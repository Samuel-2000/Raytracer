# interaction.py

import os
import numpy as np
import time
import threading
from queue import Queue
from typing import Dict, Optional, Tuple, List
import math
from enum import Enum
import cv2
import random

from PyQt5.QtCore import QThread, pyqtSignal

from denoiser import Denoiser
from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3, Camera
from cpp_raytracer.raytracer_cpp import Texture, SolidTexture, NoiseTexture, CheckerTexture
from cpp_raytracer.raytracer_cpp import WoodTexture, MarbleTexture, MetalTexture, ImageTexture
from cpp_raytracer.raytracer_cpp import Skybox, SkyboxType
from cpp_raytracer.raytracer_cpp import MaterialType
from utils import FrameRateLimiter

from video_renderer import CameraRecorder

class RenderMode(Enum):
    """Rendering modes for different interaction scenarios"""
    RAYTRACING = 0
    SILHOUETTE = 1
    WIREFRAME = 2

class Matrix3:
    """Simple 3x3 matrix for camera rotations"""
    
    @staticmethod
    def rotation_y(angle: float) -> 'Matrix3':
        c, s = math.cos(angle), math.sin(angle)
        return Matrix3([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_axis(axis: Vector3, angle: float) -> 'Matrix3':
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = axis.x, axis.y, axis.z
        
        return Matrix3([
            [c + (1-c)*x*x, (1-c)*x*y - s*z, (1-c)*x*z + s*y],
            [(1-c)*x*y + s*z, c + (1-c)*y*y, (1-c)*y*z - s*x],
            [(1-c)*x*z - s*y, (1-c)*y*z + s*x, c + (1-c)*z*z]
        ])
    
    def __init__(self, data):
        self.data = data
    
    def __mul__(self, vec: Vector3) -> Vector3:
        m = self.data
        return Vector3(
            m[0][0]*vec.x + m[0][1]*vec.y + m[0][2]*vec.z,
            m[1][0]*vec.x + m[1][1]*vec.y + m[1][2]*vec.z,
            m[2][0]*vec.x + m[2][1]*vec.y + m[2][2]*vec.z
        )

class CameraController:
    """Handles camera movement and rotation logic"""
    
    def __init__(self, camera: Camera, settings: Dict):
        self.camera = camera
        self.settings = settings
        
        # Camera control state
        self.keys_pressed = {
            'forward': False,  # W
            'backward': False,  # S
            'left': False,  # A
            'right': False,  # D
            'up': False,  # Space/Shift
            'down': False,  # Ctrl
        }
        self.rotating = False
        self.last_mouse_pos = None
        
        # Camera orientation frame
        self.update_camera_frame()
    
    def update_camera_frame(self):
        """Update camera orientation vectors"""
        # Forward vector (camera to target)
        self.forward = (self.camera.target - self.camera.position).normalize()
        
        # Right vector (perpendicular to forward and world up)
        world_up = Vector3(0, 1, 0)
        self.right = self.forward.cross(world_up).normalize()
        if self.right.length() == 0:
            self.right = Vector3(1, 0, 0)
        
        # Up vector (perpendicular to forward and right)
        self.up = self.right.cross(self.forward).normalize()
    
    def get_movement_vector(self) -> Vector3:
        """Calculate movement vector based on pressed keys"""
        move_vector = Vector3(0, 0, 0)
        speed = self.settings['camera_move_speed']
        
        if self.keys_pressed['forward']:
            move_vector = move_vector + self.forward * speed
        if self.keys_pressed['backward']:
            move_vector = move_vector - self.forward * speed
        if self.keys_pressed['left']:
            move_vector = move_vector - self.right * speed
        if self.keys_pressed['right']:
            move_vector = move_vector + self.right * speed
        if self.keys_pressed['up']:
            move_vector = move_vector + Vector3(0, speed, 0)
        if self.keys_pressed['down']:
            move_vector = move_vector - Vector3(0, speed, 0)
        
        return move_vector
    
    def rotate(self, dx: float, dy: float):
        """Rotate camera based on mouse movement"""
        sensitivity = self.settings['camera_rotate_speed']
        yaw = -dx * sensitivity
        pitch = -dy * sensitivity
        
        # Limit pitch to prevent flipping
        pitch = max(-1.5, min(1.5, pitch))
        
        # Current orientation
        forward = (self.camera.target - self.camera.position).normalize()
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        
        # Yaw rotation (around world up)
        yaw_rot = Matrix3.rotation_y(yaw)
        forward = yaw_rot * forward
        
        # Pitch rotation (around camera right)
        if abs(pitch) > 0.001:
            pitch_rot = Matrix3.rotation_axis(right, pitch)
            forward = pitch_rot * forward
        
        # Update camera target
        self.camera.target = self.camera.position + forward
        self.update_camera_frame()
    
    def apply_bounds(self):
        """Apply bounds to camera position"""
        self.camera.position.x = max(-20, min(20, self.camera.position.x))
        self.camera.position.y = max(0.1, min(20, self.camera.position.y))
        self.camera.position.z = max(-20, min(20, self.camera.position.z))

class TextureManager:
    """Manages texture loading and creation"""
    
    def __init__(self, base_path="textures"):
        self.base_path = base_path
        self.loaded_textures = {}
        
        # Create textures directory if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            print(f"Created textures directory: {base_path}")
    
    def get_texture_path(self, filename):
        """Get full path to texture file"""
        return os.path.join(self.base_path, filename)
    
    def load_image_texture(self, filename):
        """Load an image texture"""
        full_path = self.get_texture_path(filename)
        if not os.path.exists(full_path):
            print(f"Texture file not found: {full_path}")
            return None
        
        try:
            texture = ImageTexture(full_path)
            self.loaded_textures[filename] = texture
            return texture
        except Exception as e:
            print(f"Failed to load texture {filename}: {e}")
            return None
    
    def create_wood_texture(self, scale=5.0, color=None):
        """Create a wood texture"""
        if color is None:
            color = Vector3(0.6, 0.4, 0.2)
        return WoodTexture(scale, color)
    
    def create_marble_texture(self, scale=3.0, color=None):
        """Create a marble texture"""
        if color is None:
            color = Vector3(0.8, 0.8, 0.9)
        return MarbleTexture(scale, color)
    
    def create_metal_texture(self, roughness_variation=0.1):
        """Create a metal texture"""
        return MetalTexture(roughness_variation)
    
    def create_noise_texture(self, scale=1.0):
        """Create a noise texture"""
        return NoiseTexture(scale)
    
    def create_checker_texture(self, color1=None, color2=None, scale=10.0):
        """Create a checkerboard texture"""
        if color1 is None:
            color1 = Vector3(0.9, 0.9, 0.9)
        if color2 is None:
            color2 = Vector3(0.1, 0.1, 0.1)
        return CheckerTexture(color1, color2, scale)

class MaterialPresets:
    """Predefined material presets"""
    
    @staticmethod
    def create_wood():
        """Create wood material"""
        material = Material()
        material.material_type = MaterialType.WOOD
        material.albedo = Vector3(0.6, 0.4, 0.2)
        material.metallic = 0.0
        material.roughness = 0.7
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_plastic(color=None):
        """Create plastic material"""
        if color is None:
            color = Vector3(0.8, 0.8, 0.8)
        material = Material()
        material.material_type = MaterialType.PLASTIC
        material.albedo = color
        material.metallic = 0.0
        material.roughness = 0.3
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_metal(color=None):
        """Create metal material"""
        if color is None:
            color = Vector3(0.8, 0.8, 0.8)
        material = Material()
        material.material_type = MaterialType.METAL
        material.albedo = color
        material.metallic = 1.0
        material.roughness = 0.1
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_rusty_metal():
        """Create rusty metal material"""
        material = Material()
        material.material_type = MaterialType.RUSTY_METAL
        material.albedo = Vector3(0.7, 0.3, 0.1)
        material.metallic = 0.8
        material.roughness = 0.6
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_marble():
        """Create marble material"""
        material = Material()
        material.material_type = MaterialType.MARBLE
        material.albedo = Vector3(0.9, 0.9, 0.85)
        material.metallic = 0.0
        material.roughness = 0.2
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_glass(color=None):
        """Create glass material"""
        if color is None:
            color = Vector3(0.95, 0.95, 0.95)
        material = Material()
        material.material_type = MaterialType.GLASS
        material.albedo = color
        material.metallic = 0.0
        material.roughness = 0.0
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_mirror():
        """Create mirror material"""
        material = Material()
        material.material_type = MaterialType.MIRROR
        material.albedo = Vector3(1.0, 1.0, 1.0)
        material.metallic = 1.0
        material.roughness = 0.0
        material.ior = 1.5
        return material
    
    @staticmethod
    def create_rubber(color=None):
        """Create rubber material"""
        if color is None:
            color = Vector3(0.1, 0.1, 0.1)
        material = Material()
        material.material_type = MaterialType.RUBBER
        material.albedo = color
        material.metallic = 0.0
        material.roughness = 0.9
        material.ior = 1.5
        return material
    
    @staticmethod
    def get_preset_names():
        """Get list of available preset names"""
        return [
            "Custom",
            "Diffuse",
            "Metal",
            "Dielectric",
            "Plastic",
            "Wood",
            "Marble",
            "Rusty Metal",
            "Glass",
            "Mirror",
            "Rubber"
        ]

class SkyboxManager:
    """Manages skybox creation and configuration"""
    
    @staticmethod
    def create_default():
        """Create default skybox - returns a Skybox object"""
        skybox = Skybox()
        skybox.set_type(SkyboxType.GRADIENT)
        skybox.set_colors(
            Vector3(0.5, 0.7, 1.0),  # Sky blue
            Vector3(0.8, 0.9, 1.0)   # Light blue
        )
        return skybox
    
    @staticmethod
    def create_sunset():
        """Create sunset skybox"""
        skybox = Skybox()
        skybox.set_type(SkyboxType.SUNSET)
        skybox.set_colors(
            Vector3(1.0, 0.6, 0.1),  # Sunset orange
            Vector3(0.9, 0.4, 0.2),  # Deep orange
            Vector3(0.4, 0.2, 0.6)   # Purple
        )
        return skybox
    
    @staticmethod
    def create_atmosphere():
        """Create atmospheric skybox"""
        skybox = Skybox()
        skybox.set_type(SkyboxType.ATMOSPHERE)
        skybox.set_atmosphere_colors(
            Vector3(0.4, 0.6, 0.8),  # Horizon
            Vector3(0.1, 0.2, 0.4),  # Zenith
            Vector3(0.2, 0.3, 0.1)   # Ground
        )
        return skybox
    
    @staticmethod
    def create_night():
        """Create night skybox"""
        skybox = Skybox()
        skybox.set_type(SkyboxType.SOLID)
        skybox.set_colors(Vector3(0.05, 0.05, 0.1))
        return skybox
    
    @staticmethod
    def create_studio():
        """Create studio lighting skybox"""
        skybox = Skybox()
        skybox.set_type(SkyboxType.GRADIENT)
        skybox.set_colors(
            Vector3(0.2, 0.2, 0.2),  # Dark gray
            Vector3(0.5, 0.5, 0.5)   # Light gray
        )
        return skybox

class ObjectDragger:
    """Handles object dragging logic"""
    
    def __init__(self, scene: Scene, camera_controller: CameraController, settings: Dict):
        self.scene = scene
        self.camera_controller = camera_controller
        self.settings = settings
        
        self.dragging = False
        self.selected_object_id = -1
        self.drag_start_pos = None
        self.drag_start_object_pos = None
        self.lock_x = self.lock_y = self.lock_z = False
    
    def start_drag(self, x: float, y: float) -> bool:
        """Start dragging an object at screen coordinates (x, y)"""
        # This will be called from RayTracerInteraction.start_object_dragging
        return False
    
    def update_drag(self, dx: float, dy: float):
        """Update object position during dragging"""
        if not self.dragging:
            return
        
        obj = self._get_selected_object()
        if not obj:
            return
        
        speed = self.settings['move_speed'] * 2.0
        
        # Convert screen movement to world movement
        world_dx = self.camera_controller.right * dx * 2.0
        world_dy = self.camera_controller.up * (-dy) * 2.0
        
        # Apply dimension locks
        if self.lock_x:
            world_dx.x = 0
            world_dy.x = 0
        if self.lock_y:
            world_dx.y = 0
            world_dy.y = 0
        if self.lock_z:
            world_dx.z = 0
            world_dy.z = 0
        
        # Calculate new position
        move_vector = (world_dx + world_dy) * speed
        new_pos = self.drag_start_object_pos + move_vector
        
        # Apply bounds
        new_pos.x = max(-8, min(8, new_pos.x))
        new_pos.y = max(0.1, min(8, new_pos.y))
        new_pos.z = max(-8, min(2, new_pos.z))
        
        # Update object
        obj.center = new_pos
    
    def stop_drag(self):
        """Stop dragging object"""
        self.dragging = False
        self.lock_x = self.lock_y = self.lock_z = False
    
    def set_dimension_lock(self, dimension: str, state: bool):
        """Lock/unlock a dimension for dragging"""
        if dimension == 'x':
            self.lock_x = state
        elif dimension == 'y':
            self.lock_y = state
        elif dimension == 'z':
            self.lock_z = state
    
    def _get_selected_object(self) -> Optional[Sphere]:
        """Get currently selected object"""
        for sphere in self.scene.spheres:
            if sphere.object_id == self.selected_object_id:
                return sphere
        return None

class RenderStateManager:
    """Manages rendering state and mode transitions"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        self.previous_mode = RenderMode.RAYTRACING
        self.current_mode = RenderMode.RAYTRACING
        self.is_rendering = False
        
        # Buffers for fast rendering modes
        self.silhouette_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.wireframe_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Interaction state
        self.interaction_in_progress = False
        self.last_interaction_time = 0
        self.interaction_timeout = 0.5  # 500ms timeout
    
    def set_mode(self, mode: RenderMode):
        """Set rendering mode with proper state management"""
        if mode != self.current_mode:
            self.previous_mode = self.current_mode
            self.current_mode = mode
        
        # Stop ray tracing when switching to fast modes
        if mode != RenderMode.RAYTRACING:
            self.is_rendering = False
    
    def start_interaction(self):
        """Start user interaction"""
        self.interaction_in_progress = True
        self.last_interaction_time = time.time()
        
        # Store previous mode if it was ray tracing
        if self.current_mode == RenderMode.RAYTRACING:
            self.previous_mode = RenderMode.RAYTRACING
        
        # Switch to wireframe for responsiveness
        self.set_mode(RenderMode.WIREFRAME)
    
    def update_interaction(self):
        """Update last interaction time"""
        self.last_interaction_time = time.time()
    
    def should_return_to_raytracing(self) -> bool:
        """Check if should return to ray tracing mode"""
        current_time = time.time()
        return (
            self.interaction_in_progress and
            current_time - self.last_interaction_time > self.interaction_timeout and
            self.previous_mode == RenderMode.RAYTRACING and
            not self.interaction_in_progress  # Double check
        )
    
    def return_to_previous_mode(self):
        """Return to previous rendering mode"""
        if self.previous_mode == RenderMode.RAYTRACING:
            # Reset interaction state
            self.interaction_in_progress = False
            
            # Switch back to ray tracing
            self.current_mode = RenderMode.RAYTRACING
            self.is_rendering = True
        else:
            self.current_mode = self.previous_mode

class Renderer:
    """Handles different rendering modes"""
    
    def __init__(self, width: int, height: int, camera: Camera, scene: Scene):
        self.width = width
        self.height = height
        self.camera = camera
        self.scene = scene
        
        # Buffers
        self.silhouette_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.wireframe_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    def render_silhouette(self, selected_object_id: int = -1) -> np.ndarray:
        """Render silhouette view for fast object editing"""
        self.silhouette_buffer.fill(0)
        width, height = self.width, self.height
        
        # Use the same camera setup as in C++ Camera::get_ray()
        fov = self.camera.fov * 3.14159 / 180.0
        aspect_ratio = width / height
        tan_fov = math.tan(fov / 2.0)
        
        # Camera basis vectors (same as in C++)
        forward = (self.camera.target - self.camera.position).normalize()
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        up = right.cross(forward).normalize()
        
        # Helper function to project 3D point to 2D screen
        def project_point(point: Vector3):
            # Convert from world to camera space
            obj_pos = point - self.camera.position
            
            # Compute coordinates in camera basis
            z_cam = obj_pos.dot(forward)
            if z_cam <= 0.001:  # Behind or too close to camera
                return None
            
            x_cam = obj_pos.dot(right)
            y_cam = obj_pos.dot(up)
            
            # Apply perspective projection (same as C++ get_ray but inverted)
            # FIX: Added * 0.5 factor to match C++ ray generation
            x_screen = (x_cam / (z_cam * tan_fov * aspect_ratio) * 0.5 + 0.5) * width
            y_screen = (0.5 - y_cam / (z_cam * tan_fov) * 0.5) * height
            
            # Clamp to screen bounds
            x_screen = max(0, min(width - 1, x_screen))
            y_screen = max(0, min(height - 1, y_screen))
            
            return (int(x_screen), int(y_screen), z_cam)
        
        # Render all spheres
        for sphere in self.scene.spheres:
            if sphere.object_id == 0:  # Skip ground
                continue
            
            # Project sphere center
            projected = project_point(sphere.center)
            if projected is None:
                continue
                
            x_screen, y_screen, z_cam = projected
            
            # Calculate projected radius using perspective - FIXED
            # The correct formula: radius_pixels = (sphere_radius / distance) * (screen_height / (2 * tan_fov))
            sphere_radius_pixels = (sphere.radius / z_cam) * (height / (2 * tan_fov))
            radius = max(2, int(sphere_radius_pixels))
            
            if 0 <= x_screen < width and 0 <= y_screen < height:
                center = (int(x_screen), int(y_screen))
                
                # Color coding
                if sphere.object_id == selected_object_id:
                    color = (255, 255, 0)  # Yellow for selected
                    thickness = 3
                else:
                    color = (200, 200, 200)  # Gray for others
                    thickness = 1
                
                cv2.circle(self.silhouette_buffer, center, radius, color, thickness)
                
                # Crosshair for selected
                if sphere.object_id == selected_object_id:
                    cv2.line(self.silhouette_buffer,
                            (center[0] - 10, center[1]),
                            (center[0] + 10, center[1]),
                            (0, 255, 255), 2)
                    cv2.line(self.silhouette_buffer,
                            (center[0], center[1] - 10),
                            (center[0], center[1] + 10),
                            (0, 255, 255), 2)
        
        return self.silhouette_buffer.astype(np.float32) / 255.0

    def render_wireframe(self, selected_object_id: int = -1) -> np.ndarray:
        """Render wireframe view for fast camera navigation"""
        self.wireframe_buffer.fill(0)
        width, height = self.width, self.height
        
        # Camera parameters
        fov = self.camera.fov * 3.14159 / 180.0
        aspect_ratio = width / height
        tan_fov = math.tan(fov / 2.0)  # Changed from np.tan to math.tan for consistency
        
        forward = (self.camera.target - self.camera.position).normalize()
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        up = right.cross(forward).normalize()
        
        # Helper function with corrected projection
        def project_point(point: Vector3) -> Optional[Tuple[int, int]]:
            obj_pos = point - self.camera.position
            z_cam = obj_pos.dot(forward)
            
            if z_cam <= 0.1:
                return None
            
            x_cam = obj_pos.dot(right)
            y_cam = obj_pos.dot(up)
            
            # Correct projection - FIXED: Added * 0.5 factor
            x_screen = (x_cam / (z_cam * tan_fov * aspect_ratio) * 0.5 + 0.5) * width
            y_screen = (0.5 - y_cam / (z_cam * tan_fov) * 0.5) * height
            
            # Clamp to screen bounds
            x_screen = max(0, min(width - 1, x_screen))
            y_screen = max(0, min(height - 1, y_screen))
            
            return (int(x_screen), int(y_screen))
        
        # Draw ground grid
        self._render_grid(project_point)
        
        # Draw spheres
        for sphere in self.scene.spheres:
            if sphere.object_id == 0:
                continue
            
            center_screen = project_point(sphere.center)
            if center_screen:
                # Calculate screen radius - FIXED
                distance = (sphere.center - self.camera.position).dot(forward)
                if distance > 0:
                    radius_screen = (sphere.radius / distance) * (height / (2 * tan_fov))
                    radius_screen = max(2, int(radius_screen))
                    
                    # Color
                    if sphere.object_id == selected_object_id:
                        color = (255, 255, 0)
                        thickness = 2
                    else:
                        color = (200, 200, 200)
                        thickness = 1
                    
                    cv2.circle(self.wireframe_buffer, center_screen, radius_screen, color, thickness)
                    
                    # Axes for selected
                    if sphere.object_id == selected_object_id:
                        self._render_axes(sphere, center_screen, project_point)
        
        return self.wireframe_buffer.astype(np.float32) / 255.0
    
    def _render_grid(self, project_point):
        """Render ground grid"""
        grid_size = 10
        grid_step = 1.0
        
        for i in range(-grid_size, grid_size + 1):
            x = i * grid_step
            
            # X lines
            for j in range(-grid_size, grid_size):
                z1 = j * grid_step
                z2 = (j + 1) * grid_step
                
                p1 = Vector3(x, 0, z1)
                p2 = Vector3(x, 0, z2)
                
                s1 = project_point(p1)
                s2 = project_point(p2)
                
                if s1 and s2:
                    cv2.line(self.wireframe_buffer, s1, s2, (80, 80, 80), 1)
            
            # Z lines
            for j in range(-grid_size, grid_size):
                x1 = j * grid_step
                x2 = (j + 1) * grid_step
                
                p1 = Vector3(x1, 0, x)
                p2 = Vector3(x2, 0, x)
                
                s1 = project_point(p1)
                s2 = project_point(p2)
                
                if s1 and s2:
                    cv2.line(self.wireframe_buffer, s1, s2, (80, 80, 80), 1)
    
    def _render_axes(self, sphere: Sphere, center_screen: Tuple[int, int], project_point):
        """Render XYZ axes for selected object"""
        axes = [
            (Vector3(0.5, 0, 0), (255, 0, 0)),   # X - Red
            (Vector3(0, 0.5, 0), (0, 255, 0)),   # Y - Green
            (Vector3(0, 0, -0.5), (0, 0, 255))   # Z - Blue
        ]
        
        for axis_vec, axis_color in axes:
            end = sphere.center + axis_vec
            end_screen = project_point(end)
            if end_screen:
                cv2.line(self.wireframe_buffer, center_screen, end_screen, axis_color, 2)





class SceneManager:
    """Manages scene creation and object manipulation"""
    
    @staticmethod
    def create_interactive_scene(texture_manager=None) -> Scene:
        """Create a scene with interactive objects"""
        scene = Scene()
        
        # Set default skybox
        skybox = SkyboxManager.create_default()
        print(f"Skybox type: {type(skybox)}")
        print(f"Skybox is None: {skybox is None}")
        scene.set_skybox(skybox)
        
        # Ground with texture
        ground_material = Material()
        ground_material.albedo = Vector3(0.9, 0.9, 0.9)
        ground_material.roughness = 0.8
        ground_material.material_type = MaterialType.DIFFUSE
        
        # Add checker texture to ground if available
        if texture_manager:
            checker_texture = texture_manager.create_checker_texture(
                Vector3(0.9, 0.9, 0.9),
                Vector3(0.7, 0.7, 0.7),
                scale=5.0
            )
            ground_material.albedo_texture = checker_texture
        
        ground = Sphere()
        ground.center = Vector3(0, -100.5, 0)
        ground.radius = 100.0
        ground.material = ground_material
        ground.object_id = 0
        ground.name = "Ground"
        scene.add_sphere(ground)
        
        # Interactive objects with various materials
        objects_data = [
            # Main spheres with different materials
            {"pos": (-2.0, 0.5, -3.0), "preset": "Metal", "radius": 0.5, "name": "Metal Sphere"},
            {"pos": (0.0, 0.5, -3.0), "preset": "Plastic", "radius": 0.5, "name": "Plastic Sphere"},
            {"pos": (2.0, 0.5, -3.0), "preset": "Glass", "radius": 0.5, "name": "Glass Sphere"},
            
            # Material showcase
            {"pos": (-1.0, 0.3, -1.5), "preset": "Wood", "radius": 0.3, "name": "Wood Sphere"},
            {"pos": (1.0, 0.3, -1.5), "preset": "Marble", "radius": 0.3, "name": "Marble Sphere"},
            {"pos": (0.0, 0.3, 0.0), "preset": "Rusty Metal", "radius": 0.3, "name": "Rusty Metal"},
            
            # Lights - these are not material presets but have emission
            {"pos": (0, 3, -1), "preset": "light", "emission": (10, 10, 8), "radius": 0.3, "name": "Main Light"},
            {"pos": (-2, 2, 0), "preset": "light", "emission": (5, 3, 2), "radius": 0.2, "name": "Warm Light"},
            {"pos": (2, 2, 0), "preset": "light", "emission": (2, 3, 5), "radius": 0.2, "name": "Cool Light"},
        ]
        
        for i, data in enumerate(objects_data, 1):
            if data["preset"] == "light":
                material = Material()
                material.albedo = Vector3(1, 1, 1)
                material.emission = Vector3(*data["emission"])
                material.metallic = 0.0
                material.roughness = 0.1
                material.material_type = MaterialType.DIFFUSE  # Lights are diffuse
            else:
                # Use appropriate preset
                if data["preset"] == "Wood":
                    material = MaterialPresets.create_wood()
                    if texture_manager:
                        wood_texture = texture_manager.create_wood_texture(
                            scale=8.0,
                            color=Vector3(0.6, 0.4, 0.2)
                        )
                        material.albedo_texture = wood_texture
                elif data["preset"] == "Metal":
                    material = MaterialPresets.create_metal()
                    if texture_manager:
                        metal_texture = texture_manager.create_metal_texture(roughness_variation=0.05)
                        material.albedo_texture = metal_texture
                        material.roughness_texture = metal_texture
                elif data["preset"] == "Plastic":
                    material = MaterialPresets.create_plastic()
                elif data["preset"] == "Glass":
                    material = MaterialPresets.create_glass()
                elif data["preset"] == "Marble":
                    material = MaterialPresets.create_marble()
                    if texture_manager:
                        marble_texture = texture_manager.create_marble_texture(
                            scale=4.0,
                            color=Vector3(0.9, 0.9, 0.85)
                        )
                        material.albedo_texture = marble_texture
                elif data["preset"] == "Rusty Metal":
                    material = MaterialPresets.create_rusty_metal()
                    if texture_manager:
                        rust_texture = texture_manager.create_noise_texture(scale=3.0)
                        material.albedo_texture = rust_texture
            
            sphere = Sphere()
            sphere.center = Vector3(*data["pos"])
            sphere.radius = data["radius"]
            sphere.material = material
            sphere.object_id = i
            sphere.name = data["name"]
            scene.add_sphere(sphere)
        
        scene.build_bvh()
        return scene
    



class SceneGenerator:
    """Generates test scenes for benchmarking"""
    
    def __init__(self):
        #self.scene_names = ['simple', 'many_objects_100', 'many_objects_1k', 'many_objects_10k', 'many_objects_100k', 'many_objects_1M']
        self.scene_names = ['simple', 'many_objects_100', 'many_objects_1k', 'many_objects_10k']
    
    def get_scene_names(self):
        return self.scene_names
    
    def create_scene(self, name, texture_manager):
        if name == 'simple':
            return SceneManager.create_interactive_scene(texture_manager)
        elif name.startswith('many_objects'):
            num = int(name.split('_')[-1].replace('k','000').replace('M','000000'))
            return self._create_many_objects_scene(num, texture_manager)
        else:
            return None
    
    def _create_many_objects_scene(self, num_objects, texture_manager):
        scene = Scene()
        
        # Ground
        ground = Sphere()
        ground.center = Vector3(0, -100.5, 0)
        ground.radius = 100.0
        ground.material = Material()
        ground.material.albedo = Vector3(0.7, 0.7, 0.7)
        ground.material.roughness = 0.8
        ground.material.material_type = MaterialType.DIFFUSE
        ground.object_id = 0
        ground.name = "Ground"
        scene.add_sphere(ground)
        
        # Dynamický rozsah podľa počtu objektov
        # Pre 10k objektov max_range ~ 10, pre 1M ~ 50
        max_range = min(50, 5 * (num_objects ** (1/3)))
        min_range = -max_range
        
        random.seed(42)
        for i in range(1, num_objects + 1):
            sphere = Sphere()
            sphere.radius = random.uniform(0.1, 0.5)
            sphere.center = Vector3(
                random.uniform(min_range, max_range),
                random.uniform(0.2, 5),
                random.uniform(min_range, max_range)
            )
            mat_type = random.choice([MaterialType.DIFFUSE, MaterialType.METAL, MaterialType.PLASTIC])
            sphere.material = Material()
            sphere.material.material_type = mat_type
            sphere.material.albedo = Vector3(random.random(), random.random(), random.random())
            sphere.material.metallic = 0.0 if mat_type == MaterialType.DIFFUSE else 0.9
            sphere.material.roughness = random.uniform(0.1, 0.7)
            sphere.object_id = i
            sphere.name = f"Obj_{i}"
            scene.add_sphere(sphere)
        
        scene.build_bvh()
        return scene
    




class BenchmarkRunner(QThread):
    """Thread for running benchmarks"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str)       # csv_path
    error = pyqtSignal(str)
    #current_test = pyqtSignal(str, bool, bool, bool, bool, int, int)  # scene, bvh, adaptive, subsample, neural, current, total
    current_test = pyqtSignal(str, bool, bool, bool, int, int)  # scene, bvh, dynamic, simd, current, total
    
    def __init__(self, raytracer):
        super().__init__()
        self.raytracer = raytracer
        self.stopped = False
        self.MAX_OBJECTS_FOR_NO_BVH = 2000   # prah počtu objektov
    
    def run(self):
        try:
            combos = []

            #for bvh in [False, True]:
            #    for adaptive in [False, True]:
            #        for subsample in [False, True]:
            #            for neural in [False, True]:
            #                combos.append((bvh, adaptive, subsample, neural))
                            
            for bvh in [False, True]:
                for dynamic in [False, True]:
                    for simd in [False, True]:
                        combos.append((bvh, dynamic, simd))



            scenes = self.raytracer.get_available_scenes()
            results = []
            total = len(combos) * len(scenes)
            current = 0
            
            original_bvh = self.raytracer.settings['bvh_enabled']
            #original_adaptive = self.raytracer.settings['adaptive_supersampling']
            #original_subsample = self.raytracer.settings['subsampling']
            #original_neural = self.raytracer.settings['neural_denoising']
            original_dynamic = self.raytracer.settings['dynamic_bvh']
            original_simd = self.raytracer.settings['SIMD_ray_hit']

            for scene_name in scenes:
                self.raytracer.load_scene(scene_name)
                obj_count = self.raytracer.get_object_count()
                print(f"\n=== Testing scene: {scene_name} ===\n")

            #   for (bvh, adaptive, subsample, neural) in combos:

                for (bvh, dynamic, simd) in combos:
                    if self.stopped:
                        return
                    
                    # Preskočíme kombináciu, ak BVH vypnuté a scéna má priveľa objektov
                    if not bvh and obj_count > self.MAX_OBJECTS_FOR_NO_BVH:
                        print(f"\n  Skipping: BVH={bvh} (scene has {obj_count} objects > {self.MAX_OBJECTS_FOR_NO_BVH})\n")
                        continue

                    # Preskočíme kombináciu, ak BVH vypnuté a dynamic zapnuté (nedáva zmysel)
                    if not bvh and dynamic:
                        print(f"\n  Skipping: BVH={bvh} and DYN={dynamic} doesn't make sense.)\n")
                        continue


                    self.raytracer.set_bvh_enabled(bvh)
                    #self.raytracer.set_adaptive_supersampling(adaptive)
                    #self.raytracer.set_subsampling(subsample)
                    #self.raytracer.set_neural_denoising(neural)
                    self.raytracer.set_dynamic_bvh(dynamic)
                    self.raytracer.set_SIMD_ray_hit(simd)

                #   print(f"  Running: BVH={bvh}, AS={adaptive}, SS={subsample}, ND={neural}")
                    print(f"  Running: BVH={bvh}, DYN={dynamic}, SIMD={simd}")

                    #self.current_test.emit(scene_name, bvh, adaptive, subsample, neural, current+1, total)
                    self.current_test.emit(scene_name, bvh, dynamic, simd, current+1, total)
                    
                    start = time.time()
                    self.raytracer.render_state.set_mode(RenderMode.RAYTRACING)
                    self.raytracer.restart_rendering()
                    while (self.raytracer.render_state.current_mode == RenderMode.RAYTRACING 
                           and self.raytracer.total_samples < self.raytracer.settings['max_samples']):
                        time.sleep(0.05)
                    elapsed = time.time() - start
                    
                    results.append({
                        'scene': scene_name,
                        'bvh': bvh,
                        #'adaptive': adaptive,
                        #'subsampling': subsample,
                        #'neural': neural,
                        'dynamic_bvh': dynamic,
                        'simd': simd,
                        'time': elapsed,
                        'samples': self.raytracer.total_samples
                    })
                    current += 1
                    self.progress.emit(current, total)
            
            self.raytracer.set_bvh_enabled(original_bvh)
            #self.raytracer.set_adaptive_supersampling(original_adaptive)
            #self.raytracer.set_subsampling(original_subsample)
            #self.raytracer.set_neural_denoising(original_neural)
            self.raytracer.set_dynamic_bvh(original_dynamic)
            self.raytracer.set_SIMD_ray_hit(original_simd)
            
            import csv
            csv_path = 'benchmark_results.csv'
            with open(csv_path, 'w', newline='') as f:
                #writer = csv.DictWriter(f, fieldnames=['scene','bvh','adaptive','subsampling','neural','time','samples'])
                writer = csv.DictWriter(f, fieldnames=['scene', 'bvh', 'dynamic_bvh', 'simd', 'time','samples'])
                writer.writeheader()
                writer.writerows(results)
            
            self.finished.emit(csv_path)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self.stopped = True




class RayTracerInteraction:
    """Main class for interactive ray tracing with skybox and textures"""
    
    def __init__(self, width: int = 640, height: int = 480, debug_mode: bool = False):
        self.width = width
        self.height = height
        
        # Initialize texture manager
        self.texture_manager = TextureManager()
        
        # Initialize C++ ray tracer
        self.ray_tracer = RayTracer()
        self.scene = SceneManager.create_interactive_scene(self.texture_manager)
        self.ray_tracer.set_scene(self.scene)
        
        # Get camera as a copy from C++ (we push updates back with set_camera)
        self.camera = self.ray_tracer.get_camera()
        self._init_camera()
        # Immediately push our initialized camera back to C++ so both sides are in sync
        self.ray_tracer.set_camera(self.camera)

        # Camera recording
        self.camera_recorder = CameraRecorder(self)
        self.recording_mode = False

        
        # Settings
        self.settings = {
            'max_samples': 32,
            'samples_per_batch': 8,
            'max_depth': 4,
            'exposure': 1.5,
            'enhance_image': True,
            'show_denoisers': False,
            'selected_denoisers': ['bilateral'],
            'selected_object': 1,
            'move_speed': 0.3,
            'camera_move_speed': 0.1,
            'camera_rotate_speed': 0.5,
            'bvh_enabled': True,
            #'adaptive_supersampling': False,
            #'subsampling': False,
            #'neural_denoising': False,
            'dynamic_bvh': False,
            'SIMD_ray_hit': False,
        }
        
        # Initialize components
        self.camera_controller = CameraController(self.camera, self.settings)
        self.object_dragger = ObjectDragger(self.scene, self.camera_controller, self.settings)
        self.render_state = RenderStateManager(width, height)
        self.renderer = Renderer(width, height, self.camera, self.scene)
        
        # Ray tracing state
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Thread safety
        self.render_lock = threading.RLock()
        
        # Denoiser
        self.denoiser = Denoiser()
        
        # GUI reference for callbacks (set by GUI)
        self._gui = None
        
        # Key state tracking
        self._last_key_states = {}
        self._last_key_event_time = 0
        
        # Fix: Camera auto-movement prevention
        self._camera_auto_move_fix = False
        self._last_manual_movement = 0
        
        # Start camera movement thread
        self.camera_move_active = True
        self.camera_move_thread = threading.Thread(target=self._camera_move_worker, daemon=True)
        self.camera_move_thread.start()

        self.scene_generator = SceneGenerator()
        self.available_scenes = self.scene_generator.get_scene_names()
        self.load_scene('simple')
        
        self.benchmark_runner = None
        
        print(f"✓ Initialized Interactive Ray Tracer ({width}x{height})")
        print("Controls:")
        print("  Camera Movement: WASD + Space/Shift")
        print("  Camera Rotation: Right Mouse Button + Drag")
        print("  Object Selection: Left Click")
        print("  Object Dragging: Press X/Y/Z to lock dimension, then Left Click + Drag")
        print("  Cancel Operation: ESC")
        print("  Manual Mode Switching: Use buttons in top-left")
        print("=" * 50)
    

    def toggle_recording_mode(self):
        """Toggle camera recording mode"""
        with self.render_lock:
            if not self.recording_mode:
                # Start recording
                self.recording_mode = True
                self.camera_recorder.start_recording()
                
                # Switch to wireframe mode for performance
                if self.render_state.current_mode != RenderMode.WIREFRAME:
                    self.render_state.set_mode(RenderMode.WIREFRAME)
                    if self._gui:
                        self._gui.manual_mode_change = True
                        self._gui.wireframe_btn.setChecked(True)
                
                print("Recording started")
                return True
            else:
                # Stop recording
                self.recording_mode = False
                self.camera_recorder.stop_recording()
                
                # Return to ray tracing
                self.render_state.set_mode(RenderMode.RAYTRACING)
                if self._gui:
                    self._gui.on_raytrace_mode()
                
                print("Recording stopped")
                return True

    def update_recording(self):
        """Update recording with current camera frame"""
        if self.recording_mode:
            self.camera_recorder.record_frame()

    def get_recorded_frames(self):
        """Get all recorded frames"""
        return self.camera_recorder.get_all_frames()

    def clear_recording(self):
        """Clear all recorded frames"""
        self.camera_recorder.clear_recording()


    def _init_camera(self):
        """Initialize camera position and orientation"""
        self.camera.position = Vector3(0, 1, 5)
        self.camera.target = Vector3(0, 1, 4)
        self.camera.up = Vector3(0, 1, 0)
        self.camera.fov = 45.0


    def reset_camera_and_rerender(self):
        with self.render_lock:
            # Reset camera to defaults
            self._init_camera()

            # Push camera to C++
            self.ray_tracer.set_camera(self.camera)

            # Force an immediate visual update
            self.render_state.start_interaction()
            self._process_frame_for_display(0.0)

            # Restart ray tracing cleanly
            self.render_state.set_mode(RenderMode.RAYTRACING)
            self.restart_rendering()
    

    def set_object_color(self, r: float, g: float, b: float, apply_immediate: bool = True):
        """Nastaví albedo (RGB) vybranému objektu."""
        obj = self.get_selected_object()
        if not obj:
            return
        obj.material.albedo = Vector3(r, g, b)
        # ak je light, udržiavať intenzitu
        if hasattr(obj.material, 'emission'):
            if (obj.material.emission.x + obj.material.emission.y + obj.material.emission.z) > 0.001:
                avg = (obj.material.emission.x + obj.material.emission.y + obj.material.emission.z) / 3.0
                obj.material.emission = Vector3(r * avg, g * avg, b * avg)

        if apply_immediate:
            self.ray_tracer.set_scene(self.scene)
            self.restart_rendering()

    def set_object_color_hsv(self, h: float, s: float, v: float, apply_immediate: bool = True):
        """Nastaví farbu pomocou HSV (h in degrees 0-360, s/v in 0-1)."""
        # Convert HSV -> RGB
        h_norm = (h % 360) / 360.0
        i = int(h_norm * 6)
        f = h_norm * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        self.set_object_color(r, g, b, apply_immediate=apply_immediate)

    # ==================== SKYBOX METHODS ====================
    
    def set_skybox_type(self, skybox_type: str):
        """Set skybox type"""
        with self.render_lock:
            skybox = self.scene.get_skybox()
            if not skybox:
                skybox = Skybox()
                self.scene.set_skybox(skybox)
            
            if skybox_type == "gradient":
                skybox.set_type(SkyboxType.GRADIENT)
                skybox.set_colors(Vector3(0.5, 0.7, 1.0), Vector3(0.8, 0.9, 1.0))
            elif skybox_type == "sunset":
                skybox.set_type(SkyboxType.SUNSET)
                skybox.set_colors(Vector3(1.0, 0.6, 0.1), Vector3(0.9, 0.4, 0.2), Vector3(0.4, 0.2, 0.6))
            elif skybox_type == "atmosphere":
                skybox.set_type(SkyboxType.ATMOSPHERE)
                skybox.set_atmosphere_colors(
                    Vector3(0.4, 0.6, 0.8),
                    Vector3(0.1, 0.2, 0.4),
                    Vector3(0.2, 0.3, 0.1)
                )
            elif skybox_type == "solid":
                skybox.set_type(SkyboxType.SOLID)
                skybox.set_colors(Vector3(0.1, 0.1, 0.1))
            elif skybox_type == "night":
                skybox.set_type(SkyboxType.SOLID)
                skybox.set_colors(Vector3(0.05, 0.05, 0.1))
            elif skybox_type == "studio":
                skybox.set_type(SkyboxType.GRADIENT)
                skybox.set_colors(Vector3(0.2, 0.2, 0.2), Vector3(0.5, 0.5, 0.5))
            
            self.restart_rendering()
    
    def set_skybox_colors(self, color1: Vector3, color2: Vector3 = None, color3: Vector3 = None):
        """Set skybox colors"""
        with self.render_lock:
            skybox = self.scene.get_skybox()
            if skybox:
                skybox.set_colors(color1, color2 or Vector3(0,0,0), color3 or Vector3(0,0,0))
                self.restart_rendering()
    
    def load_skybox_image(self, filename: str):
        """Load skybox from image file"""
        with self.render_lock:
            skybox = self.scene.get_skybox()
            if not skybox:
                skybox = Skybox()
                self.scene.set_skybox(skybox)
            
            success = skybox.load_image(filename)
            if success:
                skybox.set_type(SkyboxType.IMAGE)
                self.restart_rendering()
            return success
    
    # ==================== MATERIAL PRESETS ====================
    
    def apply_material_preset(self, preset_name: str):
        """Apply material preset to selected object"""
        obj = self.get_selected_object()
        if not obj:
            return False
        
        with self.render_lock:
            if preset_name == "Wood":
                obj.material = MaterialPresets.create_wood()
                # Add wood texture
                wood_texture = self.texture_manager.create_wood_texture(
                    scale=8.0,
                    color=Vector3(0.6, 0.4, 0.2)
                )
                obj.material.albedo_texture = wood_texture
                
            elif preset_name == "Plastic":
                obj.material = MaterialPresets.create_plastic()
                obj.material.albedo = Vector3(0.8, 0.8, 0.8)
                
            elif preset_name == "Metal":
                obj.material = MaterialPresets.create_metal()
                metal_texture = self.texture_manager.create_metal_texture(roughness_variation=0.05)
                obj.material.albedo_texture = metal_texture
                obj.material.roughness_texture = metal_texture
                
            elif preset_name == "Rusty Metal":
                obj.material = MaterialPresets.create_rusty_metal()
                rust_texture = self.texture_manager.create_noise_texture(scale=3.0)
                obj.material.albedo_texture = rust_texture
                
            elif preset_name == "Marble":
                obj.material = MaterialPresets.create_marble()
                marble_texture = self.texture_manager.create_marble_texture(
                    scale=4.0,
                    color=Vector3(0.9, 0.9, 0.85)
                )
                obj.material.albedo_texture = marble_texture
                
            elif preset_name == "Glass":
                obj.material = MaterialPresets.create_glass()
                
            elif preset_name == "Mirror":
                obj.material = MaterialPresets.create_mirror()
                
            elif preset_name == "Rubber":
                obj.material = MaterialPresets.create_rubber()
                
            else:  # Custom/Diffuse
                obj.material = Material()
                obj.material.material_type = MaterialType.DIFFUSE  # Use enum
            
            # Update scene
            self.ray_tracer.set_scene(self.scene)
            self.restart_rendering()
            
            # Update GUI
            if self._gui:
                self._gui.control_panel.update_material_sliders()
            
            return True
    
    # ==================== TEXTURE METHODS ====================
    
    def apply_texture_to_object(self, texture_type: str, params: dict = None):
        """Apply texture to selected object - FIXED IMAGE TEXTURE"""
        obj = self.get_selected_object()
        if not obj:
            return False
        
        if params is None:
            params = {}
        
        with self.render_lock:
            if texture_type == "none":
                obj.material.albedo_texture = None
                obj.material.roughness_texture = None
                
            elif texture_type == "noise":
                scale = params.get('scale', 1.0)
                texture = self.texture_manager.create_noise_texture(scale)
                obj.material.albedo_texture = texture
                
            elif texture_type == "checker":
                color1 = params.get('color1', Vector3(0.9, 0.9, 0.9))
                color2 = params.get('color2', Vector3(0.1, 0.1, 0.1))
                scale = params.get('scale', 10.0)
                texture = self.texture_manager.create_checker_texture(color1, color2, scale)
                obj.material.albedo_texture = texture
                
            elif texture_type == "wood":
                scale = params.get('scale', 5.0)
                color = params.get('color', Vector3(0.6, 0.4, 0.2))
                texture = self.texture_manager.create_wood_texture(scale, color)
                obj.material.albedo_texture = texture
                
            elif texture_type == "marble":
                scale = params.get('scale', 3.0)
                color = params.get('color', Vector3(0.8, 0.8, 0.9))
                texture = self.texture_manager.create_marble_texture(scale, color)
                obj.material.albedo_texture = texture
                
            elif texture_type == "metal":
                roughness_var = params.get('roughness_variation', 0.1)
                texture = self.texture_manager.create_metal_texture(roughness_var)
                obj.material.albedo_texture = texture
                obj.material.roughness_texture = texture
                
            elif texture_type == "image":
                filename = params.get('filename')
                if filename:
                    try:
                        # Load the image texture
                        texture = ImageTexture(filename)
                        if texture:
                            obj.material.albedo_texture = texture
                            print(f"Applied image texture: {filename}")
                        else:
                            print(f"Failed to load image texture: {filename}")
                            return False
                    except Exception as e:
                        print(f"Error loading image texture: {e}")
                        return False
                else:
                    print("No filename provided for image texture")
                    return False
            
            # Update scene
            self.ray_tracer.set_scene(self.scene)
            self.restart_rendering()
            return True
    
    
    def load_texture_from_file(self, filename: str):
        """Load texture from file and apply to selected object"""
        return self.apply_texture_to_object("image", {"filename": filename})
    
    # ==================== UTILITY METHODS ====================
    
    def get_available_skyboxes(self):
        """Get list of available skybox types"""
        return [
            "gradient",
            "sunset",
            "atmosphere",
            "solid",
            "night",
            "studio"
        ]
    
    def get_available_material_presets(self):
        """Get list of available material presets"""
        return MaterialPresets.get_preset_names()
    
    def get_available_textures(self):
        """Get list of available texture types"""
        return [
            "none",
            "noise",
            "checker",
            "wood",
            "marble",
            "metal",
            "image"
        ]
    
    def get_selected_object(self) -> Optional[Sphere]:
        """Get currently selected object"""
        selected_idx = self.settings['selected_object']
        return self._get_sphere_by_id(selected_idx)
    
    def select_object_by_click(self, x: float, y: float) -> bool:
        """Select object by screen coordinates using ray casting"""
        try:
            with self.render_lock:
                # Convert screen coordinates to NDC
                ndc_x = (2.0 * x - 1.0)
                ndc_y = (1.0 - 2.0 * y)
                
                # Get camera parameters
                camera = self.camera
                fov = camera.fov * 3.14159 / 180.0
                aspect_ratio = self.width / self.height
                tan_fov = math.tan(fov / 2.0)
                
                # Calculate camera basis
                forward = (camera.target - camera.position).normalize()
                right = forward.cross(Vector3(0, 1, 0)).normalize()
                up = right.cross(forward).normalize()
                
                # Calculate ray direction
                ray_dir_x = ndc_x * tan_fov * aspect_ratio
                ray_dir_y = ndc_y * tan_fov
                ray_dir = (forward + right * ray_dir_x + up * ray_dir_y).normalize()
                
                # Find closest intersected object
                closest_t = float('inf')
                closest_obj_id = -1
                
                for sphere in self.scene.spheres:
                    # Skip ground for selection
                    if sphere.object_id == 0:
                        continue
                        
                    # Sphere intersection test
                    oc = camera.position - sphere.center
                    a = ray_dir.dot(ray_dir)
                    b = 2.0 * oc.dot(ray_dir)
                    c = oc.dot(oc) - sphere.radius * sphere.radius
                    discriminant = b * b - 4 * a * c
                    
                    if discriminant > 0:
                        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                        if t > 0.001 and t < closest_t:
                            closest_t = t
                            closest_obj_id = sphere.object_id
                
                if closest_obj_id >= 0:
                    self.settings['selected_object'] = closest_obj_id
                    self.object_dragger.selected_object_id = closest_obj_id
                    
                    # Update GUI if available
                    if self._gui:
                        try:
                            self._gui.control_panel.object_select.setCurrentIndex(closest_obj_id)
                            self._gui.control_panel.update_object_info()
                            self._gui.control_panel.update_material_sliders()
                        except:
                            pass
                    
                    return True
                    
        except Exception as e:
            print(f"Object selection error: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def move_object(self, dx: float, dy: float, dz: float):
        """Move selected object with keyboard"""
        with self.render_lock:
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:
                speed = self.settings['move_speed']
                
                # Calculate movement in world space
                if abs(dx) > 0:
                    obj.center.x += dx * speed
                if abs(dy) > 0:
                    obj.center.y += dy * speed
                if abs(dz) > 0:
                    obj.center.z += dz * speed
                
                # Apply bounds
                obj.center.x = max(-8, min(8, obj.center.x))
                obj.center.y = max(0.1, min(8, obj.center.y))
                obj.center.z = max(-8, min(2, obj.center.z))
                
                # Update scene
                self.ray_tracer.set_scene(self.scene)
                self.restart_rendering()
                
                # Update GUI if available
                if self._gui:
                    self._gui.control_panel.update_object_info()
    
    def update_object_material(self, property_name: str, value: float):
        """Update material property of selected object"""
        obj = self.get_selected_object()
        if obj:
            if property_name == 'albedo':
                obj.material.albedo = Vector3(value, value, value)
            elif property_name == 'metallic':
                obj.material.metallic = value
            elif property_name == 'roughness':
                obj.material.roughness = value
            
            self.restart_rendering()
    
    def update_object_material_immediate(self):
        """Update material immediately and restart rendering"""
        with self.render_lock:
            self.ray_tracer.set_scene(self.scene)
            self.restart_rendering()
    
    def update_light_intensity(self, intensity: float):
        """Update light intensity for selected light"""
        obj = self.get_selected_object()
        if obj and hasattr(obj.material, 'emission'):
            emission = obj.material.emission
            
            # Check if it's a light (has non-zero emission)
            is_light = emission.x > 0.1 or emission.y > 0.1 or emission.z > 0.1
            
            if is_light:
                # Map intensity to emission color preserving ratios
                current_max = max(emission.x, emission.y, emission.z)
                if current_max > 0:
                    scale = intensity / current_max
                    obj.material.emission = Vector3(
                        emission.x * scale,
                        emission.y * scale,
                        emission.z * scale
                    )
                
                # Update scene
                self.ray_tracer.set_scene(self.scene)
                self.restart_rendering()
    
    def add_object_to_scene(self):
        """Add a new sphere to the scene"""
        with self.render_lock:
            # Find next available object ID
            max_id = 0
            for sphere in self.scene.spheres:
                if sphere.object_id > max_id:
                    max_id = sphere.object_id

            # Create new sphere
            new_sphere = Sphere()
            new_sphere.center = Vector3(0, 2, -3)  # Default position
            new_sphere.radius = 0.5

            # Create material
            material = Material()
            material.albedo = Vector3(0.8, 0.8, 0.8)
            material.metallic = 0.0
            material.roughness = 0.5
            material.emission = Vector3(0, 0, 0)
            material.ior = 1.5

            new_sphere.material = material
            new_sphere.object_id = max_id + 1
            new_sphere.name = f"Sphere {max_id + 1}"

            # Use C++ add_sphere so the native Scene container is updated correctly
            try:
                self.scene.add_sphere(new_sphere)
            except Exception:
                # Fallback if binding fails
                self.scene.spheres.append(new_sphere)

            # Rebuild BVH and notify the ray tracer
            try:
                self.scene.build_bvh()
            except Exception:
                pass
            self.ray_tracer.set_scene(self.scene)

            # Update selected object
            self.settings['selected_object'] = new_sphere.object_id
            self.object_dragger.selected_object_id = new_sphere.object_id

            # GUI updates (best-effort)
            if self._gui:
                try:
                    self._gui.control_panel.update_object_list()
                    self._gui.control_panel.object_select.setCurrentIndex(new_sphere.object_id)
                    self._gui.control_panel.update_object_info()
                    self._gui.control_panel.update_material_sliders()
                except:
                    pass

            print(f"Added new sphere with ID {new_sphere.object_id}")
            self.restart_rendering()
            return new_sphere.object_id

    
    def remove_object_from_scene(self, object_id: int):
        """Remove object from scene by ID"""
        with self.render_lock:
            removed = False
            # Prefer calling C++ remove_sphere if available
            if hasattr(self.scene, "remove_sphere"):
                try:
                    self.scene.remove_sphere(object_id)
                    removed = True
                except Exception:
                    removed = False

            if not removed:
                # Fallback to removing in Python (this may or may not reflect to C++ depending on binding)
                for i, sphere in enumerate(list(self.scene.spheres)):
                    if sphere.object_id == object_id:
                        del self.scene.spheres[i]
                        removed = True
                        break

            if not removed:
                print(f"Object with ID {object_id} not found")
                return False

            # Rebuild BVH and set scene
            try:
                self.scene.build_bvh()
            except Exception:
                pass
            self.ray_tracer.set_scene(self.scene)

            # Update selected object to next available
            self.settings['selected_object'] = 0
            self.object_dragger.selected_object_id = 0
            for sphere in self.scene.spheres:
                if sphere.object_id > 0:
                    self.settings['selected_object'] = sphere.object_id
                    self.object_dragger.selected_object_id = sphere.object_id
                    break

            # GUI updates
            if self._gui:
                try:
                    self._gui.control_panel.update_object_list()
                    self._gui.control_panel.update_object_info()
                    self._gui.control_panel.update_material_sliders()
                except:
                    pass

            self.restart_rendering()
            return True

    
    def _get_sphere_by_id(self, object_id: int) -> Optional[Sphere]:
        """Helper method to get sphere by object ID"""
        for sphere in self.scene.spheres:
            if sphere.object_id == object_id:
                return sphere
        return None
    


# Add these methods to the RayTracerInteraction class:

    def enable_floor(self):
        """Enable the floor (ground sphere)"""
        with self.render_lock:
            # Check if ground exists
            ground_exists = any(sphere.object_id == 0 for sphere in self.scene.spheres)
            
            if not ground_exists:
                print("Creating ground sphere...")
                
                # Create ground with texture
                ground_material = Material()
                ground_material.albedo = Vector3(0.9, 0.9, 0.9)
                ground_material.roughness = 0.8
                ground_material.material_type = MaterialType.DIFFUSE
                
                # Add checker texture to ground
                checker_texture = self.texture_manager.create_checker_texture(
                    Vector3(0.9, 0.9, 0.9),
                    Vector3(0.7, 0.7, 0.7),
                    scale=5.0
                )
                ground_material.albedo_texture = checker_texture
                
                ground = Sphere()
                ground.center = Vector3(0, -100.5, 0)
                ground.radius = 100.0
                ground.material = ground_material
                ground.object_id = 0
                ground.name = "Ground"
                
                # Add to both Python list AND C++ scene
                self.scene.spheres.append(ground)
                
                # Use C++ add_sphere to ensure it's in the native container
                try:
                    self.scene.add_sphere(ground)
                except Exception as e:
                    print(f"Warning: Could not add sphere via C++ method: {e}")
                    # Fall back to Python list only
                
                # Rebuild BVH and update scene
                try:
                    self.scene.build_bvh()
                except Exception as e:
                    print(f"Warning: Could not build BVH: {e}")
                
                # Force update the ray tracer with the new scene
                self.ray_tracer.set_scene(self.scene)
                self.restart_rendering()
                print(f"Floor enabled. Total spheres: {len(self.scene.spheres)}")
            else:
                print("Ground already exists")

    def disable_floor(self):
        """Disable the floor (ground sphere)"""
        with self.render_lock:
            # Remove ground sphere (object_id = 0) from Python list
            original_count = len(self.scene.spheres)
            self.scene.spheres = [sphere for sphere in self.scene.spheres if sphere.object_id != 0]
            
            if len(self.scene.spheres) < original_count:
                print(f"Removed ground. Remaining spheres: {len(self.scene.spheres)}")
                
                # Try to use C++ remove_sphere if available
                try:
                    if hasattr(self.scene, "remove_sphere"):
                        self.scene.remove_sphere(0)
                except Exception as e:
                    print(f"Warning: Could not remove sphere via C++ method: {e}")
                    # Continue with Python list only
                
                # Rebuild BVH and update scene
                try:
                    self.scene.build_bvh()
                except Exception as e:
                    print(f"Warning: Could not build BVH: {e}")
                
                # Force update the ray tracer with the new scene
                self.ray_tracer.set_scene(self.scene)
                self.restart_rendering()
                print("Floor disabled")
            else:
                print("Ground not found to disable")



    
    # ------------------------------------------------------------------
    # Camera Control Methods
    # ------------------------------------------------------------------
    
    def set_camera_key_state(self, key: str, state: bool):
        """Update camera key state with better handling"""
        if key not in self.camera_controller.keys_pressed:
            return

        with self.render_lock:
            old_state = self.camera_controller.keys_pressed[key]

            # Only process if state actually changed
            if state == old_state:
                return

            self.camera_controller.keys_pressed[key] = state

            current_time = time.time()
            if state:
                self._last_manual_movement = current_time
                # Start interaction when any key is pressed
                if self.render_state.current_mode == RenderMode.RAYTRACING:
                    self.render_state.start_interaction()
                    self._process_frame_for_display(0.016)

            # If all released, perform cleanup immediately
            all_released = not any(self.camera_controller.keys_pressed.values())
            if all_released and not self.camera_controller.rotating:
                # push camera and return to raytracing immediately
                self._handle_all_keys_released()

    
    def start_camera_rotation(self, x: float, y: float):
        """Start camera rotation with mouse"""
        with self.render_lock:
            self.camera_controller.rotating = True
            self.camera_controller.last_mouse_pos = (x, y)
            self.render_state.start_interaction()
    
    def update_camera_rotation(self, dx: float, dy: float):
        """Update camera rotation based on mouse movement"""
        with self.render_lock:
            if not self.camera_controller.rotating:
                return
            
            self.render_state.update_interaction()
            self.camera_controller.rotate(dx, dy)
            
            # Also update the ray tracer's camera
            self.ray_tracer.set_camera(self.camera)
            # Force display update
            self._process_frame_for_display(0.05)
    
    def stop_camera_rotation(self):
        """Stop camera rotation and return to previous mode"""
        with self.render_lock:
            was_rotating = self.camera_controller.rotating
            self.camera_controller.rotating = False
            self.camera_controller.last_mouse_pos = None
            
            if was_rotating:
                self._handle_rotation_stopped()
    
    # ------------------------------------------------------------------
    # Object Dragging Methods
    # ------------------------------------------------------------------
    
    def start_object_dragging(self, x: float, y: float) -> bool:
        """Start dragging an object"""
        # First try to select the object
        if self.select_object_by_click(x, y):
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:  # Don't drag ground
                self.object_dragger.dragging = True
                self.object_dragger.selected_object_id = obj.object_id
                self.object_dragger.drag_start_pos = (x, y)
                self.object_dragger.drag_start_object_pos = obj.center
                
                # Update render state
                if self.render_state.current_mode == RenderMode.RAYTRACING:
                    self.render_state.set_mode(RenderMode.SILHOUETTE)
                
                return True
        return False
    
    def update_object_dragging(self, dx: float, dy: float):
        """Update object position during dragging"""
        if not self.object_dragger.dragging:
            return
        
        self.object_dragger.update_drag(dx, dy)
        
        # Update ray tracer scene
        self.ray_tracer.set_scene(self.scene)
        self._process_frame_for_display(0.016)
    
    def stop_object_dragging(self):
        """Stop dragging object"""
        self.object_dragger.stop_drag()
        self.render_state.set_mode(RenderMode.RAYTRACING)
        self.restart_rendering()
    
    def set_dimension_lock(self, dimension: str, state: bool):
        """Lock/unlock a dimension for dragging"""
        self.object_dragger.set_dimension_lock(dimension, state)
    
    # ------------------------------------------------------------------
    # Rendering Methods
    # ------------------------------------------------------------------
    
    def restart_rendering(self):
        """Restart ray tracing"""
        with self.render_lock:
            self.render_state.is_rendering = False
            time.sleep(0.02)
            
            self.accumulated_image = None
            self.total_samples = 0
            self.frame_queue = Queue()
            
            self.start_rendering()
    
    def start_rendering(self):
        """Start progressive rendering"""
        if self.render_state.is_rendering:
            return
        
        self.render_state.is_rendering = True
        self.accumulated_image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.total_samples = 0
        
        render_thread = threading.Thread(target=self._render_worker)
        render_thread.daemon = True
        render_thread.start()
    
    # ------------------------------------------------------------------
    # Internal Worker Methods
    # ------------------------------------------------------------------
    
    def _camera_move_worker(self):
        """Continuous camera movement worker thread with frame limiting"""
        limiter = FrameRateLimiter(30)
        
        while self.camera_move_active:
            try:
                current_time = time.time()
                
                if self.recording_mode:
                    self.update_recording()
                
                # Check for manual movement and prevent auto-movement
                keys_pressed = any(self.camera_controller.keys_pressed.values())
                is_moving = keys_pressed or self.camera_controller.rotating
                
                if is_moving:
                    # Update interaction time to prevent premature return to raytracing
                    self._last_manual_movement = current_time
                    self.render_state.update_interaction()
                    
                    # Process movement if we're supposed to be moving
                    if limiter.should_update():
                        self._process_camera_movement()
                        limiter.update()
                
                # Check if should return to ray tracing after interaction timeout
                time_since_last_manual = current_time - self._last_manual_movement
                
                if (self.render_state.should_return_to_raytracing() and
                    not any(self.camera_controller.keys_pressed.values()) and
                    not self.camera_controller.rotating and
                    time_since_last_manual > 0.5):  # 500ms delay after manual movement
                    
                    with self.render_lock:
                        # Double-check no keys are pressed
                        if not any(self.camera_controller.keys_pressed.values()) and not self.camera_controller.rotating:
                            # Switch back to ray tracing
                            self.render_state.set_mode(RenderMode.RAYTRACING)
                            self.restart_rendering()
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Camera worker error: {e}")
                time.sleep(0.1)
    
    def _process_camera_movement(self):
        """Process continuous camera movement"""
        with self.render_lock:
            if not any(self.camera_controller.keys_pressed.values()):
                return
                
            move_vector = self.camera_controller.get_movement_vector()
            
            if move_vector.length() > 0:
                # Apply movement to BOTH the camera and ray tracer camera
                self.camera.position = self.camera.position + move_vector
                self.camera.target = self.camera.target + move_vector
                
                # Update ray tracer camera
                self.ray_tracer.set_camera(self.camera)
                
                # Apply bounds
                self.camera_controller.apply_bounds()
                self.camera_controller.update_camera_frame()
                
                # Ensure we're in wireframe mode during movement
                if self.render_state.current_mode != RenderMode.WIREFRAME:
                    self.render_state.set_mode(RenderMode.WIREFRAME)
                
                # Force a wireframe update
                self._process_frame_for_display(0.05)
    
    def _render_worker(self):
        """Worker function for ray tracing"""
        try:
            while (self.render_state.is_rendering and 
                   self.total_samples < self.settings['max_samples']):
                
                start_time = time.time()
                
                with self.render_lock:
                    result = self.ray_tracer.render(
                        self.width, self.height,
                        self.settings['samples_per_batch'],
                        self.settings['max_depth']
                    )
                
                if result is None or len(result) == 0:
                    continue
                
                # Process batch
                batch_image = np.array(result, dtype=np.float32).reshape(
                    (self.height, self.width, 3)
                )
                render_time = time.time() - start_time
                
                batch_samples = self.settings['samples_per_batch']
                
                if self.total_samples == 0:
                    self.accumulated_image = batch_image
                    self.total_samples = batch_samples
                else:
                    total_old = self.total_samples
                    total_new = self.total_samples + batch_samples
                    
                    weight_old = total_old / total_new
                    weight_new = batch_samples / total_new
                    
                    self.accumulated_image = (
                        self.accumulated_image * weight_old +
                        batch_image * weight_new
                    )
                    self.total_samples = total_new
                
                # Send frame if needed
                if (self.total_samples % self.settings['samples_per_batch'] == 0 or
                    self.total_samples >= self.settings['max_samples']):
                    self._process_frame_for_display(render_time)
                
                time.sleep(0.005)
                
        except Exception as e:
            print(f"Rendering error: {e}")
            import traceback
            traceback.print_exc()
        
        self.frame_queue.put({'done': True})
        self.render_state.is_rendering = False
    
    # ------------------------------------------------------------------
    # Frame Processing Methods
    # ------------------------------------------------------------------
    
    def _process_frame_for_display(self, render_time: float):
        """Process frame for display based on current mode"""
        if self.render_state.current_mode == RenderMode.SILHOUETTE:
            display_image = self.renderer.render_silhouette(
                self.object_dragger.selected_object_id
            )
            enhanced_image = display_image
            mode_str = "silhouette"
            denoised_images = {}
            
        elif self.render_state.current_mode == RenderMode.WIREFRAME:
            display_image = self.renderer.render_wireframe(
                self.object_dragger.selected_object_id
            )
            enhanced_image = display_image
            mode_str = "wireframe"
            denoised_images = {}
            
        else:  # RAYTRACING
            if self.accumulated_image is None:
                return
            
            display_image = self._tone_map(self.accumulated_image, self.settings['exposure'])
            enhanced_image = self._enhance_display(display_image) if self.settings['enhance_image'] else display_image
            mode_str = "raytracing"
            
            # Apply denoisers if needed
            denoised_images = {}
            if self.settings['show_denoisers'] and self.settings['selected_denoisers']:
                for method in self.settings['selected_denoisers']:
                    try:
                        denoised_images[method] = self.denoiser.denoise(display_image, method)
                    except Exception as e:
                        print(f"Denoising error: {e}")
        
        frame_data = {
            'display': display_image,
            'enhanced': enhanced_image,
            'denoised': denoised_images,
            'samples': self.total_samples,
            'render_time': render_time,
            'mode': mode_str,
            'is_raytracing': self.render_state.current_mode == RenderMode.RAYTRACING
        }
        
        self.frame_queue.put(frame_data)
    
    # ------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------
    
    def _handle_all_keys_released(self):
        """Handle when all movement keys are released"""
        print(f"All keys released, previous mode: {self.render_state.previous_mode}")
        
        if self.render_state.previous_mode == RenderMode.RAYTRACING:
            # Small delay to ensure all key events are processed
            time.sleep(0.02)
            
            # Double-check no keys are pressed
            if not any(self.camera_controller.keys_pressed.values()):
                # Update ray tracer camera
                self.ray_tracer.set_camera(self.camera)
                self.render_state.set_mode(RenderMode.RAYTRACING)
                self.restart_rendering()
        else:
            self.render_state.return_to_previous_mode()
            self._process_frame_for_display(0.016)
    
    def _handle_rotation_stopped(self):
        """Handle when camera rotation stops"""
        if self.render_state.previous_mode == RenderMode.RAYTRACING:
            # Reset interaction state
            self.render_state.interaction_in_progress = False
            
            # Short delay before returning to ray tracing
            time.sleep(0.05)
            
            # Switch back to ray tracing
            self.render_state.set_mode(RenderMode.RAYTRACING)
            print("Camera rotation stopped, returning to ray tracing")
            
            # Restart ray tracing
            self.restart_rendering()
        else:
            # Return to whatever mode we were in before
            self.render_state.return_to_previous_mode()
            self._process_frame_for_display(0.016)
    
    def _tone_map(self, image: np.ndarray, exposure: float) -> np.ndarray:
        """Apply tone mapping"""
        image = image * exposure
        image = image / (1.0 + image)
        return np.clip(image, 0.0, 1.0)
    
    def _enhance_display(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast"""
        min_val = np.percentile(image, 2)
        max_val = np.percentile(image, 98)
        
        if max_val > min_val:
            enhanced = (image - min_val) / (max_val - min_val)
            return np.clip(enhanced, 0, 1)
        return image
    

    def resize(self, width: int, height: int):
        """Resize the renderer without resetting the scene"""
        with self.render_lock:
            print(f"Resizing from {self.width}x{self.height} to {width}x{height}")
            
            # Store the current scene and camera
            current_scene = self.scene
            current_camera = self.camera
            current_settings = self.settings.copy()
            
            # Update dimensions
            self.width = width
            self.height = height
            
            # Update camera aspect ratio
            self.camera.aspect_ratio = width / height
            
            # Update renderer dimensions
            self.render_state.width = width
            self.render_state.height = height
            self.renderer.width = width
            self.renderer.height = height
            
            # Update renderer camera reference
            self.renderer.camera = self.camera
            
            # Resize buffers
            self.render_state.silhouette_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self.render_state.wireframe_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self.renderer.silhouette_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self.renderer.wireframe_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Restart rendering
            self.restart_rendering()
            print(f"Resize complete: {self.width}x{self.height}")
    
    # ------------------------------------------------------------------
    # Public Getter Methods
    # ------------------------------------------------------------------
    
    def get_object_count(self) -> int:
        """Get number of interactive objects (excluding ground)"""
        return len(self.scene.spheres) - 1
    
    def has_frames(self) -> bool:
        """Check if frames are available"""
        return not self.frame_queue.empty()
    
    def get_frame(self) -> Optional[Dict]:
        """Get next frame"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
    
    def stop_rendering(self):
        """Stop all rendering"""
        self.render_state.is_rendering = False
        self.camera_move_active = False
        if self.camera_move_thread:
            self.camera_move_thread.join(timeout=1.0)




    # ==================== OPTIMIZATION METHODS ====================
    
    def set_bvh_enabled(self, enabled):
        with self.render_lock:
            self.settings['bvh_enabled'] = enabled
            self.scene.use_bvh = enabled
            
            if enabled:
                self.scene.build_bvh()   # Vytvoríme BVH pre aktuálnu scénu
            
            # Aktualizácia v C++ (aj keď scéna zostáva rovnaká, potrebujeme preniesť zmenu use_bvh)
            self.ray_tracer.set_scene(self.scene)
            
            self.restart_rendering()
    
    #def set_adaptive_supersampling(self, enabled):
    #    self.settings['adaptive_supersampling'] = enabled
    #    # TODO: implement in C++
    #    self.restart_rendering()
    #
    #def set_subsampling(self, enabled):
    #    self.settings['subsampling'] = enabled
    #    # TODO: implement in C++
    #    self.restart_rendering()
    #
    #def set_neural_denoising(self, enabled):
    #    self.settings['neural_denoising'] = enabled
    #    # TODO: implement in C++
    #    self.restart_rendering()

    # Dynamicky upravovať bvh hierarchiu v čase (pri pohybe objektov preč z ich obálky)
    # bez tohoto sa staticky pri pohybe rebuildí celý bvh strom, čo je pomalé.
    def set_dynamic_bvh(self, enabled):
        self.settings['dynamic_bvh'] = enabled
        self.scene.dynamic_bvh = enabled   # sets the C++ member
        self.ray_tracer.set_scene(self.scene)
        self.restart_rendering()

    # SIMD bvh ray hit (susedné paprsky pravdepodobne trafia rovnaký objekt)
    def set_SIMD_ray_hit(self, enabled):
        self.settings['SIMD_ray_hit'] = enabled
        self.scene.simd_ray_hit = enabled    # sets the C++ member
        self.ray_tracer.set_scene(self.scene)
        self.restart_rendering()


    
    # ==================== DENOISER METHODS ====================
    
    def apply_denoisers_to_final(self):
        """Apply selected denoisers to the final image"""
        if self.accumulated_image is None:
            return
        
        display_image = self._tone_map(self.accumulated_image, self.settings['exposure'])
        enhanced_image = self._enhance_display(display_image) if self.settings['enhance_image'] else display_image
        
        denoised_images = {}
        if self.settings['show_denoisers'] and self.settings['selected_denoisers']:
            for method in self.settings['selected_denoisers']:
                try:
                    denoised_images[method] = self.denoiser.denoise(display_image, method)
                except Exception as e:
                    print(f"Denoising error: {e}")
        
        frame_data = {
            'display': display_image,
            'enhanced': enhanced_image,
            'denoised': denoised_images,
            'samples': self.total_samples,
            'render_time': 0.0,
            'mode': 'raytracing',
            'is_raytracing': True
        }
        self.frame_queue.put(frame_data)
    
    # ==================== SCENE METHODS ====================
    
    def load_scene(self, scene_name):
        with self.render_lock:
            new_scene = self.scene_generator.create_scene(scene_name, self.texture_manager)
            if new_scene is not None:
                self.scene = new_scene
                
                # Ak je BVH zapnutý, vytvoríme ho
                if self.settings['bvh_enabled']:
                    self.scene.build_bvh()
                else:
                    self.scene.use_bvh = False
                
                # Aktualizácia C++ ray tracera a renderera
                self.ray_tracer.set_scene(self.scene)
                self.object_dragger.scene = self.scene # Ensure object dragger uses the new scene
                self.renderer.scene = self.scene   # pre wireframe/silhouette
                
                self.restart_rendering()
                return True
            return False
    
    def get_available_scenes(self):
        return self.scene_generator.get_scene_names()
    
    # ==================== BENCHMARK METHODS ====================
    
    def run_benchmarks(self):
        self.benchmark_runner = BenchmarkRunner(self)
        self.benchmark_runner.progress.connect(self._benchmark_progress)
        self.benchmark_runner.finished.connect(self._benchmark_finished)
        self.benchmark_runner.error.connect(self._benchmark_error)
        self.benchmark_runner.current_test.connect(self._benchmark_current_test)
        self.benchmark_runner.start()
    
    def _benchmark_progress(self, current, total):
        if self._gui:
            self._gui.update_benchmark_progress(current, total)
    
    def _benchmark_finished(self, csv_path):
        if self._gui:
            self._gui.benchmark_finished(csv_path)
    
    def _benchmark_error(self, error_msg):
        if self._gui:
            self._gui.benchmark_error(error_msg)

    #def _benchmark_current_test(self, scene, bvh, adaptive, subsample, neural, current, total):
    #    if self._gui:
    #        self._gui.update_current_test(scene, bvh, adaptive, subsample, neural, current, total)

    def _benchmark_current_test(self, scene, bvh, dynamic, simd, current, total):
        if self._gui:
            self._gui.update_current_test(scene, bvh, dynamic, simd, current, total)
