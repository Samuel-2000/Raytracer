import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QSlider, QCheckBox, QComboBox, QLabel, QPushButton,
                             QTabWidget, QSplitter, QProgressBar, QSpinBox, QDoubleSpinBox,
                             QColorDialog, QLineEdit, QFormLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QKeyEvent, QIntValidator
import cv2

from interaction import RayTracerInteraction, RenderMode
from cpp_raytracer.raytracer_cpp import Vector3, Material, Sphere

class RenderThread(QThread):
    """Thread for handling rendering updates"""
    frame_ready = pyqtSignal(dict)
    rendering_finished = pyqtSignal()
    
    def __init__(self, raytracer):
        super().__init__()
        self.raytracer = raytracer
        self.running = True
    
    def run(self):
        """Main rendering loop"""
        self.raytracer.start_rendering()
        
        while self.running:
            while self.raytracer.has_frames():
                frame = self.raytracer.get_frame()
                if frame is None:
                    break
                
                if 'done' in frame:
                    self.rendering_finished.emit()
                    break
                
                self.frame_ready.emit(frame)
            
            self.msleep(16)  # ~60 FPS
    
    def stop(self):
        """Stop the rendering thread"""
        self.running = False
        self.raytracer.stop_rendering()
        self.wait()

class ImageDisplay(QLabel):
    """Custom image display widget with mouse interaction"""
    mouse_moved = pyqtSignal(float, float)
    mouse_pressed = pyqtSignal(float, float, int)
    mouse_released = pyqtSignal(int)
    right_click = pyqtSignal(float, float)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #444; background-color: #1a1a1a;")
        self.setMinimumSize(400, 300)
        
        self.dragging = False
        self.drag_button = None
        self.last_pos = None
    
    def set_image(self, image_array):
        """Set image from numpy array"""
        if image_array is None or image_array.size == 0:
            return
        
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        
        image_8bit = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        
        # Remove or modify this line:
        # image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB)
        
        # Use image_8bit directly if it's already RGB:
        q_image = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image))
    
    def mousePressEvent(self, event):
        button = event.button()
        if button in [Qt.LeftButton, Qt.RightButton]:
            self.dragging = True
            self.drag_button = button
            self.last_pos = event.pos()
            
            if self.pixmap():
                pixmap_size = self.pixmap().size()
                label_size = self.size()
                
                x_offset = (label_size.width() - pixmap_size.width()) / 2
                y_offset = (label_size.height() - pixmap_size.height()) / 2
                
                norm_x = (event.x() - x_offset) / pixmap_size.width()
                norm_y = (event.y() - y_offset) / pixmap_size.height()
                
                if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                    if button == Qt.RightButton:
                        self.right_click.emit(norm_x, norm_y)
                    self.mouse_pressed.emit(norm_x, norm_y, button)
    
    def mouseReleaseEvent(self, event):
        button = event.button()
        if button == self.drag_button:
            self.dragging = False
            self.drag_button = None
            self.last_pos = None
            self.mouse_released.emit(button)
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.last_pos and self.pixmap():
            current_pos = event.pos()
            dx = current_pos.x() - self.last_pos.x()
            dy = current_pos.y() - self.last_pos.y()
            
            pixmap_size = self.pixmap().size()
            norm_dx = dx / pixmap_size.width()
            norm_dy = dy / pixmap_size.height()
            
            self.mouse_moved.emit(norm_dx, norm_dy)
            self.last_pos = current_pos

class ScrollableTabbedControlPanel(QWidget):
    """Scrollable tabbed control panel with original functionality"""
    def __init__(self, raytracer):
        super().__init__()
        self.raytracer = raytracer
        self.material_update_timer = QTimer()
        self.material_update_timer.setSingleShot(True)
        self.material_update_timer.timeout.connect(self.apply_material_changes)
        self.pending_material_changes = {}
        
        self.setup_ui()
        self.update_object_info()
        self.update_camera_info()
    
    def setup_ui(self):
        """Setup the control panel UI with tabs"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # Create tabs
        self.render_tab = self.create_render_tab()
        self.scene_tab = self.create_scene_tab()
        self.camera_tab = self.create_camera_tab()
        self.object_tab = self.create_object_tab()
        self.material_tab = self.create_material_tab()
        self.denoiser_tab = self.create_denoiser_tab()
        
        # Add tabs
        self.tabs.addTab(self.render_tab, "Render")
        self.tabs.addTab(self.scene_tab, "Scene")
        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.object_tab, "Object")
        self.tabs.addTab(self.material_tab, "Material")
        self.tabs.addTab(self.denoiser_tab, "Denoiser")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def create_render_tab(self):
        """Create rendering controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Rendering settings group
        render_group = QGroupBox("Rendering Settings")
        render_layout = QVBoxLayout()
        
        # Max Samples
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Max Samples:"))
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 1024)
        self.max_samples.setValue(self.raytracer.settings["max_samples"])
        self.max_samples.valueChanged.connect(self.on_settings_changed)
        samples_layout.addWidget(self.max_samples)
        render_layout.addLayout(samples_layout)
        
        # Samples per Batch
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Samples/Batch:"))
        self.samples_batch = QSpinBox()
        self.samples_batch.setRange(1, 64)
        self.samples_batch.setValue(self.raytracer.settings["samples_per_batch"])
        self.samples_batch.valueChanged.connect(self.on_settings_changed)
        batch_layout.addWidget(self.samples_batch)
        render_layout.addLayout(batch_layout)
        
        # Max Depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 32)
        self.max_depth.setValue(self.raytracer.settings["max_depth"])
        self.max_depth.valueChanged.connect(self.on_settings_changed)
        depth_layout.addWidget(self.max_depth)
        render_layout.addLayout(depth_layout)
        
        # Exposure
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Exposure:"))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.1, 5.0)
        self.exposure.setSingleStep(0.1)
        self.exposure.setValue(self.raytracer.settings["exposure"])
        self.exposure.valueChanged.connect(self.on_settings_changed)
        exposure_layout.addWidget(self.exposure)
        render_layout.addLayout(exposure_layout)
        
        # Enhance image
        self.enhance_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_checkbox.setChecked(self.raytracer.settings["enhance_image"])
        self.enhance_checkbox.toggled.connect(self.on_enhance_changed)
        render_layout.addWidget(self.enhance_checkbox)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        
        # Viewport resolution group
        res_group = QGroupBox("Viewport Resolution")
        res_layout = QHBoxLayout()
        self.res_w = QLineEdit(str(self.raytracer.width))
        self.res_h = QLineEdit(str(self.raytracer.height))
        self.res_w.setValidator(QIntValidator(1, 4096))
        self.res_h.setValidator(QIntValidator(1, 4096))
        res_layout.addWidget(QLabel("W:"))
        res_layout.addWidget(self.res_w)
        res_layout.addWidget(QLabel("H:"))
        res_layout.addWidget(self.res_h)
        apply_res_btn = QPushButton("Apply")
        apply_res_btn.clicked.connect(self.on_apply_resolution)
        res_layout.addWidget(apply_res_btn)
        res_group.setLayout(res_layout)
        layout.addWidget(res_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_scene_tab(self):
        """Create scene management tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Scene management group
        scene_group = QGroupBox("Scene Management")
        scene_layout = QVBoxLayout()
        
        # Object count label
        self.object_count_label = QLabel(f"Objects: {self.raytracer.get_object_count()}")
        scene_layout.addWidget(self.object_count_label)
        
        # Add object button
        add_object_btn = QPushButton("Add Sphere")
        add_object_btn.clicked.connect(self.add_object)
        scene_layout.addWidget(add_object_btn)
        
        # Remove object button
        remove_object_btn = QPushButton("Remove Selected")
        remove_object_btn.clicked.connect(self.remove_object)
        scene_layout.addWidget(remove_object_btn)
        
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)
        
        # Texture controls group
        texture_group = QGroupBox("Texture / Material")
        texture_layout = QVBoxLayout()
        
        # Texture type selection
        tex_row = QHBoxLayout()
        tex_row.addWidget(QLabel("Type:"))
        self.texture_select = QComboBox()
        self.texture_select.addItems(["none", "noise"])
        self.texture_select.currentTextChanged.connect(self.on_texture_type_changed)
        tex_row.addWidget(self.texture_select)
        texture_layout.addLayout(tex_row)
        
        # Texture parameters
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Scale:"))
        self.tex_scale = QDoubleSpinBox()
        self.tex_scale.setRange(0.01, 10.0)
        self.tex_scale.setSingleStep(0.1)
        self.tex_scale.setValue(1.0)
        param_row.addWidget(self.tex_scale)
        param_row.addWidget(QLabel("Octaves:"))
        self.tex_octaves = QSpinBox()
        self.tex_octaves.setRange(1, 8)
        self.tex_octaves.setValue(3)
        param_row.addWidget(self.tex_octaves)
        texture_layout.addLayout(param_row)
        
        # Tint controls
        tint_row = QHBoxLayout()
        tint_row.addWidget(QLabel("Tint H:"))
        self.tint_h = QSpinBox()
        self.tint_h.setRange(0, 360)
        self.tint_h.setValue(0)
        tint_row.addWidget(self.tint_h)
        tint_row.addWidget(QLabel("S:"))
        self.tint_s = QSpinBox()
        self.tint_s.setRange(0, 100)
        self.tint_s.setValue(0)
        tint_row.addWidget(self.tint_s)
        texture_layout.addLayout(tint_row)
        
        # Apply texture button
        apply_tex_btn = QPushButton("Apply Texture to Selected")
        apply_tex_btn.clicked.connect(self.apply_texture_to_selected)
        texture_layout.addWidget(apply_tex_btn)
        
        texture_group.setLayout(texture_layout)
        layout.addWidget(texture_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_camera_tab(self):
        """Create camera controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Camera position group
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout()
        
        # X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.cam_x = QDoubleSpinBox()
        self.cam_x.setRange(-20, 20)
        self.cam_x.setSingleStep(0.1)
        self.cam_x.setValue(0.0)
        self.cam_x.valueChanged.connect(self.on_camera_pos_changed)
        x_layout.addWidget(self.cam_x)
        pos_layout.addLayout(x_layout)
        
        # Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.cam_y = QDoubleSpinBox()
        self.cam_y.setRange(-20, 20)
        self.cam_y.setSingleStep(0.1)
        self.cam_y.setValue(2.0)
        self.cam_y.valueChanged.connect(self.on_camera_pos_changed)
        y_layout.addWidget(self.cam_y)
        pos_layout.addLayout(y_layout)
        
        # Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.cam_z = QDoubleSpinBox()
        self.cam_z.setRange(-20, 20)
        self.cam_z.setSingleStep(0.1)
        self.cam_z.setValue(5.0)
        self.cam_z.valueChanged.connect(self.on_camera_pos_changed)
        z_layout.addWidget(self.cam_z)
        pos_layout.addLayout(z_layout)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Camera target group
        target_group = QGroupBox("Target")
        target_layout = QVBoxLayout()
        
        # Target X
        tx_layout = QHBoxLayout()
        tx_layout.addWidget(QLabel("X:"))
        self.target_x = QDoubleSpinBox()
        self.target_x.setRange(-20, 20)
        self.target_x.setSingleStep(0.1)
        self.target_x.setValue(0.0)
        self.target_x.valueChanged.connect(self.on_camera_target_changed)
        tx_layout.addWidget(self.target_x)
        target_layout.addLayout(tx_layout)
        
        # Target Y
        ty_layout = QHBoxLayout()
        ty_layout.addWidget(QLabel("Y:"))
        self.target_y = QDoubleSpinBox()
        self.target_y.setRange(-20, 20)
        self.target_y.setSingleStep(0.1)
        self.target_y.setValue(0.0)
        self.target_y.valueChanged.connect(self.on_camera_target_changed)
        ty_layout.addWidget(self.target_y)
        target_layout.addLayout(ty_layout)
        
        # Target Z
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Z:"))
        self.target_z = QDoubleSpinBox()
        self.target_z.setRange(-20, 20)
        self.target_z.setSingleStep(0.1)
        self.target_z.setValue(-1.0)
        self.target_z.valueChanged.connect(self.on_camera_target_changed)
        tz_layout.addWidget(self.target_z)
        target_layout.addLayout(tz_layout)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Camera settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("FOV:"))
        self.fov = QDoubleSpinBox()
        self.fov.setRange(10, 120)
        self.fov.setSingleStep(1.0)
        self.fov.setValue(45.0)
        self.fov.valueChanged.connect(self.on_camera_fov_changed)
        fov_layout.addWidget(self.fov)
        settings_layout.addLayout(fov_layout)
        
        # Speed controls
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Move Speed:"))
        self.move_speed = QDoubleSpinBox()
        self.move_speed.setRange(0.01, 1.0)
        self.move_speed.setSingleStep(0.01)
        self.move_speed.setValue(self.raytracer.settings["camera_move_speed"])
        self.move_speed.valueChanged.connect(self.on_move_speed_changed)
        speed_layout.addWidget(self.move_speed)
        settings_layout.addLayout(speed_layout)
        
        # Rotate speed
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(QLabel("Rotate Speed:"))
        self.rotate_speed = QDoubleSpinBox()
        self.rotate_speed.setRange(0.01, 2.0)
        self.rotate_speed.setSingleStep(0.05)
        self.rotate_speed.setValue(self.raytracer.settings["camera_rotate_speed"])
        self.rotate_speed.valueChanged.connect(self.on_rotate_speed_changed)
        rotate_layout.addWidget(self.rotate_speed)
        settings_layout.addLayout(rotate_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self.reset_camera)
        settings_layout.addWidget(reset_btn)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_object_tab(self):
        """Create object controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Object selection group
        selection_group = QGroupBox("Object Selection")
        selection_layout = QVBoxLayout()
        
        # Object selection
        obj_layout = QHBoxLayout()
        obj_layout.addWidget(QLabel("Object:"))
        self.object_select = QComboBox()
        self.update_object_list()
        self.object_select.currentIndexChanged.connect(self.on_object_selected)
        obj_layout.addWidget(self.object_select)
        selection_layout.addLayout(obj_layout)
        
        # Object info
        self.object_info = QLabel("Selected: None")
        self.object_info.setStyleSheet("color: #aaa; font-style: italic;")
        selection_layout.addWidget(self.object_info)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Movement buttons group
        move_group = QGroupBox("Keyboard Movement")
        move_layout = QVBoxLayout()
        
        # Horizontal movement
        horiz_layout = QHBoxLayout()
        self.btn_left = QPushButton("← Left (J)")
        self.btn_right = QPushButton("Right (L) →")
        horiz_layout.addWidget(self.btn_left)
        horiz_layout.addWidget(self.btn_right)
        move_layout.addLayout(horiz_layout)
        
        # Vertical movement
        vert_layout = QHBoxLayout()
        self.btn_up = QPushButton("↑ Up (I)")
        self.btn_down = QPushButton("Down (K) ↓")
        vert_layout.addWidget(self.btn_up)
        vert_layout.addWidget(self.btn_down)
        move_layout.addLayout(vert_layout)
        
        # Depth movement
        depth_layout = QHBoxLayout()
        self.btn_forward = QPushButton("↗ Forward (U)")
        self.btn_backward = QPushButton("Backward (O) ↙")
        depth_layout.addWidget(self.btn_forward)
        depth_layout.addWidget(self.btn_backward)
        move_layout.addLayout(depth_layout)
        
        # Object move speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Object Speed:"))
        self.object_speed = QDoubleSpinBox()
        self.object_speed.setRange(0.01, 2.0)
        self.object_speed.setSingleStep(0.05)
        self.object_speed.setValue(self.raytracer.settings["move_speed"])
        self.object_speed.valueChanged.connect(self.on_object_speed_changed)
        speed_layout.addWidget(self.object_speed)
        move_layout.addLayout(speed_layout)
        
        move_group.setLayout(move_layout)
        layout.addWidget(move_group)
        
        # Connect buttons
        self.btn_left.clicked.connect(lambda: self._move_object(-1, 0, 0))
        self.btn_right.clicked.connect(lambda: self._move_object(1, 0, 0))
        self.btn_up.clicked.connect(lambda: self._move_object(0, 1, 0))
        self.btn_down.clicked.connect(lambda: self._move_object(0, -1, 0))
        self.btn_forward.clicked.connect(lambda: self._move_object(0, 0, -1))
        self.btn_backward.clicked.connect(lambda: self._move_object(0, 0, 1))
        
        # Dimension locks
        lock_group = QGroupBox("Dimension Locks (for dragging)")
        lock_layout = QHBoxLayout()
        self.lock_x = QCheckBox("X")
        self.lock_y = QCheckBox("Y")
        self.lock_z = QCheckBox("Z")
        self.lock_x.toggled.connect(lambda s: self.raytracer.set_dimension_lock('x', s))
        self.lock_y.toggled.connect(lambda s: self.raytracer.set_dimension_lock('y', s))
        self.lock_z.toggled.connect(lambda s: self.raytracer.set_dimension_lock('z', s))
        lock_layout.addWidget(self.lock_x)
        lock_layout.addWidget(self.lock_y)
        lock_layout.addWidget(self.lock_z)
        lock_group.setLayout(lock_layout)
        layout.addWidget(lock_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_material_tab(self):
        """Create material controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Color controls group
        color_group = QGroupBox("Color")
        color_layout = QVBoxLayout()
        
        # Albedo (color) controls
        # Red
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("R:"))
        self.color_r = QSlider(Qt.Horizontal)
        self.color_r.setRange(0, 100)
        self.color_r.setValue(90)
        self.color_r.sliderReleased.connect(self.on_material_slider_released)
        self.color_r.valueChanged.connect(self.on_material_value_changed)
        r_layout.addWidget(self.color_r)
        color_layout.addLayout(r_layout)
        
        # Green
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("G:"))
        self.color_g = QSlider(Qt.Horizontal)
        self.color_g.setRange(0, 100)
        self.color_g.setValue(90)
        self.color_g.sliderReleased.connect(self.on_material_slider_released)
        self.color_g.valueChanged.connect(self.on_material_value_changed)
        g_layout.addWidget(self.color_g)
        color_layout.addLayout(g_layout)
        
        # Blue
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("B:"))
        self.color_b = QSlider(Qt.Horizontal)
        self.color_b.setRange(0, 100)
        self.color_b.setValue(90)
        self.color_b.sliderReleased.connect(self.on_material_slider_released)
        self.color_b.valueChanged.connect(self.on_material_value_changed)
        b_layout.addWidget(self.color_b)
        color_layout.addLayout(b_layout)
        
        # Color picker button
        picker_layout = QHBoxLayout()
        picker_btn = QPushButton("Open Color Picker")
        picker_btn.clicked.connect(self.open_color_picker)
        picker_layout.addWidget(picker_btn)
        color_layout.addLayout(picker_layout)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # HSV picker group
        hsv_group = QGroupBox("HSV Picker")
        hsv_layout = QFormLayout()
        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.setRange(0, 360)
        self.h_slider.setValue(0)
        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setRange(0, 100)
        self.s_slider.setValue(100)
        self.v_slider = QSlider(Qt.Horizontal)
        self.v_slider.setRange(0, 100)
        self.v_slider.setValue(90)
        
        for s in (self.h_slider, self.s_slider, self.v_slider):
            s.valueChanged.connect(self.on_hsv_changed)
        
        hsv_layout.addRow("Hue", self.h_slider)
        hsv_layout.addRow("Saturation", self.s_slider)
        hsv_layout.addRow("Value", self.v_slider)
        
        apply_hsv_btn = QPushButton("Apply HSV to Selected")
        apply_hsv_btn.clicked.connect(self.apply_hsv_to_selected)
        hsv_layout.addRow(apply_hsv_btn)
        
        hsv_group.setLayout(hsv_layout)
        layout.addWidget(hsv_group)
        
        # Material properties group (only show for non-lights)
        self.material_props_group = QGroupBox("Material Properties")
        material_props_layout = QVBoxLayout()
        
        # Metallic with delayed update
        metallic_layout = QHBoxLayout()
        metallic_layout.addWidget(QLabel("Metallic:"))
        self.metallic = QSlider(Qt.Horizontal)
        self.metallic.setRange(0, 100)
        self.metallic.setValue(90)
        self.metallic.sliderReleased.connect(self.on_material_slider_released)
        self.metallic.valueChanged.connect(self.on_material_value_changed)
        metallic_layout.addWidget(self.metallic)
        material_props_layout.addLayout(metallic_layout)
        
        # Roughness with delayed update
        roughness_layout = QHBoxLayout()
        roughness_layout.addWidget(QLabel("Roughness:"))
        self.roughness = QSlider(Qt.Horizontal)
        self.roughness.setRange(0, 100)
        self.roughness.setValue(10)
        self.roughness.sliderReleased.connect(self.on_material_slider_released)
        self.roughness.valueChanged.connect(self.on_material_value_changed)
        roughness_layout.addWidget(self.roughness)
        material_props_layout.addLayout(roughness_layout)
        
        self.material_props_group.setLayout(material_props_layout)
        layout.addWidget(self.material_props_group)
        
        # Light intensity group (only show for lights)
        self.light_group = QGroupBox("Light Properties")
        light_layout = QVBoxLayout()
        
        light_intensity_layout = QHBoxLayout()
        light_intensity_layout.addWidget(QLabel("Light Power:"))
        self.light_intensity = QDoubleSpinBox()
        self.light_intensity.setRange(0.1, 100.0)
        self.light_intensity.setSingleStep(0.5)
        self.light_intensity.setValue(15.0)
        self.light_intensity.setDecimals(1)
        self.light_intensity.valueChanged.connect(self.on_light_intensity_changed)
        light_intensity_layout.addWidget(self.light_intensity)
        light_layout.addLayout(light_intensity_layout)
        
        self.light_group.setLayout(light_layout)
        layout.addWidget(self.light_group)
        
        # Initially hide both groups until we know object type
        self.material_props_group.setVisible(False)
        self.light_group.setVisible(False)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_denoiser_tab(self):
        """Create denoiser controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Denoiser settings group
        denoiser_group = QGroupBox("Denoiser Settings")
        denoiser_layout = QVBoxLayout()
        
        self.show_denoisers = QCheckBox("Show Denoisers")
        self.show_denoisers.setChecked(self.raytracer.settings['show_denoisers'])
        self.show_denoisers.toggled.connect(self.on_show_denoisers_changed)
        denoiser_layout.addWidget(self.show_denoisers)
        
        denoiser_methods_layout = QVBoxLayout()
        denoiser_methods_layout.addWidget(QLabel("Denoiser Methods:"))
        
        self.denoiser_bilateral = QCheckBox("Bilateral")
        self.denoiser_bilateral.setChecked('bilateral' in self.raytracer.settings['selected_denoisers'])
        self.denoiser_bilateral.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_methods_layout.addWidget(self.denoiser_bilateral)
        
        self.denoiser_nlmeans = QCheckBox("NL-Means")
        self.denoiser_nlmeans.setChecked('nlmeans' in self.raytracer.settings['selected_denoisers'])
        self.denoiser_nlmeans.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_methods_layout.addWidget(self.denoiser_nlmeans)
        
        self.denoiser_gaussian = QCheckBox("Gaussian")
        self.denoiser_gaussian.setChecked('gaussian' in self.raytracer.settings['selected_denoisers'])
        self.denoiser_gaussian.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_methods_layout.addWidget(self.denoiser_gaussian)
        
        self.denoiser_median = QCheckBox("Median")
        self.denoiser_median.setChecked('median' in self.raytracer.settings['selected_denoisers'])
        self.denoiser_median.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_methods_layout.addWidget(self.denoiser_median)
        
        denoiser_layout.addLayout(denoiser_methods_layout)
        denoiser_group.setLayout(denoiser_layout)
        layout.addWidget(denoiser_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    
    def on_settings_changed(self):
        """Handle settings changes"""
        self.raytracer.settings['max_samples'] = self.max_samples.value()
        self.raytracer.settings['samples_per_batch'] = self.samples_batch.value()
        self.raytracer.settings['max_depth'] = self.max_depth.value()
        self.raytracer.settings['exposure'] = self.exposure.value()
        self.raytracer.restart_rendering()
    
    def on_enhance_changed(self, checked):
        """Handle enhance contrast toggle"""
        self.raytracer.settings['enhance_image'] = checked
    
    def on_camera_pos_changed(self):
        """Update camera position"""
        pos = Vector3(self.cam_x.value(), self.cam_y.value(), self.cam_z.value())
        self.raytracer.camera.position = pos
        
        # Update ray tracer camera
        rt_camera = self.raytracer.ray_tracer.get_camera()
        rt_camera.position = pos
        
        # Update camera controller frame
        self.raytracer.camera_controller.update_camera_frame()
        self.raytracer.restart_rendering()
    
    def on_camera_target_changed(self):
        """Update camera target"""
        target = Vector3(self.target_x.value(), self.target_y.value(), self.target_z.value())
        self.raytracer.camera.target = target
        
        # Update ray tracer camera
        rt_camera = self.raytracer.ray_tracer.get_camera()
        rt_camera.target = target
        
        # Update camera controller frame
        self.raytracer.camera_controller.update_camera_frame()
        self.raytracer.restart_rendering()
    
    def on_camera_fov_changed(self):
        """Update camera FOV"""
        self.raytracer.camera.fov = self.fov.value()
        
        # Update ray tracer camera
        rt_camera = self.raytracer.ray_tracer.get_camera()
        rt_camera.fov = self.fov.value()
        
        self.raytracer.restart_rendering()
    
    def on_move_speed_changed(self):
        """Update camera movement speed"""
        self.raytracer.settings['camera_move_speed'] = self.move_speed.value()
        self.raytracer.camera_controller.settings['camera_move_speed'] = self.move_speed.value()
    
    def on_rotate_speed_changed(self):
        """Update camera rotation speed"""
        self.raytracer.settings['camera_rotate_speed'] = self.rotate_speed.value()
        self.raytracer.camera_controller.settings['camera_rotate_speed'] = self.rotate_speed.value()
    
    def on_object_speed_changed(self):
        """Update object movement speed"""
        self.raytracer.settings['move_speed'] = self.object_speed.value()
    
    def reset_camera(self):
        """Reset camera to default"""
        self.raytracer.reset_camera_and_rerender()
        self.update_camera_info()
    
    def update_camera_info(self):
        """Update camera controls from current camera"""
        camera = self.raytracer.camera
        if camera:
            self.cam_x.blockSignals(True)
            self.cam_y.blockSignals(True)
            self.cam_z.blockSignals(True)
            self.target_x.blockSignals(True)
            self.target_y.blockSignals(True)
            self.target_z.blockSignals(True)
            self.fov.blockSignals(True)
            
            self.cam_x.setValue(camera.position.x)
            self.cam_y.setValue(camera.position.y)
            self.cam_z.setValue(camera.position.z)
            self.target_x.setValue(camera.target.x)
            self.target_y.setValue(camera.target.y)
            self.target_z.setValue(camera.target.z)
            self.fov.setValue(camera.fov)
            
            self.cam_x.blockSignals(False)
            self.cam_y.blockSignals(False)
            self.cam_z.blockSignals(False)
            self.target_x.blockSignals(False)
            self.target_y.blockSignals(False)
            self.target_z.blockSignals(False)
            self.fov.blockSignals(False)
    
    def on_object_selected(self, index):
        """Handle object selection"""
        self.raytracer.settings['selected_object'] = index
        self.update_object_info()
        self.update_material_sliders()
    
    def update_object_info(self):
        """Update object information display"""
        obj = self.raytracer.get_selected_object()
        if obj:
            name = obj.name if hasattr(obj, 'name') and obj.name else f"Object {obj.object_id}"
            pos = obj.center
            self.object_info.setText(f"Selected: {name} at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        else:
            self.object_info.setText("Selected: None")
    
    def update_material_sliders(self):
        """Update material sliders to match selected object"""
        obj = self.raytracer.get_selected_object()
        
        if obj:
            mat = obj.material
            
            self.metallic.blockSignals(True)
            self.roughness.blockSignals(True)
            self.color_r.blockSignals(True)
            self.color_g.blockSignals(True)
            self.color_b.blockSignals(True)
            self.light_intensity.blockSignals(True)
            
            if hasattr(mat.albedo, 'x'):
                # Convert from Vector3(x=R, y=G, z=B) to RGB sliders
                self.color_r.setValue(int(mat.albedo.x * 100))  # Red channel
                self.color_g.setValue(int(mat.albedo.y * 100))  # Green channel
                self.color_b.setValue(int(mat.albedo.z * 100))  # Blue channel
            
            # Check if it's a light source
            is_light = False
            if hasattr(mat, 'emission'):
                emission = mat.emission
                is_light = emission.x > 0.1 or emission.y > 0.1 or emission.z > 0.1
            
            # Show/hide appropriate controls
            if is_light:
                # For lights: show light controls, hide material controls
                self.material_props_group.setVisible(False)
                self.light_group.setVisible(True)
                
                # Update light intensity
                if hasattr(mat.emission, 'x'):
                    emission = mat.emission
                    avg_emission = (emission.x + emission.y + emission.z) / 3.0
                    self.light_intensity.setValue(avg_emission)
            else:
                # For non-lights: show material controls, hide light controls
                self.material_props_group.setVisible(True)
                self.light_group.setVisible(False)
                
                # Update material properties
                self.metallic.setValue(int(mat.metallic * 100))
                self.roughness.setValue(int(mat.roughness * 100))
            
            self.metallic.blockSignals(False)
            self.roughness.blockSignals(False)
            self.color_r.blockSignals(False)
            self.color_g.blockSignals(False)
            self.color_b.blockSignals(False)
            self.light_intensity.blockSignals(False)
    
    def on_material_value_changed(self):
        """Handle material value changes without immediate update"""
        # Just update display, don't apply changes yet
        self.material_update_timer.stop()
        self.material_update_timer.start(1000)  # 1 second delay
    
    def on_material_slider_released(self):
        """Apply material changes when slider is released"""
        self.apply_material_changes()
    
    def apply_material_changes(self):
        """Apply all pending material changes"""
        if self.material_update_timer.isActive():
            self.material_update_timer.stop()
        
        obj = self.raytracer.get_selected_object()
        if obj:
            # RGB sliders to Vector3(R, G, B)
            r = self.color_r.value() / 100.0
            g = self.color_g.value() / 100.0
            b = self.color_b.value() / 100.0
            obj.material.albedo = Vector3(r, g, b)
            
            # Check if it's a light source
            is_light = False
            if hasattr(obj.material, 'emission'):
                emission = obj.material.emission
                is_light = emission.x > 0.1 or emission.y > 0.1 or emission.z > 0.1
            
            # Only update metallic/roughness for non-lights
            if not is_light:
                obj.material.metallic = self.metallic.value() / 100.0
                obj.material.roughness = self.roughness.value() / 100.0
                # Update the material in the ray tracer
                self.raytracer.ray_tracer.set_scene(self.raytracer.scene)
                self.raytracer.restart_rendering()
            else:
                # For lights, update emission color to match albedo with current intensity
                current_intensity = (obj.material.emission.x + 
                                   obj.material.emission.y + 
                                   obj.material.emission.z) / 3.0
                if current_intensity > 0:
                    obj.material.emission = Vector3(
                        r * current_intensity,
                        g * current_intensity,
                        b * current_intensity
                    )
                    # Update the scene
                    self.raytracer.ray_tracer.set_scene(self.raytracer.scene)
                    self.raytracer.restart_rendering()
    
    def on_light_intensity_changed(self, value):
        """Handle light intensity changes"""
        obj = self.raytracer.get_selected_object()
        if obj and hasattr(obj.material, 'emission'):
            emission = obj.material.emission
            
            # Check if it's a light (has non-zero emission)
            is_light = emission.x > 0.1 or emission.y > 0.1 or emission.z > 0.1
            
            if is_light:
                # Get current albedo color
                albedo = obj.material.albedo
                
                # Scale the albedo by the intensity to get emission color
                obj.material.emission = Vector3(
                    albedo.x * value,
                    albedo.y * value,
                    albedo.z * value
                )
                
                # Update the scene
                self.raytracer.ray_tracer.set_scene(self.raytracer.scene)
                self.raytracer.restart_rendering()
    
    def on_show_denoisers_changed(self, checked):
        """Handle show denoisers toggle"""
        self.raytracer.settings['show_denoisers'] = checked
    
    def on_denoiser_selection_changed(self):
        """Handle denoiser selection changes"""
        selected = []
        if self.denoiser_bilateral.isChecked():
            selected.append('bilateral')
        if self.denoiser_nlmeans.isChecked():
            selected.append('nlmeans')
        if self.denoiser_gaussian.isChecked():
            selected.append('gaussian')
        if self.denoiser_median.isChecked():
            selected.append('median')
        
        self.raytracer.settings['selected_denoisers'] = selected
    
    # ------------------------------------------------------------------
    # Object Management Methods
    # ------------------------------------------------------------------
    
    def _move_object(self, dx, dy, dz):
        """Helper method to move selected object"""
        self.raytracer.move_object(dx, dy, dz)
        self.update_object_info()
    
    def update_object_list(self):
        """Update object dropdown list"""
        current_index = self.object_select.currentIndex()
        self.object_select.clear()
        
        # Populate objects
        for i in range(self.raytracer.get_object_count() + 1):
            sphere = self.raytracer._get_sphere_by_id(i)
            if sphere:
                if hasattr(sphere, 'name') and sphere.name:
                    self.object_select.addItem(sphere.name)
                else:
                    name = "Ground" if i == 0 else f"Object {i}"
                    self.object_select.addItem(name)
            else:
                name = "Ground" if i == 0 else f"Object {i}"
                self.object_select.addItem(name)
        
        # Restore selection
        if current_index < self.object_select.count():
            self.object_select.setCurrentIndex(current_index)
        else:
            self.object_select.setCurrentIndex(self.raytracer.settings['selected_object'])
    
    def update_object_count(self):
        """Update object count display"""
        count = self.raytracer.get_object_count()
        self.object_count_label.setText(f"Objects: {count}")
    
    def add_object(self):
        """Add a new object to the scene"""
        try:
            # Generate unique name and ID
            obj_count = self.raytracer.get_object_count()
            new_id = self.raytracer.add_object_to_scene()
            
            if new_id is None:
                print("Failed to add object")
                return
            
            # Update UI
            self.update_object_list()
            self.update_object_count()
            
            # Select the new object
            self.object_select.setCurrentIndex(new_id)
            self.on_object_selected(new_id)
            
            print(f"Added object with ID: {new_id}")
            
        except Exception as e:
            print(f"Error adding object: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_object(self):
        """Remove selected object from scene"""
        try:
            obj = self.raytracer.get_selected_object()
            if obj and obj.object_id > 0:  # Don't remove ground
                print(f"Removing object: {obj.object_id}")
                success = self.raytracer.remove_object_from_scene(obj.object_id)
                if success:
                    self.update_object_list()
                    self.update_object_count()
                    # Update the selection to the first available object
                    if self.object_select.count() > 0:
                        self.object_select.setCurrentIndex(0)
                        self.on_object_selected(0)
                else:
                    print("Failed to remove object")
        except Exception as e:
            print(f"Error removing object: {e}")
            import traceback
            traceback.print_exc()
    
    # ------------------------------------------------------------------
    # New functionality from interaction.py
    # ------------------------------------------------------------------
    
    def open_color_picker(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor()
        if not color.isValid():
            return
        r = color.red() / 255.0
        g = color.green() / 255.0
        b = color.blue() / 255.0

        # Update sliders to match
        self.color_r.blockSignals(True)
        self.color_g.blockSignals(True)
        self.color_b.blockSignals(True)
        self.color_r.setValue(int(r * 100))
        self.color_g.setValue(int(g * 100))
        self.color_b.setValue(int(b * 100))
        self.color_r.blockSignals(False)
        self.color_g.blockSignals(False)
        self.color_b.blockSignals(False)

        # Apply the color
        self.raytracer.set_object_color(r, g, b)
    
    def on_hsv_changed(self, _=None):
        """Update RGB sliders to show selected HSV"""
        h = self.h_slider.value()
        s = self.s_slider.value() / 100.0
        v = self.v_slider.value() / 100.0
        
        # Convert HSV->RGB
        h_norm = (h % 360) / 360.0
        i = int(h_norm * 6)
        f = h_norm * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i_mod = i % 6
        
        if i_mod == 0:
            r, g, b = v, t, p
        elif i_mod == 1:
            r, g, b = q, v, p
        elif i_mod == 2:
            r, g, b = p, v, t
        elif i_mod == 3:
            r, g, b = p, q, v
        elif i_mod == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        # Update RGB sliders (but don't trigger material update)
        self.color_r.blockSignals(True)
        self.color_g.blockSignals(True)
        self.color_b.blockSignals(True)
        self.color_r.setValue(int(r * 100))
        self.color_g.setValue(int(g * 100))
        self.color_b.setValue(int(b * 100))
        self.color_r.blockSignals(False)
        self.color_g.blockSignals(False)
        self.color_b.blockSignals(False)
    
    def apply_hsv_to_selected(self):
        """Apply HSV color to selected object"""
        h = self.h_slider.value()
        s = self.s_slider.value() / 100.0
        v = self.v_slider.value() / 100.0
        self.raytracer.set_object_color_hsv(h, s, v)
    
    def on_texture_type_changed(self, txt):
        """Handle texture type change"""
        # Currently just a placeholder
        pass
    
    def apply_texture_to_selected(self):
        """Apply texture to selected object"""
        tex_type = self.texture_select.currentText()
        params = {
            'scale': float(self.tex_scale.value()),
            'octaves': int(self.tex_octaves.value()),
            'tint_hsv': (int(self.tint_h.value()), int(self.tint_s.value())/100.0, 1.0) if self.tint_s.value() > 0 else None
        }
        success = self.raytracer.set_object_texture(tex_type, params)
        if not success:
            print("Texture apply failed or unknown texture type")
    
    def on_apply_resolution(self):
        """Apply new viewport resolution"""
        try:
            w = int(self.res_w.text())
            h = int(self.res_h.text())
            if w <= 0 or h <= 0:
                raise ValueError("Invalid resolution")
            self.raytracer.resize_viewport(w, h)
        except Exception as e:
            print(f"Invalid resolution: {e}")

class GUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.raytracer = RayTracerInteraction(640, 480)
        self.render_thread = None
        
        # Give raytracer a reference to the GUI for callbacks
        self.raytracer._gui = self
        
        # Camera key mapping
        self.camera_keys = {
            Qt.Key_W: 'forward',
            Qt.Key_S: 'backward',
            Qt.Key_A: 'left',
            Qt.Key_D: 'right',
            Qt.Key_Space: 'up',
            Qt.Key_Control: 'down',
        }
        
        # Object movement keys (different from camera keys)
        self.object_keys = {
            Qt.Key_I: (0, 1, 0),   # Up
            Qt.Key_K: (0, -1, 0),  # Down
            Qt.Key_J: (-1, 0, 0),  # Left
            Qt.Key_L: (1, 0, 0),   # Right
            Qt.Key_U: (0, 0, -1),  # Forward
            Qt.Key_O: (0, 0, 1),   # Backward
        }
        
        # Object dragging state
        self.dragging_object = False
        self.dimension_locks = {'x': False, 'y': False, 'z': False}
        
        # Track manual mode changes vs automatic ones
        self.manual_mode_change = False
        
        self.setup_ui()
        self.setup_rendering()

        # Timer for updating camera controls
        self.camera_update_timer = QTimer()
        self.camera_update_timer.timeout.connect(self.update_camera_controls)
        self.camera_update_timer.start(100)  # Update every 100ms
        
        # Fix: Prevent auto-movement by clearing key states on focus loss
        self._key_states_cleared = False
    
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("C++ Ray Tracer - Interactive Controls")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side - Image displays
        left_widget = self.create_image_displays()
        main_layout.addWidget(left_widget, 3)
        
        # Right side - Scrollable control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMaximumWidth(500)
        
        self.control_panel = ScrollableTabbedControlPanel(self.raytracer)
        scroll_area.setWidget(self.control_panel)
        
        main_layout.addWidget(scroll_area, 1)
        
        # Status bar
        self.status_label = QLabel("Ready to render...")
        self.statusBar().addWidget(self.status_label)
        
        # Mode indicator
        self.mode_label = QLabel("Mode: Ray Tracing")
        self.mode_label.setStyleSheet("color: #88c; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.mode_label)
        
        # Lock status
        self.lock_label = QLabel("Locks: None")
        self.statusBar().addPermanentWidget(self.lock_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Focus policy
        self.setFocusPolicy(Qt.StrongFocus)
    
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #88c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #88c;
                border: 1px solid #55a;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #666;
                background: #333;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #88c;
                background: #55a;
            }
            QComboBox {
                border: 1px solid #555;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 6em;
                background: #333;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #555;
                border-radius: 3px;
                padding: 1px;
                background: #333;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #333;
                color: #aaa;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #444;
                color: #fff;
                border-bottom: 2px solid #88c;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
                color: white;
            }
            QPushButton:hover {
                background-color: #555;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollBar:vertical {
                border: none;
                background: #333;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
        """)
    
    def create_image_displays(self):
        """Create the image display area"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Mode selector
        mode_widget = QWidget()
        mode_layout = QHBoxLayout()
        mode_widget.setLayout(mode_layout)
        
        self.raytrace_btn = QPushButton("Ray Tracing")
        self.raytrace_btn.setCheckable(True)
        self.raytrace_btn.setChecked(True)
        self.raytrace_btn.clicked.connect(self.on_raytrace_mode)
        mode_layout.addWidget(self.raytrace_btn)
        
        self.wireframe_btn = QPushButton("Wireframe")
        self.wireframe_btn.setCheckable(True)
        self.wireframe_btn.clicked.connect(self.on_wireframe_mode)
        mode_layout.addWidget(self.wireframe_btn)
        
        self.silhouette_btn = QPushButton("Silhouette")
        self.silhouette_btn.setCheckable(True)
        self.silhouette_btn.clicked.connect(self.on_silhouette_mode)
        mode_layout.addWidget(self.silhouette_btn)
        
        mode_layout.addStretch()
        layout.addWidget(mode_widget)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Main view tab
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        main_tab.setLayout(main_layout)
        
        self.main_display = ImageDisplay()
        self.main_display.mouse_pressed.connect(self.on_mouse_press)
        self.main_display.mouse_moved.connect(self.on_mouse_drag)
        self.main_display.mouse_released.connect(self.on_mouse_release)
        self.main_display.right_click.connect(self.on_right_click)
        main_layout.addWidget(self.main_display)
        
        self.tabs.addTab(main_tab, "Main View")
        
        # Enhanced view tab
        enhanced_tab = QWidget()
        enhanced_layout = QVBoxLayout()
        enhanced_tab.setLayout(enhanced_layout)
        
        self.enhanced_display = ImageDisplay()
        enhanced_layout.addWidget(self.enhanced_display)
        
        self.tabs.addTab(enhanced_tab, "Enhanced View")
        
        # Denoiser views tab
        denoiser_tab = QWidget()
        denoiser_layout = QVBoxLayout()
        denoiser_tab.setLayout(denoiser_layout)
        
        denoiser_grid = QHBoxLayout()
        
        self.denoiser_displays = {}
        methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
        for method in methods:
            display_widget = QWidget()
            display_layout = QVBoxLayout()
            display_widget.setLayout(display_layout)
            
            label = QLabel(f"{method.title()} Denoised")
            label.setAlignment(Qt.AlignCenter)
            display_layout.addWidget(label)
            
            display = ImageDisplay()
            display.setMinimumSize(300, 200)
            display_layout.addWidget(display)
            
            denoiser_grid.addWidget(display_widget)
            self.denoiser_displays[method] = display
        
        denoiser_layout.addLayout(denoiser_grid)
        self.tabs.addTab(denoiser_tab, "Denoiser Views")
        
        layout.addWidget(self.tabs)
        
        # Instructions
        instructions = QLabel(
            "<b>Controls:</b> WASD+Space/Ctrl to move camera | "
            "<b>Right Click + Drag</b> to rotate camera | "
            "<b>Hold X/Y/Z + Left Click + Drag</b> to move object | "
            "<b>IJKLOU</b> to move selected object | "
            "<b>ESC</b> to cancel"
        )
        instructions.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 10px;
                padding: 5px;
                background-color: #222;
                border-radius: 3px;
            }
        """)
        layout.addWidget(instructions)
        
        return widget
    
    def setup_rendering(self):
        """Setup rendering thread"""
        self.render_thread = RenderThread(self.raytracer)
        self.render_thread.frame_ready.connect(self.on_frame_ready)
        self.render_thread.rendering_finished.connect(self.on_rendering_finished)
        self.render_thread.start()

    def update_camera_controls(self):
        """Update camera control values from current camera state"""
        if self.raytracer.camera:
            camera = self.raytracer.camera
            
            # Update position controls
            self.control_panel.cam_x.blockSignals(True)
            self.control_panel.cam_y.blockSignals(True)
            self.control_panel.cam_z.blockSignals(True)
            
            self.control_panel.cam_x.setValue(camera.position.x)
            self.control_panel.cam_y.setValue(camera.position.y)
            self.control_panel.cam_z.setValue(camera.position.z)
            
            self.control_panel.cam_x.blockSignals(False)
            self.control_panel.cam_y.blockSignals(False)
            self.control_panel.cam_z.blockSignals(False)
            
            # Update target controls
            self.control_panel.target_x.blockSignals(True)
            self.control_panel.target_y.blockSignals(True)
            self.control_panel.target_z.blockSignals(True)
            
            self.control_panel.target_x.setValue(camera.target.x)
            self.control_panel.target_y.setValue(camera.target.y)
            self.control_panel.target_z.setValue(camera.target.z)
            
            self.control_panel.target_x.blockSignals(False)
            self.control_panel.target_y.blockSignals(False)
            self.control_panel.target_z.blockSignals(False)
    
    def on_raytrace_mode(self):
        """Switch to ray tracing mode"""
        self.raytrace_btn.setChecked(True)
        self.wireframe_btn.setChecked(False)
        self.silhouette_btn.setChecked(False)
        self.mode_label.setText("Mode: Ray Tracing")
        self.mode_label.setStyleSheet("color: #88c; font-weight: bold;")
        
        # Update render state
        self.raytracer.render_state.set_mode(RenderMode.RAYTRACING)
        self.raytracer.restart_rendering()
    
    def on_wireframe_mode(self):
        """Switch to wireframe mode - manual"""
        self.manual_mode_change = True
        self.raytrace_btn.setChecked(False)
        self.wireframe_btn.setChecked(True)
        self.silhouette_btn.setChecked(False)
        
        self.mode_label.setText("Mode: Wireframe")
        self.mode_label.setStyleSheet("color: #0f0; font-weight: bold;")
        
        # Update render state
        self.raytracer.render_state.set_mode(RenderMode.WIREFRAME)
        self.raytracer.render_state.previous_mode = RenderMode.WIREFRAME
        
        # Force a frame update
        self.raytracer._process_frame_for_display(0.016)
        self.manual_mode_change = False
    
    def on_silhouette_mode(self):
        """Switch to silhouette mode - manual"""
        self.manual_mode_change = True
        self.raytrace_btn.setChecked(False)
        self.wireframe_btn.setChecked(False)
        self.silhouette_btn.setChecked(True)
        
        self.mode_label.setText("Mode: Silhouette")
        self.mode_label.setStyleSheet("color: #ff0; font-weight: bold;")
        
        # Update render state
        self.raytracer.render_state.set_mode(RenderMode.SILHOUETTE)
        self.raytracer.render_state.previous_mode = RenderMode.SILHOUETTE
        
        # Force a frame update
        self.raytracer._process_frame_for_display(0.016)
        self.manual_mode_change = False
    
    def on_frame_ready(self, frame_data):
        """Handle new frame from render thread"""
        # Update displays
        self.main_display.set_image(frame_data['display'])
        self.enhanced_display.set_image(frame_data['enhanced'])
        
        # Update denoiser displays if needed
        if 'denoised' in frame_data:
            for method, image in frame_data['denoised'].items():
                if method in self.denoiser_displays:
                    self.denoiser_displays[method].set_image(image)
        
        # Update status
        mode = frame_data.get('mode', 'raytracing')
        if mode == 'wireframe':
            status = "Wireframe Mode - Right Drag to Rotate, WASD to Move"
        elif mode == 'silhouette':
            if self.dragging_object:
                locks = self.get_lock_string()
                status = f"Dragging Object - Locks: {locks}"
            else:
                status = "Silhouette Mode - Hold X/Y/Z + Drag to Move Objects"
        else:
            if frame_data['is_raytracing']:
                status = (f"Samples: {frame_data['samples']} | "
                         f"Batch Time: {frame_data['render_time']:.3f}s")
            else:
                status = "Ray Tracing Mode"
        
        self.status_label.setText(status)
        
        # Update progress bar
        if frame_data.get('is_raytracing', False):
            max_samples = self.raytracer.settings['max_samples']
            progress = min(100, int((frame_data['samples'] / max_samples) * 100))
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(progress < 100)
        else:
            self.progress_bar.setVisible(False)
    
    def on_rendering_finished(self):
        """Handle rendering completion"""
        self.status_label.setText("Rendering Complete!")
        self.progress_bar.setVisible(False)
    
    def on_mouse_press(self, x: float, y: float, button: int):
        """Handle mouse press"""
        if button == Qt.LeftButton:
            # Check if any dimension is locked
            any_lock = any(self.dimension_locks.values())
            
            if any_lock:
                # Start object dragging
                if self.raytracer.start_object_dragging(x, y):
                    self.dragging_object = True
                    # Only switch to silhouette if not already in it
                    if not self.silhouette_btn.isChecked() and not self.manual_mode_change:
                        self.on_silhouette_mode()
            else:
                # Simple object selection
                if self.raytracer.select_object_by_click(x, y):
                    # Update control panel
                    idx = self.raytracer.settings['selected_object']
                    self.control_panel.object_select.setCurrentIndex(idx)
                    self.control_panel.update_object_info()
                    self.control_panel.update_material_sliders()
        
        elif button == Qt.RightButton:
            # Start camera rotation
            self.raytracer.start_camera_rotation(x, y)
            # Only switch to wireframe if not manually in another mode
            if not self.wireframe_btn.isChecked() and not self.manual_mode_change:
                self.on_wireframe_mode()
    
    def on_right_click(self, x: float, y: float):
        """Handle right click (alternative)"""
        self.raytracer.start_camera_rotation(x, y)
        if not self.wireframe_btn.isChecked():
            self.on_wireframe_mode()
    
    def on_mouse_drag(self, dx: float, dy: float):
        """Handle mouse dragging"""
        if self.dragging_object:
            # Object dragging
            self.raytracer.update_object_dragging(dx, dy)
            
            # Update object info
            obj = self.raytracer.get_selected_object()
            if obj:
                pos = obj.center
                name = obj.name if hasattr(obj, 'name') and obj.name else f"Object {obj.object_id}"
                self.control_panel.object_info.setText(
                    f"Dragging: {name} at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
                )
        
        elif self.raytracer.camera_controller.rotating:
            # Camera rotation
            self.raytracer.update_camera_rotation(dx, dy)
    
    def on_mouse_release(self, button: int):
        """Handle mouse release with proper mode restoration"""
        if button == Qt.LeftButton and self.dragging_object:
            self.raytracer.stop_object_dragging()
            self.dragging_object = False
            self.dimension_locks = {'x': False, 'y': False, 'z': False}
            self.update_lock_status()
            
            # Update control panel
            self.control_panel.update_object_info()
            self.control_panel.update_material_sliders()
            
            # Always return to ray tracing after dragging
            self.on_raytrace_mode()
        
        elif button == Qt.RightButton:
            self.raytracer.stop_camera_rotation()
            # Always return to ray tracing after camera rotation
            self.on_raytrace_mode()
    
    def keyPressEvent(self, event):
        """Handle keyboard input with better debouncing"""
        key = event.key()

        self.manual_mode_change = False
        
        # Camera movement keys
        if key in self.camera_keys:
            key_name = self.camera_keys[key]
            # Only send if key state is changing
            if not self.raytracer.camera_controller.keys_pressed.get(key_name, False):
                self.raytracer.set_camera_key_state(key_name, True)
            event.accept()
            return
        
        # Object movement keys
        if key in self.object_keys:
            dx, dy, dz = self.object_keys[key]
            self.control_panel._move_object(dx, dy, dz)
            event.accept()
            return
        
        # Dimension locking for object dragging
        if key == Qt.Key_X:
            self.dimension_locks['x'] = not self.dimension_locks['x']
            self.raytracer.set_dimension_lock('x', self.dimension_locks['x'])
            self.control_panel.lock_x.setChecked(self.dimension_locks['x'])
            self.update_lock_status()
            event.accept()
        
        elif key == Qt.Key_Y:
            self.dimension_locks['y'] = not self.dimension_locks['y']
            self.raytracer.set_dimension_lock('y', self.dimension_locks['y'])
            self.control_panel.lock_y.setChecked(self.dimension_locks['y'])
            self.update_lock_status()
            event.accept()
        
        elif key == Qt.Key_Z:
            self.dimension_locks['z'] = not self.dimension_locks['z']
            self.raytracer.set_dimension_lock('z', self.dimension_locks['z'])
            self.control_panel.lock_z.setChecked(self.dimension_locks['z'])
            self.update_lock_status()
            event.accept()
        
        # Escape key to cancel operations
        elif key == Qt.Key_Escape:
            if self.dragging_object:
                self.raytracer.stop_object_dragging()
                self.dragging_object = False
                self.dimension_locks = {'x': False, 'y': False, 'z': False}
                self.update_lock_status()
                self.control_panel.lock_x.setChecked(False)
                self.control_panel.lock_y.setChecked(False)
                self.control_panel.lock_z.setChecked(False)
                
                # Return to ray tracing if not manually in another mode
                if not self.manual_mode_change:
                    self.on_raytrace_mode()
            elif self.raytracer.camera_controller.rotating:
                self.raytracer.stop_camera_rotation()
                if not self.manual_mode_change:
                    self.on_raytrace_mode()
            
            event.accept()
        
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release with better state management"""
        key = event.key()
        
        if key in self.camera_keys:
            key_name = self.camera_keys[key]
            if self.raytracer.camera_controller.keys_pressed.get(key_name, False):
                self.raytracer.set_camera_key_state(key_name, False)
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    # Fix: Clear key states when window loses focus
    def focusOutEvent(self, event):
        """Clear all key states when window loses focus"""
        for key_name in self.camera_keys.values():
            self.raytracer.set_camera_key_state(key_name, False)
        self._key_states_cleared = True
        super().focusOutEvent(event)
    
    def focusInEvent(self, event):
        """Reset key states cleared flag when window gains focus"""
        self._key_states_cleared = False
        super().focusInEvent(event)
    
    def update_lock_status(self):
        """Update lock status display"""
        locks = []
        for dim, locked in self.dimension_locks.items():
            if locked:
                locks.append(dim.upper())
        
        if locks:
            self.lock_label.setText(f"Locks: {', '.join(locks)}")
            self.lock_label.setStyleSheet("color: #ff9900; font-weight: bold;")
        else:
            self.lock_label.setText("Locks: None")
            self.lock_label.setStyleSheet("color: #888;")
    
    def get_lock_string(self):
        """Get string representation of active locks"""
        locks = [dim.upper() for dim, locked in self.dimension_locks.items() if locked]
        return ', '.join(locks) if locks else "None"
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.render_thread:
            self.render_thread.stop()
        self.raytracer.stop_rendering()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = GUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()