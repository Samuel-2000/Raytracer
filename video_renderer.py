import cv2
import numpy as np
import os
from queue import Queue
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QCheckBox, QLineEdit, QFileDialog, QProgressBar,
                             QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import threading
import time

class CameraRecorder:
    """Records camera positions over time"""
    
    def __init__(self, raytracer):
        self.raytracer = raytracer
        self.recording = False
        self.frames = []  # List of (position, target, up, fov, timestamp)
        self.current_segment = []
        self.segments = []  # List of segments (list of frames)
        
    def start_recording(self):
        """Start recording camera movements"""
        self.recording = True
        self.current_segment = []
        print("Started recording camera path")
        
    def stop_recording(self):
        """Stop recording"""
        if self.recording:
            self.recording = False
            if self.current_segment:
                self.segments.append(self.current_segment.copy())
                self.current_segment = []
            print(f"Stopped recording. Total segments: {len(self.segments)}")
            return True
        return False
    
    def record_frame(self):
        """Record current camera state"""
        if not self.recording:
            return
            
        camera = self.raytracer.camera
        frame = {
            'position': camera.position,
            'target': camera.target,
            'up': camera.up,
            'fov': camera.fov,
            'timestamp': time.time()
        }
        self.current_segment.append(frame)
        
        # Limit recording to prevent memory issues
        if len(self.current_segment) > 1000:
            self.segments.append(self.current_segment[-500:].copy())
            self.current_segment = self.current_segment[-500:]
    
    def clear_recording(self):
        """Clear all recorded frames"""
        self.frames = []
        self.segments = []
        self.current_segment = []
        self.recording = False
    
    def get_total_frames(self):
        """Get total number of recorded frames"""
        total = len(self.current_segment)
        for segment in self.segments:
            total += len(segment)
        return total
    
    def get_all_frames(self):
        """Get all frames from all segments combined"""
        all_frames = []
        for segment in self.segments:
            all_frames.extend(segment)
        all_frames.extend(self.current_segment)
        return all_frames

class VideoRenderer(QThread):
    """Thread for rendering video frames"""
    frame_rendered = pyqtSignal(int, int)  # current_frame, total_frames
    video_finished = pyqtSignal(str)  # output_path
    rendering_error = pyqtSignal(str)  # error_message
    
    def __init__(self, raytracer, camera_frames, render_settings):
        super().__init__()
        self.raytracer = raytracer
        self.camera_frames = camera_frames
        self.settings = render_settings
        self.stopped = False
        
    def run(self):
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.settings['output_path'])
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.settings.get('fps', 30)
            out = cv2.VideoWriter(
                self.settings['output_path'],
                fourcc,
                fps,
                (self.settings['width'], self.settings['height'])
            )
            
            total_frames = len(self.camera_frames)
            
            for i, camera_frame in enumerate(self.camera_frames):
                if self.stopped:
                    break
                
                # Set camera to recorded position
                self.raytracer.camera.position = camera_frame['position']
                self.raytracer.camera.target = camera_frame['target']
                self.raytracer.camera.up = camera_frame['up']
                self.raytracer.camera.fov = camera_frame['fov']
                
                # Update C++ camera
                self.raytracer.ray_tracer.set_camera(self.raytracer.camera)
                
                # Render frame
                try:
                    # Save current settings
                    orig_max_samples = self.raytracer.settings['max_samples']
                    orig_samples_batch = self.raytracer.settings['samples_per_batch']
                    
                    # Apply video settings
                    self.raytracer.settings['max_samples'] = self.settings['samples_per_frame']
                    self.raytracer.settings['samples_per_batch'] = min(
                        self.settings['samples_per_frame'],
                        self.settings.get('samples_per_batch', 8)
                    )
                    
                    # Render the frame
                    result = self.raytracer.ray_tracer.render(
                        self.settings['width'],
                        self.settings['height'],
                        self.settings['samples_per_frame'],
                        self.settings.get('max_depth', 4)
                    )
                    
                    # Restore original settings
                    self.raytracer.settings['max_samples'] = orig_max_samples
                    self.raytracer.settings['samples_per_batch'] = orig_samples_batch
                    
                    if result is None or len(result) == 0:
                        raise Exception("Render returned no data")
                    
                    # Convert to image
                    image_array = np.array(result, dtype=np.float32).reshape(
                        (self.settings['height'], self.settings['width'], 3)
                    )
                    
                    # Apply tone mapping
                    exposure = self.settings.get('exposure', 1.5)
                    image_array = image_array * exposure
                    image_array = image_array / (1.0 + image_array)
                    image_array = np.clip(image_array, 0.0, 1.0)
                    
                    # Apply contrast enhancement if enabled
                    if self.settings.get('enhance_contrast', False):
                        min_val = np.percentile(image_array, 2)
                        max_val = np.percentile(image_array, 98)
                        if max_val > min_val:
                            image_array = (image_array - min_val) / (max_val - min_val)
                            image_array = np.clip(image_array, 0, 1)
                    
                    # Convert to 8-bit and BGR for OpenCV
                    image_8bit = (image_array * 255).astype(np.uint8)
                    image_bgr = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
                    
                    # Write frame to video
                    out.write(image_bgr)
                    
                    # Emit progress
                    self.frame_rendered.emit(i + 1, total_frames)
                    
                except Exception as e:
                    print(f"Error rendering frame {i}: {e}")
                    continue
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
            
            out.release()
            
            if not self.stopped:
                self.video_finished.emit(self.settings['output_path'])
                
        except Exception as e:
            self.rendering_error.emit(str(e))
    
    def stop(self):
        """Stop the rendering process"""
        self.stopped = True
        self.wait()

class VideoRenderDialog(QDialog):
    """Dialog for video rendering settings"""
    
    def __init__(self, parent, raytracer, camera_frames):
        super().__init__(parent)
        self.raytracer = raytracer
        self.camera_frames = camera_frames
        self.video_renderer = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the video rendering dialog UI"""
        self.setWindowTitle("Video Rendering Settings")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Video info
        info_group = QGroupBox("Recording Info")
        info_layout = QVBoxLayout()
        self.info_label = QLabel(f"Total frames to render: {len(self.camera_frames)}")
        info_layout.addWidget(self.info_label)
        
        duration = len(self.camera_frames) / 30  # Assuming 30 FPS
        self.duration_label = QLabel(f"Estimated duration: {duration:.1f} seconds")
        info_layout.addWidget(self.duration_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Quality settings
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QGridLayout()
        
        # Resolution
        quality_layout.addWidget(QLabel("Width:"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(self.raytracer.width)
        quality_layout.addWidget(self.width_spin, 0, 1)
        
        quality_layout.addWidget(QLabel("Height:"), 0, 2)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(self.raytracer.height)
        quality_layout.addWidget(self.height_spin, 0, 3)
        
        # Samples per frame
        quality_layout.addWidget(QLabel("Samples per frame:"), 1, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 1024)
        self.samples_spin.setValue(64)
        quality_layout.addWidget(self.samples_spin, 1, 1)
        
        # Max depth
        quality_layout.addWidget(QLabel("Max depth:"), 1, 2)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 32)
        self.depth_spin.setValue(4)
        quality_layout.addWidget(self.depth_spin, 1, 3)
        
        # FPS
        quality_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        quality_layout.addWidget(self.fps_spin, 2, 1)
        
        # Exposure
        quality_layout.addWidget(QLabel("Exposure:"), 2, 2)
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 5.0)
        self.exposure_spin.setValue(1.5)
        self.exposure_spin.setSingleStep(0.1)
        quality_layout.addWidget(self.exposure_spin, 2, 3)
        
        # Enhance contrast
        self.enhance_checkbox = QCheckBox("Enhance contrast")
        self.enhance_checkbox.setChecked(True)
        quality_layout.addWidget(self.enhance_checkbox, 3, 0, 1, 2)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Output file:"))
        self.file_edit = QLineEdit("output/video.mp4")
        file_layout.addWidget(self.file_edit)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_file)
        file_layout.addWidget(self.browse_btn)
        output_layout.addLayout(file_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.render_btn = QPushButton("Start Rendering")
        self.render_btn.clicked.connect(self.start_rendering)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.render_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def browse_output_file(self):
        """Browse for output file location"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "video.mp4", "MP4 files (*.mp4)"
        )
        if filename:
            self.file_edit.setText(filename)
    
    def start_rendering(self):
        """Start the video rendering process"""
        # Disable UI
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get settings
        settings = {
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'samples_per_frame': self.samples_spin.value(),
            'max_depth': self.depth_spin.value(),
            'fps': self.fps_spin.value(),
            'exposure': self.exposure_spin.value(),
            'enhance_contrast': self.enhance_checkbox.isChecked(),
            'output_path': self.file_edit.text()
        }
        
        # Create and start video renderer
        self.video_renderer = VideoRenderer(self.raytracer, self.camera_frames, settings)
        self.video_renderer.frame_rendered.connect(self.update_progress)
        self.video_renderer.video_finished.connect(self.rendering_finished)
        self.video_renderer.rendering_error.connect(self.rendering_error)
        self.video_renderer.start()
    
    def update_progress(self, current_frame, total_frames):
        """Update progress bar"""
        progress = int((current_frame / total_frames) * 100)
        self.progress_bar.setValue(progress)
        
    def rendering_finished(self, output_path):
        """Handle successful rendering completion"""
        QMessageBox.information(self, "Success", f"Video saved to:\n{output_path}")
        self.accept()
    
    def rendering_error(self, error_message):
        """Handle rendering error"""
        QMessageBox.critical(self, "Rendering Error", f"Error: {error_message}")
        self.render_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.video_renderer and self.video_renderer.isRunning():
            self.video_renderer.stop()
        event.accept()

class SegmentDialog(QDialog):
    """Dialog for handling recording segments"""
    
    CUT_AND_CONTINUE = 1
    RENDER_NOW = 2
    
    def __init__(self, parent, total_frames):
        super().__init__(parent)
        self.total_frames = total_frames
        self.result_code = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup segment dialog UI"""
        self.setWindowTitle("Recording Complete")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Info
        info_label = QLabel(
            f"Recording stopped.\n"
            f"Total frames recorded: {self.total_frames}\n\n"
            f"Do you want to:"
        )
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Buttons
        self.cut_btn = QPushButton("Cut and Continue Recording")
        self.cut_btn.clicked.connect(lambda: self.set_result(self.CUT_AND_CONTINUE))
        layout.addWidget(self.cut_btn)
        
        self.render_btn = QPushButton("Render Video Now")
        self.render_btn.clicked.connect(lambda: self.set_result(self.RENDER_NOW))
        layout.addWidget(self.render_btn)
        
        self.cancel_btn = QPushButton("Cancel (Discard Recording)")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        
        self.setLayout(layout)
    
    def set_result(self, result_code):
        """Set result and accept dialog"""
        self.result_code = result_code
        self.accept()