#!/usr/bin/env python3
"""Real-time IMU visualization module."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
import time


class IMUVisualizer:
    """Real-time visualization of IMU orientation and movement."""
    
    def __init__(self, threshold=1.0):
        """
        Initialize the visualizer.
        
        Args:
            threshold: Movement threshold in m/s²
        """
        self.threshold = threshold
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('IMU Movement Detection Visualization', fontsize=14)
        
        # 3D orientation plot
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax_3d.set_title('Orientation (3D Coordinate Frame)')
        self.ax_3d.set_xlabel('X (Forward)')
        self.ax_3d.set_ylabel('Y (Right)')
        self.ax_3d.set_zlabel('Z (Up)')
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([-1.5, 1.5])
        
        # Movement vector plot (2D)
        self.ax_movement = self.fig.add_subplot(2, 2, 2)
        self.ax_movement.set_title('Movement Vector (XY plane)')
        self.ax_movement.set_xlabel('X (Forward/Backward)')
        self.ax_movement.set_ylabel('Y (Right/Left)')
        self.ax_movement.set_xlim([-2, 2])
        self.ax_movement.set_ylim([-2, 2])
        self.ax_movement.grid(True)
        self.ax_movement.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_movement.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Movement magnitude over time
        self.ax_magnitude = self.fig.add_subplot(2, 2, 3)
        self.ax_magnitude.set_title('Movement Magnitude Over Time')
        self.ax_magnitude.set_xlabel('Time (s)')
        self.ax_magnitude.set_ylabel('Magnitude (m/s²)')
        self.ax_magnitude.grid(True)
        
        # Direction indicator
        self.ax_direction = self.fig.add_subplot(2, 2, 4)
        self.ax_direction.set_title('Current Direction')
        self.ax_direction.axis('off')
        
        # Data buffers
        self.magnitude_history = []
        self.time_history = []
        
        # Current state
        self.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_movement = None
        self.start_time = time.time()
        
        # Thread lock for data access
        self.lock = threading.Lock()
        
        plt.tight_layout()
    
    def update_orientation(self, quaternion):
        """Update current orientation quaternion."""
        with self.lock:
            self.current_quaternion = quaternion.copy()
    
    def update_movement(self, movement):
        """Update current movement detection."""
        with self.lock:
            self.current_movement = movement
            if movement and movement['direction'] != 'none':
                self.magnitude_history.append(movement['magnitude'])
                self.time_history.append(time.time() - self.start_time)
                # Keep only last 200 points
                if len(self.magnitude_history) > 200:
                    self.magnitude_history.pop(0)
                    self.time_history.pop(0)
    
    def draw_coordinate_frame(self, ax, quaternion, rotate_vector_func, scale=1.0):
        """Draw a 3D coordinate frame rotated by quaternion."""
        # Unit vectors in world frame
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # Rotate to body frame
        x_rotated = rotate_vector_func(x_axis, quaternion)
        y_rotated = rotate_vector_func(y_axis, quaternion)
        z_rotated = rotate_vector_func(z_axis, quaternion)
        
        # Origin
        origin = np.array([0.0, 0.0, 0.0])
        
        # Draw axes
        ax.plot([origin[0], x_rotated[0] * scale], 
                [origin[1], x_rotated[1] * scale],
                [origin[2], x_rotated[2] * scale], 'r-', linewidth=2, label='X (Forward)')
        ax.plot([origin[0], y_rotated[0] * scale],
                [origin[1], y_rotated[1] * scale],
                [origin[2], y_rotated[2] * scale], 'g-', linewidth=2, label='Y (Right)')
        ax.plot([origin[0], z_rotated[0] * scale],
                [origin[1], z_rotated[1] * scale],
                [origin[2], z_rotated[2] * scale], 'b-', linewidth=2, label='Z (Up)')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)
    
    def animate(self, frame, rotate_vector_func):
        """Animation update function."""
        with self.lock:
            # Clear axes
            self.ax_3d.clear()
            self.ax_movement.clear()
            self.ax_magnitude.clear()
            self.ax_direction.clear()
            
            # Re-setup 3D axes
            self.ax_3d.set_title('Orientation (3D Coordinate Frame)')
            self.ax_3d.set_xlabel('X (Forward)')
            self.ax_3d.set_ylabel('Y (Right)')
            self.ax_3d.set_zlabel('Z (Up)')
            self.ax_3d.set_xlim([-1.5, 1.5])
            self.ax_3d.set_ylim([-1.5, 1.5])
            self.ax_3d.set_zlim([-1.5, 1.5])
            
            # Draw coordinate frame
            self.draw_coordinate_frame(self.ax_3d, self.current_quaternion, 
                                      rotate_vector_func, scale=1.0)
            
            # Movement vector plot
            self.ax_movement.set_title('Movement Vector (XY plane)')
            self.ax_movement.set_xlabel('X (Forward/Backward)')
            self.ax_movement.set_ylabel('Y (Right/Left)')
            self.ax_movement.set_xlim([-2, 2])
            self.ax_movement.set_ylim([-2, 2])
            self.ax_movement.grid(True)
            self.ax_movement.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            self.ax_movement.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            if self.current_movement and self.current_movement['direction'] != 'none':
                vec = self.current_movement['vector']
                # Draw arrow from origin to movement vector
                self.ax_movement.arrow(0, 0, vec[0], vec[1], 
                                      head_width=0.1, head_length=0.1, 
                                      fc='red', ec='red', linewidth=2)
                self.ax_movement.text(vec[0] + 0.1, vec[1] + 0.1, 
                                    self.current_movement['direction'], 
                                    fontsize=10, color='red', weight='bold')
            
            # Magnitude over time
            self.ax_magnitude.set_title('Movement Magnitude Over Time')
            self.ax_magnitude.set_xlabel('Time (s)')
            self.ax_magnitude.set_ylabel('Magnitude (m/s²)')
            self.ax_magnitude.grid(True)
            
            if len(self.time_history) > 0:
                self.ax_magnitude.plot(self.time_history, self.magnitude_history, 
                                      'b-', linewidth=1.5, label='Magnitude')
                self.ax_magnitude.axhline(y=self.threshold, color='r', linestyle='--', 
                                         label=f'Threshold ({self.threshold} m/s²)')
                self.ax_magnitude.legend(loc='upper right')
                # Auto-scale x-axis to show recent data
                if len(self.time_history) > 1:
                    time_range = self.time_history[-1] - self.time_history[0]
                    if time_range > 10:  # Show last 10 seconds
                        self.ax_magnitude.set_xlim([self.time_history[-1] - 10, 
                                                   self.time_history[-1] + 1])
            
            # Direction indicator
            self.ax_direction.set_title('Current Direction')
            self.ax_direction.axis('off')
            
            if self.current_movement and self.current_movement['direction'] != 'none':
                direction = self.current_movement['direction']
                confidence = self.current_movement['confidence']
                magnitude = self.current_movement['magnitude']
                
                self.ax_direction.text(0.5, 0.7, direction.upper(), 
                                      fontsize=48, ha='center', va='center',
                                      weight='bold', color='blue')
                self.ax_direction.text(0.5, 0.4, f'Confidence: {confidence:.1f}%',
                                      fontsize=16, ha='center', va='center')
                self.ax_direction.text(0.5, 0.2, f'Magnitude: {magnitude:.3f} m/s²',
                                      fontsize=14, ha='center', va='center')
            else:
                self.ax_direction.text(0.5, 0.5, 'NO MOVEMENT', 
                                      fontsize=32, ha='center', va='center',
                                      color='gray')
    
    def start(self, rotate_vector_func):
        """Start the animation."""
        def animate_wrapper(frame):
            self.animate(frame, rotate_vector_func)
        
        self.ani = FuncAnimation(self.fig, animate_wrapper, interval=50, blit=False, cache_frame_data=False)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Non-blocking show
        
    def update_plot(self):
        """Manually trigger plot update (for non-blocking mode)."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

