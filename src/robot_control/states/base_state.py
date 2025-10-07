"""
Base State Class

Abstract base class for all action states in the robotic manipulation system.
Defines the interface that all action states must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any
import time

class BaseState(ABC):
    """Abstract base class for all action states."""
    
    def __init__(self, sensor_manager, robot_arm):
        """
        Initialize base state.
        
        Args:
            sensor_manager: SensorManager instance for accessing sensors
            robot_arm: RobotArm instance for controlling robot hardware
        """
        self.sensor_manager = sensor_manager
        self.robot_arm = robot_arm
        self.start_time = None
        self.max_execution_time = 30.0  # Maximum execution time in seconds
    
    @abstractmethod
    def execute(self) -> Tuple[bool, Optional[str]]:
        """
        Execute the action state.
        
        Returns:
            Tuple of (success, next_state)
            - success: True if action completed successfully, False otherwise
            - next_state: Next state to transition to, or None for automatic transition
        """
        pass
    
    @abstractmethod
    def get_required_sensors(self) -> List[str]:
        """
        Get list of sensors required for this state.
        
        Returns:
            List of sensor names required for this action
        """
        pass
    
    def pre_execute(self):
        """Called before execute() - setup and validation."""
        self.start_time = time.time()
        print(f"Starting {self.__class__.__name__}")
        
        # Validate required sensors are available
        required_sensors = self.get_required_sensors()
        for sensor_name in required_sensors:
            if not self.sensor_manager.is_sensor_active(sensor_name):
                print(f"Warning: Required sensor '{sensor_name}' not active")
    
    def post_execute(self, success: bool):
        """Called after execute() - cleanup and logging."""
        elapsed_time = time.time() - (self.start_time or time.time())
        status = "SUCCESS" if success else "FAILED"
        print(f"{self.__class__.__name__} {status} in {elapsed_time:.2f}s")
        
        if elapsed_time > self.max_execution_time:
            print(f"Warning: Action took longer than expected ({elapsed_time:.2f}s)")
    
    def is_timeout(self) -> bool:
        """Check if execution has exceeded maximum time."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.max_execution_time
    
    def get_execution_time(self) -> float:
        """Get current execution time."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_sensor_data(self, sensor_name: str) -> Optional[Any]:
        """Get latest data from a specific sensor."""
        return self.sensor_manager.get_sensor_data(sensor_name)
    
    def get_all_sensor_data(self) -> dict:
        """Get data from all active sensors."""
        return self.sensor_manager.get_all_active_data()
    
    def log_sensor_data(self):
        """Log current sensor data for debugging."""
        sensor_data = self.get_all_sensor_data()
        print(f"Sensor data for {self.__class__.__name__}:")
        for sensor_name, data in sensor_data.items():
            if data:
                print(f"  {sensor_name}: {type(data).__name__} data available")
            else:
                print(f"  {sensor_name}: No data")
    
    def set_max_execution_time(self, timeout: float):
        """Set maximum execution time for this state."""
        self.max_execution_time = timeout
