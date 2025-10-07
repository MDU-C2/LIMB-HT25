"""
Central Sensor Manager

Manages all sensors and dynamically activates only the sensors required for each action state.
This ensures efficient resource usage and follows the principle that "all sensors are not 
necessarily active at every action state."
"""

from typing import Dict, List, Set, Optional, Any
import time
import threading

class SensorManager:
    """Central coordinator for all sensors in the robotic manipulation system."""
    
    def __init__(self):
        self.sensors = {}
        self.active_sensors: Set[str] = set()
        self.sensor_threads = {}
        self.running = False
        
        # Initialize all available sensors
        self._initialize_sensors()
        
        # Sensor requirements per action state (based on flowchart)
        self.state_sensor_map = {
            'MOVE_to_cup': ['vision', 'imu', 'piezo'],
            'GRAB_cup': ['vision', 'pressure', 'slip'],
            'LIFT_cup': ['vision', 'pressure', 'slip', 'imu'],
            'MOVE_cup_ab': ['vision', 'pressure', 'slip', 'imu'],
            'PLACE_DOWN_cup': ['vision', 'pressure', 'slip', 'imu'],
            'RELEASE_cup': ['vision', 'pressure', 'slip'],
            'MOVE_back_hand': ['vision', 'imu', 'piezo']
        }
        
        print("✓ SensorManager initialized")
    
    def _initialize_sensors(self):
        """Initialize all available sensors."""
        # Import sensors here to avoid circular imports
        from .imu_reader import IMUReader
        from .vision_system import VisionSystemWrapper
        from .pressure_sensor import PressureSensor
        from .slip_sensor import SlipSensor
        from .piezo_sensor import PiezoSensor
        
        self.sensors = {
            'imu': IMUReader(),
            'vision': VisionSystemWrapper(),
            'pressure': PressureSensor(),
            'slip': SlipSensor(),
            'piezo': PiezoSensor()
        }
    
    def activate_sensors_for_state(self, state_name: str) -> bool:
        """
        Activate only the sensors required for the given state.
        
        Args:
            state_name: Name of the action state (e.g., 'MOVE_to_cup')
            
        Returns:
            True if all required sensors activated successfully
        """
        required_sensors = self.get_required_sensors(state_name)
        if not required_sensors:
            print(f"Warning: No sensor requirements defined for state '{state_name}'")
            return True
        
        print(f"Activating sensors for '{state_name}': {required_sensors}")
        
        # Deactivate unused sensors
        sensors_to_deactivate = self.active_sensors - set(required_sensors)
        for sensor_name in sensors_to_deactivate:
            self._deactivate_sensor(sensor_name)
        
        # Activate required sensors
        success = True
        for sensor_name in required_sensors:
            if sensor_name not in self.active_sensors:
                if self._activate_sensor(sensor_name):
                    self.active_sensors.add(sensor_name)
                    print(f"{sensor_name} activated")
                else:
                    print(f"{sensor_name} failed to activate")
                    success = False
        
        return success
    
    def _activate_sensor(self, sensor_name: str) -> bool:
        """Activate a specific sensor."""
        if sensor_name not in self.sensors:
            print(f"Error: Unknown sensor '{sensor_name}'")
            return False
        
        try:
            return self.sensors[sensor_name].activate()
        except Exception as e:
            print(f"Error activating {sensor_name}: {e}")
            return False
    
    def _deactivate_sensor(self, sensor_name: str):
        """Deactivate a specific sensor."""
        if sensor_name in self.sensors and sensor_name in self.active_sensors:
            try:
                self.sensors[sensor_name].deactivate()
                self.active_sensors.discard(sensor_name)
                print(f"{sensor_name} deactivated")
            except Exception as e:
                print(f"Error deactivating {sensor_name}: {e}")
    
    def get_sensor(self, sensor_name: str):
        """Get a sensor instance by name."""
        if sensor_name not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        return self.sensors[sensor_name]
    
    def get_required_sensors(self, state_name: str) -> List[str]:
        """Get list of sensors required for a given state."""
        return self.state_sensor_map.get(state_name, [])
    
    def get_sensor_data(self, sensor_name: str) -> Optional[Any]:
        """Get latest data from a specific sensor."""
        if sensor_name not in self.active_sensors:
            print(f"Warning: Sensor '{sensor_name}' not active")
            return None
        
        try:
            return self.sensors[sensor_name].get_latest_data()
        except Exception as e:
            print(f"Error getting data from {sensor_name}: {e}")
            return None
    
    def get_all_active_data(self) -> Dict[str, Any]:
        """Get data from all currently active sensors."""
        data = {}
        for sensor_name in self.active_sensors:
            sensor_data = self.get_sensor_data(sensor_name)
            if sensor_data is not None:
                data[sensor_name] = sensor_data
        return data
    
    def is_sensor_active(self, sensor_name: str) -> bool:
        """Check if a sensor is currently active."""
        return sensor_name in self.active_sensors
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sensor manager status."""
        return {
            'active_sensors': list(self.active_sensors),
            'available_sensors': list(self.sensors.keys()),
            'state_sensor_map': self.state_sensor_map,
            'running': self.running
        }
    
    def cleanup(self):
        """Cleanup all sensors and resources."""
        print("Cleaning up SensorManager...")
        self.running = False
        
        # Deactivate all sensors
        for sensor_name in list(self.active_sensors):
            self._deactivate_sensor(sensor_name)
        
        # Cleanup sensor resources
        for sensor_name, sensor in self.sensors.items():
            try:
                if hasattr(sensor, 'cleanup'):
                    sensor.cleanup()
            except Exception as e:
                print(f"Error cleaning up {sensor_name}: {e}")
        
        print("SensorManager cleanup completed")
    
    def add_custom_state_sensors(self, state_name: str, sensors: List[str]):
        """Add custom sensor requirements for a state."""
        self.state_sensor_map[state_name] = sensors
        print(f"Added custom sensor mapping for '{state_name}': {sensors}")
    
    def list_available_sensors(self):
        """List all available sensors."""
        print("Available sensors:")
        for sensor_name in self.sensors.keys():
            status = "active" if sensor_name in self.active_sensors else "inactive"
            print(f"  - {sensor_name}: {status}")
    
    def test_sensor(self, sensor_name: str) -> bool:
        """Test if a sensor is working properly."""
        if sensor_name not in self.sensors:
            print(f"Error: Unknown sensor '{sensor_name}'")
            return False
        
        print(f"Testing sensor: {sensor_name}")
        
        # Try to activate
        if not self._activate_sensor(sensor_name):
            return False
        
        # Try to get data
        data = self.get_sensor_data(sensor_name)
        
        # Deactivate
        self._deactivate_sensor(sensor_name)
        
        success = data is not None
        print(f"  {sensor_name} test {'passed' if success else 'failed'}")
        return success
