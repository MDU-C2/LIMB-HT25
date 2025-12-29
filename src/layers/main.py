"""
Main entry point for the LIMB-HT25 robotic control system.

This module initializes and connects all three layers:
- InputLayer: Reads sensor data from CAN and BLE
- ProcessingLayer: Performs ML inference and signal processing
- ControlLayer: Computes and sends control commands

Usage:
    python -m layers.main [--config CONFIG_FILE]
"""

from re import T
import sys
import signal
import time
import argparse
import os
from typing import Optional
import yaml

# Add parent directory to path for imports
sys.path.insert(0, sys.path[0] + "/../..")

from layers.input.input_layer import InputLayer
from layers.processing.processing_layer import ProcessingLayer
from layers.control.control_layer import ControlLayer
from shared.queues import DataQueue
from hardware.can.can_socketcan import SocketCANInterface
from hardware.ble.ble_bleak import BleakBLEInterface
from vision.system import VisionSystem


class LIMBSystem:
    """Main system class that manages all layers and their lifecycles."""

    def __init__(self, config: dict, vision_source=None):
        """
        Initialize the LIMB system from configuration dictionary

        Args:
            config: Configuration dictionary containing all system parameters
            vision_source: Vision source (optional, can be created from config)
        """
        self.running = False
        self.layers = []
        self.config = config

        # Extract hardware config
        hw_config = config.get("hardware", {})
        can_config = hw_config.get("can", {})
        ble_config = hw_config.get("ble", {})

        # Initialize hardware interfaces
        print("[System] Initializing hardware interfaces...")
        try:
            self.can_interface = SocketCANInterface(
                interface=can_config.get("interface", "can0"),
                bitrate=can_config.get("bitrate", 1000000)
            )
        except Exception as e:
            print(f"[System] ERROR: Failed to initialize CAN interface: {e}")
            raise

        try:
            self.ble_interface = BleakBLEInterface(
                device_name=ble_config.get("device_name", "LIMBServer"),
                scan_timeout=ble_config.get("scan_timeout", 10.0)
            )
        except Exception as e:
            print(f"[System] ERROR: Failed to initialize BLE interface: {e}")
            raise

        # Vision system source
        self.vision_source = vision_source

        # Create queues between layers
        print("[System] Creating data queues...")
        queue_config = config.get("queues", {})
        input_proc_config = queue_config.get("input_to_processing", {})
        proc_ctrl_config = queue_config.get("processing_to_control", {})
        
        self.input_to_processing_queue = DataQueue(max_size=input_proc_config.get("max_size", 5))
        self.processing_to_control_queue = DataQueue(max_size=proc_ctrl_config.get("max_size", 5))

        # Initialize layers
        print("[System] Initializing layers...")
        try:
            # Input layer
            input_config = config.get("input_layer", {})
            self.input_layer = InputLayer(
                can_interface=self.can_interface,
                ble_interface=self.ble_interface,
                output_queue=self.input_to_processing_queue,
                window_size=input_config.get("window_size", 100),
                sample_rate=input_config.get("sample_rate", 100.0),
                vision_source=vision_source
            )

            # Processing layer
            proc_config = config.get("processing_layer", {})
            model_config = proc_config.get("model", {})
            emg_config = proc_config.get("emg", {})
            features_config = proc_config.get("features", {})
            lstm_config = proc_config.get("lstm", {})
            imu_config = proc_config.get("imu_intention", {})
            fusion_config = proc_config.get("fusion", {})
            cf_config = fusion_config.get("complementary_filter", {})
            ekf_config = fusion_config.get("ekf", {})
            ekf_process = ekf_config.get("process_noise", {})
            ekf_measure = ekf_config.get("measurement_noise", {})
            
            self.processing_layer = ProcessingLayer(
                input_queue=self.input_to_processing_queue,
                output_queue=self.processing_to_control_queue,
                model_path=model_config.get("path"),
                model_config=model_config.get("config"),
                scaler_path=model_config.get("scaler_path"),
                # EMG parameters
                emg_fs=emg_config.get("fs", 1000.0),
                emg_lowcut=emg_config.get("lowcut", 20.0),
                emg_highcut=emg_config.get("highcut", 450.0),
                emg_notch_freq=emg_config.get("notch_freq", 50.0),
                # Feature extraction
                window_size_ms=features_config.get("window_size_ms", 200.0),
                overlap_ms=features_config.get("overlap_ms", 100.0),
                # LSTM parameters
                seq_length=lstm_config.get("seq_length", 10),
                num_classes=lstm_config.get("num_classes", 2),
                # IMU intention parameters
                imu_accel_threshold=imu_config.get("accel_threshold", 0.3),
                imu_gravity_removal=imu_config.get("gravity_removal", True),
                # Fusion parameters
                cf_alpha=cf_config.get("alpha", 0.98),
                cf_alpha_position=cf_config.get("alpha_position", 0.95),
                ekf_process_noise_pos=ekf_process.get("pos", 1.0),
                ekf_process_noise_vel=ekf_process.get("vel", 10.0),
                ekf_process_noise_orient=ekf_process.get("orient", 0.01),
                ekf_process_noise_angvel=ekf_process.get("angvel", 0.1),
                ekf_measurement_noise_vision_pos=ekf_measure.get("vision_pos", 25.0),
                ekf_measurement_noise_vision_orient=ekf_measure.get("vision_orient", 0.01),
                ekf_measurement_noise_imu_orient=ekf_measure.get("imu_orient", 0.005)
            )

            # Control layer
            ctrl_config = config.get("control_layer", {})
            self.control_layer = ControlLayer(
                input_queue=self.processing_to_control_queue,
                can_interface=self.can_interface,
                control_rate=ctrl_config.get("control_rate", 100.0),
                config=ctrl_config
            )

            self.layers = [self.input_layer, self.processing_layer, self.control_layer]

        except Exception as e:
            print(f"[System] ERROR: Failed to initialize layers: {e}")
            import traceback
            traceback.print_exc()
            raise
    
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self) -> bool:
        """Start layers"""
        print("[System] Starting layers...")


        try:
            # STart layers in order (Input -> Processing -> Control)
            for layer in self.layers:
                layer.start()
                time.sleep(0.1)

            time.sleep(1.0)

            #Verify layers are running
            all_running = all(layer.is_alive() for layer in self.layers)
            if not all_running:
                print("[System] ERROR: Some layer failed to start")
                return False

            self.running = True
            print("[System] All layers started successfully")
            return True
        
        except Exception as e:
            print(f"[System] ERROR: Failed to start layers: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """Stop all layers"""

        if not self.running:
            return

        print("\n[System] Stopping layers...")
        self.running = False

        for layer in reversed(self.layers):
            try:
                layer.stop()
                layer.join(timeout=2.0)
                if layer.is_alive():
                    print(f"[System] ERROR: Layer {layer.name} did not stop gracefully")
            except Exception as e:
                print(f"[System] ERROR: Failed to stop layer {layer.name}: {e}")

        # Stop hardware interfaces
        try:
            self.can_interface.stop()
        except Exception as e:
            print(f"[System] ERROR: Failed to stop CAN interface: {e}")
    
        try:
            self.ble_interface.stop()
        except Exception as e:
            print(f"[System] ERROR: Failed to stop BLE interface: {e}")

        print("[System] System stopped")

        # Stop vision system
        if self.vision_source is not None:
            try:
                self.vision_source.shutdown()
                print("[System] Vision system stopped")
            except Exception as e:
                print(f"[System] ERROR: Failed to stop vision system: {e}")

    def run(self):
        """Run the system (blocking until stopped)"""
        if not self.start():
            print("[System] Failed to start system")
            return
        
        print("\n[System] System running. Press Ctrl+C to stop.")
        print("=" * 60)
        
        try:
            # Monitor layers while running
            while self.running:
                time.sleep(1.0)
                
                # Check if any layer has died
                dead_layers = [layer.name for layer in self.layers if not layer.is_alive()]
                if dead_layers:
                    print(f"[System] ERROR: Layers died: {dead_layers}")
                    self.running = False
                    break
                
                # Could print queue statistics here
                # print(f"[System] Queues - Input->Processing: {self.input_to_processing_queue.dropped_count} dropped")
                # print(f"[System] Queues - Processing->Control: {self.processing_to_control_queue.dropped_count} dropped")
        
        except KeyboardInterrupt:
            print("\n[System] Interrupted by user")
        except Exception as e:
            print(f"[System] ERROR: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[System] Received signal {signum}, stopping...")
        self.running = False

def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file. If None, uses default config path.
    
    Returns:
        Configuration dictionary
    """
    # Default config path
    if config_path is None:
        # Try to find config in standard locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_paths = [
            os.path.join(script_dir, "../../config/system_config.yaml"),
            os.path.join(script_dir, "../config/system_config.yaml"),
            "config/system_config.yaml",
            "src/config/system_config.yaml"
        ]
        
        for path in default_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                config_path = abs_path
                break
    
    # Load default config structure
    default_config = {
        "hardware": {
            "can": {
                "interface": "can0",
                "bitrate": 1000000
            },
            "ble": {
                "device_name": "LIMBServer",
                "scan_timeout": 10.0
            }
        },
        "input_layer": {
            "window_size": 100,
            "sample_rate": 100.0
        },
        "processing_layer": {
            "model": {
                "path": None,
                "scaler_path": None,
                "config": {
                    "input_dim": 12,
                    "hidden_dim": 32
                }
            },
            "emg": {
                "fs": 1000.0,
                "lowcut": 20.0,
                "highcut": 450.0,
                "notch_freq": 50.0
            },
            "features": {
                "window_size_ms": 200.0,
                "overlap_ms": 100.0
            },
            "lstm": {
                "seq_length": 10,
                "num_classes": 2
            },
            "imu_intention": {
                "accel_threshold": 0.3,
                "gravity_removal": True,
                "velocity_threshold": 0.2,
                "direction_timeout": 4.0
            },
            "fusion": {
                "complementary_filter": {
                    "alpha": 0.98,
                    "alpha_position": 0.95
                },
                "ekf": {
                    "process_noise": {
                        "pos": 1.0,
                        "vel": 10.0,
                        "orient": 0.01,
                        "angvel": 0.1
                    },
                    "measurement_noise": {
                        "vision_pos": 25.0,
                        "vision_orient": 0.01,
                        "imu_orient": 0.005
                    }
                }
            }
        },
        "control_layer": {
            "control_rate": 100.0,
            "conf_threshold": 0.5
        },
        "vision": {
            "confidence_threshold": 0.5,
            "spatial_threshold": 5000,
            "tag_size": 0.05,
            "enable_visualization": False
        },
        "queues": {
            "input_to_processing": {
                "max_size": 5
            },
            "processing_to_control": {
                "max_size": 5
            }
        }
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            print(f"[Config] Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Deep merge: file config overrides defaults
            def deep_merge(default, override):
                """Recursively merge two dictionaries"""
                result = default.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            
            config = deep_merge(default_config, file_config)
            print("[Config] Configuration loaded successfully")
            return config
        except Exception as e:
            print(f"[Config] WARNING: Failed to load config file {config_path}: {e}")
            print("[Config] Using default configuration")
            return default_config
    elif config_path:
        print(f"[Config] WARNING: Config file not found: {config_path}")
        print("[Config] Using default configuration")
    
    return default_config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LIMB Bionic Control System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--can-interface", type=str, default="can0", help="CAN interface name")
    parser.add_argument("--can-bitrate", type=int, default=1000000, help="CAN bitrate in bps")
    parser.add_argument("--ble-device-name", type=str, default="LIMBServer", help="BLE device name")
    parser.add_argument("--control-rate", type=float, default=100.0, help="Control rate in Hz")
    parser.add_argument("--model-path", type=str, help="Path to the LSTM model")
    parser.add_argument("--scaler-path", type=str, help="Path to the scaler")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override with command line arguments (if provided)
    if args.can_interface:
        config.setdefault("hardware", {}).setdefault("can", {})["interface"] = args.can_interface
    if args.can_bitrate:
        config.setdefault("hardware", {}).setdefault("can", {})["bitrate"] = args.can_bitrate
    if args.ble_device_name:
        config.setdefault("hardware", {}).setdefault("ble", {})["device_name"] = args.ble_device_name
    if args.control_rate:
        config.setdefault("control_layer", {})["control_rate"] = args.control_rate
    if args.model_path:
        config.setdefault("processing_layer", {}).setdefault("model", {})["path"] = args.model_path
    if args.scaler_path:
        config.setdefault("processing_layer", {}).setdefault("model", {})["scaler_path"] = args.scaler_path
        
    # Initialize vision system
    vision_system = None
    vision_config = config.get("vision", {})
    try:
        print("[System] Initializing vision system...")
        vision_system = VisionSystem(
            model_path=vision_config.get("model_path"),
            apriltag_family=vision_config.get("apriltag_family", "TAG36H11"),
            confidence_threshold=vision_config.get("confidence_threshold", 0.5),
            spatial_threshold=vision_config.get("spatial_threshold", 5000),
            apriltag_quad_decimate=vision_config.get("apriltag_quad_decimate", 1.5),
            apriltag_quad_sigma=vision_config.get("apriltag_quad_sigma", 1.0),
            apriltag_refine_edges=vision_config.get("apriltag_refine_edges", True),
            apriltag_max_hamming=vision_config.get("apriltag_max_hamming", 1),
            tag_size=vision_config.get("tag_size", 0.05),
            enable_visualization=vision_config.get("enable_visualization", False)
        )
        # Start the pipeline
        if not vision_system.start_pipeline():
            print("[System] WARNING: Failed to start vision pipeline, continuing without vision")
            vision_system = None
        else:
            print("[System] Vision pipeline started successfully")
    except Exception as e:
        print(f"[System] WARNING: Failed to initialize vision system: {e}")
        print(f"[System] WARNING: Continuing without vision")
        vision_system = None

    try:
        system = LIMBSystem(config=config, vision_source=vision_system)

        system.run()

    except Exception as e:
        print(f"[System] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()