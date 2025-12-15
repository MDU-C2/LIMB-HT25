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
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, sys.path[0] + "/../..")

from layers.input.input_layer import InputLayer
from layers.processing.processing_layer import ProcessingLayer
from layers.control.control_layer import ControlLayer
from shared.queues import DataQueue
from hardware.can.can_socketcan import SocketCANInterface
from hardware.ble.ble_bleak import BleakBLEInterface


class LIMBSystem:
    """Main system class that manages all layers and their lifecycles."""

    def __init__(self,
                can_interface: str = "can0",
                can_bitrate: int = 1000000,
                ble_device_name: str = "LIMBServer",
                control_rate: float = 100.0,
                window_size: int = 100,
                sample_rate: float = 100.0,
                model_path: Optional[str] = None,
                scaler_path: Optional[str] = None,
                vision_source = None):

        """
        Initialize the LIMB system

        Args:
            can_interface: CAN interface name (e.g. "can0")
            can_bitrate: CAN bitrate in bps (default: 1000000 (1 Mbps))
            ble_device_name: BLE device name (default: "LIMBServer")
            control_rate: Control rate in Hz (default: 100.0)
            window_size: Window size for signal processing (default: 100)
            sample_rate: Sample rate in Hz (default: 100.0)
            model_path: Path to the LSTM model (default: None)
            scaler_path: Path to the scaler (default: None)
            vision_source: Vision source (default: None)
        """
        self.running = False
        self.layers = []

        # Initialize hardware interfaces
        print("[System] Initializing hardware interfaces...")
        try:
            self.can_interface = SocketCANInterface(
                interface=can_interface,
                bitrate=can_bitrate
            )
        except Exception as e:
            print(f"[System] ERROR: Failed to initialize CAN interface: {e}")
            raise

        try:
            self.ble_interface = BleakBLEInterface(
                device_name=ble_device_name,
                scan_timeout=10.0
            )
        except Exception as e:
            print(f"[System] ERROR: Failed to initialize BLE interface: {e}")
            raise

        # Create queues between layers
        print("[System] Creating data queues...")
        self.input_to_processing_queue = DataQueue(max_size=5)
        self.processing_to_control_queue = DataQueue(max_size=5)


        # Initialize layers
        print("[System] Initializing layers...")
        try:
            self.input_layer = InputLayer(
                can_interface=self.can_interface,
                ble_interface=self.ble_interface,
                output_queue=self.input_to_processing_queue,
                window_size=window_size,
                sample_rate=sample_rate,
                vision_source=vision_source
            )

            self.processing_layer = ProcessingLayer(
                input_queue=self.input_to_processing_queue,
                output_queue=self.processing_to_control_queue,
                model_path=model_path,
                scaler_path=scaler_path
            )

            self.control_layer = ControlLayer(
                input_queue=self.processing_to_control_queue,
                can_interface=self.can_interface,
                control_rate=control_rate
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
    """Load configuration from file"""
    # TODO: Implemnt config file loading (JSON/YAML)
    default_config = {
        "can_interface": "can0",
        "can_bitrate": 1000000,
        "ble_device_name": "LIMBServer",
        "control_rate": 100.0,
        "window_size": 100,
        "sample_rate": 100.0,
        "model_path": None,
        "scaler_path": None
    }

    if config_path:
        # TODO: Load from file
        pass

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

    # OVerride with command line arguments
    if args.can_interface:
        config["can_interface"] = args.can_interface
    if args.can_bitrate:
        config["can_bitrate"] = args.can_bitrate
    if args.ble_device_name:
        config["ble_device_name"] = args.ble_device_name
    if args.control_rate:
        config["control_rate"] = args.control_rate
    if args.model_path:
        config["model_path"] = args.model_path
    if args.scaler_path:
        config["scaler_path"] = args.scaler_path
        
    try:
        system = LIMBSystem(
            can_interface=config["can_interface"],
            can_bitrate=config["can_bitrate"],
            ble_device_name=config["ble_device_name"],
            control_rate=config["control_rate"],
            window_size=config["window_size"],
            sample_rate=config["sample_rate"],
            model_path=config["model_path"],
            scaler_path=config["scaler_path"],
            vision_source=None #TODO: Initialize vision system if needed
        )

        system.run()

    except Exception as e:
        print(f"[System] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()