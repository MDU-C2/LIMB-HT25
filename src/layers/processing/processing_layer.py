from multiprocessing import Process, Event
import time
from shared.queues import DataQueue

import numpy as np
import torch
from typing import Optional, Deque
from collections import deque
import sys
import os

from shared.queues import DataQueue
from shared.models.packet import DataPacket
from .emg_utils import preprocess_emg_signal, extract_time_domain_features
from emg.models import get_simple_lstm
from data_fusion.complementary_filter import ComplementaryFilter
from data_fusion.ekf_filter import ExtendedKalmanFilter


class ProcessingLayer(Process):
    """Processing layer: ML inference, signal processing."""

    def __init__(self, 
                input_queue: DataQueue, 
                output_queue: DataQueue,
                model_path: Optional[str] = None,
                model_config: Optional[dict] = None,
                scaler_path: Optional[str] = None,
                # EMG 
                emg_fs: float = 1000.0, # Sampling frequency in Hz
                emg_lowcut: float = 20.0, # Lower cutoff frequency in Hz
                emg_highcut: float = 450.0, # Upper cutoff frequency in Hz
                emg_notch_freq: float = 50.0, # Notch frequency in Hz
                # Feature extraction
                window_size_ms: float = 200.0, # Window size in milliseconds
                overlap_ms: float = 100.0, # Overlap in milliseconds
                # LSTM sequence parameters
                seq_length: int = 10, # Number of windows in the sequence
                num_classes: int = 2, # Number of classes for the LSTM model
                #IMU movement intention parameters
                imu_accel_threshold: float = 0.3, # Minimum acc to detect movement
                imu_gravity_removal: bool = True, # Remove gravity

                # IMU+vision fusion parameters
                cf_alpha: float = 0.98, # Complementary filter alpha
                cf_alpha_position: float = 0.95, # Trust factor for position fusion
                ekf_process_noise_pos: float = 1.0, # Position process noise (mm^2)
                ekf_process_noise_vel: float = 10.0, # Velocity process noise (mm^2/s^2)
                ekf_process_noise_orient: float = 0.01, # Orientation process noise (rad^2)
                ekf_process_noise_angvel: float = 0.1, # Angular velocity process noise (rad^2/s^2)
                ekf_measurement_noise_vision_pos: float = 25.0, # Vision position noise (mm^2)
                ekf_measurement_noise_vision_orient: float = 0.01, # Vision orientation noise (rad^2)
                ekf_measurement_noise_imu_orient: float = 0.005, # IMU orientation noise (rad^2)
    ):
        super().__init__(name="ProcessingLayer")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = Event()

        # EMG preprocessing parameters
        self.emg_fs = emg_fs
        self.emg_lowcut = emg_lowcut
        self.emg_highcut = emg_highcut
        self.emg_notch_freq = emg_notch_freq
        
        # Feature extraction parameters
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms

        # LSTM sequence parameters
        self.seq_length = seq_length
        self.num_classes = num_classes

        # IMU movement intention parameters
        self.imu_accel_threshold = imu_accel_threshold
        self.imu_gravity_removal = imu_gravity_removal

        # Feature buffer for creating sequences (stores recent vectors)
        self.feature_buffer = deque(maxlen=seq_length)

        # ML model and scaler (StandardScaler for standardization)
        self.model = None
        self.scaler = None  # StandardScaler instance for feature standardization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model if path is provided:
        if model_path:
            self._load_model(model_path, model_config, scaler_path)

        # IMU and vision fision
        # Complementary filter and Extended Kalmar Filter
        self.complementary_filter = ComplementaryFilter(
            alpha=cf_alpha,
            alpha_position=cf_alpha_position
        )
        self.ekf_filter = ExtendedKalmanFilter(
            process_noise_pos=ekf_process_noise_pos,
            process_noise_vel=ekf_process_noise_vel,
            process_noise_orient=ekf_process_noise_orient,
            process_noise_angvel=ekf_process_noise_angvel,
            measurement_noise_vision_pos=ekf_measurement_noise_vision_pos,
            measurement_noise_vision_orient=ekf_measurement_noise_vision_orient,
            measurement_noise_imu_orient=ekf_measurement_noise_imu_orient
        )

    def _load_model(self, model_path: str, model_config: Optional[dict] = None, scaler_path: Optional[str] = None):
        """Load the trained LSTM model and scaler."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if model_config:
                input_dim = model_config.get("input_dim")
                hidden_dim = model_config.get("hidden_dim")
            elif "config" in checkpoint:
                # Try to load from checkpoint metadata
                config = checkpoint.get("config", {})
                input_dim = config.get("input_dim")
                hidden_dim = config.get("hidden_dim")
            else:
                input_dim = 12 # 6 feature * 2 channels # TODO: Check that this is correct
                hidden_dim = 32 

            self.model = get_simple_lstm(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=self.num_classes).to(self.device) # TODO: Check that this works.

            # Load model weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            print(f"Model loaded from {model_path}")

            # Load StandardScaler if path is provided
            if scaler_path:
                import pickle
                from sklearn.preprocessing import StandardScaler
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                # Verify it's a StandardScaler
                if not isinstance(self.scaler, StandardScaler):
                    print(f"Warning: Scaler is not a StandardScaler, got {type(self.scaler)}")
                print(f"StandardScaler loaded from {scaler_path}")
        
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Processing layer will run without ML inference.")

    def run(self):
        
        self.running.set()

        while self.running.is_set():
            try:
                packet = self.input_queue.get(timeout=0.001) # Why timeout?

                # Check if packet is too old
                if packet.is_stale():
                    continue
     
                processed = self.process_packet(packet)

                if not self.output_queue.full():
                    self.output_queue.put(processed, block=False) # Why non-blocking?


            except:
                time.sleep(0.001) # Queue is empty, continue

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop

    def process_packet(self, packet):
        """Run ML inference and signal processing
        
        Packet includes:
        - sequence_id: int
        - timestamp: float
        - packet_age_ms: float
        - human_data: HumanDataWindow
        - sensors: SensorSnapshot
        - motors: MotorState
        - metadata: Dict[str, Any]
        """

        if "processed" not in packet.metadata:
            packet.metadata["processed"] = False

        # Process EMG data if available
        if packet.human_data and packet.human_data.emg is not None:
            features = self._process_emg(packet.human_data)

            # Run ML inference if model is loaded
            if self.model is not None and features is not None:
                prediction = self._run_ml_inference(features)
                packet.metadata["ml_prediction"] = prediction # This is a dict with the keys: class, probabilities, confidence, timestamp
            else:
                packet.metadata["ml_prediction"] = None

            packet.metadata["features"] = features.tolist() if features is not None else None
            packet.metadata["processed"] = True # TODO: Right now this is only set to True if the EMG data is processed. We need to add a flag for the IMU data?

        # Process IMU data for movement intenetion detection
        if packet.human_data and packet.human_data.imu is not None:
            movement_intention = self._process_imu_intention(packet.human_data)
            packet.metadata["movement_intention"] = movement_intention

        return packet

    def _process_emg(self, human_data) -> Optional[np.ndarray]:
        """Process EMG data: preprocessing and feature extraction
        
        Steps:
        1. Preprocess EMG (DC offset removal, band-pass, notch)
        2. Extract time domain features (Mav, RMS, WL, ZC, SSC, VAR)
        3. Return feature vector for current window
        """
        try:

            emg_data = human_data.emg # Shape: (window_size, num_channels) # TODO: is this the data format that we want or do we want it (num_channels, window_size)?

            if emg_data is None or emg_data.size == 0:
                return None

            emg_transposed = emg_data.T # Transpose to (num_channels, window_size) 

            # Step 1: Preprocess EMG (DC offset removal, band-pass, notch)
            emg_preprocessed = preprocess_emg_signal(
                signal=emg_transposed, 
                fs=self.emg_fs, 
                lowcut=self.emg_lowcut, 
                highcut=self.emg_highcut, 
                notch_freq=self.emg_notch_freq,
                order=4
            )

            # Step 2: Extract time domain features (Mav, RMS, WL, ZC, SSC, VAR)
            num_channels, window_size = emg_preprocessed.shape

            window_matrix = emg_preprocessed.flatten().reshape(1, -1) # Create single-row window matrix by flattening channels
            
            # Extract features (returns shape: (1, num_channels*6))
            feature_matrix = extract_time_domain_features(
                window_matrix=window_matrix, 
                num_channels=num_channels
            )

            feature_vector = feature_matrix[0] # Shape (num_channels * 6,)
            
            return feature_vector

        except Exception as e:
            print(f"Error processing EMG data: {e}")
            return None

    def _run_ml_inference(self, features: np.ndarray) -> Optional[dict]:
        """Run ML inference using LSTM model
        
        Steps:
        1. Add features to buffer
        2. Create sequence if buffer is full
        3. Standardize features using StandardScaler (mean=0, std=1)
        4. Run model inference
        5. Return prediction
        """

        try:
            
            # Step 1: Add features to buffer
            self.feature_buffer.append(features)

            # Check if feature buffer is full
            if len(self.feature_buffer) < self.seq_length:
                return None

            # Step 2: Create sequence from buffer
            sequence = np.array(list(self.feature_buffer))
            
            # Step 3: Standardize features using StandardScaler (mean=0, std=1)
            if self.scaler:
                sequence = self.scaler.transform(sequence)  # StandardScaler.transform() applies: (x - mean) / std

            # Convert to tesnor and add batch dimension
            # Shape: (1, seq_length, num_features) for batch_size=1
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Step 4: Run model inference
            with torch.no_grad():
                logits = self.model(sequence_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            return {
                "class": int(predicted_class),
                "probabilities": probabilities[0].cpu().numpy(),
                "confidence": float(confidence),
                "timestamp": time.time()
            }

            

        except Exception as e:
            print(f"Error running ML inference: {e}")
            return None

    def _process_imu_intention(self, human_data) -> Optional[dict]:
        """
        Process IMU data to detect high_level movement intentions


        Steps:
        1. Extract accelerometer and gyroscope data from IMU window
        2. Remove gravity (if enabled)
        3. Calculate mean acceleration in each axis
        4. Determine dominant direction based on thresholds
        5. Return movement intention with condifence
        """
        try:
            # Step 1: Extract accelerometer and gyroscope data from IMU window
            imu_data = human_data.imu # Shape: (window_size, 6) - [ax, ay, az, gx, gy, gz]
            if imu_data is None or imu_data.size == 0:
                return None
            # Extract accel and gyro data
            accel_data = imu_data[:, :3] # Shape: (window_size, 3) - [ax, ay, az]
            gyro_data = imu_data[:, 3:] # Shape: (window_size, 3) - [gx, gy, gz]

            # Step 2: Remove gravity (if enabled)
            # Simple approach: subtract mean of accel data
            if self.imu_gravity_removal:
                accel_mean_removed = accel_data - np.mean(accel_data, axis=0, keepdims=True)
                linear_accel = accel_mean_removed
            else:
                linear_accel = accel_data

            mean_accel = np.mean(linear_accel, axis=0) # Shape (3,) - [ax_mean, ay_mean, az_mean]

            # Step 3: Calculate magnitude of mean acceleration
            accel_magnitude = np.linalg.norm(mean_accel)

            # Step 4: Determine dominant direction
            direction = "none"
            confidence = 0.0

            if accel_magnitude < self.imu_accel_threshold:
                # Movement to small to detect
                direction = "none"
                confidence = 0.0

            else:
                abs_accel = np.abd(mean_accel)
                max_axis = np.argmax(abs_accel)
                max_value = abs_accel[max_axis]

                # Check if this axis dominates (at least 60% of total magnitude)
                if max_value / accel_magnitude >= 0.6:
                    # Determine direction based on sign and axis
                    if max_axis == 0:
                        direction = "forward" if mean_accel[0] > 0 else "backward"
                    elif max_axis == 1:
                        direction = "right" if mean_accel[1] > 0 else "left"
                    elif max_axis == 2:
                        direction = "up" if mean_accel[2] > 0 else "down"

                    # Confidence is baed on how dominant the axis is and magnitude
                    axis_dominance = max_value / accel_magnitude
                    magnitude_factor = min(accel_magnitude / (self.imu_accel_threshold * 2), 1.0)
                    confidence = axis_dominance * magnitude_factor
                else:
                    # Multiple axes active - less confident, but still detect primary direction
                    if max_axis == 0:
                        direction = "forward" if mean_accel[0] > 0 else "backward"
                    elif max_axis == 1:
                        direction = "right" if mean_accel[1] > 0 else "left"
                    elif max_axis == 2:
                        direction = "up" if mean_accel[2] > 0 else "down"

                    # Lower confidence
                    confidence = 0.5 * (max_value / accel_magnitude)

            return {
                "direction": direction,
                "acceleration": mean_accel.tolist(), # Shape (3,) - [ax_mean, ay_mean, az_mean]
                "magnitude": float(accel_magnitude),
                "confidence": float(confidence),
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Error processing IMU intention: {e}")
            return None


    def _process_imu_vision_fusion(self, packet) -> Optional[dict]:
        """
        Imu and vision fusion in two stages.
        
        Stage 1: Complementary filter for orientation (IMU)
        Stage 2: Extended Kalman Filter for position and velocity (IMU + Vision)

        Steps:
        1. Extract robot IMU data from packet
        2. Stage 1: Process IMU through complementary filter to get orientation
        3. Extract vision data (AprilTags) from packet
        4. Stage 2: Feed quaternion from CF and vision position to EKF
        5. Return fused pose estimate with Euler angles
        """

        try:
            # Step 1: Extract robot IMU data from packet
            robot_imu = None
            if packet.sensors and packet.sensors.robot_imu:
                robot_imu = packet.sensors.robot_imu

            # If robot IMU not found, return EKF state if initialized, otherwise None
            if robot_imu is None:
                if self.ekf_filter.initialized:
                    ekf_state = self.ekf_filter.get_state()
                    # Convert dict format to Euler angles
                    orientation = ekf_state.orientation
                    roll, pitch, yaw = ComplementaryFilter.quaternion_to_euler(orientation)
                    return {
                        "position": ekf_state.position.tolist(),
                        "velocity": ekf_state.velocity.tolist(),
                        "orientation": orientation.tolist(),
                        "orientation_euler": [roll, pitch, yaw],
                        "angular_velocity": ekf_state.angular_velocity.tolist(),
                        "initialized": True,
                        "stage": "ekf_only",
                        "timestamp": ekf_state.timestamp
                    }
                return None

            # Extract gyroscope and accelerometer from robot IMU
            # Format: {"data": [ax, ay, az, wx, wy, wz]}
            if isinstance(robot_imu, dict) and "data" in robot_imu:
                imu_data = robot_imu["data"]
                if isinstance(imu_data, (list, np.ndarray)) and len(imu_data) >= 6:
                    accel = np.array(imu_data[:3], dtype=np.float64)
                    gyro = np.array(imu_data[3:], dtype=np.float64)
                else:
                    print("Warning: Robot IMU data format invalid.")
                    return None
            else:
                print(f"Warning: Robot IMU data format not recognized: {type(robot_imu)}")
                return None
            
            # Step 2: Process IMU through complementary filter to get orientation
            current_time = packet.timestamp
            dt = None
            if hasattr(self, "_last_cf_time"):
                dt = current_time - self._last_cf_time
            else:
                dt = 0.01
            self._last_cf_time = current_time

            dt = min(dt, 0.1) # Limit dt to avoid instability

            cf_state = self.complementary_filter.predict_with_imu(
                gyro=gyro,
                accel=accel,
                dt=dt
            )

            # Extract quaternion from CF output
            cf_quaternion = cf_state["orientation"] # [w, x, y, z]

            # Convert to Euler angles for output
            roll, pitch, yaw = ComplementaryFilter.quaternion_to_euler(cf_quaternion)
            cf_euler = np.array([roll, pitch, yaw])

            # Step 3: Extract vision data (AprilTags pose)
            vision_position = None
            vision_orientation = None

            if packet.sensors and packet.sensors.vision:
                vision_data = packet.sensors.vision
                if isinstance(vision_data, dict) and "apriltag_pose" in vision_data:
                    apriltag_pose = vision_data["apriltag_pose"]

                    if isinstance(apriltag_pose, dict):
                        if "position" in apriltag_pose:
                            vision_position = np.array(apriltag_pose["position"], dtype=np.float64)
                        if "orientation" in apriltag_pose:
                            vision_orientation = np.array(apriltag_pose["orientation"], dtype=np.float64)
                        
                
            # Step 4: Stage 2 - Feed quaternoin and vision into EKF
            # Initialize EKF with vision if not initialzied
            if not self.ekf_filter.initialized:
                if vision_position is not None:
                    # Initialize with vision positio and orientation
                    init_orientation = vision_orientation if vision_orientation is not None else cf_quaternion
                    self.ekf_filter.initialize(
                        position=vision_position,
                        orientation=init_orientation
                    )
                else:
                    # Cannot init without vision, return CF output only
                    return {
                        'position': cf_state['position'].tolist(),
                        'velocity': cf_state['velocity'].tolist(),
                        'orientation': cf_quaternion.tolist(),
                        'orientation_euler': cf_euler.tolist(),
                        'angular_velocity': cf_state['angular_velocity'].tolist(),
                        'initialized': False,
                        'stage': 'complementary_filter_only',
                        'timestamp': time.time()
                    }

            # Update EKF with IMU orientation from CF
            ekf_state = self.ekf_filter.update_with_imu_orientation(cf_quaternion)

            # Update EKF with vision when available
            if vision_position is not None:
                ekf_state = self.ekf_filter.update_with_vision(
                    position=vision_position,
                    orientation=vision_orientation
                )
            final_orientation = ekf_state.orientation
            final_roll, final_pitch, final_yaw = ComplementaryFilter.quaternion_to_euler(final_orientation)

            # Return final fused pose
            result = {
                "position": ekf_state.position.tolist(),
                "velocity": ekf_state.velocity.tolist(),
                "orientation": final_orientation.tolist(),
                "orientation_euler": [final_roll, final_pitch, final_yaw],
                "angular_velocity": ekf_state.angular_velocity.tolist(),
                "initialized": True,
                "stage": "ekf_fusion",
                "timestamp": ekf_state.timestamp
            }

            # Add CF intermediate result for debugging/comparison
            result["cf_orientation"] = cf_quaternion.tolist()
            result["cf_orientation_euler"] = cf_euler.tolist()

            # Add uncertainty estimates
            result["position_uncertainty"] = float(self.ekf_filter.get_position_uncertainty())
            result["orientation_uncertainty"] = float(self.ekf_filter.get_orientation_uncertainty())
            
            return result
        
        except Exception as e:
            print(f"Error processing IMU+vision fusion: {e}")
            import traceback
            traceback.print_exc()
            return None

    