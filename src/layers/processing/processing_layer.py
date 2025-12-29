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
from .imu_utils import normalize_quaternion, quaternion_multiply, quaternion_conjugate, madgwick_update
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

        # IMU movement intention state variables
        self.imu_gravity_removal_method = "madgwick"
        self.imu_velocity_threshold = 0.2 # m/s
        self.imu_direction_timeout = 4.0 # seconds
        
        # Velocity tracking state
        self.imu_velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] in m/s
        self.imu_last_timestamp = None
        
        # Stillness detection parameters
        self.GRAVITY = 9.81  # m/s²
        self.ACCEL_STILL_THRESH = 0.5  # m/s²
        self.GYRO_STILL_THRESH = 0.1  # rad/s
        self.DEADBAND_THRESH = 0.15  # m/s²
        self.VELOCITY_DECAY = 0.95  # Decay factor per sample
        self.GRAVITY_EMA_ALPHA = 0.05  # EMA smoothing for gravity vector
        
        # Madgwick filter state (for 'madgwick' method)
        self.imu_q_ws = None  # Quaternion [w, x, y, z] (world to sensor)
        self.imu_q_ws_initialized = False
        self.imu_madgwick_beta = 0.05  # Madgwick filter gain
        self.imu_gyro_bias = np.array([0.0, 0.0, 0.0])
        self.imu_still_time_for_bias = 0.0
        self.imu_bias_update_duration = 0.5  # seconds
        
        # Direction detection state
        self.imu_last_direction = None
        self.imu_last_direction_time = None

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
        Process IMU data to detect high-level movement intentions using velocity-based detection.
        
        Steps:
        1. Extract accelerometer and gyroscope data from IMU window
        2. Compute time step (dt) from timestamps
        3. Initialize Madgwick filter quaternion when device is still
        4. Update gyro bias when device is still
        5. Update Madgwick filter to estimate orientation
        6. Remove gravity from acceleration using rotation matrix
        7. Detect stillness and reset velocity if still
        8. Integrate linear acceleration to get velocity (with deadband and decay)
        9. Determine dominant direction based on velocity (with timeout to prevent rapid switching)
        10. Return movement intention with confidence
        
        Returns:
            dict with keys:
                - direction: str ("forward", "backward", "left", "right", "up", "down", "none")
                - confidence: float - confidence in direction detection (0.0 to 1.0)
                - is_still: bool - whether device is currently still
                - timestamp: float - current timestamp
        """
        try:
            imu_data = human_data.imu
            if imu_data is None or imu_data.size == 0:
                return None

            # Extract accel and gyro data
            accel_data = imu_data[:, :3]
            gyro_data = imu_data[:, 3:]

            # Use mean values for processing 
            accel_raw = np.mean(accel_data, axis=0)
            gyro_raw = np.mean(gyro_data, axis=0)

            current_timestamp = time.time()
            if self.imu_last_timestamp is not None:
                dt = current_timestamp - self.imu_last_timestamp
                dt = min(max(dt, 0.0), 0.1)
            else:
                dt = 0.01
            self.imu_last_timestamp = current_timestamp

            # Remove gravity using Madgwick filter
            if not self.imu_q_ws_initialized:
                accel_mag_init = np.linalg.norm(accel_raw)
                gyro_mag_init = np.linalg.norm(gyro_raw)
                accel_diff_init = abs(accel_mag_init - self.GRAVITY)

                is_still_init = (accel_diff_init < self.ACCEL_STILL_THRESH and gyro_mag_init < self.GYRO_STILL_THRESH)

                if is_still_init and accel_mag_init > 0.1:
                    # Initialize quaternion from gravity vector
                    g_s_normalized = accel_raw / accel_mag_init
                    g_w_normalized = np.array([0.0, 0.0, 1.0])
                    
                    # Find rotation that aligns g_s to g_w using Rodrigues' rotation formula
                    v = np.cross(g_s_normalized, g_w_normalized)
                    s = np.linalg.norm(v)
                    c = np.dot(g_s_normalized, g_w_normalized)
                    
                    if s < 1e-6:
                        # Already aligned or opposite
                        if c > 0:
                            self.imu_q_ws = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
                        else:
                            # 180 degree rotation - choose arbitrary perpendicular axis
                            self.imu_q_ws = np.array([0.0, 1.0, 0.0, 0.0])
                    else:
                        # Rodrigues' rotation formula to rotation matrix
                        vx = np.array([
                            [0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                        ])
                        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
                        
                        # Convert rotation matrix to quaternion
                        trace = np.trace(R)
                        if trace > 0:
                            s_q = np.sqrt(trace + 1.0) * 2
                            w = 0.25 * s_q
                            x = (R[2, 1] - R[1, 2]) / s_q
                            y = (R[0, 2] - R[2, 0]) / s_q
                            z = (R[1, 0] - R[0, 1]) / s_q
                            self.imu_q_ws = normalize_quaternion(np.array([w, x, y, z]))
                        else:
                            # Fallback to identity
                            self.imu_q_ws = np.array([1.0, 0.0, 0.0, 0.0])
                    self.imu_q_ws_initialized = True

            # Update gyro bias when still
            accel_mag_check = np.linalg.norm(accel_raw)
            gyro_mag_check = np.linalg.norm(gyro_raw)
            accel_diff_check = abs(accel_mag_check - self.GRAVITY)

            is_still_for_bias = (accel_diff_check < self.ACCEL_STILL_THRESH and gyro_mag_check < self.GYRO_STILL_THRESH)
            if is_still_for_bias:
                self.imu_still_time_for_bias += dt
                if self.imu_still_time_for_bias >= self.imu_bias_update_duration:
                    BIAS_EMA_ALPHA = 0.02
                    self.imu_gyro_bias = (1 - BIAS_EMA_ALPHA) * self.imu_gyro_bias + BIAS_EMA_ALPHA * gyro_raw
            else:
                self.imu_still_time_for_bias = 0.0

            # Apply gyro bias correction
            gyro_corrected = gyro_raw - self.imu_gyro_bias

            # Update Madgwick filter
            if self.imu_q_ws_initialized:
                self.imu_q_ws = madgwick_update(self.imu_q_ws, accel_raw, gyro_corrected, dt, self.imu_madgwick_beta)

                # Compute gravity in sensor frame using rotation matrix
                w, x, y, z = self.imu_q_ws
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])

                # Gravity in world frame (Z up)
                g_world = np.array([0.0, 0.0, self.GRAVITY])
                # Rotate to sensor frame: g_sensor = R^T * g_world
                g_sensor = R.T @ g_world

                # Remove gravity from acceleration
                linear_accel = accel_raw - g_sensor
            else:
                linear_accel = accel_raw - np.array([0.0, 0.0, self.GRAVITY])
        
            # Detect stillness reset velocity
            a_lin_mag = np.linalg.norm(linear_accel)
            gyro_mag = np.linalg.norm(gyro_raw)
            
            is_still = (a_lin_mag < self.ACCEL_STILL_THRESH and gyro_mag < self.GYRO_STILL_THRESH)
            if is_still:
                self.imu_velocity = np.array([0.0, 0.0, 0.0])
            else:
                if a_lin_mag < self.DEADBAND_THRESH:
                    # Decay velocity when acceleration is small (reduces drift)
                    self.imu_velocity *= self.VELOCITY_DECAY
                else:
                    self.imu_velocity += linear_accel * dt

            direction = "none"
            confidence = 0.0
            
            # Check if enough time has passed since last direction detection
            time_since_last = (current_timestamp - self.imu_last_direction_time 
                            if self.imu_last_direction_time is not None else float("inf"))
            can_detect_new = time_since_last >= self.imu_direction_timeout
            
            if can_detect_new:
                # Find the axis with the largest absolute velocity
                abs_velocities = np.abs(self.imu_velocity)
                max_idx = np.argmax(abs_velocities)
                max_vel_mag = abs_velocities[max_idx]
                
                if max_vel_mag > self.imu_velocity_threshold:
                    # Determine direction based on dominant axis and sign
                    if max_idx == 0:
                        direction = "forward" if self.imu_velocity[0] < 0 else "backward"
                    elif max_idx == 1:
                        direction = "left" if self.imu_velocity[1] < 0 else "right"
                    else:
                        direction = "up" if self.imu_velocity[2] > 0 else "down"
                    
                    # Only update if direction changed
                    if direction != self.imu_displayed_direction:
                        self.imu_displayed_direction = direction
                        self.imu_last_direction = direction
                        self.imu_last_direction_time = current_timestamp
                    
                    # Calculate confidence based on velocity magnitude and dominance
                    total_vel_mag = np.linalg.norm(self.imu_velocity)
                    axis_dominance = max_vel_mag / total_vel_mag if total_vel_mag > 0 else 0.0
                    mag_factor = min(max_vel_mag / (self.imu_velocity_threshold * 2), 1.0)
                    confidence = axis_dominance * mag_factor
                else:
                    # Below threshold
                    self.imu_displayed_direction = None
                    direction = "none"
                    confidence = 0.0
            else:
                # During timeout, keep showing last displayed direction
                if self.imu_displayed_direction is not None:
                    direction = self.imu_displayed_direction
                    # Recalculate confidence for display (lower during timeout)
                    abs_velocities = np.abs(self.imu_velocity)
                    max_idx = np.argmax(abs_velocities)
                    max_vel_mag = abs_velocities[max_idx]
                    total_vel_mag = np.linalg.norm(self.imu_velocity)
                    if total_vel_mag > 0:
                        axis_dominance = max_vel_mag / total_vel_mag
                        mag_factor = min(max_vel_mag / (self.imu_velocity_threshold * 2), 1.0)
                        confidence = axis_dominance * mag_factor * 0.5  # Lower confidence during timeout
                else:
                    direction = "none"
                    confidence = 0.0

            return {
                "direction": direction,
                "confidence": float(confidence),
                "is_still": bool(is_still),
                "timestamp": current_timestamp
            }
        except Exception as e:
            print(f"Error processing IMU intention: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_imu_vision_fusion(self, packet) -> Optional[dict]:
        """
        Imu and vision fusion in two stages.
        
        Stage 1: Complementary filter for orientation (IMU)
        Stage 2: Extended Kalman Filter for position and velocity (IMU + Vision)

            # Step 2: Calculate velocity
            imu_velocity = np.zeros_like(imu_data_transposed)
            for i in range(1, len(imu_data_transposed)):
                dt = imu_data_transposed[i, 0] - imu_data_transposed[i-1, 0]
                imu_velocity[i] = imu_data_transposed[i] - imu_data_transposed[i-1] / dt
            
            return imu_velocity
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

    