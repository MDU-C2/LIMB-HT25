import sys
import os
import numpy as np
import cv2
import time

# Add your src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from imu_vision.fusion_system import FiducialDepthSystem
from imu_vision.smoothing import IMUData
from vision.tags.camera_calibration import load_calibration_json

# Your IMU reader
from serial_reader import IMUSerialReader

def main():
    print("IMU-Vision System")
    print("="*50)

    # 1. Load camera calibration
    try:
        camera_matrix, dist_coeffs = load_calibration_json("vision/tags/camera_calibration.json")
        print("Camera calibration loaded successfully")
    except:
        camera_matrix = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        print("Using default camera matrix")

    # 2. Create fusion system
    fusion_system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        enable_imu_smoothin=True,
        smoothing_method="complementary"
    )
    print("Fusion system created successfully")

    # 3. Connect to IMU
    imu_reader = IMUSerialReader(port="/dev/usbmodem1101") # CHECK IF THIS IS CORRECT
    if not imu_reader.connect():
        print("Failed to connect to IMU")
        return
    print("Connected to IMU successfully")

    # 4. Initalize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Failed to open camera")
        return
    print("Camera opened successfully")

    # 5. Main loop
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Read IMU data
            imu_raw = imu_reader.read_imu_data()
            imu_data = None

            if imu_raw:
                    # Convert to IMUData format
                    imu_data = IMUData(
                        angular_velocity=np.array([
                            imu_raw['gyro']['x'],
                            imu_raw['gyro']['y'], 
                            imu_raw['gyro']['z']
                        ]),
                        linear_acceleration=np.array([
                            imu_raw['accel']['x'],
                            imu_raw['accel']['y'],
                            imu_raw['accel']['z']
                        ]),
                        timestamp=time.time()
                    )

            result = fusion_system.process_frame(
                tag_detection=None,
                cup_detection=None,
                imu_data=imu_data
            )

            # Display results
            print(f"Hand pose detected: {result['hand_pose']['detected']}")
            if imu_data:
                print(f"IMU: Accel=({imu_data.linear_acceleration[0]:.2f}, "
                      f"{imu_data.linear_acceleration[1]:.2f}, "
                      f"{imu_data.linear_acceleration[2]:.2f})")
            
            # Show camera frame
            cv2.imshow("IMU-Vision Integration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        imu_reader.disconnect()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()