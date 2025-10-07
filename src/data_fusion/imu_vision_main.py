import sys
import os
import numpy as np
import cv2
import time

# Add your src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from data_fusion.fusion_system import FiducialDepthSystem
from data_fusion.smoothing import IMUData
from vision.tags.utils.camera_calibration import load_calibration_json

# IMU reader
from sensors.imu_reader import IMUReader

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
        enable_imu_smoothing=True,
        smoothing_method="complementary"
    )
    print("Fusion system created successfully")

    # 3. Connect to IMU
    imu_reader = IMUReader(port="/dev/cu.usbmodem1101")  # Or use None for auto-detect
    if not imu_reader.activate():
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
            
            # Read IMU data (already in IMUData format from the reader)
            imu_data = imu_reader.get_latest_data()

            result = fusion_system.process_frame(
                tag_detection_result=None,
                cup_detection_result=None,
                imu_data=imu_data
            )

            # Display results
            #print(f"Hand pose detected: {result['hand_pose']['detected']}")
            #print(imu_data)
            if imu_data:
                print(f"IMU: Accel=({imu_data.linear_acceleration[0]:.2f}, "
                      f"{imu_data.linear_acceleration[1]:.2f}, "
                      f"{imu_data.linear_acceleration[2]:.2f})       ", end="\r", flush=True)
            
            # Show camera frame
            cv2.imshow("IMU-Vision Integration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        imu_reader.deactivate()
        camera.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    main()