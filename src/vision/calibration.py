import depthai as dai
import json
import numpy as np

def extract_camera_intrinsics(calibration_data, camera_socket=0):
    """
    Extract camera matrix and distortion coefficients from calibration JSON.
    
    Args:
        calibration_data: Parsed JSON calibration data
        camera_socket: Camera socket number (0=RGB, 1=Left, 2=Right)
    
    Returns:
        tuple: (camera_matrix, distortion_coeffs) as numpy arrays
    """
    try:
        # Navigate to camera data array
        camera_data_array = calibration_data['cameraData']
        
        # Find camera with matching socket
        camera_data = None
        for camera_entry in camera_data_array:
            if camera_entry[0] == camera_socket:
                camera_data = camera_entry[1]
                break
        
        if camera_data is None:
            print(f"Error: Could not find camera with socket {camera_socket}")
            print(f"Available cameras: {[entry[0] for entry in camera_data_array]}")
            return None, None
        
        # Extract intrinsic matrix
        intrinsics = camera_data['intrinsicMatrix']
        camera_matrix = np.array([
            [intrinsics[0][0], intrinsics[0][1], intrinsics[0][2]],
            [intrinsics[1][0], intrinsics[1][1], intrinsics[1][2]],
            [intrinsics[2][0], intrinsics[2][1], intrinsics[2][2]]
        ], dtype=np.float32)
        
        # Extract distortion coefficients (take first 5 coefficients)
        distortion = camera_data['distortionCoeff']
        dist_coeffs = np.array(distortion[:5], dtype=np.float32)
        
        return camera_matrix, dist_coeffs
        
    except KeyError as e:
        print(f"Error: Could not find camera data: {e}")
        return None, None
    except Exception as e:
        print(f"Error extracting camera intrinsics: {e}")
        return None, None

def print_camera_info(camera_matrix, dist_coeffs, camera_name="RGB Camera"):
    """Print camera calibration information in a readable format."""
    print(f"\n{camera_name} Calibration:")
    print("=" * 50)
    
    if camera_matrix is not None:
        print(f"Camera Matrix (3x3):")
        print(f"  fx = {camera_matrix[0,0]:.2f}")
        print(f"  fy = {camera_matrix[1,1]:.2f}")
        print(f"  cx = {camera_matrix[0,2]:.2f}")
        print(f"  cy = {camera_matrix[1,2]:.2f}")
        print(f"Full matrix:\n{camera_matrix}")
    
    if dist_coeffs is not None:
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {dist_coeffs[0]:.6f}")
        print(f"  k2 = {dist_coeffs[1]:.6f}")
        print(f"  p1 = {dist_coeffs[2]:.6f}")
        print(f"  p2 = {dist_coeffs[3]:.6f}")
        print(f"  k3 = {dist_coeffs[4]:.6f}")
        print(f"Full coefficients: {dist_coeffs}")

# Main execution
device = dai.Device()
print(f"EEPROM available: {device.isEepromAvailable()}")

choice = input("Enter 'u' for user calibration, 'f' for factory calibration: ")

if choice == 'u':
    try:
        user_calibration_json = json.dumps(device.readCalibration().eepromToJson(), indent=2)
        calibration_data = device.readCalibration().eepromToJson()
        
        print("User calibration JSON:")
        print(user_calibration_json)
        
        # Show available cameras
        camera_data_array = calibration_data['cameraData']
        print(f"\nAvailable cameras: {[entry[0] for entry in camera_data_array]}")
        
        # Extract camera intrinsics for RGB camera (socket 0)
        camera_matrix, dist_coeffs = extract_camera_intrinsics(calibration_data, camera_socket=0)
        print_camera_info(camera_matrix, dist_coeffs, "User RGB Camera (Socket 0)")
        
    except Exception as e:
        print(f"No user calibration: {e}")

elif choice == 'f':
    try:
        factory_calibration_json = json.dumps(device.readFactoryCalibration().eepromToJson(), indent=2)
        calibration_data = device.readFactoryCalibration().eepromToJson()
        
        print("Factory calibration JSON:")
        print(factory_calibration_json)
        
        # Show available cameras
        camera_data_array = calibration_data['cameraData']
        print(f"\nAvailable cameras: {[entry[0] for entry in camera_data_array]}")
        
        # Extract camera intrinsics for RGB camera (socket 0)
        camera_matrix, dist_coeffs = extract_camera_intrinsics(calibration_data, camera_socket=0)
        print_camera_info(camera_matrix, dist_coeffs, "Factory RGB Camera (Socket 0)")
        
        # Also try to extract for other cameras if available
        for socket_id in [1, 2]:
            if any(entry[0] == socket_id for entry in camera_data_array):
                cam_matrix, dist_coeffs_other = extract_camera_intrinsics(calibration_data, camera_socket=socket_id)
                if cam_matrix is not None:
                    camera_name = f"Factory Camera Socket {socket_id}"
                    print_camera_info(cam_matrix, dist_coeffs_other, camera_name)
        
        # Save RGB camera calibration to file for use in other scripts
        if camera_matrix is not None and dist_coeffs is not None:
            calibration_dict = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'source': 'factory_calibration',
                'camera_socket': 0,
                'resolution': f"{calibration_data['cameraData'][0][1]['width']}x{calibration_data['cameraData'][0][1]['height']}"
            }
            
            with open('camera_calibration.json', 'w') as f:
                json.dump(calibration_dict, f, indent=2)
            print(f"\nRGB Camera calibration saved to 'camera_calibration.json'")
        
    except Exception as e:
        print(f"No factory calibration: {e}")

else:
    print("Invalid choice")

def load_calibration_from_file(filename='camera_calibration.json'):
    """
    Load camera calibration from a saved JSON file.
    
    Args:
        filename: Path to the calibration JSON file
    
    Returns:
        tuple: (camera_matrix, distortion_coeffs) as numpy arrays, or (None, None) if failed
    """
    try:
        with open(filename, 'r') as f:
            calibration_dict = json.load(f)
        
        camera_matrix = np.array(calibration_dict['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(calibration_dict['distortion_coefficients'], dtype=np.float32)
        
        print(f"Loaded calibration from {filename}")
        print(f"Source: {calibration_dict.get('source', 'unknown')}")
        
        return camera_matrix, dist_coeffs
        
    except FileNotFoundError:
        print(f"Calibration file '{filename}' not found")
        return None, None
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None

# Example usage for loading saved calibration
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Camera Calibration Tool")
    print("="*60)
    
    # After running the calibration extraction above, you can load it like this:
    # camera_matrix, dist_coeffs = load_calibration_from_file('camera_calibration.json')
    # if camera_matrix is not None:
    #     print_camera_info(camera_matrix, dist_coeffs, "Loaded Camera")





