import cv2
import numpy as np

from tags.tag_detector import TagDetectionResult
from tags.tag_alignment import TagAlignmentDetector, TagAlignment
from cup import CupDetectionResult

def draw_axis_on_tag(frame, camera_matrix, dist_coeffs, rvecs, tvecs, length=0.01):
    """
    Draw coordinate axes on the frame for all detected tags.
    
    Args:
        frame: Input frame
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors for all tags (shape: N, 1, 3)
        tvecs: Translation vectors for all tags (shape: N, 1, 3)
        length: Length of the axes
    """

    if rvecs is None or tvecs is None or len(rvecs) == 0 or len(tvecs) == 0:
        return frame
    
    try:
        # Define the axis points in 3D space
        axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
        
        # Draw axes for each detected tag
        for i in range(len(rvecs)):
            if i < len(tvecs):
                # Reshape rvec and tvec to the correct format for projectPoints
                # estimatePoseSingleMarkers returns arrays with shape (1, 1, 3), so we need to flatten them first
                rvec = rvecs[i].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)
                tvec = tvecs[i].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)

                # Project 3D points to 2D
                imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                imgpts = np.int32(imgpts).reshape(-1, 2)
                
                # Draw the axes for this tag
                frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X axis - Red
                frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Y axis - Green
                frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Z axis - Blue
                
    except Exception as e:
        print(f"Error drawing axes: {e}")

    return frame


def calculate_camera_relative_coordinates(tag_detection_result):
    """
    Calculate camera-relative coordinates for all detected tags.
    
    Args:
        tag_detection_result: TagDetectionResult from TagDetector
    
    Returns:
        Dictionary mapping tag_id to coordinate information
    """
    coordinates = {}
    
    if (tag_detection_result.tag_ids is None or 
        tag_detection_result.tvecs is None or 
        tag_detection_result.rvecs is None):
        return coordinates
    
    for i, tag_id in enumerate(tag_detection_result.tag_ids):
        if (i < len(tag_detection_result.tvecs) and 
            i < len(tag_detection_result.rvecs)):
            
            # Get position and rotation
            tvec = tag_detection_result.tvecs[i].flatten()
            rvec = tag_detection_result.rvecs[i].flatten()
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler angles (roll, pitch, yaw)
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0
            
            # Convert to degrees
            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            
            # Calculate distance from camera
            distance = np.linalg.norm(tvec)
            
            coordinates[int(tag_id)] = {
                'position': {
                    'x': float(tvec[0]),  # meters
                    'y': float(tvec[1]),  # meters
                    'z': float(tvec[2])   # meters
                },
                'orientation': {
                    'roll': float(roll_deg),   # degrees
                    'pitch': float(pitch_deg), # degrees
                    'yaw': float(yaw_deg)      # degrees
                },
                'distance': float(distance),  # meters from camera
                'coordinate_system': 'camera_relative'
            }
    
    return coordinates


def draw_tag_coordinates(frame, tag_detection_result, camera_matrix, dist_coeffs):
    """
    Draw coordinate information for each detected tag on the frame.
    
    Args:
        frame: Input frame
        tag_detection_result: TagDetectionResult from TagDetector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    if (tag_detection_result.tag_ids is None or 
        tag_detection_result.tvecs is None or 
        tag_detection_result.corners is None):
        return frame
    
    # Calculate coordinates
    coordinates = calculate_camera_relative_coordinates(tag_detection_result)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    
    for i, tag_id in enumerate(tag_detection_result.tag_ids):
        if (i < len(tag_detection_result.corners) and 
            int(tag_id) in coordinates):
            
            # Get tag corners
            corners = tag_detection_result.corners[i][0]  # Shape: (4, 2)
            
            # Calculate center of tag
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            
            # Get coordinate information
            coord_info = coordinates[int(tag_id)]
            pos = coord_info['position']
            orient = coord_info['orientation']
            distance = coord_info['distance']
            
            # Create coordinate text
            coord_text = [
                f"Tag {tag_id}:",
                f"Pos: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})m",
                f"Dist: {distance:.2f}m",
                f"RPY: ({orient['roll']:.1f}, {orient['pitch']:.1f}, {orient['yaw']:.1f})deg"
            ]
            
            # Position text above the tag
            text_x = center_x - 80
            text_y = center_y - 80
            
            # Draw background rectangle for each line
            for j, text_line in enumerate(coord_text):
                (text_w, text_h), _ = cv2.getTextSize(text_line, font, scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (text_x - 2, text_y - text_h - 2 + j * 15),
                    (text_x + text_w + 2, text_y + 2 + j * 15),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame, 
                    text_line, 
                    (text_x, text_y + j * 15), 
                    font, 
                    scale, 
                    (255, 255, 255), 
                    thickness, 
                    cv2.LINE_AA
                )
    
    return frame


def draw_alignment_line(frame, camera_matrix, dist_coeffs, tag1_pos, tag2_pos, color=(0, 255, 255), thickness=2):
    """
    Draw a line connecting two tags to show alignment.
    
    Args:
        frame: Input frame
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        tag1_pos: 3D position of first tag
        tag2_pos: 3D position of second tag
        color: Line color (BGR)
        thickness: Line thickness
    """
    try:
        # Project 3D points to 2D
        points_3d = np.array([tag1_pos, tag2_pos], dtype=np.float32).reshape(-1, 3)
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)
        
        # Draw line between the two points
        cv2.line(frame, tuple(points_2d[0]), tuple(points_2d[1]), color, thickness)
        
        # Draw small circles at the tag positions
        cv2.circle(frame, tuple(points_2d[0]), 5, color, -1)
        cv2.circle(frame, tuple(points_2d[1]), 5, color, -1)
        
    except Exception as e:
        print(f"Error drawing alignment line: {e}")


def draw_alignment_info(frame, alignments, camera_matrix, dist_coeffs, tag_detection_result):
    """
    Draw alignment information on the frame.
    
    Args:
        frame: Input frame
        alignments: List of TagAlignment objects
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        tag_detection_result: TagDetectionResult for getting tag positions
    """
    if not alignments or tag_detection_result.tag_ids is None:
        return frame
    
    # Create a mapping from tag_id to position
    tag_positions = {}
    if (tag_detection_result.tag_ids is not None and 
        tag_detection_result.tvecs is not None):
        for i, tag_id in enumerate(tag_detection_result.tag_ids):
            if i < len(tag_detection_result.tvecs):
                tag_positions[int(tag_id)] = tag_detection_result.tvecs[i].flatten()
    
    # Draw alignment lines and info
    for i, alignment in enumerate(alignments):
        if alignment.tag1_id in tag_positions and alignment.tag2_id in tag_positions:
            # Choose color based on alignment type
            if alignment.alignment_type == "vertical":
                color = (0, 255, 0)  # Green for vertical
            elif alignment.alignment_type == "horizontal":
                color = (255, 0, 0)  # Blue for horizontal
            elif alignment.alignment_type == "parallel":
                color = (0, 255, 255)  # Yellow for parallel
            else:
                color = (255, 255, 255)  # White for other
            
            # Draw alignment line
            draw_alignment_line(
                frame, camera_matrix, dist_coeffs,
                tag_positions[alignment.tag1_id],
                tag_positions[alignment.tag2_id],
                color, 3
            )
    
    return frame


def draw_alignment_status(frame, alignments, alignment_detector):
    """
    Draw alignment status text on the frame.
    
    Args:
        frame: Input frame
        alignments: List of TagAlignment objects
        alignment_detector: TagAlignmentDetector instance
    """
    if not alignments:
        # No alignments detected
        text = "No alignments detected"
        color = (0, 0, 255)  # Red
    else:
        # Show alignment count and details
        text = f"Alignments: {len(alignments)}"
        color = (0, 255, 0)  # Green
    
    # Draw main status
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    margin = 10
    x_left = margin
    y_top = margin + text_h
    
    # Draw background rectangle
    bg_margin = 6
    cv2.rectangle(
        frame,
        (x_left - bg_margin, y_top - text_h - bg_margin),
        (x_left + text_w + bg_margin, y_top + bg_margin // 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x_left, y_top), font, scale, color, thickness, cv2.LINE_AA)
    
    # Draw detailed alignment information
    if alignments:
        y_offset = y_top + 25
        for i, alignment in enumerate(alignments[:3]):  # Show max 3 alignments
            desc = alignment_detector.get_alignment_description(alignment)
            conf_text = f"({alignment.confidence:.2f})"
            
            # Choose color based on alignment type
            if alignment.alignment_type == "vertical":
                color = (0, 255, 0)  # Green
            elif alignment.alignment_type == "horizontal":
                color = (255, 0, 0)  # Blue
            elif alignment.alignment_type == "parallel":
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            # Draw alignment description
            cv2.putText(frame, desc, (x_left, y_offset), font, 0.5, color, 1, cv2.LINE_AA)
            #cv2.putText(frame, conf_text, (x_left + 200, y_offset), font, 0.5, color, 1, cv2.LINE_AA) # Add this to include confidence
            y_offset += 20
    
    return frame


def visualize_tag_detection(frame_bgr: np.ndarray, tag_detection_result: TagDetectionResult, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    '''
    Visualize the tag detection result with alignment detection.
    '''

    try:
        # Initialize alignment detector
        alignment_detector = TagAlignmentDetector()
        
        # Draw detected markers
        if (tag_detection_result.tag_ids is not None and 
            len(tag_detection_result.tag_ids) > 0 and 
            tag_detection_result.corners is not None and 
            len(tag_detection_result.corners) > 0):
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners=tag_detection_result.corners, ids=np.array(tag_detection_result.tag_ids))
        
            # Draw coordinate axes
            if tag_detection_result.rvecs is not None and tag_detection_result.tvecs is not None:
                frame_bgr = draw_axis_on_tag(frame_bgr, camera_matrix, dist_coeffs, rvecs=tag_detection_result.rvecs, tvecs=tag_detection_result.tvecs, length=0.01)
            
            # Draw tag coordinates
            frame_bgr = draw_tag_coordinates(frame_bgr, tag_detection_result, camera_matrix, dist_coeffs)
            
            # Detect alignments
            alignments = alignment_detector.detect_alignments(tag_detection_result)
            
            # Draw alignment information
            if alignments:
                frame_bgr = draw_alignment_info(frame_bgr, alignments, camera_matrix, dist_coeffs, tag_detection_result)
            
            # Draw alignment status
            frame_bgr = draw_alignment_status(frame_bgr, alignments, alignment_detector)
            
            # Print alignment info to terminal
            '''
            if alignments:
                print(f"Detected {len(alignments)} alignments:", end=" ")
                for alignment in alignments:
                    desc = alignment_detector.get_alignment_description(alignment)
                    print(f"{desc}({alignment.confidence:.2f})", end=" ")
                print()  # New line
            else:
                print("No alignments detected", end="\r")
            '''
    except Exception as e:
        print(f"Error visualizing tag detection: {e}")

    return frame_bgr

def visualize_cup_detection(frame_bgr: np.ndarray, cup_detection_result: CupDetectionResult) -> np.ndarray:

    if cup_detection_result.detected and cup_detection_result.bounding_box is not None:
        x, y, w, h = cup_detection_result.bounding_box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if cup_detection_result.pixel_center is not None:
            cx, cy = cup_detection_result.pixel_center
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
    # Draw detection flag in top-right corner
    flag_text = "True" if cup_detection_result.detected else "False"
    color = (0, 200, 0) if cup_detection_result.detected else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(flag_text, font, scale, thickness)
    margin = 10
    x_right = frame_bgr.shape[1] - text_w - margin
    y_top = margin + text_h
    bg_margin = 6
    cv2.rectangle(
        frame_bgr,
        (x_right - bg_margin, y_top - text_h - bg_margin),
        (x_right + text_w + bg_margin, y_top + bg_margin // 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame_bgr, flag_text, (x_right, y_top), font, scale, color, thickness, cv2.LINE_AA)

    return frame_bgr