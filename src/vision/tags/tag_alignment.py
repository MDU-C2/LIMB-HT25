from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class TagAlignment:
    """Represents an alignment between two tags."""
    tag1_id: int
    tag2_id: int
    alignment_type: str  # 'vertical', 'horizontal', 'parallel', etc.
    confidence: float  # 0.0 to 1.0
    distance_m: float
    angle_deg: float


@dataclass
class TagInfo:
    """Information about a detected tag."""
    tag_id: int
    position: np.ndarray  # 3D position in camera frame
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    transform_matrix: np.ndarray  # 4x4 transform matrix


class RobotArmTagConfig:
    """Configuration for robot arm tag placement."""
    
    # Define tag IDs and their expected positions on the robot arm
    TAG_POSITIONS = {
        0: "upper_arm_top", 
        1: "forearm_top",     
        2: "hand_top",
        3: "upper_arm_bottom",
        4: "forearm_bottom",
        5: "hand_bottom",
        6: "hand_side",
    }
    
    # Define body parts and their associated tags
    BODY_PARTS = {
        "upper_arm": [0, 3],  # upper_arm_top, upper_arm_bottom
        "forearm": [1, 4],    # forearm_top, forearm_bottom
        "hand": [2, 5, 6]     # hand_top, hand_bottom, hand_side
    }
    
    # Define expected alignments between tags
    EXPECTED_ALIGNMENTS = [
        # Vertical alignments (tags should be vertically aligned)
        
        #(0, 1, "parallel"),    # upper_arm_top with forearm_top
        #(1, 2, "parallel"),    # forearm_top with hand_top
        (0, 1, "vertical"),    # upper_arm_top with forearm_top
        (1, 2, "vertical"),    # forearm_top with hand_top
        #(0, 1, "horizontal"),    # upper_arm_top with forearm_top
        #(1, 2, "horizontal"),    # forearm_top with hand_top
        (3, 4, "parallel"),    # upper_arm_bottom with forearm_bottom
        (4, 5, "parallel"),    # forearm_bottom with hand_bottom
        
        
        (1, 5, "parallel"),    # forearm_top with hand_bottom
        (1, 6, "parallel"),    # forearm_top with hand_side
        (4, 6, "parallel"),    # forearm_bottom with hand_side

        # Could also add 0-2, 3-5, 0-6, 3-6
    ]


class TagAlignmentDetector:
    """Detects alignments between ArUco tags on the robot arm."""
    
    def __init__(self, alignment_threshold_deg: float = 15.0, distance_threshold_m: float = 0.5):
        """
        Initialize the alignment detector.
        
        Args:
            alignment_threshold_deg: Maximum angle deviation for alignment (degrees)
            distance_threshold_m: Maximum distance for considering tags as aligned (meters)
        """
        self.alignment_threshold_deg = alignment_threshold_deg
        self.distance_threshold_m = distance_threshold_m
        self.config = RobotArmTagConfig()
    
    def extract_tag_info(self, tag_detection_result) -> Dict[int, TagInfo]:
        """
        Extract tag information from detection result.
        
        Args:
            tag_detection_result: TagDetectionResult from TagDetector
            
        Returns:
            Dictionary mapping tag_id to TagInfo
        """
        tag_info_dict = {}
        
        if (tag_detection_result.tag_ids is None or 
            tag_detection_result.rvecs is None or 
            tag_detection_result.tvecs is None):
            return tag_info_dict
        
        for i, tag_id in enumerate(tag_detection_result.tag_ids):
            if i < len(tag_detection_result.rvecs) and i < len(tag_detection_result.tvecs):
                rvec = tag_detection_result.rvecs[i].flatten()
                tvec = tag_detection_result.tvecs[i].flatten()
                
                # Convert to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4 transform matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = tvec
                
                tag_info = TagInfo(
                    tag_id=int(tag_id),
                    position=tvec,
                    rotation_matrix=rotation_matrix,
                    transform_matrix=transform_matrix
                )
                
                tag_info_dict[int(tag_id)] = tag_info
        
        return tag_info_dict
    
    def check_vertical_alignment(self, tag1: TagInfo, tag2: TagInfo) -> Tuple[bool, float, float]:
        """
        Check if two tags are vertically aligned.
        
        Args:
            tag1: First tag info
            tag2: Second tag info
            
        Returns:
            Tuple of (is_aligned, confidence, angle_deg)
        """
        # Calculate the vector between the two tags
        vector = tag2.position - tag1.position
        
        # For vertical alignment, we expect the X and Z components to be small
        # compared to the Y component (assuming Y is up/down)
        horizontal_distance = np.sqrt(vector[0]**2 + vector[2]**2)
        vertical_distance = abs(vector[1])
        
        if vertical_distance < 0.01:  # Too close to determine alignment
            return False, 0.0, 0.0
        
        # Calculate the angle from vertical
        angle_rad = np.arctan2(horizontal_distance, vertical_distance)
        angle_deg = np.degrees(angle_rad)
        
        # Check if within threshold
        is_aligned = angle_deg <= self.alignment_threshold_deg
        
        # Calculate confidence (higher confidence for smaller angles)
        confidence = max(0.0, 1.0 - (angle_deg / self.alignment_threshold_deg))
        
        return is_aligned, confidence, angle_deg
    
    def check_horizontal_alignment(self, tag1: TagInfo, tag2: TagInfo) -> Tuple[bool, float, float]:
        """
        Check if two tags are horizontally aligned.
        
        Args:
            tag1: First tag info
            tag2: Second tag info
            
        Returns:
            Tuple of (is_aligned, confidence, angle_deg)
        """
        # Calculate the vector between the two tags
        vector = tag2.position - tag1.position
        
        # For horizontal alignment, we expect the Y component to be small
        # compared to the X and Z components
        horizontal_distance = np.sqrt(vector[0]**2 + vector[2]**2)
        vertical_distance = abs(vector[1])
        
        if horizontal_distance < 0.01:  # Too close to determine alignment
            return False, 0.0, 0.0
        
        # Calculate the angle from horizontal
        angle_rad = np.arctan2(vertical_distance, horizontal_distance)
        angle_deg = np.degrees(angle_rad)
        
        # Check if within threshold
        is_aligned = angle_deg <= self.alignment_threshold_deg
        
        # Calculate confidence (higher confidence for smaller angles)
        confidence = max(0.0, 1.0 - (angle_deg / self.alignment_threshold_deg))
        
        return is_aligned, confidence, angle_deg
    
    def check_parallel_alignment(self, tag1: TagInfo, tag2: TagInfo) -> Tuple[bool, float, float]:
        """
        Check if two tags are parallel (have similar orientations).
        
        Args:
            tag1: First tag info
            tag2: Second tag info
            
        Returns:
            Tuple of (is_aligned, confidence, angle_deg)
        """
        # Compare the Z-axes of both tags (assuming Z points out of the tag)
        z1 = tag1.rotation_matrix[:, 2]
        z2 = tag2.rotation_matrix[:, 2]
        
        # Calculate the angle between the Z-axes
        dot_product = np.dot(z1, z2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Check if within threshold
        is_aligned = angle_deg <= self.alignment_threshold_deg
        
        # Calculate confidence (higher confidence for smaller angles)
        confidence = max(0.0, 1.0 - (angle_deg / self.alignment_threshold_deg))
        
        return is_aligned, confidence, angle_deg
    
    def detect_alignments(self, tag_detection_result) -> List[TagAlignment]:
        """
        Detect all alignments between detected tags.
        
        Args:
            tag_detection_result: TagDetectionResult from TagDetector
            
        Returns:
            List of TagAlignment objects
        """
        alignments = []
        tag_info_dict = self.extract_tag_info(tag_detection_result)
        
        # Check each expected alignment
        for tag1_id, tag2_id, alignment_type in self.config.EXPECTED_ALIGNMENTS:
            if tag1_id in tag_info_dict and tag2_id in tag_info_dict:
                tag1 = tag_info_dict[tag1_id]
                tag2 = tag_info_dict[tag2_id]
                
                # Calculate distance between tags
                distance = np.linalg.norm(tag2.position - tag1.position)
                
                # Skip if tags are too far apart
                if distance > self.distance_threshold_m:
                    continue
                
                # Check alignment based on type
                if alignment_type == "vertical":
                    is_aligned, confidence, angle_deg = self.check_vertical_alignment(tag1, tag2)
                elif alignment_type == "horizontal":
                    is_aligned, confidence, angle_deg = self.check_horizontal_alignment(tag1, tag2)
                elif alignment_type == "parallel":
                    is_aligned, confidence, angle_deg = self.check_parallel_alignment(tag1, tag2)
                else:
                    continue
                
                if is_aligned:
                    alignment = TagAlignment(
                        tag1_id=tag1_id,
                        tag2_id=tag2_id,
                        alignment_type=alignment_type,
                        confidence=confidence,
                        distance_m=distance,
                        angle_deg=angle_deg
                    )
                    alignments.append(alignment)
        
        return alignments
    
    def get_tag_name(self, tag_id: int) -> str:
        """Get the human-readable name for a tag ID."""
        return self.config.TAG_POSITIONS.get(tag_id, f"tag_{tag_id}")
    
    def get_alignment_description(self, alignment: TagAlignment) -> str:
        """Get a human-readable description of an alignment."""
        tag1_name = self.get_tag_name(alignment.tag1_id)
        tag2_name = self.get_tag_name(alignment.tag2_id)
        
        return f"{tag1_name} <-> {tag2_name}"# ({alignment.alignment_type})"
    
    def detect_states(self, alignments: List[TagAlignment]) -> List[str]:
        """
        Detect states based on aligned tags from different body parts.
        A state consists of one tag from upper_arm, one from forearm, and one from hand.
        
        Args:
            alignments: List of TagAlignment objects
            
        Returns:
            List of state strings (e.g., ["state_0-1-6", "state_3-4-5"])
        """
        if not alignments:
            return []
        
        # Build a graph of which tags are aligned with each other
        aligned_tags = {}
        for alignment in alignments:
            tag1_id = alignment.tag1_id
            tag2_id = alignment.tag2_id
            
            if tag1_id not in aligned_tags:
                aligned_tags[tag1_id] = set()
            if tag2_id not in aligned_tags:
                aligned_tags[tag2_id] = set()
            
            aligned_tags[tag1_id].add(tag2_id)
            aligned_tags[tag2_id].add(tag1_id)
        
        # Find all detected tags organized by body part
        detected_tags_by_part = {
            "upper_arm": [],
            "forearm": [],
            "hand": []
        }
        
        for tag_id in aligned_tags.keys():
            for part_name, tag_ids in self.config.BODY_PARTS.items():
                if tag_id in tag_ids:
                    detected_tags_by_part[part_name].append(tag_id)
                    break
        
        # Find all valid states (one tag from each body part that are all aligned together)
        states = []
        
        for upper_arm_tag in detected_tags_by_part["upper_arm"]:
            for forearm_tag in detected_tags_by_part["forearm"]:
                for hand_tag in detected_tags_by_part["hand"]:
                    # Check if all three tags are aligned with each other
                    # This means: upper_arm-forearm, forearm-hand, and upper_arm-hand alignments exist
                    upper_forearm_aligned = (
                        upper_arm_tag in aligned_tags and 
                        forearm_tag in aligned_tags[upper_arm_tag]
                    )
                    forearm_hand_aligned = (
                        forearm_tag in aligned_tags and 
                        hand_tag in aligned_tags[forearm_tag]
                    )
                    
                    # For a valid state, we need at least upper_arm-forearm and forearm-hand
                    if upper_forearm_aligned and forearm_hand_aligned:
                        state_name = f"state_{upper_arm_tag}-{forearm_tag}-{hand_tag}"
                        if state_name not in states:
                            states.append(state_name)
        
        return states
