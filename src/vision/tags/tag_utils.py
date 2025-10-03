"""
Utility functions for working with TagDetectionResult and its new fields.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .tag_detector import TagDetectionResult

def get_tag_transform_by_id(tag_result: TagDetectionResult, tag_id: int) -> Optional[np.ndarray]:
    """
    Get the 4x4 transform matrix for a specific tag ID.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        tag_id: ID of the tag to find
        
    Returns:
        4x4 transform matrix if found, None otherwise
    """
    if tag_result.transforms is None or tag_result.tag_ids is None:
        return None
    
    try:
        tag_index = np.where(tag_result.tag_ids == tag_id)[0][0]
        return tag_result.transforms[tag_index]
    except (IndexError, ValueError):
        return None

def get_tag_reprojection_error_by_id(tag_result: TagDetectionResult, tag_id: int) -> Optional[float]:
    """
    Get the reprojection error for a specific tag ID.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        tag_id: ID of the tag to find
        
    Returns:
        Reprojection error if found, None otherwise
    """
    if tag_result.reproj_errors is None or tag_result.tag_ids is None:
        return None
    
    try:
        tag_index = np.where(tag_result.tag_ids == tag_id)[0][0]
        return float(tag_result.reproj_errors[tag_index])
    except (IndexError, ValueError):
        return None

def get_tag_timestamp_by_id(tag_result: TagDetectionResult, tag_id: int) -> Optional[float]:
    """
    Get the timestamp for a specific tag ID.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        tag_id: ID of the tag to find
        
    Returns:
        Timestamp if found, None otherwise
    """
    if tag_result.timestamps is None or tag_result.tag_ids is None:
        return None
    
    try:
        tag_index = np.where(tag_result.tag_ids == tag_id)[0][0]
        return tag_result.timestamps[tag_index]
    except (IndexError, ValueError):
        return None

def filter_tags_by_reprojection_error(tag_result: TagDetectionResult, 
                                    max_error: float = 1.0) -> TagDetectionResult:
    """
    Filter tags based on reprojection error threshold.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        max_error: Maximum allowed reprojection error
        
    Returns:
        New TagDetectionResult with only tags below the error threshold
    """
    if tag_result.reproj_errors is None:
        return tag_result
    
    # Find indices of tags with acceptable reprojection error
    good_indices = np.where(tag_result.reproj_errors <= max_error)[0]
    
    if len(good_indices) == 0:
        # Return empty result
        return TagDetectionResult(
            tag_ids=np.array([]),
            rvecs=None,
            tvecs=None,
            corners=None,
            transforms=None,
            reproj_errors=None,
            timestamps=None
        )
    
    # Filter all arrays
    filtered_tag_ids = tag_result.tag_ids[good_indices]
    
    filtered_rvecs = None
    if tag_result.rvecs is not None:
        filtered_rvecs = tag_result.rvecs[good_indices]
    
    filtered_tvecs = None
    if tag_result.tvecs is not None:
        filtered_tvecs = tag_result.tvecs[good_indices]
    
    filtered_corners = None
    if tag_result.corners is not None:
        filtered_corners = [tag_result.corners[i] for i in good_indices]
    
    filtered_transforms = None
    if tag_result.transforms is not None:
        filtered_transforms = [tag_result.transforms[i] for i in good_indices]
    
    filtered_reproj_errors = tag_result.reproj_errors[good_indices]
    
    filtered_timestamps = None
    if tag_result.timestamps is not None:
        filtered_timestamps = [tag_result.timestamps[i] for i in good_indices]
    
    return TagDetectionResult(
        tag_ids=filtered_tag_ids,
        rvecs=filtered_rvecs,
        tvecs=filtered_tvecs,
        corners=filtered_corners,
        transforms=filtered_transforms,
        reproj_errors=filtered_reproj_errors,
        timestamps=filtered_timestamps
    )

def get_tag_statistics(tag_result: TagDetectionResult) -> Dict[str, Any]:
    """
    Get statistics about the tag detection results.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_tags": len(tag_result.tag_ids) if tag_result.tag_ids is not None else 0,
        "tag_ids": tag_result.tag_ids.tolist() if tag_result.tag_ids is not None else [],
        "has_transforms": tag_result.transforms is not None,
        "has_reproj_errors": tag_result.reproj_errors is not None,
        "has_timestamps": tag_result.timestamps is not None
    }
    
    if tag_result.reproj_errors is not None and len(tag_result.reproj_errors) > 0:
        stats.update({
            "mean_reproj_error": float(np.mean(tag_result.reproj_errors)),
            "std_reproj_error": float(np.std(tag_result.reproj_errors)),
            "min_reproj_error": float(np.min(tag_result.reproj_errors)),
            "max_reproj_error": float(np.max(tag_result.reproj_errors))
        })
    
    if tag_result.timestamps is not None and len(tag_result.timestamps) > 0:
        stats.update({
            "timestamp_range": max(tag_result.timestamps) - min(tag_result.timestamps),
            "oldest_timestamp": min(tag_result.timestamps),
            "newest_timestamp": max(tag_result.timestamps)
        })
    
    return stats

def compare_tag_detections(result1: TagDetectionResult, result2: TagDetectionResult) -> Dict[str, Any]:
    """
    Compare two tag detection results.
    
    Args:
        result1: First TagDetectionResult
        result2: Second TagDetectionResult
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "same_tag_count": len(result1.tag_ids) == len(result2.tag_ids),
        "same_tag_ids": np.array_equal(result1.tag_ids, result2.tag_ids) if both_have_ids(result1, result2) else False,
        "timestamp_difference": None
    }
    
    if result1.timestamps is not None and result2.timestamps is not None:
        if len(result1.timestamps) > 0 and len(result2.timestamps) > 0:
            comparison["timestamp_difference"] = abs(min(result1.timestamps) - min(result2.timestamps))
    
    return comparison

def both_have_ids(result1: TagDetectionResult, result2: TagDetectionResult) -> bool:
    """Helper function to check if both results have tag IDs."""
    return (result1.tag_ids is not None and result2.tag_ids is not None)

def extract_transform_matrices(tag_result: TagDetectionResult) -> Dict[int, np.ndarray]:
    """
    Extract transform matrices as a dictionary mapping tag_id -> transform_matrix.
    
    Args:
        tag_result: TagDetectionResult containing tag data
        
    Returns:
        Dictionary mapping tag IDs to their 4x4 transform matrices
    """
    if tag_result.transforms is None or tag_result.tag_ids is None:
        return {}
    
    return {int(tag_id): transform for tag_id, transform in zip(tag_result.tag_ids, tag_result.transforms)}
