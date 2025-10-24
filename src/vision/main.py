#!/usr/bin/env python3
"""
Main entry point for OAK-D Vision System with SpatialVisualizer.

This version uses pipeline.run() with SpatialVisualizer HostNode for proper visualization.
"""

import cv2
from system import VisionSystem
import time

def main():
    """Run the vision system with SpatialVisualizer."""
    print("="*60)
    print("OAK-D Vision System with SpatialVisualizer")
    print("="*60)
    print("\nInitializing...")
    
    # Create vision system with visualization enabled
    vision_system = VisionSystem(
        confidence_threshold=0.5,
        spatial_threshold=5000,  # 5 meters
        tag_size=0.05,  # 5 cm AprilTags
        enable_visualization=True  # Enable SpatialVisualizer
    )
    
    print("Starting pipeline with SpatialVisualizer...")
    
    # Create pipeline and start it directly with run() for SpatialVisualizer
    try:
        
        vision_system.run_pipeline()
        
        print("\nVision system running with SpatialVisualizer!")
        print("You should see three windows:")
        print("  - RGB: Color video with cup detections")
        print("  - Depth: Color-coded depth map")
        print("  - AprilTags: Grayscale with 3D coordinate axes")
        print("\nPress 'q' in any window to quit.\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        vision_system.shutdown()
        print("Vision system stopped.")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    main()
