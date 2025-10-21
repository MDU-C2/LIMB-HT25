#!/usr/bin/env python3
"""
Main entry point for OAK-D Vision System with SpatialVisualizer.
"""

import cv2
from system import VisionSystem
import time

def main():
    """Run the vision system with live visualization."""
    print("="*60)
    print("OAK-D Vision System with SpatialVisualizer")
    print("="*60)
    print("\nInitializing...")
    
    # Create and start vision system
    vision_system = VisionSystem(
        confidence_threshold=0.5,
        spatial_threshold=5000  # 5 meters
    )
    
    print("Starting pipeline...")
    if not vision_system.start_pipeline():
        print("Failed to start pipeline!")
        return 1
    
    print("\nVision system running!")
    print("Two windows will appear: 'RGB' and 'Depth'")
    print("Press 'q' in either window to quit.\n")
    

    vision_system.start_pipeline()
    # Main loop - visualization happens in the SpatialVisualizer node
    try:
        while vision_system.is_pipeline_running():
            time.sleep(1)
            #key = cv2.waitKey(1)
            #if key == ord('q'):
            #    print("\nStopping pipeline...")
            #    break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        #print(f"is_pipeline_running: {vision_system.is_pipeline_running()}")
        #vision_system.shutdown()
        cv2.destroyAllWindows()
        print("Vision system stopped.")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    main()
