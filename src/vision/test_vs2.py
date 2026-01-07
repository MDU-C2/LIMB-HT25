#!/usr/bin/env python3
"""
Simple test script for VisionSystem2 (now VisionSystem) from system2.py

This script demonstrates how to use the vision system to detect cups
and calculate robot arm angles.
"""

import time
import sys
from system2 import VisionSystem

def main():
    """Run a simple test of the vision system."""
    print("="*60)
    print("Vision System Test")
    print("="*60)
    
    # Initialize vision system
    print("\nInitializing vision system...")
    vision = VisionSystem(
        blob_path="cup.blob",
        label_name="Cup",
        confidence_threshold=0.6,
        enable_visualization=True
    )
    
    # Start pipeline
    print("Starting pipeline...")
    if not vision.start_pipeline():
        print("Failed to start pipeline!")
        return 1
    
    print("\nVision system running!")
    print("Looking for cups...")
    print("Press Ctrl+C to stop\n")
    
    try:
        frame_count = 0
        while True:
            # Update vision system (polls queues and processes data)
            vision.update()
            
            # Get latest detection
            detection = vision.get_latest_detection()
            angles = vision.get_latest_angles()
            
            # Print results every 30 frames (~1 second at 30fps)
            if frame_count % 30 == 0:
                if detection:
                    print(f"\n🎯 Cup detected!")
                    print(f"   Position (mm): X={detection['x_mm']}, Y={detection['y_mm']}, Z={detection['z_mm']}")
                    print(f"   Confidence: {detection['confidence']:.2%}")
                    
                    if angles:
                        print(f"   Robot Position (m): X={angles['x_m']:.3f}, Y={angles['y_m']:.3f}, Z={angles['z_m']:.3f}")
                        print(f"   Angles: Shoulder Z={angles['sh_z']:.1f}°, Shoulder Y={angles['sh_y']:.1f}°, Elbow={angles['elb']:.1f}°")
                else:
                    print("   No cup detected...")
            
            frame_count += 1
            time.sleep(0.01)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nShutting down vision system...")
        vision.shutdown()
        print("Test completed!")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

