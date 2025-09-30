#!/usr/bin/env python3
"""Test script to verify both running methods work."""

import subprocess
import sys
import os

def test_module_method():
    """Test running as module from src folder."""
    print("Testing module method (python -m imu_vision.quick_start)...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "imu_vision.quick_start"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Module method works!")
            return True
        else:
            print(f"❌ Module method failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Module method error: {e}")
        return False

def test_direct_method():
    """Test running directly from src folder."""
    print("Testing direct method (python imu_vision/quick_start.py)...")
    try:
        result = subprocess.run([
            sys.executable, "imu_vision/quick_start.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Direct method works!")
            return True
        else:
            print(f"❌ Direct method failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Direct method error: {e}")
        return False

def main():
    """Test both methods."""
    print("=" * 60)
    print("Testing Both Running Methods")
    print("=" * 60)
    
    # Change to src directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    module_works = test_module_method()
    direct_works = test_direct_method()
    
    print("\n" + "=" * 60)
    if module_works and direct_works:
        print("✅ Both methods work correctly!")
        print("You can use either:")
        print("  python -m imu_vision.quick_start")
        print("  python imu_vision/quick_start.py")
    elif module_works:
        print("✅ Module method works!")
        print("Use: python -m imu_vision.quick_start")
    elif direct_works:
        print("✅ Direct method works!")
        print("Use: python imu_vision/quick_start.py")
    else:
        print("❌ Both methods failed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
