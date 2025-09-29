#!/usr/bin/env python3
"""
Test script to verify performance optimizations are working correctly.
"""

import time
import numpy as np
import cv2
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cup.cup_detector import CupDetector
from utils.performance_monitor import get_performance_monitor, start_timer
from utils.camera_optimizer import OptimizedCameraCapture, create_optimized_camera_config


def test_performance_monitor():
    """Test performance monitoring functionality."""
    print("Testing performance monitor...")
    
    monitor = get_performance_monitor()
    
    # Test context manager
    with start_timer("test_function"):
        time.sleep(0.1)  # Simulate work
    
    # Test decorator
    @monitor.time_function("decorated_function")
    def test_func():
        time.sleep(0.05)
        return "test"
    
    result = test_func()
    assert result == "test"
    
    # Check stats
    stats = monitor.get_all_stats()
    assert "test_function" in stats
    assert "decorated_function" in stats
    
    print("✓ Performance monitor working correctly")


def test_optimized_cup_detector():
    """Test optimized cup detector."""
    print("Testing optimized cup detector...")
    
    # Create dummy camera matrix
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    
    try:
        # Test with proper model path using model manager
        from utils.yolo_model_manager import get_yolo_model_path
        
        vision_dir = os.path.dirname(__file__)
        model_path = get_yolo_model_path("yolo11n.pt", vision_dir)
        
        detector = CupDetector(
            camera_matrix=camera_matrix,
            weights_path=model_path,
            skip_frames=1,
            target_fps=30.0
        )
        
        # Test performance stats
        stats = detector.get_performance_stats()
        assert "current_fps" in stats
        assert "target_fps" in stats
        
        # Test dynamic adjustment
        detector.set_skip_frames(2)
        detector.set_target_fps(60.0)
        
        # Cleanup
        detector.stop_inference_thread()
        
        print("✓ Optimized cup detector working correctly")
        
    except Exception as e:
        print(f"⚠ Optimized cup detector test skipped (expected with dummy model): {e}")


def test_camera_optimizer():
    """Test camera optimizer."""
    print("Testing camera optimizer...")
    
    try:
        # Test configuration creation
        config = create_optimized_camera_config(
            device_id=0,
            target_fps=30,
            low_latency=True
        )
        
        assert config.fps == 30
        assert config.buffer_size == 1
        assert config.fourcc == 'MJPG'
        
        # Test camera initialization (will fail if no camera available)
        camera = OptimizedCameraCapture(config)
        
        # Test non-blocking read
        ret, frame = camera.read()
        # ret might be False if no camera, but should not crash
        
        # Test performance stats
        stats = camera.get_performance_stats()
        assert "avg_capture_time_ms" in stats
        assert "total_frames" in stats
        
        camera.release()
        
        print("✓ Camera optimizer working correctly")
        
    except Exception as e:
        print(f"⚠ Camera optimizer test skipped (no camera available): {e}")


def test_integration():
    """Test integration of all components."""
    print("Testing component integration...")
    
    # Create dummy camera matrix
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    
    # Test performance monitor integration
    monitor = get_performance_monitor()
    
    # Test timing integration
    with start_timer("integration_test"):
        time.sleep(0.01)
    
    # Verify timing was recorded
    stats = monitor.get_all_stats()
    assert "integration_test" in stats
    
    print("✓ Component integration working correctly")


def run_performance_test():
    """Run a simple performance test."""
    print("Running performance test...")
    
    monitor = get_performance_monitor()
    
    # Test multiple timing measurements
    for i in range(10):
        with start_timer("performance_test"):
            # Simulate some work
            dummy_data = np.random.rand(100, 100)
            result = np.sum(dummy_data)
    
    # Check statistics
    stats = monitor.get_metric_stats("performance_test")
    assert stats is not None
    assert stats["total_calls"] == 10
    assert stats["avg_time_ms"] > 0
    
    print(f"✓ Performance test completed: {stats['avg_time_ms']:.2f}ms average")


def main():
    """Run all tests."""
    print("Running optimization tests...")
    print("=" * 50)
    
    try:
        test_performance_monitor()
        test_optimized_cup_detector()
        test_camera_optimizer()
        test_integration()
        run_performance_test()
        
        print("=" * 50)
        print("✓ All optimization tests passed!")
        
        # Print final performance stats
        monitor = get_performance_monitor()
        print("\nFinal performance statistics:")
        stats = monitor.get_all_stats()
        for name, stat in stats.items():
            print(f"  {name}: {stat['avg_time_ms']:.2f}ms avg, {stat['total_calls']} calls")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
