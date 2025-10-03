import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class PerformanceMetric:
    """Individual performance metric with timing and statistics."""
    name: str
    times: deque = field(default_factory=lambda: deque(maxlen=100))  # Keep last 100 measurements
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def add_measurement(self, elapsed_time: float):
        """Add a new timing measurement."""
        self.times.append(elapsed_time)
        self.total_calls += 1
        self.total_time += elapsed_time
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.times:
            return {
                'avg_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'std_time_ms': 0.0,
                'total_calls': 0,
                'total_time_ms': 0.0,
                'calls_per_second': 0.0
            }
        
        times_array = np.array(self.times)
        avg_time = np.mean(times_array)
        std_time = np.std(times_array)
        
        return {
            'avg_time_ms': avg_time * 1000,  # Convert to milliseconds
            'min_time_ms': self.min_time * 1000,
            'max_time_ms': self.max_time * 1000,
            'std_time_ms': std_time * 1000,
            'total_calls': self.total_calls,
            'total_time_ms': self.total_time * 1000,
            'calls_per_second': self.total_calls / max(self.total_time, 0.001)
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring for vision system components."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: How often to update and print stats (seconds)
        """
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # System-wide metrics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)  # Keep last 30 FPS measurements
        self.last_frame_time = self.start_time
        
        # Threading for background monitoring
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitoring_worker(self):
        """Background worker for continuous monitoring."""
        while self.running:
            time.sleep(self.update_interval)
            self._update_system_stats()
            self._print_performance_summary()
    
    def _update_system_stats(self):
        """Update system-wide performance statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed > 0 and self.frame_count > 0:
            current_fps = self.frame_count / elapsed
            self.fps_history.append(current_fps)
            self.frame_count = 0
        
        self.last_update_time = current_time
    
    def _print_performance_summary(self):
        """Print a summary of current performance."""
        print("\n" + "="*60)
        print("PERFORMANCE MONITOR SUMMARY")
        print("="*60)
        
        # System FPS
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"System FPS: {avg_fps:.1f} (avg over last {len(self.fps_history)} measurements)")
        
        # Individual metrics
        for name, metric in self.metrics.items():
            stats = metric.get_stats()
            print(f"\n{name}:")
            print(f"  Avg: {stats['avg_time_ms']:.2f}ms")
            print(f"  Min: {stats['min_time_ms']:.2f}ms")
            print(f"  Max: {stats['max_time_ms']:.2f}ms")
            print(f"  Std: {stats['std_time_ms']:.2f}ms")
            print(f"  Calls/sec: {stats['calls_per_second']:.1f}")
        
        print("="*60)
    
    def time_function(self, name: str):
        """Decorator for timing function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                with self.lock:
                    if name not in self.metrics:
                        self.metrics[name] = PerformanceMetric(name)
                    self.metrics[name].add_measurement(elapsed_time)
                
                return result
            return wrapper
        return decorator
    
    def start_timer(self, name: str):
        """Start timing a named operation."""
        if name not in self.metrics:
            with self.lock:
                self.metrics[name] = PerformanceMetric(name)
        return TimerContext(self.metrics[name])
    
    def record_frame(self):
        """Record that a frame was processed (for FPS calculation)."""
        self.frame_count += 1
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific metric."""
        if name in self.metrics:
            return self.metrics[name].get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: metric.get_stats() for name, metric in self.metrics.items()}
    
    def get_system_fps(self) -> float:
        """Get current system FPS."""
        if self.fps_history:
            return float(np.mean(self.fps_history))
        return 0.0
    
    def reset_stats(self):
        """Reset all performance statistics."""
        with self.lock:
            self.metrics.clear()
            self.frame_count = 0
            self.fps_history.clear()
            self.start_time = time.time()
            self.last_update_time = self.start_time


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, metric: PerformanceMetric):
        self.metric = metric
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.metric.add_measurement(elapsed_time)


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def time_function(name: str):
    """Convenience decorator using global monitor."""
    return get_performance_monitor().time_function(name)

def start_timer(name: str):
    """Convenience function to start timing using global monitor."""
    return get_performance_monitor().start_timer(name)
