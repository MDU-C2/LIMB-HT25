import numpy as np

class WindowBuffer:

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffer = [] # Some buffer?

    def is_full(self) -> bool:
        return len(self.buffer) >= self.window_size

    def get_window(self):
        """Get complete window as nparray"""
        pass