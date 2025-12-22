from multiprocessing import Queue
from typing import Optional
import time

class DataQueue:
    """
    Queue for data packets between layers.
    Drops the oldest packet if the queue is full.
    """

    def __init__(self, max_size: int = 5):
        self._queue = Queue(maxsize=max_size)
        self.max_size = max_size
        self.dropped_count = 0 # Count of packets dropped due to full queue

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """Put item, dropping oldest if full."""
        if self.full():
            try:
                self.get_nowait()
                self.dropped_count += 1
            except:
                pass
        
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None):
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self):
        return self._queue.get_nowait()

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def size(self):
        return self._queue.qsize()
