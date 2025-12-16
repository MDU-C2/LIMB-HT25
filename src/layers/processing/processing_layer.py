from multiprocessing import Process, Event
import time
from shared.queues import DataQueue

class ProcessingLayer(Process):
    """Processing layer: ML inference, signal processing."""

    def __init__(self, input_queue: DataQueue, output_queue: DataQueue):
        super().__init__(name="ProcessingLayer")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = Event()

        # Maybe some feature extraction..?

    def run(self):
        
        self.running.set()

        while self.running.is_set():

            try:
                packet = self.input_queue.get(timeout=0.001) # Why timeout?

                # Check if packet is too old
                if packet.is_stale():  # TODO: Implement this function
                    continue

                
                processed = self.process_packet(packet)

                if not self.output_queue.full():
                    self.output_queue.put(processed, block=False) # Why non-blocking?


            except:
                time.sleep(0.001) # Queue is empty, continue

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop

    def process_packet(self, packet):
        """Run ML inference and signal processing"""
        features = 0 # TODO: Implement this function (self.feature_extractor.extract(packet.human_packet)) This does the signal processing
        
        prediction = self.ml_model.predict(features)

        packet.metadata["ml_prediction"] = prediction
        packet.metadata["processed"] = True
        return packet