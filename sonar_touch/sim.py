import queue, threading
import numpy as np
import time


class SimulatedRecorder:
    def __init__(self):
        self.n_channels = 4
        self.sample_rate = 44100
        self.block_size = 4096
        self.audio_queue = queue.Queue()

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        t_step = self.block_size / self.sample_rate
        while True:
            data = np.random.normal(size=(self.n_channels, self.block_size), scale=0.0021)
            self.audio_queue.put(data)
            time.sleep(t_step)
