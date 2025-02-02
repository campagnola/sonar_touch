import threading
import jack
import numpy as np
import queue
import time
import teleprox


class JackAudioRecorder:
    def __init__(self, n_channels=4, client_name="Sonar"):
        self.n_channels = n_channels
        self.client = jack.Client(client_name)
        self.audio_queue = queue.Queue()
 
        # Register input ports
        self.client.inports.clear()
        for i in range(n_channels):
            self.client.inports.register(f"input_{i+1}")

        # Set JACK process callback
        self.client.set_process_callback(self._process)

    def start(self):
        self.recording = True
        self.client.activate()
        self.connect(['system:capture_1', 'system:capture_2', 'system:capture_3', 'system:capture_4'])

    def connect(self, ports):
        """Connect the input ports to the specified JACK ports."""
        for i, port in enumerate(ports):
            self.client.inports[i].connect(port)

    def list_ports(self):
        """List all available JACK ports."""
        ports = self.client.get_ports()
        for i, port in enumerate(ports):
            print(f"{i}: {port}")

    def _process(self, frames):
        """JACK process callback - receives and queues audio data."""
        if not self.recording:
            return
        data = [port.get_array() for port in self.client.inports]
        self.audio_queue.put(data)
        now = time.time()
 
    def stop(self):
        self.recording = False
        self.client.deactivate()


class AudioRingBuffer:
    """Maintins a ring buffer.

    In a background thread, pull audio data from a queue and copy into the ring buffer.

    For every chunk written to the ring buffer, invoke *callback* with the first written buffer index.
    """
    def __init__(self, buffer, queue, callback):
        self.buffer = buffer
        self.queue = queue
        self.callback = callback
        self.buffer_index = 0

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while True:
            data = self.queue.get()
            start_index = self.buffer_index % self.buffer.shape[1]
            block_size = data[0].shape[0]
            stop_index = start_index + block_size
            assert self.buffer.shape[0] == len(data)
            assert self.buffer.shape[1] % block_size == 0
            for i, chan in enumerate(data):
                self.buffer[i, start_index:start_index + block_size] = chan
            self.callback(self.buffer_index, start_index, block_size, _sync='off')
            self.buffer_index += block_size


class BackgroundRecorder:
    """Records audio from JACK in a background process.

    This ensures that jack has a high priority so we don't block the jack server.
    (If we do, jack will disconnect us.)

    Data is sent via shared memory into self.audio_queue in the main process.
    """
    def __init__(self):
        # start background process for recording from jack
        self.proc = teleprox.start_process("sonar_touch_record")
        proc_audio = self.proc.client._import('sonar_touch.audio')
        
        # Create the JACK audio recorder in child process
        self.recorder = proc_audio.JackAudioRecorder()

        self.sample_rate = self.recorder.client.samplerate._get_value()
        self.block_size = self.recorder.client.blocksize._get_value()
        self.n_channels = self.recorder.n_channels._get_value()
        assert self.n_channels == 4
        assert self.block_size > 0
        self.n_blocks = 10  # number of blocks in ring buffer

        # create shared memory
        self.shmem = teleprox.shmem.SharedNDArray.zeros((self.n_channels, 2 * self.block_size * self.n_blocks), dtype='float32')
        rshmem = self.proc.client.transfer(self.shmem)

        # create ring buffer and rpc server to receive ring callbacks
        callback_server = teleprox.RPCServer()
        callback_server.run_in_thread()
        self.audio_queue = queue.Queue()
        def new_audio_event(sample_id, start_index, block_size):
            block = self.shmem.data[:, start_index:start_index + block_size].copy()
            self.audio_queue.put(block)
        callback = callback_server.get_proxy(new_audio_event)
        self.ring_buffer = proc_audio.AudioRingBuffer(rshmem.data, self.recorder.audio_queue, callback)

        # start recorder
        self.recorder.start()
