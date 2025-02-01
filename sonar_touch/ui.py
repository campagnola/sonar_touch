import numpy as np
import pyqtgraph as pg

app = pg.mkQApp()


class MainWindow(pg.QtWidgets.QWidget):
    def __init__(self, audio_queue, sample_rate, block_size):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.init_ui()

        self.blocks = []

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.plot_data)
        self.timer.start(50)
        
    def init_ui(self):
        self.setWindowTitle("Sonar Touch")
        layout = pg.QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(-1, 1)
        layout.addWidget(self.plot)

        self.trigger_plot = pg.PlotWidget()
        self.trigger_plot.setYRange(-.2, .2)
        layout.addWidget(self.trigger_plot)

        self.resize(800, 600)
        self.show()

    def plot_data(self):
        if self.audio_queue.qsize() == 0:
            return
        while self.audio_queue.qsize() > 0:
            self.blocks.append(self.audio_queue.get())

        search_blocks = 4
        pad_blocks = 2
        pad_samples = pad_blocks * self.block_size
        n_blocks = search_blocks + 2 * pad_blocks
        if len(self.blocks) > n_blocks:
            discard = len(self.blocks) - n_blocks
            # if discard > 1:
            #     print(f"Discarding {discard} blocks")
            self.blocks = self.blocks[discard:]

        data = np.concatenate(self.blocks, axis=1)

        # plot all data in the buffer
        trigger_threshold = 0.02
        self.plot.clear()
        for i,chan in enumerate(data):
            self.plot.plot(chan, pen=(i, 4))
        self.plot.addLine(y=trigger_threshold, pen='w')
        # self.plot.addLine(x=pad_samples, pen='w')
        # self.plot.addLine(x=data.shape[1] - pad_samples, pen='w')

        # search a small subset for trigggers
        search_data = data[:, pad_samples:-pad_samples]
        mask = search_data > trigger_threshold
        mask_change = mask[:, 1:] & ~mask[:, :-1]
        triggers = [np.argwhere(mask_change[i])[:, 0] for i in range(mask_change.shape[0])]
        triggers = [t[0] for t in triggers if len(t) > 0]
        if len(triggers) == 0:
            return
        trigger = min(triggers)

        trigger_index = trigger + pad_samples
        # print(trigger_index, data[:, trigger_index - 1:trigger_index + 1])
        plot_duration = 0.01
        pad_samples = int(plot_duration * self.sample_rate)
        plot_data = data[:, trigger_index - pad_samples:trigger_index + pad_samples]

        self.trigger_plot.clear()
        for i,chan in enumerate(plot_data):
            self.trigger_plot.plot(chan, pen=(i, 4))
        self.trigger_plot.addLine(y=trigger_threshold, pen='w')
        self.trigger_plot.addLine(x=pad_samples, pen='w')
