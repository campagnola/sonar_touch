import pyqtgraph as pg

app = pg.mkQApp()


class MainWindow(pg.QtWidgets.QWidget):
    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.init_ui()

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
        self.show()
        self.resize(800, 600)

    def plot_data(self):
        if self.audio_queue.qsize() == 0:
            return
        while self.audio_queue.qsize() > 0:
            data = self.audio_queue.get()

        self.plot.clear()
        for i,chan in enumerate(data):
            self.plot.plot(chan, pen=(i, 4))
