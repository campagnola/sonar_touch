import sys, queue
import pyqtgraph as pg
from sonar_touch.audio import BackgroundRecorder
from sonar_touch.ui import MainWindow


def excepthook(type, value, traceback):
    sys.__excepthook__(type, value, traceback)

sys.excepthook = excepthook

sys.setswitchinterval(0.001)

if __name__ == "__main__":
    app = pg.mkQApp()

    recorder = BackgroundRecorder()

    # Start the main application
    window = MainWindow(recorder.audio_queue, recorder.sample_rate, recorder.block_size)

    # pg.QtCore.QTimer.singleShot(1000, recorder.start)
    # app.exec_()
