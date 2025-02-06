import sys, argparse
import pyqtgraph as pg
from sonar_touch.audio import BackgroundRecorder
from sonar_touch.sim import SimulatedRecorder
from sonar_touch.ui import MainWindow


def excepthook(type, value, traceback):
    sys.__excepthook__(type, value, traceback)

sys.excepthook = excepthook

sys.setswitchinterval(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # flag to simulate audio rther than record
    parser.add_argument("--sim", action="store_true", default=False, help="Simulate audio input")
    args = parser.parse_args()


    app = pg.mkQApp()

    if args.sim:
        recorder = SimulatedRecorder()
    else:
        recorder = BackgroundRecorder()

    # Start the main application
    window = MainWindow(recorder.audio_queue, recorder.sample_rate, recorder.block_size)

    if sys.flags.interactive == 0:
        app.exec_()
    # pg.QtCore.QTimer.singleShot(1000, recorder.start)
    # app.exec_()
