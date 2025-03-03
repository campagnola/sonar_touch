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
    # project flag requires a filename argument
    parser.add_argument("--project", type=str, help="Load a project file")
    parser.add_argument("--train", action="store_true", default=False, help="Enter training mode on startup")
    args = parser.parse_args()


    app = pg.mkQApp()

    if args.sim:
        recorder = SimulatedRecorder()
    else:
        recorder = BackgroundRecorder()

    # Start the main application
    window = MainWindow(recorder.audio_queue, recorder.sample_rate, recorder.block_size)

    if args.project:
        window.load_project(args.project)

    if args.train:
        window.start_training()

    if sys.flags.interactive == 0:
        app.exec_()
    # pg.QtCore.QTimer.singleShot(1000, recorder.start)
    # app.exec_()
