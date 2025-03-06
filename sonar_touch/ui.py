import time
import numpy as np
import pyqtgraph as pg
import torch
import coorx

from sonar_touch.audio import RollingBuffer
from sonar_touch.project import SonarTouchProject
from sonar_touch.training import TrainingDataCollector

app = pg.mkQApp()


class MainWindow(pg.QtWidgets.QMainWindow):

    def __init__(self, audio_queue, sample_rate, block_size):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.init_ui()

        self.last_trigger_time = 0
        self.trigger_threshold = 0.01
        self.refractory_period = 0.2
        self.trigger_padding = (0.01, 0.03)
        self.full_buffer_length = 2.0
        self.buffer = RollingBuffer(int(self.full_buffer_length * self.sample_rate / self.block_size) + 1)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.handle_audio_data)
        self.timer.start(50)

        self.project = None
        self.trainer = None
        self.model = None        
        
    def init_ui(self):
        self.setWindowTitle("Sonar Touch")

        # file menu for loading a project folder
        file_menu = self.menuBar().addMenu("&File")
        self.load_action = file_menu.addAction("&Load")
        self.load_action.triggered.connect(self.load_project_triggered)

        self.train_action = file_menu.addAction("&Start Training")
        self.train_action.triggered.connect(self.start_training)

        layout = pg.QtWidgets.QGridLayout()
        self.setCentralWidget(pg.QtWidgets.QWidget())
        self.centralWidget().setLayout(layout)

        self.cw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.cw, 0, 0)
        self.plot = self.cw.addPlot(row=0, col=0)
        self.plot.setYRange(-1, 1)

        self.trigger_plot = self.cw.addPlot(row=1, col=0)
        self.trigger_plot.setYRange(-.2, .2)

        self.resize(1200, 600)
        self.show()

        # second window for projection
        self.projected_view = ProjectedView()
        self.projected_view.projection_roi.sigRegionChangeFinished.connect(self.projection_roi_changed)

        quit_shortcut = pg.QtWidgets.QShortcut(pg.QtGui.QKeySequence("Ctrl+Q"), self)
        quit_shortcut.setContext(pg.QtCore.Qt.ApplicationShortcut)
        quit_shortcut.activated.connect(self.close)

    def handle_audio_data(self):
        if self.audio_queue.qsize() == 0:
            return

        # read all available data from the queue into rolling buffer
        self.buffer.add_from_queue(self.audio_queue)
        sample_index, data = self.buffer.get_data()

        # plot all data in the buffer
        self.plot.clear()
        t = np.arange(data.shape[1]) / self.sample_rate
        for i,chan in enumerate(data):
            self.plot.plot(t, chan, pen=(i, 4))
        self.plot.addLine(y=self.trigger_threshold, pen='w')

        now = time.perf_counter()
        if now - self.last_trigger_time < 0.5:
            return

        # look for a trigger
        trigger_result = self.buffer.get_trigger(
            self.trigger_threshold, 
            pre_padding=self.trigger_padding[0] * self.sample_rate, 
            post_padding=self.trigger_padding[1] * self.sample_rate,
            refractory_period=self.refractory_period * self.sample_rate,
        )
        if trigger_result is None:
            return
        
        plot_data = trigger_result['data']
        trigger_index = trigger_result['index']
        self.last_trigger_time = now

        self.trigger_plot.clear()
        t = (np.arange(plot_data.shape[1]) - trigger_index) / self.sample_rate
        for i,chan in enumerate(plot_data):
            self.trigger_plot.plot(t, chan, pen=(i, 4))
        self.trigger_plot.addLine(y=self.trigger_threshold, pen='w')
        self.trigger_plot.addLine(x=0, pen='w')

        if self.trainer is not None and self.trainer.run:
            self.trainer.trigger_detected(trigger_result)
        elif self.model is not None:
            self.predict(trigger_result)
            
    def close(self):
        self.timer.stop()
        self.projected_view.close()
        return super().close()

    def load_project_triggered(self):
        folder = pg.QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder == "":
            self.project = None
            return
        self.load_project(folder)
        
    def load_project(self, folder):
        self.project = SonarTouchProject(folder)
        if 'projection_roi_state' in self.project.state:
            self.projected_view.projection_roi.setState(self.project.state['projection_roi_state'])
        models = self.project.list_models()
        if len(models) > 0:
            self.model = self.project.load_model(models[-1])

    def projection_roi_changed(self):
        if self.project is not None:
            self.project.save(
                projection_roi_state=self.projected_view.projection_roi.saveState()
            )

    def start_training(self):
        if self.project is None:
            raise ValueError("No project loaded")
        if self.trainer is None:
            self.trainer = TrainingDataCollector(self)
            self.trainer.start()
            self.train_action.setText("&Stop Training")
        else:
            self.trainer.stop()
            self.train_action.setText("&Start Training")

    def predict(self, trigger):
        if self.model is None:
            return
        data = trigger['data']
        tensor = torch.tensor(data.reshape(1, 4, -1), dtype=torch.float32).to(self.model.device)
        location = self.model(tensor).detach().cpu().numpy()[0]
        self.projected_view.set_target(location)


class ProjectionROI(pg.PolyLineROI):
    def __init__(self):
        pos = [[0, 0], [1920, 0], [1920, 1280], [0, 1280]]
        pg.PolyLineROI.__init__(self, pos, closed=True)

    def transform(self):
        pts = self.saveState()['points']
        tr = coorx.linear.Homography2DTransform()
        tr.set_mapping(pts, [[0, 0], [1, 0], [1, 1], [0, 1]])
        return tr

    def setState(self, state):
        self.blockSignals(True)
        try:
            super().setState(state)
        finally:
            self.blockSignals(False)
        self.sigRegionChanged.emit(self)
        self.sigRegionChangeFinished.emit(self)


class ProjectedView(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.view = self.addViewBox()
        self.view.setRange(xRange=[0, 1920], yRange=[0, 1080], padding=0)
        self.grid = pg.PlotCurveItem(pen=0.5)
        self.view.addItem(self.grid)
        self.view.setMouseEnabled(False, False)

        self.projection_roi = ProjectionROI()
        self.projection_roi.sigRegionChanged.connect(self.projection_roi_changed)
        self.view.addItem(self.projection_roi)
        self.view.setAspectLocked(True)

        self.target = pg.TargetItem()
        self.view.addItem(self.target)
        self.target.setVisible(False)
        self.target_pos = [0, 0]

        self.view.scene().sigMouseClicked.connect(self.mouse_clicked)

        # move projected view to second monitor if available
        screens = pg.QtWidgets.QApplication.screens()
        if len(screens) > 1:
            for screen in screens:
                if screen != pg.QtWidgets.QApplication.primaryScreen():
                    break
            self.show()  # show the window to get a window handle
            self.windowHandle().setScreen(screen)
            self.setGeometry(screen.geometry())
            self.showFullScreen()


    def projection_roi_changed(self):
        tr = self.projection_roi.transform()
        pts = np.empty((20, 2), dtype=float)
        n_lines = 5
        for i,x in enumerate(np.linspace(0, 1, n_lines)):
            pts[i*2] = [0, x]
            pts[i*2 + 1] = [1, x]
            pts[(i + n_lines) * 2] = [x, 0]
            pts[(i + n_lines) * 2 + 1] = [x, 1]
        mapped = tr.imap(pts)
        self.grid.setData(mapped[:, 0], mapped[:, 1], connect='pairs')
        self.update_target()

    def update_target(self):
        tr = self.projection_roi.transform()
        self.target.setPos(*tr.imap(self.target_pos))

    def mouse_clicked(self, ev):
        # self.mouse_clicked
        pass

    def set_target(self, target):
        self.target_pos = target
        self.target.setVisible(True)
        self.update_target()
