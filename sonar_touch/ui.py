import time
import numpy as np
import pyqtgraph as pg
import coorx

from sonar_touch.project import SonarTouchProject

app = pg.mkQApp()


class MainWindow(pg.QtWidgets.QMainWindow):
    def __init__(self, audio_queue, sample_rate, block_size):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.init_ui()

        self.search_blocks = 4
        self.pad_blocks = 2
        self.buffer = RollingBuffer(self.search_blocks + 2 * self.pad_blocks)
        self.last_trigger_time = 0
        self.trigger_threshold = 0.01

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.handle_audio_data)
        self.timer.start(50)

        self.project = None
        
    def init_ui(self):
        self.setWindowTitle("Sonar Touch")

        # file menu for loading a project folder
        file_menu = self.menuBar().addMenu("&File")
        load_action = file_menu.addAction("&Load")
        load_action.triggered.connect(self.load_project_triggered)

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
        self.projected_view.show()
        self.projected_view.projection_roi.sigRegionChangeFinished.connect(self.projection_roi_changed)

        quit_shortcut = pg.QtWidgets.QShortcut(pg.QtGui.QKeySequence("Ctrl+Q"), self)
        quit_shortcut.setContext(pg.QtCore.Qt.ApplicationShortcut)
        quit_shortcut.activated.connect(self.close)

    def set_target(self, target):
        for view in self.projected_view, self.local_view:
            view.clear()
            scatter = pg.ScatterPlotItem([target], size=30, pen='y')
            view.addItem(scatter)

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

        # search a small subset for trigggers
        pad_samples = self.pad_blocks * self.block_size        
        search_data = data[:, pad_samples:-pad_samples]
        mask = np.abs(search_data) > self.trigger_threshold
        mask_change = mask[:, 1:] & ~mask[:, :-1]
        triggers = [np.argwhere(mask_change[i])[:, 0] for i in range(mask_change.shape[0])]
        triggers = [t[0] for t in triggers if len(t) > 0]
        if len(triggers) == 0:
            return
        
        # got a trigger
        trigger = min(triggers)
        self.last_trigger_time = now

        trigger_index = trigger + pad_samples
        # print(trigger_index, data[:, trigger_index - 1:trigger_index + 1])
        plot_duration = 0.01
        pad_samples = int(plot_duration * self.sample_rate)
        plot_data = data[:, trigger_index - int(pad_samples*0.5):trigger_index + int(pad_samples*1.5)]

        self.trigger_plot.clear()
        t = (np.arange(plot_data.shape[1]) - pad_samples*0.5) / self.sample_rate
        for i,chan in enumerate(plot_data):
            self.trigger_plot.plot(t, chan, pen=(i, 4))
        self.trigger_plot.addLine(y=self.trigger_threshold, pen='w')
        self.trigger_plot.addLine(x=0, pen='w')
            
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

    def projection_roi_changed(self):
        if self.project is not None:
            self.project.save(
                projection_roi_state=self.projected_view.projection_roi.saveState()
            )


class RollingBuffer:
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.blocks = []

    def add_from_queue(self, queue):
        while queue.qsize() > 0:
            self.blocks.append(queue.get())

        while queue.qsize() > 0:
            self.blocks.append(queue.get())

        if len(self.blocks) > self.n_blocks:
            discard = len(self.blocks) - self.n_blocks
            self.blocks = self.blocks[discard:]

    def get_data(self):
        first_index = self.blocks[0][0]
        return first_index, np.concatenate([block_data for (sample_index, block_data) in self.blocks], axis=1)


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
        self.grid = pg.PlotCurveItem(pen=0.5)
        self.view.addItem(self.grid)

        self.projection_roi = ProjectionROI()
        self.projection_roi.sigRegionChanged.connect(self.projection_roi_changed)
        self.view.addItem(self.projection_roi)
        self.view.setAspectLocked(True)

        self.resize(1920, 1080)

        self.view.scene().sigMouseClicked.connect(self.mouse_clicked)

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

    def mouse_clicked(self, ev):
        self.mouse_cliecked