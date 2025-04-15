import time
import os
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.console
import torch
import coorx

from sonar_touch.audio import RollingBuffer
from sonar_touch.project import SonarTouchProject
from sonar_touch.training import TrainingDataCollector

app = pg.mkQApp()


class MainWindow(pg.QtWidgets.QMainWindow):

    detected_tap = pg.QtCore.pyqtSignal(object)  # location of tap

    def __init__(self, audio_queue, sample_rate, block_size):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.plotting_enabled = True

        self.last_trigger_time = 0
        self.trigger_threshold = 0.04
        self.refractory_period = 0.2
        self.trigger_padding = (0.01, 0.03)
        self.full_buffer_length = 2.0
        self.buffer = RollingBuffer(int(self.full_buffer_length * self.sample_rate / self.block_size) + 1)

        self.init_ui()

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.handle_audio_data)
        self.timer.start(50)

        self.project: SonarTouchProject|None = None
        self.trainer = None
        self.model = None        

    def enable_plotting(self, enable):
        """Enable or disable plotting"""
        self.plotting_enabled = enable

    def init_ui(self):
        self.setWindowTitle("Sonar Touch")

        # file menu for loading a project folder
        file_menu = self.menuBar().addMenu("&File")
        self.load_action = file_menu.addAction("&Load")
        self.load_action.triggered.connect(self.load_project_triggered)

        self.train_action = file_menu.addAction("&Start Training")
        self.train_action.triggered.connect(self.start_training)

        self.console = pg.console.ConsoleWidget(namespace={'win': self})
        self.console_action = file_menu.addAction("&Console")
        self.console_action.triggered.connect(self.console.show)

        # Main layout with splitter
        main_layout = pg.QtWidgets.QHBoxLayout()
        self.setCentralWidget(pg.QtWidgets.QWidget())
        self.centralWidget().setLayout(main_layout)
        
        # Create splitter for left panel and main content
        self.splitter = pg.QtWidgets.QSplitter()
        main_layout.addWidget(self.splitter)
        
        # Left panel for training data browser
        self.left_panel = pg.QtWidgets.QWidget()
        left_layout = pg.QtWidgets.QGridLayout()
        self.left_panel.setLayout(left_layout)
        self.splitter.addWidget(self.left_panel)
        
        # Training data tree widget
        self.training_tree = pg.QtWidgets.QTreeWidget()
        self.training_tree.setHeaderLabels(["ID", "Tapper", "Location", "Date", "File"])
        self.training_tree.setSelectionMode(pg.QtWidgets.QAbstractItemView.ContiguousSelection)
        self.training_tree.itemSelectionChanged.connect(self.training_example_selected)
        left_layout.addWidget(self.training_tree, 0, 0, 1, 2)
        
        # editable tapper combo box
        self.tapper_combo = pg.QtWidgets.QComboBox()
        left_layout.addWidget(self.tapper_combo, left_layout.rowCount(), 0, 1, 2)
        self.tapper_combo.setEditable(True)

        # trigger threshold
        self.trigger_threshold_spin = pg.SpinBox(value=self.trigger_threshold, minStep=0.001, dec=True, compactHeight=False)
        self.trigger_threshold_spin.sigValueChanged.connect(self.trigger_threshold_spin_changed)
        left_layout.addWidget(pg.QtWidgets.QLabel("Trigger Threshold"), left_layout.rowCount(), 0, 1, 1)
        left_layout.addWidget(self.trigger_threshold_spin, left_layout.rowCount() - 1, 1, 1, 1)
        
        # Training example audio plot
        self.example_plot = pg.PlotWidget()
        self.example_plot.setTitle("Selected Training Example")
        self.example_plot.setLabel('bottom', 'Time', 's')
        self.example_plot.setYRange(-1, 1)
        self.example_plot.setMaximumHeight(200)
        left_layout.addWidget(self.example_plot, left_layout.rowCount(), 0, 1, 2)
        
        # Right panel for main content
        self.right_panel = pg.QtWidgets.QWidget()
        right_layout = pg.QtWidgets.QVBoxLayout()
        self.right_panel.setLayout(right_layout)
        self.splitter.addWidget(self.right_panel)
        
        # Main audio plots
        self.cw = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.cw)
        self.plot = self.cw.addPlot(row=0, col=0)
        self.plot.setYRange(-1, 1)

        self.trigger_plot = self.cw.addPlot(row=1, col=0)
        self.trigger_plot.setYRange(-.2, .2)
        
        # Set initial splitter sizes
        self.splitter.setSizes([300, 900])

        self.resize(1200, 600)
        self.show()

        # second window for projection
        self.projected_view = ProjectedView()
        self.projected_view.projection_roi.sigRegionChangeFinished.connect(self.projection_roi_changed)

        quit_shortcut = pg.QtWidgets.QShortcut(pg.QtGui.QKeySequence("Ctrl+Q"), self)
        quit_shortcut.setContext(pg.QtCore.Qt.ApplicationShortcut)
        quit_shortcut.activated.connect(self.close)

        # Add delete shortcut for training examples
        delete_shortcut = pg.QtWidgets.QShortcut(pg.QtGui.QKeySequence("Del"), self)
        delete_shortcut.activated.connect(self.delete_training_examples)

    def handle_audio_data(self):
        if self.audio_queue.qsize() == 0:
            return

        # read all available data from the queue into rolling buffer
        self.buffer.add_from_queue(self.audio_queue)
        sample_index, data = self.buffer.get_data()

        # plot all data in the buffer
        if self.plotting_enabled:
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

        if self.plotting_enabled:
            self.trigger_plot.clear()
            t = (np.arange(plot_data.shape[1]) - trigger_index) / self.sample_rate
            for i,chan in enumerate(plot_data):
                self.trigger_plot.plot(t, chan, pen=(i, 4))
            self.trigger_plot.addLine(y=self.trigger_threshold, pen='w')
            self.trigger_plot.addLine(x=0, pen='w')

        if self.trainer is not None and self.trainer.run:
            self.trainer.trigger_detected(trigger_result, self.sample_rate, self.tapper_combo.currentText())
        elif self.model is not None:
            self.predict(trigger_result)
            
    def close(self):
        self.timer.stop()
        self.projected_view.close()
        return super().close()

    def trigger_threshold_spin_changed(self):
        self.trigger_threshold = self.trigger_threshold_spin.value()

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
        print("Available models:", models)
        if len(models) > 0:
            self.model = self.project.load_model(models[-1])
        
        self.load_training_examples()

        self.tapper_combo.clear()
        self.tapper_combo.addItems(self.project.list_tappers())


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
            # Refresh training examples after stopping training
            self.load_training_examples()

    def predict(self, trigger):
        if self.model is None:
            return
        data = trigger['data']
        tensor = torch.tensor(data.reshape(1, 4, -1), dtype=torch.float32).to(self.model.device)
        location = self.model(tensor).detach().cpu().numpy()[0]
        self.projected_view.set_target(location)
        self.detected_tap.emit(location)
        
    def load_training_examples(self):
        """Load all training examples from the project folder and populate the tree widget"""
        if self.project is None:
            return
            
        self.training_tree.clear()
        
        # Get training index
        if not hasattr(self.project, 'training_index') or not self.project.training_index:
            return
            
        # Add each example to the tree
        for entry in self.project.training_index:
            item = pg.QtWidgets.QTreeWidgetItem([
                str(entry['id']),
                entry['tapper'],
                f"({entry['location'][0]:.1f}, {entry['location'][1]:.1f})",
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp'])),
                entry['filename'],
            ])
            item.record = entry
            self.training_tree.addTopLevelItem(item)
            
    def training_example_selected(self):
        """Handle selection of a training example in the tree"""
        if self.project is None:
            return
        selected_items = self.training_tree.selectedItems()
        if not selected_items:
            self.example_plot.clear()
            return
        item = selected_items[0]
        
        # Load the training example data
        example_data = self.project.load_training_example(item.record)
        location = item.record['location']
        
        # Display the audio data
        self.example_plot.clear()
        t = np.arange(example_data.shape[1]) / self.sample_rate
        for i, chan in enumerate(example_data):
            self.example_plot.plot(t, chan, pen=(i, 4))
            
        # Update the target in the projected view
        self.projected_view.set_target(location)
            
    def delete_training_examples(self):
        """Delete the selected training example after confirmation"""
        selected_items = self.training_tree.selectedItems()
        if not selected_items:
            return
        
        # Confirm deletion
        confirm = pg.QtWidgets.QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Delete {len(selected_items)} training examples?",
            pg.QtWidgets.QMessageBox.Yes | pg.QtWidgets.QMessageBox.No
        )
        
        if confirm == pg.QtWidgets.QMessageBox.Yes:
            for item in selected_items:
                self.delete_training_example(item.record)
                self.training_tree.takeTopLevelItem(self.training_tree.indexOfTopLevelItem(item))
            self.example_plot.clear()

    def delete_training_example(self, entry):
        """Delete a single training example"""
        if self.project is None:
            raise ValueError("No project loaded")

        filename = entry['filename']
        example_path = os.path.join(self.project.project_path, filename)
        os.remove(example_path)
        
        # Update the training index
        self.project.training_index.remove(entry)

        # Rewrite the index file
        self.project.save_training_index()


class ProjectionROI(pg.PolyLineROI):
    def __init__(self):
        pos = [[0, 0], [1920, 0], [1920, 1280], [0, 1280]]
        pg.PolyLineROI.__init__(self, pos, closed=True)
        self.selected_handle = None
        for h in self.handles:
            h['item'].hoverPen = pg.mkPen((255, 255, 0), width=3)
        
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
        
    def select_handle(self, index):
        """Select a handle by index (0-3) and update its appearance"""
        # Reset all handles to default appearance
        for h in self.handles:
            h['item'].currentPen = h['item'].pen
            h['item'].update()
            
        # Set the selected handle
        if 0 <= index < len(self.handles):
            self.selected_handle = self.handles[index]['item']
            self.selected_handle.currentPen = self.selected_handle.hoverPen
            self.selected_handle.update()
        else:
            self.selected_handle = None
            
    def move_selected_handle(self, dx, dy):
        """Move the selected handle by the specified delta"""
        if self.selected_handle is not None:
            pos = self.selected_handle.pos()
            self.selected_handle.setPos(pos.x() + dx, pos.y() + dy)
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
        
        # Enable keyboard focus for key events
        self.setup_keyboard_shortcuts()
        self.setFocus()

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
        
    def setup_keyboard_shortcuts(self):
        """Make the window focusable to receive keyboard events"""
        self.setFocusPolicy(pg.QtCore.Qt.StrongFocus)
    
    def keyPressEvent(self, event):
        """Handle key press events for arrow keys"""
        key = event.key()
        
        # Handle number keys 1-4 for corner selection
        if pg.QtCore.Qt.Key_1 <= key <= pg.QtCore.Qt.Key_4:
            self.projection_roi.select_handle(key - pg.QtCore.Qt.Key_1)
            event.accept()
            return
            
        # Handle arrow keys for moving the selected corner
        if key in (pg.QtCore.Qt.Key_Left, pg.QtCore.Qt.Key_Right, 
                  pg.QtCore.Qt.Key_Up, pg.QtCore.Qt.Key_Down):
            dx, dy = 0, 0
            if key == pg.QtCore.Qt.Key_Left:
                dx = -1
            elif key == pg.QtCore.Qt.Key_Right:
                dx = 1
            elif key == pg.QtCore.Qt.Key_Up:
                dy = -1
            elif key == pg.QtCore.Qt.Key_Down:
                dy = 1
                
            self.projection_roi.move_selected_handle(dx, dy)
            event.accept()
            return
            
        super().keyPressEvent(event)
