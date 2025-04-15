import time
import numpy as np
import pyqtgraph as pg
import coorx
from sonar_touch.astrolabe.grid import AzimuthalGrid
from sonar_touch.astrolabe.timeline import Timeline
from .transforms import SphericalTransform, LambertAzimuthalEqualAreaTransform
from .star_vis import StarTracks, StarVisualization


class StarViewBox(pg.ViewBox):
    def __init__(self, star_catalog):
        pg.ViewBox.__init__(self)
        self.setAspectLocked()
        self.stars = star_catalog

        self.play_speed = 25000  # 25,000 years per second
        self.speed = self.play_speed
        self.slew_time = 0.3  # 63% after 0.3 second
        self.last_update = time.perf_counter()
        self.paused = False
        self.time = 0
        self.target_time = None
        self.start_time = -200000
        self.stop_time = 200000

        self.initial_visible_radius = 2
        self.zoom = 1.0

        self.angle = [0, 0]
        self.rotation_tr = coorx.AffineTransform(dims=(3, 3))
        self.perspective_tr = coorx.linear.PerspectiveTransform()
        self.perspective_tr.set_perspective(fovy=60, aspect=1.0, znear=0.001, zfar=100000.0)
        self.projection = coorx.CompositeTransform([
            # SphericalTransform().inverse,
            self.rotation_tr,
            SphericalTransform(),
            # MercatorSphericalTransform()
            # self.perspective_tr,
            LambertAzimuthalEqualAreaTransform(),
        ])

        self.star_item = StarVisualization(self.stars, self.projection)
        self.addItem(self.star_item.scatter)
        self.star_item.scatter.sigClicked.connect(self.scatter_clicked)

        self.travelers = StarTracks(
            self.stars,
            pen=(255, 255, 255, 80), 
            time_range=(self.start_time, self.stop_time),
            stars_to_draw=[
                'Vega', 'Sirius', 'Capella', 'Arcturus', 'Altair', 'Aljanah', 'Rigil Kentaurus', 'Toliman',
                'Procyon', 'Pollux', 'Aldebaran'
            ],
        )
        self.addItem(self.travelers)
        self.travelers.setZValue(-1)

        self.grid = AzimuthalGrid(pen=(255, 255, 255, 50))
        self.grid.setZValue(-2)
        self.addItem(self.grid)

        self.timeline = Timeline(
            endpoints=[[50, 50], [400, 50]], 
            time_range=(self.start_time, self.stop_time), 
            pen=pg.mkPen((255, 255, 255, 128), width=2),
        )
        self.timeline.setParentItem(self)

        self.update_scene()
        self.set_zoom(1.0)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(16)

    def set_zoom(self, z):
        self.zoom = z
        visible_radius = self.initial_visible_radius / self.zoom
        self.setXRange(-visible_radius, visible_radius)
        self.setYRange(-visible_radius, visible_radius)
        self.star_item.set_zoom(self.zoom)

    def wheelEvent(self, event, axis=None):
        event.accept()
        if event.delta() > 0:
            self.set_zoom(self.zoom*1.1)
        else:
            self.set_zoom(self.zoom/1.1)

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()
        global e
        e = ev
        if ev.isStart():
            return
        if ev.isFinish():
            return
        delta = e.lastScenePos() - e.scenePos()
        self.rotation_tr.rotate(delta.y() * 0.3, axis=(0, 1, 0))
        self.rotation_tr.rotate(-delta.x() * 0.3, axis=(1, 0, 0))
        self.update_scene()

    def update_scene(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            self.star_item.update_transform(self.projection)
            self.travelers.update_transform(self.projection)
            self.grid.update_transform(self.projection)

    def update_time(self):
        now = time.perf_counter()
        dt = now - self.last_update
        self.last_update = now

        if self.paused:
            if self.target_time is not None:
                # calculate slew amount for this dt
                slew_amount = dt / self.slew_time
                t = slew_amount * self.target_time + (1 - slew_amount) * self.time
                self.set_time(t)
        else:
            t = self.time + self.speed * dt
            if t > self.stop_time:
                t = self.start_time
            self.set_time(t)

    def set_time(self, time):
        self.time = time
        self.star_item.set_time(self.time)
        self.timeline.set_time(self.time)
        self.update_scene()

    def pause(self, pause=True):
        self.paused = pause
        self.target_time = None

    def scatter_clicked(self, item, points):
        if len(points) == 0:
            return
        for pt in points:
            rec = self.stars.data.iloc[pt.data()]
            print(rec['name'])

    def keyPressEvent(self, ev):
        ev.accept()
        if ev.text() == '-':
            self.play_speed *= 0.8
        elif ev.text() in ['+', '=']:
            self.play_speed /= 0.8
        elif ev.text() == ' ':
            self.speed = self.play_speed
            self.pause(not self.paused)
        elif ev.text() != '' and ev.text() in '12345':
            t = (float(ev.text()) - 1) / 4
            t = self.start_time + t * (self.stop_time - self.start_time)
            self.slew_to_time(t)
        # arrow keys
        elif ev.key() == pg.QtCore.Qt.Key_Left:
            self.speed = -10000
            self.pause(False)
        elif ev.key() == pg.QtCore.Qt.Key_Right:
            self.speed = 10000
            self.pause(False)
        elif ev.key() == pg.QtCore.Qt.Key_Up:
            self.speed = -50000
            self.pause(False)
        elif ev.key() == pg.QtCore.Qt.Key_Down:
            self.speed = 50000
            self.pause(False)
        else:
            ev.ignore()

    def keyReleaseEvent(self, event):
        event.accept()
        if event.key() in (pg.QtCore.Qt.Key_Left, pg.QtCore.Qt.Key_Right, pg.QtCore.Qt.Key_Up, pg.QtCore.Qt.Key_Down):
            self.pause(True)
        else:
            event.ignore()

    def slew_to_time(self, time):
        self.paused = True
        self.target_time = time
