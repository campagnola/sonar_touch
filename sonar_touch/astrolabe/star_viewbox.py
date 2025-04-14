import time
import numpy as np
import pyqtgraph as pg
import coorx
from sonar_touch.astrolabe.grid import AzimuthalGrid
from .transforms import SphericalTransform, LambertAzimuthalEqualAreaTransform
from .star_vis import StarTracks, StarVisualization


class StarViewBox(pg.ViewBox):
    def __init__(self, star_catalog):
        pg.ViewBox.__init__(self)
        self.setAspectLocked()
        self.stars = star_catalog

        self.speed = 25000  # 25,000 years per second
        self.slew_time = 0.3  # 63% after 0.3 second
        self.last_update = time.perf_counter()
        self.paused = False
        self.time = 0
        self.target_time = None
        self.start_time = -200000
        self.stop_time = 200000

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

        self.update_scene()

        self.setXRange(-2, 2)
        self.setYRange(-2, 2)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(16)

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
            self.speed *= 0.8
        elif ev.text() in ['+', '=']:
            self.speed /= 0.8
        elif ev.text() == ' ':
            self.pause(not self.paused)
        elif ev.text() in '12345':
            t = (float(ev.text()) - 1) / 4
            t = self.start_time + t * (self.stop_time - self.start_time)
            self.slew_to_time(t)
        else:
            ev.ignore()

    def slew_to_time(self, time):
        self.paused = True
        self.target_time = time
