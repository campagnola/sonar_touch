import re
import time
import numpy as np
import pyqtgraph as pg
import coorx
from sonar_touch.astrolabe.grid import AzimuthalGrid
from sonar_touch.astrolabe.timeline import Timeline
from .transforms import SphericalTransform, LambertAzimuthalEqualAreaTransform
from .star_vis import StarTracks, StarVisualization
from .stars import constellations


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
        self.target_view = None

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

        self.home_view = self.get_current_view()

        self.star_item = StarVisualization(self.stars, self.projection)
        self.addItem(self.star_item.scatter)
        self.star_item.scatter.sigClicked.connect(self.scatter_clicked)

        self.travelers = StarTracks(
            self.stars,
            pen=(255, 255, 255, 80), 
            time_range=(self.start_time, self.stop_time),
            stars_to_draw=[
                'Vega', 'Sirius', 'Capella', 'Arcturus', 'Altair', 'Aljanah', 'Rigil Kentaurus', 'Toliman',
                'Procyon', 'Pollux', 'Aldebaran', 'Tabit',
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

        self.constellation_text = pg.QtWidgets.QGraphicsTextItem()
        self.constellation_text.setParentItem(self)
        self.constellation_text.setZValue(1)
        self.constellation_text.setDefaultTextColor(pg.mkColor(255, 255, 255, 255))
        self.constellation_text.setTextWidth(200)
        # self.constellation_text.setFont(pg.mkFont('Arial', 12))

        self.update_scene_transforms()
        self.set_zoom(1.0)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.timed_update)
        self.timer.start(16)

    def get_current_view(self):
        look = self.rotation_tr.inverse.map([0., 0., 1.])
        look_up = self.rotation_tr.inverse.map([0., 1., 1.])
        right = np.cross(look, look_up)
        up = np.cross(right, look)
        up /= np.linalg.norm(up)
        return {'look': [float(x) for x in look], 'up': [float(x) for x in up], 'zoom': self.zoom}

    def set_view(self, view):
        right = np.cross(view['look'], view['up'])
        right /= np.linalg.norm(right)
        self.rotation_tr.set_mapping(
            np.vstack([
                [0, 0, 0],
                view['look'],
                right,
                view['up'],
            ]),
            np.vstack([
                [0, 0, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, 1, 0],
            ])
        )
        self.set_zoom(view['zoom'])
        self.update_scene_transforms()

    def focus_constellation(self, constellation):
        if constellation is None:
            self.constellation_text.setPlainText('')
            return
        
        desc = re.sub('\s+', ' ', constellations[constellation]['desc'])

        self.constellation_text.setHtml(f'<div style="text-align: right"><b>{constellation}</b><br><br><span style="color: #CCC">{desc}</span></div>')
        self.update_text_pos()

        self.slew_to_view(constellations[constellation]['view'])

    def go_home(self):
        self.slew_to_view(self.home_view)

    def update_text_pos(self):
        self.constellation_text.setPos(self.width() - self.constellation_text.boundingRect().width() - 10,
                                       self.height() - self.constellation_text.boundingRect().height() - 10)

    def resizeEvent(self, ev):
        self.update_text_pos()
        return super().resizeEvent(ev)

    def set_zoom(self, z):
        self.zoom = z
        visible_radius = self.initial_visible_radius / self.zoom
        self.setXRange(-visible_radius, visible_radius)
        self.setYRange(-visible_radius, visible_radius)
        self.star_item.set_zoom(self.zoom)

    def wheelEvent(self, event, axis=None):
        event.accept()
        if event.delta() > 0:
            self.set_zoom(self.zoom*1.3)
        else:
            self.set_zoom(self.zoom/1.3)

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()
        global e
        e = ev
        if ev.isStart():
            return
        if ev.isFinish():
            return
        delta = e.lastScenePos() - e.scenePos()
        self.rotation_tr.rotate(delta.y() * 0.3 / self.zoom, axis=(0, 1, 0))
        self.rotation_tr.rotate(-delta.x() * 0.3 / self.zoom, axis=(1, 0, 0))
        self.update_scene_transforms()

    def update_scene_transforms(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            self.star_item.update_transform(self.projection)
            self.travelers.update_transform(self.projection)
            self.grid.update_transform(self.projection)

    def slew_to_view(self, view):
        self.target_view = view

    def timed_update(self):
        now = time.perf_counter()
        dt = now - self.last_update
        self.last_update = now
        slew_amount = dt / self.slew_time

        # play or slew to target time
        if self.paused:
            if self.target_time is not None:
                # calculate slew amount for this dt
                t = slew_amount * self.target_time + (1 - slew_amount) * self.time
                if np.abs(t - self.target_time) < 10:
                    t = self.target_time
                    self.target_time = None
                self.set_time(t)
        else:
            t = self.time + self.speed * dt
            if t > self.stop_time:
                t = self.start_time
            self.set_time(t)

        # slew to target position / orientation / zoom
        if self.target_view is not None:
            view = self.get_current_view()
            target = self.target_view
            next_view = {
                'look': slew_amount * np.array(target['look']) + (1 - slew_amount) * np.array(view['look']),
                'up': slew_amount * np.array(target['up']) + (1 - slew_amount) * np.array(view['up']),
                'zoom': slew_amount * target['zoom'] + (1 - slew_amount) * view['zoom'],
            }
            self.set_view(next_view)

            if np.allclose(next_view['look'], target['look']) and \
               np.allclose(next_view['up'], target['up']) and \
               np.allclose(next_view['zoom'], target['zoom']):
                self.target_view = None

    def set_time(self, time):
        self.time = time
        self.star_item.set_time(self.time)
        self.timeline.set_time(self.time)
        self.update_scene_transforms()

    def pause(self, pause=True):
        self.paused = pause
        self.target_time = None

    def scatter_clicked(self, item, points):
        if len(points) == 0:
            return

        max_mag, max_pt = None, None        
        for pt in points:
            rec = self.stars.data.iloc[pt.data()]
            if max_mag is None or rec['magnitude'] < max_mag:
                max_mag = rec['magnitude']
                max_pt = rec

        if not hasattr(self, '_last_click') or self._last_click is None:
            self._last_click = max_pt
        else:
            print(f'[{self._last_click['hip_id']}, {max_pt["hip_id"]}],') 
            self._last_click = None

    def keyPressEvent(self, ev):
        ev.accept()
        if ev.text() == '-':
            self.play_speed *= 0.8
            self.speed = self.play_speed
        elif ev.text() in ['+', '=']:
            self.play_speed /= 0.8
            self.speed = self.play_speed
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
