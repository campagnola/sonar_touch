import time
import numpy as np
import pyqtgraph as pg
import coorx
from .transforms import SphericalTransform, LambertAzimuthalEqualAreaTransform
from .color import btvt_to_rgb


class StarViewBox(pg.ViewBox):
    def __init__(self, data):
        pg.ViewBox.__init__(self)
        self.setAspectLocked()
        self.data = data

        self.speed = 25000  # 25,000 years per second
        self.slew_time = 0.3  # 63% after 0.3 second
        self.last_update = time.perf_counter()
        self.paused = False
        self.time = 0
        self.target_time = None
        self.start_time = -200000
        self.stop_time = 200000

        deg_to_rad = np.pi / 180.0
        mas_to_rad = deg_to_rad / 3600 / 1000

        positions = np.zeros((len(data), 3))
        positions[:, 0] = -data['ra_degrees_j2000'] * deg_to_rad
        positions[:, 1] = data['dec_degrees_j2000'] * deg_to_rad
        positions[:, 2] = 1 / data['parallax_mas'] * 1000.0
        tr = SphericalTransform()
        self.positions = tr.imap(positions)

        positions[:, 0] += data['ra_mas_per_year'] * mas_to_rad
        positions[:, 1] += data['dec_mas_per_year'] * mas_to_rad
        self.vectors = tr.imap(positions) - self.positions

        self.magnitudes = np.asarray(data['magnitude'])
        self.names = np.asarray(data['name'])

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

        self.update_positions()

        magnitudes = self.magnitudes
        brightness = 15 * ((5.5 - magnitudes) / 5.5)**2
        self.sizes = np.clip(brightness, 1, np.inf)
        self.alphas = 255 * np.clip(brightness, 0.0, 1.0)

        self.brushes = []
        for i, (index, row) in enumerate(data.iterrows()):
            bv = row['bv']
            if np.isnan(bv):
                color = (255, 255, 255)
            else:
                # B-V = 0.850 * (BT-VT)
                color = np.array(btvt_to_rgb(bv))
                mix = 0.5
                color = (mix * color + (1 - mix) * 255)
            alpha = self.alphas[i]
            self.brushes.append(pg.mkBrush(color[0], color[1], color[2], alpha))

        self.scatter = pg.ScatterPlotItem(
            pos=self.mapped_pos,
            size=self.sizes,
            pen=None,
            brush=self.brushes,
            symbol='o',
            pxMode=True,
            data=np.arange(len(self.data)),
        )
        self.addItem(self.scatter)

        self.scatter.sigClicked.connect(self.scatter_clicked)

        self.traveler_lines = pg.PlotCurveItem(pen=(255, 255, 255, 70))
        self.addItem(self.traveler_lines)

        self.update_stars()

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
        self.update_stars()

    def update_positions(self):
        pos = self.positions + self.vectors * self.time
        self.mapped_pos = self.projection.map(pos)[..., :2]

    def update_stars(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            self.update_positions()
            self.update_lines()
            self.scatter.setData(
                pos=self.mapped_pos,
                size=self.sizes,
                brush=self.brushes,
                data=np.arange(len(self.data)),
            )

    def update_lines(self):
        stars_to_draw = [
            'Vega', 'Sirius', 'Capella', 'Arcturus', 'Altair', 'Aljanah', 'Rigil Kentaurus', 'Toliman',
            'Procyon', 'Pollux',
        ]
        verts = []
        connect = []
        for star in stars_to_draw:
            ind = np.argwhere(self.names == star)[0,0]
            pos = self.mapped_pos[ind]
            vec = self.vectors[ind]
            npts = 1000 * np.clip(int(np.linalg.norm(vec)), 2, 100)
            pos = self.positions[ind:ind+1, :] + self.vectors[ind:ind+1, :] * np.linspace(self.start_time, self.stop_time, npts)[:, np.newaxis]
            verts.append(pos)
            connect.append(np.ones(pos.shape[0], dtype=bool))
            connect[-1][-1] = False

        # iso-declination lines
        for dec in range(-90, 90, 30):
            dec = dec * np.pi / 180
            npts = 512
            pos = np.empty((npts, 3))
            pos[:, 0] = np.linspace(0, 2*np.pi, npts)
            pos[:, 1] = dec
            pos[:, 2] = 100
            pos = SphericalTransform().imap(pos)
            verts.append(pos)
            connect.append(np.ones(pos.shape[0], dtype=bool))
            connect[-1][-1] = False

        # iso-ascension lines
        for ra in range(0, 180, 30):
            ra = ra * np.pi / 180
            npts = 512
            pos = np.empty((npts, 3))
            pos[:, 0] = ra
            pos[:, 1] = np.linspace(0, 2*np.pi, npts)
            pos[:, 2] = 100
            pos = SphericalTransform().imap(pos)
            verts.append(pos)
            connect.append(np.ones(pos.shape[0], dtype=bool))
            connect[-1][-1] = False

        verts = self.projection.map(np.concatenate(verts))
        connect = np.concatenate(connect)
        self.traveler_lines.setData(verts[:,0], verts[:,1], connect=connect)

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
        self.update_stars()

    def pause(self, pause=True):
        self.paused = pause
        self.target_time = None

    def scatter_clicked(self, item, points):
        if len(points) == 0:
            return
        print(f"Clicked on {len(points)} points")
        for pt in points:
            rec = self.data.iloc[pt.data()]
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
