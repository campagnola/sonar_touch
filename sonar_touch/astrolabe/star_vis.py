import numpy as np
import pyqtgraph as pg
from .color import btvt_to_rgb


class StarVisualization:
    def __init__(self, stars, transform):
        self.stars = stars
        self.transform = transform
        self.time = 0
        self.pos_at_time = None
        self.mapped_pos = None

        magnitudes = self.stars.magnitudes
        brightness = 15 * ((5.5 - magnitudes) / 5.5)**2
        self.sizes = np.clip(brightness, 1, np.inf)
        self.alphas = 255 * np.clip(brightness, 0.0, 1.0)

        self.brushes = []
        for i, (index, row) in enumerate(self.stars.data.iterrows()):
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
            pos=self.get_mapped_pos(),
            size=self.sizes,
            pen=None,
            brush=self.brushes,
            symbol='o',
            pxMode=True,
            data=np.arange(len(self.stars.data)),
        )

    def set_time(self, time):
        self.time = time
        self.pos_at_time = None
        self.update_stars()

    def update_transform(self, transform):
        self.transform = transform
        self.update_stars()

    def get_mapped_pos(self):
        if self.pos_at_time is None:
            self.pos_at_time = self.stars.positions + self.stars.vectors * self.time
            self.mapped_pos = None
        if self.mapped_pos is None:
            self.mapped_pos = self.transform.map(self.pos_at_time)[..., :2]
        return self.mapped_pos

    def update_stars(self):
        self.scatter.setData(
            pos=self.get_mapped_pos(),
            size=self.sizes,
            brush=self.brushes,
            data=np.arange(len(self.stars.data)),
        )


class StarTracks(pg.PlotCurveItem):
    def __init__(self, stars, pen, time_range, stars_to_draw):
        super().__init__()
        self.stars = stars
        self.setPen(pen)
        self.start_time = time_range[0]
        self.stop_time = time_range[1]

        verts = []
        connect = []
        for star in stars_to_draw:
            ind = np.argwhere(self.stars.names == star)[0,0]
            vec = self.stars.vectors[ind]
            npts = 1000 * np.clip(int(np.linalg.norm(vec)), 2, 100)
            pos = self.stars.positions[ind:ind+1, :] + self.stars.vectors[ind:ind+1, :] * np.linspace(self.start_time, self.stop_time, npts)[:, np.newaxis]
            verts.append(pos)
            connect.append(np.ones(pos.shape[0], dtype=bool))
            connect[-1][-1] = False

        self.verts = np.concatenate(verts)
        self.connect = np.concatenate(connect)

    def update_transform(self, transform):
        verts = transform.map(self.verts)
        self.setData(verts[:,0], verts[:,1], connect=self.connect)


