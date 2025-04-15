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
        self.zoom = 1

        magnitudes = self.stars.magnitudes
        # brightness = ((5.5 - magnitudes) / 5.5)**2
        self._sizes = None
        # self.alphas = 255 * np.clip(brightness, 0.0, 1.0)
        self._brushes = None

        self.scatter = pg.ScatterPlotItem(
            pos=self.get_mapped_pos(),
            size=self.sizes,
            brush=self.brushes,
            pen=None,
            symbol='o',
            pxMode=True,
            data=np.arange(len(self.stars.data)),
        )

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        sizes = np.clip(15 * self.zoom * ((5.5 - self.stars.magnitudes) / 5.5)**2, 0, 15)
        # quantize sizes to help with scatter plot performance
        self._sizes = np.exp((np.log(sizes)*5).astype(np.int32)/5)
        return self._sizes

    @property
    def brushes(self):
        if self._brushes is not None:
            return self._brushes

        # how much bv color to mix with white
        mix = 0.5
        # make a limited set of brushes to assist with scatter plot performance
        bv_mean = self.stars.data['bv'].mean()
        bv_std = self.stars.data['bv'].std()
        bv_brushes = {np.nan: pg.mkBrush(255, 255, 255)}
        def get_bv_brush(bv):
            if np.isnan(bv):
                return bv_brushes[np.nan]
            else:
                # B-V = 0.850 * (BT-VT)
                # quantize bv
                bv = int((bv - bv_mean) / bv_std) * bv_std + bv_mean                
                if bv not in bv_brushes:
                    color = np.array(btvt_to_rgb(bv))
                    bv_brushes[bv] = pg.mkBrush(mix * color + (1 - mix) * 255)
                return bv_brushes[bv]

        self._brushes = []
        for i, (index, row) in enumerate(self.stars.data.iterrows()):
            bv = row['bv']
            self._brushes.append(get_bv_brush(bv))
        return self._brushes

    def set_zoom(self, zoom):
        self.zoom = zoom
        self._sizes = None
        self.update_stars(update_sizes=True)

    def set_time(self, time):
        self.time = time
        self.pos_at_time = None
        self.update_stars()

    def update_transform(self, transform):
        self.transform = transform
        self.mapped_pos = None
        self.update_stars()

    def get_mapped_pos(self):
        if self.pos_at_time is None:
            self.pos_at_time = self.stars.positions + self.stars.vectors * self.time
            self.mapped_pos = None
        if self.mapped_pos is None:
            self.mapped_pos = self.transform.map(self.pos_at_time)[..., :2]
        return self.mapped_pos

    def update_stars(self, update_sizes=False):
        args = {
            'pos': self.get_mapped_pos(),
            'size': self.sizes,
            'brush': self.brushes,
            'data': np.arange(len(self.stars.data)),
        }
        if update_sizes:
            args['size'] = self.sizes
        self.scatter.setData(**args)   


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


