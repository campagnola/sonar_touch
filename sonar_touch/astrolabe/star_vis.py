import numpy as np
import pyqtgraph as pg
from .color import btvt_to_rgb
from .stars import constellations


# make some custom scatter plot symbols for stars
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

def star_scale(i):
    return 1.3**(i - 1)

for i in range(1, 6):
    path = pg.QtGui.QPainterPath()
    npts = 32  # must be multiple of 4
    for j in range(npts):
        th = j * 2*np.pi/npts
        x,y = 0.5 * np.cos(th), 0.5 * np.sin(th)
        if j not in (0, npts//4, npts//2, 3*npts//4):
            x /= star_scale(i)
            y /= star_scale(i)
        if j == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    # path.closeSubpath()

    Symbols[str(i)] = path


class StarVisualization:
    def __init__(self, stars, transform):
        self.stars = stars
        self.transform = transform
        self.time = 0
        self.pos_at_time = None
        self.mapped_pos = None
        self.zoom = 1

        self._sizes = None
        self._symbols = None
        self._brushes = None

        self.scatter = pg.ScatterPlotItem(
            pos=self.get_mapped_pos(),
            size=self.sizes,
            brush=self.brushes,
            pen=None,
            symbol=self.symbols,
            pxMode=True,
            data=np.arange(len(self.stars.data)),
        )

        self.constellations = {
            name: Constellation(self.stars, cdata, pen=pg.mkPen((130, 180, 255, 100), width=2))
            for name, cdata in constellations.items()
        }
        for constellation in self.constellations.values():
            constellation.setParentItem(self.scatter)
            constellation.setFlag(pg.QtWidgets.QGraphicsItem.ItemStacksBehindParent)
            constellation.setZValue(-1)

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        normalized_magnitude = (6.5 - self.stars.magnitudes) / 6.5
        exponent = 3  # larger exponent = more contrast between small and large stars
        scale = 10  # scale factor for all stars
        max_size = 12  # largest circular star before switching to spiny star shapes

        sizes = scale * self.zoom * normalized_magnitude**exponent
        clipped_sizes = np.clip(sizes, 0, max_size)  # max size before we use different symbols rather than size
        # quantize sizes to help with scatter plot performance
        quantized_sizes = np.exp((np.log(clipped_sizes)*5).astype(np.int32)/5)
        clip_ratio = sizes / quantized_sizes
        star_symbol_num = np.clip(np.log2(clip_ratio*4).astype(int), 1, 5)
        
        # larger star spines for stars larger than max
        self._sizes = quantized_sizes * star_scale(star_symbol_num)
        self._symbols = star_symbol_num.astype('U1')

        return self._sizes

    @property
    def symbols(self):
        if self._symbols is None:
            self.sizes  # forces calculation of symbols
        return self._symbols


    @property
    def brushes(self):
        if self._brushes is not None:
            return self._brushes

        # how much bv color to mix with white
        mix = 0.7
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
            'symbol': self.symbols,
            'brush': self.brushes,
            'data': np.arange(len(self.stars.data)),
        }
        if update_sizes:
            args['size'] = self.sizes
        self.scatter.setData(**args)
        for constellation in self.constellations.values():
            constellation.update_positions(self.get_mapped_pos())


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


class Constellation(pg.PlotCurveItem):
    def __init__(self, stars, cdata, pen):
        super().__init__()
        self.stars = stars
        self.cdata = cdata
        self.setPen(pen)

        inds = []
        hip_id_lookup = {}
        for i,(_, row) in enumerate(self.stars.data.iterrows()):
            hip_id_lookup.setdefault(row['hip_id'], i)
        for star1_id, star2_id in cdata['lines']:
            ind1 = hip_id_lookup[star1_id]
            ind2 = hip_id_lookup[star2_id]
            inds.extend([ind1, ind2])
        self.indices = inds

    def update_positions(self, positions):
        verts = positions[self.indices]
        self.setData(verts[:,0], verts[:,1], connect='pairs')


