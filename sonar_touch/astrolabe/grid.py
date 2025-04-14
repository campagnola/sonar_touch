import numpy as np
import pyqtgraph as pg
from .transforms import SphericalTransform


class AzimuthalGrid(pg.PlotCurveItem):
    def __init__(self, pen):
        super().__init__()

        self.setPen(pen)
        
        verts = []
        connect = []

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

        self.verts = np.concatenate(verts)
        self.connect = np.concatenate(connect)

    def update_transform(self, transform):
        verts = transform.map(self.verts)
        self.setData(verts[:,0], verts[:,1], connect=self.connect)
