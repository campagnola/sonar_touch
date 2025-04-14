import numpy as np
import pyqtgraph as pg
from .color import btvt_to_rgb


class StarVisualization:
    def __init__(self, stars, pos):
        self.stars = stars

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
            pos=pos,
            size=self.sizes,
            pen=None,
            brush=self.brushes,
            symbol='o',
            pxMode=True,
            data=np.arange(len(self.stars.data)),
        )

    def update_stars(self, pos):
        self.scatter.setData(
            pos=pos,
            size=self.sizes,
            brush=self.brushes,
            data=np.arange(len(self.stars.data)),
        )

