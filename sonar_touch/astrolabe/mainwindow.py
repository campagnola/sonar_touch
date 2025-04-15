import pyqtgraph as pg
from sonar_touch.astrolabe.stars import load_bigsky, StarCatalog
from sonar_touch.astrolabe.star_viewbox import StarViewBox


def main():
    pg.mkQApp()
    pg.setConfigOption('antialias', True)

    w = pg.GraphicsLayoutWidget()
    w.resize(800, 600)
    star_data = load_bigsky()
    catalog = StarCatalog(star_data)
    view = StarViewBox(catalog)
    w.addItem(view)
    w.show()

    return w, view