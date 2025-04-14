import pyqtgraph as pg
from sonar_touch.astrolabe.stars import load_bigsky
from sonar_touch.astrolabe.star_viewbox import StarViewBox


pg.dbg()


if __name__ == '__main__':

    pg.mkQApp()
    pg.setConfigOption('antialias', True)

    w = pg.GraphicsLayoutWidget()
    w.resize(800, 600)


    stars = load_bigsky()


    view = StarViewBox(
        data=stars,
    )
    w.addItem(view)
    w.show()

