from sonar_touch.astrolabe.mainwindow import main
import pyqtgraph as pg

pg.dbg()
win, view = main()
win.showMaximized()

