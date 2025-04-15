import pyqtgraph as pg
import numpy as np

class Timeline(pg.QtWidgets.QGraphicsItem):
    def __init__(self, endpoints, time_range, pen):
        super().__init__()

        self.setFlag(pg.QtWidgets.QGraphicsItem.ItemHasNoContents)
        self.time_range = time_range

        self.time = 0

        pen = pg.mkPen(pen)
        self.line = pg.QtWidgets.QGraphicsPathItem()
        self.line.setPen(pen)
        self.line.setParentItem(self)      
        self.set_positions(endpoints) 

        self.label = pg.TextItem(
            text='',
            anchor=(0.5, 1.3),
            color='w',
        )

        self.label.setParentItem(self)

        self.marker = pg.QtWidgets.QGraphicsPathItem()
        # draw a triangle
        tri_scale = 10
        path = pg.QtGui.QPainterPath()
        path.moveTo(0, 0)
        path.lineTo(-tri_scale, -tri_scale * 3**0.5)
        path.lineTo(tri_scale, -tri_scale * 3**0.5)
        path.lineTo(0, 0)
        self.marker.setPath(path)
        self.marker.setPen(pen)
        self.marker.setBrush(pg.mkBrush((0, 0, 128, 255)))
        self.marker.setParentItem(self)

    def boundingRect(self):
        return pg.QtCore.QRectF()
        
    def paint(self, *args, **kwds):
        pass

    def set_positions(self, positions):
        positions = np.array(positions)
        self.positions = positions

        path = pg.QtGui.QPainterPath()

        p1 = positions[0]
        p2 = positions[1]
        path.moveTo(*p1)
        path.lineTo(*p2)

        npts = 7
        tick = np.array([0, 10])
        for i,p in enumerate(np.linspace(p1, p2, npts)):
            path.moveTo(*p)
            tick_scale = 1 if i in (0, (npts-1)//2, npts-1) else 0.5                
            path.lineTo(*(p + tick * tick_scale))
        
        self.line.setPath(path)        

    def set_time(self, time):
        self.time = time
        self.label.setText(f'{int(time/1000):d} ky')
        self.label.setFont(pg.QtGui.QFont('Arial', 20, weight=1))

        fraction_of_range = (time - self.time_range[0]) / (self.time_range[1] - self.time_range[0])
        pt = self.positions[0] + fraction_of_range * (self.positions[1] - self.positions[0])
        self.marker.setPos(pt[0], pt[1])
        self.label.setPos(pt[0], pt[1])

