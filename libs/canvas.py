from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSplitter
from PyQt5.QtCore import QPointF, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPainterPath, QFont

TICK_FIRST, TICK_MIDDLE, TICK_LAST =  range(3); TICK_HORIGENTAL, TICK_VIRTICAL = False, True
TICK_FONT_SIZE = 8

class gpr_spl(QSplitter):
    resized = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super(gpr_spl, self).__init__(*args, **kwargs)
    
    def resizeEvent(self, ev):
        super(gpr_spl, self).resizeEvent(ev)
        self.resized.emit()
    
class Tick(QLabel):
    def __init__(self, v_h, tick_loc = TICK_MIDDLE):#vertical: True, horigental: False
        super(Tick, self).__init__()
        tick_font = QFont(); tick_font.setPointSize(TICK_FONT_SIZE); self.setFont(tick_font)
        self.setContentsMargins(0,0,0,0)
        if v_h == TICK_VIRTICAL:
            self.setText('-'); self.setFixedWidth(5)
            if tick_loc == TICK_MIDDLE: self.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            elif tick_loc == TICK_FIRST: self.setAlignment(Qt.AlignRight|Qt.AlignTop)
            else: self.setAlignment(Qt.AlignRight|Qt.AlignBottom); self.setText("_")
        else:
            self.setText('|'); self.setFixedHeight(5)
            if tick_loc == TICK_MIDDLE: self.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
            elif tick_loc == TICK_FIRST: self.setAlignment(Qt.AlignLeft|Qt.AlignTop)
            else: self.setAlignment(Qt.AlignRight|Qt.AlignTop)

class CustomAxis(QWidget):
    def __init__(self, *args, **kwargs):
        super(CustomAxis, self).__init__(*args, **kwargs)
        self.tick_wdg = QWidget(); self.label_wdg = QWidget(); self.label_font = QFont()
        self.label_font.setPointSize(TICK_FONT_SIZE)
        self.label_lbl_list = []; self.tick_list = []
        self.orient = None

    def setAxis(self, num_ticks, v_h):
        """
        축 방향(가로축, 세로축), 틱의 갯수 설정
        """
        if v_h == TICK_VIRTICAL:# 수직축
            self.layout = QHBoxLayout(); self.tick_layout = QVBoxLayout(); self.label_layout = QVBoxLayout()
            self.layout.addWidget(self.label_wdg); self.layout.addWidget(self.tick_wdg)
            self.orient = TICK_VIRTICAL

        else:# 수평축 
            self.layout = QVBoxLayout(); self.tick_layout = QHBoxLayout(); self.label_layout = QHBoxLayout()
            self.layout.addWidget(self.tick_wdg); self.layout.addWidget(self.label_wdg)
            self.orient = TICK_HORIGENTAL
            
        self.layout.setContentsMargins(0,0,0,0)
        self.tick_layout.setContentsMargins(0,0,0,0); self.tick_layout.setSpacing(0)
        self.label_layout.setContentsMargins(0,0,0,0); self.label_layout.setSpacing(0)
        self.setLayout(self.layout)
        self.tick_wdg.setLayout(self.tick_layout); self.label_wdg.setLayout(self.label_layout)

        for i in range(num_ticks):
            unit_lbl = QLabel(); unit_lbl.setFont(self.label_font)
            self.label_layout.setStretch(i, 1); self.tick_layout.setStretch(i, 1)
            
            if i==0: 
                if self.orient == TICK_VIRTICAL: unit_lbl.setAlignment(Qt.AlignRight|Qt.AlignTop)
                else: unit_lbl.setAlignment(Qt.AlignLeft|Qt.AlignTop)
                tick = Tick(self.orient, TICK_FIRST)
            
            elif i==num_ticks-1: 
                if self.orient == TICK_VIRTICAL: unit_lbl.setAlignment(Qt.AlignRight|Qt.AlignBottom) 
                else: unit_lbl.setAlignment(Qt.AlignRight|Qt.AlignTop)
                tick = Tick(self.orient, TICK_LAST)
            
            else: 
                if self.orient == TICK_VIRTICAL: unit_lbl.setAlignment(Qt.AlignRight|Qt.AlignVCenter) 
                else: unit_lbl.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
                tick = Tick(self.orient, TICK_MIDDLE)
                self.label_layout.setStretch(i, 0); self.tick_layout.setStretch(i, 0)

            self.label_layout.addWidget(unit_lbl); self.tick_layout.addWidget(tick)
            self.label_lbl_list.append(unit_lbl); self.tick_list.append(tick)

        self.half_lbls = [self.tick_list[0], self.tick_list[-1], self.label_lbl_list[0], self.label_lbl_list[-1]]
        self.resizeEvent()

    def resizeEvent(self, ev=None):
        ticks = len(self.label_lbl_list)
        if self.orient==TICK_HORIGENTAL: 
            half_w = int(round(self.width()/(ticks-1)/2))
            for lbl in self.half_lbls: lbl.setFixedWidth(half_w)
        else:
            half_h = int(round(self.height()/(ticks-1)/2))
            for lbl in self.half_lbls: lbl.setFixedHeight(half_h)

    def showTickValue(self, v_range, unit_label):
        min_value, max_value = v_range
        ticks = len(self.label_lbl_list)
        term = (max_value-min_value)/(ticks-1)
        
        for i in range(ticks):
            if i==0: tick_num = min_value
            elif i==ticks-1: tick_num = max_value
            else: tick_num = min_value+term*i
            
            if unit_label=='km': tick_num = round(tick_num, 3)
            elif unit_label=='m': tick_num = round(tick_num, 1)
            else: tick_num = round(tick_num)
            
            label_text = str(tick_num)+unit_label
            self.label_lbl_list[i].setText(label_text)
            
class Graph_Canvas(QWidget):
    def __init__(self, *args, **kwargs):
        super(Graph_Canvas, self).__init__(*args, **kwargs)
        self._painter = QPainter(); self.graph_array = None

    def paintEvent(self, event):
        if self.graph_array is None: return super(Graph_Canvas, self).paintEvent(event)

        w, h = self.width(), self.height()
        a_len = self.graph_array.shape[0]; w_scale, h_scale = w/2, a_len/h
        
        p = self._painter; p.begin(self); p.scale(1.0, 1.0); p.translate(w/2, 0)
        p.setPen(QPen(QColor(255, 0, 0, 255), 1))
        
        path = QPainterPath(); path.moveTo(0, 0)
        for d in range(h): path.lineTo(self.graph_array[int(d*h_scale)]*w_scale, d)
        p.drawPath(path)

        p.end()

    def reset_state(self): self.graph_array = None; self.update()
    def load_scan(self, a_scan): self.graph_array = a_scan; self.repaint()
    def sizeHint(self): return super(Graph_Canvas, self).minimumSizeHint()
    def minimumSizeHint(self): return super(Graph_Canvas, self).minimumSizeHint()

class Canvas(QWidget):
    scrollRequest, ch_changed = pyqtSignal(int), pyqtSignal()
    cur_chs = (0, 0, 0)
    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        self._painter = QPainter()
        self.pixmap = None
        self.scan_class = -1

    def mousePressEvent(self, ev):
        if self.scan_class<0 or self.pixmap is None: return
        
        p = self.pos_cnvt(ev.pos())
        x, y = int(round(p.x())), min(int(round(p.y())), self.pixmap.height()-1)
        b, f, t = Canvas.cur_chs
        if self.scan_class==0: f=x; t=y
        elif self.scan_class==1: b=x; t=y
        elif self.scan_class==2: f=x; b=y
        Canvas.cur_chs = (b, f, t)
        self.ch_changed.emit()
        
    def paintEvent(self, event):
        if self.pixmap is None: return super(Canvas, self).paintEvent(event)
        fx, fy = self.width()/self.area_width, self.height()/self.pixmap.height()  
        p = self._painter; p.begin(self); p.scale(fx, fy)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.translate(QPointF(self.area_offset, 0))
        p.drawPixmap(self.pixmap_offset, 0, self.pixmap)
        
        if self.scan_class>=0:
            b, f, t = Canvas.cur_chs
            if self.scan_class==0: chp = QPointF(f, t)
            elif self.scan_class==1: chp = QPointF(b, t)
            else : chp = QPointF(f, b)
            
            x, y = chp.x(), chp.y()
            p.setPen(QPen(QColor(255, 0, 0, 255), 1/fx))
            p.drawLine(QPointF(x, 0), QPointF(x, self.pixmap.height()))
            p.setPen(QPen(QColor(255, 0, 0, 255), 1/fy))
            p.drawLine(QPointF(self.pixmap_offset, y), QPointF(self.pixmap_offset+self.pixmap.width(), y))
        
        self.setAutoFillBackground(True); pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(0, 0, 0, 255)); self.setPalette(pal)
        p.end()
    
    def pos_cnvt(self, p):
        fx = self.width()/self.area_width; fy = self.height()/self.pixmap.height()
        x, y = p.x(), p.y(); return QPointF(x/fx-self.area_offset, y/fy)
        
    def wheelEvent(self, ev):
        delta = ev.angleDelta().y(); 
        self.scrollRequest.emit(delta)
        ev.accept()
    
    def load_pixmap(self, amin, amax, poff, pixmap):
        self.pixmap = pixmap
        self.area_width = amax-amin
        self.area_offset = -amin; self.pixmap_offset = poff
        self.repaint()
    
    def reset_state(self):self.pixmap = None; self.setAutoFillBackground(False); self.repaint()
    def sizeHint(self): return super(Canvas, self).minimumSizeHint()
    def minimumSizeHint(self): return super(Canvas, self).minimumSizeHint()