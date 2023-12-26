import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import  *
from PyQt5 import uic

from libs.utils import *
from libs.nm_reader import *
from libs.canvas import Canvas

__appname__ = 'NM GPR Viewer'
class MainWindow(QMainWindow, uic.loadUiType("mainwindow.ui")[0]):
    def __init__(self): # 메인 윈도우 초기화
        super(MainWindow, self).__init__(); self.setupUi(self)
        
        self.settings = Settings(); self.settings.load(); self.set_ui(); self.slot_connect()
        self.last_open_dir = self.settings.get(SETTING_LAST_OPEN_DIR, None)
        self.last_export_dir = self.settings.get(SETTING_LAST_EXPORT_DIR, None)
        
        self.nm_reader = NM_Reader()
        self.nm_reader.set_gvs([w.value() for w in self.gpr_ctrl[1:]])
        
        self.centralwidget.setEnabled(False)
        self.exportting = False
        
    def gvsChanged(self):
        gvs = [w.value() for w in self.gpr_ctrl[1:]]
        self.nm_reader.set_gvs(gvs)
        self.load_scans()
    
    def openDir(self):
        dir_path = self.last_open_dir if os.path.exists(self.last_open_dir) else '.'
        selected_dir = QFileDialog.getExistingDirectory(self, 'Select Folder to open', dir_path)
        self.close_file(); self.file_tree.clear()
        self.nm_reader.open_dir(selected_dir)
        
        root = self.file_tree.invisibleRootItem()
        self.file_tree.expandItem(root)

        dir_name = os.path.split(selected_dir)[1]
        dir_node = QTreeWidgetItem(root, [dir_name, NM_NAMES[self.nm_reader.nm_type]])
        self.file_tree.expandItem(dir_node)

        id_list = self.nm_reader.ids()
        if len(id_list)>0:
            for rps_id in id_list:
                dist = self.nm_reader.gpr_shapes[rps_id][0]*self.nm_reader.dx
                QTreeWidgetItem(dir_node, [rps_id, '{:.2f}m'.format(dist)])
            self.last_open_dir = selected_dir
        
        d_opt = self.nm_reader.settings[NM_NAMES[self.nm_reader.nm_type]]['export']
        self.sel_ch_edt.setText(', '.join(str(ch) for ch in d_opt['b scan']['select']))
        self.sel_dp_edt.setText(', '.join(str(ch) for ch in d_opt['c scan']['select']))
        self.img_len_edt.setText(str(d_opt['dist/image']))
        sizes, bsize, tsize = [d_opt[s]['size'] for s in ['line scan', 'b scan', 'c scan']]
        sizes.extend(bsize); sizes.extend(tsize); sizes = [str(s) for s in sizes]
        for i in range(6): self.img_size_edt[i].setText(sizes[i])
        
    def fileTreeItemSelected(self): self.open_file(self.file_tree.currentItem().text(0))
    
    def open_file(self, rid):
        self.close_file(); self.nm_reader.open_file(rid)
        if self.nm_reader.cur_id is None: return
        if self.nm_reader.nm_type>=LT1: self.gprmode_wdg.setHidden(False)
        else: self.gprmode_wdg.setHidden(True)
        
        dist = self.nm_reader.dx*self.nm_reader.gpr_shape()[0]
        self.x_pos_edt.setMaximum(dist); self.centralwidget.setEnabled(True)
        self.load_scans()
        
    def close_file(self):
        self.nm_reader.close_file()
        for canvas in self.gpr_canvas: canvas.reset_state()
        self.line_canvas.reset_state()
        self.ascan_canvas.reset_state()
        self.x_pos_edt.setValue(0)
        self.displayAxis()
        Canvas.cur_chs = self.nm_reader.chs
        self.centralwidget.setEnabled(False)
            
    def load_scans(self):
        if self.nm_reader.cur_id is None or self.exportting: return
        cur_pos, x_len = self.x_pos_edt.value(), self.x_len_spn.value()/2
        ms, me = cur_pos-x_len, cur_pos+x_len
        # GPR 스캔
        vv = 'vv' if self.vv_rdo.isChecked() else 'hh'
        tr = self.trim_spn.value()
        g_scans = self.nm_reader.gpr_scans((ms, me), tr, vv)
        Canvas.cur_chs = self.nm_reader.chs
        
        # 노면 영상
        ctrls = [w.value() for w in self.road_ctrl]
        lscan = self.nm_reader.road_image((ms, me), ctrls)
        
        # 출력
        for sc, canvas in enumerate(self.gpr_canvas):
            amin, amax, p_off, pixmap = g_scans[sc+1]
            canvas.load_pixmap(amin, amax, p_off, pixmap)
        self.ascan_canvas.load_scan(g_scans[0])
        
        # 노면 스캔
        #ctrls = [w.value() for w in self.road_ctrl]
        amin, amax, poff, pixmap = lscan
        self.line_canvas.load_pixmap(amin, amax, poff, pixmap)
        
        self.displayAxis()
            
    def setPosEdit(self, step): # gpr 출력 구간 변화
        if self.nm_reader.cur_id is None: return
        cur_pos, x_len = self.x_pos_edt.value(), self.x_len_spn.value()
        txt2step = {'first':float('-inf'), 'last':float('inf'), 'next':x_len, 'prev':-x_len}
        if step in txt2step: step = txt2step[step]
        else: step = -x_len/20 if step>0 else x_len/20
        self.x_pos_edt.setValue(cur_pos+step)

    def chPointChanged(self): # 채널 에디터 값 설정 후 loadGPR 호출
        self.nm_reader.chs = Canvas.cur_chs
        self.load_scans()

    def displayAxis(self):
        if self.nm_reader.cur_id is None: return
        
        h = int(round(512*self.trim_spn.value()))
        for ax in self.axises['ns']: ax.showTickValue((0, h), 'ns')

        x_pos = self.x_pos_edt.value(); x_len = self.x_len_spn.value()/2
        m_start, m_end = x_pos-x_len, x_pos+x_len; x_range = [m_start, m_end]
        for ax in self.axises['dist']: ax.showTickValue(x_range, 'm')
        rw, aw = self.nm_reader.road_width/2, self.nm_reader.ant_width/2
        self.axises['road width'].showTickValue((-rw, rw), 'm')
        self.axises['fscan width'].showTickValue((-aw, aw), 'm')
        self.axises['tscan width'].showTickValue((-aw, aw), 'm')
        self.axises['ascan width'].showTickValue((-5000, 5000), '')
    
    def spl_resized(self):
        if self.spl_to_resize: self.init_canvas_split()

    def spl_moved(self):
        cur_spl = self.sender(); sizes = cur_spl.sizes()
        for spl in self.hor_spls: spl.setSizes(sizes)

    def init_canvas_split(self):
        screen_w, screen_h = self.spl_main.width(), self.spl_main.height()
        w_l, w_r = screen_w//8, screen_w//8; w_c = screen_w-w_l-w_r; w_sizes = (w_l, w_c, w_r)
        for spl in self.hor_spls: spl.setSizes(w_sizes)

        h_road = screen_h//3; h_gpr = screen_h-h_road; r_g_sizes = (h_road, h_gpr); self.spl_main.setSizes(r_g_sizes)
        h_t = h_gpr//4; h_bf = h_gpr-h_t; self.spl_gpr.setSizes((h_t, h_bf))

        if self.spl_to_resize: self.spl_to_resize -= 1
    
    def error_message(self, title, message):
        return QMessageBox.critical(self, title, '<p><b>%s</b></p>%s' % (title, message))
        
    def validate_exp_opt(self):
        def intlist(text):
            try: return [int(t.replace(' ', '')) for t in text.split(',')]
            except: return None
        def txt2float(text):
            try: return float(text)
            except: return None
            
        chs, dps = intlist(self.sel_ch_edt.text()), intlist(self.sel_dp_edt.text())
        whs = [txt2float(w.text()) for w in self.img_size_edt]
        img_len = txt2float(self.img_len_edt.text())
        if chs is None: self.error_message('Invalid export option', 'Check the channel list'); return {}
        if dps is None: self.error_message('Invalid export option', 'Check the depth list'); return {}
        if None in whs: self.error_message('Invalid export option', 'Check the image size'); return {}
        if img_len is None: self.error_message('Invalid export option', 'Check the Distance per image'); return {}
        
        return {'channels, depths':(chs, dps), 'print size':((whs[0], whs[1]),(whs[2], whs[3]), (whs[4], whs[5])), 'dist':img_len}
    
    def export(self):
        if self.nm_reader.cur_id is None: return
        opt = self.validate_exp_opt()
        if not opt: return
        t_dir = self.last_export_dir if os.path.exists(self.last_export_dir) and os.path.isdir(self.last_export_dir) else ''
        t_dir = QFileDialog.getExistingDirectory(self, 'Select folder to save', t_dir)
        if not t_dir: return
        
        rids = [self.nm_reader.cur_id] if self.cur_file_rdo.isChecked() else self.nm_reader.ids()
        opt['latro mode'] = self.vv_rdo.isChecked()
        self.last_export_dir = t_dir
        
        self.setEnabled(False)
        for rid in rids:
            QApplication.processEvents()
            self.exp_state_lbl.setText('save images from %s'%rid)
            self.nm_reader.export_file(rid, t_dir, opt, [w.value() for w in self.road_ctrl])
            
        self.exp_state_lbl.setText('All images have been saved to %s'%t_dir)
        self.setEnabled(True)
                          
    def set_ui(self):
        # 윈도우 창 복원
        size = self.settings.get(SETTING_WIN_SIZE, QSize(600, 500)); position = QPoint(0, 0)
        saved_position = self.settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break

        self.resize(size); self.move(position)
        self.restoreState(self.settings.get(SETTING_WIN_STATE, QByteArray()))
        # 주요 위젯 그룹화
        self.hor_spls = [self.spl_gpr_b, self.spl_gpr_t, self.spl_road]; self.spl_to_resize = 8
        self.gpr_canvas = [self.b_canvas, self.f_canvas, self.t_canvas]
        self.line_canvas = self.r_canvas; self.ascan_canvas = self.a_canvas
        self.road_ctrl = [self.r_bright_sld, self.r_contrast_sld]
        self.gpr_ctrl = [self.trim_spn, self.gv_1_sld, self.gv_2_sld, self.gv_3_sld, self.gv_4_sld, self.gv_5_sld]
        self.file_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.init_axis()
        self.export_gbx.setVisible(False)
        self.label_file_dck.setTitleBarWidget(QWidget())
        
        self.img_size_edt = [self.linea_w_edt, self.linea_h_edt, self.bscan_w_edt, self.bscan_h_edt, self.cscan_w_edt, self.cscan_h_edt]
        self.ch_sel_edts = [self.sel_ch_edt, self.sel_dp_edt]
        
    def init_axis(self):
        self.axises = {'dist':[self.bX_wdg, self.tX_wdg, self.rX_wdg], 'ns':[self.bY_wdg, self.fY_wdg, self.aY_wdg],
                        'road width':self.rY_wdg, 'fscan width': self.fX_wdg, 'tscan width': self.tY_wdg,'ascan width': self.aX_wdg}

        for ax in self.axises['dist']: ax.setAxis(9, False) # 수평축
        for ax in self.axises['ns']: ax.setAxis(9, True) # 수직축
        self.axises['fscan width'].setAxis(5, False); self.axises['tscan width'].setAxis(5, True)
        self.axises['road width'].setAxis(5, True); self.axises['ascan width'].setAxis(3, False)

    def slot_connect(self):
        self.export_btn.pressed.connect(self.export)
        self.spl_main.resized.connect(self.spl_resized)# 초기 spl 모양 설정
        for spl in self.hor_spls: spl.splitterMoved.connect(self.spl_moved)
        for w in self.road_ctrl: w.valueChanged.connect(self.load_scans)
        for w in self.gpr_ctrl: w.valueChanged.connect(self.gvsChanged)
        for sc, canvas in enumerate(self.gpr_canvas):
            canvas.scan_class = sc; canvas.ch_changed.connect(self.chPointChanged)
            if sc!=F: canvas.scrollRequest.connect(self.setPosEdit)
            
        self.hh_rdo.toggled.connect(self.load_scans)# 라트로 모드 한덩
        
        self.actionOpen_dir.triggered.connect(self.openDir)
        self.actionInit_Separator.triggered.connect(self.init_canvas_split)
        self.actionClose.triggered.connect(self.close_file)
        self.file_tree.clicked.connect(self.fileTreeItemSelected)
        # self.actionexport_Images.triggered.connect(self.exportDataSet)
        
        # 거리 조정 관련
        self.line_canvas.scrollRequest.connect(self.setPosEdit)
        self.first_btn.clicked.connect(lambda: self.setPosEdit('first'))
        self.prev_btn.clicked.connect(lambda: self.setPosEdit('prev'))
        self.next_btn.clicked.connect(lambda: self.setPosEdit('next'))
        self.last_btn.clicked.connect(lambda: self.setPosEdit('last'))
        
        self.x_pos_edt.valueChanged.connect(self.load_scans)
        self.x_len_spn.valueChanged.connect(self.load_scans)
        
    def closeEvent(self, event):
        settings = self.settings
        settings[SETTING_WIN_SIZE] = self.size(); settings[SETTING_WIN_POSE] = self.pos(); settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LAST_OPEN_DIR] = self.last_open_dir if self.last_open_dir and os.path.exists(self.last_open_dir) else ''
        settings[SETTING_LAST_EXPORT_DIR] = self.last_export_dir if self.last_export_dir and os.path.exists(self.last_export_dir) else ''
        settings.save()
        
if __name__ == "__main__" :
    app = QApplication(sys.argv); app.setApplicationName(__appname__)
    myWindow = MainWindow(); myWindow.show()
    app.exec_()