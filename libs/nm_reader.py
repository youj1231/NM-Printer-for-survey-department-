import os.path as osp, numpy as np, os, re, cv2, qimage2ndarray, copy, json
from PIL import Image as pimg
from PyQt5.QtGui import QPixmap
from scipy.interpolate import interp1d

dirlist = os.listdir
NM1, NM2, NM3, LT1, LT2 = range(5)
NM_NAMES = ["NM-1", "NM-2", "NM-3", "LATRO-1", "LATRO-2"]

class NM_Reader(object):
    linea_scale = 8

    def __init__(self):
        self.cur_id = None; self.chs = (0,0,0)
        self.gpr_list, self.linea_list = {}, {}
        with open('nmsettings.json', 'r') as fp: self.settings = json.load(fp)
    
    def clear(self):
        self.close_file()
        self.gpr_list, self.linea_list = {}, {}
        self.gpr_shapes, self.linea_shapes = {}, {}
    
    def set_gvs(self, gvs):
        gv_x = [0, 127, 255, 383, 511]; gv_y = [1.06**gv for gv in gvs]
        self.gvs = interp1d(gv_x, gv_y, kind='linear')
    
    def rpsnamepattern(self, rps_name):
        if not rps_name.endswith('.rps'): return None
        _dt_ = re.search(r'_\d{4}-\d{2}-\d{2}_\(\d{2}_\d{2}_\d{2}\)_', rps_name)
        if not _dt_ : return None
        
        _dt_start, _dt_end, _dt_ = _dt_.start(), _dt_.end(), _dt_.group()
        r_mode = rps_name[:_dt_start]
        cmnt_idx = rps_name[_dt_end:-4]
        splt = cmnt_idx.split('_'); splt_n = len(splt)
        if splt_n<2: return None

        try: idx = int(splt[-1])
        except: return None
        cmnt = '_'.join(splt[:-1]); r_id = _dt_[1:]+cmnt
        
        return r_mode, r_id, idx
    
    def rps_select(self, dir_path, tmds):
        pttns = [self.rpsnamepattern(r) for r in dirlist(dir_path)]
        pttns = [p for p in pttns if p is not None]
        mds = set([p[0] for p in pttns]); tmd = None
        for md in tmds:
            if md in mds: tmd = md; break
        if tmd is None: return {}
        
        id2n = {}
        for _, rid, rdx in [p for p in pttns if p[0]==tmd]:
            id2n[rid] = max(id2n[rid], rdx) if rid in id2n else rdx
            
        return {rid:[osp.join(dir_path, '%s_%s_%d.rps'%(tmd, rid, i)) for i in range(id2n[rid]+1)] for rid in id2n.keys()}
        
    def open_dir(self, dir_path):
        def valid_dir(d): return osp.exists(d) and osp.isdir(d)
        def list_shape(rps_list):
            n = len(rps_list)
            with open(rps_list[0], 'rb') as fp: fp.seek(24); fst_n, w, h = np.fromfile(fp, 'uint32', 3)
            if n==1: return fst_n, w, h
            with open(rps_list[-1], 'rb') as fp: fp.seek(24); lst_n = np.fromfile(fp, 'uint32', 1)[0]
            return int(fst_n*(n-1)+lst_n), w, h
        
        self.clear()
        # 폴더 검색
        gdir = osp.join(dir_path, 'GPR'); ldir = None
        for d in [osp.join(dir_path, l) for l in ('Linea', 'LineScan')]: 
            if valid_dir(d): ldir = d; break
        if not (valid_dir(gdir) and ldir is not None): return
        # 파일 딕셔너리 생성 & 프레임 검사
        id2gpr = self.rps_select(gdir, ['FINAL_GPR', 'GPR']); id2linea = self.rps_select(ldir, ['Linea', 'Line'])
        id2gshape = {rid:list_shape(id2gpr[rid]) for rid in id2gpr.keys()}
        gpr_list = {rid:id2gpr[rid] for rid in id2linea.keys() if rid in id2gpr and id2gshape[rid][0]>0}
        gpr_shapes = {rid: id2gshape[rid] for rid in gpr_list.keys()}
        linea_list = {rid:id2linea[rid] for rid in gpr_list.keys() if rid in id2linea}
        linea_shapes = {rid:list_shape(linea_list[rid]) for rid in linea_list}
        if not (gpr_list and linea_list) : return
        
        # nm_type 검사
        rid = list(gpr_list.keys())[0]
        chs, samples = gpr_shapes[rid][1:]; lw, lh = linea_shapes[rid][1:]
        if samples!=512: return
        nm = NM2 if chs==46 else NM3 if chs==22 else None
        if nm is None and chs==32: nm = NM1 if lw==4096 else LT1 if lh==512 else LT2 if lh==128 else None
        if nm is None: return
        
        self.gpr_list, self.gpr_shapes = gpr_list, gpr_shapes
        self.linea_list, self.linea_shapes = linea_list, linea_shapes
        # nm_type 설정
        self.nm_type = nm; cur_setting = self.settings[NM_NAMES[nm]]#; self.default_exp_setting = cur_setting['export']
        self.dx=cur_setting['dx']; self.gab = cur_setting['gab']
        self.road_width = cur_setting['road width']; self.ant_width = cur_setting['antena width']
        
    def ids(self): return list(self.gpr_list.keys())
    def gpr_shape(self): return self.gpr_shapes[self.cur_id] if self.cur_id is not None else None
    
    def open_file(self, rid):
        self.close_file()
        if not rid in self.ids(): return
        
        # gpr_offset 설정
        gpr_fps = [open(f, 'rb') for f in self.gpr_list[rid]]
        linea_fps = [open(f, 'rb') for f in self.linea_list[rid]]
        # max_frm 설정
        gpr_fps[0].seek(16); idx_off = np.fromfile(gpr_fps[0], 'uint64', 1)[0]
        n, w, h = np.fromfile(gpr_fps[0], 'uint32', 3); frm_size = w*h+6
        self.gfrm_off = int(idx_off-n*frm_size*2); self.gpr_fst_n = n
        linea_fps[0].seek(24); self.linea_fst_n = np.fromfile(linea_fps[0], 'uint32', 1)[0]

        # linea 이미지 배열 생성
        frm_n, frm_w, frm_h = self.linea_shapes[rid]
        self.sg_w, sg_h = frm_h//self.linea_scale, frm_w//self.linea_scale
        self.linea_image = np.zeros((sg_h, self.sg_w*frm_n, 3), 'float16')
        self.linea_loaded = np.zeros((frm_n,), 'uint8'); self.sg_idx = []
        for fp in linea_fps:
            fp.seek(16); idx_off = np.fromfile(fp, 'uint64', 1)[0]; n = np.fromfile(fp, 'uint32', 1)[0]
            fp.seek(idx_off); self.sg_idx.append(np.fromfile(fp, 'uint64', n)+12)
            
        # gpr 설정
        self.cur_id = rid; self.gpr_cache = (0, 0, None)
        self.gpr_fps, self.linea_fps = gpr_fps, linea_fps
        
    def close_file(self):
        if self.cur_id is None: return
        for fp in self.gpr_fps: fp.close()
        for fp in self.linea_fps: fp.close()
        
        self.gpr_fps, self.linea_fps = [],[]
        self.linea_image, self.linea_loaded = None, None
        self.sg_idx = None
        self.gpr_cache = (0, 0, None); self.chs = (0, 0, 0)
        self.cur_id = None
        
    def load_gpr(self, fs, fe):
        cfs, cfe, cgpr = self.gpr_cache
        _, w, h = self.gpr_shape(); gpr = np.zeros((fe-fs, h, w), 'float32')
        ccfs, ccfe = max(fs, cfs), min(fe, cfe)
        if ccfs<ccfe: gpr[ccfs-fs:ccfe-fs] = cgpr[ccfs-cfs:ccfe-cfs]
        
        for frm_id in range(fs, fe):
            if ccfs<=frm_id<ccfe: continue
            fp_id = frm_id//self.gpr_fst_n; fp = self.gpr_fps[fp_id]
            frm_num = frm_id%self.gpr_fst_n
            fp.seek(int(self.gfrm_off+frm_num*(w*h*2+12)+12))
            gpr[frm_id-fs] = np.fromfile(fp, 'uint16', w*h).reshape((h, w)).astype('float32')-2**15
                
        self.gpr_cache = (fs, fe, gpr)
        
    def load_seg(self, seg_id):
        fp_id = seg_id//self.linea_fst_n; fp = self.linea_fps[fp_id]
        sg_off = self.sg_idx[fp_id][seg_id%self.linea_fst_n]; fp.seek(sg_off)
        
        seg = np.fromfile(fp, 'uint8', np.fromfile(fp, 'uint32', 1)[0])
        seg = cv2.cvtColor(cv2.imdecode(seg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        seg = cv2.resize(seg, (self.linea_image.shape[0], self.sg_w)).transpose((1,0,2))
        
        l_s = seg_id*self.sg_w; l_e = l_s+self.sg_w
        self.linea_image[:, l_s:l_e] = seg.astype('float16'); self.linea_loaded[seg_id] = 1
    
    def scan2pixmap(self, scan, gpr=True):
        if gpr: qimg = qimage2ndarray.gray2qimage(scan, (-5000, 5000))
        else: qimg = qimage2ndarray.array2qimage(scan, (0, 255))
        return QPixmap.fromImage(qimg)
    
    def road_image(self, ms_me, br_cts, raw=False):
        if self.cur_id is None : return
        n = self.linea_image.shape[1]; dx = self.gpr_shape()[0]*self.dx/n
        pmin, pmax = [int(round((m+self.gab)/dx)) for m in ms_me]
        ls, le = max(0, pmin), min(n-1, pmax)
        if ls>=le: return None
        
        for seg_id in range(ls//self.sg_w, le//self.sg_w+1):
            if not self.linea_loaded[seg_id]: self.load_seg(seg_id)
        
        raw_scan = self.linea_image[:, ls:le]
        br, cts = br_cts; cts = 1.015**cts; scan_mean = raw_scan.mean()
        scan = raw_scan-scan_mean; scan = scan*cts; scan = scan+scan_mean+br
        if raw: return pmin, pmax, ls, self.scan2pixmap(scan, False), raw_scan
        else: return pmin, pmax, ls, self.scan2pixmap(scan, False)
    
    def gpr_scans(self, ms_me, tr, lt_mode = 'vv'):
        if self.cur_id is None: return
        pmin, pmax = [int(round(m/self.dx)) for m in ms_me]
        fs, fe = max(0, pmin), min(self.gpr_shape()[0], pmax)
        if fs>fe: return None
        
        self.load_gpr(fs, fe); gpr = self.gpr_cache[2]
        n, h, w = gpr.shape; h = int(round(h*tr))
        b, f, t = self.chs; f = min(fe-1, max(fs, f)); self.chs = (b, f, t)
        vv = True if lt_mode=='vv' else False
        
        # 라트로모드 조정 관련
        if self.nm_type>=LT1: b=b if vv else b+16
        
        scans = (gpr[f-fs, :h, b]/5000, gpr[:, :h, b].T, gpr[f-fs, :h, :], gpr[:, t, :].T)
        ascan, bscan, fscan, tscan = copy.deepcopy(scans)
        
        if self.nm_type>=LT1:
            fscan = fscan[:, :16] if vv else fscan[:, 16:]; tscan = tscan[:16] if vv else tscan[16:]; w = w//2
        
        # 게인값 설정
        for d in range(h): gv = self.gvs(d); bscan[d], fscan[d] = bscan[d]*gv, fscan[d]*gv
        tscan = tscan*self.gvs(t)
        
        bscan, fscan, tscan = self.scan2pixmap(bscan), self.scan2pixmap(fscan), self.scan2pixmap(tscan)
        return ascan, (pmin, pmax, fs, bscan), (0, fscan.width(), 0, fscan), (pmin, pmax, fs, tscan)
    
    #####################################################
    ########### 여기서부터 이미지 출력 관련 ###############
    #####################################################
    
    def linea_full_scan(self, fps, dsize, br_cts):
        segs = []
        for fp in fps: 
            fp.seek(16); off=np.fromfile(fp, 'uint64', 1)[0]
            n, w, h = np.fromfile(fp, 'uint32', 3)
            fp.seek(off); idx_list = np.fromfile(fp, 'uint64', n)+12
            for idx in idx_list:
                fp.seek(idx); seg = np.fromfile(fp, 'uint8', np.fromfile(fp, 'uint32', 1)[0])
                seg = cv2.cvtColor(cv2.imdecode(seg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                segs.append(cv2.resize(seg, (w//16, h//16)))
        
        scan = np.concatenate(segs).transpose((1, 0, 2))
        scan = cv2.resize(scan, dsize).astype('float32')
        br, cts = br_cts; cts = 1.015**cts; scan_mean = scan.mean()
        scan = scan-scan_mean; scan = scan*cts; scan = scan+scan_mean+br
        return scan.clip(0, 255).astype('uint8')
    
    def gpr_full_scans(self, fps, chs, dps, bdsize, cdsize, vv=True):
        ch2scans, dp2scans = {ch:[] for ch in chs}, {dp:[]for dp in dps}
        for fp in fps:
            fp.seek(16); idx_off = np.fromfile(fp, 'uint64', 1)[0]
            n, w, h = np.fromfile(fp, 'uint32', 3); frm_size = w*h+6
            fp.seek(int(idx_off-n*frm_size*2)); gpr = np.fromfile(fp, 'uint16', n*frm_size)
            gpr = gpr.reshape((n, frm_size))[:, 6:].reshape((n, h, w)).astype('float32')-2**15
            if self.nm_type>=LT1 and vv: gpr = gpr[:, :, :16]
            elif self.nm_type>=LT1 and not vv: gpr = gpr[:, :, 16:]
            for ch in chs: ch2scans[ch].append(gpr[:,:,ch].T)
            for dp in dps: dp2scans[dp].append(gpr[:,dp,:].T)
        
        for ch in chs: 
            scan = np.concatenate(ch2scans[ch], axis=1)
            for d in range(512): scan[d] = scan[d]*self.gvs(d)
            scan = qimage2ndarray.raw_view(qimage2ndarray.gray2qimage(scan, (-5000, 5000)))
            ch2scans[ch] = cv2.resize(scan, bdsize)
            
        for dp in dps: 
            scan = np.concatenate(dp2scans[dp], axis=1)*self.gvs(dp)
            scan = qimage2ndarray.raw_view(qimage2ndarray.gray2qimage(scan, (-5000, 5000)))
            dp2scans[dp] = cv2.resize(scan, cdsize)
            
        return ch2scans, dp2scans
    
    def seg_split(self, total_scan, seg_len, gpr=True):
        h, n = total_scan.shape[:2]
        last_len = n%seg_len; n_seg = n//seg_len if last_len==0 else n//seg_len+1

        segs = []
        for i in range(n_seg):
            sp = i*seg_len; ep = sp+seg_len
            if ep<=n: seg = total_scan[:, sp:ep]
            else:
                lseg = total_scan[:, sp:]
                rseg = np.zeros((h, ep-n), 'uint8') if gpr else np.zeros((h, ep-n, 3), 'uint8')
                seg = np.concatenate((lseg, rseg), axis=1)
            segs.append(seg)
            
        return segs
    
    def save_scan(self, scan, save_path, dpi):
        save_path = save_path+'.jpg'; dir_path = osp.split(save_path)[0]
        if not osp.exists(dir_path): os.makedirs(dir_path, exist_ok=True)
        
        img = pimg.fromarray(scan); img.info['dpi']=(dpi, dpi)
        img.save(save_path, dpi=(dpi, dpi))
        
    def export_file(self, rid, save_dir, exp_opt, br_cts, dpi=254):
        chs, dps = exp_opt['channels, depths']; img_len = exp_opt['dist']; vv = exp_opt['latro mode']
        
        segs_size = [[int(round(wh*dpi/2.54)) for wh in s] for s in exp_opt['print size']]
        gn = self.gpr_shapes[rid][0]; dist = gn*self.dx
        rest_len = dist%img_len; n_term = dist//img_len; fn = n_term+rest_len/img_len
        ldsize, bdsize, tdsize = [(int(round(s[0]*fn)), s[1]) for s in segs_size]
        
        if rid==self.cur_id: l_fps, g_fps = self.linea_fps, self.gpr_fps
        else:
            l_fps = [open(f, 'rb') for f in self.linea_list[rid]]
            g_fps = [open(f, 'rb') for f in self.gpr_list[rid]]
        
        line_scans = self.seg_split(self.linea_full_scan(l_fps, ldsize, br_cts), segs_size[0][0], False)
        ch2scans, dp2scans = self.gpr_full_scans(g_fps, chs, dps, (bdsize), (tdsize), vv)
        ch2scans = {ch:self.seg_split(ch2scans[ch], segs_size[1][0]) for ch in chs}
        dp2scans = {dp:self.seg_split(dp2scans[dp], segs_size[2][0]) for dp in dps}
        
        for i in range(len(line_scans)):
            ms = i*img_len; me = ms+img_len; 
            msme = '%s-%s'%("{:05d}".format(int(ms)), "{:05d}".format(int(me)))
            path = osp.join(save_dir, rid, msme, 'road')
            self.save_scan(line_scans[i], path, dpi)
            
            pref = 'vv_' if vv and self.nm_type>=LT1 else 'hh_' if not vv and self.nm_type>=LT1 else ''
            for ch in chs:
                path = osp.join(save_dir, rid, msme, pref+"channel_{:02d}".format(ch))
                self.save_scan(ch2scans[ch][i], path, dpi)
            
            for dp in dps:
                path = osp.join(save_dir, rid, msme, pref+"layer_{:03d}".format(dp))
                self.save_scan(dp2scans[dp][i], path, dpi)
                
        if rid!=self.cur_id:
            for fp in l_fps: fp.close()
            for fp in g_fps: fp.close()