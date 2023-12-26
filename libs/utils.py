import  json, pickle, os

SETTING_WIN_SIZE = 'window/size'; SETTING_WIN_POSE = 'window/position'; SETTING_WIN_STATE = 'window/state'
SETTING_LAST_OPEN_DIR = 'lastOpenDir'; SETTING_LAST_EXPORT_DIR = 'lastExportDir'
B, F, T = range(3)

class Settings(object):
    def __init__(self): self.data = {}; self.path = 'windowSettings.pkl'
    def __setitem__(self, key, value): self.data[key] = value
    def __getitem__(self, key): return self.data[key]
    def get(self, key, default=None): return self.data[key] if key in self.data else default
    def save(self): f=open(self.path, 'wb'); pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL); f.close()
    def load(self):
        try: f=open(self.path, 'rb'); self.data=pickle.load(f); f.close(); return True
        except: return False

    def reset(self):
        if os.path.exists(self.path): os.remove(self.path)
        self.data = {}; self.path = None
        
def saveJson(save_path, data, multi_line=True):
    try:
        with open(save_path, 'w') as fp:
            if multi_line: json.dump(data, fp, indent=2)
            else: json.dump(data, fp)
        return True
    except:
        return False

def loadJson(json_path):
    try: 
        with open(json_path, 'r') as fp: json_data = json.load(fp)
    except: 
        json_data = None
    
    return json_data