import json, numpy as np, h5py, remfile
from pynwb import NWBHDF5IO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import grid_lib as gl
scan = gl.list_assets()
def ext(s):
    nwb = gl.open_nwb(s['asset_id'])
    ss = nwb.processing['behavior']['Position'].spatial_series['SpatialSeriesLED1']
    d = np.asarray(ss.data[:])
    d = d[np.isfinite(d).all(1)]
    lo = np.percentile(d, 0.5, axis=0); hi = np.percentile(d, 99.5, axis=0)
    return dict(path=s['path'], w=float(hi[0]-lo[0]), h=float(hi[1]-lo[1]),
                cx=float((hi[0]+lo[0])/2), cy=float((hi[1]+lo[1])/2), unit=ss.unit)
with ThreadPoolExecutor(8) as ex:
    r = list(tqdm(ex.map(ext, scan), total=len(scan)))
json.dump(r, open('session_extent.json','w'), indent=1)
w = np.array([x['w'] for x in r]); h = np.array([x['h'] for x in r])
print('width  pct', np.percentile(w,[0,5,25,50,75,95,100]).round(1))
print('height pct', np.percentile(h,[0,5,25,50,75,95,100]).round(1))
print('centers x', np.percentile([x['cx'] for x in r],[0,50,100]).round(1), 'y', np.percentile([x['cy'] for x in r],[0,50,100]).round(1))
print('n sessions with w>120:', (w>120).sum(), ' w<80:', (w<80).sum())
