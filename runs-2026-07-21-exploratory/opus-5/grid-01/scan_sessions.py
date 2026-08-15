"""Scan all 000582 assets: unit count, session duration, position extent."""
import h5py, remfile, json, warnings
import numpy as np
from pynwb import NWBHDF5IO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
warnings.filterwarnings('ignore')

assets = json.load(open('assets.json'))
DC = remfile.DiskCache('/tmp/remfile_cache')

def url_of(a):
    return f"https://api.dandiarchive.org/api/dandisets/000582/versions/draft/assets/{a['id']}/download/"

def scan(a):
    h5 = h5py.File(remfile.File(url_of(a), disk_cache=DC), 'r')
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwb = io.read()
    pos = nwb.processing['behavior']['Position']
    keys = list(pos.spatial_series.keys())
    ss = pos.spatial_series[keys[0]]
    xy = ss.data[:]
    t = ss.timestamps[:]
    u = nwb.units
    n = len(u)
    counts = [len(u['spike_times'][i]) for i in range(n)]
    dur = float(t[-1] - t[0])
    out = dict(path=a['path'], subject=a['path'].split('/')[0], n_units=n,
               dur=dur, leds=keys,
               xrange=[float(np.nanmin(xy[:,0])), float(np.nanmax(xy[:,0]))],
               yrange=[float(np.nanmin(xy[:,1])), float(np.nanmax(xy[:,1]))],
               frac_nan=float(np.isnan(xy).any(axis=1).mean()),
               rates=[c/dur for c in counts])
    io.close(); h5.close()
    return out

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(tqdm(ex.map(scan, assets), total=len(assets)))
json.dump(results, open('session_scan.json','w'), indent=1)

import collections
print('total units', sum(r['n_units'] for r in results))
print('LED configs', collections.Counter(tuple(r['leds']) for r in results))
print('durations', np.percentile([r['dur'] for r in results],[0,50,100]))
xs = np.array([r['xrange'] for r in results]); ys=np.array([r['yrange'] for r in results])
print('x extent', np.percentile(xs[:,1]-xs[:,0],[0,50,100]))
print('y extent', np.percentile(ys[:,1]-ys[:,0],[0,50,100]))
print('nan frac', np.percentile([r['frac_nan'] for r in results],[0,50,90,100]))
