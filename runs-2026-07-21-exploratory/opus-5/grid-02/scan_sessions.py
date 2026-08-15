"""Scan all DANDI 000582 assets: how many units, position streams, duration."""
import json, h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

assets = json.load(open('/tmp/assets_582.json'))
BASE = 'https://api.dandiarchive.org/api/dandisets/000582/versions/draft/assets/%s/download/'
cache = remfile.DiskCache('/tmp/remfile_cache_grid')

def scan(a):
    f = remfile.File(BASE % a['asset_id'], disk_cache=cache)
    io = NWBHDF5IO(file=h5py.File(f, 'r'), load_namespaces=True)
    nwb = io.read()
    pos = nwb.processing['behavior']['Position']
    ss = pos.spatial_series[list(pos.spatial_series)[0]]
    t = ss.timestamps[:]
    u = nwb.units
    names = list(u['unit_name'].data[:]) if u is not None else []
    hist = list(u['histology'].data[:]) if u is not None else []
    nsp = [len(u['spike_times'][i]) for i in range(len(u))] if u is not None else []
    return dict(path=a['path'], asset_id=a['asset_id'], n_units=len(names),
                units=[str(x) for x in names], histology=[str(x) for x in hist], n_spikes=nsp,
                led_keys=list(pos.spatial_series), dur=float(t[-1] - t[0]), n_pos=len(t))

with ThreadPoolExecutor(8) as ex:
    res = list(tqdm(ex.map(scan, assets), total=len(assets)))
json.dump(res, open('session_scan.json', 'w'), indent=1)
print('total units', sum(r['n_units'] for r in res))
import collections
print(collections.Counter(h for r in res for h in r['histology']))
print(collections.Counter(tuple(r['led_keys']) for r in res))
print('durations', np.percentile([r['dur'] for r in res], [0,25,50,75,100]))
