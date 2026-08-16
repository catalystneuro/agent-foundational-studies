import remfile, h5py, numpy as np, pickle, os, sys
from tqdm import tqdm

ASSET = sys.argv[1] if len(sys.argv)>1 else "58703c97-c0a9-4736-b684-73c85c1a444a"
OUT = sys.argv[2] if len(sys.argv)>2 else "cache_715093703.pkl"
url = f"https://api.dandiarchive.org/api/dandisets/000021/versions/draft/assets/{ASSET}/download/"
h = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache')), 'r')

def s(a): return np.array([x.decode() if isinstance(x,(bytes,np.bytes_)) else x for x in a])

e = h['general/extracellular_ephys/electrodes']
chan2loc = dict(zip(e['id'][:], s(e['location'][:])))

u = h['units']
qual = s(u['quality'][:])
reg = np.array([chan2loc.get(p,'?') for p in u['peak_channel_id'][:]])
uid = u['id'][:]
snr = u['snr'][:]; iso = u['isolation_distance'][:]; isiv = u['isi_violations'][:]
amp_cut = u['amplitude_cutoff'][:]; pres = u['presence_ratio'][:]
wdur = u['waveform_duration'][:]

TARGET = ['VISp','VISl','VISrl','VISam','VISpm','LGd']
sel = (qual=='good') & np.isin(reg, TARGET) & (isiv<0.5) & (amp_cut<0.1) & (pres>0.9)
print('selected units:', sel.sum(), {r:int((reg[sel]==r).sum()) for r in TARGET})

idx = u['spike_times_index'][:]
starts = np.concatenate([[0], idx[:-1]])
st_ds = u['spike_times']
spikes = {}
for i in tqdm(np.where(sel)[0], desc='spike_times'):
    spikes[int(uid[i])] = st_ds[starts[i]:idx[i]]

meta = dict(unit_id=uid[sel], region=reg[sel], snr=snr[sel], isolation=iso[sel],
            isi_viol=isiv[sel], waveform_duration=wdur[sel])

stim = {}
for name in ['drifting_gratings_presentations','static_gratings_presentations']:
    g = h['intervals'][name]
    d = {}
    for c in g.keys():
        if c in ('tags','tags_index','timeseries','timeseries_index','id'): continue
        arr = g[c][:]
        d[c] = s(arr) if arr.dtype.kind in 'SO' else arr
    stim[name] = d

# running speed
rs = None
for path in ['processing/running/running_speed','processing/running/running_speed/running_speed']:
    if path in h:
        node = h[path]
        if 'data' in node:
            rs = dict(t=node['timestamps'][:], v=node['data'][:])
        break
print('running speed:', None if rs is None else rs['v'].shape)

pickle.dump(dict(meta=meta, spikes=spikes, stim=stim, running=rs,
                 session_id=h['general/session_id'][()].decode()), open(OUT,'wb'))
print('saved', OUT, os.path.getsize(OUT)/1e6, 'MB')
