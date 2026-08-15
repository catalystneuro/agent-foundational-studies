"""Stream once from DANDI 000044 (Achilles session) and cache arrays locally."""
import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
import os
OUT='/tmp/replay_cache'; os.makedirs(OUT, exist_ok=True)
s3='https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028'
h5=h5py.File(remfile.File(s3, disk_cache=remfile.DiskCache('/tmp/remfile_cache')),'r')
nwb=NWBHDF5IO(file=h5, load_namespaces=True).read()

# epochs
ep=nwb.epochs.to_dataframe()
np.save(f'{OUT}/epochs.npy', ep[['start_time','stop_time','label']].to_records(index=False))
print('epochs saved'); print(ep)

# units: excitatory CA1
ct=nwb.units['cell_type'][:]; loc=nwb.units['location'][:]
st=nwb.units['spike_times']
n=len(nwb.units)
spikes={}
exc_idx=[]
for i in range(n):
    if str(ct[i])=='excitatory':
        spikes[i]=np.asarray(st[i])
        exc_idx.append(i)
np.save(f'{OUT}/spikes.npy', np.array(list(spikes.items()), dtype=object), allow_pickle=True)
np.save(f'{OUT}/unit_loc.npy', np.array([str(loc[i]) for i in exc_idx]))
print(f'saved {len(exc_idx)} excitatory units')

# linearized position
lin=nwb.processing['behavior'].data_interfaces['1.6mLinearMazeLinearizedPosition']
ss=list(lin.spatial_series.values())[0]
xl=np.asarray(ss.data[:]).ravel()
tl=ss.starting_time+np.arange(xl.size)/ss.rate
np.save(f'{OUT}/lin_t.npy', tl); np.save(f'{OUT}/lin_x.npy', xl)
# 2D position (for speed)
p2=nwb.processing['behavior'].data_interfaces['1.6mLinearMazePosition']
s2=list(p2.spatial_series.values())[0]
xy=np.asarray(s2.data[:])
np.save(f'{OUT}/pos2d.npy', xy)
print('position saved', xl.shape, xy.shape)

# LFP: best ripple channel = 2, plus channel 1 and 3 as backups; save full traces
es=list(nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series.values())[0]
fs=es.rate
for ch in [2,1,3]:
    print('reading LFP channel', ch)
    lfp=np.asarray(es.data[:,ch]).astype(np.int16)
    np.save(f'{OUT}/lfp_ch{ch}.npy', lfp)
np.save(f'{OUT}/lfp_fs.npy', np.array([fs]))
print('LFP saved, fs=', fs, 'n=', lfp.size)
