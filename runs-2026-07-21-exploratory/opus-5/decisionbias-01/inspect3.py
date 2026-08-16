import h5py, remfile, numpy as np, time
from pynwb import NWBHDF5IO
url='https://dandiarchive.s3.amazonaws.com/blobs/275/e0a/275e0aae-5c90-4637-afff-3944f319762c'
rf=remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5=h5py.File(rf,'r'); io=NWBHDF5IO(file=h5, load_namespaces=True); nwbfile=io.read()
u=nwbfile.units
print('quality', np.unique(np.round(u['ibl_quality_score'][:],3), return_counts=True))
print('ks label', np.unique(u['kilosort2_label'][:], return_counts=True))
print('fr', np.percentile(u['firing_rate'][:],[5,25,50,75,95]))
w=nwbfile.processing['wheel']
for k,v in w.data_interfaces.items(): print(k, type(v).__name__)
vel=w['WheelVelocitySmoothed']
print('vel', vel.data.shape, vel.rate if hasattr(vel,'rate') else None, vel.timestamps if vel.timestamps is None else vel.timestamps.shape)
mi=w['WheelMovementIntervals']
print(type(mi), mi.colnames if hasattr(mi,'colnames') else '')
t0=time.time()
st = u['spike_times'].target.data
print('total spikes', st.shape)
arr = st[:]
print('load spikes s', time.time()-t0)
