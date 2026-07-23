import remfile, h5py, numpy as np
url = "https://api.dandiarchive.org/api/dandisets/000021/versions/draft/assets/58703c97-c0a9-4736-b684-73c85c1a444a/download/"
h = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache')), 'r')
u = h['units']
print(list(u.keys()))
print('spike_times shape', u['spike_times'].shape, u['spike_times'].chunks)
e = h['general/extracellular_ephys/electrodes']
print(list(e.keys()))
loc = np.array([x.decode() if isinstance(x,bytes) else x for x in e['location'][:]])
ids = e['id'][:]
import collections
print(collections.Counter(loc).most_common(30))
pc = u['peak_channel_id'][:]
m = dict(zip(ids, loc))
ureg = np.array([m.get(p,'?') for p in pc])
qual = np.array([x.decode() if isinstance(x,bytes) else x for x in u['quality'][:]])
print(collections.Counter(ureg[qual=='good']).most_common(20))
