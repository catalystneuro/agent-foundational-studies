import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
c = DandiAPIClient(); d = c.get_dandiset('000021')
a = d.get_asset_by_path('sub-726162193/sub-726162193_ses-750749662.nwb')
url = a.get_content_url(follow_redirects=1, strip_query=True)
rf = remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5 = h5py.File(rf,'r'); io = NWBHDF5IO(file=h5, load_namespaces=True); nwbfile = io.read()
r = nwbfile.processing['running']
print(list(r.data_interfaces))
for k,v in r.data_interfaces.items():
    print(k, type(v).__name__, getattr(v,'data',None).shape if hasattr(v,'data') else '')
    print('   ts:', v.timestamps.shape if v.timestamps is not None else 'rate', v.unit)
ut = nwbfile.units
print('spike_times row 0:', ut['spike_times'][0][:5], len(ut['spike_times'][0]))
print('id dtype', ut.id[:5])
import time
t0=time.time()
st = ut['spike_times'][10]
print('one unit read', time.time()-t0, len(st))
print('cols quality sample:', ut['quality'].data[:5], ut['snr'].data[:5])
