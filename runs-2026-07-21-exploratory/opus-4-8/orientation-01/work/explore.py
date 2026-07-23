import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap
from dandi.dandiapi import DandiAPIClient

c = DandiAPIClient()
d = c.get_dandiset('000021')
a = d.get_asset_by_path('sub-707296975/sub-707296975_ses-721123822.nwb')
url = a.get_content_url(follow_redirects=1, strip_query=True)
print(url)

disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rf = remfile.File(url, disk_cache=disk_cache)
h5 = h5py.File(rf, 'r')
io = NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile = io.read()
print('=== session_description:', nwbfile.session_description)
print('=== intervals:')
for k, v in nwbfile.intervals.items():
    print('  ', k, len(v), list(v.colnames))
print('=== units colnames:', list(nwbfile.units.colnames))
print('=== n units:', len(nwbfile.units))
print('=== electrodes colnames:', list(nwbfile.electrodes.colnames))
print('=== acquisition:', list(nwbfile.acquisition))
print('=== processing:', list(nwbfile.processing))
