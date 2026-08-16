import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import pynapple as nap

path='sub-NYU-11/sub-NYU-11_ses-6713a4a7-faed-4df2-acab-ee4e63326f8d_desc-processed_behavior+ecephys.nwb'
c=DandiAPIClient(); ds=c.get_dandiset('000409')
a=next(ds.get_assets_by_glob(path))
url=a.get_content_url(follow_redirects=1, strip_query=True)
print(url)
rf=remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5=h5py.File(rf,'r')
io=NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile=io.read()
print(nwbfile.session_description)
print('--- intervals:', list(nwbfile.intervals.keys()))
tr=nwbfile.trials
print('trial cols:', tr.colnames)
print('n trials', len(tr))
print('--- units cols:', nwbfile.units.colnames)
print('n units', len(nwbfile.units))
print('--- acquisition:', list(nwbfile.acquisition.keys()))
print('--- processing:', {k:list(v.data_interfaces.keys()) for k,v in nwbfile.processing.items()})
