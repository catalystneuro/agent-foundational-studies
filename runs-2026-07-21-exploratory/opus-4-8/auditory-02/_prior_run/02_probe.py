import json, urllib.request, numpy as np
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap

assets = json.load(open('assets.json'))
a = assets[0]
s3 = urllib.request.urlopen(urllib.request.Request(a['s3'], method='HEAD')).url
f = remfile.File(s3, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5 = h5py.File(f, 'r')
print('units keys:', list(h5['units'].keys()))
if 'general/extracellular_ephys' in h5:
    print('ephys:', list(h5['general/extracellular_ephys'].keys()))
io = NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile = io.read()
tr = nwbfile.trials.to_dataframe()
print('freqs:', np.unique(tr.stim_frequency.values), 'counts:', np.unique(tr.stim_frequency.values, return_counts=True)[1])
print('amps:', np.unique(tr.stim_amplitude.values))
print('ITI median', np.median(np.diff(tr.start_time.values)))

nwb = nap.NWBFile(nwbfile)
print(nwb)
units = nwb['units']
print(type(units), len(units))
print(units.get_info('rate') if 'rate' in units.metadata_columns else units)
pupil = nwb['PupilTracking'] if 'PupilTracking' in nwb.keys() else None
print('behavior keys', [k for k in nwb.keys()])
