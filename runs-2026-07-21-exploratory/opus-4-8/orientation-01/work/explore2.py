import h5py, remfile, numpy as np, pandas as pd
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

c = DandiAPIClient()
d = c.get_dandiset('000021')
a = d.get_asset_by_path('sub-707296975/sub-707296975_ses-721123822.nwb')
url = a.get_content_url(follow_redirects=1, strip_query=True)
rf = remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5 = h5py.File(rf, 'r')

print('units/spike_times size:', h5['units/spike_times'].shape, h5['units/spike_times'].dtype)
print('units/spike_amplitudes size:', h5['units/spike_amplitudes'].shape)
print('waveform_mean:', h5['units/waveform_mean'].shape)
print('units/spike_times chunks:', h5['units/spike_times'].chunks, h5['units/spike_times'].compression)

io = NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile = io.read()

dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
print('\n--- drifting gratings ---')
print(dg[['start_time','stop_time','orientation','temporal_frequency','contrast','spatial_frequency']].head())
print('orientations:', sorted(pd.unique(dg['orientation'].astype(str))))
print('tfs:', sorted(pd.unique(dg['temporal_frequency'].astype(str))))
print('durations:', np.round((dg['stop_time']-dg['start_time']).describe(),3).to_dict())

sg = nwbfile.intervals['static_gratings_presentations'].to_dataframe()
print('\n--- static gratings ---')
print('orientations:', sorted(pd.unique(sg['orientation'].astype(str))))
print('sfs:', sorted(pd.unique(sg['spatial_frequency'].astype(str))))
print('phases:', sorted(pd.unique(sg['phase'].astype(str)))[:10])
print('durations:', np.round((sg['stop_time']-sg['start_time']).describe(),3).to_dict())

el = nwbfile.electrodes.to_dataframe()
print('\n--- electrodes locations ---')
print(el['location'].value_counts().head(25))
print('electrode index name:', el.index.name)
