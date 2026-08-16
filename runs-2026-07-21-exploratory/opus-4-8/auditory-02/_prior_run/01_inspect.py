import json, urllib.request
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap

assets = json.load(open('assets.json'))
a = assets[0]
# resolve to direct S3 url
req = urllib.request.Request(a['s3'], method='HEAD')
s3 = urllib.request.urlopen(req).url
print('S3:', s3)

cache = remfile.DiskCache('/tmp/remfile_cache')
f = remfile.File(s3, disk_cache=cache)
h5 = h5py.File(f, 'r')
io = NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile = io.read()
print(nwbfile)

print("\n=== TRIALS ===")
tr = nwbfile.trials.to_dataframe()
print(tr.columns.tolist())
print(tr.head(10))
print(tr.describe(include='all'))
print("\n=== UNITS ===")
u = nwbfile.units
print(u.colnames)
udf = u.to_dataframe()
print(udf.drop(columns=[c for c in udf.columns if 'spike_times' in c or 'waveform' in c]).head(20))
print('n units', len(udf))
print("\n=== spontaneous ===")
print(nwbfile.intervals['spontaneous_blocks'].to_dataframe())
print("\n=== behavior ===")
print(nwbfile.processing['behavior'])
