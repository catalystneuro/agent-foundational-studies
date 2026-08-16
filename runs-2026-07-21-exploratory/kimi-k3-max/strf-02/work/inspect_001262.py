import h5py, remfile
from pynwb import NWBHDF5IO

url = "https://dandiarchive.s3.amazonaws.com/blobs/06c/4fb/06c4fbc1-6248-46d1-a215-5f8271229fed"
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5 = h5py.File(rem_file, "r")

def show(name, obj):
    print(name, '|', type(obj).__name__, getattr(obj, 'shape', ''), end='')
    if isinstance(obj, h5py.Dataset) and obj.shape == ():
        try: print(' =', obj[()], end='')
        except Exception: pass
    print()
h5.visititems(show)
