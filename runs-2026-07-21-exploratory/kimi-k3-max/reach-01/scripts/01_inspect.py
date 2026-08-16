import remfile, h5py
from pynwb import NWBHDF5IO

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"D  {name}  shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"G  {name}")

h5py_file.visititems(visit)
