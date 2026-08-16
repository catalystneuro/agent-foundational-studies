import h5py, numpy as np, remfile, requests
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
dl = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
r = requests.get(dl, allow_redirects=False, timeout=120)
s3_url = r.headers["Location"]
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("states:", nwb["states"].label)
st = nwb["states"]
for lab in np.unique(st.label):
    m = np.asarray(st.label) == lab
    print(lab, "intervals:", int(m.sum()), "tot s:", float((np.asarray(st.end)[m]-np.asarray(st.start)[m]).sum()))

# LFP rate attribute
ds = h5py_file["processing/ecephys/LFP/LFP/starting_time"]
print("LFP rate attr:", ds.attrs.get("rate"), "conversion:", h5py_file["processing/ecephys/LFP/LFP/data"].attrs.get("conversion"))
print("LFP shape:", h5py_file["processing/ecephys/LFP/LFP/data"].shape)
units = nwb["units"]
ct = units.get_info("cell_type")
print("cell_type counts:", ct.value_counts().to_dict() if ct is not None else None)
print("n units:", len(units))
pos = nwb["1.6mLinearMazeSpatialSeries"]
pt = np.asarray(pos.t)
print("pos n:", len(pt), "median dt:", np.median(np.diff(pt)))
