"""Check electrodes table / unit region info."""
import h5py
from pynwb import NWBHDF5IO
import remfile

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache_hd")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

print("=== electrodes ===")
df = nwbfile.electrodes.to_dataframe()
print(df.columns.tolist())
print(df.head(10))
print(df['location'].value_counts() if 'location' in df.columns else "no location col")
print()
print("=== units table columns ===")
print(nwbfile.units.colnames if nwbfile.units else "no units")
print()
print("=== general ===")
print("session:", nwbfile.identifier, nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id if nwbfile.subject else None)
print("devices:", list(nwbfile.devices.keys()))
for name, grp in nwbfile.electrode_groups.items():
    print("egroup:", name, "|", grp.description, "|", grp.location)
