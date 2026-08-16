import h5py, remfile, numpy as np, pynapple as nap
from pynwb import NWBHDF5IO

ASSET = "56d2e2d7-ba41-40a1-b017-b81871f3f0c3"
URL = f"https://api.dandiarchive.org/api/dandisets/000056/versions/draft/assets/{ASSET}/download/"
rem_file = remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

u = nwbfile.units
print("units columns:", u.colnames, "| all cols:", [c.name for c in u.columns])
print("electrodes cols:", nwbfile.electrodes.colnames)
edf = nwbfile.electrodes.to_dataframe()
print(edf[["location", "group_name"]].drop_duplicates().to_string())
print("\nelectrode_groups:")
for k, g in nwbfile.electrode_groups.items():
    print("  ", k, "|", g.location, "|", g.description)
print("\ndevices:", list(nwbfile.devices))

print("\nraw h5 units keys:", list(h5f["units"].keys()))
print("raw h5 /:", list(h5f.keys()))
print("general:", list(h5f["general"].keys()))
if "subject" in h5f["general"]:
    print("subject keys:", list(h5f["general"]["subject"].keys()))

pos = nwbfile.processing["behavior"]["SubjectPosition"]
for k, s in pos.spatial_series.items():
    print(k, s.data.shape, s.unit, "rate" if s.rate else "ts", s.timestamps[:3] if s.timestamps is not None else s.starting_time)

st = nwbfile.processing["behavior"]["states"].to_dataframe()
print(st.head(20)); print(st["label"].value_counts() if "label" in st else st.columns)

nwb = nap.NWBFile(nwbfile)
print(nwb["units"])
print(nwb["states"])
