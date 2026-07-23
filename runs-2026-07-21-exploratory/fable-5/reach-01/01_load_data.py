"""Load and inspect the MC_Maze NWB file (DANDI 000128) via streaming."""
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

print("session_description:", nwbfile.session_description)
print("identifier:", nwbfile.identifier)
print("subject:", nwbfile.subject)
print("institution:", nwbfile.institution)
print("lab:", nwbfile.lab)
print("keywords:", nwbfile.keywords[:] if nwbfile.keywords is not None else None)
print()
print("=== acquisition ===", list(nwbfile.acquisition))
print("=== processing ===", list(nwbfile.processing))
for name, mod in nwbfile.processing.items():
    print(f"  {name}: {list(mod.data_interfaces)}")
    for dname, di in mod.data_interfaces.items():
        print("    ", dname, type(di).__name__)
        if hasattr(di, "time_series"):
            for ts_name, ts in di.time_series.items():
                print("       ", ts_name, ts.data.shape, getattr(ts, "unit", None),
                      "rate=", getattr(ts, "rate", None))
print()
print("=== units ===")
print(nwbfile.units.colnames, "n=", len(nwbfile.units))
print()
print("=== trials ===")
print(nwbfile.trials.colnames)
print("n trials:", len(nwbfile.trials))
tdf = nwbfile.trials.to_dataframe()
print(tdf.head(10).to_string())
print()
for c in tdf.columns:
    v = tdf[c]
    if v.dtype == object:
        print(c, "object; example:", np.asarray(v.iloc[0]).shape if hasattr(v.iloc[0], "__len__") else v.iloc[0])
    else:
        print(c, v.dtype, "unique:", v.nunique(), "min/max:", np.nanmin(v.values), np.nanmax(v.values))
