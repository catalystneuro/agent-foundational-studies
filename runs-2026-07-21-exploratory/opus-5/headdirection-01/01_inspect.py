"""Inspect the structure of a Dandiset 000056 session."""
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

S3 = "https://dandiarchive.s3.amazonaws.com/blobs/417/5cc/4175cc90-74f1-4af7-b0b9-e13e0c39f6b8"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rf = remfile.File(S3, disk_cache=disk_cache)
h5 = h5py.File(rf, "r")
io = NWBHDF5IO(file=h5, load_namespaces=True)
nwbfile = io.read()

print("=" * 70)
print("session_description:", nwbfile.session_description)
print("identifier:", nwbfile.identifier)
print("session_start_time:", nwbfile.session_start_time)
print("subject:", nwbfile.subject)
print("=" * 70)
print("acquisition:", list(nwbfile.acquisition.keys()))
print("processing:", list(nwbfile.processing.keys()))
for name, mod in nwbfile.processing.items():
    print(f"  [{name}]", list(mod.data_interfaces.keys()))
    for dname, di in mod.data_interfaces.items():
        print("    ", dname, type(di).__name__)
        if hasattr(di, "spatial_series"):
            for sname, ss in di.spatial_series.items():
                print("       spatial_series:", sname, ss.data.shape, ss.unit,
                      "rate" if ss.rate else "timestamps")
        if hasattr(di, "time_series"):
            for sname, ss in di.time_series.items():
                print("       time_series:", sname, ss.data.shape, ss.unit)
print("=" * 70)
print("intervals:", list(nwbfile.intervals.keys()) if nwbfile.intervals else None)
if nwbfile.intervals:
    for k, v in nwbfile.intervals.items():
        print(f"  {k}: n={len(v)} cols={v.colnames}")
        print(v.to_dataframe().head(20))
print("=" * 70)
print("units colnames:", nwbfile.units.colnames)
print("n units:", len(nwbfile.units))
udf = nwbfile.units.to_dataframe()
print(udf.drop(columns=[c for c in udf.columns if "spike_times" in c]).head(20))
print("=" * 70)
print("electrodes cols:", nwbfile.electrodes.colnames)
print(nwbfile.electrodes.to_dataframe().head())
print("regions:", nwbfile.electrodes.to_dataframe()["location"].unique()
      if "location" in nwbfile.electrodes.colnames else "n/a")
print("=" * 70)
nwbnap = nap.NWBFile(nwbfile)
print(nwbnap)
