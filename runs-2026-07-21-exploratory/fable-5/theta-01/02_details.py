"""Second-pass inspection: rates, epochs, unit metadata, electrode locations."""
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

S3 = "https://api.dandiarchive.org/api/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/"
rem_file = remfile.File(S3, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

lfp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
print("LFP rate:", lfp.rate, "start:", lfp.starting_time, "unit:", lfp.unit,
      "conversion:", lfp.conversion, "n_samples:", lfp.data.shape)
print("LFP duration (s):", lfp.data.shape[0] / lfp.rate)

pos = nwbfile.processing["behavior"]["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
print("\nPosition rate:", pos.rate, "start:", pos.starting_time, "unit:", pos.unit,
      "conversion:", pos.conversion, "ref:", pos.reference_frame)
lin = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]["1.6mLinearMazeLinearizedTimeSeries"]
print("Linearized rate:", lin.rate, "start:", lin.starting_time, "unit:", lin.unit, "shape", lin.data.shape)

print("\n=== epochs ===")
print(nwbfile.epochs.to_dataframe())

print("\n=== behavior states (first 10) ===")
st = nwbfile.processing["behavior"]["states"].to_dataframe()
print(st.head(10))
print("state labels:", st["label"].unique(), "n =", len(st))

print("\n=== units ===")
u = nwbfile.units.to_dataframe()
print(u[["cell_type", "location", "shank_id"]].head())
print(u.groupby(["location", "cell_type"]).size())

print("\n=== electrodes ===")
e = nwbfile.electrodes.to_dataframe()
print(e.groupby(["group_name", "location"]).size())
print("bad electrodes:", int(e["bad_electrode"].sum()))
print(e.head(10))

# position data preview
print("\n=== position preview ===")
p = pos.data[:1000, :]
print(p[:5], np.nanmin(pos.data[:], axis=0), np.nanmax(pos.data[:], axis=0))
l = lin.data[:]
print("linearized range:", np.nanmin(l), np.nanmax(l), "n nan:", np.isnan(l).sum())
