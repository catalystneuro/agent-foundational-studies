import numpy as np, lindi, pynapple as nap
from pynwb import NWBHDF5IO
URL = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=lindi.LocalCache(cache_dir="/tmp/lindi_cache"))
nwbfile = NWBHDF5IO(file=f, mode="r").read()
ss = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"].spatial_series["1.6mLinearMazeLinearizedTimeSeries"]
print("starting_time", ss.starting_time, "rate", ss.rate, "unit", ss.starting_time_unit)
d = ss.data[:]
print("data range", np.nanmin(d), np.nanmax(d), "nans", np.isnan(d).sum(), d.shape)
ss2 = nwbfile.processing["behavior"]["1.6mLinearMazePosition"].spatial_series["1.6mLinearMazeSpatialSeries"]
d2 = ss2.data[:]
print("2d range", np.nanmin(d2,0), np.nanmax(d2,0), "nans", np.isnan(d2).sum())
print("epochs:", nwbfile.epochs.to_dataframe().values)
print("implied duration @39.06Hz:", d.shape[0]/39.0625)

nwb = nap.NWBFile(nwbfile)
pos = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("pynapple pos t:", pos.t[:3], pos.t[-3:], "n", len(pos))
units = nwb["units"]
print(units)
