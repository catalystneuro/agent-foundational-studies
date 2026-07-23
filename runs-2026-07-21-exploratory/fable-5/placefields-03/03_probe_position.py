"""Probe the raw SpatialSeries objects to understand the timestamp encoding."""

import numpy as np
import lindi
from pynwb import NWBHDF5IO

LINDI_URL = (
    "https://lindi.neurosift.org/dandi/dandisets/000044/assets/"
    "5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
)

local_cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()

beh = nwbfile.processing["behavior"]

for iface_name in ["1.6mLinearMazePosition", "1.6mLinearMazeLinearizedPosition"]:
    iface = beh[iface_name]
    for ss_name, ss in iface.spatial_series.items():
        print("=" * 70)
        print(iface_name, "/", ss_name)
        print("  data shape :", ss.data.shape, ss.data.dtype)
        print("  unit       :", ss.unit, "conversion:", ss.conversion)
        print("  rate       :", ss.rate)
        print("  starting_t :", ss.starting_time)
        print("  resolution :", ss.resolution)
        ts = ss.timestamps
        if ts is not None:
            t = np.asarray(ts[:])
            print("  timestamps : shape", t.shape, "first5", t[:5], "diff5", np.diff(t[:6]))
        d = np.asarray(ss.data[:])
        print("  data first 10:\n", d[:10])
        finite = np.isfinite(d).all(axis=-1) if d.ndim > 1 else np.isfinite(d)
        print("  n finite samples:", int(finite.sum()), "/", len(d))
        idx = np.where(finite)[0]
        if len(idx):
            print("  finite index range:", idx[0], idx[-1])
            # contiguous blocks of finite samples
            brk = np.where(np.diff(idx) > 1)[0]
            print("  n contiguous finite blocks:", len(brk) + 1)

print()
print("=== states table ===")
st = beh["states"]
print(st.colnames)
import pandas as pd
df = st.to_dataframe()
print(df.head(20))
print("label counts:\n", df.iloc[:, -1].value_counts() if df.shape[1] else None)
print("total rows:", len(df))
