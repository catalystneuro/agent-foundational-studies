import numpy as np, lindi, pynapple as nap
from pynwb import NWBHDF5IO

URL = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
cache = lindi.LocalCache(cache_dir="/tmp/lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=cache)
nwbfile = NWBHDF5IO(file=f, mode="r").read()

beh = nwbfile.processing["behavior"]
for name in ["1.6mLinearMazeLinearizedPosition", "1.6mLinearMazePosition", "states"]:
    di = beh[name]
    print("==", name, type(di).__name__)
    if hasattr(di, "spatial_series"):
        for sn, ss in di.spatial_series.items():
            print("   ", sn, ss.data.shape, "unit", ss.unit, "conv", ss.conversion,
                  "ts?", ss.timestamps is not None, "rate", ss.rate)
    else:
        print("   ", di)

nwb = nap.NWBFile(nwbfile)
print(nwb)
