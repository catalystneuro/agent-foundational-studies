# Cache session arrays (trials + spikes + unit meta) to a local .npz for fast iteration
import sys
import lindi
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

ASSETS = {
    "sub-92130c1b": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "sub-70bf8cbd": "81169999-c697-4eca-a635-2fd994ac183f",
    "sub-9bebfe0b": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "sub-c6e8125f": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}

def cache_session(name, asset_id):
    url = f"https://lindi.neurosift.org/dandi/dandisets/000149/assets/{asset_id}/nwb.lindi.json"
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()

    trials = nwbfile.intervals["trials"].to_dataframe()
    trials.columns = [c.replace(".npy", "") for c in trials.columns]

    st = f["units/spike_times"][:]
    sti = f["units/spike_times_index"][:]
    ks2 = np.array([x.decode() if isinstance(x, bytes) else x for x in f["units/ks2_label"][:]])
    if "brainLocationAcronyms_ccf_2017.npy" in f["units"] and f["units/brainLocationAcronyms_ccf_2017.npy"] is not None:
        regions = np.array([x.decode() if isinstance(x, bytes) else x
                            for x in f["units/brainLocationAcronyms_ccf_2017.npy"][:]])
    else:
        regions = np.array(["unknown"] * len(sti))

    np.savez_compressed(
        f"cache_{name}.npz",
        spike_times=st, spike_times_index=sti, ks2=ks2, regions=regions,
        **{f"trial_{c}": trials[c].to_numpy() for c in trials.columns},
    )
    print(name, "cached:", len(sti), "units,", len(trials), "trials,", len(st), "spikes")

if __name__ == "__main__":
    names = sys.argv[1:] or list(ASSETS)
    for n in names:
        cache_session(n, ASSETS[n])
