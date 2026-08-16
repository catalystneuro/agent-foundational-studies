"""Save per-sweep rate matrices (drifting + static gratings) for figure generation.

Loads are fast because LINDI chunks are cached locally from the previous run.
"""
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
from tqdm import tqdm

SESSIONS = {
    "715093703": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json"),
    "719161530": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "02291b99-e583-498b-9929-b68bba2c50e2/nwb.lindi.json"),
    "721123822": ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
                  "224b57e5-c9a3-46ef-85db-966713f3ccbe/nwb.lindi.json"),
}
VISUAL_AREAS = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]


def intervals_to_df(tab, cols):
    return pd.DataFrame({c: np.asarray(tab[c].data[:]) for c in cols})


def rates_for_sweeps(spike_trains, starts, stops):
    counts = np.zeros((len(spike_trains), len(starts)))
    for i, st in enumerate(spike_trains):
        counts[i] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return counts / (stops - starts)[None, :]


for sid, url in SESSIONS.items():
    print(f"=== {sid} ===")
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    elec = nwbfile.electrodes.to_dataframe()
    meta = units.metadata.copy()
    meta["structure"] = meta["peak_channel_id"].map(elec["location"])
    keep = meta[(meta["quality"] == "good") & (meta["structure"].isin(VISUAL_AREAS))]
    unit_ids = list(keep.index)

    spike_trains = [np.asarray(units[uid].times()) for uid in tqdm(unit_ids, desc="spikes")]

    # drifting gratings
    dg = intervals_to_df(nwbfile.intervals["drifting_gratings_presentations"],
                         ["start_time", "stop_time", "orientation", "temporal_frequency"])
    dg.columns = ["start", "stop", "orientation", "temporal_frequency"]
    dg = dg.dropna(subset=["orientation"]).reset_index(drop=True)
    dg["orientation"] = dg["orientation"].astype(float)
    rates = rates_for_sweeps(spike_trains, dg["start"].to_numpy(), dg["stop"].to_numpy())
    np.save(f"rates_{sid}.npy", rates)
    dg.to_csv(f"sweeps_{sid}.csv", index=False)
    pd.DataFrame({"unit_id": unit_ids, "structure": keep["structure"].to_numpy()}).to_csv(
        f"units_{sid}.csv", index=False)
    print("drifting rates:", rates.shape)

    # static gratings (finer orientation resolution: 6 orientations x 30 deg)
    sg = intervals_to_df(nwbfile.intervals["static_gratings_presentations"],
                         ["start_time", "stop_time", "orientation", "spatial_frequency", "phase"])
    sg.columns = ["start", "stop", "orientation", "spatial_frequency", "phase"]
    sg = sg.dropna(subset=["orientation"]).reset_index(drop=True)
    sg["orientation"] = sg["orientation"].astype(float)
    srates = rates_for_sweeps(spike_trains, sg["start"].to_numpy(), sg["stop"].to_numpy())
    np.save(f"rates_static_{sid}.npy", srates)
    sg.to_csv(f"sweeps_static_{sid}.csv", index=False)
    print("static rates:", srates.shape)

print("done")
