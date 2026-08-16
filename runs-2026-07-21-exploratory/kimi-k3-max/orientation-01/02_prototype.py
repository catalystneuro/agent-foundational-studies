"""Prototype: orientation/direction tuning of VISp units to drifting gratings.

Session 715093703 from DANDI 000021 (Allen Visual Coding Neuropixels).
"""
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINDI_URL = ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
             "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json")

# ---------------------------------------------------------------- load
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# ---------------------------------------------------------------- units
units = nwb["units"]  # TsGroup, index = unit id
print("n units:", len(units))

elec = nwbfile.electrodes.to_dataframe()
loc_map = elec["location"]

meta = units.metadata.copy()
meta["structure"] = meta["peak_channel_id"].map(loc_map)

visp_ids = meta[(meta["quality"] == "good") & (meta["structure"] == "VISp")].index
visp = units[list(visp_ids)]
print("good VISp units:", len(visp))

# ------------------------------------------------- drifting gratings table
tab = nwbfile.intervals["drifting_gratings_presentations"]
dg = pd.DataFrame({
    "start": np.asarray(tab["start_time"].data[:]),
    "stop": np.asarray(tab["stop_time"].data[:]),
    "orientation": np.asarray(tab["orientation"].data[:]),
    "temporal_frequency": np.asarray(tab["temporal_frequency"].data[:]),
})
dg = dg.dropna(subset=["orientation"]).reset_index(drop=True)
dg["orientation"] = dg["orientation"].astype(float)
dg["duration"] = dg["stop"] - dg["start"]
print(dg.groupby("orientation").size())

# ------------------------------------------------- spike counts per sweep
# count spikes of each unit in each presentation interval
starts = dg["start"].to_numpy()
stops = dg["stop"].to_numpy()

unit_ids = list(visp.index)
spike_trains = {uid: np.asarray(visp[uid].times()) for uid in unit_ids}

n_sweeps = len(dg)
counts = np.zeros((len(unit_ids), n_sweeps))
for i, uid in enumerate(unit_ids):
    st = spike_trains[uid]
    counts[i] = np.searchsorted(st, stops) - np.searchsorted(st, starts)

rates = counts / dg["duration"].to_numpy()[None, :]
rates_df = pd.DataFrame(rates, index=unit_ids)

# ------------------------------------------------- tuning curves
directions = np.sort(dg["orientation"].unique())
mean_rate = np.zeros((len(unit_ids), len(directions)))
sem_rate = np.zeros_like(mean_rate)
for j, d in enumerate(directions):
    cols = dg["orientation"].to_numpy() == d
    mean_rate[:, j] = rates_df.loc[:, cols].mean(axis=1)
    sem_rate[:, j] = rates_df.loc[:, cols].sem(axis=1)

tuning = pd.DataFrame(mean_rate, index=unit_ids, columns=directions)
print("\nExample tuning curves (first 5 units):")
print(tuning.head())

# ------------------------------------------------- selectivity metrics
def gosi(rate_vec, dirs_deg):
    """Global orientation selectivity index (vector magnitude at 2*theta)."""
    th = np.deg2rad(dirs_deg)
    r = np.asarray(rate_vec, dtype=float)
    if r.sum() <= 0:
        return 0.0
    return np.abs(np.sum(r * np.exp(2j * th))) / np.sum(r)

def gdsi(rate_vec, dirs_deg):
    """Global direction selectivity index (vector magnitude at theta)."""
    th = np.deg2rad(dirs_deg)
    r = np.asarray(rate_vec, dtype=float)
    if r.sum() <= 0:
        return 0.0
    return np.abs(np.sum(r * np.exp(1j * th))) / np.sum(r)

def osi_pref_orth(rate_vec, dirs_deg):
    """(R_pref_ori - R_orth) / (R_pref_ori + R_orth) on orientation-folded curve."""
    oris = (np.asarray(dirs_deg) % 180)
    uniq = np.sort(np.unique(oris))
    ori_curve = np.array([rate_vec[oris == u].mean() for u in uniq])
    i_pref = np.argmax(ori_curve)
    r_pref = ori_curve[i_pref]
    orth_angle = (uniq[i_pref] + 90) % 180
    r_orth = ori_curve[uniq == orth_angle][0]
    if r_pref + r_orth <= 0:
        return 0.0
    return (r_pref - r_orth) / (r_pref + r_orth)

metrics = pd.DataFrame(index=unit_ids)
metrics["pref_dir"] = [directions[np.argmax(tuning.loc[u].to_numpy())] for u in unit_ids]
metrics["max_rate"] = [tuning.loc[u].max() for u in unit_ids]
metrics["mean_rate"] = rates_df.mean(axis=1)
metrics["gOSI"] = [gosi(tuning.loc[u].to_numpy(), directions) for u in unit_ids]
metrics["gDSI"] = [gdsi(tuning.loc[u].to_numpy(), directions) for u in unit_ids]
metrics["OSI"] = [osi_pref_orth(tuning.loc[u].to_numpy(), directions) for u in unit_ids]
metrics["pref_ori"] = metrics["pref_dir"] % 180
print("\nMetrics summary:")
print(metrics.describe())

# ------------------------------------------------- quick validation figure
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
# pick top 8 units by gOSI with decent firing
cand = metrics[metrics["mean_rate"] > 1.0].sort_values("gOSI", ascending=False)
top = cand.index[:8]
for ax, uid in zip(axes.ravel(), top):
    tc = tuning.loc[uid].to_numpy()
    ax.errorbar(directions, tc, yerr=sem_rate[unit_ids.index(uid)], marker="o", ms=4)
    ax.set_title(f"unit {uid}\ngOSI={metrics.loc[uid,'gOSI']:.2f}, pref={metrics.loc[uid,'pref_dir']:.0f}°", fontsize=9)
    ax.set_xlabel("direction (deg)")
    ax.set_ylabel("rate (Hz)")
fig.suptitle("Direction tuning curves - top gOSI VISp units (session 715093703)")
fig.tight_layout()
fig.savefig("prototype_tuning_curves.png", dpi=150)
print("\nsaved prototype_tuning_curves.png")

metrics.to_csv("prototype_metrics.csv")
tuning.to_csv("prototype_tuning.csv")
print("saved prototype_metrics.csv, prototype_tuning.csv")
