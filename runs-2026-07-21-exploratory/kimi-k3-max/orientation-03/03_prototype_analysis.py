"""Prototype orientation-selectivity analysis on one session.

Steps:
1. Load session via LINDI, get units TsGroup + drifting gratings table.
2. Keep good units in visual cortical areas.
3. Build per-sweep spike count matrix (vectorized searchsorted).
4. Direction tuning curves at each unit's preferred temporal frequency.
5. gOSI / gDSI + permutation test.
"""
import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
LINDI_URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"
VISUAL_AREAS = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# ---------------- drifting gratings table ----------------
tab = nwbfile.intervals["drifting_gratings_presentations"]
starts = tab["start_time"].data[:]
stops = tab["stop_time"].data[:]
direction = tab["orientation"].data[:]  # drift direction in degrees (NaN = blank)
temp_freq = tab["temporal_frequency"].data[:]
durations = stops - starts
n_sweeps = len(starts)
print(f"sweeps: {n_sweeps}, blanks: {np.sum(np.isnan(direction))}")

# ---------------- units: quality + location filter ----------------
units_tab = nwbfile.units
quality = units_tab["quality"].data[:]
peak_ch = units_tab["peak_channel_id"].data[:]
unit_ids = units_tab["id"].data[:] if hasattr(units_tab["id"], "data") else units_tab.id[:]

elec_ids = nwbfile.electrodes["id"].data[:] if hasattr(nwbfile.electrodes["id"], "data") else nwbfile.electrodes.id[:]
elec_loc = nwbfile.electrodes["location"].data[:]
elec_loc = np.array([x.decode() if isinstance(x, bytes) else x for x in elec_loc])
id2loc = dict(zip(elec_ids.tolist(), elec_loc.tolist()))
loc_per_peak = np.array([id2loc[int(p)] for p in peak_ch])

unit_loc = np.array([str(x) for x in loc_per_peak])
is_good = quality == "good"
is_visual = np.isin(unit_loc, VISUAL_AREAS)
keep = is_good & is_visual
print(f"units total: {len(keep)}, good: {is_good.sum()}, good+visual: {keep.sum()}")
print("units per area:", {a: int(((unit_loc == a) & is_good).sum()) for a in VISUAL_AREAS})

# ---------------- TsGroup for kept units ----------------
ts_group = nwb["units"]
# TsGroup keys are unit IDs from the table's id column; map row mask -> unit ids
kept_indices = np.where(keep)[0]
kept_unit_ids = unit_ids[keep]
assert all(k in ts_group.keys() for k in kept_unit_ids), "unit id missing from TsGroup"
print(f"TsGroup n: {len(ts_group)}; kept unit ids: {len(kept_unit_ids)}")

# ---------------- per-sweep spike count matrix (vectorized) ----------------
kept_keys = list(kept_unit_ids)
key_to_col = {k: i for i, k in enumerate(kept_keys)}
counts = np.zeros((n_sweeps, len(kept_keys)), dtype=np.float64)

# concatenate spikes of kept units
spike_times_all = []
spike_unit_all = []
for k in kept_keys:
    t = ts_group[k].t
    spike_times_all.append(t)
    spike_unit_all.append(np.full(len(t), key_to_col[k]))
spike_times_all = np.concatenate(spike_times_all)
spike_unit_all = np.concatenate(spike_unit_all)

# assign each spike to a sweep
sweep_idx = np.searchsorted(starts, spike_times_all, side="right") - 1
valid = (sweep_idx >= 0) & (sweep_idx < n_sweeps)
valid[valid] &= spike_times_all[valid] < stops[sweep_idx[valid]]
np.add.at(counts, (sweep_idx[valid], spike_unit_all[valid]), 1.0)
rates = counts / durations[:, None]
print("count matrix:", counts.shape, "total spikes counted:", int(counts.sum()))

# ---------------- direction tuning ----------------
directions = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
tfs = np.array([1, 2, 4, 8, 15], dtype=float)
nonblank = ~np.isnan(direction)
blank_rate = rates[np.isnan(direction)].mean(axis=0)  # spontaneous rate per unit

# mean rate per (direction, tf) per unit
n_units = len(kept_keys)
resp = np.zeros((n_units, len(directions), len(tfs)))
for i, d in enumerate(directions):
    for j, tf in enumerate(tfs):
        m = nonblank & (direction == d) & (temp_freq == tf)
        resp[:, i, j] = rates[m].mean(axis=0)

# preferred TF = TF with max mean response (averaged over directions)
pref_tf_idx = resp.mean(axis=1).argmax(axis=1)
tuning = resp[np.arange(n_units), :, pref_tf_idx]  # (n_units, 8 directions)

# ---------------- gOSI / gDSI ----------------
theta = np.deg2rad(directions)
sum_r = tuning.sum(axis=1)
sum_r[sum_r == 0] = np.nan
gosi = np.abs((tuning * np.exp(2j * theta)).sum(axis=1)) / sum_r
gdsi = np.abs((tuning * np.exp(1j * theta)).sum(axis=1)) / sum_r
pref_dir = np.angle((tuning * np.exp(1j * theta)).sum(axis=1), deg=True) % 360
pref_ori = (np.angle((tuning * np.exp(2j * theta)).sum(axis=1), deg=True) / 2) % 180

print(f"\ngOSI: median {np.nanmedian(gosi):.3f}, units with gOSI>0.2: {np.sum(gosi > 0.2)}/{n_units}")
print(f"gDSI: median {np.nanmedian(gdsi):.3f}")

# ---------------- permutation test (shuffle direction labels within TF) ----------------
# Null must match the statistic: per shuffle, recompute (direction x TF) responses,
# pick each unit's preferred TF, then gOSI. Units with zero response (NaN gOSI) are excluded.
rng = np.random.default_rng(42)
n_perm = 500
gosi_null = np.full((n_perm, n_units), np.nan)
valid = ~np.isnan(gosi)
dir_labels = direction.copy()
for p in range(n_perm):
    shuffled = dir_labels.copy()
    for tf in tfs:
        m = nonblank & (temp_freq == tf)
        shuffled[m] = rng.permutation(shuffled[m])
    perm_resp = np.zeros((n_units, len(directions), len(tfs)))
    for i, d in enumerate(directions):
        for j, tf in enumerate(tfs):
            m = shuffled == d
            # keep TF balance: only average sweeps of this TF whose shuffled label is d
            m = m & (temp_freq == tf)
            if m.sum() > 0:
                perm_resp[:, i, j] = rates[m].mean(axis=0)
    pref_tf_p = perm_resp.mean(axis=1).argmax(axis=1)
    tuning_p = perm_resp[np.arange(n_units), :, pref_tf_p]
    s = tuning_p.sum(axis=1)
    s[s == 0] = np.nan
    gosi_null[p] = np.abs((tuning_p * np.exp(2j * theta)).sum(axis=1)) / s

p_values = np.ones(n_units)
null_cmp = gosi_null[:, valid] >= gosi[None, valid]  # NaN comparisons -> False
p_values[valid] = (np.sum(null_cmp, axis=0) + 1) / (n_perm + 1)
sig = (p_values < 0.01) & valid
print(f"\nsignificant orientation-selective units (p<0.01): {sig.sum()}/{n_units} ({100*sig.mean():.1f}%)")
print("median gOSI among significant:", np.nanmedian(gosi[sig]).round(3))

np.savez("prototype_results.npz", tuning=tuning, gosi=gosi, gdsi=gdsi, pref_ori=pref_ori,
         pref_dir=pref_dir, p_values=p_values, sig=sig, blank_rate=blank_rate,
         unit_loc=unit_loc[kept_indices], kept_indices=kept_indices, rates=rates,
         direction=direction, temp_freq=temp_freq, starts=starts, stops=stops)
print("saved prototype_results.npz")
