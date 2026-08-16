"""Core orientation selectivity analysis.

Computes per-trial firing rates during drifting grating presentations for all
good units in visual areas (VISp, VISl, VISpm, VISam, VISrl), LGd (thalamus),
and CA1 (hippocampus, negative control). Then computes orientation/direction
selectivity metrics and permutation-test significance.

Saves: orientation_results.csv
"""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd
import numpy as np
from tqdm import tqdm

DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

AREAS = ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'LGd', 'CA1']
N_SHUFFLES = 1000
MIN_PEAK_RESPONSE = 1.0  # Hz above blank required to count as selective
rng = np.random.default_rng(42)

# ---------------- Load ----------------
disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# ---------------- Stimulus table ----------------
dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
inv = nwb['invalid_times'].values
starts_all = dg['start_time'].values
stops_all = dg['stop_time'].values
overlap = np.zeros(len(dg), dtype=bool)
for a, b in inv:
    overlap |= (starts_all < b) & (stops_all > a)
dg = dg[~overlap].reset_index(drop=True)

is_blank = dg['orientation'].isna().values
directions = np.sort(dg['orientation'].dropna().unique())  # 8 values, 0..315
n_dirs = len(directions)
dir_to_idx = {d: i for i, d in enumerate(directions)}
trial_dir_idx = dg['orientation'].map(dir_to_idx).values  # NaN for blanks
trial_starts = dg['start_time'].values
trial_stops = dg['stop_time'].values
trial_durs = trial_stops - trial_starts

print(f"Valid presentations: {len(dg)} ({is_blank.sum()} blank), "
      f"directions: {directions}")

# ---------------- Unit selection ----------------
units = nwb['units']
meta = units.metadata.copy()
elec = nwbfile.electrodes.to_dataframe()
meta['structure'] = meta['peak_channel_id'].map(elec['location'])
sel = meta[(meta['quality'] == 'good') & (meta['structure'].isin(AREAS))]
print(f"Selected good units in {AREAS}: {len(sel)}")
print(sel['structure'].value_counts())

# ---------------- Metrics helpers ----------------
theta_rad = np.deg2rad(directions)
exp2 = np.exp(2j * theta_rad)  # orientation (axis) weights
exp1 = np.exp(1j * theta_rad)  # direction weights


def curve_metrics(dir_rates, blank_rate):
    """dir_rates: mean rate per direction (8,). Returns metrics dict."""
    sub = np.clip(dir_rates - blank_rate, 0, None)
    total = sub.sum()
    if total <= 0:
        return dict(gosi=np.nan, osi=np.nan, gdsi=np.nan, dsi=np.nan,
                    pref_dir=np.nan, pref_ori=np.nan, peak_response=0.0)
    gosi = np.abs(np.sum(sub * exp2)) / total
    gdsi = np.abs(np.sum(sub * exp1)) / total
    pref_idx = int(np.argmax(sub))
    pref_dir = directions[pref_idx]
    # orientation curve: average opposite directions
    ori_curve = np.array([sub[i] + sub[(i + n_dirs // 2) % n_dirs]
                          for i in range(n_dirs // 2)]) / 2.0
    pref_ori_idx = int(np.argmax(ori_curve))
    pref_ori = directions[pref_ori_idx]  # 0,45,90,135
    orth_ori_idx = (pref_ori_idx + 2) % (n_dirs // 2)
    o_pref, o_orth = ori_curve[pref_ori_idx], ori_curve[orth_ori_idx]
    osi = (o_pref - o_orth) / (o_pref + o_orth) if (o_pref + o_orth) > 0 else np.nan
    anti_idx = (pref_idx + n_dirs // 2) % n_dirs
    d_pref, d_anti = sub[pref_idx], sub[anti_idx]
    dsi = (d_pref - d_anti) / (d_pref + d_anti) if (d_pref + d_anti) > 0 else np.nan
    return dict(gosi=gosi, osi=osi, gdsi=gdsi, dsi=dsi,
                pref_dir=pref_dir, pref_ori=pref_ori,
                peak_response=float(dir_rates.max() - blank_rate))


def gosi_from_labels(rates, labels, blank_rate):
    """gOSI of the blank-subtracted mean-per-direction curve for given labels.

    Identical statistic to curve_metrics: per-direction means are computed
    first, then blank-subtracted and clipped."""
    sums = np.bincount(labels, weights=rates, minlength=n_dirs)
    counts = np.bincount(labels, minlength=n_dirs)
    means = sums / np.maximum(counts, 1)
    sub = np.clip(means - blank_rate, 0, None)
    total = sub.sum()
    if total <= 0:
        return 0.0
    return np.abs(np.sum(sub * exp2)) / total


# ---------------- Main loop ----------------
dir_trial_mask = ~is_blank
dir_labels_all = np.array([int(x) if not np.isnan(x) else -1 for x in trial_dir_idx])

results = []
dir_rate_curves = {}

for unit_id in tqdm(sel.index, desc="Units"):
    ts = units[unit_id].t  # spike times (s)
    # spike counts per presentation via searchsorted
    counts = np.searchsorted(ts, trial_stops) - np.searchsorted(ts, trial_starts)
    rates = counts / trial_durs

    blank_rate = rates[is_blank].mean() if is_blank.any() else 0.0
    dir_rates = np.array([rates[(trial_dir_idx == di)].mean()
                          for di in range(n_dirs)])

    m = curve_metrics(dir_rates, blank_rate)

    # permutation test on gOSI: shuffle direction labels across trials,
    # recompute the same blank-subtracted mean-curve statistic
    r = rates[dir_trial_mask]
    labels = dir_labels_all[dir_trial_mask]
    if not np.isnan(m['gosi']):
        obs = gosi_from_labels(r, labels, blank_rate)
        null = np.empty(N_SHUFFLES)
        for s in range(N_SHUFFLES):
            null[s] = gosi_from_labels(r, rng.permutation(labels), blank_rate)
        pval = (np.sum(null >= obs) + 1) / (N_SHUFFLES + 1)
        gosi_stat = obs  # identical to m['gosi'] up to trial-count weighting
    else:
        pval = 1.0
        gosi_stat = np.nan

    results.append(dict(
        unit_id=unit_id,
        structure=meta.loc[unit_id, 'structure'],
        firing_rate=meta.loc[unit_id, 'firing_rate'],
        snr=meta.loc[unit_id, 'snr'],
        blank_rate=blank_rate,
        p_value=pval,
        gosi_perm=gosi_stat,
        **m,
    ))
    dir_rate_curves[unit_id] = dir_rates

res = pd.DataFrame(results).set_index('unit_id')
res['selective'] = (res['p_value'] < 0.05) & (res['peak_response'] >= MIN_PEAK_RESPONSE)

curves_df = pd.DataFrame(dir_rate_curves,
                         index=[f"dir_{int(d)}" for d in directions]).T
curves_df.index.name = 'unit_id'
out = res.join(curves_df)
out.to_csv('orientation_results.csv')
print("\nSaved orientation_results.csv")

print("\nFraction orientation-selective per area (p<0.05 & peak>=1 Hz):")
print(res.groupby('structure')['selective'].agg(['mean', 'sum', 'count']))
print("\nMedian gOSI per area:")
print(res.groupby('structure')['gosi'].median())
