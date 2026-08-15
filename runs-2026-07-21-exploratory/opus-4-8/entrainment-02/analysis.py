"""
Core theta phase-entrainment analysis on the cached Buddy session
(DANDI:000044, Grosmark & Buzsaki hc-11). Uses pynapple for all time-series
handling. Produces validated figures and a results table.
"""
import numpy as np
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

THETA_LO, THETA_HI = 6.0, 10.0
SPEED_THRESH = 5.0       # cm/s, running threshold
CACHE = "cache/buddy_prep.npz"

# ---------------------------------------------------------------- load
d = np.load(CACHE, allow_pickle=True)
rate = float(d["rate"])
tvec = d["tvec"].astype(np.float64)
lfp_uv = d["lfp_best"].astype(np.float64)
best_ch = int(d["best_ch"])
t0, t1 = float(d["t0"]), float(d["t1"])
pos = d["pos_data"].astype(np.float64)
pos_t = d["pos_t"].astype(np.float64)
spike_times = d["spike_times"]
locations = d["locations"]
cell_types = d["cell_types"]
print(f"Loaded: LFP ch{best_ch}, {len(lfp_uv)} samples @ {rate} Hz; {len(spike_times)} units")

maze = nap.IntervalSet(start=t0, end=t1)

# ---------------------------------------------------------------- LFP -> theta phase
lfp = nap.Tsd(t=tvec, d=lfp_uv, time_support=maze)

def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

theta_filt = bandpass(lfp_uv, THETA_LO, THETA_HI, rate)
analytic = hilbert(theta_filt)
theta_phase = np.angle(analytic)          # radians, -pi..pi
theta_amp = np.abs(analytic)

theta_tsd = nap.Tsd(t=tvec, d=theta_filt, time_support=maze)
phase_tsd = nap.Tsd(t=tvec, d=theta_phase, time_support=maze)
amp_tsd = nap.Tsd(t=tvec, d=theta_amp, time_support=maze)

# ---------------------------------------------------------------- speed & running epochs
# position may have NaNs; interpolate/clean
good = np.isfinite(pos) & np.isfinite(pos_t)
pos_t, pos = pos_t[good], pos[good]
# position units are meters on a 1.6 m linear maze -> convert to cm
pos_cm = pos * 100.0
posd = nap.Tsd(t=pos_t, d=pos_cm, time_support=maze)
# speed = |d pos/dt|, smoothed
dt = np.gradient(pos_t)
speed = np.abs(np.gradient(pos_cm) / dt)
# smooth speed over ~0.25 s
from scipy.ndimage import uniform_filter1d
fs_pos = 1.0 / np.median(dt)
speed = uniform_filter1d(speed, max(1, int(0.25 * fs_pos)))
speed_tsd = nap.Tsd(t=pos_t, d=speed, time_support=maze)

run_ep = speed_tsd.threshold(SPEED_THRESH, method="above").time_support
run_ep = run_ep.drop_short_intervals(0.5)   # keep runs >0.5 s
print(f"Running epochs: {len(run_ep)} intervals, total {run_ep.tot_length():.1f} s "
      f"of {maze.tot_length():.1f} s maze")

np.save("cache/run_ep.npy", np.c_[run_ep.start, run_ep.end])

# ---------------------------------------------------------------- build TsGroup of CA1 units
is_ca1 = np.array([loc in ("lCA1", "rCA1") for loc in locations])
tsg_dict, meta_ct, meta_loc = {}, {}, {}
for i, st in enumerate(spike_times):
    if not is_ca1[i]:
        continue
    st = np.asarray(st, dtype=float)
    tsg_dict[i] = nap.Ts(t=st, time_support=maze)
    meta_ct[i] = cell_types[i]
    meta_loc[i] = locations[i]
units = nap.TsGroup(tsg_dict, time_support=maze)
ct_arr = np.array([meta_ct[k] for k in units.keys()])
print(f"CA1 units: {len(units)}  (exc={np.sum(ct_arr=='excitatory')}, inh={np.sum(ct_arr=='inhibitory')})")

# ---------------------------------------------------------------- phase-locking per unit
def circ_stats(phases):
    n = len(phases)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S)
    mrl = R / n
    mean_phase = np.arctan2(S, C)
    # Rayleigh test
    z = R**2 / n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    return mrl, mean_phase, p, n

results = []
spike_phase_by_unit = {}
for k in units.keys():
    st = units[k]
    st_run = st.restrict(run_ep)          # spikes during running
    if len(st_run) < 30:
        continue
    ph = phase_tsd.interpolate(st_run).values   # theta phase at each spike
    ph = ph[np.isfinite(ph)]
    mrl, mphase, pval, n = circ_stats(ph)
    spike_phase_by_unit[k] = ph
    results.append(dict(unit=k, cell_type=meta_ct[k], location=meta_loc[k],
                        n_spikes=n, mrl=mrl, mean_phase=mphase, rayleigh_p=pval))

import pandas as pd
res = pd.DataFrame(results).set_index("unit")
res["sig"] = res["rayleigh_p"] < 0.05
print(res.sort_values("mrl", ascending=False).head(12).to_string())
print(f"\nSignificantly phase-locked (Rayleigh p<0.05): "
      f"{res['sig'].sum()}/{len(res)} = {100*res['sig'].mean():.0f}%")
res.to_csv("cache/phase_locking_results.csv")
np.savez("cache/spike_phases.npz", **{str(k): v for k, v in spike_phase_by_unit.items()})

# stash arrays needed for plotting
np.savez_compressed("cache/analysis_arrays.npz",
                    tvec=tvec, lfp=lfp_uv, theta=theta_filt, phase=theta_phase,
                    amp=theta_amp, pos_t=pos_t, speed=speed, pos_cm=pos_cm,
                    run_start=run_ep.start, run_end=run_ep.end,
                    best_ch=best_ch)
print("analysis complete; arrays cached")
