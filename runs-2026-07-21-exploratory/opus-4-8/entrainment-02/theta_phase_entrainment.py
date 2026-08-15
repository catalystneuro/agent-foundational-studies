# %% [markdown]
# # Theta Phase Entrainment of Hippocampal CA1 Neurons
#
# This notebook demonstrates **theta phase entrainment**: the tendency of
# hippocampal CA1 neurons to fire at a preferred phase of the 6-10 Hz theta
# rhythm in the local field potential (LFP). Theta phase locking is one of the
# most robust and widely replicated phenomena in systems neuroscience, and it
# underlies phase precession and the temporal coding of space during
# navigation.
#
# **Dataset.** DANDI dandiset **000044**, *"Hippocampal spatial coding during
# navigation"* (Grosmark, Long & Buzsáki; the `hc-11` dataset). We use session
# `sub-Buddy`, a rat running back and forth on a linear maze with dense silicon
# probe recordings in dorsal CA1. The NWB file provides simultaneously recorded
# wideband LFP, spike-sorted single units with cell-type labels
# (excitatory / inhibitory), anatomical location labels, and the animal's
# linearized position on the maze.
#
# **Approach.**
# 1. Stream the LFP and spikes directly from the DANDI S3 store (no full
#    download) and cache the needed slices locally.
# 2. Select the CA1 channel with the strongest theta rhythm.
# 3. Band-pass the LFP to 6-10 Hz and extract instantaneous theta phase via the
#    Hilbert transform.
# 4. Restrict to running epochs (theta is a movement-related rhythm).
# 5. For every CA1 unit, collect the theta phase at each spike and quantify
#    phase locking with the mean resultant length (MRL) and the Rayleigh test.
#
# **Key finding.** 62 of 65 CA1 units (95%) are significantly phase locked to
# theta (Rayleigh p < 0.05), with excitatory cells preferring phases near the
# descending flank / trough of the LFP theta cycle.

# %% [markdown]
# ## Setup

# %%
import os
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.ndimage import uniform_filter1d

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

THETA_LO, THETA_HI = 6.0, 10.0     # theta band (Hz)
SPEED_THRESH = 5.0                 # running threshold (cm/s)
CACHE = "cache/buddy_prep.npz"
os.makedirs("cache", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# %% [markdown]
# ## 1. Stream and cache the data from DANDI
#
# We stream with `remfile` (byte-range reads over S3) and a local disk cache,
# so nothing is fully downloaded. We only pull what the analysis needs: LFP for
# one candidate channel per shank (to pick the best theta channel), the full
# maze-epoch LFP for that channel, spike times, unit metadata, position, and
# epoch bounds. The result is cached to an `.npz`; re-running the notebook skips
# the streaming step entirely.

# %%
def build_cache():
    import h5py, remfile
    from pynwb import NWBHDF5IO
    from dandi.dandiapi import DandiAPIClient

    client = DandiAPIClient()
    ds = client.get_dandiset("000044", "draft")
    assets = sorted((a for a in ds.get_assets() if a.path.endswith(".nwb")),
                    key=lambda a: a.size)
    asset = assets[0]  # sub-Buddy, smallest (~5.2 GB); streamed, not downloaded
    print("streaming:", asset.path)

    rem = remfile.File(asset.download_url,
                       disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    io = NWBHDF5IO(file=h5py.File(rem, "r"))
    nwb = io.read()

    # maze epoch
    edf = nwb.epochs.to_dataframe()
    maze = edf[edf["label"] == "MazeEpoch"].iloc[0]
    t0, t1 = float(maze["start_time"]), float(maze["stop_time"])

    # LFP handle
    lfp = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    rate = float(lfp.rate)
    n_t, n_ch = lfp.data.shape
    conv = float(lfp.conversion)
    i0, i1 = int(t0 * rate), int(t1 * rate)
    tvec = np.arange(i0, i1) / rate

    # one mid channel per 8-channel shank; pick the strongest-theta channel
    cand = list(range(3, n_ch, 8))
    theta_ratio, cand_lfp = {}, {}
    for ch in cand:
        x = lfp.data[i0:i1, ch].astype(np.float32) * conv * 1e6  # microvolts
        cand_lfp[ch] = x
        f, p = welch(x, fs=rate, nperseg=int(rate * 4))
        theta = p[(f >= 6) & (f <= 10)].mean()
        broad = p[(f >= 2) & (f <= 40)].mean()
        theta_ratio[ch] = theta / broad
    best_ch = max(theta_ratio, key=theta_ratio.get)

    # position (linearized)
    beh = nwb.processing["behavior"].data_interfaces
    pos_key = [k for k in beh if "LinearizedPosition" in k][0]
    ss = list(beh[pos_key].spatial_series.values())[0]
    pos_data = np.asarray(ss.data[:]).squeeze()
    pos_t = (ss.timestamps[:] if ss.timestamps is not None
             else ss.starting_time + np.arange(len(pos_data)) / ss.rate)

    # units
    udf = nwb.units.to_dataframe()
    spike_times = np.array(
        [np.asarray(udf.iloc[i]["spike_times"]) for i in range(len(udf))],
        dtype=object)

    np.savez_compressed(
        CACHE, session=asset.path, rate=rate, t0=t0, t1=t1,
        tvec=tvec.astype(np.float32), best_ch=best_ch,
        cand_channels=np.array(cand),
        cand_theta_ratio=np.array([theta_ratio[c] for c in cand]),
        lfp_best=cand_lfp[best_ch].astype(np.float32),
        pos_data=pos_data.astype(np.float32), pos_t=np.asarray(pos_t, float),
        spike_times=spike_times,
        locations=udf["location"].astype(str).values,
        cell_types=udf["cell_type"].astype(str).values,
        shank_ids=udf["shank_id"].values)
    print("cached ->", CACHE)


if not os.path.exists(CACHE):
    build_cache()

d = np.load(CACHE, allow_pickle=True)
rate = float(d["rate"])
lfp_uv = d["lfp_best"].astype(np.float64)
best_ch = int(d["best_ch"])
t0, t1 = float(d["t0"]), float(d["t1"])
# reconstruct an exact float64 time axis (cached tvec is float32, which
# quantizes timestamps at ~11,000 s and corrupts sample spacing)
tvec = t0 + np.arange(len(lfp_uv)) / rate
pos = d["pos_data"].astype(np.float64)
pos_t = d["pos_t"].astype(np.float64)
spike_times = d["spike_times"]
locations = d["locations"]
cell_types = d["cell_types"]
cand_channels = d["cand_channels"]
cand_ratio = d["cand_theta_ratio"]
print(f"session {str(d['session'])}: LFP ch{best_ch}, "
      f"{len(lfp_uv)} samples @ {rate:g} Hz; {len(spike_times)} units total")

# %% [markdown]
# ### Channel selection
#
# The theta rhythm is strongest in the CA1 pyramidal / stratum radiatum layer.
# We choose the channel with the highest ratio of theta-band (6-10 Hz) power to
# broadband (2-40 Hz) power over the maze epoch.

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
order = np.argsort(cand_channels)
ax.bar(np.arange(len(cand_channels)), cand_ratio[order], color="#4477AA")
best_pos = int(np.where(cand_channels[order] == best_ch)[0][0])
ax.bar(best_pos, cand_ratio[order][best_pos], color="#EE6677",
       label=f"selected ch {best_ch}")
ax.set_xticks(np.arange(len(cand_channels)))
ax.set_xticklabels([f"ch{c}" for c in cand_channels[order]], rotation=45, fontsize=8)
ax.set_ylabel("theta (6-10 Hz) / broadband\npower ratio")
ax.set_title("Theta-channel selection across shanks (MazeEpoch LFP)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig1_channel_selection.png")

# %% [markdown]
# ## 2. Extract theta phase
#
# We band-pass the selected LFP channel to 6-10 Hz with a zero-phase
# Butterworth filter, then take the Hilbert transform. The angle of the
# analytic signal is the instantaneous theta phase (0 = LFP peak), and its
# magnitude is the theta amplitude envelope. All time series are wrapped in
# pynapple objects restricted to the maze epoch.

# %%
maze = nap.IntervalSet(start=t0, end=t1)
lfp_tsd = nap.Tsd(t=tvec, d=lfp_uv, time_support=maze)


def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


theta_filt = bandpass(lfp_uv, THETA_LO, THETA_HI, rate)
analytic = hilbert(theta_filt)
theta_phase = np.angle(analytic)      # radians in (-pi, pi]
theta_amp = np.abs(analytic)

phase_tsd = nap.Tsd(t=tvec, d=theta_phase, time_support=maze)

# %% [markdown]
# ## 3. Speed and running epochs
#
# Hippocampal theta is a movement-related rhythm, so phase locking is measured
# during running. We compute speed from the linearized position, smooth it, and
# keep epochs above 5 cm/s lasting at least 0.5 s.

# %%
good = np.isfinite(pos) & np.isfinite(pos_t)
pos_t, pos = pos_t[good], pos[good]
pos_cm = pos * 100.0                                    # meters -> cm
dt = np.gradient(pos_t)
speed = np.abs(np.gradient(pos_cm) / dt)
fs_pos = 1.0 / np.median(dt)
speed = uniform_filter1d(speed, max(1, int(0.25 * fs_pos)))
speed_tsd = nap.Tsd(t=pos_t, d=speed, time_support=maze)

run_ep = speed_tsd.threshold(SPEED_THRESH, method="above").time_support
run_ep = run_ep.drop_short_intervals(0.5)
print(f"Running epochs: {len(run_ep)} intervals, "
      f"{run_ep.tot_length():.1f} s of {maze.tot_length():.1f} s maze")

# %% [markdown]
# ## 4. Validation: LFP, theta band, and spikes together
#
# Before quantifying, we look at 3 s of running data. The band-passed theta
# trace tracks the rhythmic troughs of the raw LFP, and spikes from strongly
# locked units cluster at consistent phases of the theta cycle.

# %%
run_start, run_end = run_ep.start, run_ep.end
seg_i0 = int(np.argmax([e - s for s, e in zip(run_start, run_end)]))
w0 = run_start[seg_i0] + 1.0
w1 = w0 + 3.0
sel = (tvec >= w0) & (tvec <= w1)
tt = tvec[sel]

# quick per-unit phase-locking so we can pick strongly locked units to raster
is_ca1 = np.array([loc in ("lCA1", "rCA1") for loc in locations])


def circ_stats(phases):
    n = len(phases)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S)
    mrl = R / n
    mean_phase = np.arctan2(S, C)
    z = R ** 2 / n                                        # Rayleigh test
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n)
                      - (24 * z - 132 * z ** 2 + 76 * z ** 3
                         - 9 * z ** 4) / (288 * n ** 2))
    return mrl, mean_phase, min(p, 1.0), n


records, spike_phase_by_unit = [], {}
for i, st in enumerate(spike_times):
    if not is_ca1[i]:
        continue
    st = np.asarray(st, float)
    ts = nap.Ts(t=st, time_support=maze).restrict(run_ep)
    if len(ts) < 30:
        continue
    ph = phase_tsd.interpolate(ts).values
    ph = ph[np.isfinite(ph)]
    mrl, mphase, pval, n = circ_stats(ph)
    spike_phase_by_unit[i] = ph
    records.append(dict(unit=i, cell_type=cell_types[i], location=locations[i],
                        n_spikes=n, mrl=mrl, mean_phase=mphase, rayleigh_p=pval))

res = pd.DataFrame(records).set_index("unit")
res["sig"] = res["rayleigh_p"] < 0.05

# %%
strong = res.sort_values("mrl", ascending=False).head(12).index.tolist()
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1.4]})
axes[0].plot(tt, lfp_uv[sel], color="0.4", lw=0.7, label="raw LFP")
axes[0].plot(tt, theta_filt[sel], color="#EE6677", lw=1.4, label="theta 6-10 Hz")
axes[0].set_ylabel("µV")
axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
axes[0].set_title(f"CA1 LFP, theta band, and spike phase (ch {best_ch}, running)")

axes[1].plot(tt, theta_filt[sel], color="#EE6677", lw=1.4)
axes[1].fill_between(tt, -theta_amp[sel], theta_amp[sel], color="#EE6677",
                     alpha=0.15, label="theta envelope")
axes[1].set_ylabel("µV")
axes[1].legend(frameon=False, loc="upper right", fontsize=8)

for row, u in enumerate(strong):
    st = np.asarray(spike_times[u], float)
    st = st[(st >= w0) & (st <= w1)]
    axes[2].plot(st, np.full_like(st, row), "|", color="#222222", ms=6, mew=1.0)
axes[2].plot(tt, (theta_phase[sel] + np.pi) / (2 * np.pi) * (len(strong) - 1),
             color="#4477AA", lw=0.8, alpha=0.6, label="theta phase (0-2π)")
axes[2].set_ylabel("unit"); axes[2].set_xlabel("time (s)")
axes[2].set_yticks([0, len(strong) - 1]); axes[2].set_yticklabels(["", ""])
axes[2].legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig2_lfp_theta_raster.png")

# %% [markdown]
# ## 5. Per-unit phase locking
#
# For each CA1 unit we already collected the theta phase at every spike during
# running. The **mean resultant length (MRL)** measures concentration
# (0 = uniform, 1 = perfectly locked), and the **Rayleigh test** gives a
# p-value against the null of a uniform phase distribution.

# %%
n_exc = int((res["cell_type"] == "excitatory").sum())
n_inh = int((res["cell_type"] == "inhibitory").sum())
sig = int(res["sig"].sum())
print(f"CA1 units analysed: {len(res)}  (exc={n_exc}, inh={n_inh})")
print(f"Significantly phase-locked (Rayleigh p<0.05): "
      f"{sig}/{len(res)} = {100 * sig / len(res):.0f}%")
print("\nTop units by MRL:")
print(res.sort_values("mrl", ascending=False).head(8).to_string())
res.to_csv("cache/phase_locking_results.csv")

# %% [markdown]
# ### Example single-unit phase distributions
#
# Polar histograms for the six most strongly locked units. Each shows a clear
# concentration of spikes at a preferred theta phase (black line = mean phase).

# %%
examples = res.sort_values("mrl", ascending=False).head(6).index.tolist()
nb = 18
edges = np.linspace(-np.pi, np.pi, nb + 1)
centers = (edges[:-1] + edges[1:]) / 2
fig, axes = plt.subplots(2, 3, figsize=(10, 7.5),
                         subplot_kw={"projection": "polar"},
                         constrained_layout=True)
for ax, u in zip(axes.ravel(), examples):
    h, _ = np.histogram(spike_phase_by_unit[u], bins=edges)
    h = h / h.sum()
    ct = res.loc[u, "cell_type"]
    color = "#EE6677" if ct == "inhibitory" else "#4477AA"
    ax.bar(centers, h, width=2 * np.pi / nb, color=color, alpha=0.85,
           edgecolor="k", lw=0.3)
    mp, mrl = res.loc[u, "mean_phase"], res.loc[u, "mrl"]
    ax.plot([mp, mp], [0, h.max()], color="k", lw=2.2)
    ax.set_rticks([])
    ax.set_thetagrids([0, 90, 180, 270], labels=["0", "", "±π", "-π/2"], fontsize=8)
    ax.set_title(f"unit {u} ({ct[:3]})  MRL={mrl:.2f}", fontsize=9, pad=8)
fig.suptitle("Spike-phase distributions for strongly theta-locked CA1 units "
             "(black line = mean phase; 0 = LFP peak)", fontsize=11)
fig.savefig("figures/fig3_example_polar.png")

# %% [markdown]
# ## 6. Population summary
#
# Across the population, phase locking is strong and near-universal:
# most CA1 units have MRL between 0.1 and 0.4, 95% pass the Rayleigh test, and
# the preferred phases cluster (excitatory cells near the descending flank of
# the theta cycle).

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
exc = res[res["cell_type"] == "excitatory"]["mrl"]
inh = res[res["cell_type"] == "inhibitory"]["mrl"]
bins = np.linspace(0, res["mrl"].max() * 1.05, 16)
axes[0].hist(exc, bins=bins, color="#4477AA", alpha=0.75, label=f"exc (n={len(exc)})")
axes[0].hist(inh, bins=bins, color="#EE6677", alpha=0.75, label=f"inh (n={len(inh)})")
axes[0].set_xlabel("mean resultant length (MRL)"); axes[0].set_ylabel("# units")
axes[0].set_title("Phase-locking strength"); axes[0].legend(frameon=False, fontsize=8)

axes[1].bar(["locked\n(p<0.05)", "not"], [sig, len(res) - sig], color=["#228833", "0.7"])
for i, v in enumerate([sig, len(res) - sig]):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)
axes[1].set_ylabel("# CA1 units")
axes[1].set_title(f"Rayleigh test: {100 * sig / len(res):.0f}% significantly locked")

mp_sig = res[res["sig"]]["mean_phase"].values
axes[2].hist(mp_sig, bins=np.linspace(-np.pi, np.pi, 19), color="#AA3377", alpha=0.8)
axes[2].set_xlabel("preferred theta phase (rad)"); axes[2].set_ylabel("# units")
axes[2].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
axes[2].set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
axes[2].set_title("Preferred phases (locked units)")
fig.tight_layout()
fig.savefig("figures/fig4_population_summary.png")

# %% [markdown]
# ### Pooled spike-phase histogram
#
# Pooling all spikes from every significantly locked unit gives the population
# phase preference: spike density is clearly modulated across the theta cycle,
# the signature of theta phase entrainment. Two cycles are shown for clarity.

# %%
all_ph = np.concatenate([spike_phase_by_unit[u] for u in res[res["sig"]].index])
all_ph_deg = np.degrees(all_ph) % 360.0
nb2 = 36
h, e2 = np.histogram(all_ph_deg, bins=np.linspace(0, 360, nb2 + 1), density=True)
c2 = (e2[:-1] + e2[1:]) / 2
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(np.concatenate([c2, c2 + 360]), np.concatenate([h, h]),
       width=360 / nb2 * 0.95, color="#4477AA", alpha=0.8)
xx = np.linspace(0, 720, 400)
ax.plot(xx, h.mean() + h.std() * np.cos(np.radians(xx)),
        color="#EE6677", lw=1.5, label="theta LFP (schematic)")
ax.set_xlabel("theta phase (deg, two cycles)"); ax.set_ylabel("spike density")
ax.set_title(f"Pooled CA1 spike-phase histogram "
             f"({len(all_ph):,} spikes, {sig} locked units)")
ax.set_xticks(np.arange(0, 721, 180)); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig5_pooled_spike_phase.png")

# %% [markdown]
# ## Conclusion
#
# Using real silicon-probe data from dorsal CA1 (DANDI 000044, sub-Buddy), we
# demonstrated theta phase entrainment end to end: from streaming the raw LFP,
# through theta extraction and running-epoch selection, to per-unit and
# population phase-locking statistics. 62 of 65 CA1 units (95%) fire at a
# significantly non-uniform theta phase during running (Rayleigh p < 0.05),
# with a population preference concentrated on one side of the theta cycle. This
# reproduces the classic finding that hippocampal principal cells and
# interneurons are entrained by the theta rhythm.
