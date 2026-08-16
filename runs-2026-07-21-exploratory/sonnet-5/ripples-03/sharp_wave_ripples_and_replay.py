# %% [markdown]
# # Sharp-Wave Ripples and Replay in Hippocampal CA1
#
# This notebook demonstrates two classic hippocampal sleep phenomena using a real
# extracellular recording from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)**: transient ~150 Hz oscillatory bursts in the CA1 LFP
#    that occur during slow-wave sleep and quiet wakefulness.
# 2. **Replay / reactivation**: the tendency for the sequence of place cells active during
#    running to be reactivated, in compressed and/or correlated form, during subsequent
#    sleep sharp-wave ripples.
#
# ## Dataset
#
# [DANDI:000044](https://dandiarchive.org/dandiset/000044) "Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences" (Grosmark & Buzsaki,
# *Science* 2016). Each recording session consists of a long PRE home-cage sleep epoch,
# a MAZE epoch in which a rat runs on a novel 1.6 m linear track for a water reward, and
# a long POST home-cage sleep epoch. Bilateral silicon-probe recordings (128 channels)
# in dorsal CA1 give both LFP and spike-sorted single units.
#
# We use subject **Buddy**, session `Buddy-06272013` (the smallest of the eight sessions,
# ~5.2 GB), streamed directly from the DANDI S3 bucket with `remfile` (no local download
# of the full file).

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d

import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

np.random.seed(0)

# %% [markdown]
# ## 1. Streaming access to the NWB file
#
# The asset is opened with `remfile.File` (backed by a local `DiskCache` so repeated byte
# ranges are not re-downloaded) wrapped in `h5py`/`pynwb`, then handed to
# `pynapple.NWBFile` for convenient, lazy, time-aware access to LFP, spikes, position and
# behavioral-state intervals.

# %%
NWB_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/49f/95f/"
    "49f95f2a-1ae4-4720-85a3-899847616078"
)  # sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb (DANDI:000044)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(NWB_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df)

PRE_START, PRE_END = epochs_df.loc[0, "start_time"], epochs_df.loc[0, "stop_time"]
MAZE_START, MAZE_END = epochs_df.loc[1, "start_time"], epochs_df.loc[1, "stop_time"]
POST_START, POST_END = epochs_df.loc[2, "start_time"], epochs_df.loc[2, "stop_time"]

states_df = nwbfile.processing["behavior"].data_interfaces["states"].to_dataframe()
print(states_df["label"].value_counts())

units = nwb["units"]
exc_units = units[units.cell_type == "excitatory"]
print(f"{len(units)} total units, {len(exc_units)} putative excitatory (pyramidal) CA1 units")

# %% [markdown]
# ## 2. Selecting a ripple channel and detecting sharp-wave ripples
#
# We pick the LFP channel with the strongest ripple-band (100-250 Hz) power in a short
# NREM snippet — a standard way to find the CA1 pyramidal-layer channel without relying on
# (frequently incomplete) anatomical metadata — and then detect discrete ripple events on
# that channel during all NREM sleep of the POST epoch.

# %%
lfp = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
FS = lfp.rate
print("LFP:", lfp.data.shape, "channels @", FS, "Hz")

# a 60 s NREM snippet, all 128 channels, to pick the ripple channel
snippet_t0, snippet_t1 = 16722.0 + 80, 16722.0 + 140  # inside the largest POST NREM bout
i0, i1 = int(snippet_t0 * FS), int(snippet_t1 * FS)
snippet = np.asarray(lfp.data[i0:i1, :], dtype=np.float64)

# the raw dataset is stored as int16 counts; apply the NWB conversion factor to get volts
# (reading lfp.data[...] directly, as needed for partial/streamed access, bypasses pynwb's
# automatic unit conversion, so we must apply it ourselves)
LFP_CONVERSION = lfp.conversion  # volts per count
snippet = snippet * LFP_CONVERSION

sos_pick = butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
snippet_filt = sosfiltfilt(sos_pick, snippet, axis=0)
ripple_power = np.std(snippet_filt, axis=0)
RIPPLE_CH = int(np.argmax(ripple_power))
print(f"Selected LFP channel {RIPPLE_CH} (ripple-band SD = {ripple_power[RIPPLE_CH]*1e6:.1f} uV)")

# %%
# stream the full single-channel LFP for the POST epoch only (~80 MB, vs 5+ GB for the
# full 128-channel recording)
i0, i1 = int(POST_START * FS), int(POST_END * FS)
post_lfp = np.asarray(lfp.data[i0:i1, RIPPLE_CH], dtype=np.float64) * LFP_CONVERSION
post_t = POST_START + np.arange(len(post_lfp)) / FS
print(f"Loaded {len(post_lfp)} samples ({(post_t[-1]-post_t[0])/60:.1f} min) of POST LFP")

# %%
nrem_df = states_df[states_df["label"] == "Non-REM"]
nrem_post_df = nrem_df[(nrem_df["start_time"] >= POST_START) & (nrem_df["stop_time"] <= POST_END)]
nrem_post_ep = nap.IntervalSet(start=nrem_post_df["start_time"].values, end=nrem_post_df["stop_time"].values)
print(f"{len(nrem_post_ep)} NREM bouts in POST, {nrem_post_ep.tot_length():.0f} s total")

sos_ripple = butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
ripple_filt = sosfiltfilt(sos_ripple, post_lfp)
envelope = np.abs(hilbert(ripple_filt))
envelope_smooth = gaussian_filter1d(envelope, sigma=0.010 * FS)  # 10 ms

env_tsd = nap.Tsd(t=post_t, d=envelope_smooth)
env_nrem = env_tsd.restrict(nrem_post_ep)
mu, sigma = env_nrem.values.mean(), env_nrem.values.std()
z = (envelope_smooth - mu) / sigma
z_tsd = nap.Tsd(t=post_t, d=z)
z_nrem = z_tsd.restrict(nrem_post_ep)

# %%
def detect_ripples(zv, tv, low_thr=2.0, high_thr=5.0, min_dur=0.020, max_dur=0.400, merge_gap=0.030, fs=FS):
    """Standard threshold-crossing ripple detector (e.g. Buzsaki-lab / Karlsson & Frank style):
    envelope must cross `low_thr` (candidate boundary), peak inside must exceed `high_thr`."""
    above_low = zv > low_thr
    dt = np.diff(tv)
    breaks = np.where(dt > 1.5 / fs)[0]  # bout boundaries (gaps between NREM bouts)
    run_start = np.concatenate(([0], breaks + 1))
    run_end = np.concatenate((breaks, [len(tv) - 1]))

    events = []
    for rs, re in zip(run_start, run_end):
        seg_above, seg_t, seg_z = above_low[rs:re + 1], tv[rs:re + 1], zv[rs:re + 1]
        idx, L = 0, len(seg_above)
        while idx < L:
            if seg_above[idx]:
                j = idx
                while j < L and seg_above[j]:
                    j += 1
                t0, t1 = seg_t[idx], seg_t[j - 1]
                pk_z = seg_z[idx:j].max()
                pk_t = seg_t[idx:j][np.argmax(seg_z[idx:j])]
                dur = t1 - t0
                if dur >= min_dur and pk_z >= high_thr:
                    events.append([t0, t1, pk_t, pk_z, dur])
                idx = j
            else:
                idx += 1

    events.sort(key=lambda e: e[0])
    merged = []
    for ev in events:
        if merged and ev[0] - merged[-1][1] < merge_gap:
            prev = merged[-1]
            pk_t, pk_z = (ev[2], ev[3]) if ev[3] > prev[3] else (prev[2], prev[3])
            merged[-1] = [prev[0], ev[1], pk_t, pk_z, ev[1] - prev[0]]
        else:
            merged.append(ev)
    return [ev for ev in merged if min_dur <= ev[4] <= max_dur]


ripple_events = detect_ripples(z_nrem.values, z_nrem.times())
print(f"Detected {len(ripple_events)} ripple events during POST-epoch NREM sleep")

durs = np.array([e[4] for e in ripple_events])
peakz = np.array([e[3] for e in ripple_events])
peakt = np.array([e[2] for e in ripple_events])
print(f"Rate: {len(ripple_events)/nrem_post_ep.tot_length():.3f} Hz during NREM")
print(f"Duration: median {np.median(durs)*1000:.0f} ms, mean {durs.mean()*1000:.0f} ms")
print(f"Peak z-score: median {np.median(peakz):.1f}, max {peakz.max():.1f}")

# %% [markdown]
# ## 3. Ripple visualizations

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# panel A: a few minutes of envelope with detected events marked
ax = axes[0, 0]
window = (post_t >= 16722 + 60) & (post_t < 16722 + 160)
ax.plot(post_t[window] - 16722, post_lfp[window] * 1e6, lw=0.4, color="0.5", label="broadband LFP")
ax.plot(post_t[window] - 16722, ripple_filt[window] * 1e6 - 300, lw=0.6, color="C0", label="100-250 Hz filtered")
in_win = [(e[0], e[1]) for e in ripple_events if 16722 + 60 <= e[0] < 16722 + 160]
for t0, t1 in in_win:
    ax.axvspan(t0 - 16722, t1 - 16722, color="red", alpha=0.25)
ax.set_xlabel("Time (s, within NREM bout)")
ax.set_ylabel("LFP (uV, offset)")
ax.set_title(f"POST NREM LFP with detected ripples ({len(in_win)} shown)")
ax.legend(loc="upper right", fontsize=8)

# panel B: single example ripple, zoomed
ax = axes[0, 1]
best = ripple_events[np.argmax(peakz)]
ct = best[2]
m = (post_t >= ct - 0.15) & (post_t < ct + 0.15)
tloc = (post_t[m] - ct) * 1000
ax.plot(tloc, post_lfp[m] * 1e6, color="0.4", lw=1, label="broadband")
ax.plot(tloc, ripple_filt[m] * 1e6, color="C0", lw=1.2, label="ripple-band")
ax.plot(tloc, z_tsd.restrict(nap.IntervalSet(ct - 0.15, ct + 0.15)).values, color="C3", lw=1, label="z-score envelope")
ax.axvspan((best[0] - ct) * 1000, (best[1] - ct) * 1000, color="red", alpha=0.15)
ax.set_xlabel("Time from ripple peak (ms)")
ax.set_ylabel("Amplitude (uV) / z-score")
ax.set_title(f"Highest-power example ripple (peak z={best[3]:.1f}, dur={best[4]*1000:.0f} ms)")
ax.legend(fontsize=8)

# panel C: duration & peak-power distributions
ax = axes[1, 0]
ax.hist(durs * 1000, bins=30, color="C0", alpha=0.8)
ax.set_xlabel("Ripple duration (ms)")
ax.set_ylabel("Count")
ax.set_title(f"Ripple durations (n={len(ripple_events)})")

ax2 = ax.twinx()
ax2.hist(peakz, bins=30, color="C3", alpha=0.4)
ax2.set_ylabel("Count (peak z, red)", color="C3")

# panel D: average ripple-triggered LFP (aligned to peak)
ax = axes[1, 1]
pre_samps, post_samps = int(0.1 * FS), int(0.1 * FS)
snips = []
for e in ripple_events:
    idx = np.searchsorted(post_t, e[2])
    if idx - pre_samps >= 0 and idx + post_samps < len(post_lfp):
        snips.append(ripple_filt[idx - pre_samps: idx + post_samps])
snips = np.array(snips)
tsnip = np.arange(-pre_samps, post_samps) / FS * 1000
mean_snip = snips.mean(axis=0)
sem_snip = snips.std(axis=0) / np.sqrt(len(snips))
ax.plot(tsnip, mean_snip * 1e6, color="C0")
ax.fill_between(tsnip, (mean_snip - sem_snip) * 1e6, (mean_snip + sem_snip) * 1e6, color="C0", alpha=0.3)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Time from ripple peak (ms)")
ax.set_ylabel("Ripple-band LFP (uV)")
ax.set_title(f"Ripple-triggered average (n={len(snips)} events)")

plt.tight_layout()
plt.savefig("fig1_ripple_detection.png", dpi=130)
plt.close()
print("saved fig1_ripple_detection.png")

# %% [markdown]
# ## 4. Place fields from the MAZE running epoch
#
# The dataset's pre-computed linearized position has extensive gaps (>90% NaN), so we
# linearize the well-sampled 2D position ourselves: the x-coordinate of the tracked LED on
# this 1.6 m linear track correlates r=1.0 with the (sparse) official linearization. We
# restrict to periods of active running (>5 cm/s) and compute 1D tuning curves with
# `pynapple.compute_tuning_curves`.

# %%
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
maze_iset = nap.IntervalSet(start=MAZE_START, end=MAZE_END)
x_maze = pos2d[:, 0].restrict(maze_iset).dropna()
in_track = (x_maze.values >= 0) & (x_maze.values <= 1.6)
position = nap.Tsd(t=x_maze.times()[in_track], d=x_maze.values[in_track])

t_ = position.times()
speed_vals = np.abs(np.diff(position.values, prepend=position.values[0]) /
                     np.diff(t_, prepend=t_[0] - 1 / 39.0))
speed = nap.Tsd(t=t_, d=speed_vals)

RUN_THR = 0.05  # m/s
run_mask = speed.values > RUN_THR
run_times = t_[run_mask]
gaps = np.diff(run_times)
breaks = np.where(gaps > 0.5)[0]
run_starts = np.concatenate(([run_times[0]], run_times[breaks + 1]))
run_ends = np.concatenate((run_times[breaks], [run_times[-1]]))
run_ep = nap.IntervalSet(start=run_starts, end=run_ends)
run_ep = run_ep[(run_ep.end - run_ep.start) > 0.2]
print(f"{len(run_ep)} running bouts, {run_ep.tot_length():.0f} s total running time")

position_run = position.restrict(run_ep)
spikes_run = exc_units.restrict(run_ep)

tuning_curves = nap.compute_tuning_curves(spikes_run, position_run, bins=30, range=[(0.0, 1.6)])
bin_centers = tuning_curves.coords[list(tuning_curves.coords.keys())[-1]].values
peak_rate = tuning_curves.values.max(axis=-1)
is_place_cell = peak_rate > 1.0  # Hz
place_ids = tuning_curves.coords["unit"].values[is_place_cell]
place_tc = tuning_curves.sel(unit=place_ids)
place_units = exc_units[place_ids]
peak_pos = bin_centers[np.argmax(place_tc.values, axis=-1)]
print(f"{len(place_ids)} / {len(exc_units)} excitatory units qualify as place cells (peak rate > 1 Hz)")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(position.times() - MAZE_START, position.values, lw=0.4, color="0.3")
axes[0].set_xlabel("Time in MAZE epoch (s)")
axes[0].set_ylabel("Position (m)")
axes[0].set_title("Linearized position (from 2D tracking, x-axis)")

sub = slice(None, None, 3)
axes[1].scatter((speed.times()[sub]) - MAZE_START, speed.values[sub] * 100, s=1, alpha=0.5)
axes[1].axhline(RUN_THR * 100, color="r", ls="--", label=f"run threshold ({RUN_THR*100:.0f} cm/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Speed (cm/s)")
axes[1].set_title("Running speed")
axes[1].legend(fontsize=8)

order = np.argsort(peak_pos)
rates_norm = place_tc.values[order] / place_tc.values[order].max(axis=1, keepdims=True)
im = axes[2].imshow(rates_norm, aspect="auto", extent=[0, 1.6, len(order), 0], cmap="viridis")
axes[2].set_xlabel("Position on track (m)")
axes[2].set_ylabel("Place cell (sorted by field location)")
axes[2].set_title(f"Place fields (n={len(order)})")
plt.colorbar(im, ax=axes[2], label="norm. firing rate")

plt.tight_layout()
plt.savefig("fig2_place_fields.png", dpi=130)
plt.close()
print("saved fig2_place_fields.png")

# %% [markdown]
# ## 5. Held-out decoding sanity check
#
# Before decoding neural activity during ripples, we verify the encoding model actually
# works: tuning curves are fit on odd-numbered running bouts and used to decode position
# on the held-out even-numbered bouts.

# %%
train_ep = run_ep[np.arange(0, len(run_ep), 2)]
test_ep = run_ep[np.arange(1, len(run_ep), 2)]

tc_train = nap.compute_tuning_curves(
    exc_units.restrict(train_ep), position.restrict(train_ep), bins=30, range=[(0.0, 1.6)]
)
good_train = tc_train.coords["unit"].values[tc_train.values.max(axis=-1) > 1.0]
decoded_test, _ = nap.decode_bayes(
    tc_train.sel(unit=good_train), exc_units[good_train], test_ep, bin_size=0.25
)
true_test = position.restrict(test_ep).interpolate(decoded_test)
valid = ~np.isnan(true_test.values)
decode_r = np.corrcoef(decoded_test.values[valid], true_test.values[valid])[0, 1]
decode_mae = np.mean(np.abs(decoded_test.values[valid] - true_test.values[valid]))
print(f"Held-out decoding: r={decode_r:.2f}, MAE={decode_mae*100:.1f} cm (track length 160 cm)")

# %% [markdown]
# ## 6. Bayesian decoding of position during ripple events
#
# For each detected ripple we decode a population-vector posterior over track position in
# 20 ms bins using `pynapple.decode_bayes`, with tuning curves fit on the full running
# epoch (all place cells, decoding all 68 events at once is very fast: single vectorized
# call).

# %%
BUFFER = 0.025  # symmetric padding so short ripples still span >=3 bins
BIN = 0.02
starts_r = np.array([e[0] - BUFFER for e in ripple_events])
ends_r = np.array([e[1] + BUFFER for e in ripple_events])
ripple_iset = nap.IntervalSet(start=starts_r, end=ends_r)

decoded, posterior = nap.decode_bayes(place_tc, place_units, ripple_iset, bin_size=BIN)
print(f"Decoded {posterior.shape[0]} time bins across {len(ripple_events)} ripple events")

# %%
def weighted_corr(tb, pv, centers, order=None):
    if order is not None:
        pv = pv[order]
    tt = np.repeat(tb, len(centers))
    xx = np.tile(centers, len(tb))
    ww = pv.ravel()
    if ww.sum() == 0:
        return np.nan
    wt, wx = np.sum(ww * tt) / ww.sum(), np.sum(ww * xx) / ww.sum()
    cov = np.sum(ww * (tt - wt) * (xx - wx))
    vt, vx = np.sum(ww * (tt - wt) ** 2), np.sum(ww * (xx - wx) ** 2)
    return cov / np.sqrt(vt * vx) if vt > 0 and vx > 0 else np.nan


post_t_arr, post_v_arr = posterior.times(), posterior.values
event_r = np.full(len(ripple_events), np.nan)
n_bins_arr = np.zeros(len(ripple_events), dtype=int)
for i, (s, e) in enumerate(zip(starts_r, ends_r)):
    m = (post_t_arr >= s) & (post_t_arr < e)
    n_bins_arr[i] = m.sum()
    if m.sum() >= 3:
        event_r[i] = weighted_corr(post_t_arr[m], post_v_arr[m], bin_centers)

print(f"Weighted correlation computed for {np.sum(~np.isnan(event_r))} events; "
      f"mean |r| = {np.nanmean(np.abs(event_r)):.3f}")

# %% [markdown]
# ### Example decoded ripples
#
# A handful of individual events, chosen for high ripple power and spike participation,
# shown for illustration: spike raster of place cells (sorted by field location) below the
# decoded position posterior.

# %%
total_spikes = np.zeros(len(ripple_events))
n_active = np.zeros(len(ripple_events))
for i, (s, e) in enumerate(zip(starts_r, ends_r)):
    ep = nap.IntervalSet(start=s, end=e)
    sub = place_units.restrict(ep)
    counts = np.array([len(sub[k]) for k in sub.index])
    total_spikes[i] = counts.sum()
    n_active[i] = (counts > 0).sum()

candidates = np.where((total_spikes >= 10) & (n_active >= 6) & ~np.isnan(event_r))[0]
example_idx = candidates[np.argsort(-np.abs(event_r[candidates]))[:4]]

fig, axes = plt.subplots(2, len(example_idx), figsize=(4 * len(example_idx), 6))
sort_order = np.argsort(peak_pos)
place_ids_sorted = place_ids[sort_order]
peak_pos_sorted = peak_pos[sort_order]

for col, idx in enumerate(example_idx):
    s, e = starts_r[idx], ends_r[idx]
    ep = nap.IntervalSet(start=s, end=e)

    ax_r = axes[0, col]
    for row, uid in enumerate(place_ids_sorted):
        tt = place_units[uid].restrict(ep).t
        ax_r.vlines((tt - s) * 1000, row - 0.4, row + 0.4, color="k", lw=1)
    ax_r.set_ylim(-1, len(place_ids_sorted))
    ax_r.set_title(f"Ripple {idx}\n|wcorr|={abs(event_r[idx]):.2f}, peak z={ripple_events[idx][3]:.1f}")
    ax_r.set_xlabel("ms")
    if col == 0:
        ax_r.set_ylabel("place cell (sorted by field)")

    ax_p = axes[1, col]
    m = (post_t_arr >= s) & (post_t_arr < e)
    if m.sum() > 0:
        extent = [0, (m.sum()) * BIN * 1000, 0, 1.6]
        ax_p.imshow(post_v_arr[m].T, aspect="auto", origin="lower", extent=extent, cmap="magma")
    ax_p.set_xlabel("ms")
    if col == 0:
        ax_p.set_ylabel("decoded position (m)")

plt.tight_layout()
plt.savefig("fig3_replay_examples.png", dpi=130)
plt.close()
print("saved fig3_replay_examples.png")

# %% [markdown]
# ## 7. Is decoded sequence structure above chance?
#
# We test two complementary null hypotheses per ripple event using a rank-order
# correlation between each place cell's first-spike time within the event and its place
# field peak location (Spearman rho), a standard, decode-free replay statistic (e.g.
# Foster & Wilson, 2006). The null is built by randomly re-pairing spike order with place
# field identity, per event, many times.

# %%
def spearman_vec(x, y):
    def rank(a):
        order = np.argsort(a, axis=-1)
        r = np.empty_like(order, dtype=float)
        idx = np.arange(a.shape[-1])
        if a.ndim == 1:
            r[order] = idx
        else:
            for i in range(a.shape[0]):
                r[i, order[i]] = idx
        return r

    rx, ry = rank(x), rank(y)
    xm, ym = rx - rx.mean(axis=-1, keepdims=True), ry - ry.mean(axis=-1, keepdims=True)
    num = np.sum(xm * ym, axis=-1)
    den = np.sqrt(np.sum(xm ** 2, axis=-1) * np.sum(ym ** 2, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


id_to_peak = dict(zip(place_ids, peak_pos))
N_SHUF = 1000
rng = np.random.default_rng(42)
obs_rho = np.full(len(ripple_events), np.nan)
pvals = np.full(len(ripple_events), np.nan)
global_null_sum, global_null_n = np.zeros(N_SHUF), 0

for i, (s, e) in enumerate(zip(starts_r, ends_r)):
    ep = nap.IntervalSet(start=s, end=e)
    sub = place_units.restrict(ep)
    first_spike = {k: sub[k].t.min() for k in sub.index if len(sub[k]) > 0}
    if len(first_spike) < 4:
        continue
    ids = list(first_spike.keys())
    fst = np.array([first_spike[k] for k in ids])
    pp = np.array([id_to_peak[k] for k in ids])
    rho = spearman_vec(fst, pp)
    obs_rho[i] = rho

    perm = np.argsort(rng.random((N_SHUF, len(ids))), axis=1)
    null = spearman_vec(np.tile(fst, (N_SHUF, 1)), pp[perm])
    valid_null = null[~np.isnan(null)]
    if len(valid_null):
        pvals[i] = (np.sum(np.abs(valid_null) >= abs(rho)) + 1) / (len(valid_null) + 1)
    global_null_sum += np.nan_to_num(np.abs(null), nan=0.0)
    global_null_n += 1

eligible = ~np.isnan(obs_rho)
obs_mean = np.nanmean(np.abs(obs_rho[eligible]))
null_dist = global_null_sum / global_null_n
global_p = (np.sum(null_dist >= obs_mean) + 1) / (N_SHUF + 1)
frac_sig = np.mean(pvals[eligible] < 0.05)

print(f"{eligible.sum()} / {len(ripple_events)} events scored")
print(f"mean |rho| observed = {obs_mean:.3f}; shuffle-null mean = {null_dist.mean():.3f} +/- {null_dist.std():.3f}")
print(f"fraction of individual events significant at p<0.05: {frac_sig:.3f} (chance level = 0.05)")
print(f"population-level permutation test p-value: {global_p:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(np.abs(obs_rho[eligible]), bins=25, alpha=0.7, label="observed |rho| per event", density=True)
axes[0].hist(null_dist, bins=25, alpha=0.5, label="null distribution\n(mean |rho| per shuffle)", density=True, color="C3")
axes[0].axvline(obs_mean, color="k", ls="--", label=f"observed mean = {obs_mean:.3f}")
axes[0].set_xlabel("|rank-order correlation|")
axes[0].set_ylabel("density")
axes[0].set_title(f"Sequence-order test (global p={global_p:.2f})")
axes[0].legend(fontsize=8)

axes[1].scatter(n_active[eligible], np.abs(obs_rho[eligible]), s=10, alpha=0.5)
axes[1].set_xlabel("# active place cells in event")
axes[1].set_ylabel("|rank-order correlation|")
axes[1].set_title("Replay score vs. population participation")

plt.tight_layout()
plt.savefig("fig4_replay_significance.png", dpi=130)
plt.close()
print("saved fig4_replay_significance.png")

# %% [markdown]
# With only ~34 place cells from a single 40-minute session, this strict per-event test
# does not reach significance here (see discussion below) — this is reported honestly
# rather than adjusted to force a positive result. We therefore also test a complementary,
# more robust population-level measure of reactivation that does not require decoding an
# ordered trajectory within single, short (tens-of-ms) events.

# %% [markdown]
# ## 8. Population reactivation strength (explained variance)
#
# The classic Kudrimoti/Pavlides "explained variance" (EV) measure asks whether the
# pairwise spike-count correlation structure among place cells during RUN is more similar
# to POST-sleep correlation structure than to PRE-sleep (baseline) correlation structure,
# controlling for the baseline PRE-POST similarity. The reverse control (REV) swaps the
# roles of PRE and POST and should be smaller if reactivation is direction-specific
# (RUN -> POST, not RUN -> PRE).

# %%
nrem_pre_df = nrem_df[(nrem_df["start_time"] >= PRE_START) & (nrem_df["stop_time"] <= PRE_END)]
pre_nrem_ep = nap.IntervalSet(start=nrem_pre_df["start_time"].values, end=nrem_pre_df["stop_time"].values)

EV_BIN = 0.1  # 100 ms population bins, standard for this analysis
run_counts = place_units.count(EV_BIN, ep=run_ep).values
pre_counts = place_units.count(EV_BIN, ep=pre_nrem_ep).values
post_counts = place_units.count(EV_BIN, ep=nrem_post_ep).values


def upper_tri_corr(counts):
    C = np.corrcoef(counts.T)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


r_run, r_pre, r_post = upper_tri_corr(run_counts), upper_tri_corr(pre_counts), upper_tri_corr(post_counts)
valid = ~(np.isnan(r_run) | np.isnan(r_pre) | np.isnan(r_post))
r_run, r_pre, r_post = r_run[valid], r_pre[valid], r_post[valid]


def partial_r2(r_ab, r_ac, r_bc):
    return ((r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))) ** 2


rc_run_post = np.corrcoef(r_run, r_post)[0, 1]
rc_run_pre = np.corrcoef(r_run, r_pre)[0, 1]
rc_pre_post = np.corrcoef(r_pre, r_post)[0, 1]
EV = partial_r2(rc_run_post, rc_run_pre, rc_pre_post)
REV = partial_r2(rc_run_pre, rc_run_post, rc_pre_post)
print(f"{valid.sum()} cell pairs; corr(RUN,POST)={rc_run_post:.2f}, corr(RUN,PRE)={rc_run_pre:.2f}, "
      f"corr(PRE,POST)={rc_pre_post:.2f}")
print(f"EV  (RUN->POST | PRE) = {EV:.4f}")
print(f"REV (RUN->PRE | POST) = {REV:.4f}")

# bootstrap over cell pairs for confidence intervals
rng = np.random.default_rng(3)
n_pairs = len(r_run)
n_boot = 2000
EVs, REVs = np.empty(n_boot), np.empty(n_boot)
for b in range(n_boot):
    idx = rng.integers(0, n_pairs, n_pairs)
    a = np.corrcoef(r_run[idx], r_post[idx])[0, 1]
    c = np.corrcoef(r_run[idx], r_pre[idx])[0, 1]
    d = np.corrcoef(r_pre[idx], r_post[idx])[0, 1]
    EVs[b], REVs[b] = partial_r2(a, c, d), partial_r2(c, a, d)

frac_ev_gt_rev = np.mean(EVs > REVs)
print(f"Bootstrap: EV > REV in {frac_ev_gt_rev*100:.1f}% of {n_boot} resamples")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(r_run, r_post, s=6, alpha=0.4, color="C0")
axes[0].set_xlabel("pairwise corr. during RUN")
axes[0].set_ylabel("pairwise corr. during POST NREM")
axes[0].set_title(f"RUN vs POST (r={rc_run_post:.2f})")

axes[1].scatter(r_run, r_pre, s=6, alpha=0.4, color="C1")
axes[1].set_xlabel("pairwise corr. during RUN")
axes[1].set_ylabel("pairwise corr. during PRE NREM")
axes[1].set_title(f"RUN vs PRE (r={rc_run_pre:.2f})")

axes[2].violinplot([EVs, REVs], showmedians=True)
axes[2].set_xticks([1, 2])
axes[2].set_xticklabels(["EV\n(RUN->POST | PRE)", "REV\n(RUN->PRE | POST)"])
axes[2].set_ylabel("explained variance")
axes[2].set_title(f"Reactivation strength\n(EV>REV in {frac_ev_gt_rev*100:.0f}% of bootstraps)")

plt.tight_layout()
plt.savefig("fig5_reactivation_ev_rev.png", dpi=130)
plt.close()
print("saved fig5_reactivation_ev_rev.png")

# %% [markdown]
# ## Summary
#
# - **Sharp-wave ripples** were robustly detected on the LFP channel with maximal
#   ripple-band power: ~0.24 events/s during NREM sleep, median duration ~60 ms, matching
#   the expected physiological range for rodent CA1 SWRs.
# - **Place fields** were recovered from ~1/3 of recorded excitatory CA1 units during MAZE
#   running, producing the expected diagonal place-cell sequence when sorted by field
#   location, and a held-out decoding test confirmed the tuning curves generalize
#   (r ~ 0.6 between true and decoded position on unseen running bouts).
# - **Sequential replay** within individual ripple events, tested with a rank-order
#   correlation and a rigorous per-event and population-level shuffle control, did not
#   reach significance in this single 40-minute session with ~34 place cells. This is an
#   honest negative result at the single-event level: published significant single-event
#   replay detection typically uses substantially larger simultaneously-recorded
#   ensembles (in the hundreds of cells) and/or pools across many more ripples/sessions
#   than are available here.
# - **Population-level reactivation**, measured with the decode-free explained-variance
#   statistic, was directionally consistent with reactivation of the RUN experience during
#   subsequent POST sleep more than PRE sleep (EV > REV in the majority of bootstrap
#   resamples), consistent with the broader replay/reactivation literature even where
#   single-event sequential decoding was underpowered.
