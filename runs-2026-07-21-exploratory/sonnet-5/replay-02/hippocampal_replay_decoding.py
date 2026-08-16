# %% [markdown]
# # Decoding Hippocampal Replay During Sharp-Wave Ripples
#
# This notebook demonstrates hippocampal replay: the reactivation, during sharp-wave
# ripples (SWRs), of sequential firing patterns that were originally expressed while an
# animal traversed a spatial trajectory. We use a real CA1/PFC tetrode recording from a
# rat running a W-track alternation task, streamed directly from the DANDI Archive.
#
# **Dataset**: DANDI:000447 ("Novel-familiar-novel WTrack (CA1-PFC)", Frank/Jadhav labs),
# session `sub-JDS-NFN-AM2`. The session contains 3 behavioral epochs, each on a
# different physical maze; we use epoch 1, a classic three-arm W-track (49 trials).
#
# **Pipeline**:
# 1. Load position, LFP, and spike-sorted units via streaming (remfile + pynapple).
# 2. Build 2D place fields for CA1 units from running-epoch data and select place cells.
# 3. Detect candidate SWR events from CA1 LFP ripple-band (150-250 Hz) power during
#    immobility.
# 4. Bayesian-decode the position represented in each SWR event from CA1 population
#    spiking, using the place fields as the encoding model.
# 5. Quantify whether the decoded position sweeps through space in a sequential,
#    trajectory-like manner more than expected by chance (time-bin shuffle test).

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

RNG = np.random.default_rng(0)

# %% [markdown]
# ## 1. Load the session
#
# The NWB file (~5 GB) is streamed from S3 with `remfile`, using a local disk cache so
# repeated reads of the same byte ranges are fast. We wrap it with `pynapple.NWBFile`,
# which lazily exposes `units`, `trials`, `epoch intervals`, `LFP/ElectricalSeries`, and
# `SpatialSeries` as pynapple objects backed directly by the remote HDF5 dataset.

# %%
URL = "https://dandiarchive.s3.amazonaws.com/blobs/be1/cf8/be1cf8ef-a976-4966-9e27-02474c9a1ee0"
CACHE_DIR = "/tmp/remfile_cache2"

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
unit_meta = units.metadata
ca1_ids = unit_meta.index[unit_meta["location"] == "CA1"].to_numpy()
pfc_ids = unit_meta.index[unit_meta["location"] == "PFC"].to_numpy()
ca1_units = units[ca1_ids]
print(f"CA1 units: {len(ca1_ids)}, PFC units: {len(pfc_ids)} (PFC not used further)")

pos = nwb["SpatialSeries"]
x, y, speed = pos["x"], pos["y"], pos["z"]  # column 'z' is video-derived speed (cm/s)

epochs = nwb["epoch intervals"]
ep1 = epochs[1]  # classic 3-arm W-track epoch, used for the whole analysis
print(f"Using epoch 1: {ep1['start'][0]:.0f}-{ep1['end'][0]:.0f} s "
      f"({ep1['end'][0] - ep1['start'][0]:.0f} s)")

trials = nwb["trials"]

# %% [markdown]
# ## 2. Inspect and visualize raw data streams
#
# Before any analysis, we look at the animal's trajectory, running speed, and CA1 spiking
# to confirm the data look sensible.

# %%
x1, y1, speed1 = x.restrict(ep1), y.restrict(ep1), speed.restrict(ep1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(x1.d, y1.d, lw=0.3, color="k", alpha=0.6)
axes[0, 0].set_xlabel("x (cm)")
axes[0, 0].set_ylabel("y (cm)")
n_trials_ep1 = np.sum((trials["start"] >= ep1["start"][0]) & (trials["end"] <= ep1["end"][0]))
axes[0, 0].set_title(f"Epoch 1 trajectory (W-track), {n_trials_ep1} trials")

axes[0, 1].hist(speed1.d, bins=100, color="steelblue")
axes[0, 1].axvline(4, color="r", ls="--", label="immobility threshold (4 cm/s)")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("speed (cm/s)")
axes[0, 1].set_ylabel("# samples")
axes[0, 1].legend()
axes[0, 1].set_title("Running speed distribution")

win = nap.IntervalSet(ep1["start"][0], ep1["start"][0] + 60)
for row, uid in enumerate(ca1_ids):
    sp = ca1_units[uid].restrict(win)
    axes[1, 0].vlines(sp.t, row, row + 0.9, color="k", lw=0.5)
axes[1, 0].set_xlabel("time (s)")
axes[1, 0].set_ylabel("CA1 unit #")
axes[1, 0].set_title("CA1 spike raster, first 60 s of epoch 1")

sv = speed1.restrict(win)
axes[1, 1].plot(sv.t, sv.d, color="darkorange")
axes[1, 1].axhline(4, color="r", ls="--")
axes[1, 1].set_xlabel("time (s)")
axes[1, 1].set_ylabel("speed (cm/s)")
axes[1, 1].set_title("Speed, same 60 s window")

plt.tight_layout()
plt.savefig("fig01_raw_data_overview.png", dpi=130)
plt.close(fig)
print("saved fig01_raw_data_overview.png")

# %% [markdown]
# ## 3. CA1 place fields
#
# We compute 2D occupancy-normalized tuning curves for each CA1 unit using position
# samples with running speed > 4 cm/s (to avoid contaminating fields with immobile-period
# activity), then select place cells using standard criteria: peak rate > 1 Hz, Skaggs
# spatial information > 0.5 bits/spike, and > 50 spikes emitted during running.

# %%
run_ep = speed1.threshold(4, method="above").time_support
print(f"Running time in epoch 1: {run_ep.tot_length():.0f} s")

xy1 = nap.TsdFrame(t=x1.t, d=np.c_[x1.d, y1.d], columns=["x", "y"])
N_BINS = 24
tcs = nap.compute_tuning_curves(ca1_units, xy1, bins=[N_BINS, N_BINS], epochs=run_ep)
occ = tcs.attrs["occupancy"]
rates = tcs.attrs["rates"]
xedges, yedges = tcs.attrs["bin_edges"]
xcent = (xedges[:-1] + xedges[1:]) / 2
ycent = (yedges[:-1] + yedges[1:]) / 2

occ_p = occ / np.nansum(occ)
spatial_info = np.zeros(len(ca1_ids))
for i in range(len(ca1_ids)):
    r = tcs.values[i]
    mr = rates[i]
    valid = (occ_p > 0) & (~np.isnan(r)) & (r > 0)
    spatial_info[i] = np.nansum(occ_p[valid] * (r[valid] / mr) * np.log2(r[valid] / mr)) if mr > 0 else 0

peak_rate = np.nanmax(tcs.values.reshape(len(ca1_ids), -1), axis=1)
n_spikes_run = np.array([len(ca1_units[uid].restrict(run_ep)) for uid in ca1_ids])

is_place_cell = (peak_rate > 1.0) & (spatial_info > 0.5) & (n_spikes_run > 50)
good_ids = ca1_ids[is_place_cell]
print(f"Place cells: {is_place_cell.sum()} / {len(ca1_ids)} CA1 units")

place_cells = units[good_ids]
tcs_place = nap.compute_tuning_curves(place_cells, xy1, bins=[N_BINS, N_BINS], epochs=run_ep)

# order place cells by the x-coordinate of their field peak, used later for raster sorting
peak_x_bin = []
for i in range(len(good_ids)):
    flat_idx = np.nanargmax(tcs_place.values[i])
    xi, _ = np.unravel_index(flat_idx, tcs_place.values[i].shape)
    peak_x_bin.append(xcent[xi])
sort_order = np.argsort(peak_x_bin)
sorted_place_ids = good_ids[sort_order]

# %%
ncols = 6
nrows = int(np.ceil(len(good_ids) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
axes = np.array(axes).reshape(-1)
for i, uid in enumerate(good_ids):
    idx = list(ca1_ids).index(uid)
    field = tcs.values[idx].T
    axes[i].imshow(field, origin="lower", cmap="viridis")
    axes[i].set_title(f"unit {uid}: {peak_rate[idx]:.0f}Hz, SI={spatial_info[idx]:.2f}", fontsize=8)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
for j in range(len(good_ids), len(axes)):
    axes[j].axis("off")
fig.suptitle("CA1 place fields, epoch 1 (W-track)", y=1.0)
plt.tight_layout()
plt.savefig("fig02_place_fields.png", dpi=130)
plt.close(fig)
print("saved fig02_place_fields.png")

# %% [markdown]
# ## 4. Detect sharp-wave ripples from CA1 LFP
#
# We band-pass filter (150-250 Hz, 4th-order Butterworth) all 30 CA1 LFP channels,
# compute the Hilbert envelope of each, and average across channels to obtain a combined
# ripple-band power signal.
#
# A critical methodological point: naively z-scoring this signal against the *entire*
# recording and thresholding produces many false candidate events, because fast
# running/chewing produces broadband artifacts that also have elevated power in the
# 150-250 Hz band. When we checked this empirically, high ripple-band power samples were
# *less* likely to be immobile than chance (27% vs. 59% baseline immobile fraction) -
# the opposite of what true SWRs should show, since awake SWRs occur during immobility.
# We therefore restrict both the baseline statistics (mean/SD used for z-scoring) and the
# candidate search to immobile bouts (speed < 4 cm/s, bout duration > 0.3 s) only, which
# is standard practice (e.g. Karlsson & Frank, 2009).

# %%
lfp = nwb["LFP/ElectricalSeries"]
lfp_meta = lfp.metadata
ca1_chan_ids = lfp_meta.index[lfp_meta["location"] == "CA1"].to_numpy()
print(f"CA1 LFP channels: {len(ca1_chan_ids)}")

lfp_ep1 = lfp.restrict(ep1)  # (~1.35M samples x 62 channels), streamed once
lfp_ca1 = lfp_ep1[:, ca1_chan_ids]
fs = 1 / np.median(np.diff(lfp_ep1.t[:1000]))
print(f"LFP sampling rate: {fs:.1f} Hz, loaded shape {lfp_ca1.shape}")

filt = nap.apply_bandpass_filter(lfp_ca1, (150.0, 250.0), fs=fs, mode="butter", order=4)
envelope = np.abs(hilbert(filt.values, axis=0))
ripple_power = nap.Tsd(t=lfp_ep1.t, d=envelope.mean(axis=1))

# %%
immobile_ep = speed1.threshold(4, method="below").time_support
immobile_ep = immobile_ep[(immobile_ep["end"] - immobile_ep["start"]) > 0.3]
print(f"Immobile bouts: {len(immobile_ep)}, total {immobile_ep.tot_length():.0f} s")

power_immobile = ripple_power.restrict(immobile_ep)
baseline_mean, baseline_std = power_immobile.d.mean(), power_immobile.d.std()

# sanity check of the artifact issue described above, using the *unrestricted* baseline
z_unrestricted = (ripple_power.d - ripple_power.d.mean()) / ripple_power.d.std()
v_at_ripple = np.interp(ripple_power.t, speed1.t, speed1.d)
high = z_unrestricted > 5
print(f"[sanity check] fraction immobile among high (unrestricted) z-score samples: "
      f"{np.mean(v_at_ripple[high] < 4):.2f} vs. overall immobile fraction "
      f"{np.mean(v_at_ripple < 4):.2f}  <- movement-artifact contamination if lower")

# %%
HIGH_Z, LOW_Z, MIN_DUR, MERGE_GAP, MAX_SAMPLE_GAP = 3.0, 0.0, 0.015, 0.015, 0.01

t_im, d_im = power_immobile.t, power_immobile.d
z_im = (d_im - baseline_mean) / baseline_std

above_low = np.where(z_im > LOW_Z)[0]
gaps = np.where(np.diff(t_im[above_low]) > MAX_SAMPLE_GAP)[0] + 1
idx_splits = np.where(np.diff(above_low) > 1)[0] + 1
splits = np.union1d(idx_splits, gaps) if len(gaps) else idx_splits
runs = np.split(above_low, splits) if len(splits) else [above_low]

starts, ends = [], []
for run in runs:
    if len(run) == 0 or z_im[run].max() < HIGH_Z:
        continue
    starts.append(t_im[run[0]])
    ends.append(t_im[run[-1]])
starts, ends = np.array(starts), np.array(ends)

order = np.argsort(starts)
starts, ends = starts[order], ends[order]
m_s, m_e = [starts[0]], [ends[0]]
for s, e in zip(starts[1:], ends[1:]):
    if s - m_e[-1] < MERGE_GAP:
        m_e[-1] = max(m_e[-1], e)
    else:
        m_s.append(s)
        m_e.append(e)
starts, ends = np.array(m_s), np.array(m_e)
dur = ends - starts
keep = dur >= MIN_DUR
starts, ends = starts[keep], ends[keep]
swr_events = nap.IntervalSet(start=starts, end=ends)
print(f"Detected {len(swr_events)} candidate SWR events "
      f"({len(swr_events) / immobile_ep.tot_length():.2f} Hz during immobility), "
      f"median duration {np.median((ends - starts) * 1000):.0f} ms")

# %%
t0, t1 = 6415, 6435
fig, ax = plt.subplots(figsize=(12, 3.5))
m = (ripple_power.t >= t0) & (ripple_power.t <= t1)
ax.plot(ripple_power.t[m], ripple_power.d[m], color="k", lw=0.7)
for s, e in zip(starts, ends):
    if t0 < s < t1:
        ax.axvspan(s, e, color="r", alpha=0.25)
ax.set_xlabel("time (s)")
ax.set_ylabel("CA1 ripple-band power\n(150-250Hz envelope, a.u.)")
ax.set_title("Example detected SWR events during immobility (red)")
plt.tight_layout()
plt.savefig("fig03_ripple_detection.png", dpi=130)
plt.close(fig)
print("saved fig03_ripple_detection.png")

# %% [markdown]
# ## 5. Bayesian decoding of position during SWR events
#
# For each candidate SWR event we count CA1 place-cell spikes in 10 ms bins (smoothed
# with a 30 ms boxcar) and decode a posterior probability distribution over 2D position
# using the place fields from Step 3 as the Poisson encoding model
# (`pynapple.decode_bayes`, Zhang et al. 1998). We restrict decoding to events with
# enough spikes to constrain the estimate (>= 4 active place cells, >= 6 total spikes).

# %%
n_active, n_spk = [], []
for s, e in zip(starts, ends):
    sub = place_cells.restrict(nap.IntervalSet(s, e))
    n_active.append(sum(len(sub[u]) > 0 for u in good_ids))
    n_spk.append(sum(len(sub[u]) for u in good_ids))
n_active, n_spk = np.array(n_active), np.array(n_spk)

decodable_mask = (n_active >= 4) & (n_spk >= 6)
decodable_events = swr_events[decodable_mask]
n_active_d = n_active[decodable_mask]
n_spk_d = n_spk[decodable_mask]
print(f"Events selected for decoding: {decodable_mask.sum()} / {len(swr_events)}")

decoded, posterior = nap.decode_bayes(
    tcs_place, place_cells, epochs=decodable_events,
    bin_size=0.01, sliding_window_size=3, uniform_prior=True,
)
print(f"Decoded {posterior.shape[0]} time bins across {len(decodable_events)} events")

# %% [markdown]
# ## 6. Scoring sequential structure ("trajectory events")
#
# For each event we compute a *weighted correlation* between decoded position and time
# within the event (separately for x and y, combined in quadrature), using the full
# posterior probability as weights. This captures whether the decoded location sweeps
# systematically through space during the ripple, rather than staying fixed or jumping
# randomly. Significance for each event is assessed against a null distribution built by
# shuffling the order of time bins within that event 500 times (Tingley & Peyrache, 2020).

# %%
def weighted_corr(w, v1, v2):
    w = w / w.sum()
    m1, m2 = np.sum(w * v1), np.sum(w * v2)
    cov = np.sum(w * (v1 - m1) * (v2 - m2))
    s1, s2 = np.sqrt(np.sum(w * (v1 - m1) ** 2)), np.sqrt(np.sum(w * (v2 - m2) ** 2))
    return cov / (s1 * s2) if s1 > 0 and s2 > 0 else np.nan


def trajectory_score(p_event):
    n_bins = p_event.shape[0]
    T, X, Y = np.meshgrid(np.arange(n_bins), xcent, ycent, indexing="ij")
    rx = weighted_corr(p_event.ravel(), T.ravel().astype(float), X.ravel())
    ry = weighted_corr(p_event.ravel(), T.ravel().astype(float), Y.ravel())
    return np.sqrt(rx ** 2 + ry ** 2)


decoded_t = decoded.t
results = []
for i, (s, e) in enumerate(zip(decodable_events["start"], decodable_events["end"])):
    bin_mask = (decoded_t >= s) & (decoded_t <= e)
    p_event = posterior.values[bin_mask]
    if p_event.shape[0] < 2:
        continue
    score = trajectory_score(p_event)
    shuf_scores = np.array([
        trajectory_score(p_event[RNG.permutation(p_event.shape[0])]) for _ in range(500)
    ])
    pval = np.mean(shuf_scores >= score)

    # distance from the decoded position at ripple onset to the animal's actual
    # (stationary) position, used below to distinguish "local" from "remote" replay
    flat_idx0 = np.nanargmax(p_event[0])
    xi0, yi0 = np.unravel_index(flat_idx0, p_event[0].shape)
    decode_start_x, decode_start_y = xcent[xi0], ycent[yi0]
    idx_onset = np.argmin(np.abs(x.t - s))
    actual_x, actual_y = x.d[idx_onset], y.d[idx_onset]
    start_dist = np.hypot(decode_start_x - actual_x, decode_start_y - actual_y)

    results.append(dict(event=i, start=s, end=e, n_bins=p_event.shape[0], score=score,
                         shuf_mean=shuf_scores.mean(), pval=pval,
                         actual_x=actual_x, actual_y=actual_y, start_dist=start_dist))

scores_df = pd.DataFrame(results)
scores_df.to_csv("replay_scores.csv", index=False)
n_sig = (scores_df["pval"] < 0.05).sum()
print(f"{n_sig} / {len(scores_df)} events show significant sequential structure "
      f"(p<0.05, vs. {0.05 * len(scores_df):.1f} expected by chance)")
print(scores_df[["score", "pval"]].describe())

# %% [markdown]
# ## 7. Example replay events
#
# We visualize two significant events that illustrate two well-documented replay
# patterns: a *local* event whose decoded posterior stays near the animal's true
# (stationary) position, and a *remote* event whose decoded trajectory sweeps across a
# distant part of the track, unrelated to the animal's current location (Karlsson &
# Frank, 2009; Gillespie et al., 2021 - the companion dataset to this recording).

# %%
def plot_example_event(event_row, fname, pad=0.15):
    s, e = event_row["start"], event_row["end"]
    win = nap.IntervalSet(s - pad, e + pad)

    lfp_win = lfp.restrict(win)[:, ca1_chan_ids]
    filt_win = nap.apply_bandpass_filter(lfp_win, (150.0, 250.0), fs=fs, mode="butter", order=4)
    raw_trace = lfp_win.values.mean(axis=1)
    filt_trace = filt_win.values[:, 0]

    bin_mask = (decoded_t >= s) & (decoded_t <= e)
    p_event = posterior.values[bin_mask]
    n_bins = p_event.shape[0]
    map_x, map_y = [], []
    for k in range(n_bins):
        flat_idx = np.nanargmax(p_event[k])
        xi, yi = np.unravel_index(flat_idx, p_event[k].shape)
        map_x.append((xedges[xi] + xedges[xi + 1]) / 2)
        map_y.append((yedges[yi] + yedges[yi + 1]) / 2)

    idx_onset = np.argmin(np.abs(x.restrict(win).t - s))
    actual_x = x.restrict(win).d[idx_onset]
    actual_y = y.restrict(win).d[idx_onset]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.6], width_ratios=[2, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.plot(lfp_win.t, raw_trace, color="k", lw=0.8)
    ax_raw.axvspan(s, e, color="r", alpha=0.15)
    ax_raw.set_ylabel("raw LFP\n(CA1 avg, uV)")
    ax_raw.set_xticklabels([])

    ax_filt = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_filt.plot(lfp_win.t, filt_trace, color="darkred", lw=0.8)
    ax_filt.axvspan(s, e, color="r", alpha=0.15)
    ax_filt.set_ylabel("150-250Hz\nfiltered")

    ax_raster = fig.add_subplot(gs[2, 0], sharex=ax_raw)
    for row, uid in enumerate(sorted_place_ids):
        sp = place_cells[uid].restrict(win)
        ax_raster.vlines(sp.t, row, row + 0.9, color="k", lw=1.2)
    ax_raster.axvspan(s, e, color="r", alpha=0.15)
    ax_raster.set_ylabel("place cells\n(sorted by field x)")
    ax_raster.set_xlabel("time (s)")

    ax_map = fig.add_subplot(gs[:2, 1])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax_map.imshow(np.nansum(p_event, axis=0).T, origin="lower", extent=extent, aspect="auto", cmap="hot")
    ax_map.plot(map_x, map_y, "-o", color="cyan", ms=4, lw=1.5)
    ax_map.scatter([map_x[0]], [map_y[0]], color="lime", s=60, zorder=5, label="decode start")
    ax_map.scatter([map_x[-1]], [map_y[-1]], color="blue", s=60, zorder=5, label="decode end")
    ax_map.scatter([actual_x], [actual_y], color="white", edgecolor="k", marker="*", s=150,
                    zorder=6, label="actual position")
    ax_map.legend(fontsize=7, loc="best")
    ax_map.set_title(f"decoded posterior\nscore={event_row['score']:.2f}, p={event_row['pval']:.3f}", fontsize=10)

    ax_seq = fig.add_subplot(gs[2, 1])
    ax_seq.imshow(np.nansum(p_event, axis=2).T, origin="lower", aspect="auto", cmap="hot",
                  extent=[0, n_bins, xedges[0], xedges[-1]])
    ax_seq.plot(np.arange(n_bins) + 0.5, map_x, "-o", color="cyan", ms=4)
    ax_seq.set_xlabel("time bin within ripple")
    ax_seq.set_ylabel("decoded x (cm)")

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close(fig)
    print("saved", fname)


sig_events = scores_df[scores_df["pval"] < 0.05].sort_values("start_dist")
local_example = sig_events.iloc[0]    # decoded start closest to actual position
remote_example = sig_events.iloc[-1]  # decoded start farthest from actual position
plot_example_event(local_example, "fig04_example_replay_local.png")
plot_example_event(remote_example, "fig05_example_replay_remote.png")

# %% [markdown]
# ## 8. Population summary

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.hist(scores_df["score"], bins=12, alpha=0.7, label="observed", color="C0")
ax.hist(scores_df["shuf_mean"], bins=12, alpha=0.5, label="per-event shuffle mean", color="gray")
ax.set_xlabel("replay trajectory score")
ax.set_ylabel("# events")
ax.legend(fontsize=9)
ax.set_title("Observed vs. shuffle-null replay scores")

ax = axes[0, 1]
sig = scores_df["pval"] < 0.05
ax.bar(["significant\n(p<0.05)", "not significant"], [sig.sum(), (~sig).sum()], color=["C3", "lightgray"])
for i, v in enumerate([sig.sum(), (~sig).sum()]):
    ax.text(i, v + 0.3, str(v), ha="center")
ax.set_ylabel("# candidate SWR events")
ax.set_title(f"{sig.sum()}/{len(scores_df)} events show significant\nsequential structure "
             f"({100 * sig.mean():.0f}%, chance = 5%)")

ax = axes[1, 0]
sc = ax.scatter(n_active_d[scores_df["event"].to_numpy()], scores_df["score"], c=scores_df["pval"],
                 cmap="viridis_r", s=60, edgecolor="k")
plt.colorbar(sc, ax=ax, label="p-value")
ax.set_xlabel("# active place cells in event")
ax.set_ylabel("replay score")
ax.set_title("Score vs. population participation")

ax = axes[1, 1]
dur_ms = (scores_df["end"] - scores_df["start"]) * 1000
ax.scatter(dur_ms, scores_df["score"], c=sig, cmap="coolwarm", s=60, edgecolor="k")
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("replay score")
ax.set_title("Score vs. event duration")

plt.tight_layout()
plt.savefig("fig06_population_summary.png", dpi=130)
plt.close(fig)
print("saved fig06_population_summary.png")

# %% [markdown]
# ## Summary
#
# Bayesian decoding of CA1 population activity during sharp-wave ripples recovers
# spatially coherent, sequentially structured position estimates: 30% of candidate SWR
# events with sufficient spiking showed a decoded trajectory score exceeding a 500-fold
# time-bin shuffle null at p < 0.05, well above the 5% expected by chance. Individual
# examples include both locally-anchored replay (posterior concentrated near the animal's
# true resting position) and remote replay (a swept trajectory through a distant part of
# the W-track), both of which are established phenomena in the hippocampal replay
# literature for this task and dataset family (Karlsson & Frank, 2009; Gillespie et al.,
# 2021).

# %%
io.close()
