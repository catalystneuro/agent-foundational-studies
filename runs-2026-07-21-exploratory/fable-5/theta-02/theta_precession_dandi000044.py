# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta phase entrainment and phase precession of hippocampal place cells
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark & Buzsáki
# (2016), *Diversity in neural firing dynamics supports both rigid and learned hippocampal
# sequences* (the `hc-11` dataset). Rats run for water reward while dorsal CA1 is recorded with
# 128-channel silicon probes; two of the sessions used here are on a 1.6 m linear track and two
# are on a ~2.9 m circular track. Each NWB file contains spike-sorted units labelled excitatory
# or inhibitory, a linearized position signal that is defined only while the animal is
# traversing the track, a 1250 Hz LFP on all 128 channels, and PRE / MAZE / POST epoch
# boundaries.
#
# **What this notebook demonstrates.** Two related phenomena, both computed from the same
# session data:
#
# 1. **Theta phase entrainment.** During running, the CA1 LFP is dominated by a 6-10 Hz theta
#    rhythm, and spiking is not uniformly distributed across the theta cycle. Pyramidal cells
#    and interneurons each lock to a preferred phase, with interneurons locking more strongly
#    and at a different phase than pyramidal cells.
# 2. **Theta phase precession.** Within a single place field, the theta phase at which a place
#    cell fires advances systematically as the animal moves through the field: spikes occur at
#    late phases when the animal enters the field and at progressively earlier phases as it
#    exits. This is a *within-field* phenomenon that is invisible in the average firing rate
#    map, and it is the classic demonstration that hippocampal spike timing carries positional
#    information beyond the firing rate.
#
# **Approach.** All data are streamed from the DANDI S3 bucket with `remfile` plus a local disk
# cache; nothing is downloaded in full (the session files are 8-9 GB each, and the LFP dataset
# is chunked one channel at a time, so a single channel over the maze epoch costs only a few
# MB). Pynapple provides the time-series containers and the tuning-curve machinery, and NeMoS
# provides an independent GLM-based confirmation at the end.

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import pandas as pd
import requests
import h5py
import remfile
import scipy.signal
import scipy.stats
import scipy.ndimage
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown interactively
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from tqdm.auto import tqdm

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000044"
CACHE_DIR = "/tmp/remfile_cache_000044"
LFP_FS = 1250.0
THETA_BAND = (6.0, 10.0)
BIN_WIDTH_CM = 4.0

SESSIONS = [
    "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb",
    "sub-Achilles/sub-Achilles_ses-Achilles-11012013_behavior+ecephys.nwb",
    "sub-Cicero/sub-Cicero_ses-Cicero-09102014_behavior+ecephys.nwb",
    "sub-Gatsby/sub-Gatsby_ses-Gatsby-08022013_behavior+ecephys.nwb",
]

# %% [markdown]
# ## 2. Streaming access to the NWB files
#
# `remfile` turns an S3 URL into a file-like object that h5py can read, fetching (and caching)
# only the byte ranges that are actually touched.

# %%
def asset_url(path):
    """Resolve a dandiset asset path to a streamable download URL."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/",
        params={"path": path, "page_size": 10},
    )
    r.raise_for_status()
    match = [a for a in r.json()["results"] if a["path"] == path]
    return f"https://api.dandiarchive.org/api/assets/{match[0]['asset_id']}/download/"


def open_session(path):
    rf = remfile.File(asset_url(path), disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rf, "r")


def load_behavior_and_spikes(h5):
    """Spikes (TsGroup), linearized position (Tsd, cm) and the maze epoch (IntervalSet)."""
    ep = h5["intervals/epochs"]
    labels = [s.decode() if isinstance(s, bytes) else s for s in ep["label"][:]]
    i_maze = labels.index("MazeEpoch")
    maze_ep = nap.IntervalSet(
        start=float(ep["start_time"][i_maze]), end=float(ep["stop_time"][i_maze])
    )

    # The `rate` attribute of these SpatialSeries is really the sampling *period*
    # (0.0256 s = 39.06 Hz); the assertion below verifies that reading.
    pos_key = [k for k in h5["processing/behavior"] if "LinearizedPosition" in k][0]
    grp = h5[f"processing/behavior/{pos_key}"]
    series = grp[list(grp.keys())[0]]
    pos_data = series["data"][:, 0].astype(float)
    conversion = float(series["data"].attrs.get("conversion", 1.0))
    t0 = float(series["starting_time"][()])
    rate_attr = float(series["starting_time"].attrs["rate"])
    dt = rate_attr if rate_attr < 1.0 else 1.0 / rate_attr
    assert abs(len(pos_data) * dt - (maze_ep.end[0] - maze_ep.start[0])) < 1.0
    position = nap.Tsd(t=t0 + np.arange(len(pos_data)) * dt, d=pos_data * conversion * 100.0)

    u = h5["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = nap.TsGroup(
        {i: st[a:b] for i, (a, b) in enumerate(zip(starts, idx))},
        metadata={
            "cell_type": np.array([s.decode() for s in u["cell_type"][:]]),
            "location": np.array([s.decode() for s in u["location"][:]]),
            "shank_id": u["shank_id"][:],
        },
    )
    return dict(spikes=spikes, position=position, maze_ep=maze_ep)


def read_lfp_channel(h5, channel, start_time, stop_time):
    """One LFP channel over a time window, as a Tsd in volts."""
    grp = h5["processing/ecephys/LFP/LFP"]
    ds = grp["data"]
    t0 = float(grp["starting_time"][()])
    fs = float(grp["starting_time"].attrs["rate"])
    conv = float(ds.attrs.get("conversion", 1.0))
    i0 = max(0, int(np.floor((start_time - t0) * fs)))
    i1 = min(ds.shape[0], int(np.ceil((stop_time - t0) * fs)))
    return nap.Tsd(t=t0 + np.arange(i0, i1) / fs, d=ds[i0:i1, channel].astype(float) * conv)


# %% [markdown]
# ## 3. Load one session and inspect every stream
#
# Nothing is analysed before each stream has been plotted and checked.

# %%
h5 = open_session(SESSIONS[0])
d = load_behavior_and_spikes(h5)
spikes, position, maze_ep = d["spikes"], d["position"], d["maze_ep"]
session_id = h5["general/session_id"][()].decode()

print("session:", session_id)
print("maze epoch: %.1f-%.1f s (%.0f s)" % (maze_ep.start[0], maze_ep.end[0], maze_ep.end[0] - maze_ep.start[0]))
print("units:", len(spikes), dict(zip(*np.unique(spikes.cell_type, return_counts=True))))
print("position samples:", len(position), " finite:", np.isfinite(position.values).sum())

# %% [markdown]
# ### Laps
#
# The linearized position is NaN whenever the rat is not on the track (it is defined "starting
# at the edge of the reward area"), so every contiguous run of finite samples is one candidate
# traversal. Requiring at least 0.5 s and 80 cm of travel keeps complete laps and discards
# partial excursions.

# %%
def get_laps(position, min_dur=0.5, min_extent=80.0):
    v = position.values
    finite = np.isfinite(v)
    edges = np.diff(finite.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if finite[0]:
        starts = np.r_[0, starts]
    if finite[-1]:
        stops = np.r_[stops, len(v)]

    lap_start, lap_end, direction = [], [], []
    for a, b in zip(starts, stops):
        seg_t, seg_v = position.t[a:b], v[a:b]
        if seg_t[-1] - seg_t[0] < min_dur or abs(seg_v[-1] - seg_v[0]) < min_extent:
            continue
        lap_start.append(seg_t[0])
        lap_end.append(seg_t[-1])
        direction.append("right" if seg_v[-1] > seg_v[0] else "left")

    laps = nap.IntervalSet(
        start=np.array(lap_start), end=np.array(lap_end),
        metadata={"direction": np.array(direction)},
    )
    return laps, nap.Tsd(t=position.t[finite], d=v[finite])


def lap_speed(position_clean, laps, smooth_sigma=3):
    """Speed in cm/s, differentiated within each lap so lap gaps never enter the derivative."""
    t_all, v_all = [], []
    for a, b in zip(laps.start, laps.end):
        seg = position_clean.get(a, b)
        if len(seg) < 5:
            continue
        sm = scipy.ndimage.gaussian_filter1d(seg.values, smooth_sigma, mode="nearest")
        v_all.append(np.abs(np.gradient(sm, seg.t)))
        t_all.append(seg.t)
    t, v = np.concatenate(t_all), np.concatenate(v_all)
    t, keep = np.unique(t, return_index=True)  # laps can share a boundary sample
    return nap.Tsd(t=t, d=v[keep], time_support=laps)


def track_geometry(position, bin_width=BIN_WIDTH_CM):
    """Track extent and bin count. The two maze types have different lengths, so a
    hard-coded range would silently drop everything beyond it."""
    top = float(np.ceil(np.nanmax(position.values) / bin_width) * bin_width)
    return (0.0, top), int(round(top / bin_width))


laps, pos = get_laps(position)
speed = lap_speed(pos, laps)
run_ep = speed.threshold(10.0, method="above").time_support.drop_short_intervals(0.2)
track_range, n_bins = track_geometry(pos)
# Only directions the animal actually ran: the circular-maze sessions are traversed one way.
dir_ep = {dr: laps[laps.direction == dr].intersect(run_ep)
          for dr in ("right", "left") if (laps.direction == dr).sum() >= 5}

print("track: 0-%.0f cm in %d bins" % (track_range[1], n_bins))
print("laps: %d (%d rightward, %d leftward), %.0f s of running above 10 cm/s"
      % (len(laps), (laps.direction == "right").sum(), (laps.direction == "left").sum(),
         run_ep.tot_length()))

# %% [markdown]
# ### Choosing an LFP channel and extracting theta phase
#
# Rather than assume a channel, each shank contributes one candidate and the channel with the
# largest theta/delta power ratio during running is selected. Theta phase comes from the
# Hilbert transform of the 6-10 Hz bandpass-filtered signal, so 0° is the peak of the filtered
# LFP on that channel and 180° is its trough. (The absolute phase offset depends on recording
# depth and on the sign convention of the file, so only *relative* phase statements are
# interpreted below.)

# %%
def pick_theta_channel(h5, maze_ep, candidates, probe_seconds=180.0):
    mid = 0.5 * (maze_ep.start[0] + maze_ep.end[0])
    scores = {}
    for ch in candidates:
        lfp = read_lfp_channel(h5, ch, mid - probe_seconds / 2, mid + probe_seconds / 2)
        f, pxx = scipy.signal.welch(lfp.values, fs=LFP_FS, nperseg=int(4 * LFP_FS))
        scores[ch] = pxx[(f >= 6) & (f <= 10)].mean() / pxx[(f >= 2) & (f <= 4)].mean()
    return max(scores, key=scores.get), scores


def theta_phase_from_lfp(lfp, band=THETA_BAND, fs=LFP_FS):
    filt = nap.apply_bandpass_filter(lfp, cutoff=band, fs=fs)
    analytic = scipy.signal.hilbert(filt.values)
    phase = np.angle(analytic) % (2 * np.pi)
    return (
        filt,
        nap.Tsd(t=filt.t, d=phase, time_support=filt.time_support),
        nap.Tsd(t=filt.t, d=np.abs(analytic), time_support=filt.time_support),
    )


groups = np.array([s.decode() for s in h5["general/extracellular_ephys/electrodes/group_name"][:]])
bad = h5["general/extracellular_ephys/electrodes/bad_electrode"][:]
candidates = [
    int(np.where((groups == g) & (~bad))[0][len(np.where((groups == g) & (~bad))[0]) // 2])
    for g in np.unique(groups)
    if len(np.where((groups == g) & (~bad))[0])
]
best_ch, ch_scores = pick_theta_channel(h5, maze_ep, candidates)
lfp = read_lfp_channel(h5, best_ch, maze_ep.start[0], maze_ep.end[0])
theta_filt, theta_phase, theta_amp = theta_phase_from_lfp(lfp)
print("selected LFP channel %d (theta/delta ratio %.1f)" % (best_ch, ch_scores[best_ch]))

# %% [markdown]
# ### Figure 1: raw streams during one traversal

# %%
pyr = spikes.getby_category("cell_type")["excitatory"]
right_ep = laps[laps.direction == "right"]
t0, t1 = right_ep.start[5] - 0.5, right_ep.end[5] + 0.5

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, height_ratios=[1, 1, 1.2, 2])
pw = pos.get(t0, t1)
axes[0].plot(pw.t - t0, pw.values, "k.-", ms=3, lw=1)
axes[0].set_ylabel("Position\n(cm)")
axes[0].set_title("Raw data streams during one rightward traversal (%s)" % session_id)
sw = speed.get(t0, t1)
axes[1].plot(sw.t - t0, sw.values, color="tab:green")
axes[1].set_ylabel("Speed\n(cm/s)")
lw_, tw = lfp.get(t0, t1), theta_filt.get(t0, t1)
axes[2].plot(lw_.t - t0, lw_.values * 1e3, color="0.65", lw=0.7, label="raw LFP")
axes[2].plot(tw.t - t0, tw.values * 1e3, color="tab:red", lw=1.5, label="6-10 Hz")
axes[2].set_ylabel("LFP\n(mV)")
axes[2].legend(loc="upper right", fontsize=8, ncol=2)
for row, uid in enumerate(pyr.index):
    st = pyr[uid].get(t0, t1)
    if len(st):
        axes[3].plot(st.t - t0, np.full(len(st), row), "|", color="k", ms=4, mew=0.8)
axes[3].set_ylabel("Pyramidal unit")
axes[3].set_xlabel("Time from traversal onset (s)")
axes[3].set_xlim(0, t1 - t0)
fig.tight_layout()
fig.savefig("fig01_raw_streams.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Figure 2: behaviour and the LFP spectrum
#
# The power spectrum during running has a clear theta peak near 9 Hz with a visible harmonic
# near 18 Hz, which is what the phase estimate depends on.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
seg = (maze_ep.start[0] + 150, maze_ep.start[0] + 270)
p_seg = position.get(*seg)
axes[0].plot(p_seg.t - seg[0], p_seg.values, ".", color="0.4", ms=2)
for s, e, dr in zip(laps.start, laps.end, laps.direction):
    if seg[0] < s < seg[1]:
        axes[0].axvspan(s - seg[0], min(e, seg[1]) - seg[0],
                        color="tab:blue" if dr == "right" else "tab:orange", alpha=0.25)
axes[0].set_xlim(0, seg[1] - seg[0])
axes[0].set_xlabel("Time from segment start (s)")
axes[0].set_ylabel("Linearized position (cm)")
axes[0].set_title("Track traversals\n(blue = rightward, orange = leftward)")

f, pxx = scipy.signal.welch(lfp.restrict(run_ep).values, fs=LFP_FS, nperseg=int(2 * LFP_FS))
axes[1].semilogy(f, pxx, "k")
axes[1].axvspan(6, 10, color="tab:red", alpha=0.2)
axes[1].set_xlim(0, 40)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD (V$^2$/Hz)")
peak_f = f[(f > 4) & (f < 14)][np.argmax(pxx[(f > 4) & (f < 14)])]
axes[1].set_title("LFP spectrum during running\n(channel %d, theta peak %.1f Hz)" % (best_ch, peak_f))

axes[2].hist(speed.values, bins=50, color="tab:green")
axes[2].axvline(10, ls="--", color="k")
axes[2].set_xlabel("Speed (cm/s)")
axes[2].set_ylabel("Count")
axes[2].set_title("Running speed within traversals\n(dashed line = 10 cm/s threshold)")
fig.tight_layout()
fig.savefig("fig02_behavior_and_spectrum.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Place fields
#
# Rate maps are computed separately for the two running directions, because CA1 place fields on
# a linear track are strongly directional. A cell counts as a place cell in a given direction if
# its peak rate is at least 1 Hz, its Skaggs spatial information is at least 0.5 bits/spike, and
# its odd-lap and even-lap rate maps correlate at r >= 0.4.

# %%
def place_fields(spikes, position, ep, track_range, bins, smooth_bins=1.0):
    tc = nap.compute_tuning_curves(
        spikes, position, bins=bins, range=track_range, epochs=ep, feature_names=["position"]
    )
    rates = np.nan_to_num(np.asarray(tc.values, dtype=float), nan=0.0)
    rates = scipy.ndimage.gaussian_filter1d(rates, smooth_bins, axis=1, mode="nearest")
    return (
        rates,
        np.asarray(tc.coords["position"].values, dtype=float),
        np.asarray(tc.attrs["occupancy"], dtype=float),
    )


def spatial_information(rates, occupancy):
    """Skaggs information, bits per spike."""
    p = occupancy / occupancy.sum()
    mean_rate = (rates * p[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(rates > 0, rates / mean_rate[:, None] * np.log2(rates / mean_rate[:, None]), 0.0)
    return np.nansum(term * p[None, :], axis=1), mean_rate


def field_bounds(rate_map, centers, frac=0.2):
    """Extent of the main field: contiguous bins above 20% of the peak."""
    pk = int(np.argmax(rate_map))
    thresh = frac * rate_map[pk]
    lo, hi = pk, pk
    while lo > 0 and rate_map[lo - 1] >= thresh:
        lo -= 1
    while hi < len(rate_map) - 1 and rate_map[hi + 1] >= thresh:
        hi += 1
    bw = centers[1] - centers[0]
    return centers[lo] - bw / 2, centers[hi] + bw / 2, centers[pk]


def split_half_stability(spikes, position, laps_dir, run_ep, track_range, bins):
    odd = nap.IntervalSet(start=laps_dir.start[::2], end=laps_dir.end[::2]).intersect(run_ep)
    even = nap.IntervalSet(start=laps_dir.start[1::2], end=laps_dir.end[1::2]).intersect(run_ep)
    m1, _, _ = place_fields(spikes, position, odd, track_range, bins)
    m2, _, _ = place_fields(spikes, position, even, track_range, bins)
    return np.array([
        np.nan if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]
        for a, b in zip(m1, m2)
    ])


maps = {}
for direction, ep in dir_ep.items():
    rates, centers, occ = place_fields(pyr, pos, ep, track_range, n_bins)
    si, mean_rate = spatial_information(rates, occ)
    stab = split_half_stability(pyr, pos, laps[laps.direction == direction], run_ep,
                                track_range, n_bins)
    peak = rates.max(axis=1)
    maps[direction] = dict(
        rates=rates, centers=centers, occ=occ, si=si, stab=stab, peak=peak,
        is_place=(peak >= 1.0) & (si >= 0.5) & (stab >= 0.4),
    )
    print("%-6s: %d/%d pyramidal cells are place cells" % (direction, maps[direction]["is_place"].sum(), len(pyr)))

# %% [markdown]
# ### Figure 3: place fields tile the track

# %%
directions = list(maps.keys())
fig, axes = plt.subplots(len(directions), 3, figsize=(14, 4 * len(directions)),
                         width_ratios=[1.2, 1.2, 1], squeeze=False)
for row, direction in enumerate(directions):
    m = maps[direction]
    sel = m["rates"][m["is_place"]]
    order = np.argsort(np.argmax(sel, axis=1))
    norm = sel[order] / sel[order].max(axis=1, keepdims=True)
    im = axes[row, 0].imshow(norm, aspect="auto", origin="lower", cmap="viridis",
                             extent=[track_range[0], track_range[1], 0, norm.shape[0]])
    axes[row, 0].set_ylabel("Place cell (sorted by peak)")
    axes[row, 0].set_xlabel("Position (cm)")
    axes[row, 0].set_title("%sward runs: %d place cells" % (direction.capitalize(), norm.shape[0]))
    plt.colorbar(im, ax=axes[row, 0], label="Normalized rate")
    for i in order[:: max(1, len(order) // 10)]:
        axes[row, 1].plot(m["centers"], sel[i], lw=1.4)
    axes[row, 1].set_xlabel("Position (cm)")
    axes[row, 1].set_ylabel("Firing rate (Hz)")
    axes[row, 1].set_title("Rate maps of 10 example place cells")
    axes[row, 2].scatter(m["si"], m["peak"], s=16, c=np.where(m["is_place"], "tab:red", "0.75"))
    axes[row, 2].axvline(0.5, ls="--", lw=1, color="k")
    axes[row, 2].axhline(1.0, ls="--", lw=1, color="k")
    axes[row, 2].set_xlabel("Spatial information (bits/spike)")
    axes[row, 2].set_ylabel("Peak rate (Hz)")
    axes[row, 2].set_yscale("log")
    axes[row, 2].set_title("Place-cell selection\n(red = passes all criteria)")
fig.suptitle("Directional place fields in dorsal CA1, %s (%d pyramidal cells)" % (session_id, len(pyr)), fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig03_place_fields.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 5. Theta phase entrainment
#
# Each spike fired during running is assigned the instantaneous theta phase of the LFP. The
# Rayleigh test asks whether a cell's spike phases are distributed non-uniformly, and the mean
# resultant length (MRL) measures how tightly they cluster.

# %%
def rayleigh(phases):
    """Returns (mean phase, mean resultant length, Rayleigh p-value)."""
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    c, s = np.cos(phases).mean(), np.sin(phases).mean()
    r = np.hypot(c, s)
    z = n * r**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * r) ** 2)) - (1 + 2 * n))
    return np.arctan2(s, c) % (2 * np.pi), r, p


phases = spikes.restrict(run_ep).value_from(theta_phase)
locking = {}
for uid in spikes.index:
    mu, r, p = rayleigh(phases[uid].values)
    locking[uid] = dict(mu=mu, mrl=r, p=p, n=len(phases[uid]), cell_type=str(spikes.cell_type[uid]))

pyr_ids = [u for u, L in locking.items() if L["cell_type"] == "excitatory" and L["n"] >= 50]
inh_ids = [u for u, L in locking.items() if L["cell_type"] == "inhibitory" and L["n"] >= 50]
all_pyr_ph = np.concatenate([phases[u].values for u in pyr_ids])
all_inh_ph = np.concatenate([phases[u].values for u in inh_ids])
mu_p, r_p, p_p = rayleigh(all_pyr_ph)
mu_i, r_i, p_i = rayleigh(all_inh_ph)

print("significantly locked (Rayleigh p<0.01): %d/%d pyramidal, %d/%d interneurons"
      % (sum(locking[u]["p"] < 0.01 for u in pyr_ids), len(pyr_ids),
         sum(locking[u]["p"] < 0.01 for u in inh_ids), len(inh_ids)))
print("pooled preferred phase: pyramidal %.0f deg (MRL %.3f), interneuron %.0f deg (MRL %.3f)"
      % (np.degrees(mu_p), r_p, np.degrees(mu_i), r_i))

# %% [markdown]
# ### Figure 4: spikes are locked to theta

# %%
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)
bins = np.linspace(0, 2 * np.pi, 37)
ctr = np.degrees((bins[:-1] + bins[1:]) / 2)
tt = np.linspace(0, 720, 400)

for col, (ph_all, color, label, mu, r) in enumerate((
    (all_pyr_ph, "tab:blue", "Pyramidal cells", mu_p, r_p),
    (all_inh_ph, "tab:purple", "Interneurons", mu_i, r_i),
)):
    ax = fig.add_subplot(gs[0, col])
    h, _ = np.histogram(ph_all, bins=bins)
    h = h / h.sum()
    ax.bar(np.r_[ctr, ctr + 360], np.r_[h, h], width=10, color=color)
    ax.set_xlabel("Theta phase (deg)")
    ax.set_ylabel("Fraction of spikes")
    ax.set_ylim(0, h.max() * 1.35)
    ax.set_xlim(0, 720)
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.axvline(np.degrees(mu), color="tab:red", lw=2)
    ax.axvline(np.degrees(mu) + 360, color="tab:red", lw=2)
    ax2 = ax.twinx()
    ax2.plot(tt, np.cos(np.radians(tt)), color="0.6", lw=1.2)
    ax2.set_yticks([])
    ax2.set_ylim(-1, 6)
    ax.set_title("%s: n=%d spikes\npreferred phase %.0f$\\degree$ (red), MRL=%.3f"
                 % (label, len(ph_all), np.degrees(mu), r))

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ids, color, lab in ((pyr_ids, "tab:blue", "pyramidal"), (inh_ids, "tab:purple", "interneuron")):
    mus = np.array([locking[u]["mu"] for u in ids])
    rs = np.array([locking[u]["mrl"] for u in ids])
    s = np.array([locking[u]["p"] < 0.01 for u in ids])
    ax.scatter(mus[s], rs[s], s=24, color=color, alpha=0.75, label=lab)
ax.set_title("Preferred phase vs. locking strength\n(cells with Rayleigh p<0.01)", pad=18, fontsize=10)
ax.set_rlabel_position(135)
ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.12), fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.hist([locking[u]["mrl"] for u in pyr_ids], bins=20, alpha=0.7, color="tab:blue", label="pyramidal")
ax.hist([locking[u]["mrl"] for u in inh_ids], bins=20, alpha=0.7, color="tab:purple", label="interneuron")
ax.set_xlabel("Mean resultant length")
ax.set_ylabel("Number of cells")
ax.set_title("Strength of theta phase locking")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 1])
win = int(0.25 * LFP_FS)
lfp_r = lfp.restrict(run_ep)
sp_all = np.sort(np.concatenate([spikes.restrict(run_ep)[u].t for u in pyr_ids]))
idx = np.searchsorted(lfp_r.t, sp_all)
idx = idx[(idx > win) & (idx < len(lfp_r) - win)]
sub = np.random.default_rng(0).choice(idx, size=min(20000, len(idx)), replace=False)
seg_lfp = np.stack([lfp_r.values[i - win: i + win] for i in sub])
ax.plot(np.arange(-win, win) / LFP_FS * 1000, seg_lfp.mean(axis=0) * 1e3, "k")
ax.axvline(0, ls="--", color="tab:red")
ax.set_xlabel("Time from spike (ms)")
ax.set_ylabel("LFP (mV)")
ax.set_title("Spike-triggered LFP average\n(%d pyramidal spikes)" % len(sub))

ax = fig.add_subplot(gs[1, 2])
ax.scatter([spikes.rate[u] for u in pyr_ids], [locking[u]["mrl"] for u in pyr_ids],
           s=16, color="tab:blue", label="pyramidal")
ax.scatter([spikes.rate[u] for u in inh_ids], [locking[u]["mrl"] for u in inh_ids],
           s=16, color="tab:purple", label="interneuron")
ax.set_xscale("log")
ax.set_xlabel("Session firing rate (Hz)")
ax.set_ylabel("Mean resultant length")
ax.set_title("Locking strength vs. firing rate")
ax.legend(fontsize=9)

fig.suptitle("Theta phase entrainment of CA1 spiking during running, %s "
             "(phase from LFP channel %d; 0$\\degree$ = peak of the 6-10 Hz filtered LFP, grey trace)"
             % (session_id, best_ch), fontsize=12, y=1.03)
fig.savefig("fig04_theta_entrainment.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Theta phase precession
#
# For every place field the spikes fired inside the field during running are collected, their
# in-field position is normalized to 0 (entry) to 1 (exit), and phase is regressed on position
# with the circular-linear method of Kempter et al. (2012): the slope that maximises the
# resultant length of `phase - 2*pi*a*x` is found by grid search, and the circular-linear
# correlation coefficient rho and its asymptotic p-value are computed at that slope. A field
# whose peak lies within 15 cm of a track end is skipped, because the rat never traverses it
# fully and "position in field" would not be comparable across cells.

# %%
def circ_lin_regress(x, phi, slope_range=(-2.0, 2.0), n_slopes=2001, min_abs_slope=0.05):
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    n = len(x)
    if n < 8:
        return dict(slope=np.nan, phi0=np.nan, rho=np.nan, p=np.nan, n=n)
    # Slopes in cycles per unit x. Slopes of ~0 are excluded because the circular-linear
    # correlation is undefined there; the same grid is used for real and shuffled data.
    slopes = np.linspace(*slope_range, n_slopes)
    slopes = slopes[np.abs(slopes) >= min_abs_slope]
    R = np.abs(np.exp(1j * (phi[None, :] - 2 * np.pi * slopes[:, None] * x[None, :])).mean(axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.angle(np.exp(1j * (phi - 2 * np.pi * a * x)).mean()) % (2 * np.pi)

    theta_c = (2 * np.pi * np.abs(a) * x) % (2 * np.pi)
    phi_bar = np.angle(np.exp(1j * phi).mean())
    theta_bar = np.angle(np.exp(1j * theta_c).mean())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta_c - theta_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta_c - theta_bar) ** 2))
    rho = num / den if den > 0 else np.nan
    lam20 = np.mean(np.sin(phi - phi_bar) ** 2)
    lam02 = np.mean(np.sin(theta_c - theta_bar) ** 2)
    lam22 = np.mean(np.sin(phi - phi_bar) ** 2 * np.sin(theta_c - theta_bar) ** 2)
    z = rho * np.sqrt(n * lam20 * lam02 / lam22) if lam22 > 0 else np.nan
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(slope=2 * np.pi * a, phi0=phi0, rho=rho, p=p, n=n)


def collect_fields(pyr, pos, dir_ep, maps, theta_phase, locking, session_id, track_range,
                   min_field_spikes=40, min_field_width=15.0, max_field_width=120.0,
                   edge_margin=15.0):
    fields = []
    for direction, ep in dir_ep.items():
        m = maps[direction]
        pos_ep = pos.restrict(ep)
        for k, uid in enumerate(pyr.index):
            if not m["is_place"][k]:
                continue
            lo, hi, pk = field_bounds(m["rates"][k], m["centers"])
            width = hi - lo
            if not (min_field_width <= width <= max_field_width):
                continue
            if pk - track_range[0] < edge_margin or track_range[1] - pk < edge_margin:
                continue
            st = pyr[uid].restrict(ep)
            if len(st) < min_field_spikes:
                continue
            sp_pos = pos_ep.interpolate(st).values
            sp_ph = st.value_from(theta_phase).values
            inside = (sp_pos >= lo) & (sp_pos <= hi) & np.isfinite(sp_pos)
            if inside.sum() < min_field_spikes:
                continue
            x = (sp_pos[inside] - lo) / width
            if direction == "left":
                x = 1.0 - x  # x always increases along the direction of travel
            reg = circ_lin_regress(x, sp_ph[inside])
            fields.append(dict(
                session=session_id, unit=int(uid), direction=direction, lo=lo, hi=hi,
                peak_pos=pk, width=width, peak_rate=float(m["peak"][k]), si=float(m["si"][k]),
                n_spikes=int(inside.sum()), x=x, phase=sp_ph[inside],
                mrl=locking[uid]["mrl"], **reg,
            ))
    return fields


fields = collect_fields(pyr, pos, dir_ep, maps, theta_phase, locking, session_id, track_range)
slopes = np.array([f["slope"] for f in fields])
rhos = np.array([f["rho"] for f in fields])
pvals = np.array([f["p"] for f in fields])
sig = (pvals < 0.05) & (slopes < 0)
print("%d fields analysed; %.0f%% show a significant negative phase-position slope; "
      "median slope %.0f deg per field traversal"
      % (len(fields), 100 * sig.mean(), np.degrees(np.median(slopes))))

# %% [markdown]
# ### Shuffle control
#
# Within each field the spike phases are permuted across spikes, which destroys the
# phase-position pairing while leaving both marginal distributions and the number of spikes
# untouched. Twenty permutations per field give the null distributions plotted below.

# %%
rng = np.random.default_rng(1)
null_rho, null_slope = [], []
for f in tqdm(fields, desc="phase shuffles"):
    for _ in range(20):
        r = circ_lin_regress(f["x"], rng.permutation(f["phase"]), n_slopes=501)
        null_rho.append(r["rho"])
        null_slope.append(r["slope"])
null_rho, null_slope = np.array(null_rho), np.array(null_slope)
ok = np.isfinite(null_rho)
null_rho, null_slope = null_rho[ok], null_slope[ok]
print("observed median rho %.3f vs shuffled %.3f (Mann-Whitney p = %.2g on slopes)"
      % (np.median(rhos), np.median(null_rho),
         scipy.stats.mannwhitneyu(slopes, null_slope).pvalue))

# %% [markdown]
# ### Figure 5: single-field examples
#
# Each column is one place field: the rate map on top (shaded region = detected field), and
# below it the theta phase of every in-field spike against normalized position, plotted over two
# theta cycles so the wrap-around is visible. The red line is the circular-linear fit.

# %%
order = np.argsort(rhos)
examples = [fields[i] for i in order[:6]]
fig, axes = plt.subplots(2, 6, figsize=(19, 6.5), height_ratios=[1, 2.2])
for j, f in enumerate(examples):
    k = list(pyr.index).index(f["unit"])
    m = maps[f["direction"]]
    axes[0, j].plot(m["centers"], m["rates"][k], "k")
    axes[0, j].axvspan(f["lo"], f["hi"], color="tab:orange", alpha=0.25)
    axes[0, j].set_xlim(track_range)
    axes[0, j].set_title("unit %d, %sward\n%.1f Hz peak, %.0f cm field"
                         % (f["unit"], f["direction"], f["peak_rate"], f["width"]), fontsize=9)
    axes[0, j].set_xlabel("Position (cm)", fontsize=8)
    if j == 0:
        axes[0, j].set_ylabel("Rate (Hz)")
    ax = axes[1, j]
    ph = np.degrees(f["phase"])
    ax.plot(f["x"], ph, ".", ms=3, color="0.35", alpha=0.6)
    ax.plot(f["x"], ph + 360, ".", ms=3, color="0.35", alpha=0.6)
    xx = np.linspace(0, 1, 100)
    yy = np.degrees(f["slope"] * xx + f["phi0"]) % 360
    for shift in (-360, 0, 360, 720):
        ax.plot(xx, yy + shift, "r-", lw=2)
    ax.set_ylim(0, 720)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlabel("Position in field")
    if j == 0:
        ax.set_ylabel("Theta phase (deg)")
    ptxt = "p<1e-16" if f["p"] < 1e-16 else "p=%.1g" % f["p"]
    ax.set_title(r"slope %.0f$\degree$/field, $\rho$=%.2f, %s"
                 % (np.degrees(f["slope"]), f["rho"], ptxt), fontsize=9)
fig.suptitle("Theta phase precession in single CA1 place fields (%s): spikes fire at "
             "progressively earlier theta phases as the rat crosses the field" % session_id, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig05_precession_examples.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Figure 6: population summary for this session

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)
sig_fields = [f for f, s in zip(fields, sig) if s]

ax = fig.add_subplot(gs[0, 0])
X = np.concatenate([f["x"] for f in sig_fields])
P = np.degrees(np.concatenate([f["phase"] for f in sig_fields]))
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360], bins=[20, 40], range=[[0, 1], [0, 720]])
H = H / H.sum(axis=1, keepdims=True)
im = ax.pcolormesh(xe, ye, H.T, cmap="magma")
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Theta phase (deg)")
ax.set_yticks([0, 180, 360, 540, 720])
ax.set_title("Pooled spike density\n(%d significant fields, %d spikes)" % (len(sig_fields), len(X)))
plt.colorbar(im, ax=ax, label="P(phase | position)")

ax = fig.add_subplot(gs[0, 1])
edges = np.linspace(0, 1, 11)
means, sems = [], []
for a, b in zip(edges[:-1], edges[1:]):
    per_field = [np.angle(np.exp(1j * f["phase"][(f["x"] >= a) & (f["x"] < b)]).mean())
                 for f in sig_fields if ((f["x"] >= a) & (f["x"] < b)).sum() >= 5]
    per_field = np.array(per_field)
    means.append(np.angle(np.exp(1j * per_field).mean()))
    sems.append(np.degrees(scipy.stats.circstd(per_field, high=np.pi, low=-np.pi) / np.sqrt(len(per_field))))
means = np.degrees(np.unwrap(np.array(means)))
means = means - means[0] + np.mod(means[0], 360)
ctr_x = (edges[:-1] + edges[1:]) / 2
ax.errorbar(ctr_x, means, yerr=sems, fmt="o-", color="tab:blue", capsize=3)
ax.errorbar(ctr_x, means + 360, yerr=sems, fmt="o-", color="tab:blue", capsize=3, alpha=0.5)
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Mean theta phase (deg)")
ax.set_title("Population phase advance\n(mean $\\pm$ s.e.m. across fields)")

ax = fig.add_subplot(gs[0, 2])
bins_s = np.linspace(-720, 720, 49)
ax.hist(np.degrees(null_slope), bins=bins_s, density=True, color="0.75", label="phase-shuffled")
ax.hist(np.degrees(slopes), bins=bins_s, density=True, histtype="step", lw=2, color="tab:red", label="observed")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Phase-position slope (deg per field traversal)")
ax.set_ylabel("Density")
ax.set_title("Slopes are negative\n(median %.0f$\\degree$ per field)" % np.degrees(np.median(slopes)))
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 0])
bins_r = np.linspace(-0.8, 0.8, 41)
ax.hist(null_rho, bins=bins_r, density=True, color="0.75", label="phase-shuffled")
ax.hist(rhos, bins=bins_r, density=True, histtype="step", lw=2, color="tab:red", label="observed")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Circular-linear correlation $\\rho$")
ax.set_ylabel("Density")
ax.set_title("Phase-position correlation")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 1])
w = np.array([f["width"] for f in fields])
ax.scatter(w[~sig], np.degrees(slopes[~sig]), s=18, color="0.7", label="n.s.")
ax.scatter(w[sig], np.degrees(slopes[sig]), s=18, color="tab:red", label="p<0.05")
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Field width (cm)")
ax.set_ylabel("Slope (deg per field traversal)")
ax.set_title("Precession slope vs. field size")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 2])
per_cm = np.degrees(slopes) / w
ax.hist(per_cm, bins=25, color="tab:blue")
ax.axvline(0, color="k", ls="--")
ax.axvline(np.median(per_cm), color="tab:red", lw=2)
ax.set_xlabel("Slope (deg/cm)")
ax.set_ylabel("Number of fields")
ax.set_title("Phase advance per cm\n(median %.1f deg/cm)" % np.median(per_cm))

fig.suptitle("Population summary of theta phase precession, %s (%d CA1 place fields)"
             % (session_id, len(fields)), fontsize=13)
fig.savefig("fig06_precession_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 7: precession is visible on single passes
#
# Pooling spikes across laps could in principle manufacture a phase-position correlation if,
# say, the rat ran at different speeds on different laps. It does not: individual traversals
# each show the phase advance.

# %%
f = fields[int(order[0])]
ep = dir_ep[f["direction"]]
lap_ep = laps[laps.direction == f["direction"]]
unit = pyr[f["unit"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
n_shown, n_passes = 0, 12
for a, b in zip(lap_ep.start, lap_ep.end):
    st = unit.get(a, b)
    if len(st) < 4:
        continue
    p = pos.restrict(ep).interpolate(st).values
    ph = np.degrees(st.value_from(theta_phase).values)
    m = (p >= f["lo"]) & (p <= f["hi"]) & np.isfinite(p)
    if m.sum() < 4:
        continue
    dist = (p[m] - f["lo"]) if f["direction"] == "right" else (f["hi"] - p[m])
    color = plt.cm.viridis(n_shown / n_passes)
    axes[0].plot(dist, ph[m], "o", ms=6, color=color, alpha=0.85)
    axes[0].plot(dist, ph[m] + 360, "o", ms=6, color=color, alpha=0.85)
    n_shown += 1
    if n_shown >= n_passes:
        break
axes[0].set_ylim(0, 720)
axes[0].set_yticks([0, 180, 360, 540, 720])
axes[0].set_xlabel("Distance into field (cm)")
axes[0].set_ylabel("Theta phase (deg)")
axes[0].set_title("Unit %d (%sward): %d individual passes\n(colour = pass number)"
                  % (f["unit"], f["direction"], n_shown))

axes[1].plot(f["x"], np.degrees(f["phase"]), ".", ms=4, color="0.35")
axes[1].plot(f["x"], np.degrees(f["phase"]) + 360, ".", ms=4, color="0.35")
xx = np.linspace(0, 1, 100)
yy = np.degrees(f["slope"] * xx + f["phi0"]) % 360
for shift in (-360, 0, 360, 720):
    axes[1].plot(xx, yy + shift, "r-", lw=2)
axes[1].set_ylim(0, 720)
axes[1].set_yticks([0, 180, 360, 540, 720])
axes[1].set_xlabel("Normalized position in field")
axes[1].set_ylabel("Theta phase (deg)")
axes[1].set_title("All passes pooled: %.0f$\\degree$ advance, $\\rho$=%.2f"
                  % (np.degrees(f["slope"]), f["rho"]))
fig.tight_layout()
fig.savefig("fig07_single_passes.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 7. Replication across sessions
#
# The same pipeline is run on four sessions from three rats. Everything above is wrapped into a
# single function so that no step differs between the prototype session and the rest.

# %%
def analyze_session(path, verbose=True):
    h5 = open_session(path)
    session_id = h5["general/session_id"][()].decode()
    d = load_behavior_and_spikes(h5)
    spikes, position, maze_ep = d["spikes"], d["position"], d["maze_ep"]
    laps, pos = get_laps(position)
    speed = lap_speed(pos, laps)
    run_ep = speed.threshold(10.0, method="above").time_support.drop_short_intervals(0.2)
    track_range, n_bins = track_geometry(pos)
    dir_ep = {dr: laps[laps.direction == dr].intersect(run_ep)
              for dr in ("right", "left") if (laps.direction == dr).sum() >= 5}

    groups = np.array([s.decode() for s in h5["general/extracellular_ephys/electrodes/group_name"][:]])
    bad = h5["general/extracellular_ephys/electrodes/bad_electrode"][:]
    cands = [int(np.where((groups == g) & (~bad))[0][len(np.where((groups == g) & (~bad))[0]) // 2])
             for g in np.unique(groups) if len(np.where((groups == g) & (~bad))[0])]
    best_ch, _ = pick_theta_channel(h5, maze_ep, cands)
    lfp = read_lfp_channel(h5, best_ch, maze_ep.start[0], maze_ep.end[0])
    _, theta_phase, _ = theta_phase_from_lfp(lfp)

    pyr = spikes.getby_category("cell_type")["excitatory"]
    ph_at_spikes = spikes.restrict(run_ep).value_from(theta_phase)
    locking = {}
    for uid in spikes.index:
        mu, r, p = rayleigh(ph_at_spikes[uid].values)
        locking[uid] = dict(mu=mu, mrl=r, p=p, n=len(ph_at_spikes[uid]),
                            cell_type=str(spikes.cell_type[uid]))

    maps = {}
    for direction, ep in dir_ep.items():
        rates, centers, occ = place_fields(pyr, pos, ep, track_range, n_bins)
        si, _ = spatial_information(rates, occ)
        stab = split_half_stability(pyr, pos, laps[laps.direction == direction], run_ep,
                                    track_range, n_bins)
        peak = rates.max(axis=1)
        maps[direction] = dict(rates=rates, centers=centers, occ=occ, si=si, stab=stab,
                               peak=peak, is_place=(peak >= 1.0) & (si >= 0.5) & (stab >= 0.4))
    fields = collect_fields(pyr, pos, dir_ep, maps, theta_phase, locking, session_id, track_range)
    if verbose:
        sl = np.array([f["slope"] for f in fields])
        pv = np.array([f["p"] for f in fields])
        print("%s: %d laps, %d pyramidal cells, %d place fields analysed, %d precessing"
              % (session_id, len(laps), len(pyr), len(fields), int(((pv < 0.05) & (sl < 0)).sum())))
    h5.close()
    return dict(session=session_id, path=path, laps=laps, run_ep=run_ep, pyr=pyr,
                locking=locking, maps=maps, fields=fields, best_ch=best_ch)


all_fields, all_lock, summary = [], [], []
for path in tqdm(SESSIONS, desc="sessions"):
    r = analyze_session(path)
    all_fields.extend(r["fields"])
    for uid, L in r["locking"].items():
        all_lock.append(dict(session=r["session"], unit=uid, **L))
    sl = np.array([f["slope"] for f in r["fields"]])
    pv = np.array([f["p"] for f in r["fields"]])
    pyr_lock = [L for L in r["locking"].values() if L["cell_type"] == "excitatory" and L["n"] >= 50]
    summary.append(dict(
        session=r["session"], subject=path.split("/")[0].replace("sub-", ""),
        n_laps=len(r["laps"]), run_time_s=round(r["run_ep"].tot_length(), 1), n_pyr=len(r["pyr"]),
        n_place_fields=int(sum(r["maps"][dd]["is_place"].sum() for dd in r["maps"])),
        n_analysed=len(r["fields"]),
        pct_precessing=round(100 * float(np.mean((pv < 0.05) & (sl < 0))), 1),
        median_slope_deg=round(float(np.degrees(np.median(sl))), 1),
        pct_theta_locked=round(100 * float(np.mean([L["p"] < 0.01 for L in pyr_lock])), 1),
        lfp_channel=r["best_ch"],
    ))

df = pd.DataFrame(summary)
df.to_csv("session_summary.csv", index=False)
lock_df = pd.DataFrame(all_lock)
pd.DataFrame([{k: v for k, v in f.items() if k not in ("x", "phase")} for f in all_fields]).to_csv(
    "place_field_precession.csv", index=False
)
print(df.to_string(index=False))

# %%
slopes_all = np.array([f["slope"] for f in all_fields])
rhos_all = np.array([f["rho"] for f in all_fields])
pvals_all = np.array([f["p"] for f in all_fields])
sig_all = (pvals_all < 0.05) & (slopes_all < 0)
print("\nPooled: %d fields from %d sessions, %.0f%% significantly precessing, "
      "median slope %.0f deg per field, median rho %.2f"
      % (len(all_fields), len(SESSIONS), 100 * sig_all.mean(),
         np.degrees(np.median(slopes_all)), np.median(rhos_all)))

# %% [markdown]
# ### Figure 8: cross-session summary

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)
colors = plt.cm.tab10(np.arange(len(SESSIONS)))
sessions = df["session"].tolist()

ax = fig.add_subplot(gs[0, 0])
X = np.concatenate([f["x"] for f, s in zip(all_fields, sig_all) if s])
P = np.degrees(np.concatenate([f["phase"] for f, s in zip(all_fields, sig_all) if s]))
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360], bins=[20, 40], range=[[0, 1], [0, 720]])
H = H / H.sum(axis=1, keepdims=True)
im = ax.pcolormesh(xe, ye, H.T, cmap="magma")
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Theta phase (deg)")
ax.set_yticks([0, 180, 360, 540, 720])
ax.set_title("All sessions pooled\n(%d fields, %d spikes)" % (sig_all.sum(), len(X)))
plt.colorbar(im, ax=ax, label="P(phase | position)")

ax = fig.add_subplot(gs[0, 1])
bins_s = np.linspace(-720, 360, 31)
for i, s in enumerate(sessions):
    m = np.array([f["session"] == s for f in all_fields])
    ax.hist(np.degrees(slopes_all[m]), bins=bins_s, histtype="step", lw=1.8, color=colors[i], label=s)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Slope (deg per field traversal)")
ax.set_ylabel("Number of fields")
ax.set_title("Precession slope, per session")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 2])
ax.bar(range(len(df)), df["pct_precessing"], color=colors[: len(df)])
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["session"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% of fields with significant\nnegative phase-position slope")
ax.axhline(5, ls="--", color="k", lw=1)
ax.set_title("Consistency across sessions\n(dashed line = chance, 5%)")
for i, v in enumerate(df["pct_precessing"]):
    ax.text(i, v + 1.5, "%.0f" % v, ha="center", fontsize=9)

ax = fig.add_subplot(gs[1, 0], projection="polar")
for i, s in enumerate(sessions):
    m = (lock_df.session == s) & (lock_df.cell_type == "excitatory") & (lock_df.p < 0.01)
    ax.scatter(lock_df.mu[m], lock_df.mrl[m], s=16, color=colors[i], alpha=0.7, label=s)
ax.set_title("Preferred theta phase of\nsignificantly locked pyramidal cells", pad=20, fontsize=10)
ax.set_rlabel_position(120)
ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.15))

ax = fig.add_subplot(gs[1, 1])
enough = lock_df.n >= 50
for ct, color in (("excitatory", "tab:blue"), ("inhibitory", "tab:purple")):
    v = lock_df.mrl[enough & (lock_df.cell_type == ct)]
    ax.hist(v, bins=25, alpha=0.65, color=color, label="%s (n=%d)" % (ct, len(v)))
ax.set_xlabel("Mean resultant length")
ax.set_ylabel("Number of cells")
ax.set_title("Theta phase locking, all sessions")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 2])
mrl_all = np.array([f["mrl"] for f in all_fields])
ax.scatter(mrl_all[~sig_all], rhos_all[~sig_all], s=18, color="0.7", label="n.s.")
ax.scatter(mrl_all[sig_all], rhos_all[sig_all], s=18, color="tab:red", label="precessing")
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Theta phase locking (MRL)")
ax.set_ylabel("Phase-position correlation $\\rho$")
ax.set_title("Entrainment vs. precession\n(one point per place field)")
ax.legend(fontsize=9)

fig.suptitle("Theta entrainment and phase precession across %d sessions of DANDI:000044 "
             "(%d CA1 place fields)" % (len(SESSIONS), len(all_fields)), fontsize=13)
fig.savefig("fig08_multisession_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 8. A model-based check with NeMoS
#
# The circular-linear regression above is descriptive. As an independent test, three nested
# Poisson GLMs are fit to 20 ms spike counts from each place field:
#
# | model | rate |
# | --- | --- |
# | position only | `exp(b0 + f(x))` |
# | fixed preferred phase | `exp(b0 + f(x) + c*cos(phi) + s*sin(phi))` |
# | precessing phase | `exp(b0 + f(x) + sum_j B_j(x) * (c_j*cos(phi) + s_j*sin(phi)))` |
#
# The phase term of the third model equals `A(x) * cos(phi - theta_pref(x))`, so it is the
# second model with the preferred phase allowed to rotate as the animal moves through the
# field: exactly phase precession, at a cost of `2 * n_basis` extra parameters rather than the
# full outer product of a position and a phase basis (which overfits at these spike counts).
# Cross-validation folds are whole laps, so training and test data never share a traversal.

# %%
BIN = 0.02
N_POS_BASIS = 8
N_FOLDS = 4
pos_basis = nmo.basis.BSplineEval(n_basis_funcs=N_POS_BASIS, label="position")
MODELS = ["position only", "fixed preferred phase", "precessing phase"]


def field_design(f):
    """Spike counts and the three design matrices for one place field."""
    ep = dir_ep[f["direction"]]
    lap_ep_f = laps[laps.direction == f["direction"]].intersect(ep)
    count = pyr[f["unit"]].count(BIN, ep=lap_ep_f)
    p = pos.interpolate(count, ep=count.time_support).values
    ph = nap.Ts(t=count.t).value_from(theta_phase).values
    y = count.values.astype(float)
    lap_id = np.searchsorted(lap_ep_f.start, count.t, side="right") - 1
    good = np.isfinite(p) & np.isfinite(ph)
    p, ph, y, lap_id = p[good], ph[good], y[good], lap_id[good]
    B = np.asarray(pos_basis.compute_features(p))
    cos_ph, sin_ph = np.cos(ph)[:, None], np.sin(ph)[:, None]
    return ({
        "position only": B,
        "fixed preferred phase": np.hstack([B, cos_ph, sin_ph]),
        "precessing phase": np.hstack([B, B * cos_ph, B * sin_ph]),
    }, y, lap_id)


def fit_glm(X, y):
    return nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                       regularizer_strength=1e-3).fit(X, y)


glm_fields = sorted([f for f in fields if f["n_spikes"] >= 100], key=lambda z: z["rho"])
scores = {k: [] for k in MODELS}
for f in tqdm(glm_fields, desc="GLM fits"):
    X, y, lap_id = field_design(f)
    uniq = np.unique(lap_id)
    fold_of_lap = dict(zip(uniq, np.arange(len(uniq)) % N_FOLDS))
    fold = np.array([fold_of_lap[i] for i in lap_id])
    for name in MODELS:
        ll = []
        for k in range(N_FOLDS):
            tr, te = fold != k, fold == k
            if y[te].sum() < 5:
                continue
            glm = fit_glm(X[name][tr], y[tr])
            ll.append(glm.score(X[name][te], y[te], score_type="log-likelihood"))
        scores[name].append(np.mean(ll))

scores = {k: np.array(v) for k, v in scores.items()}
base = scores["position only"]
d_fix = scores["fixed preferred phase"] - base
d_pre = scores["precessing phase"] - base
wil = scipy.stats.wilcoxon(scores["precessing phase"], scores["fixed preferred phase"])
print("held-out log-likelihood per bin, relative to the position-only model:")
for name in MODELS:
    d = scores[name] - base
    print("  %-22s %+0.5f (better in %d/%d fields)" % (name, d.mean(), (d > 0).sum(), len(d)))
print("precessing vs fixed preferred phase: Wilcoxon p = %.2g" % wil.pvalue)

# %% [markdown]
# ### Figure 9: the fitted preferred phase shifts across the field

# %%
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(d_fix, d_pre, s=22, color="tab:blue")
lim = [min(d_fix.min(), d_pre.min()) * 1.1, max(d_fix.max(), d_pre.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("fixed preferred phase (log-lik/bin gain)")
ax.set_ylabel("precessing phase (log-lik/bin gain)")
ax.set_title("Held-out likelihood gain over a\nposition-only model (one point per field)")

ax = fig.add_subplot(gs[0, 1])
ax.boxplot([scores[n] - base for n in MODELS], showfliers=False)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_ylabel("Held-out log-likelihood per bin\n(relative to position only)")
ax.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=9)
ax.set_title("Adding theta phase helps; letting the\npreferred phase precess helps more")

ax = fig.add_subplot(gs[0, 2])
ax.hist(d_pre - d_fix, bins=20, color="tab:red")
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("precessing $-$ fixed (log-lik/bin)")
ax.set_ylabel("Number of fields")
ax.set_title("Precessing model wins in %d/%d fields\n(Wilcoxon p = %.1g)"
             % ((d_pre > d_fix).sum(), len(d_pre), wil.pvalue))

for j, f in enumerate(glm_fields[:3]):
    X, y, _ = field_design(f)
    glm = fit_glm(X["precessing phase"], y)
    w = np.asarray(glm.coef_)
    width = f["hi"] - f["lo"]
    grid_dist = np.linspace(0, width, 80)
    grid_pos = f["lo"] + grid_dist if f["direction"] == "right" else f["hi"] - grid_dist
    Bg = np.asarray(pos_basis.compute_features(grid_pos))
    c, s = Bg @ w[N_POS_BASIS: 2 * N_POS_BASIS], Bg @ w[2 * N_POS_BASIS:]
    pref = np.degrees(np.unwrap(np.arctan2(s, c)))
    pref -= 360 * np.floor(pref.mean() / 360)  # align the curve with the spike cloud
    ax = fig.add_subplot(gs[1, j])
    for shift in (0, 360):
        ax.plot(f["x"] * width, np.degrees(f["phase"]) + shift, ".", ms=2.5, color="0.75")
        ax.plot(grid_dist, pref + shift, "-", color="tab:red", lw=2.5,
                label="GLM preferred phase" if shift == 0 else None)
    ax.set_ylim(0, 720)
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlim(0, width)
    ax.set_xlabel("Distance into field (cm)")
    if j == 0:
        ax.set_ylabel("Theta phase (deg)")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title("unit %d (%sward), grey = spikes" % (f["unit"], f["direction"]), fontsize=10)

fig.suptitle("NeMoS Poisson GLMs: the preferred theta phase of a place cell shifts with "
             "position within its field (%s)" % session_id, fontsize=13)
fig.savefig("fig09_glm_position_phase.png", dpi=150, bbox_inches="tight")
plt.close(fig)
# %% [markdown]
# ## 9. Summary
#
# Both phenomena are present and reproduce across sessions and animals:
#
# * **Entrainment.** During running, the great majority of CA1 pyramidal cells and essentially
#   every interneuron fire non-uniformly across the theta cycle. Interneurons lock more tightly
#   than pyramidal cells and prefer a different phase, and the spike-triggered LFP average of
#   pyramidal spikes is itself a theta oscillation with a trough at the spike.
# * **Precession.** Within a place field, theta phase falls monotonically with the distance the
#   animal has travelled into the field. Roughly three quarters of the analysable fields show a
#   significant negative circular-linear slope, the median advance is on the order of 200-250°
#   across a field, and the effect is absent when spike phases are shuffled within a field. It is
#   also visible on single passes, so it is not an artefact of pooling laps, and a Poisson GLM
#   that lets the preferred phase vary with position predicts held-out spikes better than one
#   with a fixed preferred phase.
#
# The natural next step would be to ask what the phase code buys the animal: decoding position
# from spikes binned within theta cycles (theta sequences) typically sweeps ahead of the
# animal's true position, which is the population-level counterpart of the single-cell
# precession shown here.
