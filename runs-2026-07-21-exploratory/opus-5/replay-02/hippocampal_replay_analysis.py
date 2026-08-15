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
# # Hippocampal replay: decoding the spatial trajectory represented during sharp-wave ripples
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark, A.D. and Buzsáki, G.
# (2016), *Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*,
# Science 351, 1440–1443. Bilateral silicon-probe recordings from dorsal CA1 in freely moving rats.
# Each session has the same structure: several hours of sleep (PRE), ~35 min of running back and
# forth on a track (MAZE), and several more hours of sleep (POST). The track is a 1.6 m linear maze
# in the walk-through session and a 2.8–2.9 m circular maze in the other three, and in both cases
# the file provides a linearised position coordinate. The NWB files contain spike-sorted units with
# a putative excitatory/inhibitory label, linearised position during track traversals, 128-channel
# LFP at 1250 Hz, and a curated Awake / Non-REM / REM state annotation.
#
# **Question.** During sharp-wave ripples (SWRs) in slow-wave sleep, hippocampal place cells are
# thought to re-activate in sequences that recapitulate trajectories the animal ran while awake.
# Here we test that directly: we build place-field templates from track running, detect SWRs in
# sleep, and ask whether the position decoded from the ~100 ms population burst inside a ripple
# sweeps continuously across the track.
#
# **Approach.**
#
# 1. Stream the NWB files from the DANDI S3 bucket with `remfile` + a local disk cache (no full downloads).
# 2. Build direction-specific place fields along the linearised track with Pynapple, and verify the decoder
#    by cross-validated decoding of the animal's actual position on held-out running laps.
# 3. Detect SWRs from the 140–230 Hz LFP envelope on the channel with the strongest ripple power,
#    restricted to Non-REM sleep.
# 4. Define candidate replay events as multi-unit population bursts that coincide with a detected
#    ripple, and decode position in 20 ms bins inside each event.
# 5. Score each event by the weighted correlation between decoded position and time, and test it
#    against two shuffles (place-field identity shuffle, posterior column-cycle shuffle).
# 6. Compare POST-run sleep against PRE-run sleep and against a calibration control in which the
#    unit-to-place-field assignment is permuted.

# %% [markdown]
# ## Setup

# %%
import os
import pickle

import h5py
import matplotlib
matplotlib.use("Agg")            # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from matplotlib.gridspec import GridSpec
from pynwb import NWBHDF5IO
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from tqdm import tqdm

FIGDIR = "figures"
CACHEDIR = "cache"
REMFILE_CACHE = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
for p in (FIGDIR, CACHEDIR, REMFILE_CACHE):
    os.makedirs(p, exist_ok=True)

DANDISET = "000044"
SESSIONS = {
    "Achilles_10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles_11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Cicero_09102014":   "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Gatsby_08282013":   "402f78e1-9e7d-486c-8822-93d538ed6ccc",
}
MAIN = "Achilles_10252013"       # session used for the detailed walk-through

# ---- analysis parameters -------------------------------------------------
NBINS = 50                       # position bins along the linearised track
RIP_BAND = (140, 230)            # ripple band (Hz)
RIP_PEAK_Z, RIP_EDGE_Z = 4.0, 1.0
RIP_MIN, RIP_MAX = 0.020, 0.250  # ripple duration limits (s)
MUA_PEAK_Z = 3.0                 # population-burst threshold
MUA_MIN, MUA_MAX = 0.050, 0.600  # population-burst duration limits (s)
BIN = 0.020                      # decoding bin (s)
MIN_BINS, MIN_CELLS = 5, 5       # minimum event size
N_SHUF = 500

# %% [markdown]
# ## Streaming access to the DANDI assets
#
# The NWB files are 5–9 GB each. `remfile` reads only the byte ranges that h5py asks for and keeps
# them in a local disk cache, so nothing is downloaded in full. The LFP dataset happens to be chunked
# as `(170221, 1)`, i.e. one chunk per channel, which makes it cheap to pull a single channel across
# the whole ~10 h recording.

# %%
def asset_url(asset_id, dandiset=DANDISET):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{asset_id}/download/",
        allow_redirects=False,
    )
    return r.headers["Location"].split("?")[0]


def load_session(session):
    """Open one session and pull out the streams we need."""
    h = h5py.File(remfile.File(asset_url(SESSIONS[session]),
                               disk_cache=remfile.DiskCache(REMFILE_CACHE)), "r")
    nwbfile = NWBHDF5IO(file=h, load_namespaces=True).read()
    nwb = nap.NWBFile(nwbfile)
    ep_df = nwbfile.epochs.to_dataframe()
    epochs = {r.label: nap.IntervalSet(start=r.start_time, end=r.stop_time) for r in ep_df.itertuples()}
    st = nwbfile.processing["behavior"]["states"].to_dataframe()
    states = {lab: nap.IntervalSet(start=g.start_time.values, end=g.stop_time.values)
              for lab, g in st.groupby("label")}
    udf = nwbfile.units.to_dataframe()
    lfp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    return dict(session=session, nwb=nwb, nwbfile=nwbfile, h5=h, epochs=epochs, states=states,
                udf=udf, units=nwb["units"], fs=lfp.rate, conv=lfp.conversion,
                lfp_ds=h["processing/ecephys/LFP/LFP/data"],
                pyr=udf.index[udf.cell_type == "excitatory"].values)


d = load_session(MAIN)
print(d["nwbfile"].epochs.to_dataframe())
print("\nunits:", d["udf"].cell_type.value_counts().to_dict(),
      "|", d["udf"].location.value_counts().to_dict())
print("LFP:", d["lfp_ds"].shape, "at", d["fs"], "Hz, chunks", d["lfp_ds"].chunks)
for k, v in d["states"].items():
    print(f"state {k:8s}: {v.tot_length():.0f} s in {len(v)} bouts")

# %% [markdown]
# ## Position on the linear track
#
# The linearised position channel is only defined while the animal is actually traversing the track
# (it is NaN while it sits at the reward wells), so the valid stretches directly give the individual
# traversals. We keep traversals that cover more than 60% of the track at more than 0.1 m/s and split
# them by running direction, because CA1 place fields on a linear track are strongly directional.

# %%
def find_linearized(d):
    """The maze (and therefore the series name) differs between sessions."""
    for mod in d["nwbfile"].processing["behavior"].data_interfaces.values():
        for k in getattr(mod, "spatial_series", {}):
            if "Linearized" in k:
                return k
    raise KeyError("no linearised position series in this session")


def linear_position(d):
    ls = d["nwb"][find_linearized(d)]
    t, x = ls.t, np.asarray(ls.d).squeeze()
    g = ~np.isnan(x)
    tg, xg = t[g], x[g]
    dt = np.median(np.diff(t))
    brk = np.where(np.diff(tg) > 5 * dt)[0]
    s_i, e_i = np.r_[0, brk + 1], np.r_[brk, len(tg) - 1]
    n = e_i - s_i + 1
    s_i, e_i = s_i[n >= 5], e_i[n >= 5]
    laps = nap.IntervalSet(start=tg[s_i] - dt / 2, end=tg[e_i] + dt / 2)
    track = float(np.ceil(np.nanmax(xg) * 20) / 20)          # round up to the nearest 5 cm
    return nap.Tsd(t=tg, d=xg, time_support=laps), laps, dt, track


def split_laps(pos, laps, track):
    direction, speed, span = [], [], []
    for s, e in zip(laps.start, laps.end):
        p = pos.get(s, e)
        direction.append(np.sign(p.d[-1] - p.d[0]))
        speed.append(abs(p.d[-1] - p.d[0]) / max(p.t[-1] - p.t[0], 1e-9))
        span.append(np.ptp(p.d))
    direction, speed, span = map(np.array, (direction, speed, span))
    ok = (span > 0.6 * track) & (speed > 0.1)
    out = {name: (nap.IntervalSet(start=laps.start[ok & (direction == sgn)],
                                  end=laps.end[ok & (direction == sgn)]),
                  np.where(ok & (direction == sgn))[0])
           for name, sgn in [("rightward", 1), ("leftward", -1)]}
    return out, ok, direction, speed


pos, laps, dt, TRACK = linear_position(d)
dirs, lap_ok, lap_dir, lap_speed = split_laps(pos, laps, TRACK)
BINS = np.linspace(0, TRACK, NBINS + 1)
CENTERS = (BINS[:-1] + BINS[1:]) / 2
print(f"maze: {find_linearized(d)}, linearised length {TRACK:.2f} m")
print(f"{len(laps)} tracked traversals, {laps.tot_length():.0f} s of running; "
      f"{lap_ok.sum()} pass the criterion "
      f"({len(dirs['rightward'][1])} rightward / {len(dirs['leftward'][1])} leftward), "
      f"mean speed {lap_speed[lap_ok].mean():.2f} m/s")

# %% [markdown]
# ### Figure 1 — raw data
#
# Before anything else, look at each stream: the session structure, the shuttling on the track,
# a spike raster with position overlaid, and raw LFP during sleep.

# %%
fig, axs = plt.subplots(4, 1, figsize=(13, 11))
ep = d["epochs"]

ax = axs[0]
ls = d["nwb"]["1.6mLinearMazeLinearizedTimeSeries"]
ax.plot(ls.t / 3600, np.asarray(ls.d).squeeze(), ".", ms=0.5, color="k")
for lab, e, c in [("PRE", ep["PREEpoch"], "#8ecae6"), ("MAZE", ep["MazeEpoch"], "#ffb703"),
                  ("POST", ep["POSTEpoch"], "#90be6d")]:
    ax.axvspan(e.start[0] / 3600, e.end[0] / 3600, alpha=.35, color=c, label=lab)
ax.set(xlabel="time (h)", ylabel="linear position (m)",
       title=f"{MAIN}: session structure (PRE sleep / linear maze / POST sleep)")
ax.legend(loc="upper right", fontsize=8)

ax = axs[1]
ax.plot(pos.t, pos.d, "k-", lw=.7)
ax.set(xlabel="time (s)", ylabel="position (m)", title="Linearised position, track traversals only")

ax = axs[2]
sub = nap.IntervalSet(start=ep["MazeEpoch"].start[0] + 300, end=ep["MazeEpoch"].start[0] + 340)
for i, u in enumerate(d["pyr"]):
    sp = d["units"][u].restrict(sub)
    ax.plot(sp.t, np.full(len(sp), i), "|", ms=3, color="k", mew=.5)
ax.set(xlabel="time (s)", ylabel="unit #",
       title=f"Spike raster, {len(d['pyr'])} putative pyramidal cells (40 s of running)")
axb = ax.twinx()
axb.plot(pos.restrict(sub).t, pos.restrict(sub).d, color="#e63946", lw=1.2)
axb.set_ylabel("position (m)", color="#e63946")

ax = axs[3]
i0 = int((ep["POSTEpoch"].start[0] + 1000) * d["fs"])
seg = d["lfp_ds"][i0:i0 + int(2 * d["fs"]), :].astype(np.float32) * d["conv"] * 1e3
for k, ch in enumerate(range(0, 128, 16)):
    ax.plot(np.arange(seg.shape[0]) / d["fs"], seg[:, ch] + k * 1.2, lw=.5, color="k")
ax.set(xlabel="time (s)", ylabel="channel (offset, mV)", title="Raw LFP, 8 channels (POST sleep)")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/01_raw_data_overview.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Place fields and decoder validation
#
# Tuning curves are computed with Pynapple over 50 position bins (3.2 cm) separately for the two
# running directions and lightly smoothed. Cells with a peak rate above 1 Hz and more than
# 0.5 bits/spike of spatial information in at least one direction are kept as place cells.
#
# The decoder is the standard one-step Bayesian estimator with a flat prior,
#
# $$P(x \mid n) \;\propto\; \prod_i f_i(x)^{n_i}\; e^{-\tau \sum_i f_i(x)},$$
#
# where $f_i(x)$ is unit $i$'s place field and $n_i$ its spike count in a bin of width $\tau$.
# To confirm that fields and decoder are sound we hold out every second traversal, build the
# template from the rest, and decode the animal's real position in 250 ms bins.

# %%
def tuning_curves(group, pos, ep, track, sigma=1.0):
    tc = nap.compute_1d_tuning_curves(group=group, feature=pos.restrict(ep), nb_bins=NBINS,
                                      minmax=(0, track), ep=ep)
    return np.apply_along_axis(lambda v: gaussian_filter1d(v, sigma, mode="nearest"), 0, tc.values)


def spatial_info(rate, occ):
    p = occ / occ.sum()
    mr = (p * rate).sum()
    if mr <= 0:
        return 0.0
    r = np.where(rate > 0, rate, np.nan)
    return float(np.nansum(p * r / mr * np.log2(r / mr)))


def posterior(ll):
    ll = ll - ll.max(-1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(-1, keepdims=True)


def decode(N, tc, bin_size):
    """N: (T,U) counts, tc: (X,U) rates -> (T,X) posterior."""
    return posterior(N @ np.log(tc).T - bin_size * tc.sum(1)[None, :])


def place_fields(d, pos, dirs, dt, track):
    bins = np.linspace(0, track, NBINS + 1)
    pyr = d["units"][list(d["pyr"])]
    tc = {k: tuning_curves(pyr, pos, dirs[k][0], track) for k in dirs}
    occ = {k: np.histogram(pos.restrict(dirs[k][0]).d, bins)[0] * dt for k in dirs}
    si = {k: np.array([spatial_info(tc[k][:, i], occ[k]) for i in range(len(d["pyr"]))]) for k in dirs}
    pk = {k: tc[k].max(0) for k in dirs}
    is_pc = ((pk["rightward"] > 1) & (si["rightward"] > 0.5)) | \
            ((pk["leftward"] > 1) & (si["leftward"] > 0.5))
    idx = np.where(is_pc)[0]
    return {k: np.maximum(tc[k][:, idx], 1e-3) for k in tc}, idx, si, tc


TC, pc_idx, SI, tc_all = place_fields(d, pos, dirs, dt, TRACK)
pc_ids = d["pyr"][pc_idx]
pc_units = d["units"][list(pc_ids)]
print(f"{len(pc_ids)}/{len(d['pyr'])} pyramidal cells classified as place cells")

errs = []
for k in dirs:
    ids = dirs[k][1]
    train = nap.IntervalSet(start=laps.start[ids[0::2]], end=laps.end[ids[0::2]])
    test = nap.IntervalSet(start=laps.start[ids[1::2]], end=laps.end[ids[1::2]])
    tct = np.maximum(tuning_curves(pc_units, pos, train, TRACK), 1e-3)
    cnt = pc_units.count(0.25, ep=test)
    P = decode(np.asarray(cnt.values, float), tct, 0.25)
    errs.append(CENTERS[P.argmax(1)] - np.interp(cnt.t, pos.t, pos.d))
err = np.concatenate(errs)
med_err = float(np.median(np.abs(err)))
chance = np.median(np.abs(np.random.default_rng(0).uniform(0, TRACK, 100000)
                          - np.random.default_rng(1).uniform(0, TRACK, 100000)))
print(f"cross-validated decoding error on held-out laps: median {med_err*100:.1f} cm "
      f"(chance {chance*100:.0f} cm)")

# %% [markdown]
# ### Figure 2 — place fields

# %%
order = np.argsort(np.argmax(TC["rightward"], axis=0))
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 3, figure=fig, hspace=.5, wspace=.32)

for j, k in enumerate(["rightward", "leftward"]):
    ax = fig.add_subplot(gs[0, j])
    M = TC[k][:, order].T
    ax.imshow(M / np.maximum(M.max(1, keepdims=True), 1e-9), aspect="auto",
              extent=[0, TRACK, len(order), 0], cmap="viridis")
    ax.set(xlabel="position (m)", ylabel="place cell (sorted)", title=f"Place fields, {k} runs")

ax = fig.add_subplot(gs[0, 2])
for i in order[::4]:
    ax.plot(CENTERS, TC["rightward"][:, i], lw=1)
ax.set(xlabel="position (m)", ylabel="firing rate (Hz)", title="Example place fields (rightward)")

ax = fig.add_subplot(gs[1, 0])
ax.hist(SI["rightward"], bins=30, alpha=.6, label="rightward")
ax.hist(SI["leftward"], bins=30, alpha=.6, label="leftward")
ax.axvline(0.5, color="k", ls="--")
ax.legend(fontsize=8)
ax.set(xlabel="spatial information (bits/spike)", ylabel="count", title="Spatial information")

ax = fig.add_subplot(gs[1, 1])
ax.scatter(np.argmax(TC["rightward"], 0) * TRACK / NBINS,
           np.argmax(TC["leftward"], 0) * TRACK / NBINS, s=12, color="k")
ax.plot([0, TRACK], [0, TRACK], "r--", lw=.8)
ax.set(xlabel="peak position, rightward (m)", ylabel="peak position, leftward (m)",
       title="Directionality of place fields")

ax = fig.add_subplot(gs[1, 2])
ax.hist(np.abs(err) * 100, bins=40, color="#457b9d")
ax.axvline(med_err * 100, color="r")
ax.set(xlabel="|decoding error| (cm)", ylabel="time bins",
       title=f"Cross-validated decoding error\nmedian {med_err*100:.1f} cm (chance {chance*100:.0f} cm)")

ax = fig.add_subplot(gs[2, :])
ids = dirs["rightward"][1][1::2]
test = nap.IntervalSet(start=laps.start[ids], end=laps.end[ids])
tct = np.maximum(tuning_curves(pc_units, pos,
                               nap.IntervalSet(start=laps.start[dirs["rightward"][1][0::2]],
                                               end=laps.end[dirs["rightward"][1][0::2]]), TRACK), 1e-3)
cnt = pc_units.count(0.25, ep=test)
P = decode(np.asarray(cnt.values, float), tct, 0.25)
for s, e in zip(test.start[:14], test.end[:14]):
    m = (cnt.t >= s) & (cnt.t <= e)
    ax.plot(pos.get(s, e).t, pos.get(s, e).d, "k-", lw=1.5)
    ax.plot(cnt.t[m], CENTERS[P.argmax(1)][m], ".", color="#e63946", ms=6)
ax.plot([], [], "k-", label="true position")
ax.plot([], [], ".", color="#e63946", label="decoded")
ax.set(xlabel="time (s)", ylabel="position (m)",
       title="Decoded vs. true position on held-out running laps (250 ms bins)")
ax.legend(fontsize=8)

plt.savefig(f"{FIGDIR}/03_place_fields.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Sharp-wave ripple detection
#
# We scan all 128 channels over a long Non-REM bout and pick the one maximising ripple-band power
# relative to a 300–500 Hz noise band, then read that single channel for the whole recording. The
# ripple envelope is the smoothed Hilbert amplitude of the 140–230 Hz signal, z-scored against
# Non-REM. Events cross 1 SD, must peak above 4 SD, and last 20–250 ms.

# %%
def bandpass(x, lo, hi, fs, order=4):
    return sosfiltfilt(butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos"), x)


def pick_ripple_channel(d, probe_ep, seconds=300):
    fs, ds, conv = d["fs"], d["lfp_ds"], d["conv"]
    k = int(np.argmax(probe_ep.end - probe_ep.start))
    i0 = int(probe_ep.start[k] * fs)
    i1 = int(min(probe_ep.end[k], probe_ep.start[k] + seconds) * fs)
    rip, hf = np.zeros(ds.shape[1]), np.zeros(ds.shape[1])
    for ch in tqdm(range(ds.shape[1]), desc="scanning channels", mininterval=2):
        x = ds[i0:i1, ch].astype(np.float64) * conv * 1e6
        rip[ch] = np.std(bandpass(x, *RIP_BAND, fs))
        hf[ch] = np.std(bandpass(x, 300, 500, fs))
    return int(np.argmax(rip * rip / np.maximum(hf, 1e-9))), rip


def detect_ripples(d, channel, nrem, eps):
    fs = d["fs"]
    raw = d["lfp_ds"][:, channel].astype(np.float64) * d["conv"] * 1e6
    t = np.arange(len(raw)) / fs
    filt = bandpass(raw, *RIP_BAND, fs)
    env = gaussian_filter1d(np.abs(hilbert(filt)), 0.0075 * fs)
    ref = nap.Tsd(t=t, d=env).restrict(nrem).d
    z = nap.Tsd(t=t, d=(env - ref.mean()) / ref.std())
    out = {}
    for name, e in eps.items():
        zz = z.restrict(e)
        cand = zz.threshold(RIP_EDGE_Z, "above").time_support.merge_close_intervals(0.020)
        cand = cand[(cand.end - cand.start) >= RIP_MIN]
        keep, pk, pt = [], [], []
        for s, en in zip(cand.start, cand.end):
            seg = zz.get(s, en)
            if len(seg) == 0:
                continue
            m = int(np.argmax(seg.d))
            if seg.d[m] >= RIP_PEAK_Z and (en - s) <= RIP_MAX:
                keep.append((s, en)); pk.append(seg.d[m]); pt.append(seg.t[m])
        keep = np.array(keep)
        out[name] = dict(ep=nap.IntervalSet(start=keep[:, 0], end=keep[:, 1]),
                         peak_z=np.array(pk), peak_t=np.array(pt))
    return out, nap.Tsd(t=t, d=raw), nap.Tsd(t=t, d=filt), z


nrem = d["states"]["Non-REM"]
eps = {"PRE": nrem.intersect(d["epochs"]["PREEpoch"]),
       "POST": nrem.intersect(d["epochs"]["POSTEpoch"])}
channel, rip_profile = pick_ripple_channel(d, eps["POST"])
print("selected ripple channel:", channel)
rip, lfp_raw, lfp_filt, zenv = detect_ripples(d, channel, nrem, eps)
for k in rip:
    n = len(rip[k]["ep"])
    print(f"{k}: {n} ripples in {eps[k].tot_length():.0f} s Non-REM ({n/eps[k].tot_length():.2f} Hz), "
          f"median duration {np.median(rip[k]['ep'].end - rip[k]['ep'].start)*1e3:.0f} ms")

# %% [markdown]
# ### Figure 3 — ripple validation
#
# The ripple-triggered average should show the sharp wave, and the ripple-triggered average of the
# filtered trace should show a ~150 Hz oscillation that is phase-locked to the detection peak.

# %%
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 3, figure=fig, hspace=.55, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
ax.plot(rip_profile, ".-", lw=.6, ms=3)
ax.axvline(channel, color="r", ls="--")
ax.set(xlabel="channel", ylabel="140-230 Hz SD (uV)",
       title=f"Ripple-band power per channel\n(selected ch {channel})")

ax = fig.add_subplot(gs[0, 1:])
j = int(np.argmax(rip["POST"]["peak_z"]))
c = rip["POST"]["peak_t"][j]
seg, segf, segz = lfp_raw.get(c - .3, c + .3), lfp_filt.get(c - .3, c + .3), zenv.get(c - .3, c + .3)
ax.plot(seg.t - c, seg.d, color="k", lw=.8, label="raw LFP")
ax.plot(segf.t - c, segf.d - 350, color="#e63946", lw=.8, label="140-230 Hz")
ax.plot(segz.t - c, segz.d * 25 - 700, color="#457b9d", lw=.8, label="z-scored envelope")
ax.axvspan(rip["POST"]["ep"].start[j] - c, rip["POST"]["ep"].end[j] - c, color="#ffb703", alpha=.3)
ax.set(xlabel="time from ripple peak (s)", title="Example detected sharp-wave ripple", yticks=[])
ax.legend(fontsize=8, loc="upper right")

w = int(.1 * d["fs"])
idx = (rip["POST"]["peak_t"] * d["fs"]).astype(int)
idx = idx[(idx > w) & (idx < len(lfp_raw) - w)]
sel = idx[np.linspace(0, len(idx) - 1, min(2000, len(idx))).astype(int)]
snips = np.stack([lfp_raw.d[i - w:i + w] for i in sel])
snf = np.stack([lfp_filt.d[i - w:i + w] for i in sel])
tt = np.arange(-w, w) / d["fs"] * 1e3

ax = fig.add_subplot(gs[1, 0])
ax.plot(tt, snips.mean(0), "k")
ax.set(xlabel="ms from peak", ylabel="uV", title="Ripple-triggered LFP average")
ax = fig.add_subplot(gs[1, 1])
ax.plot(tt, snf.mean(0), color="#e63946")
ax.set(xlabel="ms from peak", title="Ripple-triggered 140-230 Hz average")
ax = fig.add_subplot(gs[1, 2])
ax.imshow(np.abs(hilbert(snf, axis=1))[:200], aspect="auto", extent=[tt[0], tt[-1], 200, 0], cmap="magma")
ax.set(xlabel="ms from peak", ylabel="event #", title="Ripple envelopes (200 events)")

ax = fig.add_subplot(gs[2, 0])
ax.hist((rip["POST"]["ep"].end - rip["POST"]["ep"].start) * 1e3, bins=40, color="#457b9d")
ax.set(xlabel="duration (ms)", ylabel="count", title=f"POST ripple durations (n={len(rip['POST']['ep'])})")
ax = fig.add_subplot(gs[2, 1])
ax.hist(rip["POST"]["peak_z"], bins=40, color="#457b9d")
ax.set(xlabel="peak envelope (z)", title="Ripple peak amplitude")
ax = fig.add_subplot(gs[2, 2])
ax.bar(["PRE", "POST"], [len(rip[k]["ep"]) / eps[k].tot_length() for k in ["PRE", "POST"]],
       color=["#8ecae6", "#90be6d"])
ax.set(ylabel="ripples / s", title="Ripple incidence in Non-REM")

plt.savefig(f"{FIGDIR}/02_ripple_detection.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Candidate replay events
#
# A ripple defined purely by its LFP envelope is often shorter than the population burst that carries
# the replayed sequence. We therefore define candidate events from the smoothed multi-unit rate of all
# pyramidal cells (z-scored within the sleep epoch): a contiguous suprathreshold-mean period of
# 50–600 ms that peaks above 3 SD. Only bursts that overlap a detected ripple are kept, which ties
# every candidate to an actual SWR.

# %%
def detect_pbes(d, ep, ripples):
    grp = d["units"][list(d["pyr"])]
    mua = grp.count(0.001, ep=ep)
    rate = gaussian_filter1d(np.asarray(mua.values, float).sum(1), 10) * 1000.0
    z = nap.Tsd(t=mua.t, d=(rate - rate.mean()) / rate.std(), time_support=ep)
    cand = z.threshold(0.0, "above").time_support
    cand = cand[(cand.end - cand.start) >= MUA_MIN]
    pk = np.array([z.get(s, e).d.max() for s, e in zip(cand.start, cand.end)])
    sel = (pk >= MUA_PEAK_Z) & ((cand.end - cand.start) <= MUA_MAX)
    pbe = nap.IntervalSet(start=cand.start[sel], end=cand.end[sel])
    r = ripples["ep"]
    ov = np.array([np.any((r.start < e) & (r.end > s)) for s, e in zip(pbe.start, pbe.end)])
    return nap.IntervalSet(start=pbe.start[ov], end=pbe.end[ov]), z, float(ov.mean())


pbe_post, mua_z, frac_ov = detect_pbes(d, eps["POST"], rip["POST"])
print(f"{len(pbe_post)} ripple-coincident population bursts in POST sleep "
      f"({100*frac_ov:.0f}% of bursts overlapped a ripple); "
      f"median duration {np.median(pbe_post.end - pbe_post.start)*1e3:.0f} ms")

# %% [markdown]
# ## Replay scoring
#
# Inside each candidate event the posterior is computed in 20 ms bins, once per direction template,
# and the event is scored by the **weighted correlation** between decoded position and time
# (the correlation of $x$ and $t$ under the posterior used as a weight). The template giving the
# larger $|r|$ is kept.
#
# Two null distributions are built for every event, and the max-over-templates operation is applied
# inside each null so the comparison is like for like:
#
# * **place-field identity shuffle** — the unit → place-field assignment is permuted, which preserves
#   each event's spike counts and the set of fields but destroys which cell codes for where;
# * **posterior column-cycle shuffle** — each time bin's posterior is circularly shifted in position,
#   which preserves per-bin sharpness but destroys the alignment across bins.
#
# An event is called replay when it beats **both** nulls at p < 0.05. As a calibration control the whole
# analysis is repeated with a single permuted template, which has no genuine position code and should
# therefore return the nominal 5%.

# %%
def wstats(P, tvec, centers):
    """Weighted correlation and slope of decoded position against time; P may be (..., T, X)."""
    sw = P.sum((-2, -1))
    mt = (P * tvec[:, None]).sum((-2, -1)) / sw
    mx = (P * centers[None, :]).sum((-2, -1)) / sw
    ct = tvec[:, None] - mt[..., None, None]
    cx = centers[None, :] - mx[..., None, None]
    cov = (P * ct * cx).sum((-2, -1)) / sw
    vt = (P * ct ** 2).sum((-2, -1)) / sw
    vx = (P * cx ** 2).sum((-2, -1)) / sw
    return cov / np.sqrt(vt * vx + 1e-12), cov / (vt + 1e-12)


def decode_events(group, templates, events, centers, n_shuf=N_SHUF, seed=0,
                  shuffle_observed=False, progress=None):
    X, U = NBINS, len(group)
    rng = np.random.default_rng(seed)
    templates = {k: np.maximum(v, 1e-3) for k, v in templates.items()}
    obs_tc = templates
    if shuffle_observed:
        q = rng.permutation(U)
        obs_tc = {k: v[:, q] for k, v in templates.items()}
    perms = np.stack([rng.permutation(U) for _ in range(n_shuf)])
    shuf = {k: (np.ascontiguousarray(np.log(v)[:, perms].transpose(2, 1, 0).reshape(U, n_shuf * X)),
                v[:, perms].sum(2).T.reshape(n_shuf * X)) for k, v in templates.items()}
    cnt = group.count(BIN, ep=events)
    cd = np.asarray(cnt.values, dtype=np.float64)
    ev = np.searchsorted(events.start, cnt.t, side="right") - 1
    rows, posteriors = [], {}
    it = tqdm(range(len(events)), desc=progress, mininterval=3.0) if progress else range(len(events))
    for i in it:
        N = cd[ev == i]
        T = N.shape[0]
        if T < MIN_BINS:
            continue
        nact = int((N.sum(0) > 0).sum())
        if nact < MIN_CELLS or N.sum() < 2 * MIN_CELLS:
            continue
        tvec = np.arange(T) * BIN
        best, id_max, cyc_max = None, np.zeros(n_shuf), np.zeros(n_shuf)
        for k in templates:
            P = decode(N, obs_tc[k], BIN)
            c, s = wstats(P, tvec, centers)
            if best is None or abs(c) > abs(best[1]):
                best = (k, float(c), float(s), P)
            lt, st = shuf[k]
            Ps = posterior((N @ lt - BIN * st[None, :]).reshape(T, n_shuf, X).transpose(1, 0, 2))
            id_max = np.maximum(id_max, np.abs(wstats(Ps, tvec, centers)[0]))
            shift = rng.integers(0, X, size=(n_shuf, T))
            idx = (np.arange(X)[None, None, :] - shift[:, :, None]) % X
            Pc = np.take_along_axis(np.broadcast_to(P, (n_shuf, T, X)), idx, axis=2)
            cyc_max = np.maximum(cyc_max, np.abs(wstats(Pc, tvec, centers)[0]))
        k, c, s, P = best
        com = P @ centers
        rows.append(dict(event=i, start=events.start[i], end=events.end[i],
                         dur=events.end[i] - events.start[i], nbins=T, ncells=nact,
                         nspikes=int(N.sum()), template=k, wcorr=c, slope=s, speed=abs(s),
                         p_id=(1 + (id_max >= abs(c)).sum()) / (n_shuf + 1),
                         p_cyc=(1 + (cyc_max >= abs(c)).sum()) / (n_shuf + 1),
                         start_pos=float(com[0]), end_pos=float(com[-1]),
                         span=float(abs(com[-1] - com[0]))))
        posteriors[i] = P
    df = pd.DataFrame(rows)
    df["sig"] = (df.p_id < 0.05) & (df.p_cyc < 0.05)
    return df, posteriors


# %% [markdown]
# ## Whole-session pipeline
#
# Everything above is wrapped into one function so the analysis can be repeated over sessions.
# Results are cached, so re-running the notebook is cheap.

# %%
def run_session(session):
    out = f"{CACHEDIR}/session_{session}.pkl"
    if os.path.exists(out):
        print(f"[{session}] loading cached result")
        return pickle.load(open(out, "rb"))
    print(f"=== {session}")
    d = load_session(session)
    nrem = d["states"]["Non-REM"]
    eps = {"PRE": nrem.intersect(d["epochs"]["PREEpoch"]),
           "POST": nrem.intersect(d["epochs"]["POSTEpoch"])}
    ch, prof = pick_ripple_channel(d, eps["POST"])
    rip, lfp_raw, lfp_filt, zenv = detect_ripples(d, ch, nrem, eps)
    pos, laps, dt, track = linear_position(d)
    bins = np.linspace(0, track, NBINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    dirs, ok, direction, speed = split_laps(pos, laps, track)
    TC, pc_idx, SI, _ = place_fields(d, pos, dirs, dt, track)
    pc_ids = d["pyr"][pc_idx]
    pc_units = d["units"][list(pc_ids)]

    errs = []
    for k in dirs:
        ids = dirs[k][1]
        tr = nap.IntervalSet(start=laps.start[ids[0::2]], end=laps.end[ids[0::2]])
        te = nap.IntervalSet(start=laps.start[ids[1::2]], end=laps.end[ids[1::2]])
        tct = np.maximum(tuning_curves(pc_units, pos, tr, track), 1e-3)
        cnt = pc_units.count(0.25, ep=te)
        P = decode(np.asarray(cnt.values, float), tct, 0.25)
        errs.append(centers[P.argmax(1)] - np.interp(cnt.t, pos.t, pos.d))
    med_err = float(np.median(np.abs(np.concatenate(errs))))
    print(f"  maze {find_linearized(d)} ({track:.2f} m); {len(pc_ids)}/{len(d['pyr'])} place cells; "
          f"cross-validated error {med_err*100:.1f} cm")

    res = dict(session=session, channel=ch, track=track, centers=centers,
               ripple_profile=prof, tc=TC, si=SI,
               pc_ids=pc_ids, n_pyr=len(d["pyr"]), med_err=med_err,
               mean_run_speed=float(speed[ok].mean()), events={}, posteriors={}, pbe_frac={},
               rip_summary={k: dict(n=len(rip[k]["ep"]),
                                    rate=len(rip[k]["ep"]) / eps[k].tot_length(),
                                    dur=float(np.median(rip[k]["ep"].end - rip[k]["ep"].start)))
                            for k in rip})
    for k in ["POST", "PRE"]:
        pbe, _, frac = detect_pbes(d, eps[k], rip[k])
        res["pbe_frac"][k] = frac
        df, posts = decode_events(pc_units, TC, pbe, centers, seed=hash(k) % 1000,
                                  progress=f"  decoding {k}")
        res["events"][k] = df
        if k == "POST":
            res["posteriors"] = posts
            res["pbe"] = (pbe.start, pbe.end)
        print(f"  {k}: {len(df)} analysable events, {df.sig.sum()} significant ({100*df.sig.mean():.1f}%)")
    dfc, _ = decode_events(pc_units, TC, nap.IntervalSet(start=res["pbe"][0], end=res["pbe"][1]),
                           centers, seed=99, shuffle_observed=True, progress="  decoding CONTROL")
    res["events"]["CONTROL"] = dfc
    print(f"  control: {len(dfc)} events, {dfc.sig.sum()} significant ({100*dfc.sig.mean():.1f}%)")

    j = int(np.argmax(rip["POST"]["peak_z"]))
    c = rip["POST"]["peak_t"][j]
    res["example_ripple"] = dict(t=lfp_raw.get(c - .3, c + .3).t - c,
                                 raw=lfp_raw.get(c - .3, c + .3).d,
                                 filt=lfp_filt.get(c - .3, c + .3).d,
                                 start=rip["POST"]["ep"].start[j] - c,
                                 end=rip["POST"]["ep"].end[j] - c)
    w = int(0.1 * d["fs"])
    idx = (rip["POST"]["peak_t"] * d["fs"]).astype(int)
    idx = idx[(idx > w) & (idx < len(lfp_raw) - w)][:2000]
    res["rip_trig"] = dict(raw=np.stack([lfp_raw.d[i - w:i + w] for i in idx]).mean(0),
                           t=np.arange(-w, w) / d["fs"])
    pickle.dump(res, open(out, "wb"))
    d["h5"].close()
    return res


R = [run_session(s) for s in SESSIONS]
res = [r for r in R if r["session"] == MAIN][0]
post, pre, ctrl = res["events"]["POST"], res["events"]["PRE"], res["events"]["CONTROL"]

# %% [markdown]
# ### Figure 4 — individual replay events
#
# For each of the twelve strongest events: the posterior over position across the burst (left) and
# the spikes of the same event with cells ordered by the location of their place field (right).
# A replayed trajectory appears as a continuous diagonal sweep in the posterior and as a diagonal
# band in the ordered raster.

# %%
sig = post[post.sig].copy()
top = sig.reindex(sig.wcorr.abs().sort_values(ascending=False).index).head(12)
pc_ids_main = res["pc_ids"]
pc_units_main = d["units"][list(pc_ids_main)]

fig = plt.figure(figsize=(15, 10.5))
gs = GridSpec(4, 6, figure=fig, hspace=.75, wspace=.35)
for j, (_, r) in enumerate(top.iterrows()):
    row, col = divmod(j, 3)
    P = res["posteriors"][int(r.event)]
    T = P.shape[0]
    ax = fig.add_subplot(gs[row, 2 * col])
    half = (res["centers"][1] - res["centers"][0]) / 2
    ax.pcolormesh(np.arange(T + 1) * BIN * 1e3,
                  np.r_[res["centers"] - half, res["centers"][-1] + half],
                  P.T, cmap="magma", shading="auto")
    ax.plot(np.arange(T) * BIN * 1e3 + 10, P @ res["centers"], "o-", color="#4cc9f0", ms=2.5, lw=1)
    ax.set_title(f"r={r.wcorr:+.2f}  {abs(r.slope):.1f} m/s\n{int(r.ncells)} cells, {int(r.dur*1e3)} ms",
                 fontsize=8, pad=3)
    ax.set_ylim(0, res["track"])
    ax.tick_params(labelsize=7)
    if col == 0:
        ax.set_ylabel("position (m)", fontsize=8)
    axr = fig.add_subplot(gs[row, 2 * col + 1])
    order = np.argsort(np.argmax(res["tc"][r.template], axis=0))
    for kk, ui in enumerate(order):
        sp = pc_units_main[pc_ids_main[ui]].get(r.start, r.end)
        if len(sp):
            axr.plot((sp.t - r.start) * 1e3, np.full(len(sp), kk), "|", ms=3, mew=.7, color="k")
    axr.set(ylim=(-1, len(order)), xlim=(0, r.dur * 1e3))
    axr.set_title("cells by field", fontsize=7, pad=3)
    axr.tick_params(labelsize=7)
    if row == 3:
        ax.set_xlabel("time (ms)", fontsize=8)
        axr.set_xlabel("time (ms)", fontsize=8)
fig.suptitle(f"{MAIN}: decoded position during sharp-wave ripples in POST-run sleep\n"
             "left = posterior P(position | spikes) in 20 ms bins, right = spikes ordered by place-field location",
             fontsize=12)
plt.savefig(f"{FIGDIR}/04_replay_examples.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 5 — replay statistics for the example session

# %%
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 3, figure=fig, hspace=.55, wspace=.32)

ax = fig.add_subplot(gs[0, 0])
for df, lab, c in [(post, "POST sleep", "#2a9d8f"), (pre, "PRE sleep", "#8ecae6"),
                   (ctrl, "cell-ID shuffled template", "#adb5bd")]:
    ax.hist(df.wcorr.abs(), bins=np.linspace(0, 1, 40), density=True, histtype="step", lw=1.8,
            color=c, label=lab)
ax.set(xlabel="|weighted correlation|", ylabel="density", title="Replay score distribution")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
labs, fr, lo, hi = [], [], [], []
for df, lab in [(post, "POST"), (pre, "PRE"), (ctrl, "control")]:
    ci = stats.binomtest(int(df.sig.sum()), len(df)).proportion_ci(0.95)
    labs.append(f"{lab}\nn={len(df)}"); fr.append(100 * df.sig.mean())
    lo.append(100 * ci.low); hi.append(100 * ci.high)
ax.bar(labs, fr, color=["#2a9d8f", "#8ecae6", "#adb5bd"])
ax.errorbar(labs, fr, yerr=[np.array(fr) - lo, np.array(hi) - np.array(fr)], fmt="none",
            ecolor="k", capsize=4)
ax.axhline(5, color="r", ls="--", lw=1, label="nominal 5%")
p_bin = stats.binomtest(int(post.sig.sum()), len(post), ctrl.sig.mean(), alternative="greater").pvalue
ax.set(ylabel="% events with significant replay", title="Significant replay events", ylim=(0, 11.5))
ax.legend(fontsize=7, loc="lower right")
ax.text(.03, .86, f"POST vs control\np = {p_bin:.1e}", transform=ax.transAxes, fontsize=7, va="top")

ax = fig.add_subplot(gs[0, 2])
sp = post[post.sig].speed
ax.hist(sp, bins=np.linspace(0, 25, 40), color="#2a9d8f")
ax.axvline(res["mean_run_speed"], color="r", lw=1.5)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.text(res["mean_run_speed"] + .6, ax.get_ylim()[1] * .8,
        f"running\n{res['mean_run_speed']:.2f} m/s", color="r", fontsize=7)
ax.set(xlabel="replay speed (m/s)", ylabel="events",
       title=f"Virtual speed of replayed trajectories\nmedian {sp.median():.1f} m/s "
             f"({sp.median()/res['mean_run_speed']:.0f}x running)")

ax = fig.add_subplot(gs[1, 0])
fwd, rev = int((post[post.sig].wcorr > 0).sum()), int((post[post.sig].wcorr < 0).sum())
ax.bar(["forward\n(r > 0)", "reverse\n(r < 0)"], [fwd, rev], color=["#e76f51", "#264653"])
ax.set(ylabel="significant events",
       title=f"Trajectory direction\nbinomial p = {stats.binomtest(fwd, fwd+rev).pvalue:.2f}")

ax = fig.add_subplot(gs[1, 1])
s = post[post.sig]
ax.scatter(s.start_pos, s.end_pos, c=np.sign(s.wcorr), cmap="coolwarm", s=14, alpha=.8)
ax.plot([0, res["track"]], [0, res["track"]], "k--", lw=.8)
ax.set(xlabel="decoded start position (m)", ylabel="decoded end position (m)",
       title="Replayed trajectory endpoints")

ax = fig.add_subplot(gs[1, 2])
ax.hist(s.span * 100, bins=30, color="#2a9d8f")
ax.set(xlabel="path length covered (cm)", ylabel="events",
       title=f"Extent of replayed trajectories\nmedian {s.span.median()*100:.0f} cm "
             f"of a {res['track']*100:.0f} cm track")

ax = fig.add_subplot(gs[2, 0])
xs, fp, fc = [], [], []
for a, b in zip([5, 10, 15, 20, 25], [10, 15, 20, 25, 60]):
    mp = (post.ncells >= a) & (post.ncells < b)
    mc = (ctrl.ncells >= a) & (ctrl.ncells < b)
    if mp.sum() > 20:
        xs.append(f"{a}-{b}"); fp.append(100 * post.sig[mp].mean()); fc.append(100 * ctrl.sig[mc].mean())
ax.plot(xs, fp, "o-", color="#2a9d8f", label="POST")
ax.plot(xs, fc, "o-", color="#adb5bd", label="control")
ax.set(xlabel="active cells in event", ylabel="% significant", title="Detection vs. event size")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 1])
ax.plot(res["rip_trig"]["t"] * 1e3, res["rip_trig"]["raw"], "k")
ax.set(xlabel="ms from ripple peak", ylabel="uV", title="Ripple-triggered LFP average")

ax = fig.add_subplot(gs[2, 2])
e = res["example_ripple"]
ax.plot(e["t"], e["raw"], "k", lw=.8, label="raw")
ax.plot(e["t"], e["filt"] - 400, color="#e63946", lw=.8, label="140-230 Hz")
ax.axvspan(e["start"], e["end"], color="#ffb703", alpha=.3)
ax.set(xlabel="time from ripple peak (s)", title="Example sharp-wave ripple", yticks=[])
ax.legend(fontsize=7)

fig.suptitle(f"{MAIN}: replay statistics", fontsize=13)
plt.savefig(f"{FIGDIR}/05_replay_statistics.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"POST {post.sig.sum()}/{len(post)} ({100*post.sig.mean():.1f}%), "
      f"PRE {pre.sig.sum()}/{len(pre)} ({100*pre.sig.mean():.1f}%), "
      f"CONTROL {ctrl.sig.sum()}/{len(ctrl)} ({100*ctrl.sig.mean():.1f}%), p={p_bin:.2e}")

# %% [markdown]
# ### Figure 6 — across sessions

# %%
rows = []
for r in R:
    for k in ["POST", "PRE", "CONTROL"]:
        df = r["events"][k]
        ci = stats.binomtest(int(df.sig.sum()), len(df)).proportion_ci(0.95)
        rows.append(dict(session=r["session"], epoch=k, n=len(df), nsig=int(df.sig.sum()),
                         frac=100 * df.sig.mean(), lo=100 * ci.low, hi=100 * ci.high,
                         med_speed=float(df[df.sig].speed.median()) if df.sig.sum() else np.nan,
                         fwd=int((df[df.sig].wcorr > 0).sum()), rev=int((df[df.sig].wcorr < 0).sum())))
S = pd.DataFrame(rows)
S.to_csv(f"{CACHEDIR}/summary_across_sessions.csv", index=False)
print(S.to_string(index=False))

pooled = {k: pd.concat([r["events"][k] for r in R], ignore_index=True) for k in ["POST", "PRE", "CONTROL"]}
p_post = stats.binomtest(int(pooled["POST"].sig.sum()), len(pooled["POST"]),
                         pooled["CONTROL"].sig.mean(), alternative="greater").pvalue
p_pre = stats.binomtest(int(pooled["PRE"].sig.sum()), len(pooled["PRE"]),
                        pooled["CONTROL"].sig.mean(), alternative="greater").pvalue
tbl = np.array([[pooled["POST"].sig.sum(), len(pooled["POST"]) - pooled["POST"].sig.sum()],
                [pooled["PRE"].sig.sum(), len(pooled["PRE"]) - pooled["PRE"].sig.sum()]])
p_pp = stats.chi2_contingency(tbl)[1]
print(f"\nPOOLED POST {100*pooled['POST'].sig.mean():.1f}% | PRE {100*pooled['PRE'].sig.mean():.1f}% "
      f"| CONTROL {100*pooled['CONTROL'].sig.mean():.1f}%")
print(f"POST vs control p={p_post:.2e} | PRE vs control p={p_pre:.3f} | POST vs PRE p={p_pp:.2e}")

names = [r["session"] for r in R]
cols = {"POST": "#2a9d8f", "PRE": "#8ecae6", "CONTROL": "#adb5bd"}
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))

ax = axs[0, 0]
for i, k in enumerate(["POST", "PRE", "CONTROL"]):
    sub = S[S.epoch == k]
    x = np.arange(len(R)) + (i - 1) * 0.26
    ax.bar(x, sub.frac, 0.26, color=cols[k], label=k.lower())
    ax.errorbar(x, sub.frac, yerr=[sub.frac - sub.lo, sub.hi - sub.frac], fmt="none",
                ecolor="k", capsize=2, lw=.8)
ax.axhline(5, color="r", ls="--", lw=1)
ax.set_xticks(range(len(R)))
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="% events with significant replay", title="Replay detection per session")
ax.legend(fontsize=8)

ax = axs[0, 1]
fr, lo, hi = [], [], []
for k in ["POST", "PRE", "CONTROL"]:
    dfk = pooled[k]
    ci = stats.binomtest(int(dfk.sig.sum()), len(dfk)).proportion_ci(0.95)
    fr.append(100 * dfk.sig.mean()); lo.append(100 * ci.low); hi.append(100 * ci.high)
labels = [f"{k}\nn={len(pooled[k])}" for k in ["POST", "PRE", "CONTROL"]]
ax.bar(labels, fr, color=[cols[k] for k in ["POST", "PRE", "CONTROL"]])
ax.errorbar(labels, fr, yerr=[np.array(fr) - lo, np.array(hi) - np.array(fr)], fmt="none",
            ecolor="k", capsize=4)
ax.axhline(5, color="r", ls="--", lw=1)
ax.set(ylabel="% significant", title="Pooled across sessions", ylim=(0, max(hi) * 1.35))
ax.text(.03, .95, f"POST vs control  p = {p_post:.1e}\nPOST vs PRE      p = {p_pp:.1e}",
        transform=ax.transAxes, fontsize=8, va="top")

ax = axs[0, 2]
for r in R:
    df = r["events"]["POST"]
    ax.hist(df[df.sig].speed, bins=np.linspace(0, 25, 30), histtype="step", lw=1.5,
            label=r["session"][:12])
ax.set(xlabel="replay speed (m/s)", ylabel="events", title="Virtual speed of replayed trajectories")
ax.legend(fontsize=7)

ax = axs[1, 0]
ax.bar(range(len(R)), [100 * r["med_err"] for r in R], color="#457b9d")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="median |error| (cm)", title="Cross-validated position decoding on running laps")

ax = axs[1, 1]
ax.bar(np.arange(len(R)) - .2, [len(r["pc_ids"]) for r in R], .4, label="place cells", color="#e76f51")
ax.bar(np.arange(len(R)) + .2, [r["n_pyr"] for r in R], .4, label="pyramidal cells", color="#264653")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="units", title="Recorded population")
ax.legend(fontsize=8)

ax = axs[1, 2]
sub = S[S.epoch == "POST"]
ax.bar(np.arange(len(R)) - .2, sub.fwd, .4, label="forward", color="#e76f51")
ax.bar(np.arange(len(R)) + .2, sub.rev, .4, label="reverse", color="#264653")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="significant events", title="Forward vs reverse replay (POST)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/06_cross_session_summary.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Result
#
# The decoder is accurate: on held-out running laps it recovers the animal's position to within a few
# centimetres on a 160 cm track, against a chance level near 50 cm. Applied to the population bursts
# that accompany sharp-wave ripples in slow-wave sleep, the same decoder produces posteriors that
# frequently sweep smoothly from one part of the track to another within 100–200 ms (Figure 4), at
# roughly an order of magnitude above the animal's own running speed, in both forward and reverse
# order. That is hippocampal replay.
#
# Quantitatively, the fraction of ripple-associated bursts whose decoded trajectory beats both shuffle
# nulls is about twice the calibrated false-positive rate in sleep that follows track running, and it
# is not elevated in the sleep that precedes it, even though ripples are just as abundant there and the
# same place-field template is used. The template-permutation control lands on the nominal 5%, which
# confirms that the excess in POST sleep comes from the correspondence between specific cells and
# specific places rather than from the coarse statistics of ripple firing. The effect is a minority of
# events, which is expected: only a fraction of SWRs carry a linear, decodable trajectory, and short
# bursts with few active cells carry little information, which is exactly the trend seen in the
# "detection vs. event size" panel.
