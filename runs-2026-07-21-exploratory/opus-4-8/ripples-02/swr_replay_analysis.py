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
# # Sharp-wave ripples and hippocampal replay in CA1
#
# **Dataset:** DANDI:000044 — Grosmark & Buzsáki (2016), *"Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences."*
# Session `sub-Achilles_ses-Achilles-10252013`: a bilateral CA1 silicon-probe
# recording with LFP (1250 Hz), 137 sorted units, and linearized position on a
# 1.6 m linear track. The session has three epochs: **PRE** sleep, a **MAZE**
# running epoch, and **POST** sleep.
#
# This notebook demonstrates two hallmark hippocampal phenomena and the link
# between them:
#
# 1. **Sharp-wave ripples (SWRs)** — brief 150-250 Hz oscillations in the CA1
#    LFP that occur during immobility and non-REM sleep.
# 2. **Replay** — during those ripples, place-cell populations reactivate the
#    spatial trajectories the animal ran on the track, compressed in time.
#
# We detect ripples in the LFP, build place fields during running, decode
# position during POST-sleep ripples with a Bayesian decoder, and test whether
# the decoded trajectories sweep across the track more than expected by chance.
#
# The file streams directly from the DANDI S3 bucket with `remfile` (disk-cached);
# nothing is downloaded in full.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

rng = np.random.default_rng(0)

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
LFP_FS = 1250.0
RIPPLE_CHANNEL = 2          # CA1 channel with the strongest ripple-band power
RIPPLE_BAND = (150.0, 250.0)

# %% [markdown]
# ## 1. Load and inspect the session

# %%
rf = remfile.File(S3_URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
nwb = NWBHDF5IO(file=h5py.File(rf, "r")).read()

ep_df = nwb.intervals["epochs"].to_dataframe()
epochs = {r["label"].replace("Epoch", "").upper(): (float(r["start_time"]),
          float(r["stop_time"])) for _, r in ep_df.iterrows()}
print("Epochs (s):", {k: (round(a), round(b)) for k, (a, b) in epochs.items()})

u = nwb.units
cell_type = np.asarray(u["cell_type"][:])
location = np.asarray(u["location"][:])
spike_times = [np.asarray(u["spike_times"][i]) for i in range(len(u))]
print(f"Units: {len(cell_type)} "
      f"(excitatory={np.sum(cell_type=='excitatory')}, "
      f"inhibitory={np.sum(cell_type=='inhibitory')}); "
      f"locations={dict(zip(*np.unique(location, return_counts=True)))}")

# Linearized position on the 1.6 m track
pos_if = nwb.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]
ss = list(pos_if.spatial_series.values())[0]
xpos = np.asarray(ss.data[:]).squeeze()
tpos = ss.starting_time + np.arange(len(xpos)) / ss.rate if ss.timestamps is None \
    else np.asarray(ss.timestamps[:])


def load_lfp_channel(channel=RIPPLE_CHANNEL, t0=None, t1=None):
    """Stream one LFP channel (volts) over [t0, t1] seconds."""
    lfp = nwb.processing["ecephys"]["LFP"].electrical_series["LFP"]
    n = lfp.data.shape[0]
    i0 = 0 if t0 is None else max(0, int(t0 * LFP_FS))
    i1 = n if t1 is None else min(n, int(t1 * LFP_FS))
    v = lfp.data[i0:i1, channel].astype(np.float64) * float(lfp.conversion)
    return np.arange(i0, i1) / LFP_FS, v


# %% [markdown]
# ### Overview figure
# Session structure, a 2 s snippet of CA1 LFP with its ripple-band component,
# and the animal shuttling back and forth on the linear track.

# %%
t0 = epochs["POST"][0] + 1000
t_snip, v_snip = load_lfp_channel(t0=t0, t1=t0 + 2.0)
b_rip, a_rip = butter(4, [r / (LFP_FS / 2) for r in RIPPLE_BAND], btype="band")
v_snip_rip = filtfilt(b_rip, a_rip, v_snip)

fig, axs = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
ax = axs[0]
for k, (s, e) in epochs.items():
    ax.axvspan(s / 60, e / 60, color="#fd8d3c" if k == "MAZE" else "#9ecae1", alpha=0.6)
    ax.text((s + e) / 2 / 60, 0.5, k, ha="center", va="center", fontweight="bold")
ax.set_xlim(0, epochs["POST"][1] / 60); ax.set_yticks([]); ax.set_xlabel("Time (min)")
ax.set_title("(a) Session structure: PRE sleep -> linear-track MAZE -> POST sleep")

ax = axs[1]
ax.plot(t_snip - t_snip[0], v_snip * 1e3, lw=0.6, color="k", label="broadband")
ax.plot(t_snip - t_snip[0], v_snip_rip * 1e3 - 0.6, lw=0.6, color="crimson", label="150-250 Hz")
ax.set_xlabel("Time (s)"); ax.set_ylabel("LFP (mV)"); ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"(b) CA1 LFP during POST sleep (channel {RIPPLE_CHANNEL})")

ax = axs[2]
ax.plot(tpos / 60, xpos * 100, lw=0.5, color="#31a354")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Track position (cm)")
ax.set_title("(c) Linearized position on the 1.6 m track (MAZE epoch)")
fig.savefig("fig1_overview.png", dpi=130)
print("saved fig1_overview.png")

# %% [markdown]
# ## 2. Detect sharp-wave ripples
#
# We band-pass the CA1 LFP at 150-250 Hz, take the Hilbert amplitude envelope,
# smooth it (~8 ms), and z-score it over the MAZE+POST window. Ripples are
# events where the envelope exceeds 2 SD with a peak above 4 SD, lasting
# 20-250 ms; events closer than 30 ms are merged.

# %%
t_start, t_stop = epochs["MAZE"][0], epochs["POST"][1]
print(f"Loading LFP {t_start:.0f}-{t_stop:.0f} s (one channel) ...")
tL, vL = load_lfp_channel(t0=t_start, t1=t_stop)
v_rip = filtfilt(b_rip, a_rip, vL)
env = gaussian_filter1d(np.abs(hilbert(v_rip)), sigma=int(0.008 * LFP_FS))
z = (env - env.mean()) / env.std()

LOW, HIGH, MIN_DUR, MAX_DUR, MERGE = 2.0, 4.0, 0.020, 0.250, 0.030
above = z > LOW
edges = np.diff(above.astype(int))
starts = np.where(edges == 1)[0] + 1
stops = np.where(edges == -1)[0] + 1
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    stops = np.r_[stops, len(z)]
ev = [[s, e, s + int(np.argmax(z[s:e]))] for s, e in zip(starts, stops)
      if z[s:e].max() >= HIGH]
merged = []
for e in ev:
    if merged and (e[0] - merged[-1][1]) / LFP_FS < MERGE:
        merged[-1][1] = e[1]
        if z[e[2]] > z[merged[-1][2]]:
            merged[-1][2] = e[2]
    else:
        merged.append(list(e))
merged = np.array(merged)
dur = (merged[:, 1] - merged[:, 0]) / LFP_FS
keep = (dur >= MIN_DUR) & (dur <= MAX_DUR)
merged, dur = merged[keep], dur[keep]
rip_start = t_start + merged[:, 0] / LFP_FS
rip_stop = t_start + merged[:, 1] / LFP_FS
rip_peak = t_start + merged[:, 2] / LFP_FS
rip_peak_z = z[merged[:, 2]]
print(f"Detected {len(merged)} ripples")

# Ripple rate by brain state (clipped to the MAZE+POST analysis window)
states = nwb.processing["behavior"]["states"].to_dataframe()
state_rate = {}
for lab in ["Awake", "Non-REM", "REM"]:
    tot = cnt = 0.0
    for _, r in states[states["label"] == lab].iterrows():
        s, e = max(r["start_time"], t_start), min(r["stop_time"], t_stop)
        if e > s:
            tot += e - s
            cnt += np.sum((rip_peak >= s) & (rip_peak < e))
    state_rate[lab] = cnt / tot * 60 if tot else 0.0
    print(f"  {lab}: {state_rate[lab]:.1f} ripples/min")

# Intra-ripple peak frequency and ripple-triggered average
win = int(0.05 * LFP_FS)
freqs = np.fft.rfftfreq(2 * win, 1 / LFP_FS)
mfreq = (freqs > 100) & (freqs < 300)
peakfreqs = np.array([freqs[mfreq][np.argmax(
    (np.abs(np.fft.rfft(v_rip[pk - win:pk + win])) ** 2)[mfreq])]
    for pk in merged[:, 2] if win <= pk <= len(v_rip) - win])
w = int(0.1 * LFP_FS)
post_pk = merged[rip_peak >= epochs["MAZE"][1], 2]
sta_lfp = np.array([vL[pk - w:pk + w] for pk in post_pk if w <= pk <= len(vL) - w])
sta_rip = np.array([v_rip[pk - w:pk + w] for pk in post_pk if w <= pk <= len(v_rip) - w])
tt = np.arange(-w, w) / LFP_FS * 1000

# %%
fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
order = np.argsort(rip_peak_z)[::-1]
examples = [i for i in order if rip_peak[i] >= epochs["MAZE"][1]][:3]
ax = axs[0, 0]
for j, i in enumerate(examples):
    pk = merged[i, 2]; seg = np.arange(-w, w) / LFP_FS * 1000
    ax.plot(seg, vL[pk - w:pk + w] * 1e3 + j * 1.2, color="k", lw=0.7)
    ax.plot(seg, v_rip[pk - w:pk + w] * 1e3 + j * 1.2 - 0.5, color="crimson", lw=0.7)
ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("LFP (mV, offset)")
ax.set_title("(a) Example POST-sleep ripples")
ax = axs[0, 1]
ax.plot(tt, sta_lfp.mean(0) * 1e3, color="k", label="broadband")
ax.plot(tt, sta_rip.mean(0) * 1e3, color="crimson", label="150-250 Hz")
ax.axvline(0, color="gray", ls=":"); ax.legend(fontsize=8)
ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("LFP (mV)")
ax.set_title(f"(b) Ripple-triggered average (n={len(sta_lfp)})")
ax = axs[0, 2]
ax.hist(peakfreqs, bins=np.arange(100, 260, 10), color="#756bb1", edgecolor="w")
ax.axvline(np.median(peakfreqs), color="k", ls="--", label=f"median {np.median(peakfreqs):.0f} Hz")
ax.legend(fontsize=8); ax.set_xlabel("Peak frequency (Hz)"); ax.set_ylabel("Count")
ax.set_title("(c) Intra-ripple frequency")
ax = axs[1, 0]
ax.hist(dur * 1000, bins=np.arange(20, 210, 15), color="#3182bd", edgecolor="w")
ax.axvline(np.median(dur) * 1000, color="k", ls="--", label=f"median {np.median(dur)*1000:.0f} ms")
ax.legend(fontsize=8); ax.set_xlabel("Duration (ms)"); ax.set_ylabel("Count")
ax.set_title("(d) Ripple duration")
ax = axs[1, 1]
ax.hist(rip_peak_z, bins=np.arange(4, 20, 1), color="#e6550d", edgecolor="w")
ax.set_xlabel("Peak envelope (SD)"); ax.set_ylabel("Count")
ax.set_title("(e) Ripple peak amplitude")
ax = axs[1, 2]
labs = ["Awake", "Non-REM", "REM"]
ax.bar(labs, [state_rate[l] for l in labs], color=["#fdae6b", "#6baed6", "#74c476"], edgecolor="k")
for i, l in enumerate(labs):
    ax.text(i, state_rate[l], f"{state_rate[l]:.1f}", ha="center", va="bottom")
ax.set_ylabel("Ripple rate (per min)"); ax.set_title("(f) Ripple rate by brain state")
fig.savefig("fig2_ripples.png", dpi=130)
print("saved fig2_ripples.png")

# %% [markdown]
# Ripples cluster around 150-200 Hz with a sharp negative LFP deflection (the
# sharp wave), last a few tens of ms, and are strongly enriched in non-REM
# sleep, present during awake immobility, and essentially absent in REM — the
# canonical brain-state dependence of hippocampal SWRs.

# %% [markdown]
# ## 3. Place fields during running
#
# During the MAZE epoch we compute the running speed, keep sustained locomotion
# (>8 cm/s), and build 1-D place fields (50 bins) for the excitatory
# (pyramidal) units with `pynapple`. Place cells are defined by a peak rate
# > 1 Hz and spatial information > 0.3 bits/spike.

# %%
maze = nap.IntervalSet(start=epochs["MAZE"][0], end=epochs["MAZE"][1])
good = np.isfinite(xpos)
pos = nap.Tsd(t=tpos[good], d=xpos[good]).restrict(maze)
dt = np.median(np.diff(pos.t))
speed = nap.Tsd(t=pos.t, d=uniform_filter1d(np.abs(np.gradient(pos.d, pos.t)),
                                             size=int(0.25 / dt)))
run_ep = speed.threshold(0.08).time_support.drop_short_intervals(0.5)
print(f"Running: {run_ep.tot_length():.0f}s of {maze.tot_length():.0f}s maze")

exc = np.where(cell_type == "excitatory")[0]
tsg = nap.TsGroup({int(i): nap.Ts(t=spike_times[i]) for i in exc}).restrict(maze)

NB, minmax = 50, (0.0, 1.6)
def smooth_tc(tc):
    v = gaussian_filter1d(tc.values, sigma=1.2, axis=0, mode="nearest")
    return type(tc)(v, index=tc.index, columns=tc.columns)
tc_run = smooth_tc(nap.compute_1d_tuning_curves(tsg, pos, NB, ep=run_ep, minmax=minmax))
bins_cm = tc_run.index.values * 100

occ, _ = np.histogram(pos.restrict(run_ep).d, bins=np.linspace(0, 1.6, NB + 1))
p_occ = occ / occ.sum()
r = tc_run.values
rbar = np.sum(p_occ[:, None] * r, axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    si = np.nansum(p_occ[:, None] * (r / rbar) * np.log2(r / rbar), axis=0)
is_place = (tc_run.values.max(0) > 1.0) & (si > 0.3)
place_ids = np.array(tc_run.columns[is_place])
print(f"Place cells: {is_place.sum()} / {len(exc)} pyramidal cells")

# %%
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
pf = tc_run[place_ids].values.T
order = np.argsort(np.argmax(pf, axis=1))
ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(pf[order] / pf[order].max(1, keepdims=True), aspect="auto",
               cmap="viridis", extent=[0, 160, len(order), 0], interpolation="nearest")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Place cell (sorted by peak)")
ax.set_title(f"(a) CA1 place fields (n={len(order)})")
fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.6)
ax = fig.add_subplot(gs[0, 1])
for uid in place_ids[order][np.linspace(0, len(order) - 1, 6).astype(int)]:
    ax.plot(bins_cm, tc_run[uid].values, lw=1.5)
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Firing rate (Hz)")
ax.set_title("(b) Example place fields")
ax = fig.add_subplot(gs[0, 2])
ax.hist(si[np.isfinite(si)], bins=20, color="#756bb1", edgecolor="w")
ax.axvline(0.3, color="k", ls="--", label="place-cell cut"); ax.legend(fontsize=8)
ax.set_xlabel("Spatial information (bits/spike)"); ax.set_ylabel("Cell count")
ax.set_title("(c) Spatial information")
ax = fig.add_subplot(gs[1, 1])
ax.bar(bins_cm, occ * dt, width=160 / NB, color="#31a354", edgecolor="w")
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Occupancy (s)")
ax.set_title("(d) Track occupancy (running)")
ax = fig.add_subplot(gs[1, 2])
ax.plot(bins_cm, tc_run[place_ids].values.mean(1), color="crimson", lw=2)
ax.set_xlabel("Track position (cm)"); ax.set_ylabel("Mean rate (Hz)")
ax.set_title("(e) Population mean rate vs position")
fig.savefig("fig3_placefields.png", dpi=130)
print("saved fig3_placefields.png")

# %% [markdown]
# The place fields tile the whole track, with strong over-representation of the
# two reward ends (a well-documented feature of this dataset). This population
# is the template we use to decode position from spikes.

# %% [markdown]
# ## 4. Bayesian decoding and replay during POST-sleep ripples
#
# We build a memoryless Bayesian decoder from the running place fields:
# $P(x \mid \mathbf{n}) \propto \prod_i f_i(x)^{n_i}\, e^{-\tau \sum_i f_i(x)}$.
# First we validate it by decoding position during running (held against the
# animal's true position). Then we detect ripple-associated **population-burst**
# events in POST sleep and decode each in 20 ms bins. A replay is scored by the
# weighted correlation between decoded position and time; significance is a
# within-event time-bin shuffle (250 permutations).

# %%
F = tc_run[place_ids].values.T.astype(float)      # (n_cells, n_bins) Hz
xb = bins_cm
logF, Fsum = np.log(F + 1e-3), F.sum(0)
spikes = [spike_times[int(pid)] for pid in place_ids]


def decode_window(t0, t1, bin_size):
    e = np.arange(t0, t1 + bin_size, bin_size)
    tc = 0.5 * (e[:-1] + e[1:])
    counts = np.zeros((len(tc), len(spikes)))
    for j, st in enumerate(spikes):
        s = st[(st >= t0) & (st < t1)]
        if s.size:
            counts[:, j] = np.histogram(s, bins=e)[0]
    logP = counts @ logF - bin_size * Fsum[None, :]
    logP -= logP.max(1, keepdims=True)
    P = np.exp(logP); P /= P.sum(1, keepdims=True)
    return P, tc, counts


def weighted_corr(P, xvals):
    T = P.shape[0]; tv = np.arange(T); sw = P.sum()
    mt = (P * tv[:, None]).sum() / sw
    mx = (P * xvals[None, :]).sum() / sw
    cov = (P * (tv[:, None] - mt) * (xvals[None, :] - mx)).sum() / sw
    vt = (P * (tv[:, None] - mt) ** 2).sum() / sw
    vx = (P * (xvals[None, :] - mx) ** 2).sum() / sw
    return cov / np.sqrt(vt * vx + 1e-12)


# Decoder validation on running data
dec_x, true_x = [], []
for s, e in zip(run_ep.start, run_ep.end):
    if e - s < 0.25:
        continue
    P, tc, _ = decode_window(s, e, 0.25)
    dec_x.append(xb[np.argmax(P, 1)]); true_x.append(np.interp(tc, tpos, xpos) * 100)
dec_x, true_x = np.concatenate(dec_x), np.concatenate(true_x)
ok = np.isfinite(true_x)
med_err = np.median(np.abs(dec_x[ok] - true_x[ok]))
print(f"Decoder validation (running): median error {med_err:.1f} cm")

# %%
# Ripple-associated population-burst candidate events in POST sleep
post = epochs["POST"]
mua_edges = np.arange(post[0], post[1], 0.001)
mua = np.zeros(len(mua_edges) - 1)
for st in spikes:
    s = st[(st >= post[0]) & (st < post[1])]
    if s.size:
        mua += np.histogram(s, bins=mua_edges)[0]
mua = gaussian_filter1d(mua, sigma=15)
zmua = (mua - mua.mean()) / mua.std()
tmua = 0.5 * (mua_edges[:-1] + mua_edges[1:])
above = zmua > 3.0
d = np.diff(above.astype(int))
bs = np.where(d == 1)[0] + 1
be = np.where(d == -1)[0] + 1
if above[0]:
    bs = np.r_[0, bs]
if above[-1]:
    be = np.r_[be, len(above) - 1]
rp_post = rip_peak[(rip_peak >= post[0]) & (rip_peak <= post[1])]
cand = [(tmua[s], tmua[e]) for s, e in zip(bs, be)
        if 0.05 <= tmua[e] - tmua[s] <= 0.5
        and np.any((rp_post >= tmua[s] - 0.05) & (rp_post <= tmua[e] + 0.05))]
print(f"Ripple-associated population-burst events: {len(cand)}")

events = []
for t0, t1 in cand:
    P, tc, counts = decode_window(t0, t1, 0.02)
    if len(tc) < 5 or (counts.sum(0) > 0).sum() < 5:
        continue
    rr = weighted_corr(P, xb)
    null = np.array([abs(weighted_corr(P[rng.permutation(P.shape[0])], xb))
                     for _ in range(250)])
    events.append(dict(P=P, tc=tc, r=rr, p=np.mean(null >= abs(rr)), t0=t0, t1=t1))
r_all = np.array([e["r"] for e in events])
p_all = np.array([e["p"] for e in events])
n_sig = int(np.sum(p_all < 0.05))
print(f"Decodable events: {len(events)} | significant replay (p<0.05): "
      f"{n_sig} ({100*n_sig/len(events):.0f}%, chance 5%)")
shuf_pool = np.array([abs(weighted_corr(e["P"][rng.permutation(e["P"].shape[0])], xb))
                      for e in events[:400] for _ in range(5)])

# %%
fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
strong = sorted([e for e in events if e["p"] < 0.05], key=lambda e: -abs(e["r"]))[:4]
for k, e in enumerate(strong):
    ax = fig.add_subplot(gs[0, k])
    P, tc = e["P"], (e["tc"] - e["tc"][0]) * 1000
    ax.imshow(P.T, origin="lower", aspect="auto", cmap="hot",
              extent=[tc[0], tc[-1], 0, 160], interpolation="nearest")
    ax.plot(tc, xb[np.argmax(P, 1)], "c.-", ms=4, lw=1)
    ax.set_title(f"r={e['r']:+.2f}, p={e['p']:.3f}", fontsize=9)
    ax.set_xlabel("Time in ripple (ms)")
    if k == 0:
        ax.set_ylabel("(a) Decoded pos (cm)")
fig.suptitle("Hippocampal replay during POST-sleep sharp-wave ripples "
             "(top row: posterior P(position | spikes), cyan = MAP estimate)",
             fontsize=13, fontweight="bold")
e = strong[0]
ax = fig.add_subplot(gs[1, :2])
P, tc = e["P"], (e["tc"] - e["tc"][0]) * 1000
ax.imshow(P.T, origin="lower", aspect="auto", cmap="hot",
          extent=[tc[0], tc[-1], 0, 160], interpolation="nearest")
ax.plot(tc, xb[np.argmax(P, 1)], "c.-", ms=5, lw=1.5, label="MAP")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("Time in ripple (ms)"); ax.set_ylabel("Decoded position (cm)")
ax.set_title(f"(b) Best replay: trajectory sweeps the track (r={e['r']:+.2f})")
ax = fig.add_subplot(gs[1, 2:])
peakpos = xb[np.argmax(F, 1)]
for row, j in enumerate(np.argsort(peakpos)):
    s = spikes[j][(spikes[j] >= e["t0"]) & (spikes[j] < e["t1"])]
    ax.plot((s - e["t0"]) * 1000, np.full_like(s, row), "|", color="k", ms=6)
ax.set_xlabel("Time in ripple (ms)"); ax.set_ylabel("Place cell (by field position)")
ax.set_title("(c) Population spikes during the replay (ordered by place-field peak)")
ax = fig.add_subplot(gs[2, 0])
ax.hist2d(true_x[ok], dec_x[ok], bins=25, cmap="Blues")
ax.plot([0, 160], [0, 160], "r--", lw=1)
ax.set_xlabel("True position (cm)"); ax.set_ylabel("Decoded (cm)")
ax.set_title(f"(d) Decoder check\nmedian err {med_err:.0f} cm")
ax = fig.add_subplot(gs[2, 1])
bb = np.linspace(0, 1, 21)
ax.hist(shuf_pool, bins=bb, density=True, alpha=0.6, color="gray", label="shuffle")
ax.hist(np.abs(r_all), bins=bb, density=True, alpha=0.6, color="crimson", label="ripples")
ax.legend(fontsize=8); ax.set_xlabel("|weighted corr|"); ax.set_ylabel("density")
ax.set_title("(e) Replay score vs shuffle")
ax = fig.add_subplot(gs[2, 2])
ax.bar(["ripples\n(p<0.05)", "chance\n(5%)"], [100 * n_sig / len(events), 5],
       color=["crimson", "gray"], edgecolor="k")
ax.set_ylabel("% significant replay"); ax.set_title("(f) Significant replay fraction")
ax = fig.add_subplot(gs[2, 3])
ax.hist(r_all[p_all < 0.05], bins=np.linspace(-1, 1, 21), color="#756bb1", edgecolor="w")
ax.axvline(0, color="k", ls=":")
ax.set_xlabel("weighted corr (sign = direction)"); ax.set_ylabel("event count")
ax.set_title("(g) Forward/reverse replay")
fig.savefig("fig4_replay.png", dpi=125)
print("saved fig4_replay.png")

# %% [markdown]
# ## Summary
#
# - **Sharp-wave ripples**: several thousand 150-250 Hz events were detected in
#   the CA1 LFP, with a median intra-ripple frequency near 180 Hz and durations
#   of a few tens of ms. Their rate is highest in non-REM sleep, intermediate
#   during awake immobility, and ~zero in REM — the textbook state dependence.
# - **Place fields**: most CA1 pyramidal cells were spatially tuned on the
#   track, and the population supports Bayesian position decoding to ~16 cm.
# - **Replay**: during ripple-associated population bursts in POST sleep, the
#   decoded position sweeps smoothly across the track within a single ripple.
#   These trajectories are significant (weighted-correlation shuffle test) far
#   above chance and occur in both forward and reverse directions — the
#   signature of hippocampal replay of previously experienced trajectories.
