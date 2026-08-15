# %% [markdown]
# # Theta phase entrainment of hippocampal CA1 neurons
#
# This notebook demonstrates **theta phase entrainment** (spike phase-locking to the
# hippocampal theta rhythm) in area CA1, using publicly available data from the
# DANDI Archive.
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark &
# Buzsáki (2016), *"Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences."* Bilateral silicon-probe recordings from dorsal CA1 in rats
# running on a 1.6 m linear maze, with simultaneous local field potential (LFP, 1250 Hz),
# spike-sorted single units (labelled pyramidal / interneuron), linearized position, and
# scored brain states (Awake / Non-REM / REM).
#
# **Phenomenon.** During active locomotion (and REM sleep), the CA1 LFP is dominated by a
# 6–12 Hz *theta* oscillation. Principal cells and interneurons do not fire uniformly
# across the theta cycle; they preferentially discharge near a particular theta phase.
# We quantify this with the phase of theta (from the Hilbert transform of the
# band-passed LFP) sampled at each spike, and summarize each neuron's locking with the
# **mean resultant length (MRL)**, its **preferred phase**, and the **Rayleigh test** for
# non-uniformity.
#
# **Approach.**
# 1. Stream one session, reconstruct timing, and pick the CA1 channel with the strongest
#    theta (highest theta/delta power ratio).
# 2. Define RUN epochs from running speed on the linear maze.
# 3. Extract instantaneous theta phase and read it at every spike.
# 4. Compute per-neuron circular statistics; compare pyramidal cells vs. interneurons.
# 5. Confirm the population result on a second animal/session.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import remfile, h5py
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d
import pynapple as nap

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
CACHE = "/tmp/remfile_cache"
THETA_BAND = (6, 12)
PYR_COLOR, INT_COLOR = "#2c6fbb", "#d1495b"

# %% [markdown]
# ## Streaming and data-access helpers
#
# The LFP is chunked per-channel `(170221, 1)`, so reading a single channel across the
# whole session is cheap. One quirk of this NWB conversion: the linearized-position
# `SpatialSeries` stores its sampling *period* (~0.0256 s ≈ 39 Hz) in the `rate` field
# rather than a frequency, so we rebuild its timestamps manually.

# %%
def open_session(dandiset_id, path_suffix):
    client = DandiAPIClient()
    d = client.get_dandiset(dandiset_id)
    asset = next(a for a in d.get_assets() if a.path.endswith(path_suffix))
    rf = remfile.File(asset.download_url, disk_cache=remfile.DiskCache(CACHE))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read(), io

def get_lfp_series(nwb):
    return nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]

def read_lfp_channel(es, ch, t0=None, t1=None):
    fs, n = es.rate, es.data.shape[0]
    i0 = 0 if t0 is None else max(0, int(t0 * fs))
    i1 = n if t1 is None else min(n, int(t1 * fs))
    d = es.data[i0:i1, ch].astype(np.float32) * es.conversion
    return nap.Tsd(t=np.arange(i0, i1) / fs + es.starting_time, d=d)

def read_lfp_block(es, t0, t1):
    fs = es.rate
    i0, i1 = max(0, int(t0 * fs)), min(es.data.shape[0], int(t1 * fs))
    return es.data[i0:i1, :].astype(np.float32) * es.conversion

def bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)

def theta_delta_ratio(block, fs):
    tp = np.mean(bandpass(block, fs, *THETA_BAND) ** 2, axis=0)
    dp = np.mean(bandpass(block, fs, 1, 4) ** 2, axis=0)
    return tp / dp

def theta_phase_tsd(lfp_tsd, fs, band=THETA_BAND):
    """Hilbert phase of band-passed LFP. angle==0 at the theta PEAK, +/-pi at the TROUGH."""
    filt = bandpass(lfp_tsd.values, fs, *band)
    analytic = hilbert(filt)
    t = lfp_tsd.index.values
    return (nap.Tsd(t=t, d=np.angle(analytic)),
            nap.Tsd(t=t, d=filt),
            nap.Tsd(t=t, d=np.abs(analytic)))

def reconstruct_position(nwb):
    sp = (nwb.processing["behavior"].data_interfaces["1.6mLinearMazeLinearizedPosition"]
          .spatial_series["1.6mLinearMazeLinearizedTimeSeries"])
    t = sp.starting_time + np.arange(sp.data.shape[0]) * sp.rate  # sp.rate is the PERIOD
    d = sp.data[:].ravel()
    v = ~np.isnan(d)
    return nap.Tsd(t=t[v], d=d[v])

def run_epochs(pos, speed_thresh=0.08, max_speed=4.0, min_dur=0.5, smooth=5):
    v = np.abs(np.gradient(pos.values, pos.index.values))
    v = uniform_filter1d(v, size=smooth)
    v[v > max_speed] = 0.0                      # clip maze-end reset teleports
    speed = nap.Tsd(t=pos.index.values, d=v)
    run = speed.threshold(speed_thresh).time_support
    return run.drop_short_intervals(min_dur).merge_close_intervals(0.5), speed

def get_state_epochs(nwb, label):
    st = nwb.processing["behavior"].data_interfaces["states"].to_dataframe()
    sub = st[st["label"] == label]
    return nap.IntervalSet(start=sub["start_time"].values, end=sub["stop_time"].values)

# %% [markdown]
# ## Circular statistics
#
# For a set of spike phases we compute the mean resultant length (MRL, 0 = uniform,
# 1 = perfectly locked), the preferred (mean) phase, and the Rayleigh test of
# uniformity (Zar's approximation for the p-value).

# %%
def circ_stats(phases):
    phases = np.asarray(phases)
    n = len(phases)
    if n == 0:
        return dict(n=0, mrl=np.nan, mean_angle=np.nan, z=np.nan, p=np.nan)
    C, S = np.mean(np.cos(phases)), np.mean(np.sin(phases))
    R = np.hypot(C, S)
    z = n * R ** 2
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n)
                      - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    return dict(n=n, mrl=R, mean_angle=np.arctan2(S, C), z=z, p=float(min(max(p, 0), 1)))

# %% [markdown]
# ## Per-session analysis pipeline

# %%
def analyze_session(dandiset_id, path_suffix, min_spikes=50):
    """Full pipeline for one session. Returns a results dict."""
    nwb, io = open_session(dandiset_id, path_suffix)
    es = get_lfp_series(nwb); fs = es.rate
    units = nap.NWBFile(nwb)["units"]
    pos = reconstruct_position(nwb)
    run, speed = run_epochs(pos)

    # pick the strongest-theta channel from a 150 s running window
    ratio = theta_delta_ratio(read_lfp_block(es, pos.index.min() + 30,
                                             pos.index.min() + 180), fs)
    best_ch = int(np.argmax(ratio))

    # theta phase across the whole track window (single, cheap channel read)
    lfp = read_lfp_channel(es, best_ch, pos.index.min() - 1, pos.index.max() + 1)
    phase_tsd, filt_tsd, amp_tsd = theta_phase_tsd(lfp, fs)
    ph_run = phase_tsd.restrict(run)

    ct = units.get_info("cell_type")
    recs, spike_phases = [], {}
    for uid in units.index:
        spk = units[uid].restrict(run)
        if len(spk) < min_spikes:
            continue
        phases = spk.value_from(ph_run).values      # nearest theta phase per spike
        spike_phases[uid] = phases
        st = circ_stats(phases)
        recs.append(dict(uid=uid, cell_type=ct[uid], **st))
    io.close()
    return dict(session=path_suffix.split("_")[0] + "/" + path_suffix.split("ses-")[1][:20],
                fs=fs, best_ch=best_ch, ratio=float(ratio[best_ch]),
                run=run, speed=speed, lfp=lfp, phase=phase_tsd, filt=filt_tsd,
                amp=amp_tsd, units=units, recs=recs, spike_phases=spike_phases)

# %% [markdown]
# ## Session 1 — sub-Achilles

# %%
S1 = "sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
res1 = analyze_session("000044", S1)
recs1 = res1["recs"]
print(f"Session: {res1['session']}")
print(f"Best theta channel: {res1['best_ch']}  (theta/delta ratio = {res1['ratio']:.1f})")
print(f"RUN epochs: {len(res1['run'])} intervals, {res1['run'].tot_length():.0f} s total")
print(f"Units analyzed (>=50 RUN spikes): {len(recs1)}")

mrl = np.array([r["mrl"] for r in recs1]); pval = np.array([r["p"] for r in recs1])
cts = np.array([r["cell_type"] for r in recs1])
is_pyr, is_int = cts == "excitatory", cts == "inhibitory"
print(f"  pyramidal: n={is_pyr.sum()}, median MRL={np.median(mrl[is_pyr]):.3f}, "
      f"%locked(p<0.05)={100*(pval[is_pyr]<0.05).mean():.0f}")
print(f"  interneuron: n={is_int.sum()}, median MRL={np.median(mrl[is_int]):.3f}, "
      f"%locked(p<0.05)={100*(pval[is_int]<0.05).mean():.0f}")

# %% [markdown]
# ## Figure 1 — Raw LFP, theta band, and instantaneous phase
#
# Validation that theta extraction is correct: the 6–12 Hz band-passed signal tracks the
# dominant rhythm in the raw LFP and the Hilbert phase advances monotonically 0→360° once
# per theta cycle.

# %%
fs = res1["fs"]
t0 = res1["run"].start[0] + 5
seg = res1["lfp"].restrict(nap.IntervalSet(t0, t0 + 4))
ph, ft, am = theta_phase_tsd(seg, fs)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(seg.index.values, seg.values * 1e3, lw=0.6, color="0.55", label="raw LFP")
ax[0].plot(ft.index.values, ft.values * 1e3, lw=1.6, color="C3", label="theta (6–12 Hz)")
ax[0].set_ylabel("LFP (mV)"); ax[0].legend(loc="upper right", ncol=2, fontsize=9)
ax[0].set_title(f"CA1 LFP — {res1['session']} (channel {res1['best_ch']})")
ax[1].plot(ph.index.values, np.degrees(ph.values) % 360, lw=0.9, color="C0")
ax[1].set_ylabel("theta phase (°)"); ax[1].set_xlabel("time (s)")
ax[1].set_yticks([0, 90, 180, 270, 360])
plt.tight_layout(); plt.savefig("fig1_lfp_theta_phase.png"); plt.close()
print("saved fig1_lfp_theta_phase.png")

# %% [markdown]
# ## Figure 2 — Spike raster of an entrained interneuron over the theta cycle
#
# For one strongly locked interneuron we overlay its spikes on the theta-filtered LFP for
# a few seconds and color each spike by the theta phase at which it occurred. Spikes
# cluster at a preferred phase rather than tiling the cycle uniformly.

# %%
int_recs = sorted([r for r in recs1 if r["cell_type"] == "inhibitory"],
                  key=lambda r: -r["mrl"])
demo_uid = int_recs[0]["uid"]
w0 = res1["run"].start[0] + 20
win = nap.IntervalSet(w0, w0 + 3)
ftw = res1["filt"].restrict(win)
spk = res1["units"][demo_uid].restrict(win)
spk_ph = spk.value_from(res1["phase"].restrict(win)).values

# place each spike marker ON the theta waveform at its spike time so the phase
# clustering is visible directly on the oscillation
spk_lfp = spk.value_from(ftw).values * 1e3
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(ftw.index.values, ftw.values * 1e3, color="0.55", lw=1.3, label="theta LFP", zorder=1)
sc = ax.scatter(spk.index.values, spk_lfp, c=np.degrees(spk_ph) % 360, cmap="twilight",
                s=70, vmin=0, vmax=360, edgecolor="k", linewidth=0.4, zorder=3)
ax.set_xlabel("time (s)"); ax.set_ylabel("theta LFP (mV)")
ax.set_title(f"Interneuron unit {demo_uid}: spikes ride a fixed part of the theta cycle "
             f"(MRL={int_recs[0]['mrl']:.2f})")
cb = plt.colorbar(sc, ax=ax, pad=0.01); cb.set_label("theta phase\nat spike (°)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("fig2_entrained_raster.png"); plt.close()
print("saved fig2_entrained_raster.png")

# %% [markdown]
# ## Figure 3 — Example phase-tuning curves (polar)
#
# Spike-phase histograms for the most strongly locked pyramidal cells (top row) and
# interneurons (bottom row). The red radial line marks each cell's preferred phase; its
# length is the MRL. 0° is the peak of the local theta LFP, 180° the trough.

# %%
def polar_hist(ax, phases, color, title):
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(phases, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges)
    ax.bar(centers, counts, width=width, color=color, alpha=0.7, edgecolor="k", linewidth=0.3)
    st = circ_stats(phases)
    ax.annotate("", xy=(st["mean_angle"], st["mrl"] * counts.max() * 2.4),
                xytext=(0, 0), arrowprops=dict(color="red", width=2, headwidth=8))
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"], fontsize=8)
    ax.set_title(title, fontsize=9, pad=20)

pyr_recs = sorted([r for r in recs1 if r["cell_type"] == "excitatory" and r["n"] > 300],
                  key=lambda r: -r["mrl"])[:4]
int_top = int_recs[:4]
fig, axes = plt.subplots(2, 4, figsize=(13, 7), subplot_kw={"projection": "polar"})
for ax, r in zip(axes[0], pyr_recs):
    polar_hist(ax, res1["spike_phases"][r["uid"]], PYR_COLOR,
               f"PYR u{r['uid']}\nMRL={r['mrl']:.2f}, p={r['p']:.1e}")
for ax, r in zip(axes[1], int_top):
    polar_hist(ax, res1["spike_phases"][r["uid"]], INT_COLOR,
               f"INT u{r['uid']}\nMRL={r['mrl']:.2f}, p={r['p']:.1e}")
fig.suptitle("Theta phase tuning of individual CA1 neurons (0°=peak, 180°=trough)",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.subplots_adjust(hspace=0.5)
plt.savefig("fig3_example_polar.png", bbox_inches="tight"); plt.close()
print("saved fig3_example_polar.png")

# %% [markdown]
# ## Figure 4 — Population phase histograms (two theta cycles)
#
# Pooling all spikes from all significantly locked cells of each type, plotted over two
# theta cycles for readability, with a cosine reference (peak of theta at 0°/360°).

# %%
def pooled_phases(recs, spdict, ctype):
    return np.concatenate([spdict[r["uid"]] for r in recs
                           if r["cell_type"] == ctype and r["p"] < 0.05])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
for ax, ctype, color, name in [(axes[0], "excitatory", PYR_COLOR, "Pyramidal"),
                               (axes[1], "inhibitory", INT_COLOR, "Interneuron")]:
    ph = np.degrees(pooled_phases(recs1, res1["spike_phases"], ctype)) % 360
    bins = np.linspace(0, 360, 37)
    counts, edges = np.histogram(ph, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    x2 = np.concatenate([centers, centers + 360])
    y2 = np.concatenate([counts, counts])
    ax.bar(x2, y2, width=10, color=color, alpha=0.75, edgecolor="k", linewidth=0.3)
    xx = np.linspace(0, 720, 400)
    ax.plot(xx, counts.mean() + counts.std() * np.cos(np.radians(xx)),
            "k--", lw=1, alpha=0.6, label="theta (cos)")
    ax.set_xlim(0, 720); ax.set_xticks([0, 180, 360, 540, 720])
    ax.set_xlabel("theta phase (°)  [two cycles]"); ax.set_ylabel("spike density")
    st = circ_stats(np.radians(ph))
    ax.set_title(f"{name}: pooled, pref={np.degrees(st['mean_angle'])%360:.0f}°")
    ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig("fig4_population_phase_hist.png"); plt.close()
print("saved fig4_population_phase_hist.png")

# %% [markdown]
# ## Session 2 — sub-Cicero (population confirmation)

# %%
S2 = "sub-Cicero_ses-Cicero-09012014_behavior+ecephys.nwb"
res2 = analyze_session("000044", S2)
recs2 = res2["recs"]
print(f"Session: {res2['session']}")
print(f"Best theta channel: {res2['best_ch']} (ratio {res2['ratio']:.1f}); "
      f"RUN {res2['run'].tot_length():.0f} s; units {len(recs2)}")

# %% [markdown]
# ## Figure 5 — Population summary across both sessions
#
# Left: distribution of MRL for pyramidal cells vs interneurons (pooled across sessions).
# Middle: fraction of cells significantly locked (Rayleigh p<0.05). Right: preferred phases
# of all significantly locked cells on the theta cycle.

# %%
allrecs = recs1 + recs2
mrl = np.array([r["mrl"] for r in allrecs])
pval = np.array([r["p"] for r in allrecs])
ang = np.array([r["mean_angle"] for r in allrecs])
cts = np.array([r["cell_type"] for r in allrecs])
is_pyr, is_int = cts == "excitatory", cts == "inhibitory"

fig = plt.figure(figsize=(14, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.8, 1.0], wspace=0.35)

ax0 = fig.add_subplot(gs[0])
bins = np.linspace(0, max(mrl) * 1.02, 26)
ax0.hist(mrl[is_pyr], bins=bins, color=PYR_COLOR, alpha=0.7,
         label=f"pyramidal (n={is_pyr.sum()})", density=True)
ax0.hist(mrl[is_int], bins=bins, color=INT_COLOR, alpha=0.7,
         label=f"interneuron (n={is_int.sum()})", density=True)
ax0.axvline(np.median(mrl[is_pyr]), color=PYR_COLOR, ls="--")
ax0.axvline(np.median(mrl[is_int]), color=INT_COLOR, ls="--")
ax0.set_xlabel("mean resultant length (MRL)"); ax0.set_ylabel("density")
ax0.set_title("Strength of theta locking"); ax0.legend(fontsize=9)

ax1 = fig.add_subplot(gs[1])
frac = [100 * (pval[is_pyr] < 0.05).mean(), 100 * (pval[is_int] < 0.05).mean()]
ax1.bar(["pyramidal", "interneuron"], frac, color=[PYR_COLOR, INT_COLOR], alpha=0.8,
        edgecolor="k")
for i, f in enumerate(frac):
    ax1.text(i, f + 1, f"{f:.0f}%", ha="center", fontsize=10)
ax1.set_ylim(0, 105); ax1.set_ylabel("% significantly locked")
ax1.set_title("Rayleigh p < 0.05")

ax2 = fig.add_subplot(gs[2], projection="polar")
sig = pval < 0.05
for mask, color, name in [(is_pyr & sig, PYR_COLOR, "pyr"), (is_int & sig, INT_COLOR, "int")]:
    ax2.scatter(ang[mask], mrl[mask], c=color, s=28, alpha=0.75, label=name, edgecolor="k",
                linewidth=0.3)
ax2.set_theta_zero_location("E"); ax2.set_theta_direction(1)
ax2.set_title("Preferred phase vs MRL\n(0°=peak, 180°=trough)", fontsize=10, pad=16)
ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
fig.suptitle("CA1 theta phase entrainment — population (2 sessions, DANDI:000044)",
             fontsize=13, y=1.02)
plt.savefig("fig5_population_summary.png", bbox_inches="tight"); plt.close()
print("saved fig5_population_summary.png")

# %% [markdown]
# ## Summary of findings

# %%
print("=" * 70)
print("THETA PHASE ENTRAINMENT OF CA1 NEURONS — SUMMARY")
print("=" * 70)
for res, recs in [(res1, recs1), (res2, recs2)]:
    m = np.array([r["mrl"] for r in recs]); p = np.array([r["p"] for r in recs])
    c = np.array([r["cell_type"] for r in recs])
    print(f"\n{res['session']}  (theta ch {res['best_ch']}, {len(recs)} units)")
    for lab, key in [("pyramidal", "excitatory"), ("interneuron", "inhibitory")]:
        mk = c == key
        print(f"  {lab:11s} n={mk.sum():3d}  medianMRL={np.median(m[mk]):.3f}  "
              f"%locked={100*(p[mk]<0.05).mean():.0f}")
print("\nOverall: interneurons are more strongly and more consistently entrained to")
print("theta than pyramidal cells, and the large majority of CA1 neurons of both types")
print("are significantly phase-locked during running. This reproduces the canonical")
print("finding of theta phase entrainment in the hippocampus.")
