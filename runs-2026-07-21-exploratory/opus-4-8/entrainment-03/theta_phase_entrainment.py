# %% [markdown]
# # Theta phase entrainment of hippocampal CA1 neurons
#
# This notebook demonstrates one of the most robust phenomena in systems
# neuroscience: during active locomotion, the hippocampal local field
# potential (LFP) is dominated by a 6-10 Hz *theta* rhythm, and the spike
# times of individual CA1 neurons are not uniformly distributed across the
# theta cycle. Instead, each cell tends to fire preferentially near a
# particular phase of theta. This coupling between single-unit spiking and
# the population rhythm is called **theta phase entrainment** (or
# phase-locking).
#
# We quantify entrainment with the **mean resultant length (MRL)** of the
# distribution of theta phases at which a cell spikes, and test whether that
# distribution is significantly non-uniform with the **Rayleigh test**.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044),
# "Petersen, Buzsaki (2020) hippocampal CA1 recordings during linear-track
# and open-field running." We use four rats (Buddy, Achilles, Cicero,
# Gatsby), each with a high-density silicon-probe LFP recording, spike-sorted
# CA1 units labeled excitatory (pyramidal) or inhibitory (interneuron), and
# tracked position on a linear maze.
#
# **Pipeline.**
# 1. Stream the LFP, units, and position for one session (remfile + NWB).
# 2. Pick the LFP channel in the CA1 pyramidal layer with the strongest theta.
# 3. Band-pass filter to 6-10 Hz and take the Hilbert phase.
# 4. Define running epochs from position speed (theta is a running rhythm).
# 5. For every unit, sample the theta phase at each spike during running and
#    compute MRL + Rayleigh p.
# 6. Visualize single cells, the population, and a spike-triggered LFP average.
# 7. Repeat across all four rats and pool the results.

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

THETA_BAND = (6.0, 10.0)   # Hz
RUN_THRESH = 0.05          # m/s locomotion threshold
MIN_RUN_DUR = 0.5          # s minimum running-bout duration
C_EXC, C_INH = "#2c7fb8", "#d95f0e"

# One NWB asset (blob URL) per rat in DANDI:000044.
SESSIONS = {
    "Buddy":    "https://dandiarchive.s3.amazonaws.com/blobs/49f/95f/49f95f2a-1ae4-4720-85a3-899847616078",
    "Achilles": "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028",
    "Cicero":   "https://dandiarchive.s3.amazonaws.com/blobs/aab/623/aab6235a-0144-4072-9a35-c7a3008b9870",
    "Gatsby":   "https://dandiarchive.s3.amazonaws.com/blobs/0a1/72f/0a172fd9-8a8d-403f-bacc-2c72488ea259",
}


# %% [markdown]
# ## Helper functions

# %%
def load_session(url):
    """Stream an NWB file from S3 with local disk caching."""
    rf = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    return NWBHDF5IO(file=h5py.File(rf, "r")).read()


def rayleigh(phases):
    """Rayleigh test for circular non-uniformity.

    Returns (R, mean_phase, p): R is the mean resultant length (0 = uniform,
    1 = perfectly concentrated), mean_phase is the preferred phase in radians,
    and p is the Rayleigh p-value (Zar's approximation).
    """
    n = len(phases)
    C = np.cos(phases).sum()
    S = np.sin(phases).sum()
    R = np.hypot(C, S) / n
    mean_phase = np.arctan2(S, C)
    Z = n * R ** 2
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n)
                      - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))
    return R, mean_phase, min(p, 1.0)


def maze_epoch(nwb):
    """Return (start, stop) of the maze/running epoch in seconds."""
    ep = nwb.epochs.to_dataframe()
    maze = ep[ep.label.str.contains("Maze", case=False)].iloc[0]
    return float(maze.start_time), float(maze.stop_time)


def best_theta_channel(lfp, maze0, maze1):
    """Pick the LFP channel with the largest theta/delta power ratio.

    The CA1 pyramidal layer has the strongest theta; we score each channel by
    the ratio of 6-10 Hz to 1-4 Hz power over a 120 s window at the middle of
    the maze epoch.
    """
    fs, conv = lfp.rate, lfp.conversion
    tmid = (maze0 + maze1) / 2
    i0, i1 = int((tmid - 60) * fs), int((tmid + 60) * fs)
    seg = lfp.data[i0:i1, :].astype(np.float32) * conv * 1e3   # mV
    f, P = welch(seg, fs=fs, axis=0, nperseg=int(fs * 4))
    ratio = P[(f >= 6) & (f <= 10)].mean(0) / P[(f >= 1) & (f <= 4)].mean(0)
    return int(np.argmax(ratio)), ratio, f


def theta_phase_tsd(lfp, channel, maze0, maze1):
    """Band-pass the chosen channel to theta and return raw/theta/phase/amp Tsd."""
    fs, conv = lfp.rate, lfp.conversion
    m0, m1 = int(maze0 * fs), int(maze1 * fs)
    raw = lfp.data[m0:m1, channel].astype(np.float32) * conv * 1e3   # mV
    t = m0 / fs + np.arange(raw.size) / fs
    b, a = butter(3, [THETA_BAND[0] / (fs / 2), THETA_BAND[1] / (fs / 2)], btype="band")
    theta = filtfilt(b, a, raw)
    analytic = hilbert(theta)
    return (nap.Tsd(t=t, d=raw),
            nap.Tsd(t=t, d=theta),
            nap.Tsd(t=t, d=np.angle(analytic)),   # 0 rad = theta peak
            nap.Tsd(t=t, d=np.abs(analytic)))


def running_epochs(nwb):
    """Derive running epochs (nap.IntervalSet) from position speed."""
    beh = nwb.processing["behavior"].data_interfaces
    posname = [k for k in beh if "Position" in k and "Linearized" not in k][0]
    ss = nwb.processing["behavior"][posname].spatial_series
    pos = ss[list(ss.keys())[0]]
    xy = pos.data[:] * pos.conversion
    pt = pos.starting_time + np.arange(xy.shape[0]) / pos.rate
    # linearly interpolate NaN gaps in each coordinate
    for k in range(xy.shape[1]):
        col = xy[:, k]
        nan = np.isnan(col)
        if nan.any():
            col[nan] = np.interp(pt[nan], pt[~nan], col[~nan])
            xy[:, k] = col
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    speed = np.concatenate([[0], d]) * pos.rate            # m/s
    w = max(1, int(0.25 * pos.rate))                       # ~0.25 s smoothing
    speed = np.convolve(speed, np.ones(w) / w, "same")
    speed_tsd = nap.Tsd(t=pt, d=speed)
    run_ep = speed_tsd.threshold(RUN_THRESH).time_support.drop_short_intervals(MIN_RUN_DUR)
    return run_ep, speed_tsd


def phase_locking(nwb, phase_tsd, run_ep):
    """Per-unit phase locking during running. Returns a DataFrame + phase dict."""
    udf = nwb.units.to_dataframe()
    rows, phase_by_unit = [], {}
    for i, r in udf.iterrows():
        s = nap.Ts(t=np.asarray(r["spike_times"])).restrict(run_ep)
        n = len(s)
        if n < 30:
            rows.append(dict(unit=i, n=n, mrl=np.nan, pref=np.nan, p=np.nan,
                             cell_type=r["cell_type"]))
            continue
        ph = s.value_from(phase_tsd).values
        R, mu, p = rayleigh(ph)
        rows.append(dict(unit=i, n=n, mrl=R, pref=mu, p=p, cell_type=r["cell_type"]))
        phase_by_unit[i] = ph
    df = pd.DataFrame(rows)
    df["significant"] = df.p < 0.05
    return df, phase_by_unit


# %% [markdown]
# ## Prototype on one session (rat "Buddy")
#
# Load the LFP, units, and position, then walk through channel selection,
# theta extraction, and running epochs so each stream can be sanity-checked
# before the statistics.

# %%
name = "Buddy"
nwb = load_session(SESSIONS[name])
lfp = nwb.processing["ecephys"]["LFP"].electrical_series["LFP"]
maze0, maze1 = maze_epoch(nwb)
print(f"{name}: maze epoch {maze0:.0f}-{maze1:.0f}s "
      f"({(maze1 - maze0) / 60:.1f} min), LFP fs={lfp.rate} Hz, "
      f"{lfp.data.shape[1]} channels")

best, ratio, freqs = best_theta_channel(lfp, maze0, maze1)
print(f"Best theta channel: {best} (theta/delta ratio = {ratio[best]:.2f})")

lfp_raw, lfp_theta, phase_tsd, amp_tsd = theta_phase_tsd(lfp, best, maze0, maze1)
run_ep, speed_tsd = running_epochs(nwb)
print(f"Running: {run_ep.tot_length():.0f}s in {len(run_ep)} bouts "
      f"({100 * run_ep.tot_length() / (maze1 - maze0):.0f}% of maze)")

udf = nwb.units.to_dataframe()
print(f"Units: {len(udf)} "
      f"({(udf.cell_type == 'excitatory').sum()} excitatory, "
      f"{(udf.cell_type == 'inhibitory').sum()} inhibitory)")

# %% [markdown]
# ### Channel selection
#
# Theta power is sharply peaked at the pyramidal layer. The left panel shows
# the theta/delta ratio across the probe; the right panel confirms that on the
# selected channel the raw LFP carries a clean theta oscillation and the
# Hilbert phase (red sawtooth) advances smoothly.

# %%
fs = lfp.rate
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot(np.arange(len(ratio)), ratio, "-o", ms=3, color=C_EXC)
ax[0].axvline(best, color="r", ls="--", label=f"best ch {best}")
ax[0].set_xlabel("LFP channel"); ax[0].set_ylabel("theta/delta power ratio")
ax[0].set_title("Theta prominence across channels"); ax[0].legend()

t0 = maze0 + (maze1 - maze0) / 2
sl = (lfp_raw.t >= t0) & (lfp_raw.t <= t0 + 3)
ax[1].plot(lfp_raw.t[sl], lfp_raw.d[sl], color="0.6", lw=0.7, label="raw LFP")
ax[1].plot(lfp_theta.t[sl], lfp_theta.d[sl], color=C_EXC, lw=1.8, label="theta 6-10 Hz")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("LFP (mV)")
axp = ax[1].twinx()
axp.plot(phase_tsd.t[sl], phase_tsd.d[sl], color="r", lw=0.6, alpha=0.6)
axp.set_ylabel("theta phase (rad)", color="r")
ax[1].set_title(f"Best channel {best}: raw, theta, phase")
ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig_channel_selection.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig_channel_selection.png")

# %% [markdown]
# ## Per-unit phase locking (running only)
#
# Restrict every spike train to running epochs, sample the theta phase at each
# spike, and compute MRL + Rayleigh p.

# %%
results, phase_by_unit = phase_locking(nwb, phase_tsd, run_ep)
results.to_csv("phase_locking_results.csv", index=False)
sig = results.dropna(subset=["mrl"]).copy()
print(f"Phase-locked (Rayleigh p<0.05): "
      f"{int(sig.significant.sum())}/{len(sig)} units "
      f"({100 * sig.significant.mean():.0f}%)")
print(sig.sort_values("mrl", ascending=False).head(6).to_string(index=False))

# %% [markdown]
# ### Raw LFP, theta, spike raster, and speed during a running bout
#
# The clearest single-trial view of entrainment. Interneurons (orange) fire on
# nearly every theta cycle; pyramidal cells (blue) fire more sparsely but still
# cluster relative to the ongoing theta wave (light blue, overlaid).

# %%
spike_times = {i: np.asarray(st) for i, st in enumerate(udf["spike_times"])}
cell_type = udf["cell_type"].values
run_starts = np.asarray(run_ep.start)
run_ends = np.asarray(run_ep.end)

win = 4.0
lb = max(zip(run_starts, run_ends), key=lambda x: x[1] - x[0])   # longest bout
t_start = lb[0] + 2.0
sl = (lfp_raw.t >= t_start) & (lfp_raw.t <= t_start + win)
tt = lfp_raw.t[sl]


def active(ct, k):
    cand = [i for i in spike_times if cell_type[i] == ct
            and np.sum((spike_times[i] >= t_start) & (spike_times[i] <= t_start + win)) >= 2]
    cand.sort(key=lambda i: -np.sum((spike_times[i] >= t_start) & (spike_times[i] <= t_start + win)))
    return cand[:k]


raster_units = sorted(active("excitatory", 10) + active("inhibitory", 6),
                      key=lambda i: cell_type[i])

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                       gridspec_kw={"height_ratios": [2, 3, 1], "hspace": 0.12})
ax[0].plot(tt, lfp_raw.d[sl], color="0.6", lw=0.7, label="raw LFP")
ax[0].plot(tt, lfp_theta.d[sl], color="C0", lw=1.8, label="theta (6-10 Hz)")
ax[0].set_ylabel("LFP (mV)"); ax[0].legend(loc="upper right", ncol=2, fontsize=9)
ax[0].set_title(f"CA1 LFP channel {best}, theta band, and simultaneous spiking during running")

for row, i in enumerate(raster_units):
    st = spike_times[i]; st = st[(st >= t_start) & (st <= t_start + win)]
    col = C_INH if cell_type[i] == "inhibitory" else C_EXC
    ax[1].vlines(st, row + 0.6, row + 1.4, color=col, lw=1.2)
th_n = (lfp_theta.d[sl] - lfp_theta.d[sl].min()) / (lfp_theta.d[sl].max() - lfp_theta.d[sl].min())
ax[1].plot(tt, th_n * len(raster_units), color="C0", lw=1.0, alpha=0.35, zorder=0)
ax[1].set_ylim(0.5, len(raster_units) + 0.5); ax[1].set_ylabel("unit (sorted by type)")
ax[1].legend([Line2D([0], [0], color=C_EXC, lw=2), Line2D([0], [0], color=C_INH, lw=2),
              Line2D([0], [0], color="C0", lw=1, alpha=0.4)],
             ["excitatory", "inhibitory", "theta (ref)"], loc="upper right", fontsize=9, ncol=3)

ps = (speed_tsd.t >= t_start) & (speed_tsd.t <= t_start + win)
ax[2].plot(speed_tsd.t[ps], speed_tsd.d[ps] * 100, color="k", lw=1.2)
ax[2].axhline(RUN_THRESH * 100, color="r", ls="--", lw=0.8, label=f"run thresh {RUN_THRESH*100:.0f} cm/s")
ax[2].set_ylabel("speed (cm/s)"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8)
plt.savefig("fig2_lfp_raster.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig2_lfp_raster.png")

# %% [markdown]
# ### Example spike-phase distributions
#
# Polar histograms of theta phase at spike time for the two most strongly
# locked interneurons and two pyramidal cells. The black radial line is the
# mean direction scaled by MRL. All four are strongly non-uniform.

# %%
avail = list(phase_by_unit.keys())
Ravail = results[results.unit.isin(avail)]
inh = Ravail[Ravail.cell_type == "inhibitory"].sort_values("mrl", ascending=False)
exc = Ravail[Ravail.cell_type == "excitatory"].sort_values("mrl", ascending=False)
exc_ex = exc[exc.n >= 500]
examples = list(inh.unit.head(2)) + list(exc_ex.unit.head(2))

nb = 30
bins = np.linspace(-np.pi, np.pi, nb + 1)
ctr = (bins[:-1] + bins[1:]) / 2
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), subplot_kw={"projection": "polar"})
for ax, u in zip(axes, examples):
    ph = phase_by_unit[u]; row = results[results.unit == u].iloc[0]
    h, _ = np.histogram(ph, bins=bins); h = h / h.sum()
    col = C_INH if row.cell_type == "inhibitory" else C_EXC
    ax.bar(ctr, h, width=2 * np.pi / nb, color=col, alpha=0.8, edgecolor="k", lw=0.3)
    ax.plot([row.pref, row.pref], [0, row.mrl * max(h) * 3.0], color="k", lw=2.5)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlabel_position(112)
    ax.set_title(f"unit {u} ({row.cell_type[:3]})   MRL={row.mrl:.2f}\n"
                 f"pref={np.degrees(row.pref):.0f}°   Rayleigh p={row.p:.1e}",
                 fontsize=10, pad=24)
    ax.set_yticklabels([])
fig.suptitle("Spike-theta-phase distributions (0° = theta peak; arrow = mean direction × MRL)",
             y=1.10, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("fig3_example_polar.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig3_example_polar.png")

# %% [markdown]
# ### Population summary for this session
#
# (a) Distribution of MRL by cell type. (b) Preferred-phase rose for the
# significantly locked units. (c) Interneurons lock more strongly than
# pyramidal cells (Mann-Whitney). (d) Fraction of each type that is
# significantly entrained.

# %%
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

axa = fig.add_subplot(gs[0, 0])
be = np.linspace(0, sig.mrl.max() * 1.05, 20)
axa.hist(sig[sig.cell_type == "excitatory"].mrl, be, color=C_EXC, alpha=0.7, label="excitatory")
axa.hist(sig[sig.cell_type == "inhibitory"].mrl, be, color=C_INH, alpha=0.7, label="inhibitory")
axa.set_xlabel("mean resultant length (phase-locking strength)"); axa.set_ylabel("# units")
axa.set_title("(a) Phase-locking strength by cell type"); axa.legend()

axb = fig.add_subplot(gs[0, 1], projection="polar")
ss = sig[sig.significant]
edges = np.linspace(-np.pi, np.pi, 13)
c = (edges[:-1] + edges[1:]) / 2
for ct, col in [("excitatory", C_EXC), ("inhibitory", C_INH)]:
    h, _ = np.histogram(ss[ss.cell_type == ct].pref.values, bins=edges)
    axb.bar(c, h, width=2 * np.pi / 12, color=col, alpha=0.6, label=ct, edgecolor="k", lw=0.3)
axb.set_theta_zero_location("E"); axb.set_title("(b) Preferred theta phase\n(0°=peak)", pad=20)
axb.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

axc = fig.add_subplot(gs[1, 0])
rng = np.random.default_rng(0)
for j, (ct, col) in enumerate([("excitatory", C_EXC), ("inhibitory", C_INH)]):
    v = sig[sig.cell_type == ct].mrl.values
    axc.scatter(rng.normal(j, 0.06, len(v)), v, color=col, alpha=0.7, s=30)
    axc.hlines(np.median(v), j - 0.2, j + 0.2, color="k", lw=2)
axc.set_xticks([0, 1]); axc.set_xticklabels(["excitatory", "inhibitory"])
axc.set_ylabel("mean resultant length"); axc.set_title("(c) MRL: interneurons lock more strongly")
u_, pmw = mannwhitneyu(sig[sig.cell_type == "excitatory"].mrl,
                       sig[sig.cell_type == "inhibitory"].mrl)
axc.text(0.5, axc.get_ylim()[1] * 0.92, f"Mann-Whitney p={pmw:.1e}", ha="center", fontsize=9)

axd = fig.add_subplot(gs[1, 1])
frac = sig.groupby("cell_type").significant.mean() * 100
ns = sig.groupby("cell_type").size()
bars = axd.bar(["excitatory", "inhibitory"],
               [frac.get("excitatory", 0), frac.get("inhibitory", 0)],
               color=[C_EXC, C_INH], alpha=0.8)
for b_, ct in zip(bars, ["excitatory", "inhibitory"]):
    n_sig = int(sig[(sig.cell_type == ct) & sig.significant].shape[0])
    axd.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 1,
             f"{n_sig}/{ns[ct]}", ha="center", fontsize=10)
axd.set_ylabel("% significantly phase-locked (p<0.05)"); axd.set_ylim(0, 108)
axd.set_title("(d) Fraction of units entrained to theta")
fig.suptitle(f"Theta phase entrainment across the CA1 population "
             f"(n={len(sig)} units, session {name})", y=0.98, fontsize=13)
plt.savefig("fig4_population.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig4_population.png")

# %% [markdown]
# ### Spike-triggered LFP average
#
# An independent confirmation that does not use the Hilbert phase at all.
# Averaging the raw LFP around the spikes of the most strongly locked cell
# reveals a clean ~8 Hz oscillation centered on the spike, which is exactly
# what phase entrainment predicts. The right panel plots preferred phase
# against locking strength for all significant units.

# %%
u = int(inh.unit.iloc[0])
st = spike_times[u]
in_run = np.zeros(len(st), bool)
for s0, s1 in zip(run_starts, run_ends):
    in_run |= (st >= s0) & (st <= s1)
st = st[in_run]
half = int(0.25 * fs)                       # +/- 250 ms
idx = np.searchsorted(lfp_raw.t, st)
idx = idx[(idx > half) & (idx < len(lfp_raw.d) - half)]
rng = np.random.default_rng(0)
if len(idx) > 4000:
    idx = rng.choice(idx, 4000, replace=False)
snips = np.stack([lfp_raw.d[j - half:j + half] for j in idx])
lag = np.arange(-half, half) / fs * 1000
sta = snips.mean(0); sem = snips.std(0) / np.sqrt(len(snips))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(lag, sta, color=C_INH, lw=2)
ax[0].fill_between(lag, sta - sem, sta + sem, color=C_INH, alpha=0.3)
ax[0].axvline(0, color="k", ls="--", lw=0.8)
ax[0].set_xlabel("time from spike (ms)"); ax[0].set_ylabel("mean LFP (mV)")
ax[0].set_title(f"(a) Spike-triggered LFP average, unit {u} (interneuron)\n"
                f"{len(idx)} spikes — rhythmic ~8 Hz theta around spikes")
for ct, col in [("excitatory", C_EXC), ("inhibitory", C_INH)]:
    s = sig[(sig.cell_type == ct) & sig.significant]
    ax[1].scatter(np.degrees(s.pref), s.mrl, color=col, alpha=0.75, s=40, label=ct)
ax[1].set_xlabel("preferred theta phase (deg, 0=peak)"); ax[1].set_ylabel("MRL")
ax[1].set_xlim(-180, 180); ax[1].set_xticks([-180, -90, 0, 90, 180])
ax[1].set_title("(b) Preferred phase vs locking strength (sig. units)"); ax[1].legend()
plt.tight_layout()
plt.savefig("fig5_sta_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig5_sta_summary.png")

# %% [markdown]
# ## Scale across all four rats
#
# Run the identical pipeline on every session and pool the per-unit results.
# This confirms the effect is not idiosyncratic to one animal.

# %%
def analyze(name, url):
    nwb = load_session(url)
    lfp = nwb.processing["ecephys"]["LFP"].electrical_series["LFP"]
    maze0, maze1 = maze_epoch(nwb)
    best, ratio, _ = best_theta_channel(lfp, maze0, maze1)
    _, _, phase_tsd, _ = theta_phase_tsd(lfp, best, maze0, maze1)
    run_ep, _ = running_epochs(nwb)
    df, _ = phase_locking(nwb, phase_tsd, run_ep)
    df = df.dropna(subset=["mrl"]).copy()
    df.insert(0, "session", name)
    return df, best, float(ratio[best])


all_df = []
for nm, url in tqdm(SESSIONS.items(), desc="sessions"):
    df, ch, rt = analyze(nm, url)
    exc = df[df.cell_type == "excitatory"]; inh_ = df[df.cell_type == "inhibitory"]
    print(f"{nm}: ch{ch} (th/dl={rt:.1f}) | {len(df)} units | "
          f"sig {100 * df.significant.mean():.0f}% | "
          f"MRL exc={exc.mrl.median():.3f} inh={inh_.mrl.median():.3f}")
    all_df.append(df)
A = pd.concat(all_df, ignore_index=True)
A.to_csv("phase_locking_all_sessions.csv", index=False)

# %% [markdown]
# ### Cross-session figure
#
# (a) Median MRL and (b) fraction entrained per rat, split by cell type; in
# every animal interneurons lock more strongly and essentially all interneurons
# are significant. (c) Pooled preferred-phase rose across all significant units.

# %%
sess = list(SESSIONS.keys())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(sess)); wdt = 0.38
for j, (ct, col) in enumerate([("excitatory", C_EXC), ("inhibitory", C_INH)]):
    med = [A[(A.session == s) & (A.cell_type == ct)].mrl.median() for s in sess]
    ax[0].bar(x + (j - 0.5) * wdt, med, wdt, color=col, label=ct, alpha=0.85)
ax[0].set_xticks(x); ax[0].set_xticklabels(sess); ax[0].set_ylabel("median MRL")
ax[0].set_title("(a) Phase-locking strength by session"); ax[0].legend()

for j, (ct, col) in enumerate([("excitatory", C_EXC), ("inhibitory", C_INH)]):
    frac = [100 * A[(A.session == s) & (A.cell_type == ct)].significant.mean() for s in sess]
    ax[1].bar(x + (j - 0.5) * wdt, frac, wdt, color=col, label=ct, alpha=0.85)
ax[1].set_xticks(x); ax[1].set_xticklabels(sess); ax[1].set_ylabel("% phase-locked (p<0.05)")
ax[1].set_ylim(0, 108); ax[1].set_title("(b) Fraction entrained by session"); ax[1].legend()

axr = fig.add_subplot(1, 3, 3, projection="polar"); ax[2].remove()
edges = np.linspace(-np.pi, np.pi, 13); c = (edges[:-1] + edges[1:]) / 2
for ct, col in [("excitatory", C_EXC), ("inhibitory", C_INH)]:
    h, _ = np.histogram(A[(A.cell_type == ct) & A.significant].pref.values, bins=edges)
    axr.bar(c, h, width=2 * np.pi / 12, color=col, alpha=0.6, label=ct, edgecolor="k", lw=0.3)
axr.set_theta_zero_location("E"); axr.set_title("(c) Pooled preferred phase\n(0°=peak)", pad=18)
axr.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9)

ne, ni = (A.cell_type == "excitatory").sum(), (A.cell_type == "inhibitory").sum()
u_, pmw = mannwhitneyu(A[A.cell_type == "excitatory"].mrl, A[A.cell_type == "inhibitory"].mrl)
fig.suptitle(f"Theta phase entrainment across 4 rats / {len(A)} CA1 units "
             f"({ne} exc, {ni} inh; MRL exc<inh, Mann-Whitney p={pmw:.1e})",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("fig6_cross_session.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved fig6_cross_session.png")

# %% [markdown]
# ## Summary
#
# Across four rats and several hundred CA1 units, theta phase entrainment is
# nearly every unit fires at a significantly non-uniform theta phase during
# running. Inhibitory interneurons entrain more strongly than excitatory
# pyramidal cells (higher MRL, Mann-Whitney p shown above), consistent with
# the classic picture in which interneuron networks pace the theta rhythm
# while pyramidal cells participate more sparsely. The spike-triggered LFP
# average provides an independent, phase-free confirmation: averaging the raw
# LFP around a cell's spikes recovers a clean ~8 Hz oscillation.

# %%
print(f"POOLED: {len(A)} units across {len(sess)} rats")
print(f"  significant: {100 * A.significant.mean():.0f}%")
print(f"  median MRL exc={A[A.cell_type == 'excitatory'].mrl.median():.3f} "
      f"inh={A[A.cell_type == 'inhibitory'].mrl.median():.3f}  MW p={pmw:.2e}")
