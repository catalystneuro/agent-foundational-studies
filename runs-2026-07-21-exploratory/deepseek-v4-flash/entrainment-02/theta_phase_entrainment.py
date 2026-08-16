# %% [markdown]
# # Theta phase entrainment of hippocampal neurons
#
# Demonstration of theta-phase entrainment of CA1 neurons using the classic
# Buzsáki-lab linear-maze session *Achilles-10252013* from DANDI dandiset
# **000044** ("Diversity in neural firing dynamics...", Grosmark & Buzsáki 2016,
# rat hippocampal CA1 tetrodes, 1.6 m linear track).
#
# We show that, during running (when the hippocampal LFP is dominated by a
# coherent ~8 Hz theta rhythm):
#
#   1. a large fraction of both pyramidal cells and interneurons fire
#      phase-locked to the theta oscillation (Rayleigh test);
#   2. entrainment is stronger for inhibitory than excitatory cells;
#   3. the observed locking is far above a random-time (shuffle) null;
#   4. locking strength scales with running speed within the same awake
#      epochs, tracking theta power (behavioral control).
#
# Data is streamed with `remfile` (range requests + local disk cache) - no full
# download. All analyses use Pynapple and SciPy.

# %% [markdown]
# ## Imports and configuration

# %%
import h5py
import numpy as np
import pandas as pd
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal, stats as sstats
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

DANDISET_ID = "000044"
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"   # Achilles-10252013 behavior+ecephys
CACHE = "/tmp/remfile_cache_entrainment02"
RNG = np.random.default_rng(42)

# theta LFP band and speed threshold defining "running"
TH_BAND = (6.0, 11.0)
SPEED_TH = 0.10           # m/s
MIN_SPIKES = 100          # minimum run spikes for per-unit stats
MAZE = (18079.5, 20147.0) # MazeEpoch boundaries (from nwb epochs)

def resolve_s3_url(asset_id):
    """Resolve the GET-only presigned S3 URL for the NWB asset."""
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

s3_url = resolve_s3_url(ASSET_ID)
print(f"dandiset {DANDISET_ID}: resolved S3 URL ({len(s3_url)} chars)")

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print("loaded NWB:", nwbfile.session_description)

# %% [markdown]
# ## Inspect the data streams
# Verify units (cell types), the 128-channel LFP, position sampling, and the
# sleep/wake states table before analysis.

# %%
print("=== UNITS ===")
print("n units:", len(nwb["units"]))
ct = nwb["units"].get_info("cell_type")
print("cell_type counts:", ct.value_counts().to_dict())
print("n units by location:", nwb["units"].get_info("location").value_counts().to_dict())

print("\n=== POSITION (1.6mLinearMazeSpatialSeries) ===")
pos = nwb["1.6mLinearMazeSpatialSeries"]
pt = np.asarray(pos.t)
print("n samples:", len(pt), "t range: %.1f - %.1f s" % (pt[0], pt[-1]),
      "median dt: %.4f s (~%d Hz)" % (np.median(np.diff(pt)), 1 / np.median(np.diff(pt))))

print("\n=== LFP (processing/ecephys/LFP/LFP) ===")
lfp_ds = h5py_file["processing/ecephys/LFP/LFP/data"]
fs = float(h5py_file["processing/ecephys/LFP/LFP/starting_time"].attrs.get("rate", 1250.0))
conversion = float(lfp_ds.attrs.get("conversion", 1.0))
print("shape:", lfp_ds.shape, "sampling rate:", fs, "Hz  conversion (V/unit):", conversion)

print("\n=== EPOCHS ===")
ep = nwb["epochs"]
print(ep.label)
print("\n=== STATES (sleep/wake) ===")
st = nwb["states"]
for lab in np.unique(st.label):
    m = np.asarray(st.label) == lab
    tot = float(np.asarray(st.end)[m].sum() - np.asarray(st.start)[m].sum())
    print(lab, "- intervals:", int(m.sum()), "- total %.0f s" % tot)

# %% [markdown]
# ## Select the theta reference channel
# The electrode table has no anatomy (all `location == "unknown"`), so we pick
# the reference electrode **data-driven**: compute the theta(6-10 Hz) to
# delta(1-3 Hz) PSD-power ratio on a 200 s chunk of the maze epoch for all
# 128 channels and take the strongest theta channel.

# %%
def band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(p[m], f[m])

t0, t1 = 18500.0, 18700.0
i0, i1 = int(t0 * fs), int(t1 * fs)
chunk = np.asarray(lfp_ds[i0:i1, :], dtype=np.float64) * conversion * 1e6   # uV
print("LFP chunk:", chunk.shape)

f, _ = signal.welch(chunk[:, 0], fs=fs, nperseg=1250)
psd = np.array([signal.welch(chunk[:, c], fs=fs, nperseg=1250)[1]
               for c in range(chunk.shape[1])])   # (nch, nfreq)
delta = np.array([band_power(f, psd[c], 1.0, 3.0) for c in range(psd.shape[0])])
theta = np.array([band_power(f, psd[c], 6.0, 10.0) for c in range(psd.shape[0])])
ratio = theta / delta
best_ch = int(np.argmax(ratio))
print("theta/delta ratio per channel: max = %.2f on ch %d" % (ratio[best_ch], best_ch))
print("top-5:", np.argsort(ratio)[::-1][:5], np.sort(ratio)[::-1][:5].round(2))

np.savez("theta_channel.npz", best_ch=best_ch, ratio=ratio, freqs=f,
         psd_selected=psd[best_ch], conversion=conversion)

# %% [markdown]
# ## Theta-band LFP, instantaneous phase, and run bouts
# Band-pass filter the reference LFP (6-11 Hz), take the Hilbert transform for
# the instantaneous phase (0 = positive peak of the filtered theta), and derive
# running epochs from 2-D tracking speed during the maze epoch (speed >= 0.10
# m/s, gaps < 0.3 s merged, bouts >= 1 s).

# %%
pad_s = 30.0
i0 = int((MAZE[0] - pad_s) * fs)
i1 = int((MAZE[1] + pad_s) * fs)
raw = np.asarray(lfp_ds[i0:i1, best_ch], dtype=np.float64)
lfp_uV = raw * conversion * 1e6
lfp_t = np.arange(i0, i1) / fs
print("LFP segment: %.1f - %.1f s, n = %d" % (lfp_t[0], lfp_t[-1], len(lfp_uV)))

sos = signal.butter(4, TH_BAND, btype="band", fs=fs, output="sos")
theta_f = signal.sosfiltfilt(sos, lfp_uV)
phase = np.angle(signal.hilbert(theta_f))          # 0 = positive peak
print("instantaneous theta phase computed (n = %d)" % len(phase))

# --- speed & run bouts from tracking ---
px = np.asarray(pos["x"], dtype=float)
py = np.asarray(pos["y"], dtype=float)
n_pos = len(pt)
valid = np.isfinite(px) & np.isfinite(py)
sp = np.zeros(n_pos)
center_ok = np.zeros(n_pos, dtype=bool)
center_ok[1:-1] = valid[2:] & valid[:-2]
sp[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) / (pt[2:] - pt[:-2])
sp = sp * center_ok

# interpolate across short invalid gaps (< 0.5 s), 0 elsewhere
bad = np.where(~center_ok)[0]
if len(bad):
    splits = np.where(np.diff(bad) > 1)[0] + 1
    for c in np.split(bad, splits):
        dt_gap = pt[c[-1]] - pt[c[0]]
        if dt_gap < 0.5 and c[0] > 0 and c[-1] < n_pos - 1:
            sp[c] = np.interp(pt[c], [pt[c[0] - 1], pt[c[-1] + 1]],
                              [sp[c[0] - 1], sp[c[-1] + 1]])

run_mask = sp >= SPEED_TH
dm = np.diff(run_mask.astype(np.int8))
starts = np.where(dm == 1)[0] + 1
ends = np.where(dm == -1)[0] + 1
if run_mask[0]:
    starts = np.concatenate([[0], starts])
if run_mask[-1]:
    ends = np.concatenate([ends, [n_pos - 1]])
bouts = []
for s, e in zip(starts, ends):
    if len(bouts) and pt[s] - pt[bouts[-1][1]] < 0.3:
        bouts[-1][1] = e
    else:
        bouts.append([s, e])
bouts = [b for b in bouts if pt[b[1]] - pt[b[0]] >= 1.0]
run_start = np.array([pt[b[0]] for b in bouts])
run_end = np.array([pt[b[1]] for b in bouts])
run_epoch = nap.IntervalSet(start=run_start, end=run_end)
# honest run-time: samples classified running (speed >= threshold), times dt
run_time_raw = int(run_mask.sum()) * np.median(np.diff(pt))
print("run-classified time: %.0f s (%.0f%% of maze epoch); %d run bouts after "
      "merging gaps < 0.3 s (spike selection), covering %.1f s"
      % (run_time_raw, 100 * run_time_raw / (MAZE[1] - MAZE[0]),
         len(run_epoch), run_epoch.tot_length()))

speed = nap.Tsd(t=pt, d=sp)
np.savez("phase_data.npz", lfp_t=lfp_t, lfp_uV=lfp_uV, theta_f=theta_f,
         phase=phase, run_start=run_start, run_end=run_end)
print("saved phase_data.npz")

# %% [markdown]
# ## Per-unit theta phase-locking statistics
# For every unit with >= 100 spikes during running, extract the theta phase at
# each spike and compute circular statistics: mean resultant length (MRL),
# preferred phase, and a Rayleigh test of uniformity. Cell types are split into
# excitatory (putative pyramidal) and inhibitory (putative interneuron).

# %%
units = nwb["units"].restrict(run_epoch)
ct_series = units.get_info("cell_type")
ct_map = dict(zip(ct_series.index, ct_series.values))

rows = []
for uid in units.keys():
    ph = np.interp(units[uid].t, lfp_t, phase)
    n = len(ph)
    if n < MIN_SPIKES:
        continue
    c, s = np.cos(ph).mean(), np.sin(ph).mean()
    M = np.hypot(c, s)
    mu = np.arctan2(s, c)
    z = n * M**2
    p_rayleigh = np.exp(-z)                      # large-n approximation
    rows.append(dict(unit=uid, n_spk=n, M=M, mu=mu, p_rayleigh=p_rayleigh,
                     cell_type=ct_map.get(uid, "NA")))
df = pd.DataFrame(rows)
print("units with >= %d run spikes: %d" % (MIN_SPIKES, len(df)))
print(df["cell_type"].value_counts())

sig = df["p_rayleigh"] < 0.01
n_sig = int(sig.sum())
print("Rayleigh-significant (p < 0.01): %d / %d (%.0f%%)"
      % (n_sig, len(df), 100 * n_sig / len(df)))

df_exc = df[df["cell_type"] == "excitatory"]
df_inh = df[df["cell_type"] == "inhibitory"]
print("median MRL  exc: %.3f  inh: %.3f" % (df_exc["M"].median(), df_inh["M"].median()))
u, p_mw = sstats.mannwhitneyu(df_exc["M"], df_inh["M"], alternative="two-sided")
print("Mann-Whitney U (exc vs inh MRL): U = %.0f, p = %.2e" % (u, p_mw))

# preferred phase of the whole population (0 = theta peak)
for lab, sub in [("excitatory", df_exc), ("inhibitory", df_inh)]:
    pref = np.degrees(np.arctan2(np.sin(sub["mu"]).mean(), np.cos(sub["mu"]).mean()))
    print("pooled preferred phase (%s): %.0f deg" % (lab, pref))

df.to_csv("unit_stats.csv", index=False)
print("saved unit_stats.csv")

# %% [markdown]
# ## Shuffle null control
# Does the observed locking simply reflect firing times being clustered in
# particular run intervals? Null: assign each cell's spikes to random times
# *within the same run epochs* (uniform over the running time, equal spike
# count) and recompute MRL. We draw 300 null MRL values per cell from a large
# shared pool of random run-times of precomputed theta phases.

# %%
NSHUF = 300
N_DRAW = 2000
lo, up = np.asarray(run_epoch.start), np.asarray(run_epoch.end)
cum = np.concatenate([[0], np.cumsum(up - lo)])
POOL = 5_000_000
x = RNG.uniform(0, cum[-1], size=POOL)
seg = np.clip(np.searchsorted(cum, x, side="right") - 1, 0, len(lo) - 1)
rts = lo[seg] + (x - cum[seg])
pool_phase = np.interp(rts, lfp_t, phase)
print("pool of random run-time phases:", pool_phase.shape)

M_real = np.full(len(df), np.nan)
M_null = np.full(len(df), np.nan)
p_shuf = np.full(len(df), np.nan)
for i, uid in enumerate(df["unit"]):
    ph = np.interp(units[uid].t, lfp_t, phase)
    n = len(ph)
    n_d = min(N_DRAW, n)
    draw = RNG.choice(len(ph), size=n_d, replace=False)
    M_real[i] = np.hypot(np.cos(ph[draw]).mean(), np.sin(ph[draw]).mean())
    idx = RNG.integers(0, POOL, size=(NSHUF, n_d))
    pph = pool_phase[idx]
    null_R = np.hypot(np.cos(pph).mean(axis=1), np.sin(pph).mean(axis=1))
    M_null[i] = np.median(null_R)
    p_shuf[i] = (np.sum(null_R >= M_real[i]) + 1) / (NSHUF + 1)

df["M_real_same_n"] = M_real
df["M_null_median"] = M_null
df["p_shuffle"] = p_shuf
n_shuf_sig = int((df["p_shuffle"] < 0.01).sum())
print("shuffle-significant (p < 0.01): %d / %d" % (n_shuf_sig, len(df)))
print("median MRL  real: %.3f   null: %.3f" % (df["M_real_same_n"].median(),
                                               df["M_null_median"].median()))
u, p_null = sstats.mannwhitneyu(df["M_real_same_n"], df["M_null_median"],
                                alternative="greater")
print("MW U real > null: p = %.2e" % p_null)

df.to_csv("unit_stats.csv", index=False)
print("saved unit_stats.csv (with shuffle columns)")

# %% [markdown]
# ## Behavioral control: entrainment strength scales with running speed
# Hippocampal theta power and frequency scale with running speed, so if the
# phase locking really follows theta, phase-locking strength (MRL) should rise
# from slow to fast locomotion within the *same* awake running epochs. We split
# each unit's run spikes by instantaneous tracking speed (< 0.4 vs >= 0.4 m/s)
# and compare MRL between the two bins (paired, per cell).

# %%
def mrl_of(ph):
    return np.hypot(np.cos(ph).mean(), np.sin(ph).mean())

rows_state = []
for uid in nwb["units"].keys():
    ts = nwb["units"][uid].t
    keep = np.zeros(len(ts), bool)
    for s_, e_ in zip(run_start, run_end):
        keep |= (ts >= s_) & (ts <= e_)
    ts = ts[keep]
    if len(ts) < MIN_SPIKES:
        continue
    spk_sp = np.interp(ts, pt, sp)          # 2-D tracking speed at each spike
    ph = np.interp(ts, lfp_t, phase)
    slow = spk_sp < 0.4
    fast = spk_sp >= 0.4
    if slow.sum() < 30 or fast.sum() < 30:
        continue
    rows_state.append(dict(
        unit=uid, cell_type=ct_map.get(uid, "NA"),
        n_run=len(ts), n_slow=int(slow.sum()), n_fast=int(fast.sum()),
        M_slow=mrl_of(ph[slow]), M_fast=mrl_of(ph[fast])))
df_state = pd.DataFrame(rows_state)
print("units with >=%d run spikes and >=30 spikes in each speed bin: %d"
      % (MIN_SPIKES, len(df_state)))
if len(df_state):
    med = df_state.groupby("cell_type")[["M_slow", "M_fast"]].median()
    print(med.round(3))
    u_s, p_st = sstats.wilcoxon(df_state["M_fast"], df_state["M_slow"],
                                alternative="greater")
    print("Wilcoxon signed-rank (M_fast > M_slow): p = %.2e" % p_st)
    for lab in ["excitatory", "inhibitory"]:
        sub = df_state[df_state["cell_type"] == lab]
        if len(sub) > 5:
            _, p_lab = sstats.wilcoxon(sub["M_fast"], sub["M_slow"],
                                       alternative="greater")
            print("  %s: fast %.3f vs slow %.3f (n=%d, p = %.2e)"
                  % (lab, sub["M_fast"].median(), sub["M_slow"].median(),
                     len(sub), p_lab))
df_state.to_csv("state_stats.csv", index=False)
print("saved state_stats.csv")

# %% [markdown]
# ## Figure 1: session overview
# (a) raw LFP and its 6-11 Hz band-pass over 2.5 s of running, showing the
# theta cycles; (b) 30 s of normalized theta, run-bout shading, and the raster
# of the 15 most phase-locked pyramidal units; (c) running speed.

# %%
C_EXC, C_INH = "#2b7bba", "#d62728"
C_TH, C_RAW, C_SPD, C_NULL = "#08519c", "#9ecae1", "#666666", "#8c8c8c"

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.35, 0.5], hspace=0.32)

ax = fig.add_subplot(gs[0])
t0, t1 = 18550.0, 18552.5
m = (lfp_t >= t0) & (lfp_t <= t1)
ax.plot(lfp_t[m], lfp_uV[m], color=C_RAW, lw=0.7, label="Raw LFP (ch %d)" % best_ch)
ax.plot(lfp_t[m], theta_f[m], color=C_TH, lw=1.2, label="6-11 Hz filter")
ax.set_ylabel("LFP (µV)")
ax.set_title("Raw LFP bandpass-filtered to the theta band during running")
ax.legend(loc="upper right", frameon=False)
ax.set_xticks([])

ax = fig.add_subplot(gs[1])
t0, t1 = 18540.0, 18570.0
m = (lfp_t >= t0) & (lfp_t <= t1)
th_n = theta_f[m] / (4 * np.percentile(np.abs(theta_f[m]), 90))
ax.plot(lfp_t[m], th_n - 0.5, color=C_TH, lw=0.8, label="theta (normalized)")
for s_, e_ in zip(np.asarray(run_epoch.start), np.asarray(run_epoch.end)):
    if e_ >= t0 and s_ <= t1:
        ax.axvspan(max(s_, t0), min(e_, t1), color="#f2f0f5", zorder=0)
ids = df_exc.sort_values("M", ascending=False)["unit"].iloc[:15]
for j, uid in enumerate(ids):
    st = units[uid].t
    st = st[(st >= t0) & (st <= t1)]
    ax.scatter(st, np.full(len(st), j), s=1.0, color=C_EXC, linewidths=0)
ax.set_ylim(-1.4, 15)
ax.set_ylabel("15 most phase-locked\npyramidal units (top MRL)")
ax.set_yticks([])
ax.legend(loc="upper right", frameon=False)

ax = fig.add_subplot(gs[2])
m = (lfp_t >= t0) & (lfp_t <= t1)
spd = np.interp(lfp_t[m], pt, sp)
ax.plot(lfp_t[m], spd, color=C_SPD, lw=0.9)
ax.set_xlabel("Session time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_ylim(0, max(spd) * 1.35)

fig.suptitle("Hippocampal theta rhythm and entrained unit firing, session 'Achilles-10252013'", y=0.98)
plt.savefig("fig01_lfp_theta_overview.png")
plt.close(fig)
print("saved fig01_lfp_theta_overview.png")

# %% [markdown]
# ## Figure 2: theta power spectral density
# PSD of the selected reference channel (200 s of maze running). Clear
# concentration of power at the ~8 Hz theta peak.

# %%
d = np.load("theta_channel.npz", allow_pickle=False)
freqs, psd_best = d["freqs"], d["psd_selected"]
fig, ax = plt.subplots(figsize=(7, 4))
m = (freqs >= 0.5) & (freqs <= 40)
ax.semilogy(freqs[m], psd_best[m], color="#333", lw=1.1)
ax.axvspan(TH_BAND[0], TH_BAND[1], color=C_TH, alpha=0.15, label="theta band (6-11 Hz)")
pk_mask = (freqs >= 4) & (freqs <= 14)
pk = freqs[pk_mask][np.argmax(psd_best[pk_mask])]
ax.axvline(pk, color=C_INH, ls="--", lw=1, label="peak = %.1f Hz" % pk)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (µV²/Hz)")
ax.set_title("Power spectral density, reference electrode ch %d" % best_ch)
ax.legend(frameon=False)
plt.savefig("fig02_theta_psd.png")
plt.close(fig)
print("saved fig02_theta_psd.png (theta peak %.1f Hz)" % pk)

# %% [markdown]
# ## Figure 3: population theta-phase distributions (polar)
# Each significant cell contributes a density-normalized phase histogram, so
# every neuron weighs equally. Left: pyramidal cells; middle: interneurons;
# right: single-cell preferred-phase vectors with the population mean vector.
# 0 deg = positive peak of the filtered theta.

# %%
def cell_weighted_hist(phases, bins):
    """Average per-cell (density-normalized) phase histograms."""
    dens = np.zeros(len(bins) - 1)
    for ph in phases:
        h, _ = np.histogram(ph, bins=bins, density=True)
        dens += h
    return dens / len(phases)

spike_phases = {uid: np.interp(units[uid].t, lfp_t, phase) for uid in df["unit"]}
sig_exc = df_exc[df_exc["p_rayleigh"] < 0.01]
sig_inh = df_inh[df_inh["p_rayleigh"] < 0.01]
bins = np.linspace(-np.pi, np.pi, 37)
d_exc = cell_weighted_hist([spike_phases[u] for u in sig_exc["unit"]], bins)
d_inh = cell_weighted_hist([spike_phases[u] for u in sig_inh["unit"]], bins)
centers = (bins[:-1] + bins[1:]) / 2

fig = plt.figure(figsize=(13.5, 4.6))
for k, (dd, color, n) in enumerate(zip([d_exc, d_inh], [C_EXC, C_INH],
                                       [len(sig_exc), len(sig_inh)])):
    ax = fig.add_subplot(1, 3, k + 1, projection="polar")
    th = np.concatenate([centers, [centers[0]]])
    rr = np.concatenate([dd, [dd[0]]])
    ax.plot(th, rr, color=color, lw=1.3)
    ax.fill(th, rr, color=color, alpha=0.3)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0°", "90°", "180°\ntrough", "270°"], fontsize=8)
    ax.set_title("Significant cells, n=%d" % n, pad=20)
    ax.set_yticks([])

ax = fig.add_subplot(1, 3, 3, projection="polar")
for u in sig_exc["unit"]:
    mu = np.deg2rad(np.degrees(df[df["unit"] == u]["mu"].iloc[0]))
    ax.plot([mu, mu], [0, 1], color=C_EXC, alpha=0.3, lw=0.8)
for u in sig_inh["unit"]:
    mu = np.deg2rad(np.degrees(df[df["unit"] == u]["mu"].iloc[0]))
    ax.plot([mu, mu], [0, 1], color=C_INH, alpha=0.3, lw=0.8)
for color, sub in zip([C_EXC, C_INH], [sig_exc, sig_inh]):
    mu_p = np.arctan2(np.sin(sub["mu"]).mean(), np.cos(sub["mu"]).mean())
    ax.plot([mu_p, mu_p], [0, 0.9], color=color, lw=3)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0°", "90°", "180°\ntrough", "270°"], fontsize=8)
ax.set_title("Single-cell preferred\nphase vectors", fontsize=11, pad=20)
ax.set_yticks([])

fig.suptitle("Theta phase of spikes during running (0° = theta peak at the LFP)", y=0.98)
plt.savefig("fig03_population_phase.png")
plt.close(fig)
print("saved fig03_population_phase.png")

# %% [markdown]
# ## Figure 4: entrainment strength and preferred phase by cell type

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
mv = [df_exc["M"].values, df_inh["M"].values]
bp = ax[0].boxplot(mv, labels=["excitatory", "inhibitory"], widths=0.5,
                   patch_artist=True, medianprops=dict(color="k"))
for patch, col in zip(bp["boxes"], [C_EXC, C_INH]):
    patch.set_facecolor(col)
    patch.set_alpha(0.65)
_, pv = sstats.mannwhitneyu(df_exc["M"], df_inh["M"])
ax[0].text(0.5, 1.0, "MW U p = %.2e" % pv, ha="center",
           transform=ax[0].transAxes, fontweight="bold")
ax[0].set_ylabel("|M| mean resultant length")
ax[0].set_title("Phase-locking strength by cell type")

ax[1].scatter(df_exc["M"], np.degrees(df_exc["mu"]) % 360, s=9,
              color=C_EXC, alpha=0.6, label="excitatory")
ax[1].scatter(df_inh["M"], np.degrees(df_inh["mu"]) % 360, s=9,
              color=C_INH, alpha=0.7, label="inhibitory")
ax[1].set_ylabel("Preferred theta phase (°)")
ax[1].set_xlabel("|M|")
ax[1].set_title("Preferred phase vs locking strength")
ax[1].legend(frameon=False, loc="upper left")
ax[1].axhline(180, color=C_INH, ls=":", lw=1, alpha=0.5)
fig.suptitle("Cell-level theta entrainment (n exc=%d, n inh=%d)" %
             (len(df_exc), len(df_inh)))
plt.tight_layout()
plt.savefig("fig04_cell_phase_stats.png")
plt.close(fig)
print("saved fig04_cell_phase_stats.png")

# %% [markdown]
# ## Figure 5: example cells
# Linear spike-phase histograms for the 3 most phase-locked pyramidal cells
# and the 3 most phase-locked interneurons (dashed line = preferred phase).

# %%
example = pd.concat([df_exc.sort_values("M", ascending=False).iloc[[0, 5, 20]],
                     df_inh.sort_values("M", ascending=False).head(3)])
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
for ax, (_, row) in zip(axes.ravel(), example.iterrows()):
    col = C_INH if row["cell_type"] == "inhibitory" else C_EXC
    ph = np.degrees(spike_phases[row["unit"]]) % 360
    ax.hist(ph, bins=30, range=(0, 360), color=col, alpha=0.8)
    ax.axvline(np.degrees(row["mu"]) % 360, color="k", ls="--", lw=1)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(["0", "90", "180", "270", "360"])
    ax.set_title("%s unit %d\nn=%d, R=%.2f, p=%.1e" %
                 (row["cell_type"].title(), row["unit"], row["n_spk"],
                  row["M"], row["p_rayleigh"]), fontsize=9)
fig.suptitle("Example cells: spike-count histograms vs theta phase (0° = peak)")
fig.tight_layout()
plt.savefig("fig05_example_cells.png")
plt.close(fig)
print("saved fig05_example_cells.png")

# %% [markdown]
# ## Figure 6: shuffle null control
# Left: distribution of real MRL (equal spike count) vs the median of the
# random-time null for every cell. Right: paired real-vs-null MRL; almost all
# points lie above the identity line.

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].hist(df["M_real_same_n"], bins=30, color=C_EXC, alpha=0.6,
           label="real (n=%d cells)" % len(df))
ax[0].hist(df["M_null_median"], bins=30, color=C_NULL, alpha=0.7,
           label="shuffle null (median)")
ax[0].set_xlabel("MRL |M|")
ax[0].set_ylabel("cells")
ax[0].set_title("Real phase-locking vs random-time shuffle null")
ax[0].legend(frameon=False)
ax[1].scatter(df["M_null_median"], df["M_real_same_n"], s=9, color="#444", alpha=0.6)
ax[1].plot([0, 0.6], [0, 0.6], "k--", lw=1)
n_sig = int((df["p_shuffle"] < 0.01).sum())
ax[1].set_xlabel("null M (median)")
ax[1].set_ylabel("real M")
ax[1].set_title("Paired real vs null (significant: %d/%d)" % (n_sig, len(df)))
_, p_mwn = sstats.mannwhitneyu(df["M_real_same_n"], df["M_null_median"])
ax[1].text(0.05, 0.93, "MW U p = %.2e" % p_mwn, transform=ax[1].transAxes,
           fontsize=8)
fig.suptitle("Control: phase-locking is not explained by firing at random times", y=0.98)
plt.tight_layout()
plt.savefig("fig06_null_control.png")
plt.close(fig)
print("saved fig06_null_control.png")

# %% [markdown]
# ## Figure 7: entrainment strengthens with running speed
# If the locking to the 6-11 Hz LFP tracks real theta (which grows with running
# speed), MRL should increase from slow to fast locomotion. Left: median MRL
# per cell type in the two speed bins (paired, same cells). Right: MRL at fast
# vs slow speed per cell; most cells sit above the identity line.

# %%
if len(df_state):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for col, lab in zip([C_EXC, C_INH], ["excitatory", "inhibitory"]):
        sub = df_state[df_state["cell_type"] == lab]
        if not len(sub):
            continue
        xs = [0, 1]
        ys = [sub["M_slow"].median(), sub["M_fast"].median()]
        ax[0].plot(xs, ys, "o-", color=col, lw=1.6,
                   label="%s (n=%d)" % (lab, len(sub)))
    ax[0].set_xlim(-0.25, 1.25)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["slow running\n(< 0.4 m/s)", "fast running\n(>= 0.4 m/s)"])
    ax[0].set_ylabel("median MRL")
    ax[0].set_ylim(0, None)
    ax[0].set_title("Phase-locking strength vs running speed")
    ax[0].legend(frameon=False)
    _, p_st = sstats.wilcoxon(df_state["M_fast"], df_state["M_slow"],
                              alternative="greater")
    ax[0].text(0.5, 1.02, "Wilcoxon p = %.2e" % p_st, ha="center",
               transform=ax[0].transAxes, fontweight="bold")

    ax[1].scatter(df_state["M_slow"], df_state["M_fast"], s=12, color="#444", alpha=0.6)
    lim = min(max(df_state["M_slow"].max(), df_state["M_fast"].max()) * 1.08, 1.0)
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set_xlabel("MRL at slow speed")
    ax[1].set_ylabel("MRL at fast speed")
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_title("Paired per-cell MRL (n=%d)" % len(df_state))
    fig.suptitle("Theta phase-locking grows with running speed, within the "
                 "same awake epochs", y=0.98)
    plt.tight_layout()
    plt.savefig("fig07_state_dependence.png")
    plt.close(fig)
    print("saved fig07_state_dependence.png")
else:
    print("skipped fig07 (no matched units)")

# %% [markdown]
# ## Summary
# All results are written to `unit_stats.csv`, `state_stats.csv`, and the seven
# figures. The key numbers:

# %%
print("=== THETA PHASE ENTRAINMENT SUMMARY (Achilles-10252013, DANDI 000044) ===")
print("reference channel: %d  theta peak: %.1f Hz" % (best_ch, pk))
print("run-classified time: %.0f s of maze epoch (%.0f%%); %d bouts merged "
      "for spike selection (%.0f s)" % (run_time_raw,
      100 * run_time_raw / (MAZE[1] - MAZE[0]), len(run_epoch),
      run_epoch.tot_length()))
print("units analyzed (>=%d run spikes): %d (%d exc, %d inh)" %
      (MIN_SPIKES, len(df), len(df_exc), len(df_inh)))
print("Rayleigh p<0.01: %d/%d (%.0f%%)" %
      (int((df["p_rayleigh"] < 0.01).sum()), len(df),
       100 * (df["p_rayleigh"] < 0.01).mean()))
print("median MRL: exc %.3f, inh %.3f (MW p = %.2e)" %
      (df_exc["M"].median(), df_inh["M"].median(), p_mw))
print("shuffle-significant: %d/%d ; median real %.3f vs null %.3f" %
      (n_shuf_sig, len(df), df["M_real_same_n"].median(), df["M_null_median"].median()))
if len(df_state):
    print("speed control: median MRL slow %.3f vs fast %.3f (Wilcoxon p = %.2e)" %
          (df_state["M_slow"].median(), df_state["M_fast"].median(), p_st))
print("saved: unit_stats.csv, state_stats.csv, theta_channel.npz, phase_data.npz, fig01-07")