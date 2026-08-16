# %% [markdown]
# # Theta Phase Entrainment of Hippocampal CA1 Neurons (DANDI 000044)
#
# This notebook demonstrates theta phase entrainment (spike-phase locking) of
# hippocampal CA1 neurons during spatial behavior, using a classic Buzsaki lab
# session from the DANDI Archive: dandiset **000044**, session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` (rat CA1 tetrode
# recording, 137 sorted units, 128-channel LFP at 1250 Hz, 1.6 m linear maze).
#
# **Phenomenon.** During active exploration, the hippocampal local field
# potential (LFP) is dominated by the theta rhythm (4-12 Hz). The spike times
# of CA1 neurons are not uniform with respect to this rhythm: most pyramidal
# cells and interneurons preferentially fire at particular phases of the
# ongoing theta cycle. This "entrainment" is quantified per unit by the mean
# resultant length (MRL) of the spike-phase distribution and tested with the
# Rayleigh test for circular uniformity.
#
# **Pipeline.**
# 1. Stream the NWB file from DANDI with remfile + disk cache (no full download).
# 2. Pick the LFP channel with the strongest theta/delta power ratio as the
#    theta reference.
# 3. Bandpass filter (4-12 Hz) and extract instantaneous theta phase with the
#    Hilbert transform.
# 4. Restrict analysis to running bouts (smoothed speed > 10 cm/s) in the maze
#    epoch, where theta is strong.
# 5. For each unit, interpolate the theta phase at its spike times and compute
#    MRL, preferred phase, and a Rayleigh p-value; compare against a
#    random-time control that preserves spike count and the common phase
#    distribution.

# %% [markdown]
# ## Setup

# %%
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# %% [markdown]
# ## Streaming Access to the NWB File
#
# We resolve the S3 blob URL through the DANDI API. The `/download/` endpoint
# replies with a 302 redirect to a presigned URL; the bare blob URL (query
# string stripped) also serves range requests, which is what remfile needs.

# %%
DANDISET = "000044"
VERSION = "0.250624.0426"
ASSET_ID = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # sub-Achilles ses-Achilles-10252013

dl = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"
r = requests.get(dl, allow_redirects=False, timeout=30)
s3_url = r.headers["Location"].split("?")[0]
with open("s3_url.txt", "w") as f:
    f.write(s3_url + "\n")
print("S3 blob URL:", s3_url)

disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment03")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
print(f"{len(units)} units")
print(units.get_info("cell_type").value_counts())
print(units.get_info("location").value_counts())

es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs = es.rate            # 1250 Hz
conv = es.conversion    # int16 -> volts
print(f"LFP: {es.data.shape[1]} channels, {es.data.shape[0]} samples, {fs} Hz")

epochs = nwb["epochs"]
maze = epochs[epochs.label == "MazeEpoch"]
maze_start, maze_end = float(maze.start[0]), float(maze.end[0])
print(f"Maze epoch: {maze_start}-{maze_end:.0f} s")

# %% [markdown]
# ## Selecting a Theta Reference Channel
#
# The units' `shank_id` metadata does not map onto the electrode groups of the
# LFP, so per-shank LFP matching is not possible. Instead we use a single
# reference channel for the whole session: the channel with the highest
# theta (4-12 Hz) to delta (1-4 Hz) power ratio, measured on a 200 s chunk
# from the middle of the maze epoch.

# %%
t0, dur = 18500.0, 200.0
i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
chunk = es.data[i0:i1, :].astype(np.float64) * conv

freqs, psd = signal.welch(chunk, fs=fs, nperseg=int(4 * fs), axis=0)
theta_band = (freqs >= 4) & (freqs <= 12)
delta_band = (freqs >= 1) & (freqs < 4)
theta_power = np.trapezoid(psd[theta_band], freqs[theta_band], axis=0)
delta_power = np.trapezoid(psd[delta_band], freqs[delta_band], axis=0)
ratio = theta_power / delta_power

best_ch = int(np.argmax(ratio))
psd_best = psd[:, best_ch]
peak_f = freqs[theta_band][np.argmax(psd_best[theta_band])]
print(f"reference channel: {best_ch} (theta/delta = {ratio[best_ch]:.1f}, "
      f"theta peak {peak_f:.2f} Hz)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(np.arange(128), ratio, "o-", ms=3, lw=0.8, color="0.3")
axes[0].axvline(best_ch, color="crimson", ls="--", lw=1)
axes[0].set_xlabel("LFP channel")
axes[0].set_ylabel("theta (4-12 Hz) / delta (1-4 Hz) power")
axes[0].set_title(f"Theta/delta ratio per channel (best: ch {best_ch})")
band = freqs <= 60
axes[1].semilogy(freqs[band], psd_best[band], color="k", lw=1.2)
axes[1].axvspan(4, 12, color="tab:blue", alpha=0.15, label="theta band")
axes[1].axvline(peak_f, color="crimson", ls="--", lw=1, label=f"peak {peak_f:.2f} Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD (V$^2$/Hz)")
axes[1].set_title(f"PSD of reference channel {best_ch} (maze chunk)")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig_theta_channel_selection.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Theta Phase Extraction and Running Bouts
#
# We load the full maze epoch of the reference channel (with a 10 s buffer to
# avoid filter edge effects), bandpass filter in the theta band, and take the
# angle of the Hilbert transform as the instantaneous phase. Phase 0 is the
# peak of the filtered LFP; +/-pi is the trough.
#
# Running speed comes from the 2D position series. Note a data quirk: the
# SpatialSeries stores the sampling *period* (0.0256 s) in the `rate` field,
# so the timestamps pynapple constructs are spaced 39 s apart. We reconstruct
# correct timestamps from the true 39.06 Hz rate. Speed is interpolated only
# across short tracking dropouts (< 0.5 s); long dropouts are treated as
# stationary. Running bouts are intervals with smoothed speed above 10 cm/s,
# merging gaps under 0.5 s and keeping bouts of at least 0.5 s.

# %%
buf = 10.0
i0 = int((maze_start - buf) * fs)
i1 = int((maze_end + buf) * fs)
lfp = es.data[i0:i1, best_ch].astype(np.float64) * conv
t_lfp = np.arange(i0, i1) / fs

b, a = signal.butter(4, [4, 12], btype="bandpass", fs=fs)
lfp_filt = signal.filtfilt(b, a, lfp)
phase = np.angle(signal.hilbert(lfp_filt))

keep = (t_lfp >= maze_start) & (t_lfp <= maze_end)
t_lfp, lfp, lfp_filt, phase = t_lfp[keep], lfp[keep], lfp_filt[keep], phase[keep]

# --- position and speed ---
pos = nwb["1.6mLinearMazeSpatialSeries"]
xy = pos.values
ss = nwbfile.processing["behavior"]["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
fs_pos = 1.0 / float(ss.rate)  # stored 'rate' is actually the sampling period
pt = maze_start + np.arange(xy.shape[0]) / fs_pos
valid = np.isfinite(xy).all(axis=1)
print(f"position: {fs_pos:.1f} Hz, {valid.mean()*100:.0f}% valid samples in maze epoch")

dt = np.diff(pt)
step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
ok = valid[:-1] & valid[1:] & (dt > 0)
spd_t = (pt[:-1] + pt[1:]) / 2
spd = np.full_like(spd_t, np.nan, dtype=np.float64)
spd[ok] = step[ok] / dt[ok]

okv = np.isfinite(spd)
spd_i = np.interp(spd_t, spd_t[okv], spd[okv])
inv = ~valid
edges_inv = np.diff(inv.astype(int))
g_starts = np.where(edges_inv == 1)[0] + 1
g_ends = np.where(edges_inv == -1)[0] + 1
if inv[0]:
    g_starts = np.r_[0, g_starts]
if inv[-1]:
    g_ends = np.r_[g_ends, len(inv)]
for gs, ge in zip(g_starts, g_ends):
    if (ge - gs) / fs_pos >= 0.5:
        in_gap = (spd_t >= pt[gs]) & (spd_t <= pt[min(ge, len(pt) - 1)])
        spd_i[in_gap] = 0.0

sig_s = 0.25 * fs_pos
w = signal.windows.gaussian(int(sig_s * 8) | 1, std=sig_s)
w /= w.sum()
spd_s = np.convolve(spd_i, w, mode="same")

# --- running bouts ---
above = spd_s > 0.10
edges = np.diff(above.astype(int))
starts = spd_t[:-1][edges == 1]
ends = spd_t[:-1][edges == -1]
if above[0]:
    starts = np.r_[spd_t[0], starts]
if above[-1]:
    ends = np.r_[ends, spd_t[-1]]
merged = []
for s, e in np.column_stack([starts, ends]):
    if merged and s - merged[-1][1] < 0.5:
        merged[-1][1] = e
    else:
        merged.append([s, e])
bouts = np.array([b_ for b_ in merged if b_[1] - b_[0] >= 0.5])
run_ep = nap.IntervalSet(start=bouts[:, 0], end=bouts[:, 1])
run_total = float(np.sum(bouts[:, 1] - bouts[:, 0]))
print(f"running bouts: {len(bouts)}, total {run_total:.0f} s of {maze_end - maze_start:.0f} s maze epoch")

# %% [markdown]
# ## Raw Data Validation: LFP, Theta Phase, and Spikes
#
# A 3 s window during a long running bout: raw LFP on the reference channel,
# the theta-filtered trace with the instantaneous phase, and a raster of the
# eight most strongly phase-locked units. The rhythmic modulation of the
# interneurons is already visible by eye.

# %%
bout_lens = bouts[:, 1] - bouts[:, 0]
long_bout = bouts[np.argmax(bout_lens)]
t_mid = long_bout.mean()
w0, w1 = t_mid - 6, t_mid + 6
seg = es.data[int(w0 * fs):int(w1 * fs), best_ch].astype(np.float64) * conv
t_seg = np.arange(int(w0 * fs), int(w1 * fs)) / fs
seg_filt = signal.filtfilt(b, a, seg)
seg_phase = np.angle(signal.hilbert(seg_filt))
show = (t_seg >= t_mid - 1.5) & (t_seg <= t_mid + 1.5)
t_show = t_seg[show] - t_seg[show][0]
t_abs0 = t_seg[show][0]

# (top units are picked after the stats loop; placeholder replaced below)

# %% [markdown]
# ## Spike-Phase Locking Statistics
#
# For every unit with at least 50 spikes inside running bouts, we interpolate
# the theta phase at each spike time (on the unit circle, to avoid wraparound
# artifacts) and compute:
#
# - **MRL** (mean resultant length): 0 = uniform over phases, 1 = all spikes at
#   one phase.
# - **Rayleigh p-value** for circular uniformity (Zar's approximation).
# - **Preferred phase**: angle of the mean resultant vector.
#
# As a control, we repeat the measurement with the same number of spikes drawn
# at random times inside the running bouts. This preserves spike count and the
# (non-uniform) marginal distribution of theta phase itself, so any remaining
# structure in the real data must come from genuine spike-phase coupling.

# %%
cos_p, sin_p = np.cos(phase), np.sin(phase)

def phases_at(times):
    c = np.interp(times, t_lfp, cos_p)
    s = np.interp(times, t_lfp, sin_p)
    return np.arctan2(s, c)

def rayleigh_p(n, R):
    z = n * R**2
    p = np.exp(-z)
    if n < 50:
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * R) ** 2)) - (1 + 2 * n))
    return min(p, 1.0)

cell_type = units.get_info("cell_type")
location = units.get_info("location")

rows = []
spike_phases = {}
for uid in tqdm(list(units.keys()), desc="units"):
    spk = units[uid].restrict(run_ep)
    n = len(spk)
    if n < 50:
        continue
    ph = phases_at(spk.t)
    v = np.mean(np.exp(1j * ph))
    rows.append(dict(unit=uid, cell_type=cell_type[uid], location=location[uid],
                     n_spikes=n, rate=n / run_total, mrl=np.abs(v),
                     rayleigh_p=rayleigh_p(n, np.abs(v)), pref_phase=np.angle(v)))
    spike_phases[uid] = ph
res = pd.DataFrame(rows)

# random-time control
bout_lens = bouts[:, 1] - bouts[:, 0]
cum = np.cumsum(bout_lens)

def random_run_times(n):
    u = rng.random(n) * cum[-1]
    idx = np.searchsorted(cum, u)
    prev = np.r_[0, cum[:-1]][idx]
    return bouts[idx, 0] + (u - prev)

res["mrl_control"] = [np.abs(np.mean(np.exp(1j * phases_at(random_run_times(int(n))))))
                      for n in tqdm(res["n_spikes"], desc="control")]
res.to_csv("unit_phase_locking.csv", index=False)

sig_mask = res["rayleigh_p"] < 0.01
exc = res["cell_type"] == "excitatory"
inh = res["cell_type"] == "inhibitory"
print(f"\nunits analyzed: {len(res)}; locked (Rayleigh p<0.01): {sig_mask.sum()} ({sig_mask.mean()*100:.0f}%)")
for ct, m in [("excitatory", exc), ("inhibitory", inh)]:
    sub = res[m]
    s = sub["rayleigh_p"] < 0.01
    pooled = np.angle(np.mean(np.exp(1j * sub.loc[s, "pref_phase"])))
    print(f"{ct}: {s.sum()}/{len(sub)} locked; median MRL {sub['mrl'].median():.3f} "
          f"(control {sub['mrl_control'].median():.3f}); pooled preferred phase {np.degrees(pooled):.0f} deg")
mw_p = stats.mannwhitneyu(res.loc[exc, "mrl"], res.loc[inh, "mrl"]).pvalue
print(f"exc vs inh MRL, Mann-Whitney p = {mw_p:.1e}")

# %% [markdown]
# ## Figure: Raw LFP, Theta Phase, and Spike Raster

# %%
top_units = res[sig_mask].nlargest(8, "mrl")["unit"].tolist()

fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.2, 1.2, 1.6]))
axes[0].plot(t_show, seg[show] * 1e3, color="0.4", lw=0.7)
axes[0].set_ylabel("raw LFP (mV)")
axes[0].set_title(f"CA1 LFP channel {best_ch} during running (t = {t_mid:.1f} s, maze epoch)")
axes[1].plot(t_show, seg_filt[show] * 1e3, color="tab:blue", lw=1.0)
axes[1].set_ylabel("theta-filtered\n4-12 Hz (mV)")
ax_ph = axes[1].twinx()
ax_ph.plot(t_show, seg_phase[show], color="tab:orange", lw=0.8, alpha=0.7)
ax_ph.set_ylabel("theta phase (rad)", color="tab:orange")
ax_ph.set_yticks([-np.pi, 0, np.pi])
ax_ph.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
ax_ph.tick_params(axis="y", colors="tab:orange")
for i, uid in enumerate(top_units):
    spk = units[uid].t
    spk = spk[(spk >= t_abs0) & (spk <= t_abs0 + 3.0)] - t_abs0
    axes[2].vlines(spk, i + 0.1, i + 0.9, color="k", lw=0.8)
axes[2].set_yticks(np.arange(8) + 0.5)
axes[2].set_yticklabels([f"u{uid} ({res[res.unit == uid]['cell_type'].iloc[0][:3]})"
                         for uid in top_units], fontsize=8)
axes[2].set_ylabel("units (top MRL)")
axes[2].set_xlabel("Time within 3 s window (s)")
axes[2].set_ylim(0, 8.2)
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig("fig_raw_lfp_theta.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Figure: Running Speed and Bout Selection

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
ds = np.arange(0, len(spd_t), 25)
axes[0].plot(spd_t[ds], spd_s[ds] * 100, color="k", lw=0.6)
axes[0].axhline(10, color="crimson", ls="--", lw=1, label="10 cm/s threshold")
for s, e in bouts[:: max(1, len(bouts) // 400)]:
    axes[0].axvspan(s, e, color="tab:blue", alpha=0.06, lw=0)
axes[0].set_ylabel("speed (cm/s)")
axes[0].set_xlabel("Time (s)")
axes[0].set_title("Running speed over the full maze epoch (blue shading: running bouts used)")
axes[0].legend(loc="upper right")
axes[0].set_xlim(maze_start, maze_end)
z0 = long_bout[0] - 20
z1 = z0 + 120
zm = (spd_t >= z0) & (spd_t <= z1)
axes[1].plot(spd_t[zm], spd_s[zm] * 100, color="k", lw=1.0)
axes[1].axhline(10, color="crimson", ls="--", lw=1)
for s, e in bouts[(bouts[:, 1] > z0) & (bouts[:, 0] < z1)]:
    axes[1].axvspan(max(s, z0), min(e, z1), color="tab:blue", alpha=0.15, lw=0)
axes[1].set_ylabel("speed (cm/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Zoom: 120 s around the longest running bout")
axes[1].set_xlim(z0, z1)
fig.tight_layout()
fig.savefig("fig_speed_running_bouts.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Figure: Example Spike-Phase Histograms

# %%
examples = []
for ct, df in [("excitatory", res[exc & sig_mask]), ("inhibitory", res[inh & sig_mask])]:
    examples += df.nlargest(2, "mrl")["unit"].tolist()

fig, axes = plt.subplots(1, 4, subplot_kw=dict(projection="polar"), figsize=(13, 4.6))
for ax, uid in zip(axes, examples):
    ph = spike_phases[uid]
    row = res[res.unit == uid].iloc[0]
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges_h = np.histogram(ph, bins=bins)
    centers = (edges_h[:-1] + edges_h[1:]) / 2
    heights = counts / counts.sum()
    ax.bar(centers, heights, width=np.diff(edges_h), color="tab:blue", alpha=0.75,
           edgecolor="k", lw=0.4)
    ax.plot([row["pref_phase"], row["pref_phase"]], [0, heights.max()], color="crimson", lw=2)
    ax.set_theta_zero_location("N")
    ax.set_title(f"unit {uid} ({row['cell_type'][:3]})\nMRL={row['mrl']:.2f}, "
                 f"p={row['rayleigh_p']:.0e}, n={int(row['n_spikes'])}", fontsize=9, pad=26)
    ax.set_rticks([])
fig.suptitle("Spike-phase histograms of strongly locked units (0 = theta LFP peak)", y=1.04)
fig.tight_layout()
fig.savefig("fig_example_phase_histograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure: Population Distributions of Locking Strength

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
bins = np.linspace(0, 0.55, 40)
axes[0].hist(res.loc[exc, "mrl"], bins=bins, alpha=0.7, color="tab:blue",
             label=f"excitatory (n={exc.sum()})", density=True)
axes[0].hist(res.loc[inh, "mrl"], bins=bins, alpha=0.7, color="tab:red",
             label=f"inhibitory (n={inh.sum()})", density=True)
axes[0].hist(res["mrl_control"], bins=bins, histtype="step", color="k", lw=1.5,
             label="random-time control", density=True)
axes[0].set_xlabel("Mean resultant length (MRL)")
axes[0].set_ylabel("density")
axes[0].set_title("Theta phase-locking strength by cell type")
axes[0].legend(fontsize=9)

data = [res.loc[exc, "mrl"], res.loc[inh, "mrl"], res["mrl_control"]]
bp = axes[1].boxplot(data, tick_labels=["excitatory", "inhibitory", "control"],
                     showfliers=False, widths=0.5, patch_artist=True)
for patch, c in zip(bp["boxes"], ["tab:blue", "tab:red", "0.7"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for i, d in enumerate(data):
    axes[1].scatter(np.full(len(d), i + 1) + np.random.default_rng(0).normal(0, 0.05, len(d)),
                    d, s=6, color="k", alpha=0.35, zorder=3)
axes[1].set_ylabel("MRL")
axes[1].set_title(f"exc vs inh Mann-Whitney p = {mw_p:.1e}")
fig.tight_layout()
fig.savefig("fig_mrl_distributions.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Figure: Preferred-Phase Distribution and Locking vs Firing Rate

# %%
fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection="polar"), figsize=(10.5, 4.8))
for ax, (ct, m, color) in zip(axes, [("excitatory", exc, "tab:blue"), ("inhibitory", inh, "tab:red")]):
    ph = res.loc[m & sig_mask, "pref_phase"].values
    bins = np.linspace(-np.pi, np.pi, 19)
    counts, edges_h = np.histogram(ph, bins=bins)
    centers = (edges_h[:-1] + edges_h[1:]) / 2
    ax.bar(centers, counts, width=np.diff(edges_h), color=color, alpha=0.75, edgecolor="k", lw=0.4)
    pooled = np.angle(np.mean(np.exp(1j * ph)))
    ax.plot([pooled, pooled], [0, counts.max()], color="k", lw=2, ls="--")
    ax.set_theta_zero_location("N")
    ax.set_title(f"{ct} (n={len(ph)} locked)\npooled preferred phase {np.degrees(pooled):.0f}°",
                 fontsize=10, pad=26)
    ax.set_rticks([])
fig.suptitle("Preferred theta phase of significantly locked units (0 = LFP peak)", y=1.04)
fig.tight_layout()
fig.savefig("fig_preferred_phase_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m, color, label in [(exc, "tab:blue", "excitatory"), (inh, "tab:red", "inhibitory")]:
    s = m & sig_mask
    ax.scatter(res.loc[s, "rate"], res.loc[s, "mrl"], s=16, color=color, alpha=0.8,
               label=f"{label} locked", edgecolor="k", lw=0.3)
    ax.scatter(res.loc[m & ~sig_mask, "rate"], res.loc[m & ~sig_mask, "mrl"], s=16,
               facecolor="none", edgecolor=color, alpha=0.8, label=f"{label} n.s.")
ax.set_xscale("log")
ax.set_xlabel("firing rate during running bouts (Hz)")
ax.set_ylabel("MRL")
ax.set_title("Phase-locking strength vs firing rate")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig_mrl_vs_rate.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Results
#
# In this session, **104 of 137 CA1 units (76%) are significantly entrained to
# theta** (Rayleigh p < 0.01) during running: 88/120 putative pyramidal cells
# and 16/17 putative interneurons. Locking is much stronger in interneurons
# (median MRL 0.26) than in pyramidal cells (median MRL 0.11;
# Mann-Whitney p ~ 6e-5), while the random-time control sits near 0.02, so the
# effect is not a byproduct of the theta waveform's own phase distribution.
# Preferred phases are tightly clustered and cell-type specific: pyramidal
# cells concentrate around -139 degrees relative to the reference-channel LFP
# peak (descending phase, approaching the trough), interneurons around +123
# degrees. Because the exact layer of the reference channel is unknown, the
# absolute phase values are convention-dependent; the robust findings are the
# prevalence, strength, and cell-type specificity of the locking.
#
# Locking strength is only weakly related to firing rate: many sparse
# pyramidal cells (place cells firing only in their field) still show
# significant entrainment, consistent with theta organizing spike timing
# independently of overall rate.
