# %% [markdown]
# # Theta Phase Entrainment of Hippocampal CA1 Neurons
#
# This notebook demonstrates theta phase entrainment (phase locking) of hippocampal
# neurons using data from the DANDI Archive. During active exploration, the rodent
# hippocampus generates a prominent 4-12 Hz theta oscillation, and CA1 pyramidal
# cells and interneurons fire preferentially at particular phases of this rhythm.
# This phase organization is thought to structure spike timing for encoding and
# plasticity.
#
# We use the classic Buzsaki lab session `sub-Achilles_ses-Achilles-10252013` from
# dandiset 000044 ("Diversity in neural firing dynamics supports..."), which contains
# 137 sorted CA1 units (120 excitatory, 17 inhibitory) and a 128-channel, 1250 Hz LFP
# recording spanning a ~35 minute linear-maze epoch. We extract the instantaneous
# theta phase from a reference LFP channel, compute the theta phase at every spike
# during running bouts, and quantify phase locking with the mean resultant length
# (circular concentration) and the Rayleigh test, with a spike-train circular-shift
# null as a control.
#
# Phase convention: 0 rad is the peak of the 4-12 Hz filtered LFP on the reference
# channel, and +/-pi is the trough.

# %% [markdown]
# ## Setup and Data Access
#
# We stream the NWB file with remfile plus a disk cache, resolving a fresh presigned
# S3 URL from the DANDI API at runtime (the presigned URLs expire, so this must be
# done each run).

# %%
import os
import pickle

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from tqdm import tqdm

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# Resolve a fresh presigned URL for the asset (GET-only redirect; do not HEAD it).
api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api, params={"path": ASSET_PATH})
r.raise_for_status()
asset_id = r.json()["results"][0]["asset_id"]
r2 = requests.get(f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
                  allow_redirects=False)
s3_url = r2.headers["Location"]
print("Resolved asset:", ASSET_PATH)

disk_cache = remfile.DiskCache("/tmp/remfile_cache_theta_entrainment")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The file contains a `units` TsGroup (137 CA1 units with `cell_type` and `location`
# metadata), an `epochs` IntervalSet (PRE / Maze / POST), the maze position
# (`1.6mLinearMazeSpatialSeries`, ~39 Hz), and a 128-channel LFP ElectricalSeries at
# 1250 Hz. The LFP is far too large to load whole (43.6M samples x 128 channels), so
# we read channel slices directly from the underlying HDF5 dataset.

# %%
FS_LFP = 1250.0
lfp_dset = h5py_file["processing/ecephys/LFP/LFP/data"]
CONV_UV = lfp_dset.attrs["conversion"] * 1e6  # int16 -> microvolts
print("LFP dataset:", lfp_dset.shape, lfp_dset.dtype)
print("LFP rate:", h5py_file["processing/ecephys/LFP/LFP/starting_time"].attrs["rate"], "Hz")

epochs = nwb["epochs"]
print(epochs)
MAZE_T0, MAZE_T1 = 18079.5, 20147.0  # MazeEpoch bounds (s)

# %% [markdown]
# ## Choosing a Theta Reference Channel
#
# The electrodes table does not annotate which channel sits in the CA1 pyramidal
# layer, so we pick the reference channel data-driven: we read a 60 s window of all
# 128 channels from inside the maze epoch, compute the power spectrum of each, and
# take the channel with the largest theta/delta power ratio. Theta phase reverses
# across layers, so all phase values below are relative to this one reference
# channel; only relative differences between units are layer-independent.

# %%
t0_win, t1_win = 18500.0, 18560.0
i0_win, i1_win = int(t0_win * FS_LFP), int(t1_win * FS_LFP)
win = lfp_dset[i0_win:i1_win, :].astype(np.float64) * CONV_UV

f_psd, psd = signal.welch(win, fs=FS_LFP, nperseg=4096, axis=0)
theta_band = (f_psd >= 4) & (f_psd <= 12)
delta_band = (f_psd >= 1) & (f_psd <= 4)
theta_delta_ratio = psd[theta_band].mean(axis=0) / psd[delta_band].mean(axis=0)
REF_CH = int(np.argmax(theta_delta_ratio))
print(f"Reference channel: {REF_CH} (theta/delta ratio {theta_delta_ratio[REF_CH]:.2f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(f_psd, psd[:, REF_CH], color="k")
axes[0].axvspan(4, 12, color="orange", alpha=0.2, label="theta (4-12 Hz)")
axes[0].set_xlim(0, 60)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("PSD (uV$^2$/Hz)")
axes[0].set_title(f"Power spectrum, channel {REF_CH}")
axes[0].legend()
axes[1].plot(theta_delta_ratio, "o-", markersize=3, color="steelblue")
axes[1].axvline(REF_CH, color="r", ls="--", label=f"reference ch {REF_CH}")
axes[1].set_xlabel("LFP channel")
axes[1].set_ylabel("Theta / delta power ratio")
axes[1].set_title("Channel selection")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_psd_channel_selection.png", dpi=150)
plt.close()

# %% [markdown]
# The spectrum shows a clear theta peak near 8 Hz, as expected during maze running.
#
# ## Theta Phase Extraction
#
# We read the full maze epoch (~34.5 min) for the reference channel, bandpass filter
# at 4-12 Hz (4th-order Butterworth, zero-phase), and take the angle of the Hilbert
# transform as the instantaneous theta phase.

# %%
i0, i1 = int(MAZE_T0 * FS_LFP), int(MAZE_T1 * FS_LFP)
print("Reading maze-epoch LFP for reference channel...", flush=True)
lfp = lfp_dset[i0:i1, REF_CH].astype(np.float64) * CONV_UV
t_lfp = np.arange(i0, i1) / FS_LFP
print(f"  {lfp.shape[0]} samples, {t_lfp[0]:.1f}-{t_lfp[-1]:.1f} s")

b, a = signal.butter(4, [4, 12], btype="band", fs=FS_LFP)
lfp_theta = signal.filtfilt(b, a, lfp)
analytic = signal.hilbert(lfp_theta)
theta_phase = np.angle(analytic)      # -pi..pi, 0 = peak of filtered signal
theta_amp = np.abs(analytic)

# %% [markdown]
# ## Defining Run Epochs
#
# Theta is strongest during locomotion, and the animal spends much of the maze epoch
# sitting at the reward platforms. We compute running speed from the 2D position,
# smooth it, and keep bouts faster than 10 cm/s, merging gaps shorter than 0.5 s and
# dropping bouts shorter than 1 s. Phase locking is assessed on spikes inside these
# run bouts only.

# %%
pos = nwb["1.6mLinearMazeSpatialSeries"]
xy = pos.values
t_pos = pos.index.values
dt_pos = np.median(np.diff(t_pos))
speed = np.sqrt(np.gradient(xy[:, 0], dt_pos) ** 2 + np.gradient(xy[:, 1], dt_pos) ** 2)
kern = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
kern /= kern.sum()
speed_s = np.convolve(speed, kern, mode="same")
speed_s[np.isnan(xy[:, 0])] = np.nan

is_run = speed_s > 0.10  # m/s
d_run = np.diff(is_run.astype(int))
starts, stops = t_pos[:-1][d_run == 1], t_pos[:-1][d_run == -1]
if is_run[0]:
    starts = np.r_[t_pos[0], starts]
if is_run[-1]:
    stops = np.r_[stops, t_pos[-1]]
merged_s, merged_e = [], []
for s, e in zip(starts, stops):
    if merged_s and s - merged_e[-1] < 0.5:
        merged_e[-1] = e
    else:
        merged_s.append(s)
        merged_e.append(e)
run_ep = nap.IntervalSet(
    start=[s for s, e in zip(merged_s, merged_e) if e - s >= 1.0],
    end=[e for s, e in zip(merged_s, merged_e) if e - s >= 1.0],
)
print(f"Run bouts: {len(run_ep)}, total {run_ep.tot_length():.0f} s "
      f"({100 * run_ep.tot_length() / (MAZE_T1 - MAZE_T0):.0f}% of maze epoch)")

# %% [markdown]
# ## Visual Validation: LFP, Theta Phase, and Spikes
#
# Before computing any statistics we inspect a 2 s snippet: the raw LFP with its
# 4-12 Hz component, the instantaneous phase, and a raster of 40 CA1 units. If
# entrainment is present, spikes should cluster at particular phases of the theta
# cycles rather than sprinkling uniformly.

# %%
units = nwb["units"]
unit_ids = list(units.keys())
cell_type = units.get_info("cell_type")

# Pick a 2 s window in the middle of the run bout with the strongest median theta
# amplitude, so the rhythm and its spike alignment are both clearly visible.
best_bout, best_amp = None, -np.inf
for s, e in zip(run_ep.start, run_ep.end):
    if e - s < 4:
        continue
    mm = (t_lfp >= s + 1) & (t_lfp < e - 1)
    med = np.median(theta_amp[mm])
    if med > best_amp:
        best_amp, best_bout = med, (s, e)
tt0 = 0.5 * (best_bout[0] + best_bout[1]) - 1.0
tt1 = tt0 + 2.0
m = (t_lfp >= tt0) & (t_lfp < tt1)
t_rel = t_lfp[m] - tt0
snippet_ep = nap.IntervalSet(start=[tt0], end=[tt1])

# Raster the 40 units with the most run-bout spikes, sorted by preferred phase, so
# that phase alignment appears as bands tracking the phase sawtooth.
cos_tmp, sin_tmp = np.cos(theta_phase), np.sin(theta_phase)
rate_rows = []
for uid in unit_ids:
    spk = units[uid].restrict(run_ep).index.values
    if len(spk) < 50:
        continue
    ph = np.arctan2(np.interp(spk, t_lfp, sin_tmp), np.interp(spk, t_lfp, cos_tmp))
    rate_rows.append((uid, len(spk), np.angle(np.mean(np.exp(1j * ph)))))
rate_rows.sort(key=lambda r: -r[1])
top40 = sorted(rate_rows[:40], key=lambda r: r[2])

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 1, 1.4]))
axes[0].plot(t_rel, lfp[m], lw=0.5, color="0.4", label="raw LFP")
axes[0].plot(t_rel, lfp_theta[m], lw=1.2, color="darkorange", label="4-12 Hz")
axes[0].set_ylabel("LFP (uV)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"Raw LFP and theta component (channel {REF_CH}, during a run bout)")

axes[1].plot(t_rel, theta_phase[m], lw=0.9, color="steelblue")
axes[1].set_ylabel("Theta phase (rad)")
axes[1].set_yticks([-np.pi, 0, np.pi])
axes[1].set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
axes[1].set_title("Instantaneous theta phase (Hilbert)")

ax = axes[2]
for row_i, (uid, nspk, pref) in enumerate(top40):
    spk = units[uid].restrict(snippet_ep).index.values - tt0
    color = "firebrick" if cell_type[uid] == "inhibitory" else "k"
    ax.vlines(spk, row_i + 0.6, row_i + 1.4, color=color, lw=0.8)
ax.set_ylabel("Unit (sorted by preferred phase)")
ax.set_xlabel(f"Time from {tt0:.1f} s (s)")
ax.set_title("Spike raster, 40 most active units (black = excitatory, red = inhibitory)")
ax.set_ylim(0.5, len(top40) + 1)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_raw_lfp_phase_raster.png", dpi=150)
plt.close()
print("saved fig2")

# %% [markdown]
# ## Phase-Locking Statistics
#
# For each unit we take its spikes inside run bouts, interpolate the theta phase at
# each spike time (interpolating cos and sin separately to avoid phase-wrap
# artifacts), and compute the mean resultant length R = |mean(exp(i*phase))|, the
# preferred phase, and the Rayleigh test p-value (Zar's approximation for small
# samples). Units with fewer than 100 spikes in run bouts are excluded.

# %%
cos_p, sin_p = np.cos(theta_phase), np.sin(theta_phase)

def spike_phases(spk_t):
    return np.arctan2(np.interp(spk_t, t_lfp, sin_p), np.interp(spk_t, t_lfp, cos_p))

def rayleigh_p(n, R):
    z = n * R ** 2
    p = np.exp(-z)
    if n < 50:
        p *= (1 + (2 * z - z ** 2) / (4 * n)
              - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    return min(p, 1.0)

MIN_SPIKES = 100
rows, spk_phases, spk_times_run = [], {}, {}
for uid in tqdm(unit_ids, desc="phase locking"):
    spk = units[uid].restrict(run_ep).index.values
    if len(spk) < MIN_SPIKES:
        continue
    ph = spike_phases(spk)
    mean_vec = np.mean(np.exp(1j * ph))
    rows.append(dict(uid=int(uid), n_spikes=len(spk), R=np.abs(mean_vec),
                     pref_phase=np.angle(mean_vec), p_rayleigh=rayleigh_p(len(spk), np.abs(mean_vec)),
                     cell_type=cell_type[uid], location=units.get_info("location")[uid]))
    spk_phases[int(uid)] = ph
    spk_times_run[int(uid)] = spk

df = pd.DataFrame(rows)
df["significant"] = df["p_rayleigh"] < 0.05
df.to_csv("phase_locking_results.csv", index=False)
print(f"\nUnits analyzed (>= {MIN_SPIKES} spikes in run bouts): {len(df)}")
print(df.groupby("cell_type")[["R", "n_spikes"]].describe().round(3))
for ct in ["excitatory", "inhibitory"]:
    sub = df[df.cell_type == ct]
    print(f"{ct}: {sub.significant.sum()}/{len(sub)} significantly phase-locked "
          f"(Rayleigh p<0.05), median R = {sub.R.median():.3f}")

# %% [markdown]
# ## Circular-Shift Null Control
#
# The Rayleigh test assumes a uniform null, but finite data and non-stationary LFP
# can bias it. As a control we circularly shift the phase signal by a random lag
# (1 s to the full epoch), which destroys the spike-to-phase correspondence while
# preserving every other statistic of both spike trains and LFP, and recompute R for
# every unit. Repeating this 100 times gives a null distribution for each unit and
# for the population mean R.

# %%
rng = np.random.default_rng(42)
N_SHUF = 100
n_samp = len(theta_phase)
min_shift = int(1.0 * FS_LFP)

t_cos = np.cos(theta_phase)  # reused after rolling
shuf_R = {uid: [] for uid in spk_times_run}
shuf_mean_R = []
for i_sh in tqdm(range(N_SHUF), desc="shuffles"):
    shift = int(rng.integers(min_shift, n_samp - min_shift))
    c_sh = np.roll(cos_p, shift)
    s_sh = np.roll(sin_p, shift)
    Rs = []
    for uid, spk in spk_times_run.items():
        ph = np.arctan2(np.interp(spk, t_lfp, s_sh), np.interp(spk, t_lfp, c_sh))
        R = np.abs(np.mean(np.exp(1j * ph)))
        shuf_R[uid].append(R)
        Rs.append(R)
    shuf_mean_R.append(np.mean(Rs))

shuf_mean_R = np.array(shuf_mean_R)
df["R_null_mean"] = df["uid"].map(lambda u: np.mean(shuf_R[u]))
df["p_shuffle"] = df["uid"].map(lambda u: (np.sum(np.array(shuf_R[u]) >=
                                df.loc[df.uid == u, "R"].iloc[0]) + 1) / (N_SHUF + 1))
df.to_csv("phase_locking_results.csv", index=False)
obs_mean_R = df.R.mean()
print(f"Observed population mean R = {obs_mean_R:.4f}; "
      f"null mean R = {shuf_mean_R.mean():.4f} +/- {shuf_mean_R.std():.4f} "
      f"(p = {(np.sum(shuf_mean_R >= obs_mean_R) + 1) / (N_SHUF + 1):.4f})")
print(f"Units significant by shuffle (p<0.05): {(df.p_shuffle < 0.05).sum()}/{len(df)}")

# %% [markdown]
# ## Example Phase-Locked Neurons
#
# Phase histograms for the two most strongly locked excitatory and inhibitory units.
# The red line marks each unit's preferred (circular mean) phase.

# %%
exc_top = df[df.cell_type == "excitatory"].sort_values("R", ascending=False).head(2)
inh_top = df[df.cell_type == "inhibitory"].sort_values("R", ascending=False).head(2)
examples = pd.concat([exc_top, inh_top])

bins = np.linspace(-np.pi, np.pi, 37)
centers = 0.5 * (bins[:-1] + bins[1:])
fig = plt.figure(figsize=(13, 7))
for i, (_, row) in enumerate(examples.iterrows()):
    ph = spk_phases[row.uid]
    ax = fig.add_subplot(2, 4, i + 1)
    ax.hist(ph, bins=bins, color="steelblue" if row.cell_type == "excitatory" else "firebrick",
            alpha=0.85)
    ax.axvline(row.pref_phase, color="k", lw=1.5, ls="--")
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
    ax.set_xlabel("Theta phase (rad)")
    ax.set_ylabel("Spike count")
    ax.set_title(f"unit {row.uid} ({row.cell_type})\n"
                 f"R = {row.R:.2f}, Rayleigh p = {row.p_rayleigh:.1e}", fontsize=10)
    axp = fig.add_subplot(2, 4, i + 5, projection="polar")
    counts, _ = np.histogram(ph, bins=bins)
    axp.bar(centers, counts, width=np.diff(bins),
            color="steelblue" if row.cell_type == "excitatory" else "firebrick", alpha=0.85)
    axp.set_theta_zero_location("E")
    axp.set_title(f"preferred phase = {np.degrees(row.pref_phase):.0f} deg", fontsize=10, pad=26)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_example_phase_histograms.png", dpi=150)
plt.close()
print("saved fig3")

# %% [markdown]
# ## Population Summary
#
# Across the population we expect inhibitory interneurons to lock more strongly than
# pyramidal cells, and we expect preferred phases to cluster rather than cover the
# circle uniformly. The shuffle control panel shows the observed mean R far outside
# the null distribution.

# %%
fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(2, 3, 1)
Rb = np.linspace(0, 0.6, 31)
for ct, color in [("excitatory", "steelblue"), ("inhibitory", "firebrick")]:
    sub = df[df.cell_type == ct]
    ax1.hist(sub.R, bins=Rb, alpha=0.65, color=color, label=f"{ct} (n={len(sub)})")
ax1.set_xlabel("Mean resultant length R")
ax1.set_ylabel("# units")
ax1.legend()
ax1.set_title("Phase-locking strength by cell type")

ax2 = fig.add_subplot(2, 3, 2)
for ct, color in [("excitatory", "steelblue"), ("inhibitory", "firebrick")]:
    sub = df[df.cell_type == ct]
    ax2.scatter(sub.n_spikes, sub.R, s=10, alpha=0.6, color=color, label=ct)
ax2.set_xscale("log")
ax2.set_xlabel("# spikes in run bouts")
ax2.set_ylabel("R")
ax2.legend(fontsize=8)
ax2.set_title("Locking strength vs spike count")

ax3 = fig.add_subplot(2, 3, 3)
ax3.hist(shuf_mean_R, bins=20, color="0.6", alpha=0.8, label="circular-shift null")
ax3.axvline(obs_mean_R, color="r", lw=2, label=f"observed = {obs_mean_R:.3f}")
ax3.set_xlabel("Population mean R")
ax3.set_ylabel("# shuffles")
ax3.legend(fontsize=8)
ax3.set_title("Shuffle control (population)")

pbins = np.linspace(-np.pi, np.pi, 25)
pc = 0.5 * (pbins[:-1] + pbins[1:])
ax4 = fig.add_subplot(2, 3, 4, projection="polar")
sig_exc = df[(df.cell_type == "excitatory") & df.significant]
counts, _ = np.histogram(sig_exc.pref_phase, bins=pbins)
ax4.bar(pc, counts, width=np.diff(pbins), color="steelblue", alpha=0.85)
ax4.set_title(f"Preferred phase, excitatory (n={len(sig_exc)} sig.)", pad=24)

ax5 = fig.add_subplot(2, 3, 5, projection="polar")
sig_inh = df[(df.cell_type == "inhibitory") & df.significant]
counts, _ = np.histogram(sig_inh.pref_phase, bins=pbins)
ax5.bar(pc, counts, width=np.diff(pbins), color="firebrick", alpha=0.85)
ax5.set_title(f"Preferred phase, inhibitory (n={len(sig_inh)} sig.)", pad=24)

ax6 = fig.add_subplot(2, 3, 6)
fracs = [df[df.cell_type == ct].significant.mean() for ct in ["excitatory", "inhibitory"]]
fracs_sh = [(df[df.cell_type == ct].p_shuffle < 0.05).mean() for ct in ["excitatory", "inhibitory"]]
x = np.arange(2)
ax6.bar(x - 0.2, fracs, width=0.4, color=["steelblue", "firebrick"], alpha=0.85,
        label="Rayleigh p<0.05")
ax6.bar(x + 0.2, fracs_sh, width=0.4, color=["steelblue", "firebrick"], alpha=0.45,
        label="shuffle p<0.05")
ax6.set_xticks(x)
ax6.set_xticklabels(["excitatory", "inhibitory"])
ax6.set_ylabel("Fraction of units")
ax6.set_ylim(0, 1.05)
ax6.legend(fontsize=8)
ax6.set_title("Fraction significantly phase-locked")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_population_summary.png", dpi=150)
plt.close()
print("saved fig4")

# %% [markdown]
# ## Population Phase-Preference Map
#
# Finally, we normalize each significantly locked unit's phase histogram by its mean
# rate, sort units by their preferred phase, and plot the result as a heatmap. A
# bright band sweeping across the sorted population shows that each unit tiles a
# characteristic part of the theta cycle.

# %%
sig_df = df[df.significant].sort_values("pref_phase")
nbins = 36
pb = np.linspace(-np.pi, np.pi, nbins + 1)
pcen = np.degrees(0.5 * (pb[:-1] + pb[1:]))
maps = []
for _, row in sig_df.iterrows():
    counts, _ = np.histogram(spk_phases[row.uid], bins=pb)
    rate = counts / max(counts.mean(), 1e-9)
    # light circular smoothing
    rate = np.convolve(np.r_[rate[-2:], rate, rate[:2]],
                       np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2) /
                       np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2).sum(),
                       mode="same")[2:-2]
    maps.append(rate)
maps = np.array(maps)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(maps, aspect="auto", extent=[-180, 180, len(maps), 0], cmap="viridis")
ax.set_xlabel("Theta phase (deg, 0 = LFP peak)")
ax.set_ylabel("Units (sorted by preferred phase)")
ax.set_title(f"Normalized theta-phase tuning, {len(maps)} significantly locked CA1 units")
plt.colorbar(im, label="Rate / mean rate")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_phase_preference_heatmap.png", dpi=150)
plt.close()
print("saved fig5")

# %% [markdown]
# ## Summary
#
# The printed statistics below quantify the entrainment: the large majority of CA1
# units, of both cell classes, fire at preferred phases of the local theta rhythm
# during running, with interneurons showing roughly twice the locking strength of
# pyramidal cells, and the population mean locking strength sits far outside the
# circular-shift null.

# %%
print("=== Theta phase entrainment summary ===")
print(f"Session: {ASSET_PATH} (dandiset {DANDISET})")
print(f"Reference channel {REF_CH}, maze epoch {MAZE_T0}-{MAZE_T1} s, "
      f"{len(run_ep)} run bouts ({run_ep.tot_length():.0f} s)")
print(f"Units analyzed: {len(df)} "
      f"({(df.cell_type == 'excitatory').sum()} excitatory, "
      f"{(df.cell_type == 'inhibitory').sum()} inhibitory)")
for ct in ["excitatory", "inhibitory"]:
    sub = df[df.cell_type == ct]
    print(f"  {ct}: median R = {sub.R.median():.3f}, "
          f"significant {sub.significant.sum()}/{len(sub)} (Rayleigh), "
          f"{(sub.p_shuffle < 0.05).sum()}/{len(sub)} (shuffle)")
print(f"Population mean R = {obs_mean_R:.3f} vs null {shuf_mean_R.mean():.3f}")
print("Figures written to", FIGDIR)
