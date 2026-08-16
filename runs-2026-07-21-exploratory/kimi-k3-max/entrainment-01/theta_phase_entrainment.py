# %% [markdown]
# # Theta Phase Entrainment of Hippocampal CA1 Neurons
#
# This notebook demonstrates theta phase entrainment (phase locking) of hippocampal
# neurons using data from the DANDI Archive. During exploration, the hippocampal local
# field potential (LFP) is dominated by the theta rhythm (5-11 Hz in rats), and the
# spikes of CA1 pyramidal cells and interneurons are entrained to preferred phases of
# this rhythm (O'Keefe & Recce 1993; Buzsaki 2002). We quantify this phenomenon with
# circular statistics applied to spike times relative to the theta-filtered LFP.
#
# **Dataset**: DANDI dandiset 000044, "Diversity in neural firing dynamics supports
# both rigid and learned hippocampal sequences" (Buzsaki lab). Session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: 128-channel LFP (1250 Hz)
# and 137 sorted CA1 units (120 excitatory, 17 inhibitory) recorded while a rat ran
# ~42 laps on a 1.6 m linear maze between rest epochs.
#
# **Approach**:
# 1. Stream the NWB file with remfile (no full download).
# 2. Select the LFP channel with the strongest theta (theta/delta power ratio) as the
#    phase reference.
# 3. Bandpass-filter the maze-epoch LFP to 5-11 Hz and take the Hilbert phase.
# 4. Restrict analysis to running bouts (speed > 10 cm/s), when theta is strongest.
# 5. For each unit, compute the mean resultant length (MRL), preferred phase, and
#    Rayleigh p-value of its spike phases, plus a time-shift control.

# %% [markdown]
# ## Setup

# %%
import os

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import signal, stats
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache_theta"

# %% [markdown]
# ## Streaming Access to the NWB File
#
# We resolve the S3 blob URL through the DANDI API (the `/download/` endpoint issues a
# 302 redirect to a presigned URL; the bare blob URL supports HTTP range requests) and
# open it with remfile plus a disk cache, so only the chunks we read are fetched.

# %%
api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api, params={"path": ASSET_PATH})
asset = r.json()["results"][0]
dl = api + asset["asset_id"] + "/download/"
location = requests.get(dl, allow_redirects=False).headers["Location"]
bare_url = location.split("?")[0]
print("streaming:", bare_url)

disk_cache = remfile.DiskCache(CACHE_DIR)
h5py_file = h5py.File(remfile.File(bare_url, disk_cache=disk_cache), "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The file contains a `units` TsGroup (137 CA1 units with `cell_type` and `location`
# metadata), an `epochs` IntervalSet (PRE / Maze / POST), a 128-channel `LFP`
# ElectricalSeries at 1250 Hz, and the 2D position on the maze at ~39 Hz.

# %%
units = nwb["units"]
print("units:", len(units))
print("cell types:", np.unique(units.get_info("cell_type"), return_counts=True))
print("locations:", np.unique(units.get_info("location"), return_counts=True))
print(nwb["epochs"])

lfp_es = h5py_file["processing"]["ecephys"]["LFP"]["LFP"]
lfp_data = lfp_es["data"]
fs = float(lfp_es["starting_time"].attrs["rate"])
conversion = float(lfp_es["data"].attrs["conversion"])
maze_start, maze_end = 18079.5, 20147.0
print(f"LFP: {lfp_data.shape} at {fs} Hz; maze epoch {maze_end - maze_start:.0f} s")

# %% [markdown]
# ## Reference Channel Selection
#
# Theta phase is conventionally measured from a channel in the CA1 pyramidal cell
# layer. We do not have layer annotations, so we select the channel with the highest
# theta (5-11 Hz) to delta (1-4 Hz) power ratio, computed from a 200 s window in the
# middle of the maze epoch. This is the standard data-driven way to find a
# pyramidal-layer reference.

# %%
i0, i1 = int(maze_start * fs), int(maze_end * fs)
mid = (i0 + i1) // 2
half = int(100 * fs)
chunk = lfp_data[mid - half:mid + half, :] * conversion  # (250000, 128) V

f, psd = signal.welch(chunk, fs=fs, nperseg=int(4 * fs), axis=0)
theta_band = (f >= 5) & (f <= 11)
delta_band = (f >= 1) & (f <= 4)
theta_pow = np.trapezoid(psd[theta_band], f[theta_band], axis=0)
delta_pow = np.trapezoid(psd[delta_band], f[delta_band], axis=0)
td_ratio = theta_pow / delta_pow
ref_ch = int(np.argmax(td_ratio))
theta_peak_f = f[theta_band][np.argmax(psd[:, ref_ch][theta_band])]
print(f"reference channel {ref_ch}: theta/delta = {td_ratio[ref_ch]:.2f}, "
      f"theta peak at {theta_peak_f:.2f} Hz")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.semilogy(f, psd[:, ref_ch], color="k", lw=1.2, label=f"ref ch {ref_ch}")
ax.semilogy(f, psd[:, np.argmin(td_ratio)], color="0.6", lw=1,
            label="lowest theta/delta ch")
ax.axvspan(5, 11, color="tab:blue", alpha=0.15, label="theta (5-11 Hz)")
ax.set_xlim(0, 60)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (V$^2$/Hz)")
ax.set_title("LFP power spectrum during maze epoch")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.plot(td_ratio, np.arange(len(td_ratio)), ".", color="tab:blue", ms=4)
ax.plot(td_ratio[ref_ch], ref_ch, "o", color="tab:red", ms=8,
        label=f"selected ch {ref_ch}")
ax.set_xlabel("Theta / delta power ratio")
ax.set_ylabel("LFP channel")
ax.set_title("Reference channel selection")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig1_psd_reference_channel.png", dpi=150)
plt.close(fig)

# %% [markdown]
# The reference channel shows a clear spectral peak at ~9 Hz, the expected frequency
# of rat hippocampal theta during running.

# %% [markdown]
# ## Theta Phase Extraction and Running Epochs
#
# We load the full maze epoch for the reference channel only (2 s of padding on each
# side to absorb filter edge effects), bandpass-filter to 5-11 Hz with a zero-phase
# FIR filter, and take the angle of the Hilbert transform as the instantaneous theta
# phase. Phase 0 corresponds to the peak of the LFP oscillation and +/- pi to the
# trough. Running speed is computed from the 2D position and used to define running
# bouts (smoothed speed > 10 cm/s, at least 0.5 s long, gaps < 0.25 s merged).

# %%
pad = int(2 * fs)
lfp = lfp_data[i0 - pad:i1 + pad, ref_ch] * conversion
t_lfp_full = (np.arange(i0 - pad, i1 + pad) / fs).astype(np.float64)

ntaps = int(np.ceil(3 * fs / 5))  # 3 cycles of the low cutoff
ntaps += 1 - ntaps % 2
b = signal.firwin(ntaps, [5, 11], pass_zero=False, fs=fs)
theta_full = signal.filtfilt(b, [1.0], lfp)
phase_full = np.angle(signal.hilbert(theta_full))

sl = slice(pad, len(lfp) - pad)
lfp_c = lfp[sl]
theta_c = theta_full[sl]
phase_c = phase_full[sl]
t_lfp = t_lfp_full[sl]

# speed from 2D position
pos = nwb["1.6mLinearMazeSpatialSeries"]
pos_maze = pos.get(maze_start, maze_end)
xy = pos_maze.values
t_pos = pos_maze.index.values
dt_pos = np.median(np.diff(t_pos))
vx = np.gradient(xy[:, 0], t_pos)
vy = np.gradient(xy[:, 1], t_pos)
speed = np.sqrt(vx**2 + vy**2)
speed[np.isnan(xy[:, 0])] = np.nan
sig_s = 0.5 / dt_pos
k = signal.windows.gaussian(int(6 * sig_s) | 1, sig_s)
k /= k.sum()
speed_s = np.convolve(np.nan_to_num(speed), k, mode="same")
speed_s[np.isnan(speed)] = np.nan
print(f"position at {1/dt_pos:.1f} Hz; median speed {np.nanmedian(speed_s)*100:.1f} cm/s")

# %%
# epoch overview: speed and theta amplitude over the whole maze epoch
theta_amp = np.abs(signal.hilbert(theta_c))
ds = int(fs / 10)

fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax = axes[0]
ax.plot(t_pos, speed_s * 100, color="tab:green", lw=0.6)
ax.axhline(10, color="k", ls="--", lw=0.8, label="10 cm/s run threshold")
ax.set_ylabel("Speed (cm/s)")
ax.set_ylim(0, 120)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Maze epoch overview: running speed and theta amplitude")
ax = axes[1]
ax.plot(t_lfp[::ds], theta_amp[::ds] * 1e6, color="tab:blue", lw=0.5)
ax.set_ylabel("Theta amplitude (µV)")
ax.set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("figures/fig2_epoch_overview.png", dpi=150)
plt.close(fig)

# %% [markdown]
# The speed trace shows the ~42 track runs (fast transients up to ~80 cm/s separated
# by pauses at the reward platforms), and theta amplitude is sustained throughout the
# maze epoch. The next figure validates the phase extraction on a 3 s snippet.

# %%
t0 = t_lfp[0] + 500.0
win = (t_lfp >= t0) & (t_lfp < t0 + 3.0)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1.2]})
ax = axes[0]
ax.plot(t_lfp[win], lfp_c[win] * 1e6, color="0.5", lw=0.8, label="raw LFP")
ax.plot(t_lfp[win], theta_c[win] * 1e6, color="tab:blue", lw=1.4,
        label="theta (5-11 Hz)")
ax.set_ylabel("LFP (µV)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title(f"Theta phase extraction, reference channel {ref_ch}")

ax = axes[1]
ax.plot(t_lfp[win], phase_c[win], color="tab:purple", lw=1.0)
ax.set_ylabel("Theta phase (rad)")
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels(["-π", "0", "π"])
ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
for j, uk in enumerate(list(units.keys())[:6]):
    spk_t = units[uk].get(t0, t0 + 3.0).index.values
    ax.plot(spk_t, np.full_like(spk_t, np.pi + 0.15 + 0.05 * j), "|",
            color="tab:red", ms=6)

ax = axes[2]
ax.plot(t_pos, speed_s * 100, color="tab:green", lw=1.0)
ax.axhline(10, color="k", ls="--", lw=0.8, label="10 cm/s run threshold")
ax.set_ylabel("Speed (cm/s)")
ax.set_xlabel("Time (s)")
ax.set_xlim(t0, t0 + 3.0)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig3_theta_phase_validation.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Spike-Phase Locking Analysis
#
# For each unit we collect spikes fired during running bouts, interpolate the theta
# phase at each spike time (interpolating sin and cos separately to avoid wrap
# artifacts), and compute:
#
# - **MRL** (mean resultant length): 0 = uniform over phases, 1 = perfectly locked.
# - **Preferred phase**: angle of the mean resultant vector.
# - **Rayleigh p-value**: non-uniformity test (Fisher 1993 approximation).
# - **Time-shift control**: MRL recomputed after rolling the phase signal by a random
#   offset, which destroys spike-phase correspondence while preserving all marginal
#   statistics.
#
# Units with fewer than 100 spikes in running bouts are excluded.

# %%
# running bouts as a pynapple IntervalSet
run = (speed_s > 0.10) & ~np.isnan(speed_s)
edges = np.diff(run.astype(int))
starts = t_pos[:-1][edges == 1]
ends = t_pos[:-1][edges == -1]
if run[0]:
    starts = np.r_[t_pos[0], starts]
if run[-1]:
    ends = np.r_[ends, t_pos[-1]]

def merge_intervals(starts, ends, min_gap):
    if len(starts) == 0:
        return starts, ends
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - me[-1] < min_gap:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    return np.array(ms), np.array(me)

starts, ends = merge_intervals(starts, ends, 0.25)
dur = ends - starts
starts, ends = starts[dur >= 0.5], ends[dur >= 0.5]
run_ep = nap.IntervalSet(start=starts, end=ends)
maze_ep = nap.IntervalSet(start=[maze_start], end=[maze_end])
analysis_ep = maze_ep.intersect(run_ep)
print(f"run bouts: {len(starts)}, total {analysis_ep.tot_length('s'):.0f} s")

cos_p, sin_p = np.cos(phase_c), np.sin(phase_c)

def phase_at(times):
    return np.arctan2(np.interp(times, t_lfp, sin_p), np.interp(times, t_lfp, cos_p))

def rayleigh_p(n, R):
    """Fisher (1993) approximation of the Rayleigh test p-value."""
    z = n * R**2
    p = np.exp(-z)
    if n < 50:
        p *= (1 + (2 * z - z**2) / (4 * n)
              - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2))
    return min(p, 1.0)

# %%
records = []
keys = list(units.keys())
cell_types = np.asarray(units.get_info("cell_type"))
locations = np.asarray(units.get_info("location"))
rng = np.random.default_rng(42)
min_shift = int(fs)

for i, uk in enumerate(tqdm(keys, desc="units")):
    t_spk = units[uk].restrict(analysis_ep).index.values
    n = len(t_spk)
    if n < 100:
        continue
    ph = phase_at(t_spk)
    R = np.abs(np.mean(np.exp(1j * ph)))
    pref = np.angle(np.mean(np.exp(1j * ph)))
    p = rayleigh_p(n, R)
    shift = rng.integers(min_shift, len(phase_c) - min_shift)
    ph_s = np.arctan2(np.interp(t_spk, t_lfp, np.roll(sin_p, shift)),
                      np.interp(t_spk, t_lfp, np.roll(cos_p, shift)))
    R_s = np.abs(np.mean(np.exp(1j * ph_s)))
    records.append(dict(unit=uk, cell_type=cell_types[i], location=locations[i],
                        n_spikes=n, rate=n / analysis_ep.tot_length("s"),
                        mrl=R, pref_phase=pref, rayleigh_p=p, mrl_shuffle=R_s))

df = pd.DataFrame(records)
df.to_csv("results/phase_locking_per_unit.csv", index=False)
sig = df["rayleigh_p"] < 0.01

print(f"\nunits analyzed (>=100 spikes while running): {len(df)}")
print(f"significantly phase-locked (Rayleigh p<0.01): {sig.sum()} ({100*sig.mean():.1f}%)")
for ct in ["excitatory", "inhibitory"]:
    m = df["cell_type"] == ct
    print(f"  {ct}: {sig[m].sum()}/{m.sum()} ({100*sig[m].mean():.1f}%), "
          f"median MRL {df.loc[m, 'mrl'].median():.3f}")
mw = stats.mannwhitneyu(df.loc[df.cell_type == "excitatory", "mrl"],
                        df.loc[df.cell_type == "inhibitory", "mrl"])
print(f"MRL exc vs inh Mann-Whitney p = {mw.pvalue:.2e}")
mw2 = stats.mannwhitneyu(df["mrl"], df["mrl_shuffle"])
print(f"median MRL observed {df['mrl'].median():.3f} vs time-shift control "
      f"{df['mrl_shuffle'].median():.3f} (Mann-Whitney p = {mw2.pvalue:.2e})")

# %% [markdown]
# ## Example Units
#
# Polar histograms of spike phase for the four most strongly locked excitatory units
# and two locked inhibitory units. Each unit concentrates its spikes on a narrow range
# of theta phases.

# %%
nbins = 18
bins = np.linspace(-np.pi, np.pi, nbins + 1)
centers = (bins[:-1] + bins[1:]) / 2

sig_exc = df[(df.cell_type == "excitatory") & sig].sort_values("mrl", ascending=False)
sig_inh = df[(df.cell_type == "inhibitory") & sig].sort_values("mrl", ascending=False)
examples = list(sig_exc["unit"].iloc[:4]) + list(sig_inh["unit"].iloc[:2])
labels = [f"unit {u} (exc)" for u in sig_exc["unit"].iloc[:4]] + \
         [f"unit {u} (inh)" for u in sig_inh["unit"].iloc[:2]]

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), subplot_kw={"projection": "polar"})
for ax, u, lab in zip(axes.flat, examples, labels):
    ph = phase_at(units[u].restrict(analysis_ep).index.values)
    counts, _ = np.histogram(ph, bins=bins)
    counts = counts / counts.sum()
    ax.bar(centers, counts, width=2 * np.pi / nbins, color="tab:blue",
           edgecolor="k", lw=0.5, alpha=0.8)
    row = df[df.unit == u].iloc[0]
    ax.set_title(lab, fontsize=10, pad=10)
    ax.text(0.5, -0.13, f"MRL={row.mrl:.2f}, p={row.rayleigh_p:.0e}, n={row.n_spikes}",
            transform=ax.transAxes, ha="center", fontsize=8.5)
    ax.set_theta_zero_location("E")
    ax.set_yticks([])
    ax.set_xticklabels([])
fig.suptitle("Spike phase relative to CA1 theta (0° = LFP peak at right, "
             "180° = trough at left)", y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.94], h_pad=2.5)
fig.savefig("figures/fig4_example_phase_histograms.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Population Summary
#
# Three views of the population: the distribution of phase-locking strength by cell
# type against the time-shift control, the fraction of significantly locked units, and
# the distribution of preferred phases among locked units.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

ax = axes[0]
be = np.linspace(0, 1, 41)
ax.hist(df.loc[df.cell_type == "excitatory", "mrl"], bins=be, alpha=0.7,
        color="tab:blue", label=f"excitatory (n={sum(df.cell_type == 'excitatory')})")
ax.hist(df.loc[df.cell_type == "inhibitory", "mrl"], bins=be, alpha=0.7,
        color="tab:red", label=f"inhibitory (n={sum(df.cell_type == 'inhibitory')})")
ax.hist(df["mrl_shuffle"], bins=be, histtype="step", color="k", lw=1.2,
        label="time-shift control")
ax.set_xlabel("Mean resultant length (phase locking)")
ax.set_ylabel("Units")
ax.set_title(f"Phase-locking strength\nMann-Whitney p={mw.pvalue:.1e}")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
fracs = [100 * sig[df.cell_type == ct].mean() for ct in ["excitatory", "inhibitory"]]
bars = ax.bar(["excitatory", "inhibitory"], fracs, color=["tab:blue", "tab:red"],
              alpha=0.8)
for b_, ct in zip(bars, ["excitatory", "inhibitory"]):
    m = df.cell_type == ct
    ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 1,
            f"{sig[m].sum()}/{m.sum()}", ha="center", fontsize=9)
ax.set_ylabel("% significantly locked")
ax.set_title("Fraction phase-locked\n(Rayleigh p<0.01, >=100 spikes)")
ax.set_ylim(0, 105)

ax = axes[2]
for ct, c in [("excitatory", "tab:blue"), ("inhibitory", "tab:red")]:
    m = (df.cell_type == ct) & sig
    ax.hist(np.degrees(df.loc[m, "pref_phase"]), bins=np.linspace(-180, 180, 19),
            alpha=0.7, color=c, label=f"{ct} (n={m.sum()})")
ax.axvline(180, color="k", ls=":", lw=0.8)
ax.axvline(-180, color="k", ls=":", lw=0.8)
ax.set_xlabel("Preferred theta phase (deg; 0=peak, ±180=trough)")
ax.set_ylabel("Units")
ax.set_title("Preferred phase of locked units")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig5_population_summary.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Pooled Pyramidal-Cell Phase Preference
#
# Pooling the spikes of all significantly locked pyramidal cells shows the population
# preference for the descending phase approaching the trough of the reference theta,
# the classic signature of CA1 pyramidal-cell entrainment.

# %%
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), subplot_kw={"projection": "polar"})
all_ph = []
for u in df.loc[(df.cell_type == "excitatory") & sig, "unit"]:
    all_ph.append(phase_at(units[u].restrict(analysis_ep).index.values))
all_ph = np.concatenate(all_ph)
counts, _ = np.histogram(all_ph, bins=bins)
counts = counts / counts.sum()
ax.bar(centers, counts, width=2 * np.pi / nbins, color="tab:blue", edgecolor="k",
       lw=0.5, alpha=0.8)
ax.set_theta_zero_location("E")
ax.set_yticks([])
ax.set_xticklabels([])
R_pool = np.abs(np.mean(np.exp(1j * all_ph)))
pref_pool = np.angle(np.mean(np.exp(1j * all_ph)))
n_locked_exc = int(sig[df.cell_type == "excitatory"].sum())
ax.set_title(f"Pooled phases, {n_locked_exc} locked pyramidal cells\n"
             f"{len(all_ph):,} spikes, MRL={R_pool:.3f}, pref={np.degrees(pref_pool):.0f}°\n"
             "(0° = LFP peak at right, 180° = trough at left)",
             fontsize=10, pad=16)
fig.tight_layout()
fig.savefig("figures/fig6_pooled_phase.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Cycle-by-Cycle View
#
# Finally, the most direct view of entrainment: the spikes of one well-locked unit
# plotted on the theta-filtered LFP over 1.5 s. The spikes land at nearly the same
# phase of successive theta cycles.

# %%
cand = df[(df.cell_type == "excitatory") & sig & (df.n_spikes > 1000)].sort_values(
    "mrl", ascending=False)
best_u = int(cand["unit"].iloc[0])
spk_all = units[best_u].restrict(analysis_ep).index.values
edges_w = np.arange(spk_all[0], spk_all[-1], 0.25)
idx = np.searchsorted(spk_all, edges_w)
idx2 = np.searchsorted(spk_all, edges_w + 1.5)
t0 = edges_w[np.argmax(idx2 - idx)]  # densest 1.5 s window

win = (t_lfp >= t0) & (t_lfp < t0 + 1.5)
spk_t = units[best_u].get(t0, t0 + 1.5).index.values
spk_amp = np.interp(spk_t, t_lfp, theta_c)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_lfp[win], theta_c[win] * 1e6, color="tab:blue", lw=1.4,
        label=f"theta-filtered LFP (ch {ref_ch})")
ax.plot(spk_t, spk_amp * 1e6, "o", color="tab:red", ms=9, mfc="none", mew=1.8,
        label=f"unit {best_u} spikes (n={len(spk_t)})")
ax.set_ylabel("LFP (µV)")
ax.set_xlabel("Time (s)")
row = df[df.unit == best_u].iloc[0]
ax.set_title(f"Cycle-by-cycle entrainment: unit {best_u}, MRL={row.mrl:.2f}, "
             f"preferred phase {np.degrees(row.pref_phase):.0f}°")
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig7_spikes_on_theta.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Results
#
# In this session, 103 of 136 CA1 units with sufficient running-epoch spiking
# (75.7%) were significantly phase-locked to theta (Rayleigh p<0.01). Phase locking
# was near-universal and stronger among inhibitory interneurons (16/17 locked, median
# MRL 0.249) than among excitatory pyramidal cells (87/119 locked, median MRL 0.108;
# Mann-Whitney p = 1.5e-4), and the observed MRL distribution was far above the
# time-shift control (median 0.116 vs 0.025, p ~ 1e-33). Locked pyramidal cells
# preferred the descending phase approaching the trough of the reference-channel
# theta (pooled preferred phase about -142 deg, where 0 is the LFP peak), consistent
# with the classic description of CA1 pyramidal-cell entrainment during running
# (O'Keefe & Recce 1993). Together with the cycle-by-cycle view, these results
# demonstrate robust theta phase entrainment of hippocampal neurons in this dataset.
