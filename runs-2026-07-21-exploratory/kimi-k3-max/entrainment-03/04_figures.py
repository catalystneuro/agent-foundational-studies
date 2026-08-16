"""Figures for the theta phase entrainment analysis."""
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
import pickle
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = pd.read_csv("unit_phase_locking.csv")
an = np.load("theta_analysis.npz")
bouts, spd_t, spd_s = an["bouts"], an["spd_t"], an["spd_s"]
maze_start, maze_end = float(an["maze_start"]), float(an["maze_end"])
best_ch, fs = int(an["best_ch"]), float(an["fs"])
with open("spike_phases.pkl", "rb") as f:
    spike_phases = pickle.load(f)

s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment03")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)
es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
conv = es.conversion
units = nwb["units"]

sig_mask = res["rayleigh_p"] < 0.01
exc = res["cell_type"] == "excitatory"
inh = res["cell_type"] == "inhibitory"

# ---------- Fig 1: raw LFP, theta-filtered, phase, spike raster ----------
# pick a running bout and show 3 s in its middle
bout_lens = bouts[:, 1] - bouts[:, 0]
long_bout = bouts[np.argmax(bout_lens)]
t_mid = long_bout.mean()
w0, w1 = t_mid - 6, t_mid + 6  # load 12 s, show middle 3 s
i0, i1 = int(w0 * fs), int(w1 * fs)
seg = es.data[i0:i1, best_ch].astype(np.float64) * conv
t_seg = np.arange(i0, i1) / fs
b, a = signal.butter(4, [4, 12], btype="bandpass", fs=fs)
seg_filt = signal.filtfilt(b, a, seg)
seg_phase = np.angle(signal.hilbert(seg_filt))

show = (t_seg >= t_mid - 1.5) & (t_seg <= t_mid + 1.5)
t_show = t_seg[show] - t_seg[show][0]  # relative time within the 3 s window

# top-8 significantly locked units by MRL for the raster
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

t_abs0 = t_seg[show][0]
for i, uid in enumerate(top_units):
    spk = units[uid].t
    spk = spk[(spk >= t_abs0) & (spk <= t_abs0 + 3.0)] - t_abs0
    axes[2].vlines(spk, i + 0.1, i + 0.9, color="k", lw=0.8)
axes[2].set_yticks(np.arange(8) + 0.5)
axes[2].set_yticklabels([f"u{uid} ({res[res.unit==uid]['cell_type'].iloc[0][:3]})"
                         for uid in top_units], fontsize=8)
axes[2].set_ylabel("units (top MRL)")
axes[2].set_xlabel("Time within 3 s window (s)")
axes[2].set_ylim(0, 8.2)
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig("fig_raw_lfp_theta.png", dpi=150)
plt.close(fig)
print("saved fig_raw_lfp_theta.png")

# ---------- Fig 2: speed and running bouts ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
# full maze epoch, downsample for display
ds = np.arange(0, len(spd_t), 25)
axes[0].plot(spd_t[ds], spd_s[ds] * 100, color="k", lw=0.6)
axes[0].axhline(10, color="crimson", ls="--", lw=1, label="10 cm/s threshold")
for s, e in bouts[:: max(1, len(bouts)//400)]:
    axes[0].axvspan(s, e, color="tab:blue", alpha=0.06, lw=0)
axes[0].set_ylabel("speed (cm/s)")
axes[0].set_xlabel("Time (s)")
axes[0].set_title("Running speed over the full maze epoch (blue shading: running bouts used)")
axes[0].legend(loc="upper right")
axes[0].set_xlim(maze_start, maze_end)
# zoom into 120 s
z0 = long_bout[0] - 20
z1 = z0 + 120
zm = (spd_t >= z0) & (spd_t <= z1)
axes[1].plot(spd_t[zm], spd_s[zm] * 100, color="k", lw=1.0)
axes[1].axhline(10, color="crimson", ls="--", lw=1)
zb = bouts[(bouts[:, 1] > z0) & (bouts[:, 0] < z1)]
for s, e in zb:
    axes[1].axvspan(max(s, z0), min(e, z1), color="tab:blue", alpha=0.15, lw=0)
axes[1].set_ylabel("speed (cm/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Zoom: 120 s around the longest running bout")
axes[1].set_xlim(z0, z1)
fig.tight_layout()
fig.savefig("fig_speed_running_bouts.png", dpi=150)
plt.close(fig)
print("saved fig_speed_running_bouts.png")

# ---------- Fig 3: example phase histograms (polar) ----------
examples = []
for ct, df in [("excitatory", res[exc & sig_mask]), ("inhibitory", res[inh & sig_mask])]:
    examples += df.nlargest(2, "mrl")["unit"].tolist()

fig, axes = plt.subplots(1, 4, subplot_kw=dict(projection="polar"), figsize=(13, 4.6))
for ax, uid in zip(axes, examples):
    ph = spike_phases[uid]
    row = res[res.unit == uid].iloc[0]
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(ph, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges)
    heights = counts / counts.sum()
    ax.bar(centers, heights, width=width, color="tab:blue", alpha=0.75,
           edgecolor="k", lw=0.4)
    # mean vector, scaled to the tallest bar
    ax.plot([row["pref_phase"], row["pref_phase"]], [0, heights.max()], color="crimson", lw=2)
    ax.set_theta_zero_location("N")
    ax.set_title(f"unit {uid} ({row['cell_type'][:3]})\nMRL={row['mrl']:.2f}, "
                 f"p={row['rayleigh_p']:.0e}, n={int(row['n_spikes'])}", fontsize=9, pad=26)
    ax.set_rticks([])
fig.suptitle("Spike-phase histograms of strongly locked units (0 = theta LFP peak)", y=1.04)
fig.tight_layout()
fig.savefig("fig_example_phase_histograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig_example_phase_histograms.png")

# ---------- Fig 4: MRL distributions + stats ----------
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
# jitter points
for i, d in enumerate(data):
    axes[1].scatter(np.full(len(d), i + 1) + np.random.default_rng(0).normal(0, 0.05, len(d)),
                    d, s=6, color="k", alpha=0.35, zorder=3)
u_stat, mw_p = stats.mannwhitneyu(res.loc[exc, "mrl"], res.loc[inh, "mrl"])
axes[1].set_ylabel("MRL")
axes[1].set_title(f"exc vs inh Mann-Whitney p = {mw_p:.1e}")
fig.tight_layout()
fig.savefig("fig_mrl_distributions.png", dpi=150)
plt.close(fig)
print("saved fig_mrl_distributions.png; MW p =", mw_p)

# ---------- Fig 5: preferred phase distribution ----------
fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection="polar"), figsize=(10.5, 4.8))
for ax, (ct, m, color) in zip(axes, [("excitatory", exc, "tab:blue"), ("inhibitory", inh, "tab:red")]):
    ph = res.loc[m & sig_mask, "pref_phase"].values
    bins = np.linspace(-np.pi, np.pi, 19)
    counts, edges = np.histogram(ph, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=np.diff(edges), color=color, alpha=0.75, edgecolor="k", lw=0.4)
    pooled = np.angle(np.mean(np.exp(1j * ph)))
    pooled_r = np.abs(np.mean(np.exp(1j * ph)))
    ax.plot([pooled, pooled], [0, counts.max()], color="k", lw=2, ls="--",
            label=f"pooled mean {np.degrees(pooled):.0f}°")
    ax.set_theta_zero_location("N")
    ax.set_title(f"{ct} (n={len(ph)} locked)\npooled preferred phase {np.degrees(pooled):.0f}°",
                 fontsize=10, pad=26)
    ax.set_rticks([])
fig.suptitle("Preferred theta phase of significantly locked units (0 = LFP peak)", y=1.04)
fig.tight_layout()
fig.savefig("fig_preferred_phase_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig_preferred_phase_distribution.png")

# ---------- Fig 6: MRL vs firing rate ----------
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
print("saved fig_mrl_vs_rate.png")

# summary stats to stdout for README
print("\n=== SUMMARY ===")
print(f"units analyzed (>=50 spikes in running bouts): {len(res)}")
print(f"significantly locked (Rayleigh p<0.01): {sig_mask.sum()} ({sig_mask.mean()*100:.0f}%)")
for ct, m in [("excitatory", exc), ("inhibitory", inh)]:
    sub = res[m]
    s = sub["rayleigh_p"] < 0.01
    print(f"{ct}: {s.sum()}/{len(sub)} locked; median MRL {sub['mrl'].median():.3f}; "
          f"control {sub['mrl_control'].median():.3f}; "
          f"median pref phase {np.degrees(np.angle(np.mean(np.exp(1j*sub.loc[s,'pref_phase'])))):.0f} deg")
print(f"exc vs inh MRL Mann-Whitney p = {mw_p:.2e}")
print(f"running bouts: {len(bouts)}, total {bout_lens.sum():.0f} s")
