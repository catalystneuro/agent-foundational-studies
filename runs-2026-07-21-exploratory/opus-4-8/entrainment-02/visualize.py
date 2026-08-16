"""
Visualization of theta phase entrainment in CA1 (DANDI:000044 sub-Buddy).
Consumes the cached arrays produced by prep_data.py + analysis.py and writes
all figures to figures/*.png. Uses pynapple only where it clarifies the code;
the heavy lifting is done on cached numpy arrays.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

THETA_LO, THETA_HI = 6.0, 10.0

# ---------------------------------------------------------------- load caches
A = np.load("cache/analysis_arrays.npz")
tvec = A["tvec"]; lfp = A["lfp"]; theta = A["theta"]; phase = A["phase"]
amp = A["amp"]; pos_t = A["pos_t"]; speed = A["speed"]; pos_cm = A["pos_cm"]
run_start = A["run_start"]; run_end = A["run_end"]; best_ch = int(A["best_ch"])
rate = 1.0 / np.median(np.diff(tvec))

res = pd.read_csv("cache/phase_locking_results.csv").set_index("unit")
sp = np.load("cache/spike_phases.npz")            # per-unit spike phases (running)
spike_phase_by_unit = {int(k): sp[k] for k in sp.files}

prep = np.load("cache/buddy_prep.npz", allow_pickle=True)
cand_channels = prep["cand_channels"]
cand_ratio = prep["cand_theta_ratio"]
spike_times = prep["spike_times"]
locations = prep["locations"]

print(f"LFP {len(lfp)} samp @ {rate:.1f} Hz, best_ch={best_ch}, "
      f"{len(res)} CA1 units analysed")

# ================================================================ FIG 1
# Channel selection: theta/broadband power ratio across shanks
fig, ax = plt.subplots(figsize=(7, 3.2))
order = np.argsort(cand_channels)
ax.bar(np.arange(len(cand_channels)), cand_ratio[order], color="#4477AA")
best_pos = int(np.where(cand_channels[order] == best_ch)[0][0])
ax.bar(best_pos, cand_ratio[order][best_pos], color="#EE6677",
       label=f"selected ch {best_ch}")
ax.set_xticks(np.arange(len(cand_channels)))
ax.set_xticklabels([f"ch{c}" for c in cand_channels[order]], rotation=45, fontsize=8)
ax.set_ylabel("theta (6-10 Hz) / broadband\npower ratio")
ax.set_title("Theta-channel selection across shanks (MazeEpoch LFP)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig1_channel_selection.png")
plt.close(fig)

# ================================================================ FIG 2
# Raw LFP + theta-filtered trace + phase, with CA1 spikes overlaid.
# Pick a 3 s running window with high theta amplitude.
run_mask = np.zeros(len(tvec), bool)
for s, e in zip(run_start, run_end):
    run_mask |= (tvec >= s) & (tvec <= e)
# find a 3 s window centred in the longest run with strong theta
seg_i0 = np.argmax([e - s for s, e in zip(run_start, run_end)])
w0 = run_start[seg_i0] + 1.0
w1 = w0 + 3.0
sel = (tvec >= w0) & (tvec <= w1)
tt = tvec[sel]

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1.4]})
axes[0].plot(tt, lfp[sel], color="0.4", lw=0.7, label="raw LFP")
axes[0].plot(tt, theta[sel], color="#EE6677", lw=1.4, label="theta 6-10 Hz")
axes[0].set_ylabel("µV")
axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
axes[0].set_title(f"CA1 LFP, theta band, and spike phase (ch {best_ch}, running)")

axes[1].plot(tt, theta[sel], color="#EE6677", lw=1.4)
axes[1].fill_between(tt, -amp[sel], amp[sel], color="#EE6677", alpha=0.15,
                     label="theta envelope")
axes[1].set_ylabel("µV")
axes[1].legend(frameon=False, loc="upper right", fontsize=8)

# raster of strongly-locked CA1 units in this window
strong = res.sort_values("mrl", ascending=False).head(12).index.tolist()
for row, u in enumerate(strong):
    st = np.asarray(spike_times[u], dtype=float)
    st = st[(st >= w0) & (st <= w1)]
    axes[2].plot(st, np.full_like(st, row), "|", color="#222222", ms=6, mew=1.0)
axes[2].plot(tt, (phase[sel] + np.pi) / (2*np.pi) * (len(strong)-1),
             color="#4477AA", lw=0.8, alpha=0.6, label="theta phase (0-2π)")
axes[2].set_ylabel("unit")
axes[2].set_xlabel("time (s)")
axes[2].set_yticks([0, len(strong)-1]); axes[2].set_yticklabels(["", ""])
axes[2].legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig2_lfp_theta_raster.png")
plt.close(fig)

# ================================================================ FIG 3
# Example single-unit phase histograms (polar) for 6 strongly-locked units.
examples = res.sort_values("mrl", ascending=False).head(6).index.tolist()
nb = 18
edges = np.linspace(-np.pi, np.pi, nb + 1)
centers = (edges[:-1] + edges[1:]) / 2
fig, axes = plt.subplots(2, 3, figsize=(10, 7.5),
                         subplot_kw={"projection": "polar"},
                         constrained_layout=True)
width = 2*np.pi/nb
for ax, u in zip(axes.ravel(), examples):
    ph = spike_phase_by_unit[u]
    h, _ = np.histogram(ph, bins=edges)
    h = h / h.sum()
    ct = res.loc[u, "cell_type"]
    color = "#EE6677" if ct == "inhibitory" else "#4477AA"
    ax.bar(centers, h, width=width, bottom=0.0, color=color, alpha=0.85,
           edgecolor="k", lw=0.3)
    mp = res.loc[u, "mean_phase"]; mrl = res.loc[u, "mrl"]
    ax.plot([mp, mp], [0, h.max()], color="k", lw=2.2)
    ax.set_rticks([])
    ax.set_thetagrids([0, 90, 180, 270],
                      labels=["0", "", "±π", "-π/2"], fontsize=8)
    ax.set_title(f"unit {u} ({ct[:3]})  MRL={mrl:.2f}", fontsize=9, pad=8)
fig.suptitle("Spike-phase distributions for strongly theta-locked CA1 units "
             "(black line = mean phase; 0 = LFP peak)", fontsize=11)
fig.savefig("figures/fig3_example_polar.png")
plt.close(fig)

# ================================================================ FIG 4
# Population summary: MRL distribution, significance, mean-phase distribution.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# (a) MRL histogram, exc vs inh
exc = res[res["cell_type"] == "excitatory"]["mrl"]
inh = res[res["cell_type"] == "inhibitory"]["mrl"]
bins = np.linspace(0, res["mrl"].max()*1.05, 16)
axes[0].hist(exc, bins=bins, color="#4477AA", alpha=0.75, label=f"exc (n={len(exc)})")
axes[0].hist(inh, bins=bins, color="#EE6677", alpha=0.75, label=f"inh (n={len(inh)})")
axes[0].set_xlabel("mean resultant length (MRL)")
axes[0].set_ylabel("# units")
axes[0].set_title("Phase-locking strength")
axes[0].legend(frameon=False, fontsize=8)

# (b) significance summary
sig = int(res["sig"].sum()); tot = len(res)
axes[1].bar(["locked\n(p<0.05)", "not"], [sig, tot - sig],
            color=["#228833", "0.7"])
for i, v in enumerate([sig, tot - sig]):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)
axes[1].set_ylabel("# CA1 units")
axes[1].set_title(f"Rayleigh test: {100*sig/tot:.0f}% significantly locked")

# (c) preferred-phase distribution (significant units) on polar-ish linear axis
mp_sig = res[res["sig"]]["mean_phase"].values
axes[2].hist(mp_sig, bins=np.linspace(-np.pi, np.pi, 19), color="#AA3377", alpha=0.8)
axes[2].set_xlabel("preferred theta phase (rad)")
axes[2].set_ylabel("# units")
axes[2].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[2].set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
axes[2].set_title("Preferred phases (locked units)")
fig.tight_layout()
fig.savefig("figures/fig4_population_summary.png")
plt.close(fig)

# ================================================================ FIG 5
# Pooled spike-phase histogram across all significant units (two cycles).
all_ph = np.concatenate([spike_phase_by_unit[u] for u in res[res["sig"]].index
                         if u in spike_phase_by_unit])
all_ph_deg = np.degrees(all_ph) % 360.0        # wrap to [0, 360)
fig, ax = plt.subplots(figsize=(7, 3.4))
nb2 = 36
h, edges2 = np.histogram(all_ph_deg, bins=np.linspace(0, 360, nb2+1),
                         density=True)
c2 = (edges2[:-1] + edges2[1:]) / 2            # bin centers in deg, 0..360
# plot two cycles for readability
x2 = np.concatenate([c2, c2 + 360])
h2 = np.concatenate([h, h])
ax.bar(x2, h2, width=360/nb2*0.95, color="#4477AA", alpha=0.8)
# overlay idealized theta cycle (0/360 deg = LFP peak)
xx = np.linspace(0, 720, 400)
ax.plot(xx, h.mean() + h.std()*np.cos(np.radians(xx)),
        color="#EE6677", lw=1.5, label="theta LFP (schematic)")
ax.set_xlabel("theta phase (deg, two cycles)")
ax.set_ylabel("spike density")
ax.set_title(f"Pooled CA1 spike-phase histogram "
             f"({len(all_ph):,} spikes, {res['sig'].sum()} locked units)")
ax.legend(frameon=False, fontsize=8)
ax.set_xticks(np.arange(0, 721, 180))
fig.tight_layout()
fig.savefig("figures/fig5_pooled_spike_phase.png")
plt.close(fig)

print("Wrote 5 figures to figures/")
