"""Stage 4: figures for the single-session theta phase entrainment result."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.stats import wilcoxon

import theta_utils as tu

SESSION = "Achilles-10252013"
BEST_CH = 117
MIN_SPIKES = 100
NBINS = 18

h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)
units = meta["units"]
maze = meta["epochs"]["MazeEpoch"]
res = pd.read_csv("phase_locking_single_session.csv")

pos = meta["position"].restrict(maze)
speed = tu.compute_speed(pos, rate=meta["position_rate"])
run = speed.threshold(10.0).time_support.drop_short_intervals(1.0)
post = meta["epochs"]["POSTEpoch"]
state_epochs = {
    "RUN": run,
    "REM": meta["states"]["REM"].intersect(post).drop_short_intervals(20.0),
    "nonREM": meta["states"]["Non-REM"].intersect(post).drop_short_intervals(20.0),
}
phases, amps, raws, epochs = {}, {}, {}, {}
for name, ep in state_epochs.items():
    phases[name], amps[name], raws[name], epochs[name] = tu.phase_by_interval(
        h, BEST_CH, ep, max_total=1200.0)
lookups = {k: tu.PhaseLookup(v) for k, v in phases.items()}

incl = res["n_RUN"] >= MIN_SPIKES
locked = res["sig_RUN"].values & incl.values

# phase tuning curves (occupancy-normalised firing rate vs theta phase)
tc = nap.compute_1d_tuning_curves(units, phases["RUN"], nb_bins=NBINS,
                                  minmax=(0, 2 * np.pi), ep=epochs["RUN"])
phase_centers = tc.index.values

# =====================================================================
# Figure 4: theta cycles, spikes and the cycle-averaged population rate
# =====================================================================
sorted_units = res.loc[locked].sort_values("pref_RUN")["unit"].values

# display window: the 1.6 s of running with the strongest, cleanest theta
WIN_S = 1.6
n_w = int(WIN_S * tu.LFP_FS)
env, rawd = amps["RUN"].d, raws["RUN"].d
kern = np.ones(n_w) / n_w
mean_env = np.convolve(env, kern, mode="valid")
peak_abs = np.convolve(np.abs(rawd) > 1500, np.ones(n_w), mode="valid")
contiguous = (amps["RUN"].t[n_w - 1:] - amps["RUN"].t[:len(mean_env)]) < WIN_S * 1.01
score = np.where(contiguous & (peak_abs == 0), mean_env, -np.inf)
t0 = amps["RUN"].t[int(np.argmax(score))]
win = nap.IntervalSet(t0, t0 + WIN_S)
raw_w = raws["RUN"].restrict(win)
ph_w = phases["RUN"].restrict(win)
filt_w = tu.bandpass(raws["RUN"], *tu.THETA_BAND).restrict(win)
trough_t = ph_w.t[np.where(np.diff(np.sign(ph_w.d - np.pi)) > 0)[0]]

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 2.2, 1.4], width_ratios=[3, 1.15],
                      hspace=0.35, wspace=0.25)
ax = fig.add_subplot(gs[0, 0])
ax.plot(raw_w.t - t0, raw_w.d, color="0.7", lw=0.7, label="raw LFP")
ax.plot(filt_w.t - t0, filt_w.d, color="C0", lw=1.6, label="6-10 Hz theta")
for tt in trough_t:
    ax.axvline(tt - t0, color="0.85", lw=0.6, zorder=0)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.set_ylabel("µV")
ax.set_title(f"{SESSION}, CA1 channel {BEST_CH}: spikes track the theta cycle "
             f"({WIN_S:.1f} s of running)")
ax.tick_params(labelbottom=False)

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
for i, uid in enumerate(sorted_units):
    st = units[uid].restrict(win).t
    ct = res.loc[res.unit == uid, "cell_type"].values[0]
    ax2.vlines(st - t0, i, i + 0.95, lw=1.4,
               color="C3" if ct == "inhibitory" else "C0")
for tt in trough_t:
    ax2.axvline(tt - t0, color="0.85", lw=0.6, zorder=0)
ax2.set_ylabel("phase-locked units\n(sorted by preferred phase)")
ax2.set_ylim(0, len(sorted_units))
ax2.set_xlabel("time from window start (s);  grey lines = theta troughs")
ax2.set_xlim(0, WIN_S)
handles = [plt.Line2D([], [], color="C0", lw=2, label="excitatory"),
           plt.Line2D([], [], color="C3", lw=2, label="inhibitory")]
ax2.legend(handles=handles, fontsize=8, loc="upper right")

ax3 = fig.add_subplot(gs[2, 0])
pop_exc = tc[[u for u in sorted_units
              if res.loc[res.unit == u, "cell_type"].values[0] == "excitatory"]]
pop_inh = tc[[u for u in sorted_units
              if res.loc[res.unit == u, "cell_type"].values[0] == "inhibitory"]]
xx = np.degrees(np.concatenate([phase_centers, phase_centers + 2 * np.pi]))
for pop, c, lbl in [(pop_exc, "C0", "excitatory"), (pop_inh, "C3", "inhibitory")]:
    norm = pop / pop.mean()
    m = np.tile(norm.mean(axis=1).values, 2)
    s = np.tile((norm.std(axis=1) / np.sqrt(norm.shape[1])).values, 2)
    ax3.plot(xx, m, color=c, label=lbl)
    ax3.fill_between(xx, m - s, m + s, color=c, alpha=0.25)
ax3.plot(xx, 1 + 0.15 * np.cos(np.radians(xx)), color="0.6", lw=1, ls="--",
         label="LFP theta (schematic)")
ax3.set_xlabel("theta phase (deg;  0 = peak of filtered LFP)")
ax3.set_ylabel("normalised rate")
ax3.set_xlim(0, 720)
ax3.set_xticks(np.arange(0, 721, 90))
ax3.legend(fontsize=8, ncol=3)

ax4 = fig.add_subplot(gs[:, 1])
z = tc[sorted_units].values.T
z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, keepdims=True)
im = ax4.imshow(np.hstack([z, z]), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                extent=[0, 720, len(sorted_units), 0], interpolation="nearest")
plt.colorbar(im, ax=ax4, label="firing rate (z-scored across phase bins)",
             fraction=0.05, pad=0.03)
ax4.set_xlabel("theta phase (deg)")
ax4.set_ylabel("unit (sorted by preferred phase)")
ax4.set_title("phase tuning, all locked units")
ax4.set_xticks(np.arange(0, 721, 180))
for a in [ax, ax2, ax3]:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("fig04_theta_cycle_entrainment.png", dpi=150, bbox_inches="tight")
print("saved fig04_theta_cycle_entrainment.png")

# =====================================================================
# Figure 5: example units
# =====================================================================
cand = res.loc[locked].copy()
exc = cand[(cand.cell_type == "excitatory") & (cand.n_RUN >= 500)].nlargest(3, "mrl_RUN")
inh = cand[cand.cell_type == "inhibitory"].nlargest(3, "mrl_RUN")
examples = pd.concat([exc, inh])

fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, (_, row) in zip(axs.ravel(), examples.iterrows()):
    uid = int(row.unit)
    ph = lookups["RUN"](units[uid].t)
    counts, edges = np.histogram(ph, bins=NBINS, range=(0, 2 * np.pi))
    centers = np.degrees(edges[:-1] + np.diff(edges) / 2)
    frac = counts / counts.sum() * 100
    ax.bar(np.concatenate([centers, centers + 360]), np.tile(frac, 2),
           width=360 / NBINS * 0.95,
           color="C3" if row.cell_type == "inhibitory" else "C0", alpha=0.85)
    ax.plot(np.linspace(0, 720, 200),
            frac.mean() * (1 + 0.5 * np.cos(np.radians(np.linspace(0, 720, 200)))),
            color="0.4", lw=1, ls="--")
    ax.axvline(np.degrees(row.pref_RUN), color="k", lw=1.2)
    ax.axvline(np.degrees(row.pref_RUN) + 360, color="k", lw=1.2)
    ax.set_title(f"unit {uid} ({row.cell_type[:3]})  n={int(row.n_RUN)}\n"
                 f"MRL={row.mrl_RUN:.3f}  PPC={row.ppc_RUN:.4f}  "
                 f"Rayleigh p={row.p_RUN:.1e}", fontsize=9)
    ax.set_xlim(0, 720); ax.set_xticks(np.arange(0, 721, 180))
    ax.spines[["top", "right"]].set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel("theta phase (deg)")
for ax in axs[:, 0]:
    ax.set_ylabel("% of spikes")
fig.suptitle("Spike-phase distributions of example CA1 units during running "
             "(two theta cycles shown)", y=1.0)
fig.tight_layout()
fig.savefig("fig05_example_units.png", dpi=150)
print("saved fig05_example_units.png")

# =====================================================================
# Figure 6: population statistics
# =====================================================================
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 0.6, 31)
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = incl.values & (res.cell_type == ct).values
    ax.hist(res.loc[m, "mrl_RUN"], bins=bins, alpha=0.6, color=c,
            label=f"{ct} (n={m.sum()})")
ax.hist(res.loc[incl, "mrl_null_p95"], bins=bins, histtype="step", color="k",
        ls="--", label="jitter null, 95th pct")
ax.set_xlabel("mean resultant length (RUN)"); ax.set_ylabel("units")
ax.legend(fontsize=8); ax.set_title("strength of theta phase locking")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(res.loc[incl, "mrl_null_p95"], res.loc[incl, "mrl_RUN"], s=18,
           c=["C3" if t == "inhibitory" else "C0"
              for t in res.loc[incl, "cell_type"]])
lim = [0, res.loc[incl, "mrl_RUN"].max() * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(0, 0.12); ax.set_ylim(*lim)
ax.set_xlabel("null MRL, 95th pct (jittered spikes)")
ax.set_ylabel("observed MRL")
ax.set_title("observed vs jitter null")

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = locked & (res.cell_type == ct).values
    cnt, edges = np.histogram(res.loc[m, "pref_RUN"], bins=18, range=(0, 2 * np.pi))
    ax.bar(edges[:-1] + np.pi / 18, cnt, width=2 * np.pi / 18, color=c,
           alpha=0.6, label=ct)
    mu = tu.circ_mean(res.loc[m, "pref_RUN"])
    ax.annotate("", xy=(mu, cnt.max()), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.set_theta_zero_location("E")
ax.set_rlabel_position(285)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("preferred phase of locked units\n(0 deg = LFP theta peak)",
             pad=22, fontsize=10)
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.25, -0.12))

ax = fig.add_subplot(gs[1, 0])
ax.scatter(np.asarray(units.rates)[incl.values], res.loc[incl, "mrl_RUN"], s=18,
           c=["C3" if t == "inhibitory" else "C0"
              for t in res.loc[incl, "cell_type"]])
ax.set_xscale("log")
ax.set_xlabel("mean firing rate (Hz)"); ax.set_ylabel("MRL (RUN)")
ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color="C0", label="excitatory"),
                   plt.Line2D([], [], ls="", marker="o", color="C3", label="inhibitory")],
          fontsize=8)
ax.set_title("locking strength vs firing rate")

ax = fig.add_subplot(gs[1, 1])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = incl.values & (res.cell_type == ct).values
    v = np.sort(res.loc[m, "ppc_RUN"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=ct)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xscale("symlog", linthresh=1e-4)
ax.set_xlabel("pairwise phase consistency (spike-count unbiased)")
ax.set_ylabel("cumulative fraction of units")
ax.legend(fontsize=8); ax.set_title("PPC distribution")

ax = fig.add_subplot(gs[1, 2])
frac = [res.loc[incl & (res.cell_type == ct), "sig_RUN"].mean() * 100
        for ct in ["excitatory", "inhibitory"]]
ax.bar(["excitatory", "inhibitory"], frac, color=["C0", "C3"])
for i, f in enumerate(frac):
    ax.text(i, f + 1.5, f"{f:.0f}%", ha="center")
ax.set_ylim(0, 108); ax.set_ylabel("% significantly locked")
ax.set_title("Rayleigh test, FDR q<0.05")
for a in fig.axes:
    if a.name != "polar":
        a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{SESSION}: population summary of CA1 theta phase entrainment "
             "during running", y=0.99)
fig.savefig("fig06_population_summary.png", dpi=150, bbox_inches="tight")
print("saved fig06_population_summary.png")

# =====================================================================
# Figure 7: brain-state comparison and the spike-triggered LFP control
# =====================================================================
ok_all = ((res.n_RUN >= MIN_SPIKES) & (res.n_REM >= MIN_SPIKES)
          & (res.n_nonREM >= MIN_SPIKES)).values
fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))

ax = axs[0]
data = [res.loc[ok_all, f"mrl_{s}"].values for s in ["RUN", "REM", "nonREM"]]
parts = ax.violinplot(data, showmedians=True)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.06, 0.06, len(d)),
               d, s=6, color="0.3", alpha=0.5)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["RUN", "REM", "nonREM"])
ax.set_ylabel("MRL to 6-10 Hz phase")
ax.set_title(f"MRL alone does not separate\nthe states (n={ok_all.sum()} units)",
             fontsize=10)
w_run_nrem = wilcoxon(data[0], data[2])
w_run_rem = wilcoxon(data[0], data[1])
ax.text(0.02, 0.03, f"RUN vs nonREM  p={w_run_nrem.pvalue:.2f}\n"
                    f"RUN vs REM      p={w_run_rem.pvalue:.3f}\n"
                    "(Wilcoxon signed-rank)",
        transform=ax.transAxes, va="bottom", fontsize=7.5)

ax = axs[1]
# The MRL to a 6-10 Hz Hilbert phase is NOT diagnostic of an underlying rhythm:
# bandpass + Hilbert returns a phase for any signal, including one with no theta
# peak at all. What separates a real rhythm from a filtering artefact is whether
# the spike-triggered LFP average oscillates. Quantify that per unit as the
# theta/delta band-power ratio of the unit's own STA waveform.
sta_rhythm = {}
for name in ["RUN", "REM", "nonREM"]:
    vals = []
    for uid in res.loc[ok_all, "unit"].astype(int):
        sta, lags, nsp = tu.spike_triggered_average(raws[name], units[uid].t,
                                                    max_spikes=3000)
        vals.append(tu.theta_delta_ratio(sta - sta.mean()) if sta is not None
                    else np.nan)
    sta_rhythm[name] = np.array(vals, dtype=float)

parts = ax.violinplot([sta_rhythm[s][np.isfinite(sta_rhythm[s])]
                       for s in ["RUN", "REM", "nonREM"]], showmedians=True)
for i, s in enumerate(["RUN", "REM", "nonREM"]):
    v = sta_rhythm[s][np.isfinite(sta_rhythm[s])]
    ax.scatter(np.full(len(v), i + 1) + np.random.uniform(-0.06, 0.06, len(v)),
               v, s=6, color="0.3", alpha=0.5)
ax.set_yscale("log")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["RUN", "REM", "nonREM"])
ax.set_ylabel("theta/delta power of the STA waveform")
w_sta = wilcoxon(sta_rhythm["RUN"], sta_rhythm["nonREM"])
ax.text(0.02, 0.03, f"RUN vs nonREM  p={w_sta.pvalue:.1e}",
        transform=ax.transAxes, fontsize=7.5)
ax.set_title("spike-locked rhythmicity does\nseparate the states", fontsize=10)

# spike-triggered LFP average: a rhythm produces repeating side lobes
best_unit = int(res.loc[ok_all].sort_values("mrl_RUN", ascending=False).unit.values[0])
ax = axs[2]
for name, c in [("RUN", "C0"), ("REM", "C2"), ("nonREM", "C3")]:
    sta, lags, nsp = tu.spike_triggered_average(raws[name], units[best_unit].t)
    if sta is None:
        continue
    ax.plot(lags, sta, color=c, label=f"{name} (n={nsp})")
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("time from spike (s)"); ax.set_ylabel("LFP (µV)")
ax.set_title(f"spike-triggered LFP, unit {best_unit}")
ax.legend(fontsize=8)

ax = axs[3]
for name, c in [("RUN", "C0"), ("REM", "C2"), ("nonREM", "C3")]:
    v = np.sort(res.loc[ok_all, f"ppc_{name}"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=name)
ax.set_xscale("symlog", linthresh=1e-4)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("PPC"); ax.set_ylabel("cumulative fraction")
ax.legend(fontsize=8); ax.set_title("PPC by brain state")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Control: is the locking to a real rhythm? "
             "(6-10 Hz Hilbert phase returns a value in every state)", y=1.02)
fig.tight_layout()
fig.savefig("fig07_state_comparison.png", dpi=150, bbox_inches="tight")
print("saved fig07_state_comparison.png")
