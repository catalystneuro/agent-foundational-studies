"""Population-level figures: example units, MRL vs chance, preferred-phase structure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import theta_lib as tl

df = pd.read_csv("results/unit_stats.csv")
ph_normal = np.load("results/spike_phases_normal.npz")
sig = df.shuffle_p < 0.05

print(f"{len(df)} units, {sig.sum()} phase-locked (circular-shift shuffle, p<0.05)")

NBINS = 36
EDGES = np.linspace(0, 2 * np.pi, NBINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def hist_norm(phases, nbins=NBINS):
    h, _ = np.histogram(phases, bins=np.linspace(0, 2 * np.pi, nbins + 1))
    return h / h.mean()


# ---------------------------------------------------------------------------
# Figure 3: example units, weak to strong
# ---------------------------------------------------------------------------
order = df.sort_values("mrl", ascending=False)
picks = []
for ct in ["pyramidal", "interneuron"]:
    sel = order[order.cell_type == ct]
    picks += list(sel.index[:3])
picks += list(order.index[-2:])  # two of the least modulated units, as a contrast

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, i in zip(axes.ravel(), picks):
    r = df.loc[i]
    p = ph_normal[f"{r.session}|{int(r.unit)}"]
    h = hist_norm(p)
    x = np.concatenate([CENTERS, CENTERS + 2 * np.pi])
    ax.bar(np.degrees(x), np.concatenate([h, h]), width=360 / NBINS * 0.95,
           color="crimson" if r.shuffle_p < 0.05 else "0.6")
    ax.plot(np.degrees(x), 1 + 0.35 * np.cos(x), color="navy", lw=1.5, alpha=0.8)
    ax.axvline(np.degrees(r.pref_phase), color="k", ls="--", lw=1)
    ax.set_xlim(0, 720)
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.set_ylim(0, max(1.65, h.max() * 1.15))
    ax.set_title(f"{r.cell_type[:4]}. {r.session.split('-')[1]} u{int(r.unit)}\n"
                 f"MRL={r.mrl:.3f} (chance {r.mrl_null_mean:.3f}), n={int(r.n_spikes)}",
                 fontsize=9)
    if ax in axes[:, 0]:
        ax.set_ylabel("spike count\n(norm. to mean)")
    if ax in axes[-1, :]:
        ax.set_xlabel("theta phase (deg)")
fig.suptitle("Spike-phase distributions during running (blue: theta cycle, 0 deg = LFP peak)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig03_example_units.png", dpi=150)
print("wrote fig03_example_units.png")

# ---------------------------------------------------------------------------
# Figure 4: population summary
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

ax = axes[0, 0]
bins = np.linspace(0, max(df.mrl.max(), df.mrl_null_p95.max()) * 1.05, 40)
ax.hist(df.mrl, bins=bins, color="crimson", alpha=0.75, label="observed")
ax.hist(df.mrl_null_mean, bins=bins, color="0.4", alpha=0.7, label="circular-shift shuffle")
ax.set_xlabel("mean resultant length")
ax.set_ylabel("units")
ax.set_title(f"phase locking vs chance\n{sig.sum()}/{len(df)} units significant (p<0.05)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.scatter(df.mrl_null_p95[~sig], df.mrl[~sig], s=14, color="0.6", label="n.s.")
ax.scatter(df.mrl_null_p95[sig], df.mrl[sig], s=14, color="crimson", label="p<0.05")
lim = [0, max(df.mrl.max(), df.mrl_null_p95.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim), ax.set_ylim(lim)
ax.set_xlabel("95th pct of shuffled MRL")
ax.set_ylabel("observed MRL")
ax.set_title("every unit against its own null", fontsize=10)
ax.legend(fontsize=8)

ax = axes[0, 2]
for ct, color in [("pyramidal", "steelblue"), ("interneuron", "darkorange")]:
    d = df[df.cell_type == ct]
    ax.scatter(d.rate, d.mrl, s=16, color=color, alpha=0.8, label=f"{ct} (n={len(d)})")
ax.set_xscale("log")
ax.set_xlabel("firing rate during running (Hz)")
ax.set_ylabel("MRL")
ax.set_title("locking strength vs firing rate", fontsize=10)
ax.legend(fontsize=8)

ax = plt.subplot(2, 3, 4, projection="polar")
axes[1, 0].remove()
for ct, color in [("pyramidal", "steelblue"), ("interneuron", "darkorange")]:
    d = df[(df.cell_type == ct) & sig]
    h, _ = np.histogram(d.pref_phase, bins=np.linspace(0, 2 * np.pi, 25))
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ax.bar(theta, h / max(h.sum(), 1), width=2 * np.pi / 24, alpha=0.6, color=color, label=ct)
    mu = np.angle(np.mean(np.exp(1j * d.pref_phase)))
    ax.plot([mu, mu], [0, ax.get_ylim()[1]], color=color, lw=2.5)
ax.set_theta_zero_location("E")
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.32, -0.12))

# Permutation test on the difference in mean preferred phase between cell types.
rng = np.random.default_rng(0)
pa = df.pref_phase[(df.cell_type == "pyramidal") & sig].values
pb = df.pref_phase[(df.cell_type == "interneuron") & sig].values
cmean = lambda x: np.angle(np.mean(np.exp(1j * x)))
obs = abs(np.angle(np.exp(1j * (cmean(pa) - cmean(pb)))))
pool = np.concatenate([pa, pb])
null = [abs(np.angle(np.exp(1j * (cmean(q[:len(pa)]) - cmean(q[len(pa):])))))
        for q in (rng.permutation(pool) for _ in range(5000))]
p_phase = (np.sum(np.array(null) >= obs) + 1) / 5001
ax.set_title(f"preferred phase of locked units\nΔ = {np.degrees(obs):.0f}°, permutation p={p_phase:.2g}",
             fontsize=10, pad=22)

# Locking strength by cell type. The bias-corrected MRL is used because interneurons
# fire ~20x more spikes, and the raw MRL is biased upward at small spike counts (i.e.
# the raw statistic favours pyramidal cells, so this comparison is conservative).
ax = axes[1, 1]
data = [df.mrl_unbiased[df.cell_type == ct] for ct in ["pyramidal", "interneuron"]]
parts = ax.violinplot(data, showmedians=True)
for pc, color in zip(parts["bodies"], ["steelblue", "darkorange"]):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax.set_xticks([1, 2], [f"pyramidal\n(n={len(data[0])})", f"interneuron\n(n={len(data[1])})"])
ax.set_ylabel("bias-corrected MRL (all units)")
from scipy.stats import mannwhitneyu
u, pval = mannwhitneyu(data[0], data[1])
ax.set_title(f"medians {data[0].median():.3f} vs {data[1].median():.3f}, "
             f"Mann-Whitney p={pval:.2g}", fontsize=10)

ax = axes[1, 2]
per_session = df.groupby("session").agg(n=("mrl", "size"), frac=("shuffle_p", lambda x: (x < 0.05).mean()),
                                        mrl=("mrl", "median"))
ax.bar(range(len(per_session)), per_session.frac, color="crimson", alpha=0.8)
ax.set_xticks(range(len(per_session)),
              [s.split("-")[1] + "\n" + s.split("-")[2] for s in per_session.index], fontsize=8)
for i, (n, f) in enumerate(zip(per_session.n, per_session.frac)):
    ax.text(i, f + 0.02, f"n={n}", ha="center", fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("fraction of units phase-locked")
ax.set_title("consistency across sessions", fontsize=10)

fig.tight_layout()
fig.savefig("fig04_population_summary.png", dpi=150)
print("wrote fig04_population_summary.png")

# ---------------------------------------------------------------------------
# Figure 5: the whole population as a phase-sorted heat map
# ---------------------------------------------------------------------------
locked = df[sig].sort_values("pref_phase")
M = np.array([hist_norm(ph_normal[f"{r.session}|{int(r.unit)}"]) for _, r in locked.iterrows()])
M2 = np.concatenate([M, M], axis=1)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3.2, 1], sharex=True)
im = axes[0].imshow(M2, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, 720, 0, len(M2)], vmin=0.4, vmax=1.8)
axes[0].set_ylabel("phase-locked unit (sorted by preferred phase)")
axes[0].set_title(f"Every significantly locked unit ({len(M)} of {len(df)}), two theta cycles",
                  fontsize=11)
cb = fig.colorbar(im, ax=axes[0], pad=0.01)
cb.set_label("firing rate (norm. to unit mean)")

pop = M2.mean(axis=0)
x = np.degrees(np.concatenate([CENTERS, CENTERS + 2 * np.pi]))
axes[1].plot(x, pop, color="crimson", lw=2, label="population mean")
axes[1].plot(x, 1 + 0.06 * np.cos(np.radians(x)), color="navy", lw=1.5, label="theta cycle")
axes[1].set_xlabel("theta phase (deg)")
axes[1].set_ylabel("norm. rate")
axes[1].set_xlim(0, 720)
axes[1].set_xticks([0, 180, 360, 540, 720])
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig05_population_heatmap.png", dpi=150)
print("wrote fig05_population_heatmap.png")

# summary numbers used by the README
summary = dict(
    n_units=len(df), n_sessions=int(df.session.nunique()), n_locked=int(sig.sum()),
    frac_locked=float(sig.mean()),
    median_mrl=float(df.mrl.median()), median_mrl_null=float(df.mrl_null_mean.median()),
    median_mrl_pyr=float(df.mrl_unbiased[df.cell_type == "pyramidal"].median()),
    median_mrl_int=float(df.mrl_unbiased[df.cell_type == "interneuron"].median()),
    pref_phase_diff_deg=float(np.degrees(obs)), pref_phase_perm_p=float(p_phase),
    pref_phase_rayleigh_p_pyr=float(tl.rayleigh(pa)[2]),
    pref_phase_rayleigh_p_int=float(tl.rayleigh(pb)[2]),
    pref_phase_pyr=float(np.degrees(np.angle(np.mean(np.exp(1j * df.pref_phase[(df.cell_type == "pyramidal") & sig]))) % (2 * np.pi))),
    pref_phase_int=float(np.degrees(np.angle(np.mean(np.exp(1j * df.pref_phase[(df.cell_type == "interneuron") & sig]))) % (2 * np.pi))),
    mannwhitney_p=float(pval),
)
pd.Series(summary).to_json("results/summary.json", indent=1)
print(summary)
