"""Generate all figures for the auditory frequency tuning analysis."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "scripts")
from load_data import load_session, build_trials
import tuning_analysis as ta

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
FREQS = ta.FREQS
FREQS_KHZ = FREQS / 1000.0


# --------------------------------------------------------------------------
# 1. Raster + PSTH for a representative frequency-tuned unit
# --------------------------------------------------------------------------
def plot_unit_raster_psth(nwb, trials, unit_id, asset_id, fname):
    units = nwb["units"]
    sp = np.asarray(units[unit_id].t, dtype=float)
    starts = np.asarray(trials.start, dtype=float)
    freqs = np.asarray(trials.stim_frequency, dtype=float)

    resp, base, order, w_p, kw_p = ta.unit_stats(sp, starts, freqs)
    tuning, base_rate = ta.tuning_and_rate(resp, base, freqs, order)
    trial_i, rel_t, s_starts, s_freqs = ta.raster_relative(sp, starts, freqs)

    # One panel per frequency, ordered low -> high
    fig, axes = plt.subplots(len(FREQS), 1, figsize=(7.2, 9), sharex=True)
    pre, post = 0.020, 0.200
    for ax, f in zip(axes, FREQS):
        m = s_freqs == f
        t_idx = np.where(m)[0]
        # raster
        mask = np.isin(trial_i, t_idx)
        ax.eventplot(rel_t[mask], colors="k", linewidths=0.6)
        # PSTH
        bins = np.arange(-pre, post + 0.002, 0.002)
        occ = (bins[1:] + bins[:-1]) / 2
        srt, _ = np.histogram(rel_t[mask], bins=bins)
        rate = srt / (0.002 * len(t_idx))
        ax.plot(occ, rate, color="C0", lw=1.2)
        ax.axvline(0, color="r", lw=0.8, ls="--")
        ax.set_ylabel(f"{f/1000:.0f} kHz\n({len(t_idx)} trials)", fontsize=9)
        ax.set_ylim(0, max(rate.max() * 1.3, 1))
    axes[-1].set_xlabel("Time from tone onset (s)")
    axes[0].set_title(f"Raster (black) + PSTH (blue) for tuned unit {unit_id}\n"
                      f"session {asset_id} | Wilcoxon p={w_p:.2e}, KW p={kw_p:.2e}",
                      fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# 2. Example tuning curves: a few units spanning BF
# --------------------------------------------------------------------------
def plot_example_tuning_curves(all_df, fname):
    # units with clean single-peak tuning
    by_type = {bf: all_df[(all_df.frequency_tuned) & (all_df.best_frequency == bf)]
               for bf in FREQS}
    chosen = {}
    for bf, sub in by_type.items():
        if len(sub):
            chosen[bf] = sub.iloc[0]

    fig, axes = plt.subplots(1, len(FREQS), figsize=(13, 3.2), sharey=True)
    for ax, bf in zip(axes, FREQS):
        if bf not in chosen:
            ax.axis("off")
            continue
        r = chosen[bf]
        rates = [r[f"rate_{int(f)}"] for f in FREQS]
        ax.plot(FREQS_KHZ, rates, "-o", color="C3")
        ax.set_yscale("symlog", linthresh=1)
        best = r["best_frequency"]
        ax.axvline(best / 1000, color="k", ls=":", lw=1)
        ax.set_title(f"BF {best/1000:.0f} kHz\nunit {r['unit']}", fontsize=9)
        ax.set_xticks(FREQS_KHZ)
        ax.set_xticklabels([f"{f:.0f}" for f in FREQS_KHZ])
    for ax in axes:
        ax.set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Net rate (sp/s)")
    fig.suptitle("Example tuning curves, one per best frequency", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# 3. Population tuning heatmap (min-max normalized net rate per unit)
# --------------------------------------------------------------------------
def plot_tuning_heatmap(all_df, fname):
    # only frequency-tuned, classify mod index
    sub = all_df[all_df.frequency_tuned].copy()
    rate_cols = [f"rate_{int(f)}" for f in FREQS]
    mat = sub[rate_cols].values
    # normalized per row min-max
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1
    norm = (mat - mn) / rng
    # sort within BF groups so tuned units cluster
    order = np.argsort(sub["best_frequency"].values, kind="stable")
    norm = norm[order]
    bf = sub["best_frequency"].values[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [6, 1]})
    ax = axes[0]
    im = ax.imshow(norm, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(FREQS)))
    ax.set_xticklabels([f"{f:.0f}" for f in FREQS_KHZ])
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(f"Unit index (n={norm.shape[0]})")
    ax.set_title("Normalized net-rate tuning curves (tuned units)", fontsize=10)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Min-max normalized net rate")
    # colorbar of BF
    ax2 = axes[1]
    bf_colors = np.array([FREQS_KHZ[np.where(FREQS == b)[0][0]] for b in bf])
    norm2 = Normalize(vmin=FREQS_KHZ.min(), vmax=FREQS_KHZ.max())
    ax2.imshow(bf_colors[:, None], aspect="auto", cmap="viridis", norm=norm2)
    ax2.set_yticks([])
    ax2.set_xticks([0])
    ax2.set_xticklabels(["BF"], fontsize=8)
    ax2.set_title("Best freq", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# 4. BF distribution + tuning fraction
# --------------------------------------------------------------------------
def plot_bf_distribution(all_df, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    from collections import Counter
    cnt = Counter(all_df["best_frequency"].dropna())
    bins = FREQS_KHZ
    vals = [cnt.get(f, 0) / len(all_df) for f in FREQS]
    ax.bar(bins, vals, color="C1", width=0.55)
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{f:.0f}" for f in bins])
    ax.set_xlabel("Best frequency (kHz)")
    ax.set_ylabel("Fraction of units")
    ax.set_title(f"Best-frequency distribution (n={len(all_df)})")

    ax = axes[1]
    per_sess = all_df.groupby("session")["frequency_tuned"].mean()
    ax.bar(np.arange(len(per_sess)), per_sess.values, color="C2")
    ax.set_xticks(np.arange(len(per_sess)))
    ax.set_xticklabels([s.replace("_"," ") for s in per_sess.index], rotation=45, fontsize=7)
    ax.axhline(all_df["frequency_tuned"].mean(), color="k", ls="--", lw=1)
    ax.set_ylabel("Fraction frequency-tuned")
    ax.set_title(f"Tuning fraction per session (mean={all_df['frequency_tuned'].mean():.2f})")
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# 5. Average normalized tuning curve (population)
# --------------------------------------------------------------------------
def plot_population_avg_curve(all_df, fname):
    sub = all_df[all_df.frequency_tuned]
    rate_cols = [f"rate_{int(f)}" for f in FREQS]
    mat = sub[rate_cols].values
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1
    norm = (mat - mn) / rng
    mean_curve = norm.mean(axis=0)
    sem = norm.std(axis=0) / np.sqrt(len(norm))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(FREQS_KHZ, mean_curve, yerr=sem, fmt="-o", color="C0",
                capsize=4, lw=2)
    ax.set_xticks(FREQS_KHZ)
    ax.set_xticklabels([f"{f:.0f}" for f in FREQS_KHZ])
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized net rate")
    ax.set_title(f"Mean normalized tuning curve across tuned units (n={len(sub)})")
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


if __name__ == "__main__":
    all_df = pd.read_csv("figures/all_sessions_results.csv")
    plot_bf_distribution(all_df, f"{FIGDIR}/bf_distribution.png")
    plot_example_tuning_curves(all_df, f"{FIGDIR}/example_tuning_curves.png")
    plot_tuning_heatmap(all_df, f"{FIGDIR}/tuning_heatmap.png")
    plot_population_avg_curve(all_df, f"{FIGDIR}/population_avg_curve.png")
    print("population figures done")
