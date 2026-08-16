"""Figure builders for the theta phase-entrainment analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import theta as th

NBINS = 36
BINS = np.linspace(0, 2 * np.pi, NBINS + 1)
CENTERS = (BINS[:-1] + BINS[1:]) / 2
CT_COLOR = {"excitatory": "C0", "inhibitory": "C1"}
MIN_SPIKES = 100


def locked_both(df, states=("run", "REM")):
    """Preferred phases of units significantly locked in both of `states`.

    Units that are not locked have an essentially arbitrary preferred phase, so
    including them would dilute any comparison of phase preference between
    states with noise.
    """
    sub = df[(df.n_spikes >= MIN_SPIKES) & (df.shuffle_p < 0.05)
             & (df.state.isin(states))]
    piv = sub.pivot_table(index=["session", "unit"], columns="state",
                          values="pref_phase")
    return piv.dropna(subset=list(states))


def phase_hist(ph, density=True):
    cnt, _ = np.histogram(ph, bins=BINS)
    return cnt / cnt.sum() if density and cnt.sum() else cnt


def _two_cycles(y):
    return np.concatenate([y, y])


TWO_C = np.concatenate([CENTERS, CENTERS + 2 * np.pi])


def fig_spike_phase_excerpt(res, fname="fig03_spike_phase_excerpt.png", n_each=8):
    """Raw LFP, theta filter, and spikes of the most-locked units side by side."""
    ex = res["excerpt"]
    df = res["df"]
    run = df[df.state == "run"].copy()
    # Units that both lock well and actually fire inside the displayed window.
    run["n_in_window"] = [len(ex["spikes"][int(u)]) for u in run.unit]
    pick = []
    for ct in ["inhibitory", "excitatory"]:
        sub = run[(run.cell_type == ct) & (run.n_in_window >= 5)]
        sub = sub.sort_values("mrl", ascending=False).head(n_each)
        pick.append(sub.sort_values("pref_phase"))
    import pandas as pd
    order_df = pd.concat(pick)
    order = list(order_df.unit)
    colors = [CT_COLOR[res["cell_type"][u]] for u in order]

    ph = ex["phase"]
    troughs = ex["t"][1:][(ph[:-1] < np.pi) & (ph[1:] >= np.pi)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.7]})
    ax = axes[0]
    ax.plot(ex["t"], ex["raw"] * 1e3, color="0.65", lw=0.8, label="raw LFP")
    ax.plot(ex["t"], ex["filt"] * 1e3, color="k", lw=1.8, label="6-10 Hz theta")
    for tt in troughs:
        ax.axvline(tt, color="C3", lw=0.9, ls=":", alpha=0.8)
    ax.set_ylabel("LFP (mV)")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(f"{res['tag']} — spike times of the most theta-locked units "
                 "during a single running bout\n"
                 "dotted red lines mark theta troughs (phase = π)")

    ax = axes[1]
    for j, u in enumerate(order):
        t = ex["spikes"][u]
        ax.plot(t, np.full_like(t, j), "|", ms=10, color=colors[j])
    for tt in troughs:
        ax.axvline(tt, color="C3", lw=0.9, ls=":", alpha=0.8)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"u{int(u)} ({np.degrees(p):.0f}°)"
                        for u, p in zip(order_df.unit, order_df.pref_phase)],
                       fontsize=7)
    ax.set_ylabel("unit (preferred phase in parentheses)")
    ax.set_xlabel("time (s)")
    handles = [plt.Line2D([], [], color=c, marker="|", ls="", ms=10, label=k)
               for k, c in CT_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2,
              framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def fig_lfp_states(res, fname="fig02_theta_signal.png"):
    """Theta in the raw signal and in the power spectrum of each brain state."""
    ex, psds = res["excerpt"], res["psds"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(ex["t"] - ex["t"][0], ex["raw"] * 1e3, color="0.6", lw=0.9,
            label="raw LFP")
    ax.plot(ex["t"] - ex["t"][0], ex["filt"] * 1e3, color="C3", lw=1.8,
            label="6-10 Hz filtered")
    axp = ax.twinx()
    axp.plot(ex["t"] - ex["t"][0], ex["phase"], color="C0", lw=0.7, alpha=0.8)
    axp.set_ylabel("Hilbert phase (rad)", color="C0")
    axp.set_yticks([0, np.pi, 2 * np.pi])
    axp.set_yticklabels(["0", "π", "2π"])
    ax.set_xlim(0, 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("LFP (mV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"channel {res['best_ch']}: theta extraction during running")

    ax = axes[1]
    for state, color in zip(["run", "REM", "nonREM"], ["C2", "C4", "0.5"]):
        if state not in psds:
            continue
        f, p = psds[state]
        ax.semilogy(f, p, color=color, label=state)
    ax.axvspan(*th.THETA_BAND, color="C3", alpha=0.12)
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-11, 1e-7)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("a theta peak is present during running and REM, absent in non-REM")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def fig_example_units(res, fname="fig04_example_units.png"):
    """Phase histograms, polar plots, and shuffle nulls for two example units."""
    df, phases = res["df"], res["phases"]
    run = df[(df.state == "run") & (df.n_spikes >= 500)]
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1, 1], hspace=0.45,
                  wspace=0.32)
    for r, ct in enumerate(["excitatory", "inhibitory"]):
        sub = run[run.cell_type == ct].sort_values("mrl", ascending=False)
        row = sub.iloc[0]
        u = int(row.unit)
        ph = phases[("run", u)]
        dens = phase_hist(ph)

        ax = fig.add_subplot(gs[r, 0])
        ax.bar(TWO_C, _two_cycles(dens), width=BINS[1] - BINS[0],
               color=CT_COLOR[ct], edgecolor="none")
        x = np.linspace(0, 4 * np.pi, 400)
        ax.plot(x, dens.mean() * (1 + 0.55 * np.cos(x)), color="0.25", lw=1.3,
                ls="--", label="LFP theta (schematic)")
        ax.axvline(row.pref_phase, color="k", lw=1.5)
        ax.axvline(row.pref_phase + 2 * np.pi, color="k", lw=1.5)
        ax.set_xlim(0, 4 * np.pi)
        ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_xticklabels(["0", "π", "2π", "3π", "4π"])
        ax.set_ylabel("spike probability")
        ax.set_title(f"unit {u} ({ct})   MRL={row.mrl:.3f}, "
                     f"pref={np.degrees(row.pref_phase):.0f}°, n={int(row.n_spikes)}",
                     fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        if r == 1:
            ax.set_xlabel("theta phase (two cycles shown; 0 = peak, π = trough)")

        axp = fig.add_subplot(gs[r, 1], projection="polar")
        axp.bar(CENTERS, dens, width=BINS[1] - BINS[0], color=CT_COLOR[ct],
                alpha=0.85)
        axp.annotate("", xy=(row.pref_phase, row.mrl * dens.max() / 0.5),
                     xytext=(0, 0),
                     arrowprops=dict(color="k", width=2, headwidth=8))
        axp.set_yticklabels([])
        axp.set_title("phase distribution", fontsize=9, pad=22)

        ax2 = fig.add_subplot(gs[r, 2])
        null_lo, null_hi = row.null_mrl_mean, row.null_mrl_p95
        ax2.axvspan(0, null_hi, color="0.85", label="shuffled null (<95th pct)")
        ax2.axvline(null_lo, color="0.4", lw=1.5, label="null mean")
        ax2.axvline(row.mrl, color="r", lw=2.5, label="observed MRL")
        ax2.set_xlim(0, max(row.mrl * 1.3, null_hi * 2))
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("mean resultant length")
        ax2.set_title(f"circular-shift test, p={row.shuffle_p:.3g}", fontsize=10)
        ax2.legend(fontsize=7, loc="center left")
    fig.suptitle(f"{res['tag']} — single-unit theta phase locking during running",
                 y=0.99)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_population_map(res, fname="fig05_population_map.png"):
    """Every unit's phase histogram, sorted by preferred phase."""
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    for ax, state in zip(axes, ["run", "REM", "nonREM"]):
        sub = df[(df.state == state) & (df.n_spikes >= 500)]
        sub = sub.sort_values("pref_phase")
        mat = np.array([_two_cycles(phase_hist(phases[(state, int(u))]))
                        for u in sub.unit])
        mat = (mat - mat.mean(axis=1, keepdims=True)) / mat.std(axis=1, keepdims=True)
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, 4 * np.pi, 0, len(sub)], vmin=-2.5, vmax=2.5)
        ax.plot(sub.pref_phase, np.arange(len(sub)) + 0.5, color="c", lw=1.2)
        ax.plot(sub.pref_phase + 2 * np.pi, np.arange(len(sub)) + 0.5,
                color="c", lw=1.2, label="preferred phase")
        ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_xticklabels(["0", "π", "2π", "3π", "4π"])
        ax.set_xlabel("theta-band phase (two cycles)")
        ax.set_title(f"{state}  (n={len(sub)} units with ≥500 spikes)")
        if ax is axes[0]:
            ax.set_ylabel("unit, sorted by preferred phase")
            ax.legend(fontsize=8, loc="upper left")
        plt.colorbar(im, ax=ax, label="firing rate (z-scored across phase)")
    fig.suptitle(f"{res['tag']} — phase-resolved firing of every unit. During "
                 "running and REM the preferred phases tile the cycle;\nin "
                 "non-REM the apparent modulation is shared by all units at "
                 "one phase (population bursts, not an oscillation).")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def fig_population_stats(df, fname="fig06_population_stats.png"):
    """Pooled MRL distributions, significance counts, and preferred phases."""
    run = df[(df.state == "run") & (df.n_spikes >= MIN_SPIKES)]
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(0.6, run.mrl.max() * 1.05), 30)
    ax.hist(run.mrl, bins=bins, color="C2", alpha=0.85, label="observed")
    ax.hist(run.null_mrl_mean.dropna(), bins=bins, color="0.6", alpha=0.7,
            label="shuffled null (mean)")
    ax.set_xlabel("mean resultant length")
    ax.set_ylabel("units")
    ax.set_title("locking strength vs chance, running", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for ct, sub in run.groupby("cell_type"):
        ax.hist(sub.mrl, bins=bins, alpha=0.6, color=CT_COLOR[ct],
                label=f"{ct} (n={len(sub)}, med={sub.mrl.median():.3f})")
    ax.set_xlabel("mean resultant length")
    ax.set_ylabel("units")
    ax.set_title("interneurons lock more strongly\nthan pyramidal cells", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    frac = (df[df.n_spikes >= MIN_SPIKES]
            .groupby("state")
            .apply(lambda d: (d.shuffle_p < 0.05).mean(), include_groups=False))
    order = [s for s in ["run", "REM", "nonREM"] if s in frac.index]
    ax.bar(order, [frac[s] * 100 for s in order],
           color=["C2", "C4", "0.6"][: len(order)])
    for i, s in enumerate(order):
        n = (df[(df.state == s) & (df.n_spikes >= MIN_SPIKES)])
        ax.text(i, frac[s] * 100 + 1.5,
                f"{int((n.shuffle_p < 0.05).sum())}/{len(n)}", ha="center",
                fontsize=9)
    ax.axhline(5, color="r", ls="--", lw=1, label="chance (5%)")
    ax.set_ylabel("% units significantly locked")
    ax.set_ylim(0, 105)
    ax.set_title("significant 6-10 Hz phase modulation\nby state (see text on non-REM)",
              fontsize=10)
    ax.legend(fontsize=8)

    axp = fig.add_subplot(gs[1, 0], projection="polar")
    for ct, sub in run.groupby("cell_type"):
        sig = sub[sub.shuffle_p < 0.05]
        cnt, _ = np.histogram(sig.pref_phase, bins=BINS)
        axp.bar(CENTERS, cnt, width=BINS[1] - BINS[0], alpha=0.6,
                color=CT_COLOR[ct], label=ct)
    axp.set_yticklabels([])
    axp.set_title("preferred phase of locked units", fontsize=10, pad=24)
    axp.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.25, -0.12))

    ax = fig.add_subplot(gs[1, 1])
    for ct, sub in run.groupby("cell_type"):
        ax.scatter(sub.rate, sub.mrl, s=14, alpha=0.7, color=CT_COLOR[ct],
                   label=ct)
    ax.set_xscale("log")
    ax.set_xlabel("firing rate during running (Hz)")
    ax.set_ylabel("mean resultant length")
    ax.set_title("locking strength vs firing rate", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    keep = locked_both(df)
    ax.scatter(np.degrees(keep["run"]), np.degrees(keep["REM"]), s=14,
               alpha=0.7, color="C4")
    d = np.angle(np.exp(1j * (keep["REM"] - keep["run"])))
    rho = np.abs(np.exp(1j * d).mean())
    ax.plot([0, 360], [0, 360], "k--", lw=1)
    ax.set_xlabel("preferred phase, running (°)")
    ax.set_ylabel("preferred phase, REM (°)")
    ax.set_title(f"phase preference is partly conserved run→REM\n"
                 f"(n={len(keep)} units locked in both; concentration of the "
                 f"difference = {rho:.2f},\nmedian offset = "
                 f"{np.degrees(np.median(d)):+.0f}°)", fontsize=10)
    fig.suptitle("Population summary of theta phase entrainment "
                 f"({df.session.nunique()} sessions, {len(run)} units with "
                 f"≥{MIN_SPIKES} spikes during running)", y=0.98)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_specificity(results, fname="fig07_frequency_specificity.png"):
    """Locking is theta-specific and grows with theta amplitude."""
    import pandas as pd
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    freqs = results[0]["spectrum"].index.values
    for state, color in zip(["run", "REM", "nonREM"], ["C2", "C4", "0.5"]):
        pooled = []
        for res in results:
            cols = [c for c in res["spectrum"].columns if c[0] == state]
            if cols:
                pooled.append(res["spectrum"][cols].values)
        if not pooled:
            continue
        pooled = np.concatenate(pooled, axis=1)
        med = np.nanmedian(pooled, axis=1)
        q1, q3 = np.nanpercentile(pooled, [25, 75], axis=1)
        ax.plot(freqs, med, color=color, lw=2.5, label=state)
        ax.fill_between(freqs, q1, q3, color=color, alpha=0.18)
    ax.axvspan(*th.THETA_BAND, color="C3", alpha=0.12)
    ax.set_xlabel("centre frequency of 3 Hz band (Hz)")
    ax.set_ylabel("MRL (median, IQR shaded)")
    ax.set_title("locking peaks in the theta band during\nrunning and REM, "
                 "but not in non-REM", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    amp = pd.concat([r["amp_split"] for r in results])
    amp = amp[(amp.n_high >= MIN_SPIKES) & (amp.n_low >= MIN_SPIKES)]
    for ct, sub in amp.groupby("cell_type"):
        ax.scatter(sub.mrl_low, sub.mrl_high, s=14, alpha=0.7,
                   color=CT_COLOR[ct], label=ct)
    lim = [0, max(amp.mrl_high.max(), amp.mrl_low.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MRL, low-amplitude theta cycles")
    ax.set_ylabel("MRL, high-amplitude theta cycles")
    frac = (amp.mrl_high > amp.mrl_low).mean()
    ax.set_title(f"locking scales with theta amplitude\n"
                 f"({frac*100:.0f}% of units above the unity line)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    dfs = pd.concat([r["df"] for r in results])
    sub = dfs[(dfs.n_spikes >= MIN_SPIKES) & (dfs.n_matched >= MIN_SPIKES)]
    data = [sub[sub.state == s].mrl_matched.values
            for s in ["run", "REM", "nonREM"]]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["C2", "C4", "0.6"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    for i, d in enumerate(data):
        ax.text(i + 1, np.median(d), f" {np.median(d):.3f}", fontsize=8,
                va="center")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"run\n(n={len(data[0])})", f"REM\n(n={len(data[1])})",
                        f"non-REM\n(n={len(data[2])})"])
    ax.set_ylabel("mean resultant length")
    ax.set_title("theta-band locking by state,\nequal spike counts per unit",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def fig_sessions(df, fname="fig08_across_sessions.png"):
    """Per-session replication of the population result."""
    run = df[(df.state == "run") & (df.n_spikes >= MIN_SPIKES)]
    sessions = sorted(run.session.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    pos = np.arange(len(sessions))
    for i, s in enumerate(sessions):
        for j, ct in enumerate(["excitatory", "inhibitory"]):
            v = run[(run.session == s) & (run.cell_type == ct)].mrl.values
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), i + (j - 0.5) * 0.25)
                       + np.random.default_rng(i).normal(0, 0.03, len(v)),
                       v, s=10, alpha=0.6, color=CT_COLOR[ct],
                       label=ct if i == 0 else None)
            ax.plot([i + (j - 0.5) * 0.25 - 0.09, i + (j - 0.5) * 0.25 + 0.09],
                    [np.median(v)] * 2, color="k", lw=2)
    ax.set_xticks(pos)
    ax.set_xticklabels([s.split("_")[0] + "\n" + s.split("-")[-1]
                        for s in sessions], fontsize=8)
    ax.set_ylabel("mean resultant length (running)")
    ax.set_title("per-session locking strength (bars = medians)")
    ax.legend(fontsize=8)

    ax = axes[1]
    fr = [(run[run.session == s].shuffle_p < 0.05).mean() * 100 for s in sessions]
    ax.bar(pos, fr, color="C2")
    for i, s in enumerate(sessions):
        n = run[run.session == s]
        ax.text(i, fr[i] + 1.5, f"{int((n.shuffle_p<0.05).sum())}/{len(n)}",
                ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels([s.split("_")[0] + "\n" + s.split("-")[-1]
                        for s in sessions], fontsize=8)
    ax.axhline(5, color="r", ls="--", lw=1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% units significantly locked")
    ax.set_title("replication across sessions and animals")

    axp = plt.subplot(1, 3, 3, projection="polar")
    axes[2].remove()
    for s in sessions:
        sig = run[(run.session == s) & (run.shuffle_p < 0.05)]
        cnt, _ = np.histogram(sig.pref_phase, bins=BINS, density=True)
        axp.plot(np.append(CENTERS, CENTERS[0]), np.append(cnt, cnt[0]), lw=1.5,
                 label=f"{s.split('_')[0]} {s.split('-')[-1]}")
    axp.set_title("preferred-phase distribution per session", fontsize=10, pad=24)
    axp.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.32, -0.12))
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname
