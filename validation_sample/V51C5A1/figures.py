"""All figures for the auditory frequency-tuning analysis of DANDI:000986."""

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tuning as T

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "legend.frameon": False,
})

FREQ_COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(T.FREQS)))
FREQ_LABELS = [f"{int(f / 1000)} kHz" for f in T.FREQS]
EDGES = T.psth_edges()
CTR_MS = (EDGES[:-1] + EDGES[1:]) / 2 * 1000


def load():
    with open("results.pkl", "rb") as f:
        return pickle.load(f)


def example(results):
    return next(r for r in results if "evoked" in r)


def _freq_legend(ax, **kw):
    handles = [plt.Line2D([], [], color=c, lw=2, label=l)
               for c, l in zip(FREQ_COLORS, FREQ_LABELS)]
    ax.legend(handles=handles, title="tone frequency", **kw)


# ---------------------------------------------------------------- fig 1: raw data
def fig_raw_data(ex, raw):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), height_ratios=[2.2, 1, 1.3])
    onsets, freqs = ex["onsets"], ex["trial_freqs"]

    t0 = onsets[200]
    t1 = t0 + 15
    sel = (onsets >= t0) & (onsets < t1)

    ax = axes[0]
    for u, (uid, st) in enumerate(raw["spike_times"].items()):
        s = st[(st >= t0) & (st < t1)]
        ax.plot(s - t0, np.full_like(s, u), "|", color="0.25", ms=2.2, mew=0.5)
    for o, f in zip(onsets[sel], freqs[sel]):
        ax.axvline(o - t0, color=FREQ_COLORS[list(T.FREQS).index(f)], lw=1.4, alpha=0.75,
                   zorder=0)
    ax.set(xlim=(0, 15), ylabel="unit", xlabel="",
           title=f"a  Raw spiking, {ex['path']} ({len(raw['spike_times'])} units); "
                 "vertical lines are tone onsets, coloured by frequency")
    _freq_legend(ax, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=7.5,
                 title_fontsize=8)

    ax = axes[1]
    pop = raw["pop_rate"]
    m = (pop.t >= t0) & (pop.t < t1)
    ax.plot(pop.t[m] - t0, np.asarray(pop)[m], color="C3", lw=0.9)
    ax.set(xlim=(0, 15), ylabel="pop. rate (Hz)", xlabel="time (s)",
           title="b  Population firing rate over the same window")

    ax = axes[2]
    pup = raw["pupil"]
    ax.plot(pup.t / 60, np.asarray(pup), color="C0", lw=0.4)
    ax.set(ylabel="pupil diameter (a.u.)", xlabel="time in session (min)",
           title="c  Pupil diameter across the whole session; "
                 "shaded bands are the spontaneous (no-tone) blocks")
    for s, e in zip(raw["spont"].start, raw["spont"].end):
        ax.axvspan(s / 60, e / 60, color="0.85", zorder=0)

    fig.tight_layout()
    fig.savefig("fig01_raw_data.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 2: stimulus design + alignment
def fig_design_and_alignment(results, ex):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    onsets, freqs = ex["onsets"], ex["trial_freqs"]

    ax = axes[0]
    n = 100
    k_seq = np.searchsorted(T.FREQS, freqs[:n])
    ax.scatter(np.arange(n), k_seq, c=FREQ_COLORS[k_seq], s=26, marker="s")
    ax.set(yticks=range(len(T.FREQS)),
           yticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="trial number", ylabel="tone frequency (kHz)", ylim=(-0.6, 4.6),
           title=f"a  Tones are randomly interleaved\n"
                 f"({len(freqs)} trials, "
                 f"{np.bincount(np.searchsorted(T.FREQS, freqs)).min()}-"
                 f"{np.bincount(np.searchsorted(T.FREQS, freqs)).max()} per frequency)")

    ax = axes[1]
    grand = ex["psth"].mean(axis=(0, 1))
    ax.plot(CTR_MS, grand, color="k", lw=1.8, zorder=4)
    ax.axvspan(0, 25, color="0.7", zorder=0)
    ax.axvspan(*np.array(T.BASELINE_WIN) * 1000, color="C0", alpha=0.18, zorder=0)
    ax.axvspan(*np.array(T.EVOKED_WIN) * 1000, color="C1", alpha=0.25, zorder=0)
    ax.set_ylim(grand.min() - 0.15, grand.max() + 1.0)
    ax.annotate("25 ms tone", xy=(12, grand.max() + 0.1), xytext=(120, grand.max() + 0.75),
                fontsize=7.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.text(-50, grand.min() + 0.05, "baseline\nwindow", ha="center", va="bottom",
            fontsize=7.5, color="C0")
    ax.text(70, grand.max() * 0.72, "evoked\nwindow", ha="left", va="top", fontsize=7.5,
            color="C1")
    ax.set(xlabel="time from tone onset (ms)", ylabel="firing rate (Hz)",
           title="b  Grand-average response\nconfirms onset alignment")

    ax = axes[2]
    lat = []
    for r in results:
        p = r["psth"].mean(axis=1)  # (units, bins), averaged over frequency
        base = p[:, :20].mean(axis=1, keepdims=True)
        sd = p[:, :20].std(axis=1, keepdims=True)
        post = p[:, 20:40]  # 0-100 ms
        above = post > base + 3 * sd
        for row in above:
            hit = np.flatnonzero(row)
            if hit.size:
                lat.append(CTR_MS[20 + hit[0]])
    ax.hist(lat, bins=np.arange(0, 102, 5), color="0.4")
    ax.axvline(np.median(lat), color="C3", lw=1.5)
    ax.set(xlabel="response latency (ms)", ylabel="units",
           title=f"c  Onset latency across sessions\n(median {np.median(lat):.0f} ms, "
                 f"n = {len(lat)})")
    fig.tight_layout()
    fig.savefig("fig02_design_and_alignment.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- fig 3: example units
def fig_example_units(ex, raw, n_show=5):
    st = ex
    tuned = st["reject_tuned"]
    # pick well-tuned units spanning as many different best frequencies as possible
    picks = []
    order = np.argsort(-st["omega2"])
    for bf in range(len(T.FREQS)):
        cand = [u for u in order if tuned[u] and st["bf_idx"][u] == bf]
        if cand:
            picks.append(cand[0])
    picks = picks[:n_show] if len(picks) >= n_show else list(order[:n_show])
    picks = sorted(picks, key=lambda u: st["bf_idx"][u])

    # spike times were only cached for the first stretch of the session
    t_end = raw["t_start"] + raw["raw_span"]
    in_raw = ex["onsets"] < t_end - 0.5
    onsets, freqs = ex["onsets"], ex["trial_freqs"]
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 8.2),
                             height_ratios=[1.5, 1, 1])
    for j, u in enumerate(picks):
        uid = st["unit_ids"][u]
        stimes = raw["spike_times"][uid]

        # raster, trials grouped by frequency
        ax = axes[0, j]
        y = 0
        for k, f in enumerate(T.FREQS):
            idx = np.flatnonzero((freqs == f) & in_raw)[:120]
            for o in idx:
                s = stimes[(stimes >= onsets[o] - 0.1) & (stimes < onsets[o] + 0.3)]
                ax.plot((s - onsets[o]) * 1000, np.full_like(s, y), "|",
                        color=FREQ_COLORS[k], ms=2.4, mew=0.55)
                y += 1
            ax.axhline(y, color="0.8", lw=0.5)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set(xlim=(-100, 300), ylim=(0, y), xticklabels=[],
               title=f"unit {uid}  (BF {FREQ_LABELS[st['bf_idx'][u]]})")
        if j == 0:
            ax.set_ylabel("trials, grouped by frequency")

        ax = axes[1, j]
        for k in range(len(T.FREQS)):
            ax.plot(CTR_MS, st["psth"][u, k], color=FREQ_COLORS[k], lw=1.3)
        ax.axvspan(0, 25, color="0.88", zorder=0)
        ax.set(xlim=(-100, 300), xlabel="time from onset (ms)")
        if j == 0:
            ax.set_ylabel("firing rate (Hz)")

        ax = axes[2, j]
        ax.errorbar(np.arange(len(T.FREQS)), st["tc_delta"][u], yerr=st["tc_sem"][u],
                    color="k", marker="o", ms=5, lw=1.5, capsize=3)
        for k in range(len(T.FREQS)):
            ax.plot(k, st["tc_delta"][u][k], "o", ms=6, color=FREQ_COLORS[k], zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set(xticks=range(len(T.FREQS)),
               xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
               xlabel="frequency (kHz)")
        p = st["p_tuning"][u]
        ptxt = "$p$ < 1e-300" if p == 0 else f"$p$ = {p:.0e}"
        ax.set_title(f"{ptxt}, $\\omega^2$ = {st['omega2'][u]:.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel("evoked rate (Hz)\nbaseline-subtracted")

    axes[1, -1].legend([plt.Line2D([], [], color=c, lw=1.6) for c in FREQ_COLORS],
                       FREQ_LABELS, loc="upper right", fontsize=7)
    fig.suptitle("Single auditory-cortex units are tuned to tone frequency: "
                 "rasters, PSTHs and tuning curves", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig("fig03_example_units.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- fig 4: population tuning
def fig_population_tuning(results):
    tc = np.concatenate([r["tc_delta"][r["reject_tuned"]] for r in results])
    bf = np.concatenate([r["bf_idx"][r["reject_tuned"]] for r in results])
    norm = tc / np.abs(tc).max(axis=1, keepdims=True)

    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42, width_ratios=[1.0, 1.05, 1.05],
                          left=0.07, right=0.97, top=0.9, bottom=0.1)

    ax = fig.add_subplot(gs[:, 0])
    order = np.lexsort((-norm.max(axis=1), bf))
    im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest",
                   extent=(-0.5, len(T.FREQS) - 0.5, len(bf), 0))
    ax.set(xticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="tone frequency (kHz)", ylabel="frequency-tuned units, sorted by BF",
           title=f"a  Tuning curves of all {len(bf)} tuned units\n"
                 "(15 sessions, normalised per unit)")
    cb = fig.colorbar(im, ax=ax, location="bottom", fraction=0.055, pad=0.16,
                      aspect=28)
    cb.set_label("normalised evoked rate", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for k in range(len(T.FREQS)):
        g = norm[bf == k]
        if len(g) < 3:
            continue
        ax.errorbar(np.arange(len(T.FREQS)), g.mean(axis=0),
                    yerr=g.std(axis=0) / np.sqrt(len(g)), color=FREQ_COLORS[k],
                    marker="o", ms=4, lw=1.5, capsize=2, label=f"BF {FREQ_LABELS[k]}")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set(xticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="tone frequency (kHz)", ylabel="normalised evoked rate",
           title="b  Mean tuning curve by best frequency", ylim=(-0.85, 1.45))
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8,
              handletextpad=0.4)

    ax = fig.add_subplot(gs[0, 2])
    counts = np.bincount(bf, minlength=len(T.FREQS))
    ax.bar(range(len(T.FREQS)), 100 * counts / counts.sum(), color=FREQ_COLORS)
    ax.set(xticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="best frequency (kHz)", ylabel="% of tuned units",
           title="c  Best-frequency distribution")

    ax = fig.add_subplot(gs[1, 1])
    frac_t = [100 * r["reject_tuned"].mean() for r in results]
    frac_r = [100 * r["reject_responsive"].mean() for r in results]
    x = np.arange(len(results))
    ax.bar(x - 0.2, frac_r, 0.4, color="0.65", label="sound-responsive")
    ax.bar(x + 0.2, frac_t, 0.4, color="C3", label="frequency-tuned")
    ax.set(xticks=x, xticklabels=[r["path"].split("/")[1].replace("_behavior.nwb", "")
                                  .replace("sub-", "") for r in results],
           ylabel="% of units", ylim=(0, 128),
           title="d  Per session (FDR $q$ < 0.05)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, columnspacing=0.9)

    ax = fig.add_subplot(gs[1, 2])
    for r in results:
        c = np.bincount(r["bf_idx"][r["reject_tuned"]], minlength=len(T.FREQS))
        ax.plot(range(len(T.FREQS)), 100 * c / max(c.sum(), 1), color="0.7", lw=0.8)
    ax.plot(range(len(T.FREQS)), 100 * counts / counts.sum(), color="C3", lw=2.5,
            marker="o", label="pooled")
    ax.set(xticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="best frequency (kHz)", ylabel="% of tuned units",
           title="e  BF distribution per session\n(grey) and pooled (red)")
    ax.legend(fontsize=7)

    fig.savefig("fig04_population_tuning.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- fig 5: statistics
def fig_statistics(results):
    p_tun = np.concatenate([r["p_tuning"] for r in results])
    om = np.concatenate([r["omega2"] for r in results])
    tuned = np.concatenate([r["reject_tuned"] for r in results])
    rel = np.concatenate([r["reliability"] for r in results])
    rel_s = np.concatenate([r["reliability_shuffled"] for r in results])
    sel = np.concatenate([r["selectivity"] for r in results])
    ev = np.concatenate([r["tc_evoked"].max(axis=1) for r in results])
    bl = np.concatenate([r["baseline_rate"] for r in results])

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))

    ax = axes[0]
    ax.hist(np.log10(np.clip(p_tun, 1e-300, 1)), bins=60, color="0.45")
    ax.axvline(np.log10(0.05), color="C3", lw=1.5, ls="--")
    ax.set(xlabel="log$_{10}$ $p$, Kruskal-Wallis across frequencies\n"
                  "(clipped at $10^{-300}$)", ylabel="units",
           title=f"a  Frequency effect per unit\n{tuned.sum()}/{tuned.size} "
                 f"({tuned.mean():.0%}) tuned at FDR $q$<0.05")

    ax = axes[1]
    bins = np.linspace(-1, 1, 45)
    ax.hist(rel_s, bins=bins, color="0.75", label="frequency labels shuffled")
    ax.hist(rel, bins=bins, color="C3", alpha=0.75, label="observed")
    ax.set(xlabel="split-half tuning-curve correlation", ylabel="units",
           title=f"b  Tuning reproduces on held-out trials\n"
                 f"median $r$ = {np.median(rel):.2f} vs {np.median(rel_s):.2f} shuffled")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    ax.scatter(om[~tuned], sel[~tuned], s=5, color="0.75", label="not tuned")
    ax.scatter(om[tuned], sel[tuned], s=5, color="C3", label="tuned")
    ax.set(xlabel="$\\omega^2$ (variance explained by frequency)",
           ylabel="selectivity  $(r_{max}-r_{min})/r_{max}$", xscale="log",
           title="c  Effect size and selectivity")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[3]
    lim = (0.05, max(ev.max(), bl.max()) * 1.3)
    ax.scatter(bl[~tuned], ev[~tuned], s=5, color="0.75")
    ax.scatter(bl[tuned], ev[tuned], s=5, color="C3")
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="baseline rate (Hz)", ylabel="evoked rate at BF (Hz)",
           title="d  Tones drive units above baseline")
    fig.tight_layout()
    fig.savefig("fig05_statistics.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- fig 6: decoding
def fig_decoding(results, ex):
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5))

    ax = axes[0]
    im = ax.imshow(ex["decode_conf"], cmap="magma", vmin=0, vmax=1)
    for i in range(len(T.FREQS)):
        for j in range(len(T.FREQS)):
            v = ex["decode_conf"][i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="w" if v < 0.6 else "k")
    ax.set(xticks=range(len(T.FREQS)), yticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           yticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="decoded frequency (kHz)", ylabel="true frequency (kHz)",
           title=f"a  Single-trial decoding, example session\n"
                 f"accuracy {ex['decode_acc']:.2f} (chance 0.20)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

    ax = axes[1]
    for r in results:
        s = sorted(r["decode_vs_size"])
        ax.plot(s, [r["decode_vs_size"][k][0] for k in s], color="0.75", lw=0.8)
    allsz = sorted({k for r in results for k in r["decode_vs_size"]})
    mean = [np.mean([r["decode_vs_size"][k][0] for r in results
                     if k in r["decode_vs_size"]]) for k in allsz]
    ax.plot(allsz, mean, color="C3", lw=2.2, marker="o", label="mean over sessions")
    ax.axhline(0.2, color="k", ls="--", lw=1, label="chance")
    ax.set(xscale="log", xlabel="number of units in decoder", ylabel="accuracy",
           ylim=(0, 1), title="b  Decoding improves with population size")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    x = np.arange(len(results))
    ax.bar(x, [r["decode_acc"] for r in results], color="C3", label="observed")
    ax.bar(x, [r["decode_acc_shuffled"] for r in results], color="0.6",
           label="frequency labels shuffled")
    ax.axhline(0.2, color="k", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=[r["path"].split("/")[1].replace("_behavior.nwb", "")
                                  .replace("sub-", "") for r in results],
           ylabel="decoding accuracy", ylim=(0, 1.22),
           title="c  Every session decodes far above chance")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, columnspacing=0.9)

    ax = axes[3]
    err = np.zeros(len(T.FREQS))
    for r in results:
        c = r["decode_conf"]
        for i in range(len(T.FREQS)):
            for j in range(len(T.FREQS)):
                err[abs(i - j)] += c[i, j]
    err /= err.sum()
    ax.bar(range(len(T.FREQS)), 100 * err, color="0.45")
    ax.set(xlabel="|true - decoded| (octaves)", ylabel="% of trials",
           title="d  Errors fall on neighbouring octaves,\nas expected from "
                 "overlapping tuning")
    fig.tight_layout()
    fig.savefig("fig06_decoding.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- fig 7: GLM
def fig_glm(ex):
    with open("glm_results.pkl", "rb") as f:
        g = pickle.load(f)
    tms = g["kernel_time"] * 1000
    d_ll = (g["ll_full"] - g["ll_reduced"]) / g["bin_size"]

    # align GLM units to the statistics table
    idx = {uid: i for i, uid in enumerate(ex["unit_ids"])}
    keep = np.array([uid in idx for uid in g["unit_ids"]])
    rows = np.array([idx[uid] for uid in g["unit_ids"][keep]])

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.34)

    order = np.argsort(-d_ll)
    for n, u in enumerate(order[:4]):
        ax = fig.add_subplot(gs[0, n])
        for k in range(len(T.FREQS)):
            ax.plot(tms, g["filters"][u, k], color=FREQ_COLORS[k], lw=1.4)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set(xlabel="time from tone onset (ms)",
               title=f"unit {g['unit_ids'][u]}")
        if n == 0:
            ax.set_ylabel("GLM gain (log rate)")
            ax.legend([plt.Line2D([], [], color=c, lw=1.5) for c in FREQ_COLORS],
                      FREQ_LABELS, fontsize=6.5, loc="upper right")
    fig.text(0.5, 0.97, "a  Frequency-specific GLM response kernels "
             "(four units with the largest held-out improvement)",
             ha="center", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(d_ll, bins=45, color="0.45")
    ax.axvline(0, color="C3", lw=1.5, ls="--")
    ax.set(xlabel="$\\Delta$ held-out log-likelihood (nats/s)", ylabel="units",
           title=f"b  Frequency-specific model beats\n"
                 f"tone-only model in {np.mean(d_ll > 0):.0%} of units")

    ax = fig.add_subplot(gs[1, 1])
    peak = g["filters"][:, :, 2:12].mean(axis=2)  # 10-60 ms of each kernel
    emp = ex["tc_delta"][rows]
    glm_k = peak[keep]
    r = np.corrcoef(emp.ravel(), glm_k.ravel())[0, 1]
    ax.scatter(emp.ravel(), glm_k.ravel(), s=4, color="0.4", alpha=0.5)
    ax.set(xlabel="empirical evoked rate (Hz)", ylabel="GLM kernel, 10-60 ms",
           title=f"c  GLM recovers the measured\ntuning curves ($r$ = {r:.2f})")

    ax = fig.add_subplot(gs[1, 2])
    agree = np.mean(np.argmax(glm_k, axis=1) == ex["bf_idx"][rows])
    tuned_rows = ex["reject_tuned"][rows]
    agree_t = np.mean(np.argmax(glm_k[tuned_rows], axis=1)
                      == ex["bf_idx"][rows][tuned_rows])
    ax.bar([0, 1], [100 * agree, 100 * agree_t], color=["0.6", "C3"], width=0.6)
    ax.axhline(20, color="k", ls="--", lw=1)
    ax.set(xticks=[0, 1], xticklabels=["all units", "tuned units"],
           ylabel="% BF agreement with PSTH", ylim=(0, 105),
           title="d  Best frequency agrees between\nGLM and direct measurement")

    ax = fig.add_subplot(gs[1, 3])
    sc = ax.scatter(ex["omega2"][rows], d_ll[keep], s=6,
                    c=ex["reject_tuned"][rows], cmap=mpl.colors.ListedColormap(
                        ["0.75", "C3"]))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel="$\\omega^2$ from the PSTH analysis",
           ylabel="$\\Delta$ held-out LL (nats/s)",
           title="e  The two analyses agree on\nwhich units are tuned")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="0.75",
                                  label="not tuned"),
                       plt.Line2D([], [], marker="o", ls="", color="C3",
                                  label="tuned")], fontsize=7, loc="upper left")
    fig.savefig("fig07_glm.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- fig 8: arousal control
def fig_arousal(arousal):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    lo, hi = arousal["tc_low"], arousal["tc_high"]
    tuned = arousal["tuned"]

    ax = axes[0]
    ax.hist(arousal["pupil_at_onset"], bins=60, color="0.5")
    ax.axvline(arousal["median_pupil"], color="C3", lw=1.5)
    ax.set(xlabel="pupil diameter at tone onset (a.u.)", ylabel="trials",
           title="a  Median split on pre-tone pupil")

    ax = axes[1]
    for k in range(len(T.FREQS)):
        g = tuned & (arousal["bf_idx"] == k)
        if g.sum() < 3:
            continue
        n_lo = lo[g] / np.abs(lo[g]).max(axis=1, keepdims=True)
        n_hi = hi[g] / np.abs(hi[g]).max(axis=1, keepdims=True)
        ax.plot(range(len(T.FREQS)), n_lo.mean(axis=0), color=FREQ_COLORS[k], lw=1.5,
                ls="--")
        ax.plot(range(len(T.FREQS)), n_hi.mean(axis=0), color=FREQ_COLORS[k], lw=1.5)
    ax.set(xticks=range(len(T.FREQS)),
           xticklabels=[f"{int(f / 1000)}" for f in T.FREQS],
           xlabel="tone frequency (kHz)", ylabel="normalised evoked rate",
           ylim=(-0.75, 1.35),
           title="b  Tuning shape by arousal state\n(dashed low pupil, solid high)")
    ax.legend([plt.Line2D([], [], color=c, lw=1.5) for c in FREQ_COLORS],
              [f"BF {l}" for l in FREQ_LABELS], fontsize=6.5, ncol=3,
              loc="upper center", columnspacing=0.8, handletextpad=0.4)

    ax = axes[2]
    agree = np.mean(np.argmax(lo[tuned], axis=1) == np.argmax(hi[tuned], axis=1))
    r = np.corrcoef(lo[tuned].ravel(), hi[tuned].ravel())[0, 1]
    ax.scatter(lo[tuned].ravel(), hi[tuned].ravel(), s=5, color="0.4", alpha=0.5)
    lim = np.array([min(lo[tuned].min(), hi[tuned].min()),
                    max(lo[tuned].max(), hi[tuned].max())])
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.set(xlabel="evoked rate, low pupil (Hz)", ylabel="evoked rate, high pupil (Hz)",
           title=f"c  Tuning is preserved across arousal\n$r$ = {r:.2f}, "
                 f"same BF in {agree:.0%} of tuned units")
    fig.tight_layout()
    fig.savefig("fig08_arousal_control.png", bbox_inches="tight")
    plt.close(fig)
