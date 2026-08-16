"""Figures for the auditory frequency-tuning analysis.

Run after run_anf_batch.py and run_cortex_batch.py have cached their results.
"""

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.gridspec import GridSpec

import anf_analysis as anf
import cortex_analysis as cx
import dandi_io as dio

nap.nap_config.suppress_conversion_warnings = True
mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

FIG = "figures/"
FCOL = plt.cm.viridis(np.linspace(0.05, 0.9, 5))   # colour per cortical tone frequency


def log_freq_axis(ax, f_khz, n=4):
    """Log x axis with a few readable frequency ticks and no minor-tick clutter."""
    ax.set_xscale("log")
    ticks = np.geomspace(f_khz[0], f_khz[-1], n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.3g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


# --------------------------------------------------------------------------------------
# raw data validation
# --------------------------------------------------------------------------------------
def fig_raw_cortex(sess, spikes, seconds=8.0, path=FIG + "fig01_raw_data_cortex.png"):
    """Spike raster, tone times and simultaneous behaviour for one stretch of a session."""
    onset, freq = sess["tone_onset"], sess["frequency"]
    freqs = np.unique(freq)
    t0 = onset[len(onset) // 2]
    win = (t0 - 0.5, t0 - 0.5 + seconds)
    beh = dio.read_behavior_window(sess["file"], win[0], win[1])

    fig = plt.figure(figsize=(9, 6.2))
    gs = GridSpec(4, 1, height_ratios=[3.2, 1, 0.9, 0.9], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ep = nap.IntervalSet(start=win[0], end=win[1])
    for row, u in enumerate(spikes.index):
        t = spikes[u].restrict(ep).t
        ax.plot(t, np.full(len(t), row), "|", ms=2.2, color="0.25", mew=0.5)
    m = (onset >= win[0]) & (onset <= win[1])
    for o, fq in zip(onset[m], freq[m]):
        ax.axvspan(o, o + 0.025, color=FCOL[list(freqs).index(fq)], alpha=0.45, lw=0)
    ax.set_ylabel("unit")
    ax.set_title(f"DANDI:000986  {sess['subject']} session {sess['session']} - "
                 f"{len(spikes)} units, 25 ms tones (shaded, colour = frequency)")
    ax.set_xlim(win)

    axr = fig.add_subplot(gs[1], sharex=ax)
    rate = spikes.count(0.01, ep=ep).sum(axis=1) / (0.01 * len(spikes))
    axr.plot(rate.t, rate.values, color="C3", lw=0.8)
    axr.set_ylabel("pop. rate\n(spikes/s)")

    axp = fig.add_subplot(gs[2], sharex=ax)
    axp.plot(*beh["pupil"], color="C0", lw=1)
    axp.set_ylabel("pupil\n(norm.)")

    axs = fig.add_subplot(gs[3], sharex=ax)
    axs.plot(*beh["running"], color="C2", lw=1)
    axs.set_ylabel("running\n(cm/s)")
    axs.set_xlabel("time (s)")
    for a in (ax, axr, axp):
        a.tick_params(labelbottom=False)

    handles = [mpl.patches.Patch(color=FCOL[i], label=f"{f/1000:g} kHz")
               for i, f in enumerate(freqs)]
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=False, title="tone", title_fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_anf_sweep(fib, path=FIG + "fig02_raw_data_anf.png"):
    """Raster and PSTH of one fibre's sweeps, used to place the tone-response window."""
    spikes, _, _ = anf.fibre_to_pynapple(fib)
    starts = np.arange(len(fib["sweeps"])) * anf.SLOT
    lv, fq = fib["level"], fib["frequency"]
    cf_guess = np.unique(fq)[len(np.unique(fq)) // 2]
    on_cf = fq == cf_guess

    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.25,
                                          "wspace": 0.28})
    for col, (sel, title, sort_by, fmt) in enumerate([
            (on_cf, f"tones at {cf_guess/1000:g} kHz, all levels", lv, "{:.0f} dB"),
            (lv >= lv.max() - 5, f"tones at {lv.max():.0f} dB, all frequencies", fq,
             "{:.1f} kHz")]):
        key = sort_by[sel]
        order = np.argsort(key, kind="stable")
        idx = np.where(sel)[0][order]
        ax = axes[0, col]
        for row, i in enumerate(idx):
            t = fib["sweeps"][i]
            ax.plot(t * 1e3, np.full(len(t), row), "|", ms=2.5, color="0.2", mew=0.6)
        ax.axvspan(anf.TONE_WIN[0] * 1e3, anf.TONE_WIN[1] * 1e3, color="C1", alpha=0.15, lw=0)
        ax.set_title(title, fontsize=9)
        # label the raster rows with the stimulus value each block of sweeps came from
        vals = key[order]
        uniq = np.unique(vals)
        pos = [np.mean(np.where(vals == v)[0]) for v in uniq]
        step = max(1, len(uniq) // 8)
        ax.set_yticks(pos[::step])
        ax.set_yticklabels([fmt.format(v / (1e3 if col else 1)) for v in uniq[::step]],
                           fontsize=7)
        ax.set_ylabel("tone level" if col == 0 else "tone frequency", fontsize=8.5)

        axp = axes[1, col]
        # the PSTH pools sweeps, so pass the reference times in temporal order
        peri = nap.compute_perievent(spikes, nap.Ts(np.sort(starts[idx])), window=(0, 0.2))
        c = peri.count(0.002)
        axp.plot(c.t * 1e3, np.asarray(c.values).sum(1) / (len(idx) * 0.002), color="C0", lw=1)
        axp.axvspan(anf.TONE_WIN[0] * 1e3, anf.TONE_WIN[1] * 1e3, color="C1", alpha=0.15, lw=0)
        axp.set_xlabel("time in sweep (ms)")
        axp.set_ylabel("rate (spikes/s)" if col == 0 else "")
    fig.suptitle(f"DANDI:001262  auditory-nerve fibre {fib['fibre']} "
                 f"(shaded: 50 ms tone burst)", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# auditory nerve
# --------------------------------------------------------------------------------------
def fig_anf_examples(results, n=3, path=FIG + "fig03_anf_response_areas.png"):
    """Frequency response area and threshold curve for a few example fibres."""
    good = [r for r in results if np.isfinite(r.get("q10", np.nan))]
    good.sort(key=lambda r: r["cf"])
    picks = [good[int(q * (len(good) - 1))] for q in np.linspace(0.05, 0.95, n)]

    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.3})
    for k, r in enumerate(picks):
        f_khz, lv = r["freqs"] / 1e3, r["levels"]
        ax = axes[0, k]
        im = ax.pcolormesh(f_khz, lv, r["fra"] - r["spont"], cmap="magma", shading="nearest")
        ax.plot(f_khz, r["thresholds"], "o", color="c", ms=2.5, alpha=0.6)
        ax.plot(f_khz, r["thresholds_smooth"], "-", color="c", lw=1.6, label="threshold")
        ax.plot(r["cf"] / 1e3, r["cf_threshold"], "*", color="w", ms=13, label="CF")
        log_freq_axis(ax, f_khz)
        ax.set_title(f"{r['fibre'].split(':')[-1]}\nCF {r['cf']/1e3:.2f} kHz, "
                     f"{r['cf_threshold']:.0f} dB, Q10 {r['q10']:.2f}", fontsize=8.5)
        ax.set_xlabel("frequency (kHz)")
        if k == 0:
            ax.set_ylabel("level (dB SPL)")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        plt.colorbar(im, ax=ax, label="driven rate (spikes/s)" if k == n - 1 else "")

        axl = axes[1, k]
        for i, level in enumerate(lv):
            if i % 3:
                continue
            axl.plot(f_khz, r["fra"][i], lw=1.2,
                     color=plt.cm.plasma(i / max(1, len(lv) - 1)), label=f"{level:.0f} dB")
        axl.axhline(r["spont"], color="0.5", ls="--", lw=1)
        log_freq_axis(axl, f_khz)
        axl.set_xlabel("frequency (kHz)")
        if k == 0:
            axl.set_ylabel("rate (spikes/s)")
        axl.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.suptitle("Single auditory-nerve fibres: frequency response areas (top) and "
                 "iso-level tuning (bottom)", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_anf_iso_examples(results, n=8, path=FIG + "fig04_anf_tuning_curves.png"):
    """Fixed-level frequency tuning curves for example fibres, ordered by best frequency."""
    sel = [r for r in results if np.isfinite(r.get("iso_bf", np.nan)) and r["iso_p"] < 0.01]
    sel.sort(key=lambda r: r["iso_bf"])
    picks = [sel[int(q * (len(sel) - 1))] for q in np.linspace(0, 1, n)]
    fig, axes = plt.subplots(2, n // 2, figsize=(2.2 * (n // 2), 4.8),
                             gridspec_kw={"hspace": 0.6, "wspace": 0.42})
    for ax, r in zip(axes.ravel(), picks):
        ax.plot(r["iso_freqs"] / 1e3, r["iso_tuning"], "-o", ms=3, lw=1.2, color="C0")
        ax.axhline(r["iso_spont"], color="0.5", ls="--", lw=1)
        ax.axvline(r["iso_bf"] / 1e3, color="C3", ls=":", lw=1)
        bw = r["iso_bw_octaves"]
        ax.set_title(f"{r['fibre'].split(':')[-1]}\nBF {r['iso_bf']/1e3:.2f} kHz" +
                     (f", {bw:.2f} oct" if np.isfinite(bw) else ""), fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate (spikes/s)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("frequency (kHz)", fontsize=8)
    fig.suptitle("Auditory-nerve fibres: fixed-level frequency tuning "
                 "(dashed = spontaneous rate, dotted = best frequency)", y=1.03)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_anf_population(results, path=FIG + "fig05_anf_population.png"):
    """Threshold curves, CF and BF distributions, thresholds and sharpness across fibres."""
    tuned = [r for r in results if np.isfinite(r.get("cf", np.nan))]
    q = [r for r in tuned if np.isfinite(r["q10"])]
    iso = [r for r in results if np.isfinite(r.get("iso_bf", np.nan))]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.3})
    ax = axes[0, 0]
    # each curve is plotted relative to its own CF, which is how the common V shape shows
    for r in tuned:
        ok = np.isfinite(r["thresholds"])
        ax.plot(r["freqs"][ok] / r["cf"], r["thresholds"][ok] - r["cf_threshold"],
                lw=0.8, alpha=0.45,
                color=plt.cm.viridis(np.clip(np.log10(r["cf"] / 300) / np.log10(50), 0, 1)))
    ax.set_xscale("log")
    ax.set_xlim(0.25, 4)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(-2, 45)
    ax.axhline(10, color="k", ls=":", lw=1)
    ax.set_xlabel("frequency / CF")
    ax.set_ylabel("threshold re. CF threshold (dB)")
    ax.set_title(f"threshold curves aligned on CF ({len(tuned)} fibres)", fontsize=9)

    cfs = np.array([r["cf"] for r in tuned]) / 1e3
    ax = axes[0, 1]
    ax.plot(cfs, [r["cf_threshold"] for r in tuned], "o", ms=5, alpha=0.8, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("characteristic frequency (kHz)")
    ax.set_ylabel("threshold at CF (dB SPL)")
    ax.set_title("tuning-curve tips trace the audiogram", fontsize=9)

    ax = axes[1, 0]
    bfs = np.array([r["iso_bf"] for r in iso]) / 1e3
    ax.hist(bfs, bins=np.logspace(np.log10(bfs.min()), np.log10(bfs.max()), 22),
            color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("fibres")
    ax.set_title(f"best frequency, fixed-level sweeps (n={len(iso)})", fontsize=9)

    ax = axes[1, 1]
    bw = np.array([r["iso_bw_octaves"] for r in iso])
    ok = np.isfinite(bw)
    ax.plot(bfs[ok], bw[ok], "o", ms=4, alpha=0.7, color="C2")
    ax.set_xscale("log")
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("half-max width (octaves)")
    ax.set_title(f"iso-level tuning width (n={ok.sum()} measurable)", fontsize=9)

    ax = axes[1, 2]
    ax.hist(bw[ok], bins=np.linspace(0, 3, 25), color="C2")
    ax.axvline(np.median(bw[ok]), color="k", ls="--", lw=1.2,
               label=f"median {np.median(bw[ok]):.2f} oct")
    ax.set_xlabel("half-max width (octaves)")
    ax.set_ylabel("fibres")
    ax.set_title("tuning width distribution", fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axes[0, 2]
    qx = np.array([r["cf"] for r in q]) / 1e3
    qy = np.array([r["q10"] for r in q])
    ax.plot(qx, qy, "o", ms=4, alpha=0.7, color="C3")
    b = np.polyfit(np.log10(qx), np.log10(qy), 1)
    xs = np.logspace(np.log10(qx.min()), np.log10(qx.max()), 20)
    ax.plot(xs, 10 ** np.polyval(b, np.log10(xs)), "k--", lw=1.2,
            label=f"slope {b[0]:.2f} (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("characteristic frequency (kHz)")
    ax.set_ylabel("$Q_{10}$ = CF / bandwidth")
    ax.set_title(f"sharpness of tuning ({len(q)} fibres)", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.suptitle("DANDI:001262 - frequency tuning of the gerbil auditory nerve", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# cortex
# --------------------------------------------------------------------------------------
def fig_cortex_example_unit(sess, spikes, unit, path=FIG + "fig06_cortex_example_unit.png"):
    """Raster and PSTH by frequency for one cortical unit."""
    onset, freq = sess["tone_onset"], sess["frequency"]
    freqs = np.unique(freq)
    fig, axes = plt.subplots(2, len(freqs), figsize=(2.2 * len(freqs), 5), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.22,
                                          "wspace": 0.18})
    ymax = 0
    psths = []
    for j, fq in enumerate(freqs):
        ons = onset[freq == fq][:150]
        peri = nap.compute_perievent(spikes[unit], nap.Ts(ons), window=(-0.05, 0.15))
        ax = axes[0, j]
        for row, k in enumerate(peri.index):
            t = peri[k].t
            ax.plot(t * 1e3, np.full(len(t), row), "|", ms=2, color="0.2", mew=0.5)
        ax.axvspan(0, 25, color=FCOL[j], alpha=0.3, lw=0)
        ax.set_title(f"{fq/1000:g} kHz", fontsize=9)
        if j == 0:
            ax.set_ylabel("trial")
        else:
            ax.tick_params(labelleft=False)

        allp = nap.compute_perievent(spikes[unit], nap.Ts(onset[freq == fq]),
                                     window=(-0.05, 0.15))
        c = allp.count(0.005)
        r = np.asarray(c.values).sum(1) / ((freq == fq).sum() * 0.005)
        psths.append((c.t, r))
        ymax = max(ymax, r.max())
    for j, (t, r) in enumerate(psths):
        ax = axes[1, j]
        ax.plot(t * 1e3, r, color=FCOL[j], lw=1.2)
        ax.axvspan(0, 25, color=FCOL[j], alpha=0.2, lw=0)
        ax.set_ylim(0, ymax * 1.1)
        ax.set_xlabel("time (ms)")
        if j == 0:
            ax.set_ylabel("rate (spikes/s)")
        else:
            ax.tick_params(labelleft=False)
    fig.suptitle(f"DANDI:000986  {sess['subject']} unit {unit}: tone responses by "
                 f"frequency (150 trials shown, PSTH from all)", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cortex_tuning_examples(units, freqs, n=8, path=FIG + "fig07_cortex_tuning_curves.png"):
    """Tuning curves of example units, one per panel, ordered by best frequency."""
    sel = [u for u in units if u["p_tuned"] < 0.01 and u["peak_driven"] > 5]
    sel.sort(key=lambda u: (u["bf"], -u["peak_driven"]))
    picks = [sel[int(q * (len(sel) - 1))] for q in np.linspace(0, 1, n)]
    fig, axes = plt.subplots(2, n // 2, figsize=(2.1 * (n // 2), 4.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.42})
    for ax, u in zip(axes.ravel(), picks):
        ax.errorbar(freqs / 1000, u["tuning"], yerr=u["tuning_sem"], fmt="-o", ms=4,
                    lw=1.3, color="C0", capsize=2)
        ax.axhline(u["baseline"], color="0.5", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(freqs / 1000)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_title(f"{u['subject']} u{u['unit']}\nBF {u['bf']/1000:g} kHz, "
                     f"sel {u['selectivity']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("evoked rate\n(spikes/s)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("frequency (kHz)", fontsize=8)
    fig.suptitle("Auditory cortex: single-unit frequency tuning "
                 "(mean +/- SEM, dashed = pre-tone baseline)", y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cortex_population(res, path=FIG + "fig08_cortex_population.png"):
    """Population tuning heatmap, BF distribution, selectivity and per-session summary."""
    units, freqs = res["units"], res["freqs"]
    tuned = [u for u in units if u["p_tuned"] < 0.01]
    D = np.array([u["driven"] for u in tuned])
    norm = D / np.maximum(np.abs(D).max(1, keepdims=True), 1e-9)
    order = np.lexsort((-norm.max(1), np.argmax(norm, axis=1)))

    fig = plt.figure(figsize=(11, 6.8))
    gs = GridSpec(2, 3, hspace=0.45, wspace=0.55, width_ratios=[1.35, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("tone frequency (kHz)")
    ax.set_ylabel("unit (sorted by best frequency)")
    ax.set_title(f"normalised tuning, {len(tuned)} tuned units", fontsize=9)
    plt.colorbar(im, ax=ax, label="driven rate / peak |driven rate|",
                 fraction=0.05, pad=0.03)

    ax = fig.add_subplot(gs[0, 1])
    bf = np.array([u["bf"] for u in tuned])
    counts = [np.sum(bf == f) for f in freqs]
    ax.bar(range(len(freqs)), counts, color=FCOL)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("units")
    ax.set_title("best-frequency distribution", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist([u["selectivity"] for u in tuned], bins=20, color="0.4")
    ax.set_xlabel("selectivity index")
    ax.set_ylabel("units")
    ax.set_title("frequency selectivity", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    frac = [np.mean([u["p_tuned"] < 0.01 for u in units if u["path"] == s["path"]])
            for s in res["sessions"]]
    ax.bar(range(len(frac)), frac, color="C0")
    ax.axhline(0.01, color="r", ls="--", lw=1, label="expected under $H_0$")
    ax.set_xlabel("session")
    ax.set_ylabel("fraction tuned")
    ax.set_ylim(0, 1)
    ax.set_title("frequency-tuned units per session", fontsize=9)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    pk = np.array([u["peak_driven"] for u in units])
    ax.hist(pk[pk > -20], bins=30, color="0.4")
    ax.set_xlabel("peak driven rate (spikes/s)")
    ax.set_ylabel("units")
    ax.set_title("response magnitude", fontsize=9)
    fig.suptitle("DANDI:000986 - frequency tuning across mouse auditory cortex "
                 f"({len(res['sessions'])} sessions, {len(units)} units)", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_decoding(res, path=FIG + "fig09_cortex_decoding.png"):
    """Confusion matrix and per-session accuracy for frequency decoding."""
    freqs = res["freqs"]
    conf = np.sum([s["confusion"] for s in res["sessions"]], axis=0)
    conf = conf / conf.sum(1, keepdims=True)
    acc = [s["accuracy"] for s in res["sessions"]]
    shuf = [s["accuracy_shuffled"] for s in res["sessions"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    im = ax.imshow(conf, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("decoded frequency (kHz)")
    ax.set_ylabel("presented frequency (kHz)")
    ax.set_title("pooled confusion matrix", fontsize=9)
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if conf[i, j] < 0.6 else "k")
    plt.colorbar(im, ax=ax, label="P(decoded | presented)", fraction=0.046)

    ax = axes[1]
    x = np.arange(len(acc))
    ax.bar(x - 0.2, acc, 0.4, label="decoder", color="C0")
    ax.bar(x + 0.2, shuf, 0.4, label="shuffled labels", color="0.7")
    ax.axhline(1 / len(freqs), color="r", ls="--", lw=1, label="chance")
    ax.set_xlabel("session")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=7)
    ax.set_title("5-way frequency decoding, 50 ms of population activity", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Tone frequency decoded from auditory-cortex population activity", y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_bandwidth_comparison(anf_res, cx_res, path=FIG + "fig11_tuning_width.png"):
    """Compare tuning width in octaves between auditory nerve and cortex."""
    anf_bw = [r["iso_bw_octaves"] for r in anf_res
              if np.isfinite(r.get("iso_bw_octaves", np.nan)) and r["iso_p"] < 0.01]
    cx_bw, freqs = [], cx_res["freqs"]
    oct_axis = np.log2(freqs / freqs[0])
    for u in cx_res["units"]:
        if u["p_tuned"] >= 0.01 or u["peak_driven"] <= 0:
            continue
        d = np.clip(u["driven"], 0, None)
        half = d.max() / 2
        above = d >= half
        # width at half maximum, counting the contiguous run around the peak
        pk = int(np.argmax(d))
        lo = pk
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = pk
        while hi < len(d) - 1 and above[hi + 1]:
            hi += 1
        cx_bw.append(max(oct_axis[hi] - oct_axis[lo], 0.5))  # >= one sampling step

    anf_bw, cx_bw = np.array(anf_bw), np.array(cx_bw)
    frac_anf = float(np.mean(anf_bw >= 1.0))
    frac_cx = float(np.mean(cx_bw >= 1.0))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 5, 26)
    ax.hist(anf_bw, bins=bins, alpha=0.6, label=f"auditory nerve (n={len(anf_bw)})",
            color="C0", density=True)
    ax.hist(cx_bw, bins=bins, alpha=0.6, label=f"auditory cortex (n={len(cx_bw)})",
            color="C3", density=True)
    ax.axvspan(0, 1.0, color="0.5", alpha=0.12, lw=0)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "below the cortical\nsampling resolution",
            ha="center", va="top", fontsize=7, color="0.35")
    ax.set_xlabel("half-maximum tuning width (octaves)")
    ax.set_ylabel("density")
    ax.set_title("Tuning width measured separately in each dataset\n"
                 f"(nerve median {np.median(anf_bw):.2f} octaves) - the two distributions "
                 "are NOT directly comparable", fontsize=8.5)
    ax.legend(fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, float(np.median(anf_bw)), frac_anf, frac_cx


def fig_glm(fits, freqs, empirical, path=FIG + "fig10_glm_kernels.png"):
    """Frequency-specific tone kernels, spike-history filter and model-based tuning."""
    n = len(fits)
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 6.6),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.35})
    axes = np.atleast_2d(axes.reshape(3, n))
    for k, f in enumerate(fits):
        ax = axes[0, k]
        for j, fq in enumerate(freqs):
            ax.plot(f["t_stim"] * 1e3, f["kernels"][j], color=FCOL[j], lw=1.4,
                    label=f"{fq/1000:g} kHz")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("time from tone onset (ms)")
        ax.set_title(f"unit {f['unit']}  (held-out pseudo-$R^2$ {f['r2_test']:.3f})",
                     fontsize=8.5)
        if k == 0:
            ax.set_ylabel("tone kernel\n(log rate)")
            ax.legend(fontsize=6.5, ncol=1, frameon=False, loc="upper right")

        ax = axes[1, k]
        ax.plot(f["t_hist"] * 1e3, f["history"], color="0.2", lw=1.4)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("time since own spike (ms)")
        if k == 0:
            ax.set_ylabel("history filter\n(log rate)")

        ax = axes[2, k]
        emp = empirical[f["unit"]]
        ax.plot(freqs / 1000, emp / np.abs(emp).max(), "-o", ms=4, color="C0",
                label="spike counts")
        ax.plot(freqs / 1000, f["gain"] / np.abs(f["gain"]).max(), "-s", ms=4, color="C3",
                label="GLM kernel area")
        ax.set_xscale("log", base=2)
        ax.set_xticks(freqs / 1000)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel("frequency (kHz)")
        if k == 0:
            ax.set_ylabel("normalised\nresponse")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Poisson GLM (NeMoS): frequency-specific tone kernels, spike history "
                 "and model-based tuning", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    anf_res = pickle.load(open("results_anf.pkl", "rb"))
    cx_res = pickle.load(open("results_cortex.pkl", "rb"))

    print(fig_anf_examples(anf_res))
    print(fig_anf_iso_examples(anf_res))
    print(fig_anf_population(anf_res))
    print(fig_cortex_population(cx_res))
    print(fig_decoding(cx_res))
    print(fig_bandwidth_comparison(anf_res, cx_res))

    files = dio.list_assets("001262", "0.241205.0959")
    fib = dio.load_fibre_001262(files[0][1])
    print(fig_anf_sweep(fib))

    files = dio.list_assets("000986")
    sess = dio.load_tone_session_000986([u for p, u in files if "LA9_ses-1" in p][0])
    spikes, _, _ = cx.session_to_pynapple(sess)
    print(fig_raw_cortex(sess, spikes))
    units = [u for u in cx_res["units"] if u["session"] == sess["session"]
             and u["subject"] == sess["subject"]]
    best = max(units, key=lambda u: u["peak_driven"] * (u["selectivity"] or 0))
    print(fig_cortex_example_unit(sess, spikes, best["unit"]))
    print(fig_cortex_tuning_examples(cx_res["units"], cx_res["freqs"]))

    import glm_analysis as glm
    picks = sorted(units, key=lambda u: -u["peak_driven"])[:3]
    fits, _ = glm.fit_session(sess, spikes, [u["unit"] for u in picks])
    empirical = {u["unit"]: u["driven"] for u in picks}
    print(fig_glm(fits, cx_res["freqs"], empirical))
