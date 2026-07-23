"""Figures for the primary dataset (DANDI 000986, mouse auditory cortex)."""

import numpy as np
import pynapple as nap
from scipy import stats

import plotting as P
from plotting import khz, normalize_curves, octaves_from_bf, plt

nap.nap_config.suppress_conversion_warnings = True

D = np.load("results_000986.npz", allow_pickle=True)
UF = D["ufreq"]
COL = P.freq_colors(UF)
TUNED = D["p_anova"] < 0.01
DRIVEN = D["p_driven"] < 0.01
EW, BW = D["evoked_window"], D["base_window"]


def pfmt(p):
    return "p < 1e-300" if p == 0 else f"p = {p:.1e}"


# ----------------------------------------------------------------- figure 1
def fig_response_window():
    t = D["psth_t"] * 1000
    grand = D["psth"].mean(axis=(0, 1))
    pre = D["psth"][:, :, t < 0]
    base = pre.mean(axis=(1, 2))[:, None]
    sd = pre.std(axis=2).mean(1)[:, None]
    z = (D["psth"].mean(1) - base) / (sd + 1e-9)
    lat = np.array([t[np.argmax(zz)] for zz in z])
    order = np.argsort(lat)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    ax[0].plot(t, grand, "k", lw=1.5)
    ax[0].axvspan(0, 25, color="orange", alpha=0.35, label="tone (25 ms)")
    ax[0].axvspan(EW[0] * 1000, EW[1] * 1000, color="C2", alpha=0.18,
                  label="response window")
    ax[0].axvspan(t[0], BW[1] * 1000, color="C0", alpha=0.15, label="baseline")
    ax[0].set_xlim(t[0], t[-1])
    ax[0].set(xlabel="time from tone onset (ms)",
              ylabel=f"mean rate, {len(D['bf'])} units (Hz)",
              title="Population tone response")
    ax[0].legend(fontsize=7, loc="upper right")

    im = ax[1].imshow(z[order], aspect="auto", vmin=-6, vmax=6, cmap="RdBu_r",
                      extent=[t[0], t[-1], len(z), 0])
    ax[1].set(xlabel="time from tone onset (ms)",
              ylabel="unit (sorted by peak latency)",
              title="Single-unit responses")
    plt.colorbar(im, ax=ax[1], label="z vs pre-tone baseline")

    sel = DRIVEN & (z.max(1) > 3)
    ax[2].hist(lat[sel], bins=np.arange(0, 105, 5), color="0.3")
    ax[2].axvline(np.median(lat[sel]), color="C3", ls="--",
                  label=f"median {np.median(lat[sel]):.0f} ms")
    ax[2].set(xlabel="peak latency (ms)", ylabel="# sound-driven units",
              title=f"Response latency (n = {sel.sum()})")
    ax[2].legend(fontsize=7)
    fig.savefig("fig1_response_window.png")
    plt.close(fig)
    return lat


# ----------------------------------------------------------------- figure 2
def pick_examples(session, n=3):
    """Well-tuned units from one session: unimodal on the log-frequency axis,
    ranked by the reliability of the peak versus the rest of the curve."""
    m = np.where((D["session"] == session) & TUNED)[0]
    er, se = D["evoked_rate"][m], D["sem"][m]
    pk = er.argmax(1)
    unimodal = np.array([np.all(np.diff(e[:p + 1]) > 0) and np.all(np.diff(e[p:]) < 0)
                         for e, p in zip(er, pk)])
    others = (er.sum(1) - er.max(1)) / (er.shape[1] - 1)
    snr = (er.max(1) - others) / (se.max(1) + 1e-9)
    snr[~unimodal] = -1
    chosen = []
    for f in UF:  # spread the examples over different best frequencies
        cand = np.where(D["bf"][m] == f, snr, -1)
        if cand.max() > 0:
            chosen.append((cand.max(), int(m[cand.argmax()])))
    chosen.sort(reverse=True)
    return [i for _, i in chosen[:n]]


def fig_example_units(session, example_idx):
    from dandi_io import list_assets, load_tone_session

    url = dict(list_assets("000986"))[session]
    S = load_tone_session(url)
    spikes, onsets, freq = S["spikes"], S["onsets"], S["freq"]
    t = D["psth_t"] * 1000
    n_show = 60  # trials per frequency in the raster

    fig, axes = plt.subplots(len(example_idx), 3, figsize=(11.5, 2.6 * len(example_idx)),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    for r, i in enumerate(example_idx):
        uid = int(D["unit_ids"][i])
        st = np.asarray(spikes[uid].t)

        a = axes[r, 0]
        row = 0
        for k, f in enumerate(UF):
            ons = onsets[freq == f][:n_show]
            lo = np.searchsorted(st, ons - 0.05)
            hi = np.searchsorted(st, ons + 0.15)
            for aa, bb, o in zip(lo, hi, ons):
                x = (st[aa:bb] - o) * 1000
                a.plot(x, np.full_like(x, row), "|", color=COL[k], ms=2.5, mew=0.6)
                row += 1
            a.text(152, row - n_show / 2, f"{khz(f)}", color=COL[k], va="center",
                   fontsize=7)
        a.axvspan(0, 25, color="orange", alpha=0.25, zorder=0)
        a.set_xlim(-50, 150)
        a.set_ylim(row, -1)
        a.set_ylabel(f"trial ({n_show} per tone)")
        a.set_title(f"unit {uid} raster", fontsize=9)

        b = axes[r, 1]
        for k in range(len(UF)):
            b.plot(t, D["psth"][i, k], color=COL[k], lw=1.2, label=f"{khz(UF[k])} kHz")
        b.axvspan(0, 25, color="orange", alpha=0.25, zorder=0)
        b.set_ylabel("rate (Hz)")
        b.set_title("PSTH by tone frequency", fontsize=9)
        if r == 0:
            b.legend(fontsize=6.5, ncol=2, loc="upper right")

        c = axes[r, 2]
        c.errorbar(np.log2(UF), D["evoked_rate"][i], yerr=D["sem"][i], color="k",
                   marker="none", lw=1.2, capsize=2, zorder=1)
        for k in range(len(UF)):
            c.plot(np.log2(UF[k]), D["evoked_rate"][i, k], "o", color=COL[k], ms=7,
                   zorder=2)
        c.axhline(0, color="0.6", lw=0.8, ls="--")
        c.set_xticks(np.log2(UF), [khz(f) for f in UF])
        c.set_ylabel("evoked rate (Hz)")
        c.set_title(f"BF {khz(D['bf'][i])} kHz, {pfmt(D['p_anova'][i])}", fontsize=9)

        if r == len(example_idx) - 1:
            a.set_xlabel("time from tone onset (ms)")
            b.set_xlabel("time from tone onset (ms)")
            c.set_xlabel("tone frequency (kHz)")
    fig.suptitle(f"Tone-evoked responses of example units "
                 f"({session.split('/')[-1].replace('_behavior.nwb','')})")
    fig.savefig("fig2_example_units.png")
    plt.close(fig)
    S["io"].close()


# ----------------------------------------------------------------- figure 3
def fig_population_tuning():
    norm, ok = normalize_curves(D["evoked_rate"])
    sel = TUNED & ok
    curves, bf = norm[sel], D["bf"][sel]
    order = np.argsort(np.log2(bf) + 1e-6 * np.arange(len(bf)))

    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(curves[order], aspect="auto", vmin=0, vmax=1, cmap="magma",
                   extent=[-0.5, len(UF) - 0.5, len(curves), 0],
                   interpolation="nearest")
    ax.set_xticks(range(len(UF)), [khz(f) for f in UF])
    ax.set(xlabel="tone frequency (kHz)", ylabel="unit (sorted by best frequency)",
           title=f"Normalized tuning curves\n{sel.sum()} of {TUNED.sum()} tuned units "
                 f"with an excitatory peak")
    plt.colorbar(im, ax=ax, label="normalized evoked rate")

    ax = fig.add_subplot(gs[0, 1])
    cnt = np.array([(D["bf"][TUNED] == f).sum() for f in UF])
    ax.bar(range(len(UF)), cnt, color=COL)
    ax.set_xticks(range(len(UF)), [khz(f) for f in UF])
    ax.set_ylim(0, cnt.max() * 1.3)
    ax.set(xlabel="best frequency (kHz)", ylabel="# units",
           title="Best-frequency distribution")
    ax.text(0.03, 0.96, f"$\\chi^2$ vs uniform: {pfmt(stats.chisquare(cnt).pvalue)}",
            transform=ax.transAxes, va="top", fontsize=7.5)

    ax = fig.add_subplot(gs[0, 2])
    oct_axis = octaves_from_bf(UF, D["bf"])[sel]
    grid = np.arange(-4, 4.5, 1.0)
    prof = np.full((sel.sum(), len(grid)), np.nan)
    idx = np.searchsorted(grid, oct_axis - 1e-9)
    np.put_along_axis(prof, idx, curves, axis=1)
    m = np.nanmean(prof, 0)
    s = np.nanstd(prof, 0) / np.sqrt(np.sum(~np.isnan(prof), 0))
    ax.fill_between(grid, m - s, m + s, color="C3", alpha=0.35)
    ax.plot(grid, m, "C3o-", ms=4)
    ax.set(xlabel="octaves from best frequency", ylabel="normalized evoked rate",
           title="Mean tuning curve aligned to BF")

    ax = fig.add_subplot(gs[1, 1])
    frac = np.array([np.mean(D["p_anova"][D["session"] == s] < 0.01)
                     for s in D["session_list"]])
    nun = [int(np.sum(D["session"] == s)) for s in D["session_list"]]
    ax.bar(range(len(frac)), frac, color="0.4")
    ax.set_xticks(range(len(frac)),
                  [s.split("/")[-1].replace("_behavior.nwb", "").replace("sub-", "")
                   for s in D["session_list"]], rotation=90, fontsize=6)
    ax.axhline(0.01, color="r", ls="--", lw=0.8, label="chance (α = 0.01)")
    ax.set(ylabel="fraction frequency-tuned", ylim=(0, 1.15),
           title="Tuned fraction per session")
    ax.legend(fontsize=7, loc="lower right")
    for i, (f, n) in enumerate(zip(frac, nun)):
        ax.text(i, f + 0.02, str(n), ha="center", fontsize=5.5)

    ax = fig.add_subplot(gs[1, 2])
    depth = D["evoked_rate"].max(1) - D["evoked_rate"].min(1)
    bins = np.logspace(-2, 2, 40)
    ax.hist(depth[~TUNED], bins=bins, color="0.75", label=f"not tuned (n={(~TUNED).sum()})")
    ax.hist(depth[TUNED], bins=bins, color="C3", alpha=0.8,
            label=f"tuned (n={TUNED.sum()})")
    ax.set_xscale("log")
    ax.set(xlabel="modulation depth, max - min (Hz)", ylabel="# units",
           title="Tuning modulation depth")
    ax.legend(fontsize=7)
    fig.savefig("fig3_population_tuning.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 4
def fig_bf_by_session():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sess = D["session_list"]
    mat = np.zeros((len(sess), len(UF)))
    for i, s in enumerate(sess):
        m = (D["session"] == s) & TUNED
        c = np.array([(D["bf"][m] == f).sum() for f in UF], float)
        mat[i] = c / max(c.sum(), 1)
    im = ax[0].imshow(mat, aspect="auto", cmap="magma", vmin=0)
    ax[0].set_xticks(range(len(UF)), [khz(f) for f in UF])
    ax[0].set_yticks(range(len(sess)),
                     [s.split("/")[-1].replace("_behavior.nwb", "").replace("sub-", "")
                      for s in sess], fontsize=6.5)
    ax[0].set(xlabel="best frequency (kHz)",
              title="BF distribution per recording session")
    plt.colorbar(im, ax=ax[0], label="fraction of tuned units")

    norm, ok = normalize_curves(D["evoked_rate"])
    sel = TUNED & ok
    for k, f in enumerate(UF):
        m = sel & (D["bf"] == f)
        mu = np.nanmean(norm[m], 0)
        se = np.nanstd(norm[m], 0) / np.sqrt(m.sum())
        ax[1].fill_between(np.log2(UF), mu - se, mu + se, color=COL[k], alpha=0.25)
        ax[1].plot(np.log2(UF), mu, "o-", color=COL[k], ms=4,
                   label=f"BF {khz(f)} kHz (n={m.sum()})")
    ax[1].set_xticks(np.log2(UF), [khz(f) for f in UF])
    ax[1].set(xlabel="tone frequency (kHz)", ylabel="normalized evoked rate",
              title="Mean tuning curve by best-frequency group")
    ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig("fig4_bf_by_session.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"{len(D['bf'])} units | {DRIVEN.sum()} sound-driven | "
          f"{TUNED.sum()} frequency-tuned (ANOVA p<0.01)")
    lat = fig_response_window()
    print("median peak latency (ms):", np.median(lat[DRIVEN]))
    session = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
    ex = pick_examples(session)
    print("examples:", [(int(D["unit_ids"][i]), D["bf"][i]) for i in ex])
    fig_example_units(session, ex)
    fig_population_tuning()
    fig_bf_by_session()
    print("figures written")
