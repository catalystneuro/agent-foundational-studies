"""Figures for the spectrotemporal-receptive-field analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import strf_lib as S

FIG = "figures"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130})

Z_CRIT = 4.0          # ~p < 1e-4 per pixel against the circular-shift null
LAG_MAX_MS = 150.0    # window used to define responsiveness / BF / latency


# --------------------------------------------------------------------------
def ac_metrics(npz):
    strf, z = npz["strf"], npz["z"]
    dt, n_lags = float(npz["dt"]), int(npz["n_lags"])
    freqs = npz["freqs"]
    lags = np.arange(n_lags) * dt * 1000
    w = lags <= LAG_MAX_MS

    zw = z[:, w, :]
    responsive = np.nanmax(np.abs(zw), axis=(1, 2)) >= Z_CRIT
    suppressed = np.nanmin(zw, axis=(1, 2)) <= -Z_CRIT
    excited = np.nanmax(zw, axis=(1, 2)) >= Z_CRIT

    drive = np.clip(strf[:, w, :], 0, None).sum(axis=1)      # freq tuning
    bf_idx = np.argmax(drive, axis=1)
    bf = freqs[bf_idx]

    lat = np.full(len(strf), np.nan)
    peak_lat = np.full(len(strf), np.nan)
    for i in range(len(strf)):
        zi = z[i, w, bf_idx[i]]
        above = np.flatnonzero(zi >= 3.0)
        if len(above):
            lat[i] = lags[w][above[0]]
        peak_lat[i] = lags[w][int(np.argmax(strf[i, w, bf_idx[i]]))]
    return dict(lags=lags, freqs=freqs, responsive=responsive,
                excited=excited, suppressed=suppressed, drive=drive,
                bf=bf, bf_idx=bf_idx, latency=lat, peak_latency=peak_lat)


# --------------------------------------------------------------------------
def fig_raw():
    d = S.load_ac_session(S.asset_url("000986", "sub-LA11/sub-LA11_ses-1_behavior.nwb"))
    units, trials = d["units"], d["trials"]
    freqs = S.TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    rate = np.array([len(units[i]) for i in units.index])
    top = units.index[np.argsort(rate)[::-1][:15]]

    w0 = onsets[400]
    win = (w0 - 0.3, w0 + 7.7)
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.9, 2.4, 0.9], hspace=0.15))
    for k in range(len(freqs)):
        on = onsets[(fidx == k) & (onsets > win[0]) & (onsets < win[1])]
        ax[0].vlines(on, k - 0.4, k + 0.4, color=plt.cm.viridis(k / 4), lw=4)
    ax[0].set_yticks(range(5))
    ax[0].set_yticklabels([f"{f/1000:g} kHz" for f in freqs])
    ax[0].set_ylabel("tone pip")
    ax[0].set_title("Dandiset 000986 — mouse auditory cortex, Neuropixels, passive "
                    "25 ms pure tones at 60 dB SPL (subject LA11, session 1)")
    for r, i in enumerate(top):
        st = units[i].t
        st = st[(st > win[0]) & (st < win[1])]
        ax[1].vlines(st, r, r + 0.85, color="k", lw=0.6)
    ax[1].set_ylabel("unit (15 highest-rate)")
    ax[1].set_ylim(-0.5, len(top))
    pup = d["pupil"].restrict(__import__("pynapple").IntervalSet(*win))
    ax[2].plot(pup.t, pup.d, color="tab:purple", lw=1)
    ax[2].set_ylabel("pupil\n(norm.)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_xlim(win)
    fig.savefig(f"{FIG}/fig1_ac_raw_data.png", bbox_inches="tight")
    plt.close(fig)
    d["io"].close()


def fig_raster_psth(unit_hint=None):
    """Frequency-conditioned raster and PSTH: the raw material of the STRF."""
    import pynapple as nap

    d = S.load_ac_session(S.asset_url("000986", "sub-LA11/sub-LA11_ses-1_behavior.nwb"))
    units, trials = d["units"], d["trials"]
    freqs = S.TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)

    # pick the unit with the largest tone-evoked modulation
    best, best_score = None, -np.inf
    for uid in units.index:
        st = units[uid].t
        if len(st) < 5000:
            continue
        rel = st[:, None] - onsets[None, :250]
        ev = ((rel > 0.01) & (rel < 0.05)).sum() / 250 / 0.04
        base = ((rel > -0.15) & (rel < -0.01)).sum() / 250 / 0.14
        if ev - base > best_score:
            best, best_score = uid, ev - base
    uid = unit_hint if unit_hint is not None else best

    fig, axes = plt.subplots(2, 5, figsize=(15, 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
    bins = np.arange(-0.05, 0.2001, 0.005)
    ctr = (bins[:-1] + bins[1:]) / 2 * 1000
    for k in range(len(freqs)):
        on = onsets[fidx == k]
        pe = nap.compute_perievent(units[uid], nap.Ts(on), window=(-0.05, 0.2))
        h = np.zeros(len(bins) - 1)
        ax = axes[0, k]
        for j, key in enumerate(list(pe.keys())):
            t = pe[key].t
            h += np.histogram(t, bins=bins)[0]
            if j < 250:
                ax.plot(t * 1000, np.full(len(t), j), "|", color="k", ms=1.6,
                        mew=0.5)
        ax.axvspan(0, 25, color="gold", alpha=0.25, lw=0)
        ax.set_ylim(0, 250)
        ax.set_title(f"{freqs[k]/1000:g} kHz  ({len(on)} trials)", fontsize=9)
        axes[1, k].bar(ctr, h / len(on) / 0.005, width=5, color="tab:blue")
        axes[1, k].axvspan(0, 25, color="gold", alpha=0.25, lw=0)
        axes[1, k].set_xlabel("time from tone onset (ms)")
    axes[0, 0].set_ylabel("trial (first 250 shown)")
    axes[1, 0].set_ylabel("rate (sp/s)")
    ymax = max(a.get_ylim()[1] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(0, ymax)
    fig.suptitle(f"Frequency-conditioned raster and PSTH, unit {uid} "
                 "(sub-LA11 session 1). Stacking the five PSTHs gives the STRF",
                 y=0.97)
    fig.savefig(f"{FIG}/fig1b_ac_raster_psth.png", bbox_inches="tight")
    plt.close(fig)
    d["io"].close()
    return int(uid)


# --------------------------------------------------------------------------
def _plot_strf(ax, strf, lags, freqs, z=None, title="", cbar=True, tmax=None):
    dl = lags[1] - lags[0]
    ext = [lags[0], lags[-1] + dl, -0.5, len(freqs) - 0.5]
    v = np.abs(strf).max()
    im = ax.imshow(strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, extent=ext)
    if z is not None:
        # outline the pixels that clear the circular-shift null; upsampling
        # first keeps the contour on the pixel edges instead of cutting corners
        k = 12
        mask = np.kron((np.abs(z) >= Z_CRIT).T.astype(float), np.ones((k, k)))
        yy = np.linspace(ext[2], ext[3], mask.shape[0])
        xx = np.linspace(ext[0], ext[1], mask.shape[1])
        ax.contour(xx, yy, mask, levels=[0.5], colors="k", linewidths=0.8)
    if tmax is not None:
        ax.set_xlim(lags[0], tmax)
    ax.set_yticks(np.arange(len(freqs)))
    ax.set_yticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_title(title, fontsize=8)
    if cbar:
        plt.colorbar(im, ax=ax, pad=0.02, label="Δ rate (sp/s)")
    return im


def fig_examples(npz, m):
    strf, z = npz["strf"], npz["z"]
    lags, freqs = m["lags"], m["freqs"]
    lat_str = lambda v: "n/a" if not np.isfinite(v) else f"{v:.0f} ms"
    w = lags <= LAG_MAX_MS
    score = np.nanmax(np.abs(z[:, w, :]), axis=(1, 2))
    supp = -np.nanmin(z[:, w, :], axis=(1, 2))
    # six strong units spanning best frequencies, plus the two with the
    # strongest tone-evoked suppression
    picked, used = [], []
    for i in np.argsort(score)[::-1]:
        if not m["responsive"][i] or used.count(m["bf_idx"][i]) >= 2:
            continue
        picked.append(i)
        used.append(m["bf_idx"][i])
        if len(picked) == 6:
            break
    for i in np.argsort(supp)[::-1]:
        if i in picked:
            continue
        picked.append(i)
        if len(picked) == 8:
            break

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for k, (a, i) in enumerate(zip(axes.ravel(), picked)):
        sess = str(npz["session"][i]).split("/")[-1].replace("_behavior.nwb", "")
        tag = " (suppression)" if k >= 6 else ""
        _plot_strf(a, strf[i], lags, freqs, z[i],
                   f"{sess} u{npz['unit'][i]}{tag} | BF {m['bf'][i]/1000:g} kHz | "
                   f"lat {lat_str(m['latency'][i])}", tmax=200)
    for a in axes[-1]:
        a.set_xlabel("lag from tone onset (ms)")
    for a in axes[:, 0]:
        a.set_ylabel("tone frequency (kHz)")
    fig.suptitle("Cortical tone-pip STRFs (reverse correlation). Outline: pixels "
                 f"reaching |z| ≥ {Z_CRIT:g} against a circular-shift null", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_ac_example_strfs.png", bbox_inches="tight")
    plt.close(fig)
    return picked


def fig_population(npz, m):
    strf = npz["strf"]
    lags, freqs = m["lags"], m["freqs"]
    resp = m["responsive"]
    sessions = np.array([str(s) for s in npz["session"]])

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # (a) fraction responsive per session
    ax = fig.add_subplot(gs[0, 0])
    usess = sorted(set(sessions))
    frac = [resp[sessions == s].mean() for s in usess]
    n = [int((sessions == s).sum()) for s in usess]
    ax.bar(range(len(usess)), frac, color="tab:blue")
    ax.set_xticks(range(len(usess)))
    ax.set_xticklabels([s.split("/")[-1].replace("_behavior.nwb", "").replace("sub-", "")
                        for s in usess], rotation=90, fontsize=6)
    ax.set_ylabel("fraction tone-responsive")
    ax.set_title(f"(a) {resp.sum()}/{len(resp)} units responsive "
                 f"({resp.mean()*100:.0f}%) across {len(usess)} sessions")
    for i, (f, nn) in enumerate(zip(frac, n)):
        ax.text(i, f + 0.01, str(nn), ha="center", fontsize=5)

    # (b) best-frequency distribution
    ax = fig.add_subplot(gs[0, 1])
    counts = [np.sum(m["bf"][resp] == f) for f in freqs]
    ax.bar(range(len(freqs)), counts, color=[plt.cm.viridis(i / 4) for i in range(5)])
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("# responsive units")
    ax.set_title("(b) best frequency")

    # (c) latency distributions
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(m["latency"][resp], bins=np.arange(0, 155, 5), color="0.4",
            label="first significant lag")
    ax.hist(m["peak_latency"][resp], bins=np.arange(0, 155, 5), histtype="step",
            color="tab:red", lw=1.5, label="peak lag")
    ax.set_xlabel("lag from tone onset (ms)")
    ax.set_ylabel("# units")
    ax.legend(fontsize=7)
    ax.set_title(f"(c) onset latency (median {np.nanmedian(m['latency'][resp]):.0f} ms)")

    # (d) BF-aligned population mean STRF
    ax = fig.add_subplot(gs[1, 0])
    shifted = np.full((resp.sum(), len(lags), 2 * len(freqs) - 1), np.nan)
    for j, i in enumerate(np.flatnonzero(resp)):
        off = len(freqs) - 1 - m["bf_idx"][i]
        shifted[j, :, off:off + len(freqs)] = strf[i]
    mean_strf = np.nanmean(shifted, axis=0)
    v = np.abs(mean_strf).max()
    im = ax.imshow(mean_strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v,
                   extent=[lags[0], lags[-1], -(len(freqs) - 1), len(freqs) - 1])
    ax.set_xlabel("lag from tone onset (ms)")
    ax.set_ylabel("octaves re. best frequency")
    ax.set_title("(d) BF-aligned mean STRF")
    plt.colorbar(im, ax=ax, label="Δ rate (sp/s)")

    # (e) BF-sorted frequency tuning
    ax = fig.add_subplot(gs[1, 1])
    tune = m["drive"][resp]
    tune = tune / np.abs(tune).max(axis=1, keepdims=True)
    order = np.lexsort((tune.argmax(1), m["bf_idx"][resp]))
    im = ax.imshow(tune[order], aspect="auto", origin="lower", cmap="magma",
                   extent=[-0.5, 4.5, 0, resp.sum()])
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("tone frequency (kHz)")
    ax.set_ylabel("responsive units (sorted by BF)")
    ax.set_title("(e) normalised frequency tuning")
    plt.colorbar(im, ax=ax, label="norm. drive")

    # (f) excitation vs suppression
    ax = fig.add_subplot(gs[1, 2])
    cats = ["excit.\nonly", "suppr.\nonly", "both", "neither"]
    e, s = m["excited"], m["suppressed"]
    vals = [np.sum(e & ~s), np.sum(s & ~e), np.sum(e & s), np.sum(~e & ~s)]
    ax.bar(cats, vals, color=["tab:red", "tab:blue", "tab:purple", "0.7"])
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + 2, f"{v_}\n({v_/len(e)*100:.0f}%)", ha="center", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("# units")
    ax.set_title("(f) sign of significant STRF pixels")

    fig.suptitle("Cortical STRF population — dandiset 000986, 15 sessions, 5 mice", y=0.97)
    fig.savefig(f"{FIG}/fig3_ac_population.png", bbox_inches="tight")
    plt.close(fig)


def fig_glm(npz_sta, m):
    g = np.load("glm_strf.npz", allow_pickle=True)
    freqs, dt, W = g["freqs"], float(g["dt"]), int(g["window"])
    lags = np.arange(W) * dt * 1000
    sess = np.array([str(s) for s in npz_sta["session"]])
    ismine = sess == "sub-LA11/sub-LA11_ses-1_behavior.nwb"

    dr2 = g["r2_win"] - g["r2_null_win"]
    order = np.argsort(dr2)[::-1][:4]
    TMAX = 150

    fig = plt.figure(figsize=(16.5, 6.6))
    gs = fig.add_gridspec(2, 5, wspace=0.8, hspace=0.5,
                          width_ratios=[1, 1, 1, 1, 1.5])
    for c, i in enumerate(order):
        uid = g["unit"][i]
        j = np.flatnonzero(ismine & (npz_sta["unit"] == uid))[0]
        a = fig.add_subplot(gs[0, c])
        _plot_strf(a, npz_sta["strf"][j][: len(lags)], lags, freqs, cbar=True,
                   title=f"unit {uid} — reverse correlation", tmax=TMAX)
        a2 = fig.add_subplot(gs[1, c])
        v = np.abs(g["strf"][i]).max()
        im = a2.imshow(g["strf"][i].T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-v, vmax=v, extent=[lags[0], lags[-1], -0.5, 4.5])
        a2.set_xlim(0, TMAX)
        a2.set_yticks(range(5))
        a2.set_yticklabels([f"{f/1000:g}" for f in freqs])
        a2.set_title(f"unit {uid} — Poisson GLM (ΔR²={dr2[i]:.3f})", fontsize=8)
        a2.set_xlabel("lag from tone onset (ms)")
        plt.colorbar(im, ax=a2, pad=0.02, label="log-rate weight")
        if c == 0:
            a.set_ylabel("tone freq (kHz)")
            a2.set_ylabel("tone freq (kHz)")

    ax = fig.add_subplot(gs[:, 4])
    ax.scatter(g["r2_null_win"], g["r2_win"], c="tab:blue", s=30)
    lim = [min(g["r2_null_win"].min(), g["r2_win"].min()) - 0.01,
           max(g["r2_null_win"].max(), g["r2_win"].max()) + 0.01]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pseudo-R²: spike history only")
    ax.set_ylabel("pseudo-R²: spike history + STRF")
    ax.set_title(f"held-out prediction, tone windows\n{(dr2 > 0).sum()}/{len(dr2)} "
                 f"units improved (median Δ={np.median(dr2):.3f})", fontsize=9)
    fig.suptitle("Regularised Poisson-GLM STRFs (NeMoS) reproduce the "
                 "reverse-correlation estimate and predict held-out spikes", y=1.0)
    fig.savefig(f"{FIG}/fig4_ac_glm_strf.png", bbox_inches="tight")
    plt.close(fig)


def fig_arousal(npz, m):
    """Does pupil-indexed arousal change the STRF? (dandiset 000986)"""
    from scipy.stats import wilcoxon

    lags, freqs = m["lags"], m["freqs"]
    resp = m["responsive"]
    lo, hi = npz["strf_lo"], npz["strf_hi"]
    w = (lags >= 0) & (lags <= 100)

    late = (lags >= 60) & (lags <= 150)

    peak_lo = np.array([lo[i, w, m["bf_idx"][i]].max() for i in range(len(lo))])
    peak_hi = np.array([hi[i, w, m["bf_idx"][i]].max() for i in range(len(hi))])
    late_lo = np.array([lo[i, late, m["bf_idx"][i]].mean() for i in range(len(lo))])
    late_hi = np.array([hi[i, late, m["bf_idx"][i]].mean() for i in range(len(hi))])
    ok = resp & (peak_lo > 0) & (peak_hi > 0) & (late_lo > 0) & (late_hi > 0)
    _, p = wilcoxon(peak_hi[ok], peak_lo[ok])
    _, p_late = wilcoxon(late_hi[ok], late_lo[ok])

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    for arr, lab, col in ((lo, "small pupil (low arousal)", "tab:blue"),
                          (hi, "large pupil (high arousal)", "tab:red")):
        tr = np.stack([arr[i, :, m["bf_idx"][i]] for i in np.flatnonzero(ok)])
        ax[0].plot(lags, tr.mean(0), color=col, lw=2, label=lab)
        se = tr.std(0) / np.sqrt(len(tr))
        ax[0].fill_between(lags, tr.mean(0) - se, tr.mean(0) + se, color=col, alpha=0.25)
    ax[0].set_xlim(0, 150)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("lag from tone onset (ms)")
    ax[0].set_ylabel("Δ rate at BF (sp/s)")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"(a) tone response at BF, n={ok.sum()} units")

    for k, (a_, b_, pv, lab) in enumerate([
            (peak_lo, peak_hi, p, "peak (0–100 ms)"),
            (late_lo, late_hi, p_late, "late (60–150 ms)")]):
        g = np.log2(b_[ok] / a_[ok])
        axi = ax[k + 1]
        axi.hist(g, bins=np.arange(-4, 4.1, 0.2), color="0.4")
        axi.axvline(0, color="k", lw=1)
        axi.axvline(np.median(g), color="tab:red", lw=1.5,
                    label=f"median {np.median(g):+.2f} oct "
                          f"({2**np.median(g) - 1:+.0%})")
        axi.set_xlabel("log2 gain ratio (large / small pupil)")
        axi.set_ylabel("# units")
        axi.legend(fontsize=7)
        axi.set_title(f"({'bc'[k]}) {lab} response\nWilcoxon p = {pv:.1e}")

    fig.suptitle("Pupil-indexed arousal leaves the STRF peak almost unchanged but "
                 "boosts its late component", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig8_ac_arousal.png", bbox_inches="tight")
    plt.close(fig)
    return dict(n=int(ok.sum()), p_peak=float(p), p_late=float(p_late),
                gain_peak=float(np.median(np.log2(peak_hi[ok] / peak_lo[ok]))),
                gain_late=float(np.median(np.log2(late_hi[ok] / late_lo[ok]))))


# --------------------------------------------------------------------------
def fig_anf_examples():
    paths = [
        "sub-G190704/sub-G190704_ses-G190704-342_icephys.nwb",
        "sub-G201103/sub-G201103_ses-G201103-1p-6_icephys.nwb",
        "sub-G220301/sub-G220301_ses-G220301-1p-565_icephys.nwb",
        "sub-G151104/sub-G151104_ses-G151104-2P-530nm_icephys.nwb",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, p in zip(axes, paths):
        fb = S.load_anf_fibre(S.asset_url("001262", p))
        R, t, fq = S.anf_strf(fb, bin_ms=1.0, t_max=0.08)
        Rs = gaussian_filter(R, (1.2, 0.6))
        im = ax.pcolormesh(t * 1000, fq / 1000, Rs.T, cmap="magma", shading="auto")
        ax.axhline(fb["bf_published"] / 1000, color="c", ls="--", lw=1,
                   label="published BF")
        ax.set_xlabel("time from tone onset (ms)")
        ax.set_title(f"{p.split('/')[0][4:]}  BF={fb['bf_published']:.0f} Hz, "
                     f"{len(fq)} freqs, spont={fb['sr_published']:.0f} sp/s",
                     fontsize=8)
        plt.colorbar(im, ax=ax, label="rate (sp/s)")
        fb["h5"].close()
    axes[0].set_ylabel("tone frequency (kHz)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Dandiset 001262 — gerbil auditory-nerve fibres: frequency × time "
                 "receptive fields (50 ms tone from ~10 ms)", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_anf_example_strfs.png", bbox_inches="tight")
    plt.close(fig)


def fig_anf_population():
    a = np.load("anf_population.npz", allow_pickle=True)
    grid, aligned, prof = a["grid"], a["aligned"], a["prof"]
    t = np.arange(aligned.shape[1]) * float(a["bin_ms"])

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(a["bf_pub"], a["bf_meas"], "o", ms=4, alpha=0.7)
    lim = [min(a["bf_pub"].min(), a["bf_meas"].min()) * 0.8,
           max(a["bf_pub"].max(), a["bf_meas"].max()) * 1.2]
    ax.plot(lim, lim, "k--", lw=1)
    r = np.corrcoef(np.log2(a["bf_pub"]), np.log2(a["bf_meas"]))[0, 1]
    err = np.median(np.abs(np.log2(a["bf_meas"] / a["bf_pub"])))
    ax.set_xlabel("published best frequency (Hz)")
    ax.set_ylabel("BF from the receptive field (Hz)")
    ax.set_title(f"(a) BF recovered from the RF\nr={r:.3f}, median error {err:.3f} oct")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(a["bf_meas"] / 1000, bins=np.logspace(np.log10(0.3), np.log10(20), 22),
            color="tab:green")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("# fibres")
    ax.set_title(f"(b) CF distribution, n={len(a['bf_meas'])} fibres")

    ax = fig.add_subplot(gs[0, 2])
    norm = prof / np.maximum(prof.max(axis=1, keepdims=True), 1e-9)
    ax.plot(t, np.nanmean(norm, axis=0), color="k", lw=2)
    ax.fill_between(t, np.nanpercentile(norm, 25, axis=0),
                    np.nanpercentile(norm, 75, axis=0), color="k", alpha=0.2)
    ax.axvspan(10, 58, color="gold", alpha=0.13, zorder=0, label="tone on")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("normalised rate at BF")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) population time course at BF")

    ax = fig.add_subplot(gs[1, 0])
    mean_al = np.nanmean(aligned, axis=0)
    im = ax.pcolormesh(t, grid, mean_al.T, cmap="magma", shading="auto")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("octaves re. best frequency")
    ax.set_title("(d) BF-aligned population receptive field")
    plt.colorbar(im, ax=ax, label="rate (sp/s)")

    ax = fig.add_subplot(gs[1, 1])
    drive = np.nanmean(aligned[:, 10:60, :], axis=1) - a["spont"][:, None]
    dn = drive / np.nanmax(drive, axis=1, keepdims=True)
    ax.plot(grid, np.nanmean(dn, axis=0), "k", lw=2)
    ax.fill_between(grid, np.nanpercentile(dn, 25, axis=0),
                    np.nanpercentile(dn, 75, axis=0), color="k", alpha=0.2)
    ax.axhline(0.5, color="tab:red", ls=":", lw=1)
    half = np.nanmean(dn, axis=0) >= 0.5
    bw = grid[half][-1] - grid[half][0] if half.any() else np.nan
    ax.set_xlabel("octaves re. best frequency")
    ax.set_ylabel("normalised driven rate")
    ax.set_title(f"(e) spectral tuning\nwidth at half maximum {bw:.2f} octaves")

    ax = fig.add_subplot(gs[1, 2])
    # smooth over 5 ms first: with 5 repetitions a 1 ms bin can only take
    # values that are multiples of 200 sp/s, which would quantise the peak
    kern = np.ones(5) / 5
    sm = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, prof)
    peak = sm[:, :30].max(axis=1)
    sust = sm[:, 40:56].mean(axis=1)
    ok = peak > 0
    ax.scatter(peak[ok], sust[ok], s=22, alpha=0.7, color="tab:orange")
    ax.plot([0, peak[ok].max()], [0, peak[ok].max()], "k--", lw=1)
    ratio = np.median(sust[ok] / peak[ok])
    ax.set_xlabel("peak onset rate (sp/s, 0–30 ms)")
    ax.set_ylabel("sustained rate (sp/s, 40–56 ms)")
    ax.set_title(f"(f) onset adaptation\nmedian sustained/peak = {ratio:.2f}")

    fig.suptitle("Auditory-nerve receptive fields — dandiset 001262, Mongolian gerbil",
                 y=0.97)
    fig.savefig(f"{FIG}/fig6_anf_population.png", bbox_inches="tight")
    plt.close(fig)
    return dict(bw=bw, ratio=ratio, r=r, err=err, n=len(a["bf_meas"]))


def fig_compare(npz, m):
    """Receptive-field shape at two stages of the auditory system."""
    a = np.load("anf_population.npz", allow_pickle=True)
    grid, aligned, prof = a["grid"], a["aligned"], a["prof"]
    t_anf = np.arange(prof.shape[1]) * float(a["bin_ms"])

    strf, lags, freqs = npz["strf"], m["lags"], m["freqs"]
    resp = m["responsive"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    im = ax[0].pcolormesh(t_anf, grid, np.nanmean(aligned, axis=0).T,
                          cmap="magma", shading="auto")
    ax[0].set_xlim(0, 80)
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlabel("time from tone onset (ms)")
    ax[0].set_ylabel("octaves re. best frequency")
    ax[0].set_title(f"(a) auditory nerve, n={len(prof)} fibres\n"
                    "frequency grid ≈ 0.02–0.1 octaves")
    plt.colorbar(im, ax=ax[0], label="rate (sp/s)")

    shifted = np.full((resp.sum(), len(lags), 2 * len(freqs) - 1), np.nan)
    for j, i in enumerate(np.flatnonzero(resp)):
        off = len(freqs) - 1 - m["bf_idx"][i]
        shifted[j, :, off:off + len(freqs)] = strf[i]
    mean_strf = np.nanmean(shifted, axis=0)
    v = np.abs(mean_strf).max()
    im = ax[1].imshow(mean_strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-v, vmax=v,
                      extent=[lags[0], lags[-1], -(len(freqs) - 1), len(freqs) - 1])
    ax[1].set_xlim(0, 150)
    ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel("lag from tone onset (ms)")
    ax[1].set_ylabel("octaves re. best frequency")
    ax[1].set_title(f"(b) auditory cortex, n={resp.sum()} units\n"
                    "frequency grid = 1 octave")
    plt.colorbar(im, ax=ax[1], label="Δ rate (sp/s)")

    # temporal profiles at BF, both on the cortical 5 ms grid
    dt_ms = (lags[1] - lags[0])
    edges = np.arange(0, 155, dt_ms)
    ctr = edges[:-1] + dt_ms / 2
    n_anf = prof / np.maximum(prof.max(axis=1, keepdims=True), 1e-9)
    anf_5ms = np.stack([np.histogram(t_anf, bins=edges, weights=r)[0] /
                        np.maximum(np.histogram(t_anf, bins=edges)[0], 1)
                        for r in n_anf])
    anf_5ms[:, ctr > t_anf[-1]] = np.nan
    bf_trace = np.stack([strf[i, :, m["bf_idx"][i]] for i in np.flatnonzero(resp)])
    n_ac = bf_trace / np.maximum(bf_trace.max(axis=1, keepdims=True), 1e-9)
    n_ac = n_ac[:, : len(ctr)]

    for arr, lab, col in ((anf_5ms, f"auditory nerve (50 ms tone)", "tab:green"),
                          (n_ac, "cortex (25 ms tone)", "tab:red")):
        med = np.nanmedian(arr, axis=0)
        ax[2].plot(ctr, med, color=col, lw=2, label=lab)
        ax[2].fill_between(ctr, np.nanpercentile(arr, 40, axis=0),
                           np.nanpercentile(arr, 60, axis=0), color=col, alpha=0.25)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlim(0, 150)
    ax[2].set_xlabel("time from tone onset (ms)")
    ax[2].set_ylabel("normalised response at BF")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) temporal profile at BF\nmedian and 40–60th percentile")

    fig.suptitle("Receptive-field shape at two stages of the auditory system "
                 "(different species, tone durations and spectral sampling)", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_nerve_vs_cortex.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    npz = np.load("ac_population.npz", allow_pickle=True)
    m = ac_metrics(npz)
    print(f"units {len(m['bf'])}, responsive {m['responsive'].sum()} "
          f"({m['responsive'].mean()*100:.1f}%)")
    print("median latency", np.nanmedian(m["latency"][m["responsive"]]))
    fig_raw()
    print('raster unit', fig_raster_psth())
    fig_examples(npz, m)
    fig_population(npz, m)
    fig_glm(npz, m)
    print(fig_arousal(npz, m))
    fig_anf_examples()
    print(fig_anf_population())
    fig_compare(npz, m)
    print("figures written")
