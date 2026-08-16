"""Figure generation for the theta entrainment / phase precession analysis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import pynapple as nap

import theta_lib as tl

FIGDIR = "figures"
TWO_CYCLE_BINS = np.linspace(0, 4 * np.pi, 37)


def _phase_hist(ph, bins=18):
    """Histogram over one cycle, returned duplicated over two cycles for display."""
    h, e = np.histogram(ph, bins=bins, range=(0, 2 * np.pi))
    h = h / h.sum()
    return np.r_[h, h], np.r_[e[:-1], e[:-1] + 2 * np.pi], (e[1] - e[0])


# --------------------------------------------------------------------- fig 0


def fig_channel_selection(res, fname="00_channel_selection.png"):
    """Theta/delta ratio and theta amplitude across the 128 LFP channels."""
    ratio, th = res["channel_ratio"], res["channel_theta_pow"]
    best = res["best_ch"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(ratio, ".-", lw=0.6, color="0.4")
    ax[0].plot(best, ratio[best], "r*", ms=16)
    ax[0].set(xlabel="LFP channel", ylabel="theta / delta power ratio",
              title=f"Channel selection (best = {best})")
    ax[1].plot(np.sqrt(th) * 1e3, ".-", lw=0.6, color="0.4")
    ax[1].plot(best, np.sqrt(th[best]) * 1e3, "r*", ms=16)
    ax[1].set(xlabel="LFP channel", ylabel="theta RMS amplitude (mV)",
              title="6-12 Hz amplitude by channel")
    fr, P = res["psd"]
    m = (fr >= 1) & (fr <= 40)
    ax[2].semilogy(fr[m], P[m], "k", lw=1.4)
    ax[2].axvspan(*tl.THETA_BAND, color="crimson", alpha=0.18)
    ax[2].axvline(res["lfp_theta_freq"], color="crimson", ls="--", lw=1.5,
                  label=f"peak {res['lfp_theta_freq']:.2f} Hz")
    ax[2].set(xlabel="frequency (Hz)", ylabel="PSD (V$^2$/Hz)",
              title="LFP spectrum during running")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"LFP channel selection - {res['label']}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 1


def fig_raw_streams(res, fname="01_raw_streams.png"):
    runs, pos = res["runs"], res["pos"]
    lfp, filt, phase = res["lfp"], res["filt"], res["phase"]
    i = 3
    wide = nap.IntervalSet(start=runs.start[i] - 2, end=runs.end[i + 2] + 2)
    zoom = nap.IntervalSet(start=runs.start[i], end=runs.start[i] + 2)

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    l = lfp.restrict(wide)
    ax.plot(l.times(), l.values * 1e3, "k", lw=0.5, label="raw LFP (1250 Hz)")
    ax.plot(filt.restrict(wide).times(), filt.restrict(wide).values * 1e3, "crimson",
            lw=1.2, label="6-12 Hz")
    ax.set_ylabel("LFP (mV)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"CA1 LFP, channel {res['best_ch']} - {res['label']}", pad=8)

    ax = fig.add_subplot(gs[1, :])
    p = pos.restrict(wide)
    ax.plot(p.times(), p.values, "g.", ms=3)
    for s, e in zip(runs.start, runs.end):
        if s >= wide.start[0] and e <= wide.end[-1]:
            ax.axvspan(s, e, color="orange", alpha=0.15)
    ax.axvspan(zoom.start[0], zoom.end[-1], color="dodgerblue", alpha=0.25)
    ax.set_ylabel("linear position (m)")
    ax.set_xlabel("time (s)")
    ax.set_title("Linearized position; orange = track traversals, blue = zoom window", pad=8)

    ax = fig.add_subplot(gs[2, 0])
    l = lfp.restrict(zoom)
    f = filt.restrict(zoom)
    ax.plot(l.times() - zoom.start[0], l.values * 1e3, "k", lw=0.6)
    ax.plot(f.times() - zoom.start[0], f.values * 1e3, "crimson", lw=1.6)
    ax.set(xlabel="time from run onset (s)", ylabel="LFP (mV)",
           title="Zoom: theta rhythm during running")

    ax = fig.add_subplot(gs[2, 1])
    ph = phase.restrict(zoom)
    ax.plot(ph.times() - zoom.start[0], np.degrees(ph.values), "b", lw=1.0)
    ax.set(xlabel="time from run onset (s)", ylabel="theta phase (deg)",
           title="Hilbert phase (0 deg = peak of filtered LFP)", ylim=(0, 360))
    ax.set_yticks([0, 90, 180, 270, 360])

    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 2


def fig_place_fields(res, pc_mask, fname="02_place_fields.png"):
    df, tcs, centers = res["df"], res["tcs"], res["centers"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, d in enumerate((+1, -1)):
        units_pc = df.loc[pc_mask, "unit"].values
        M = np.stack([gaussian_filter1d(np.nan_to_num(tcs[d][u].values), 1.5, mode="nearest")
                      for u in units_pc])
        M = M / np.maximum(M.max(axis=1, keepdims=True), 1e-9)
        order = np.argsort(np.argmax(M, axis=1))
        ax = axes[j, 0]
        im = ax.imshow(M[order], aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, 1.6, 0, len(M)], vmin=0, vmax=1)
        ax.set(xlabel="position (m)", ylabel="place cell (sorted by peak)",
               title=f"{'Rightward' if d > 0 else 'Leftward'} runs (n={len(M)})")
        plt.colorbar(im, ax=ax, fraction=0.046, label="norm. rate")

    # examples: well-isolated fields away from the track ends
    central = df[pc_mask & df.peak_pos.between(0.25, 1.35)]
    top = central.sort_values("spatial_info", ascending=False).head(6)
    for k, (_, r) in enumerate(top.iterrows()):
        ax = axes[k // 3, 1 + k % 3]
        for d, c in ((+1, "tab:blue"), (-1, "tab:red")):
            tc = gaussian_filter1d(np.nan_to_num(tcs[d][r.unit].values), 1.5, mode="nearest")
            ax.plot(centers, tc, color=c, lw=1.6,
                    label="rightward" if d > 0 else "leftward")
        ax.set_title(f"unit {int(r.unit)}  SI={r.spatial_info:.2f} b/spk", fontsize=9)
        ax.set_xlabel("position (m)")
        if k % 3 == 0:
            ax.set_ylabel("rate (Hz)")
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"CA1 place fields on the 1.6 m linear track - {res['label']}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- fig 3


def fig_phase_locking(res, pc_mask, fname="03_theta_phase_locking.png"):
    df, units, runs = res["df"], res["units"], res["runs"]
    spk_phase, filt, phase = res["spk_phase"], res["filt"], res["phase"]

    exc = df[(df.cell_type == "excitatory") & (df.n_spikes_run > 100)]
    inh = df[(df.cell_type == "inhibitory") & (df.n_spikes_run > 100)]
    # use a strongly locked place cell, and show a traversal on which it is active
    cand = df[pc_mask].sort_values("mrl", ascending=False)
    ex_unit = int(cand.iloc[0].unit)
    in_unit = int(inh.sort_values("mrl", ascending=False).iloc[0].unit) if len(inh) else None
    counts = [len(units[ex_unit].restrict(runs[i:i + 1])) for i in range(len(runs))]
    best_run = int(np.argmax(counts))
    st_best = units[ex_unit].restrict(runs[best_run:best_run + 1]).times()
    centre = float(np.median(st_best))

    fig = plt.figure(figsize=(14, 8.5))
    gs = GridSpec(2, 3, hspace=0.42, wspace=0.35, height_ratios=[1, 1.15])

    # A: raster of one cell's spikes on the filtered theta trace
    ax = fig.add_subplot(gs[0, :])
    seg = nap.IntervalSet(start=centre - 0.75, end=centre + 0.75)
    f = filt.restrict(seg)
    ax.plot(f.times() - seg.start[0], f.values * 1e3, "crimson", lw=1.4)
    amp = np.max(f.values) * 1e3
    for u, y, c, lab in ((ex_unit, 1.20, "k", "pyramidal"),
                         (in_unit, 1.45, "tab:blue", "interneuron")):
        if u is None:
            continue
        st = units[u].restrict(seg).times()
        ax.plot(st - seg.start[0], np.full_like(st, y * amp), "|",
                color=c, ms=12, mew=1.4, label=f"unit {u} ({lab}), {len(st)} spikes")
    ax.set_ylim(-1.3 * amp, 1.75 * amp)
    ax.legend(fontsize=8, loc="lower center", ncol=2, framealpha=0.95)
    ax.set(xlabel="time (s)", ylabel="theta-filtered LFP (mV)",
           title="Spikes ride on the theta cycle (single 1.5 s run segment)")

    # B/C: spike-phase histograms
    for k, (u, lab, col) in enumerate(((ex_unit, "place cell", "tab:green"),
                                       (in_unit, "interneuron", "tab:blue"))):
        ax = fig.add_subplot(gs[1, k])
        if u is None:
            continue
        ph = np.asarray(spk_phase[u].values)
        h, e, w = _phase_hist(ph)
        ax.bar(np.degrees(e), h, width=np.degrees(w), align="edge", color=col, alpha=0.85)
        tt = np.linspace(0, 720, 400)
        ax.plot(tt, h.max() * (0.5 + 0.45 * np.cos(np.radians(tt))), "crimson", lw=1.2,
                label="LFP theta (peak at 0/360 deg)")
        r = df[df.unit == u].iloc[0]
        ax.legend(fontsize=7, loc="upper right")
        ax.set(xlabel="theta phase (deg)", ylabel="P(spike)", xlim=(0, 720),
               title=f"unit {u} ({lab}): MRL={r.mrl:.2f}\n"
                     f"pref={np.degrees(r.pref_phase):.0f}$\\degree$, p={r.p_rayleigh:.0e}")
        ax.set_xticks([0, 180, 360, 540, 720])

    # D: population polar plot of preferred phases
    ax = fig.add_subplot(gs[1, 2], projection="polar")
    for sub, col, lab in ((exc, "tab:green", "pyramidal"), (inh, "tab:blue", "interneuron")):
        sig = sub[sub.p_rayleigh < 0.05]
        ax.plot(sig.pref_phase, sig.mrl, "o", color=col, ms=5, alpha=0.7,
                label=f"{lab} (n={len(sig)})")
    ax.set_title("Preferred theta phase vs locking\nstrength (Rayleigh p<0.05)",
                 pad=26, fontsize=10)
    ax.set_rlabel_position(112)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.32, -0.14), fontsize=8)

    fig.suptitle(f"Theta phase entrainment of CA1 spiking - {res['label']}", y=0.98)
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 4


def fig_theta_rhythmicity(res, pc_mask, fname="04_theta_rhythmicity.png"):
    df, units, runs = res["df"], res["units"], res["runs"]
    pc_units = df.loc[pc_mask, "unit"].values
    inh_units = df.loc[df.cell_type == "inhibitory", "unit"].values

    f_lfp = res["lfp_theta_freq"]
    acorr = nap.compute_autocorrelogram(units, binsize=0.005, windowsize=0.5, ep=runs)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))
    lags = acorr.index.values

    for ax, us, lab, col in ((axes[0], pc_units, "place cells", "tab:green"),
                             (axes[1], inh_units, "interneurons", "tab:blue")):
        A = acorr[us].values.T
        A = A / np.maximum(np.nanmax(A, axis=1, keepdims=True), 1e-9)
        ax.plot(lags, np.nanmean(A, axis=0), color=col, lw=2)
        ax.fill_between(lags, np.nanmean(A, 0) - np.nanstd(A, 0) / np.sqrt(len(A)),
                        np.nanmean(A, 0) + np.nanstd(A, 0) / np.sqrt(len(A)),
                        color=col, alpha=0.3)
        for k in (-2, -1, 1, 2):
            ax.axvline(k / f_lfp, color="crimson", ls="--", lw=0.9)
        ax.set(xlabel="lag (s)", ylabel="norm. autocorrelation",
               title=f"Mean spike autocorrelogram\n{lab} (n={len(us)}); dashed = LFP theta "
                     f"({f_lfp:.1f} Hz)")

    # theta modulation index and intrinsic oscillation frequency per cell
    dt = lags[1] - lags[0]
    mods, intrinsic = {}, {}
    for us, lab in ((pc_units, "place cells"), (inh_units, "interneurons")):
        vals, freqs = [], []
        for u in us:
            a = acorr[u].values.copy()
            a[np.abs(lags) < 0.01] = np.nan
            a = np.nan_to_num(a - np.nanmean(a))
            P = np.abs(np.fft.rfft(a, n=8 * len(a))) ** 2
            fr = np.fft.rfftfreq(8 * len(a), dt)
            band = (fr >= 5) & (fr <= 12)
            vals.append(P[band].max() / P[(fr >= 1) & (fr <= 50)].mean())
            freqs.append(fr[band][np.argmax(P[band])])
        mods[lab] = np.array(vals)
        intrinsic[lab] = np.array(freqs)

    ax = axes[2]
    hi = max(0.2, np.percentile(np.r_[mods["place cells"], mods["interneurons"]], 99))
    for lab, col in (("place cells", "tab:green"), ("interneurons", "tab:blue")):
        ax.hist(mods[lab], bins=np.linspace(0, hi, 25), alpha=0.6, color=col,
                label=lab, density=True)
    ax.set(xlabel="theta modulation index of autocorrelogram", ylabel="density",
           title="Spike-train theta rhythmicity")
    ax.legend(fontsize=8)

    # intrinsic spiking frequency vs LFP theta frequency (dual-oscillator signature)
    ax = axes[3]
    keep = mods["place cells"] > np.median(mods["place cells"])
    v = intrinsic["place cells"][keep]
    ax.hist(v, bins=np.arange(5, 12.25, 0.25), color="tab:green", alpha=0.75,
            label=f"place cells (top 50% modulated, n={keep.sum()})")
    ax.axvline(f_lfp, color="crimson", lw=2, label=f"LFP theta = {f_lfp:.2f} Hz")
    ax.axvline(np.median(v), color="k", ls="--", lw=1.5,
               label=f"median intrinsic = {np.median(v):.2f} Hz")
    ax.set(xlabel="intrinsic spiking oscillation frequency (Hz)", ylabel="# cells",
           title="Place cells oscillate slightly faster\nthan the LFP theta rhythm")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(f"Theta rhythmicity of CA1 spike trains during running - {res['label']}", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return mods, intrinsic, f_lfp


# --------------------------------------------------------------------- fig 5


def fig_precession_examples(res, pc_mask, fname="05_precession_examples.png", n=6):
    df, tcs, centers = res["df"], res["tcs"], res["centers"]
    prec = {p["unit"]: p for p in res["precession"]}
    cand = df[pc_mask & (df.prec_p_shuffle < 0.05) & (df.prec_slope < 0)]
    cand = cand.sort_values("prec_rho", key=lambda s: -s.abs()).head(n)

    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.9),
                             gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.62))
    for k, (_, r) in enumerate(cand.iterrows()):
        p = prec[int(r.unit)]
        d = p["direction"]
        f0, f1 = p["field"]
        ax = axes[0, k]
        tc = gaussian_filter1d(np.nan_to_num(tcs[d][r.unit].values), 1.5, mode="nearest")
        ax.plot(centers, tc, "k", lw=1.5)
        ax.axvspan(f0, f1, color="orange", alpha=0.25)
        ax.set_title(f"unit {int(r.unit)}, {'rightward' if d > 0 else 'leftward'} runs\n"
                     f"{len(p['x'])} in-field spikes", fontsize=9)
        ax.set_xlabel("position (m)", fontsize=8)
        if k == 0:
            ax.set_ylabel("rate (Hz)")

        ax = axes[1, k]
        x, ph = p["x"], p["phase"]
        for off in (0, 2 * np.pi):
            ax.plot(x, np.degrees(ph + off), ".", ms=2.5, color="tab:blue", alpha=0.45)
        xs = np.linspace(0, 1, 100)
        for off in (-2 * np.pi, 0, 2 * np.pi, 4 * np.pi):
            ax.plot(xs, np.degrees(p["slope"] * xs + p["phi0"] + off), "crimson", lw=2)
        ax.set(xlim=(0, 1), ylim=(0, 720),
               xlabel="normalized position in field\n(along direction of travel)")
        ax.set_yticks([0, 180, 360, 540, 720])
        ax.set_title(f"slope = {np.degrees(p['slope']):.0f}$\\degree$/field,  "
                     f"$\\rho$ = {p['rho']:.2f}\nshuffle p = {r.prec_p_shuffle:.3f}",
                     fontsize=9, pad=6)
        if k == 0:
            ax.set_ylabel("theta phase (deg)")
    fig.suptitle(f"Theta phase precession: example CA1 place cells - {res['label']}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- fig 6


def fig_precession_population(dfs, precs, fname="06_precession_population.png"):
    """Pooled across sessions. dfs: concatenated place-cell table; precs: list of dicts."""
    sig = dfs[dfs.prec_p_shuffle < 0.05]
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, hspace=0.45, wspace=0.42)

    # A: pooled phase-position density for significant precessing cells
    ax = fig.add_subplot(gs[0, 0])
    keys = set(zip(sig.session, sig.unit))
    X = np.concatenate([p["x"] for p in precs if (p["session"], p["unit"]) in keys])
    P = np.concatenate([p["phase"] for p in precs if (p["session"], p["unit"]) in keys])
    X2, P2 = np.r_[X, X], np.r_[P, P + 2 * np.pi]
    H, xe, ye = np.histogram2d(X2, P2, bins=[np.linspace(0, 1, 26),
                                             np.linspace(0, 4 * np.pi, 49)])
    H = H / H.sum(axis=1, keepdims=True)
    im = ax.imshow(gaussian_filter1d(H.T, 1, axis=0), aspect="auto", origin="lower",
                   extent=[0, 1, 0, 720], cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("P(phase|position)", fontsize=8)
    ax.set(xlabel="normalized position in field", ylabel="theta phase (deg)",
           title=f"Pooled spike phase vs position\n({len(keys)} precessing cells, {len(X)} spikes)")
    ax.set_yticks([0, 180, 360, 540, 720])

    # B: circular mean phase per position bin
    ax = fig.add_subplot(gs[0, 1])
    edges = np.linspace(0, 1, 11)
    mid = 0.5 * (edges[:-1] + edges[1:])
    mus, cis = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (X >= a) & (X < b)
        _, mu, _ = tl.rayleigh(P[m])
        mus.append(mu)
        cis.append(1.0 / np.sqrt(max(m.sum(), 1)))
    mus = np.unwrap(np.array(mus))
    mus = mus - mus[0] + np.array(mus)[0]
    for off in (0, 2 * np.pi):
        ax.plot(mid, np.degrees(np.mod(mus, 2 * np.pi) + off), "o-", color="tab:purple", lw=2)
    ax.set(xlabel="normalized position in field", ylabel="circular mean phase (deg)",
           ylim=(0, 720), xlim=(0, 1),
           title="Mean spike phase advances\nacross the place field")
    ax.set_yticks([0, 180, 360, 540, 720])

    # C: slope distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(np.degrees(dfs.prec_slope), bins=30, color="0.7", label="all place cells")
    ax.hist(np.degrees(sig.prec_slope), bins=30, color="tab:red", alpha=0.85,
            label="significant (shuffle p<0.05)")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="precession slope (deg per field traversal)", ylabel="# cells",
           title=f"Slopes are predominantly negative\n"
                 f"({(sig.prec_slope < 0).sum()}/{len(sig)} negative)")
    ax.legend(fontsize=8)

    # D: observed rho vs shuffle
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(dfs.shuf_rho95, np.abs(dfs.prec_rho), "o", ms=4, alpha=0.6, color="tab:blue")
    lim = np.nanmax([dfs.shuf_rho95.max(), np.abs(dfs.prec_rho).max()]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel="95th pct of shuffled |rho|", ylabel="observed |rho|", xlim=(0, lim),
           ylim=(0, lim), title="Circular-linear correlation\nexceeds the shuffle control")

    # E: precession slope vs field width
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(dfs.field_width * 100, np.degrees(dfs.prec_slope), "o", ms=4, alpha=0.35,
            color="0.5", label="all place cells")
    ax.plot(sig.field_width * 100, np.degrees(sig.prec_slope), "o", ms=4, alpha=0.8,
            color="tab:red", label="significant")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(-360, color="crimson", ls="--", lw=1, label="one full cycle")
    ax.set(xlabel="place field width (cm)", ylabel="precession slope (deg/field)",
           title="Phase advance per field traversal")
    ax.legend(fontsize=8)

    # F: per-session summary
    ax = fig.add_subplot(gs[1, 2])
    g = dfs.groupby("session")
    labels = list(g.groups.keys())
    tot = g.size().values
    nsig = g.apply(lambda d: int((d.prec_p_shuffle < 0.05).sum()), include_groups=False).values
    y = np.arange(len(labels))
    ax.barh(y, tot, color="0.8", label="place cells")
    ax.barh(y, nsig, color="tab:red", label="significant precession")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    for i, (a, b) in enumerate(zip(nsig, tot)):
        ax.text(b + 1.0, i, f"{a}/{b} ({100 * a / b:.0f}%)", va="center", fontsize=8)
    ax.set(xlabel="# cells", title="Phase precession across sessions",
           xlim=(0, tot.max() * 1.6), ylim=(-0.8, len(labels) - 0.2))
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Theta phase precession across the CA1 place-cell population "
                 f"(DANDI:000044, {dfs.session.nunique()} sessions)", y=0.98)
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_entrainment_population(dfall, mods_pooled, intrinsic_pooled, lfp_freqs,
                               fname="08_entrainment_population.png"):
    """Pooled phase-locking summary across sessions."""
    exc = dfall[(dfall.cell_type == "excitatory") & (dfall.n_spikes_run > 100)]
    inh = dfall[(dfall.cell_type == "inhibitory") & (dfall.n_spikes_run > 100)]

    fig = plt.figure(figsize=(17.5, 4.6))
    gs = GridSpec(1, 5, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(0.6, exc.mrl.max(), inh.mrl.max()), 30)
    ax.hist(exc.mrl, bins=bins, color="tab:green", alpha=0.65, density=True, label=f"pyramidal (n={len(exc)})")
    ax.hist(inh.mrl, bins=bins, color="tab:blue", alpha=0.65, density=True, label=f"interneuron (n={len(inh)})")
    ax.set(xlabel="mean resultant length (MRL)", ylabel="density",
           title="Theta locking strength")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    frac_e = (exc.p_rayleigh < 0.05).mean()
    frac_i = (inh.p_rayleigh < 0.05).mean()
    ax.bar([0, 1], [100 * frac_e, 100 * frac_i], color=["tab:green", "tab:blue"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pyramidal", "interneuron"])
    ax.set(ylabel="% significantly phase-locked", ylim=(0, 105),
           title="Rayleigh test, p < 0.05")
    for i, v in enumerate([frac_e, frac_i]):
        ax.text(i, 100 * v + 2, f"{100 * v:.0f}%", ha="center")

    ax = fig.add_subplot(gs[0, 2], projection="polar")
    for sub, col, lab in ((exc, "tab:green", "pyramidal"), (inh, "tab:blue", "interneuron")):
        s = sub[sub.p_rayleigh < 0.05]
        h, e = np.histogram(s.pref_phase, bins=18, range=(0, 2 * np.pi))
        ax.bar(e[:-1], h / h.sum(), width=e[1] - e[0], align="edge", color=col, alpha=0.55, label=lab)
    ax.set_title("Distribution of preferred phases", pad=26, fontsize=10)
    ax.set_rlabel_position(112)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.32, -0.15), fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    hi = np.nanpercentile(np.r_[mods_pooled["place cells"], mods_pooled["interneurons"]], 99)
    for lab, col in (("place cells", "tab:green"), ("interneurons", "tab:blue")):
        ax.hist(mods_pooled[lab], bins=np.linspace(0, hi, 25),
                alpha=0.6, color=col, density=True, label=lab)
    ax.set(xlabel="theta modulation index", ylabel="density",
           title="Spike-train theta rhythmicity")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 4])
    v = intrinsic_pooled["place cells"]
    ax.hist(v, bins=np.arange(5, 12.25, 0.25), color="tab:green", alpha=0.75,
            label=f"place cells (n={len(v)})")
    for i, f in enumerate(lfp_freqs):
        ax.axvline(f, color="crimson", lw=1.5, alpha=0.8,
                   label="LFP theta (per session)" if i == 0 else None)
    ax.axvline(np.median(v), color="k", ls="--", lw=1.6,
               label=f"median intrinsic {np.median(v):.2f} Hz")
    ax.set(xlabel="intrinsic spiking frequency (Hz)", ylabel="# cells",
           title="Intrinsic oscillation exceeds\nLFP theta frequency")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.9)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Theta entrainment across {dfall.session.nunique()} CA1 sessions "
                 "(DANDI:000044)", y=1.04)
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 7


def fig_single_runs(res, pc_mask, fname="07_single_run_precession.png"):
    """Phase vs position for one cell, with individual traversals colour-coded."""
    df = res["df"]
    prec = {p["unit"]: p for p in res["precession"]}
    cand = df[pc_mask & (df.prec_p_shuffle < 0.05) & (df.prec_slope < 0)]
    cand = cand.sort_values("prec_rho", key=lambda s: -s.abs())
    u = int(cand.iloc[0].unit)
    p = prec[u]
    d = p["direction"]
    ep = res["run_ep"][d]
    spk_pos = res["spk_pos"][u].restrict(ep)
    spk_ph = res["spk_phase"][u].restrict(ep)
    f0, f1 = p["field"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), gridspec_kw=dict(wspace=0.32))

    t, pp, ph = spk_pos.times(), spk_pos.values, spk_ph.values
    run_id = np.full(len(t), -1)
    for k, (s, e) in enumerate(zip(ep.start, ep.end)):
        run_id[(t >= s) & (t <= e)] = k
    m = (pp >= f0) & (pp <= f1)
    # distance travelled into the field, always along the direction of running
    xin_all = (pp - f0) if d > 0 else (f1 - pp)

    ax = axes[0]
    sc = ax.scatter(xin_all[m] * 100, np.degrees(ph[m]), c=run_id[m], cmap="turbo", s=10)
    plt.colorbar(sc, ax=ax, label="traversal #")
    ax.set(xlabel="distance into place field (cm)", ylabel="theta phase (deg)", ylim=(0, 360),
           title=f"unit {u}: every in-field spike,\ncoloured by traversal")
    ax.set_yticks([0, 90, 180, 270, 360])

    # single traversals with their own circular-linear fits
    ax = axes[1]
    order = np.argsort([-(m & (run_id == k)).sum() for k in range(len(ep))])
    kept, cols = 0, plt.cm.tab10.colors
    for k in order:
        mm = m & (run_id == k)
        if mm.sum() < 8 or kept >= 6:
            continue
        xk = (xin_all[mm] - xin_all[mm].min()) / max(f1 - f0, 1e-9)
        ak, b0, _, _ = tl.circ_lin_regression(xk, ph[mm])
        c = cols[kept % len(cols)]
        for off in (0, 2 * np.pi):
            ax.plot(xin_all[mm] * 100, np.degrees(ph[mm] + off), "o", ms=5, color=c, alpha=0.8)
        xs = np.linspace(0, 1, 50)
        for off in (0, 2 * np.pi, 4 * np.pi):
            ax.plot((xs * (f1 - f0) + xin_all[mm].min()) * 100,
                    np.degrees(ak * xs + b0 + off), "-", color=c, lw=1.4, alpha=0.9)
        kept += 1
    ax.set(xlabel="distance into place field (cm)", ylabel="theta phase (deg)", ylim=(0, 720),
           xlim=(0, (f1 - f0) * 100),
           title=f"{kept} individual traversals,\neach with its own circular-linear fit")
    ax.set_yticks([0, 180, 360, 540, 720])

    ax = axes[2]
    for off in (0, 2 * np.pi):
        ax.plot(p["x"], np.degrees(p["phase"] + off), ".", ms=3, color="tab:blue", alpha=0.5)
    xs = np.linspace(0, 1, 100)
    for off in (-2 * np.pi, 0, 2 * np.pi, 4 * np.pi):
        ax.plot(xs, np.degrees(p["slope"] * xs + p["phi0"] + off), "crimson", lw=2)
    ax.set(xlim=(0, 1), ylim=(0, 720), xlabel="normalized position in field",
           ylabel="theta phase (deg)",
           title=f"Circular-linear fit\nslope={np.degrees(p['slope']):.0f}$\\degree$, "
                 f"$\\rho$={p['rho']:.2f}")
    ax.set_yticks([0, 180, 360, 540, 720])
    fig.suptitle(f"Single-traversal phase precession - {res['label']}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{fname}", dpi=140, bbox_inches="tight")
    plt.close(fig)
