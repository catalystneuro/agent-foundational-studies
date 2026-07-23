"""Figures for the theta entrainment / precession analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from theta_analysis import THETA_BAND, rayleigh

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130})

CDIR = {1: "#c0392b", -1: "#2471a3"}
DIRNAME = {1: "left to right", -1: "right to left"}


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------
def fig_behavior(S, name="fig01_behaviour_and_spiking.png"):
    pos, laps, lap_dir, speed = S["pos"], S["laps"], S["lap_dir"], S["speed"]
    units = S["units"]
    fig = plt.figure(figsize=(11, 7.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.32,
                  height_ratios=[1, 1, 1.1])

    ax = fig.add_subplot(gs[0, :])
    for (s, e), d in zip(zip(laps.start, laps.end), lap_dir):
        seg = pos.get(s, e)
        ax.plot(seg.t, seg.d, color=CDIR[d], lw=1.0)
    ax.set_ylabel("linearized\nposition (cm)")
    ax.set_title(f"{S['label']}: {len(laps)} track traversals on the 1.6 m linear track")
    ax.set_xlabel("time (s)")
    for d in (1, -1):
        ax.plot([], [], color=CDIR[d], label=DIRNAME[d])
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # zoom on a few laps with the spike raster underneath
    t0 = laps.start[6]
    t1 = laps.end[min(11, len(laps) - 1)]
    ax = fig.add_subplot(gs[1, :])
    seg = pos.get(t0, t1)
    ax.plot(seg.t, seg.d, "k.", ms=2)
    ax.set_ylabel("position (cm)")
    ax.set_xlim(t0, t1)
    ax.set_title("expanded view of six consecutive traversals")

    # a single traversal: place cells ordered by field position tile the run
    fwd = np.where(lap_dir == 1)[0]
    dur = laps.end[fwd] - laps.start[fwd]
    ilap = fwd[np.argsort(dur)[len(fwd) // 2]]  # a typical forward traversal
    a, b = laps.start[ilap] - 0.2, laps.end[ilap] + 0.2
    fields = S["df"][S["df"].direction == 1]
    ax = fig.add_subplot(gs[2, :2])
    for _, r in fields.iterrows():
        st = units[r.unit].get(a, b).t
        ax.plot(st - a, np.full_like(st, r.field_peak), "|", color="k",
                ms=5, mew=0.9)
    seg = pos.get(a, b)
    ax.plot(seg.t - a, seg.d, color="#c0392b", lw=1.6, label="animal position")
    ax.set_xlim(0, b - a)
    ax.set_xlabel("time from start of traversal (s)")
    ax.set_ylabel("place-field peak\nposition (cm)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("one traversal: place cells fire as the animal enters their field",
                 fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(speed.restrict(laps).d, bins=40, color="0.4")
    ax.set_xlabel("running speed (cm/s)")
    ax.set_ylabel("samples")
    ax.set_title("speed during traversals")
    save(fig, name)


# --------------------------------------------------------------------------
def fig_theta(S, name="fig02_theta_lfp.png"):
    lfp, theta, phase = S["lfp"], S["theta"], S["phase"]
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.28,
                  height_ratios=[1.2, 1, 1])

    t0 = S["laps"].start[6]
    ax = fig.add_subplot(gs[0, :])
    seg_r, seg_t = lfp.get(t0, t0 + 2), theta.get(t0, t0 + 2)
    ax.plot(seg_r.t - t0, seg_r.d, color="0.6", lw=0.8, label="raw LFP")
    ax.plot(seg_t.t - t0, seg_t.d, color="#c0392b", lw=1.5,
            label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz")
    ax.set_ylabel("LFP (mV)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"CA1 LFP during running (channel {S['best_ch']})")
    ax2 = ax.twinx()
    seg_p = phase.get(t0, t0 + 2)
    ax2.plot(seg_p.t - t0, seg_p.d, color="#2471a3", lw=0.6, alpha=0.45)
    ax2.set_ylabel("theta phase (deg)", color="#2471a3")
    ax2.set_yticks([0, 180, 360])
    ax.set_xlabel(f"time from start of a traversal (s), t0 = {t0:.1f} s")

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(S["freqs"], S["psd_run"], color="#c0392b", label="running")
    ax.semilogy(S["freqs"], S["psd_still"], color="0.4", label="rest of maze epoch")
    ax.axvspan(*THETA_BAND, color="#f1c40f", alpha=0.25)
    ax.set_xlim(0, 30)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (mV$^2$/Hz)")
    ax.legend(fontsize=8)
    ax.set_title("LFP spectrum")

    ax = fig.add_subplot(gs[1, 1])
    sc = S["chan_scan"]
    ax.plot(sc[:, 0], sc[:, 2], ".-", color="0.3", lw=0.7, ms=3)
    ax.axvline(S["best_ch"], color="#c0392b")
    ax.set_xlabel("LFP channel")
    ax.set_ylabel("theta / delta ratio")
    ax.set_title("channel selection (running periods)")

    # theta-cycle triggered average and population spike histogram
    ax = fig.add_subplot(gs[2, :])
    ph = np.asarray(phase.d)
    trough = np.where((ph[:-1] < 180) & (ph[1:] >= 180))[0]
    w = 100
    trough = trough[(trough > w) & (trough < len(ph) - w)]
    idx = trough[:, None] + np.arange(-w, w)[None, :]
    lag = np.arange(-w, w) / 1250 * 1e3
    ax.plot(lag, np.asarray(lfp.d)[idx].mean(0), color="k", label="raw LFP")
    ax.plot(lag, np.asarray(theta.d)[idx].mean(0), color="#c0392b",
            label="theta filtered")
    ax.axvline(0, color="0.5", ls=":")
    ax.legend(fontsize=8)
    ax.set_xlabel("time from theta trough (ms)")
    ax.set_ylabel("mean LFP (mV)")
    ax.set_title(f"trough-triggered average LFP ({len(trough)} cycles)")
    save(fig, name)


# --------------------------------------------------------------------------
def fig_place_fields(S, name="fig03_place_fields.png"):
    df = S["df"]
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.55)

    for k, dsign in enumerate((1, -1)):
        ax = fig.add_subplot(gs[0, k])
        sub = df[df.direction == dsign]
        tc = S["tcs"][dsign][sub.unit.values].values
        tc = tc / np.maximum(tc.max(0, keepdims=True), 1e-9)
        order = np.argsort(np.argmax(tc, axis=0))
        im = ax.imshow(tc[:, order].T, aspect="auto", origin="lower",
                       extent=[0, S["track_len"], 0, tc.shape[1]], cmap="viridis")
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("place field #")
        ax.set_title(f"{DIRNAME[dsign]}\n({tc.shape[1]} fields)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("norm. rate", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df.spatial_info, bins=25, color="0.4")
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel("fields")
    ax.set_title("spatial information")

    # a few example rate maps
    mid = df[(df.field_peak > 25) & (df.field_peak < 135)]
    best = mid.sort_values("spatial_info", ascending=False).head(9)
    best = best.iloc[np.argsort(best.field_peak.values)[[0, len(best) // 2, -1]]]
    axs = [fig.add_subplot(gs[1, j]) for j in range(3)]
    for ax, (_, r) in zip(axs, best.iterrows()):
        d = int(r.direction)
        ax.plot(S["centers"], S["tcs"][d][r.unit].values, color=CDIR[d])
        ax.axvspan(r.field_start, r.field_stop, color=CDIR[d], alpha=0.15)
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("rate (Hz)")
        ax.set_title(f"unit {int(r.unit)} {DIRNAME[d]}\nSI={r.spatial_info:.2f} bits/spk",
                     fontsize=8)
    save(fig, name)


# --------------------------------------------------------------------------
def _phase_hist_ax(ax, phases_deg, color, title, nbins=24):
    h, edges = np.histogram(phases_deg, bins=nbins, range=(0, 360))
    h = h / h.sum()
    c = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(np.r_[c, c + 360], np.r_[h, h], width=360 / nbins,
           color=color, alpha=0.85)
    xx = np.linspace(0, 720, 400)
    ax.plot(xx, h.max() * (0.45 + 0.35 * np.cos(np.deg2rad(xx))), "k-", lw=1, alpha=0.6)
    ax.set_xlim(0, 720)
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.set_xlabel("theta phase (deg)")
    ax.set_ylabel("P(spike)")
    ax.set_title(title, fontsize=8)


def fig_entrainment(S, name="fig04_theta_entrainment.png"):
    df = S["df"]
    ex = S["examples"]
    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.42)

    top = df[df.n_spikes_field >= 150].sort_values("mrl", ascending=False).head(3)
    for j, (_, r) in enumerate(top.iterrows()):
        key = (r.unit, int(r.direction))
        if key not in ex:
            continue
        ax = fig.add_subplot(gs[0, j])
        _phase_hist_ax(
            ax, ex[key]["phase"], CDIR[int(r.direction)],
            f"unit {int(r.unit)} ({DIRNAME[int(r.direction)]})\n"
            f"MRL={r.mrl:.2f}, pref={r.pref_phase:.0f}$\\degree$, "
            f"p={r.p_rayleigh:.1e}",
        )

    ax = fig.add_subplot(gs[1, 0], projection="polar")
    sig = df[df.p_rayleigh < 0.05]
    ang = np.deg2rad(sig.pref_phase.values)
    ax.scatter(ang, sig.mrl.values, s=12, c=[CDIR[d] for d in sig.direction],
               alpha=0.75)
    mu, R, p, n = rayleigh(ang)
    ax.annotate("", xy=(mu, R), xytext=(0, 0),
                arrowprops=dict(color="k", width=2, headwidth=7))
    ax.set_title(f"preferred phase of each field\n(mean {np.rad2deg(mu)%360:.0f}"
                 f"$\\degree$, R={R:.2f}, p={p:.1e})", fontsize=8, pad=32)
    ax.set_rlabel_position(135)

    ax = fig.add_subplot(gs[1, 1])
    h, edges = np.histogram(sig.pref_phase, bins=18, range=(0, 360))
    c = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(np.r_[c, c + 360], np.r_[h, h], width=20, color="#7d3c98")
    ax.set_xlim(0, 720)
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.set_xlabel("preferred theta phase (deg)")
    ax.set_ylabel("# fields")
    ax.set_title(f"population preferred phase\n({len(sig)}/{len(df)} fields, "
                 f"Rayleigh p<0.05)", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(df.mrl, bins=np.linspace(0, 0.9, 30), color="#c0392b", alpha=0.8,
            label="observed")
    ax.hist(df.mrl_jitter, bins=np.linspace(0, 0.9, 30), color="0.5", alpha=0.7,
            label="spike-jitter control")
    ax.set_xlabel("mean resultant length")
    ax.set_ylabel("# fields")
    ax.legend(fontsize=8)
    ax.set_title("locking strength vs. control", fontsize=9)

    # pooled spike-phase histogram over all fields
    ax = fig.add_subplot(gs[2, :2])
    allph = np.concatenate([v["phase"] for v in ex.values()])
    _phase_hist_ax(ax, allph, "#7d3c98",
                   f"all in-field spikes pooled (n={allph.size})", nbins=36)
    mu, R, p, n = rayleigh(np.deg2rad(allph))
    ax.axvline(np.rad2deg(mu) % 360, color="k", ls="--")
    ax.axvline(np.rad2deg(mu) % 360 + 360, color="k", ls="--")
    ax.set_title(f"all in-field spikes pooled (n={allph.size}), "
                 f"mean phase {np.rad2deg(mu)%360:.0f}$\\degree$, R={R:.3f}",
                 fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(df.mrl_jitter, df.mrl, s=10, c="k", alpha=0.6)
    lim = [0, max(df.mrl.max(), df.mrl_jitter.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MRL, jittered spikes")
    ax.set_ylabel("MRL, observed")
    ax.set_title("per-field comparison", fontsize=9)
    save(fig, name)


# --------------------------------------------------------------------------
def fig_precession_examples(S, name="fig05_precession_examples.png"):
    df = S["df"].dropna(subset=["rho"])
    ex = S["examples"]
    sig = df[df.p_shuffle < 0.05].sort_values("rho")
    sel = pd.concat([sig[sig.direction == 1].head(3), sig[sig.direction == -1].head(3)])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
        key = (r.unit, int(r.direction))
        e = ex[key]
        x, ph = e["x"], e["phase"]
        ax.scatter(np.r_[x, x], np.r_[ph, ph + 360], s=7,
                   c=CDIR[int(r.direction)], alpha=0.6, edgecolors="none")
        xx = np.linspace(0, 1, 200)
        yy = np.rad2deg(r.phi0 + 2 * np.pi * r.slope * xx)
        k0 = int(np.floor((0 - yy.max()) / 360))
        k1 = int(np.ceil((720 - yy.min()) / 360))
        for k in range(k0, k1 + 1):  # straight lines, no modulo wrap artefacts
            ax.plot(xx, yy + 360 * k, "k-", lw=1.4)
        ax.set_ylim(0, 720)
        ax.set_yticks([0, 180, 360, 540, 720])
        ax.set_xlim(0, 1)
        ax.set_xlabel("normalized position in field")
        ax.set_ylabel("theta phase (deg)")
        ax.set_title(
            f"unit {int(r.unit)}, {DIRNAME[int(r.direction)]}\n"
            f"slope {r.slope*360:.0f}$\\degree$/field, $\\rho$={r.rho:.2f}, "
            f"p={r.p_shuffle:.3f} (n={int(r.n_spikes_field)})", fontsize=8)
    fig.suptitle("Theta phase precession in single CA1 place fields", y=1.0)
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------
def fig_precession_population(S, name="fig06_precession_population.png"):
    df = S["df"].dropna(subset=["rho"])
    ex = S["examples"]
    sig = df[df.p_shuffle < 0.05]
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-3, 3, 31)
    ax.hist(df.slope, bins=bins, color="0.75", label="all fields")
    ax.hist(sig.slope, bins=bins, color="#c0392b", label="p<0.05")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("slope (theta cycles per field)")
    ax.set_ylabel("# fields")
    ax.legend(fontsize=8)
    ax.set_title(f"slopes: {(sig.slope<0).sum()}/{len(sig)} negative", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df.rho, bins=np.linspace(-0.8, 0.8, 33), color="0.75",
            label="all fields")
    ax.hist(sig.rho, bins=np.linspace(-0.8, 0.8, 33), color="#c0392b",
            label="p<0.05")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("circular-linear correlation $\\rho$")
    ax.set_ylabel("# fields")
    ax.legend(fontsize=8)
    ax.set_title(f"median $\\rho$ = {df.rho.median():.2f}", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(df.field_width, df.slope * 360, s=12, c="k", alpha=0.6)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("field width (cm)")
    ax.set_ylabel("slope (deg/field)")
    ax.set_ylim(-1100, 1100)
    ax.set_title("slope vs. field size", fontsize=9)

    # pooled density of phase against normalized position
    ax = fig.add_subplot(gs[1, :2])
    X = np.concatenate([ex[(r.unit, int(r.direction))]["x"] for _, r in sig.iterrows()])
    P = np.concatenate([ex[(r.unit, int(r.direction))]["phase"] for _, r in sig.iterrows()])
    H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360],
                               bins=[20, 40], range=[[0, 1], [0, 720]])
    from scipy.ndimage import gaussian_filter1d

    H = H / H.sum(axis=1, keepdims=True)
    H = gaussian_filter1d(H, 1.2, axis=1, mode="wrap")
    im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, 1, 0, 720])
    ax.set_xlabel("normalized position in field")
    ax.set_ylabel("theta phase (deg)")
    ax.set_yticks([0, 180, 360, 540, 720])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("P(phase | position)", fontsize=8)
    ax.set_title(f"pooled spike density, {len(sig)} precessing fields "
                 f"(n={X.size} spikes)", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    ctr = 0.5 * (xe[1:] + xe[:-1])
    mu = []
    for i in range(len(ctr)):
        m = (X >= xe[i]) & (X < xe[i + 1])
        mu.append(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(P[m]))))) % 360)
    mu = np.array(mu)
    mu_un = np.unwrap(np.deg2rad(mu))
    mu_un = np.rad2deg(mu_un - mu_un[0]) + mu[0]
    ax.plot(ctr, mu_un, "ko-", ms=4)
    sl = np.polyfit(ctr, mu_un, 1)[0]
    ax.plot(ctr, np.polyval(np.polyfit(ctr, mu_un, 1), ctr), "r-")
    ax.set_xlabel("normalized position in field")
    ax.set_ylabel("mean theta phase (deg, unwrapped)")
    ax.set_title(f"population phase advance\n{sl:.0f}$\\degree$ across the field",
                 fontsize=9)
    save(fig, name)


# --------------------------------------------------------------------------
def fig_sessions(all_df, name="fig07_across_sessions.png"):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    g = all_df.dropna(subset=["rho"]).groupby("session")

    frac = g.apply(lambda d: (d.p_shuffle < 0.05).mean(), include_groups=False)
    axes[0].bar(range(len(frac)), frac.values, color="#c0392b")
    axes[0].set_xticks(range(len(frac)))
    axes[0].set_xticklabels(frac.index, rotation=60, ha="right", fontsize=7)
    axes[0].set_ylabel("fraction of fields")
    axes[0].set_title("significant phase precession", fontsize=9)

    med = g["slope"].median() * 360
    axes[1].bar(range(len(med)), med.values, color="#2471a3")
    axes[1].set_xticks(range(len(med)))
    axes[1].set_xticklabels(med.index, rotation=60, ha="right", fontsize=7)
    axes[1].set_ylabel("median slope (deg/field)")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("precession slope", fontsize=9)

    ax = axes[2]
    for k, (s, d) in enumerate(all_df.groupby("session")):
        ang = np.deg2rad(d[d.p_rayleigh < 0.05].pref_phase.values)
        mu, R, p, n = rayleigh(ang)
        ax.plot([np.rad2deg(mu) % 360], [R], "o")
        ax.annotate(s, (np.rad2deg(mu) % 360, R), fontsize=6,
                    xytext=(4, 6 if k % 2 else -10), textcoords="offset points")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("mean preferred phase (deg)")
    ax.set_ylabel("population R")
    ax.set_title("theta entrainment per session", fontsize=9)

    ax = axes[3]
    for s, d in all_df.dropna(subset=["rho"]).groupby("session"):
        ax.hist(np.clip(d.slope * 360, -720, 720),
                bins=np.linspace(-720, 720, 25), histtype="step", label=s, lw=1.2)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("slope (deg/field), clipped to $\\pm$720")
    ax.set_ylabel("# fields")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("slope distributions", fontsize=9)
    fig.tight_layout()
    save(fig, name)
