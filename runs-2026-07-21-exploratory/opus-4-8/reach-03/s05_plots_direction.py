"""Figures for reach-direction tuning."""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import pynapple as nap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from s01_load import load_all, to_pynapple
from s03_direction_tuning import MOVE_WIN

CACHE, FIG = "cache", "figures"
NBINS = 8
EDGES = np.linspace(-np.pi, np.pi, NBINS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
CMAP = plt.get_cmap('hsv')
COLORS = [CMAP(((c + np.pi) % (2*np.pi)) / (2*np.pi)) for c in CENTERS]


def dir_bin(theta):
    return np.clip(np.digitize(theta, EDGES) - 1, 0, NBINS - 1)


def psth_by_direction(spk_unit, onsets, dbin, win=(-0.6, 0.6), bs=0.02):
    """Return (time, rate[nbins, ntime], spike-time lists per direction bin)."""
    ts = nap.Ts(t=onsets)
    pe = nap.compute_perievent(spk_unit, ts, window=win)
    edges = np.arange(win[0], win[1] + bs, bs)
    tc = edges[:-1] + bs / 2
    rates = np.zeros((NBINS, len(tc)))
    rasters = [[] for _ in range(NBINS)]
    keys = sorted(pe.keys())
    for k in keys:
        b = dbin[k]
        s = pe[k].index.values
        rasters[b].append(s)
        rates[b] += np.histogram(s, edges)[0]
    for b in range(NBINS):
        n = max(len(rasters[b]), 1)
        rates[b] /= (n * bs)
    return tc, rates, rasters


if __name__ == "__main__":
    d = load_all(); spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    df = pd.read_pickle(f"{CACHE}/dir_tuning.pkl")
    rate_mv = np.load(f"{CACHE}/rate_mv.npy")
    straight = np.load(f"{CACHE}/straight.npy")
    theta = K.dir_end.values
    db = dir_bin(theta)
    on = K.t_onset.values

    # ---------------- Fig 2: raster + PSTH for two example units ----------------
    ex = df.sort_values('r2_str', ascending=False).unit.values[:2]
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[2.2, 1])
    MAXTR = 30                      # trials shown per direction band, for legibility
    rng = np.random.default_rng(1)
    for j, u in enumerate(ex):
        sel = np.where(straight)[0]
        tcx, rates, rasters = psth_by_direction(spk[u], on[sel], db[sel])
        nonempty = [b for b in range(NBINS) if len(rasters[b]) > 0]
        axr = fig.add_subplot(gs[0, j]); y = 0; ticks = []
        for b in nonempty:
            rs = rasters[b]
            if len(rs) > MAXTR:
                rs = [rs[i] for i in rng.choice(len(rs), MAXTR, replace=False)]
            y0 = y
            for s in rs:
                axr.plot(s, np.full_like(s, y), '|', ms=3.5, color=COLORS[b],
                         markeredgewidth=.9)
                y += 1
            ticks.append((y0 + y) / 2)
            axr.axhline(y, color='k', lw=.4, alpha=.3)
        axr.axvline(0, color='k', ls='--', lw=1)
        axr.set_xlim(-0.6, 0.6); axr.set_ylim(0, y)
        axr.set_yticks(ticks)
        axr.set_yticklabels([f"{np.degrees(CENTERS[b]):.0f}$\\degree$" for b in nonempty],
                            fontsize=8)
        axr.set_ylabel('reach direction')
        axr.set_title(f"unit {u}  |  cosine $R^2$={df.loc[u,'r2_str']:.2f}, "
                      f"PD={np.degrees(df.loc[u,'pref_dir_str']):.0f}$\\degree$", pad=8)
        axp = fig.add_subplot(gs[1, j])
        for b in nonempty:
            axp.plot(tcx, rates[b], color=COLORS[b], lw=1.4,
                     label=f"{np.degrees(CENTERS[b]):.0f}$\\degree$")
        axp.axvline(0, color='k', ls='--', lw=1)
        axp.axvspan(*MOVE_WIN, color='gray', alpha=.15)
        axp.set_xlim(-0.6, 0.6); axp.set_xlabel('time from movement onset (s)')
        axp.set_ylabel('firing rate (Hz)')
        if j == 1:
            axp.legend(fontsize=7, ncol=2, title='reach dir.', title_fontsize=7,
                       loc='upper left', bbox_to_anchor=(1.01, 1.05))
    fig.suptitle('Direction-tuned responses on unobstructed reaches (shaded = analysis window)', y=0.98)
    fig.savefig(f"{FIG}/fig02_raster_psth.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 3: polar tuning curves ----------------
    top = df.sort_values('r2_str', ascending=False).unit.values[:8]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5),
                             subplot_kw=dict(projection='polar'))
    thgrid = np.linspace(-np.pi, np.pi, 200)
    for ax, u in zip(axes.ravel(), top):
        r = rate_mv[straight, u]; th = theta[straight]
        b = dir_bin(th)
        nb = np.array([(b == i).sum() for i in range(NBINS)])
        m = np.array([r[b == i].mean() if nb[i] else np.nan for i in range(NBINS)])
        se = np.array([r[b == i].std() / np.sqrt(nb[i]) if nb[i] else np.nan
                       for i in range(NBINS)])
        row = df.loc[u]
        # cosine fit on straight trials, for the smooth overlay curve
        X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        curve = beta[0] + beta[1] * np.cos(thgrid) + beta[2] * np.sin(thgrid)
        rmax = np.nanmax([np.nanmax(m + se), curve.max()]) * 1.12
        ax.plot(np.r_[CENTERS, CENTERS[0]], np.r_[m, m[0]], 'o-', color='k', ms=4, lw=1.2)
        ax.errorbar(CENTERS, m, yerr=se, fmt='none', ecolor='k', capsize=2, lw=1)
        ax.plot(thgrid, np.clip(curve, 0, None), color='crimson', lw=1.8)
        ax.annotate('', xy=(row.pref_dir_str, rmax * .95), xytext=(row.pref_dir_str, 0),
                    arrowprops=dict(arrowstyle='-|>', color='navy', lw=1.6))
        ax.set_ylim(0, rmax)
        ax.set_rlabel_position(285)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_title(f"unit {u}   $R^2$={row.r2_str:.2f}   MDI={row.mdi_str:.2f}",
                     fontsize=10, pad=22)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.4)
    fig.suptitle('Reach-direction tuning curves on unobstructed reaches\n'
                 'black = binned mean $\\pm$ SEM (Hz)   |   red = cosine fit   |   '
                 'blue arrow = preferred direction', y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4.0, w_pad=2.0)
    fig.savefig(f"{FIG}/fig03_polar_tuning.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 4: population summary ----------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    sig = df.p_str < 0.01
    ax[0, 0].hist([df.r2_str[sig], df.r2_str[~sig]], bins=25, stacked=True,
                  color=['steelblue', 'lightgray'], label=['p<0.01', 'n.s.'])
    ax[0, 0].set_xlabel('cosine $R^2$ (straight reaches)'); ax[0, 0].set_ylabel('# units')
    ax[0, 0].legend(); ax[0, 0].set_title(f'{sig.mean():.0%} of {len(df)} units significantly tuned')

    axp = fig.add_subplot(2, 3, 2, projection='polar'); ax[0, 1].remove()
    axp.hist(df.pref_dir_str[sig], bins=18, color='steelblue')
    axp.set_title('preferred directions\n(significantly tuned units)', pad=18, fontsize=10)

    ax[0, 2].hist(df.mdi_str[sig & np.isfinite(df.mdi_str)], bins=25, color='seagreen')
    ax[0, 2].set_xlabel('modulation depth index  (depth / baseline)')
    ax[0, 2].set_ylabel('# units'); ax[0, 2].set_title('tuning strength')

    ax[1, 0].scatter(df.r2, df.r2_str, s=14, c='k', alpha=.6)
    lim = [0, max(df.r2_str.max(), df.r2.max()) * 1.05]
    ax[1, 0].plot(lim, lim, 'r--', lw=1); ax[1, 0].set_xlim(lim); ax[1, 0].set_ylim(lim)
    ax[1, 0].set_xlabel('$R^2$, all trials (incl. maze/curved)')
    ax[1, 0].set_ylabel('$R^2$, unobstructed reaches')
    ax[1, 0].set_title('endpoint direction explains straight\nreaches better than curved ones')

    ax[1, 1].scatter(df.r2_plan, df.r2, s=14, c='k', alpha=.6)
    ax[1, 1].set_xlabel('$R^2$, late-delay (planning) window')
    ax[1, 1].set_ylabel('$R^2$, peri-movement window')
    ax[1, 1].set_title('planning vs execution tuning')

    dd = np.angle(np.exp(1j * (df.pref_dir_plan - df.pref_dir)))
    both = (df.r2 > 0.02) & (df.r2_plan > 0.02)     # tuning detectable in both epochs
    n = int(both.sum()); z = np.exp(1j * dd[both]).mean()
    R = np.abs(z); Z = n * R ** 2
    pray = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n))
    ax[1, 2].hist(np.degrees(dd[both]), bins=18, color='darkorange')
    ax[1, 2].axvline(np.degrees(np.angle(z)), color='k', ls='--', lw=1.5)
    ax[1, 2].set_xlabel('PD(plan) $-$ PD(move)  (deg)'); ax[1, 2].set_ylabel('# units')
    ax[1, 2].set_title(f'PD partially preserved from planning to execution\n'
                       f'n={n}, mean $\\Delta$={np.degrees(np.angle(z)):.0f}$\\degree$, '
                       f'R={R:.2f}, Rayleigh p={pray:.3f}')
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig04_population_direction.png", dpi=140)
    plt.close(fig)
    print("figs 2-4 saved")
