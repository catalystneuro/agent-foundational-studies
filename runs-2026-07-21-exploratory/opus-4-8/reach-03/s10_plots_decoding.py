"""Figures for GLM model comparison and population decoding."""
import numpy as np, pandas as pd
np.seterr(all='ignore')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

CACHE, FIG = "cache", "figures"

if __name__ == "__main__":
    D = np.load(f"{CACHE}/decoding.npz")
    B = np.load(f"{CACHE}/decoding_bayes.npz")
    true, pred, tb, trial = D['true'], D['pred'], D['tb'], D['trial']

    # ---------------- Fig 8: decoding ----------------
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.32)
    a = fig.add_subplot(gs[0, :])
    seg = (tb > 300) & (tb < 340)
    ts, tr_, pr_ = tb[seg], true[seg], pred[seg]
    brk = np.r_[False, np.diff(ts) > 0.12]          # break the line between reaches
    ts = np.where(brk, np.nan, ts)
    a.plot(ts, tr_[:, 0], 'k', lw=1.8, label='true')
    a.plot(ts, pr_[:, 0], 'crimson', lw=1.5, label='decoded $v_x$')
    a.plot(ts, tr_[:, 1] - 1600, 'k', lw=1.8)
    a.plot(ts, pr_[:, 1] - 1600, 'steelblue', lw=1.5, label='decoded $v_y$')
    a.text(ts[np.isfinite(ts)][0], 700, '$v_x$', fontsize=11)
    a.text(ts[np.isfinite(ts)][0], -900, '$v_y$', fontsize=11)
    a.set_xlabel('time (s); gaps are inter-reach intervals, which are excluded')
    a.set_ylabel('velocity (mm/s), $v_y$ offset')
    a.legend(ncol=3, fontsize=9)
    a.set_title(f"cross-validated ridge decoding of hand velocity from 182 units "
                f"($R^2$: $v_x$={D['r2'][0]:.2f}, $v_y$={D['r2'][1]:.2f})")
    for i, (lab, col) in enumerate([('$v_x$', 'crimson'), ('$v_y$', 'steelblue')]):
        ax = fig.add_subplot(gs[1, i])
        idx = np.random.default_rng(0).choice(len(true), 4000, replace=False)
        ax.scatter(true[idx, i], pred[idx, i], s=3, alpha=.15, c=col)
        lim = [-1000, 1000]
        ax.plot(lim, lim, 'k--', lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'true {lab} (mm/s)'); ax.set_ylabel(f'decoded {lab} (mm/s)')
        ax.set_title(f"{lab}:  $R^2$={D['r2'][i]:.2f}")
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.degrees(D['ang_err']), bins=36, color='seagreen', alpha=.75,
            label=f"ridge (median {np.degrees(np.median(D['ang_err'])):.0f}$\\degree$)")
    ax.hist(np.degrees(B['err']), bins=36, color='darkorange', alpha=.6,
            label=f"Bayesian (median {np.degrees(np.median(B['err'])):.0f}$\\degree$)")
    ax.axvline(90, color='k', ls='--', lw=1, label='chance median (90$\\degree$)')
    ax.set_xlabel('absolute direction error (deg)'); ax.set_ylabel('# time bins')
    ax.legend(fontsize=8); ax.set_title('decoded movement direction')
    fig.savefig(f"{FIG}/fig08_decoding.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 9: GLM model comparison ----------------
    import os
    if not os.path.exists(f"{CACHE}/glm_scores.npz"):
        print("GLM scores not ready"); raise SystemExit
    S = np.load(f"{CACHE}/glm_scores.npz")
    sc = {k: S[k].mean(0) for k in S.files}
    order = ['dir', 'speed', 'add', 'full']
    names = {'dir': 'direction', 'speed': 'speed', 'add': 'direction + speed',
             'full': 'direction $\\times$ speed'}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    parts = ax[0].violinplot([sc[k] for k in order], showmedians=True)
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels([names[k] for k in order], fontsize=9, rotation=12, ha='right')
    ax[0].set_ylabel('cross-validated pseudo-$R^2$')
    ax[0].set_title('nested Poisson GLMs (NeMoS), 50 ms bins')
    ax[0].axhline(0, color='k', lw=.6)
    for i, k in enumerate(order):
        ax[0].text(i + 1, np.median(sc[k]), f"  {np.median(sc[k]):.3f}", fontsize=8, va='bottom')
    ax[1].scatter(sc['dir'], sc['add'], s=16, c='k', alpha=.6)
    lim = [min(sc['dir'].min(), 0), max(sc['add'].max(), sc['dir'].max()) * 1.05]
    ax[1].plot(lim, lim, 'r--', lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel('direction only'); ax[1].set_ylabel('direction + speed')
    w = wilcoxon(sc['add'], sc['dir'])
    ax[1].set_title(f"{np.mean(sc['add']>sc['dir']):.0%} of units improve when speed\n"
                    f"is added (Wilcoxon p={w.pvalue:.1e})")
    g1 = sc['add'] - sc['dir']
    g2 = sc['full'] - sc['add']
    ax[2].hist([g1, g2], bins=25, color=['mediumpurple', 'darkorange'],
               label=['+ speed (separable)', '+ direction $\\times$ speed interaction'])
    ax[2].axvline(0, color='k', lw=.8)
    ax[2].set_xlabel('gain in cross-validated pseudo-$R^2$'); ax[2].set_ylabel('# units')
    ax[2].legend(fontsize=8)
    ax[2].set_title(f"separable speed term: median +{np.median(g1):.4f}\n"
                    f"further interaction: median +{np.median(g2):.4f}")
    fig.suptitle('Speed carries information beyond direction; because of the exponential link '
                 'the separable model\nis already multiplicative in rate, and the explicit '
                 'interaction adds little', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{FIG}/fig09_glm_models.png", dpi=140)
    plt.close(fig)
    print("figs 8-9 saved")
