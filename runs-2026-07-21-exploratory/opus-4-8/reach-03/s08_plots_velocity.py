"""Figures for velocity (direction x speed) tuning."""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from s01_load import load_all

CACHE, FIG = "cache", "figures"

if __name__ == "__main__":
    d = load_all()
    df = pd.read_pickle(f"{CACHE}/dir_tuning.pkl")
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    LAG = float(np.load(f"{CACHE}/best_lag.npy")[0])
    tc2d = np.load(f"{CACHE}/tc2d.npy")                    # (units, nvx, nvy)
    b2d = np.load(f"{CACHE}/tc2d_bins.npy", allow_pickle=True)
    TC = np.load(f"{CACHE}/tc_dirspeed.npy")               # (units, ndir, nspd)
    bins = np.load(f"{CACHE}/tc_dirspeed_bins.npy", allow_pickle=True)
    occ = np.load(f"{CACHE}/occ_dirspeed.npy")
    occ2d = np.load(f"{CACHE}/occ2d.npy")
    lags = np.load(f"{CACHE}/lags.npy"); R2 = np.load(f"{CACHE}/lag_r2.npy")
    ok = np.load(f"{CACHE}/lag_ok.npy"); bl = np.load(f"{CACHE}/best_lag_per_unit.npy")
    dctr, sctr = bins[0], bins[1]
    ex = df.sort_values('r2_str', ascending=False).unit.values[:6]

    # ---------------- Fig 5: 2D velocity tuning maps ----------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    vx, vy = b2d[0], b2d[1]
    ext = [vx[0], vx[-1], vy[0], vy[-1]]
    mask2d = occ2d > 1000                       # >= 1 s of hand-velocity data per bin
    for ax, u in zip(axes.ravel()[:6], ex):
        M = np.where(mask2d, tc2d[u], np.nan).T
        im = ax.imshow(M, origin='lower', extent=ext, cmap='viridis', aspect='equal')
        pdv = df.loc[u, 'pref_dir_cont']
        ax.arrow(0, 0, 650 * np.cos(pdv), 650 * np.sin(pdv), color='crimson',
                 width=20, head_width=80, length_includes_head=True)
        ax.set_title(f"unit {u}", fontsize=10)
        ax.set_xlabel('$v_x$ (mm/s)'); ax.set_ylabel('$v_y$ (mm/s)')
        plt.colorbar(im, ax=ax, fraction=.046, label='Hz')
    # occupancy of the velocity plane
    axo = axes.ravel()[6]
    im = axo.imshow(occ.T, origin='lower', cmap='magma', aspect='auto',
                    extent=[-180, 180, sctr[0], sctr[-1]])
    axo.set_xlabel('movement direction (deg)'); axo.set_ylabel('speed (mm/s)')
    axo.set_title('sampling of the direction $\\times$ speed plane', fontsize=10)
    plt.colorbar(im, ax=axo, fraction=.046, label='samples (1 kHz)')
    axc = axes.ravel()[7]
    axc.scatter(K.dir_end * 180 / np.pi, K.peak_speed, s=3, alpha=.25, c='k')
    axc.set_xlabel('reach direction (deg)'); axc.set_ylabel('peak speed (mm/s)')
    axc.set_title('trial level: peak speed is confounded\nwith direction (target geometry)',
                  fontsize=10)
    fig.suptitle(f'Hand-velocity tuning during movement (neural activity leads velocity by '
                 f'{LAG*1000:.0f} ms; red arrow = preferred direction)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{FIG}/fig05_velocity_maps.png", dpi=140)
    plt.close(fig)

    # ---------------- Fig 6: lag scan ----------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    pop = np.nanmean(R2, 1)
    ax[0].plot(lags * 1000, pop, 'k', lw=2)
    ax[0].axvline(lags[np.argmax(pop)] * 1000, color='crimson', ls='--')
    ax[0].axvline(0, color='gray', lw=.8)
    ax[0].set_xlabel('lag (ms):  neural activity leads velocity $\\rightarrow$')
    ax[0].set_ylabel('mean $R^2$ (linear velocity model)')
    ax[0].set_title(f'population optimum at {lags[np.argmax(pop)]*1000:.0f} ms')
    top = np.argsort(np.nanmax(R2, 0))[-12:]
    for u in top:
        ax[1].plot(lags * 1000, R2[:, u] / np.nanmax(R2[:, u]), lw=.9, alpha=.8)
    ax[1].axvline(0, color='gray', lw=.8)
    ax[1].set_xlabel('lag (ms)'); ax[1].set_ylabel('$R^2$ (normalised per unit)')
    ax[1].set_title('12 most velocity-related units')
    ax[2].hist(bl * 1000, bins=np.arange(-305, 306, 20), color='steelblue')
    ax[2].axvline(np.median(bl) * 1000, color='crimson', ls='--',
                  label=f'median {np.median(bl)*1000:.0f} ms')
    ax[2].set_xlabel('per-unit optimal lag (ms)'); ax[2].set_ylabel('# units')
    ax[2].legend(); ax[2].set_title(f'n={ok.sum()} units with $R^2$>0.02')
    fig.suptitle('Motor-cortical activity leads hand velocity', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{FIG}/fig06_lag_scan.png", dpi=140)
    plt.close(fig)

    # ---------------- Fig 7: speed modulation ----------------
    okb = occ > 200
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, hspace=.55, wspace=.35)
    for i, u in enumerate(ex[:4]):
        M = np.where(okb, TC[u], np.nan)
        a1 = fig.add_subplot(gs[0, i])
        im = a1.pcolormesh(np.degrees(dctr), sctr, M.T, cmap='viridis', shading='nearest')
        a1.set_xlabel('direction (deg)')
        if i == 0:
            a1.set_ylabel('speed (mm/s)')
        a1.set_title(f'unit {u}', fontsize=10)
        plt.colorbar(im, ax=a1, fraction=.046, label='Hz' if i == 3 else None)
        a2 = fig.add_subplot(gs[1, i])
        pdv = df.loc[u, 'pref_dir_cont']
        jp = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - pdv))))))
        ja = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - pdv - np.pi))))))
        a2.plot(sctr, M[jp], 'o-', color='crimson', label='preferred dir.')
        a2.plot(sctr, M[ja], 'o-', color='steelblue', label='opposite dir.')
        a2.set_xlabel('speed (mm/s)'); a2.set_ylabel('firing rate (Hz)')
        if i == 0:
            a2.legend(fontsize=8)
    sig = df.p_str < 0.01
    a = fig.add_subplot(gs[2, 0])
    a.scatter(df.speed_slope_anti[sig], df.speed_slope_pd[sig], s=16, c='k', alpha=.6)
    lim = np.nanpercentile(np.r_[df.speed_slope_pd[sig], df.speed_slope_anti[sig]], [1, 99])
    a.plot(lim, lim, 'r--', lw=1); a.axhline(0, lw=.6, c='gray'); a.axvline(0, lw=.6, c='gray')
    a.set_xlabel('slope, opposite dir.'); a.set_ylabel('slope, preferred dir.')
    a.set_title('speed slope (Hz per 100 mm/s)', fontsize=10)
    a = fig.add_subplot(gs[2, 1])
    a.hist([df.speed_slope_pd[sig].dropna(), df.speed_slope_anti[sig].dropna()], bins=20,
           color=['crimson', 'steelblue'], label=['preferred', 'opposite'])
    a.axvline(0, c='k', lw=.8); a.legend(fontsize=8)
    a.set_xlabel('speed slope (Hz per 100 mm/s)'); a.set_ylabel('# units')
    a.set_title(f'median {df.speed_slope_pd[sig].median():.2f} vs '
                f'{df.speed_slope_anti[sig].median():.2f}', fontsize=10)
    a = fig.add_subplot(gs[2, 2:])
    # population-average normalised tuning, aligned to each unit's preferred direction
    nd = len(dctr)
    aligned = np.full((TC.shape[0], nd, len(sctr)), np.nan)
    for u in range(TC.shape[0]):
        sh = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - df.pref_dir_cont[u]))))))
        M = np.where(okb, TC[u], np.nan)
        mx = np.nanmax(M)
        if mx > 0:
            aligned[u] = np.roll(M, -sh, axis=0) / mx
    A = np.nanmean(aligned[sig.values], 0)
    rel = np.degrees(np.angle(np.exp(1j * (dctr - dctr[0]))))
    order = np.argsort(np.angle(np.exp(1j * (dctr - dctr[0]))))
    for si in range(len(sctr)):
        a.plot(np.sort(rel), A[order, si], 'o-', lw=1.4,
               color=plt.get_cmap('plasma')(si / (len(sctr) - 1)),
               label=f'{sctr[si]:.0f} mm/s')
    a.set_xlabel('direction relative to each unit\'s preferred direction (deg)')
    a.set_ylabel('normalised firing rate')
    a.legend(fontsize=8, ncol=2, title='speed', title_fontsize=8)
    a.set_title(f'population-average tuning grows with speed (n={sig.sum()} tuned units)',
                fontsize=10)
    fig.suptitle('Speed scales the gain of directional tuning', fontsize=13, y=.98)
    fig.savefig(f"{FIG}/fig07_speed_gain.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("figs 5-7 saved")
