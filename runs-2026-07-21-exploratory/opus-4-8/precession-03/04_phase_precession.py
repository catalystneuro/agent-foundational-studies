"""Theta phase precession: theta phase vs. position within the place field.

For each place cell we take its spikes during running in its preferred
direction, keep those inside the place field, and read the animal's position
and the LFP theta phase at each spike time. Phase precession appears as a
negative relationship between position-in-field and theta phase, quantified
with the Kempter et al. (2012) circular-linear regression.

Saves precession_results.npz, fig_03_example_cells.png, fig_04_population.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

MAZE_LEN = 1.6
N_BINS = 50
FIELD_FRAC = 0.25   # field = contiguous region around peak with rate >= FIELD_FRAC*peak
MIN_SPIKES = 30     # min in-field spikes to fit a cell

# ---------- circular-linear regression (Kempter et al. 2012) ----------
def circ_lin_regress(x, phi):
    """x in [0,1] (position in field), phi in radians.
    Returns slope (cycles across field), phase offset (rad), rho, p-value."""
    x = np.asarray(x); phi = np.asarray(phi)
    # search slope up to +-2 cycles across the field
    a_grid = np.linspace(-4*np.pi, 4*np.pi, 2001)
    R = np.array([np.abs(np.mean(np.exp(1j*(phi - a*x)))) for a in a_grid])
    a = a_grid[np.argmax(R)]
    phi0 = np.angle(np.mean(np.exp(1j*(phi - a*x))))
    # circular-linear correlation (signed)
    theta = (a*x) % (2*np.pi)
    phibar = np.angle(np.mean(np.exp(1j*phi)))
    thetabar = np.angle(np.mean(np.exp(1j*theta)))
    sp = np.sin(phi - phibar); st = np.sin(theta - thetabar)
    rho = np.sum(sp*st) / np.sqrt(np.sum(sp**2) * np.sum(st**2))
    # p-value (Jammalamadaka & SenGupta approximation)
    n = x.size
    l20 = np.mean(sp**2); l02 = np.mean(st**2); l22 = np.mean(sp**2*st**2)
    z = rho * np.sqrt(n * l20 * l02 / l22)
    from scipy.stats import norm
    pval = 2*(1 - norm.cdf(abs(z)))
    slope_cycles = a/(2*np.pi)   # cycles of theta across the field
    return slope_cycles, phi0, rho, pval

def field_bounds(rate, bins, peak_pos):
    """Contiguous region around the peak bin where rate >= FIELD_FRAC*peak."""
    r = rate.copy()
    ipk = np.argmin(np.abs(bins - peak_pos))
    thr = FIELD_FRAC * r[ipk]
    lo = ipk
    while lo > 0 and r[lo-1] >= thr:
        lo -= 1
    hi = ipk
    while hi < len(r)-1 and r[hi+1] >= thr:
        hi += 1
    return bins[lo], bins[hi]

def main():
    d = np.load('loaded_data.npz', allow_pickle=True)
    p = np.load('preprocessed.npz', allow_pickle=True)
    pf = np.load('place_fields.npz', allow_pickle=True)

    pos = nap.Tsd(t=p['pos_t'], d=p['pos_data'])
    theta_ph = nap.Tsd(t=p['lfp_t'], d=p['theta_phase'])
    spikes = {int(u): nap.Ts(t=st) for u, st in zip(d['unit_ids'], d['spike_times'])}

    bins = pf['bins']
    tc_uids = list(pf['tc_uids'])
    tc = {'right': pf['tc_right'], 'left': pf['tc_left']}
    ep = {'right': nap.IntervalSet(pf['right_starts'], pf['right_ends']),
          'left': nap.IntervalSet(pf['left_starts'], pf['left_ends'])}

    results = []
    for uid, bdir, ppos in zip(pf['place_uid'], pf['place_dir'], pf['place_peakpos']):
        col = tc_uids.index(uid)
        rate = tc[bdir][:, col]
        f0, f1 = field_bounds(rate, bins, ppos)
        if f1 - f0 < 0.08:      # need a field of reasonable width
            continue
        # spikes in preferred-direction running, inside the field
        sp = spikes[int(uid)].restrict(ep[bdir])
        spos = sp.value_from(pos)
        infield = (spos.values >= f0) & (spos.values <= f1)
        sp_t = spos.index.values[infield]
        sp_pos = spos.values[infield]
        if sp_t.size < MIN_SPIKES:
            continue
        # theta phase at each spike
        sp_phase = np.interp(sp_t, theta_ph.index.values, np.unwrap(theta_ph.values))
        sp_phase = np.mod(sp_phase, 2*np.pi)
        # normalized position within field (flip so 0=entry for the run direction)
        xn = (sp_pos - f0) / (f1 - f0)
        if bdir == 'left':       # leftward runs traverse field from high to low pos
            xn = 1 - xn
        slope, phi0, rho, pval = circ_lin_regress(xn, sp_phase)
        results.append(dict(uid=int(uid), bdir=bdir, f0=f0, f1=f1,
                            n=sp_t.size, slope=slope, phi0=phi0, rho=rho, pval=pval,
                            xn=xn, phase=sp_phase, peak=rate.max()))

    print(f"Fit {len(results)} place cells with >= {MIN_SPIKES} in-field spikes")
    slopes = np.array([r['slope'] for r in results])
    # sign the circular-linear correlation by the regression slope so that
    # precessing cells (negative slope) show a negative correlation, per convention
    rhos = np.array([np.sign(r['slope'])*abs(r['rho']) for r in results])
    pvals = np.array([r['pval'] for r in results])
    sig = pvals < 0.05
    print(f"Significant circ-lin correlation (p<0.05): {sig.sum()}/{len(results)}")
    print(f"Negative slope (precessing): {(slopes<0).sum()}/{len(results)} "
          f"({100*(slopes<0).mean():.0f}%)")
    print(f"Among significant cells, negative slope: {(slopes[sig]<0).sum()}/{sig.sum()}")
    print(f"Median slope: {np.median(slopes):.2f} cycles/field; "
          f"median rho: {np.median(rhos):.3f}")

    # ---------- fig 3: example precessing cells ----------
    # pick 6 strongest precessing cells: significant, negative slope, largest |rho|
    cand = [i for i in range(len(results)) if pvals[i] < 0.05 and slopes[i] < 0]
    order = sorted(cand, key=lambda i: -abs(rhos[i]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), order):
        r = results[i]
        # plot two theta cycles (0-720 deg) for visual clarity
        ph_deg = np.degrees(r['phase'])
        for rep in (0, 360):
            ax.plot(r['xn'], ph_deg + rep, '.', ms=4, color='0.3', alpha=0.5)
        xx = np.linspace(0, 1, 100)
        yc = np.degrees(r['phi0']) + 360*r['slope']*xx   # regression line (deg)
        for rep in (-360, 0, 360, 720):
            ax.plot(xx, yc + rep, 'r-', lw=1.5)
        ax.set_ylim(0, 720); ax.set_xlim(0, 1)
        ax.set_title(f"unit {r['uid']} ({r['bdir']})\n"
                     f"slope={r['slope']:.2f} cyc, ρ={rhos[i]:.2f}, p={r['pval']:.1e}",
                     fontsize=9)
        ax.set_xlabel('position in field'); ax.set_ylabel('theta phase (deg)')
        ax.set_yticks([0, 180, 360, 540, 720])
    fig.suptitle('Theta phase precession in individual CA1 place cells', fontsize=13)
    plt.tight_layout()
    plt.savefig('fig_03_example_cells.png', dpi=130)
    print("Saved fig_03_example_cells.png")

    # ---------- fig 4: population ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (a) slope distribution
    ax[0].hist(slopes, bins=np.arange(-3, 1.6, 0.25), color='steelblue', edgecolor='k')
    ax[0].axvline(0, color='r', ls='--')
    ax[0].axvline(np.median(slopes), color='k', ls='-',
                  label=f'median={np.median(slopes):.2f}')
    ax[0].set_xlabel('slope (theta cycles across field)')
    ax[0].set_ylabel('# place cells'); ax[0].legend()
    ax[0].set_title(f'(a) Slopes: {100*(slopes<0).mean():.0f}% negative')
    # (b) rho distribution, significant highlighted
    ax[1].hist(rhos, bins=np.arange(-0.8, 0.85, 0.1), color='0.8', edgecolor='k', label='all')
    ax[1].hist(rhos[sig], bins=np.arange(-0.8, 0.85, 0.1), color='crimson',
               edgecolor='k', label='p<0.05')
    ax[1].axvline(0, color='k', ls='--')
    ax[1].set_xlabel('circular-linear correlation ρ')
    ax[1].set_ylabel('# place cells'); ax[1].legend()
    ax[1].set_title('(b) Phase-position correlation')
    # (c) pooled phase vs position density (significant precessing cells)
    allxn = np.concatenate([r['xn'] for i, r in enumerate(results)
                            if pvals[i] < 0.05 and slopes[i] < 0])
    allph = np.concatenate([np.degrees(r['phase']) for i, r in enumerate(results)
                            if pvals[i] < 0.05 and slopes[i] < 0])
    xn2 = np.concatenate([allxn, allxn]); ph2 = np.concatenate([allph, allph+360])
    h = ax[2].hist2d(xn2, ph2, bins=[25, 40], cmap='magma')
    plt.colorbar(h[3], ax=ax[2], label='# spikes')
    ax[2].set_xlabel('position in field'); ax[2].set_ylabel('theta phase (deg)')
    ax[2].set_title('(c) Pooled spikes (precessing cells)')
    ax[2].set_yticks([0, 180, 360, 540, 720])
    fig.suptitle('Population summary of theta phase precession (CA1, DANDI:000044 Achilles)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('fig_04_population.png', dpi=130)
    print("Saved fig_04_population.png")

    np.savez_compressed('precession_results.npz',
        uid=np.array([r['uid'] for r in results]),
        slope=slopes, rho=rhos, pval=pvals, n=np.array([r['n'] for r in results]))

if __name__ == '__main__':
    main()
