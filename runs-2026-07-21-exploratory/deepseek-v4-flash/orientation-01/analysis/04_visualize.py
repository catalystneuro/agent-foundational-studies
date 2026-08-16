"""Visualization suite for the Allen Visual Coding orientation-selectivity analysis.

Reads analysis/tuning_results_s715093703.npz (per-unit gOSI, OSI, preferred
orientation, permutation p-values) plus the raw static-gratings spike counts,
and streams spike times for a handful of example VISp units via LINDI. Writes
publication-style figures into analysis/figures/:

- activity_timeseries.png      : VISp population activity across the session
                                 (rate heatmap + per-trial unit traces)
- example_tuning_curves.png    : mean +/- SEM rate vs orientation at each
                                 unit's preferred spatial frequency
- example_raster_psth.png      : trial raster + PSTH for three selective VISp
                                 units aligned to grating onset
- population_selectivity.png   : per-region distributions of gOSI and OSI
- selectivity_fraction.png     : fraction of orientation-selective units by
                                 region (permutation p < 0.05) with bootstrap CI
- pref_orientation_hist.png    : circular histograms of preferred orientation
- gosi_vs_rate.png             : selectivity vs peak and blank firing rate

Uses the Agg backend throughout; nothing is displayed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RUN = 'analysis'
SID = '715093703'
ORIS = np.array([0., 30., 60., 90., 120., 150.])
FIGDIR = f'{RUN}/figures'
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})

REGIONS = ['VISp', 'VISl', 'VISam', 'VISpm', 'VISrl', 'LGd']
COLORS = plt.cm.tab10(np.linspace(0, 1, len(REGIONS)))


# ---------------------------------------------------------------- data load
def load_results():
    r = np.load(f'{RUN}/tuning_results_s{SID}.npz', allow_pickle=False)
    return {k: r[k] for k in r.files}


def load_static():
    d = np.load(f'{RUN}/static_s{SID}.npz', allow_pickle=False)
    out = {k: d[k] for k in d.files}
    # unify keys: npz uses orientation/spatial_freq, helpers use orient/sf
    out['orient'] = out['orientation']
    out['sf'] = out['spatial_freq']
    return out


def stream_session():
    """Open the NWB via LINDI; return (nwbfile, valid static windows)."""
    import lindi
    from pynwb import NWBHDF5IO
    url = ('https://lindi.neurosift.org/dandi/dandisets/'
           '000021/assets/58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json')
    local_cache = lindi.LocalCache(cache_dir='/tmp/lindi_cache_orient2')
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    nwbfile = NWBHDF5IO(file=f).read()
    df = nwbfile.intervals['static_gratings_presentations'].to_dataframe(
        exclude={'timeseries'}).reset_index(drop=True)
    ons = df['start_time'].values
    offs = df['stop_time'].values
    keep = np.ones(len(df), dtype=bool)
    inv = nwbfile.intervals['invalid_times'].to_dataframe(
        exclude={'timeseries'}).reset_index(drop=True)
    for _, rr in inv.iterrows():
        keep &= ~((ons < rr.stop_time) & (offs >= rr.start_time))
    idx = np.where(keep)[0]
    orient = pd.to_numeric(df['orientation'].values[idx],
                           errors='coerce').astype(float)
    sf = pd.to_numeric(df['spatial_frequency'].values[idx],
                       errors='coerce').astype(float)
    return nwbfile, dict(onset=ons[idx], offset=offs[idx],
                         orient=orient, sf=sf)


# ------------------------------------------------------------ example units
def pick_examples(res, n=5, seed=21):
    """Good, responsive, strongly selective VISp units."""
    visp = (res['region'] == 'VISp') & res['good'] & res['responsive']
    cand = np.where(visp & np.isfinite(res['gosi']) & (res['gosi'] >= 0.5)
                    & (res['gosi_perm_p'] < 0.05))[0]
    if len(cand) < n:
        cand = np.where(visp & np.isfinite(res['gosi'])
                        & (res['gosi_perm_p'] < 0.05))[0]
    rng = np.random.default_rng(seed)
    return cand[rng.permutation(len(cand))[:n]]


def sfs_sorted(d):
    return np.sort(np.unique(d['sf'][np.isfinite(d['sf'])]))


def pref_sf_value(res, d, u):
    return sfs_sorted(d)[int(res['pref_sf'][u])]


def curve_at_sf(res, d, u, sf_val):
    """Mean +/- SEM rate per orientation at one SF for one unit."""
    dur = d['offset'] - d['onset']
    rates = d['counts'][u] / dur
    ok = np.isfinite(d['orient']) & (d['sf'] == sf_val)
    mean, sem = [], []
    for o in ORIS:
        sel = ok & (d['orient'] == o)
        v = rates[sel]
        mean.append(v.mean())
        sem.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(mean), np.array(sem)


def region_mask(res, region):
    return res['good'] & (res['region'] == region) & np.isfinite(res['gosi'])


# ---------------------------------------------------------------- figures
def fig_activity_timeseries(res, d):
    """VISp population response rate over the session + example traces."""
    rates = d['counts'] / (d['offset'] - d['onset'])[None, :]
    good = res['good'] & (res['region'] == 'VISp')
    v1 = rates[good]
    order = np.argsort(v1[:, :400].mean(axis=1))

    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    lim = 500
    t0 = d['onset'][:lim] - d['onset'][0]
    im = ax0.imshow(v1[order][:, :lim], aspect='auto', cmap='viridis',
                    extent=[t0[0], t0[-1], 0, v1.shape[0]],
                    interpolation='nearest')
    ax0.set_ylabel('VISp unit (sorted by mean rate)')
    ax0.set_title('Per-trial firing rate: VISp population, first ~1.5 min')
    cb = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.02)
    cb.set_label('rate (Hz)')

    ax1 = fig.add_subplot(gs[1])
    for i in range(12):
        ax1.plot(t0, v1[order[i], :lim], lw=0.7, alpha=0.8)
    ax1.set_xlabel('time since session start (s)')
    ax1.set_ylabel('rate (Hz)')
    ax1.set_title('Example VISp units: per-trial firing rate across '
                  'consecutive static-grating presentations')
    fig.savefig(f'{FIGDIR}/activity_timeseries.png', bbox_inches='tight')
    plt.close(fig)


def fig_example_tuning(res, d):
    """4 example VISp units: mean +/- SEM tuning at preferred SF."""
    units = pick_examples(res, n=4)
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8))
    for ax_, u in zip(axs.ravel(), units):
        sf_val = pref_sf_value(res, d, u)
        mean, sem = curve_at_sf(res, d, u, sf_val)
        ax_.errorbar(ORIS, mean, yerr=sem, fmt='o-', capsize=3, lw=1.5)
        ax_.axhline(res['blank_rate'][u], color='0.6', ls='--', lw=1)
        ax_.axvline(res['pref_ori'][u], color='0.35', ls=':', lw=1)
        ax_.set_title(f'unit {u}  (gOSI={res["gosi"][u]:.2f}, '
                      f'pref SF={sf_val:.2f} c/deg)')
        ax_.set_xlabel('orientation (deg)')
        ax_.set_ylabel('rate (Hz)')
        ax_.set_xticks(ORIS)
    fig.suptitle('Orientation tuning of example VISp units (mean +/- SEM over trials)')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/example_tuning_curves.png', bbox_inches='tight')
    plt.close(fig)


def fig_example_raster_psth(nwbfile, d, res):
    """Raster + PSTH for three selective VISp units at their preferred SF."""
    units = pick_examples(res, n=6)[:3]
    pre, post = -0.05, 0.45
    binw = 0.01
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 2, width_ratios=[1.5, 1], hspace=0.55, wspace=0.2)
    for i, u in enumerate(units):
        sp = np.asarray(nwbfile.units['spike_times'][u]).astype(np.float64)
        sf_val = pref_sf_value(res, d, u)
        ok = np.isfinite(d['orient']) & (d['sf'] == sf_val)
        trials = np.where(ok)[0]
        order = np.argsort(d['orient'][trials])
        trials = trials[order]

        axr = fig.add_subplot(gs[i, 0])
        for j, tr in enumerate(trials):
            ons = d['onset'][tr]
            win = sp[(sp >= ons + pre) & (sp < ons + post)]
            axr.plot(win - ons, np.full_like(win, j), 'k|', ms=3, mew=0.6)
        axr.set_ylim(len(trials) - 0.5, -0.5)
        axr.set_xlim(pre, post)
        axr.set_yticks([])
        axr.set_xlabel('time from onset (s)')
        axr.set_ylabel(f'unit {u}\n({len(trials)} trials)')

        axp = fig.add_subplot(gs[i, 1])
        edges = np.arange(pre, post + binw, binw)
        psth = np.zeros(len(edges) - 1)
        for tr in trials:
            ons0 = d['onset'][tr]
            ts = sp[(sp >= ons0 + pre) & (sp < ons0 + post)] - ons0
            psth += np.histogram(ts, bins=edges)[0]
        psth = psth / (len(trials) * binw)
        axp.bar(edges[:-1] + binw / 2, psth, width=binw * 0.9,
                color='0.35', edgecolor='none')
        axp.set_xlim(pre, post)
        axp.set_xlabel('time from onset (s)')
        axp.set_ylabel('rate (Hz)')
        axp.set_title('PSTH (10 ms bins)')
    fig.suptitle('Single-trial responses to static gratings at preferred spatial '
                 'frequency (trials sorted by orientation)')
    fig.savefig(f'{FIGDIR}/example_raster_psth.png', bbox_inches='tight')
    plt.close(fig)


def fig_population_selectivity(res):
    """Per-region histograms of gOSI and OSI (good units only)."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, xlab in [(axs[0], 'gosi', 'global OSI  |Σ r e^{2iθ}| / Σ r'),
                          (axs[1], 'osi', 'orthogonal-pair OSI')]:
        for reg, c in zip(REGIONS, COLORS):
            m = region_mask(res, reg)
            ax.hist(res[key][m], bins=np.linspace(0, 1, 26), alpha=0.55,
                    color=c, label=f'{reg} (n={m.sum()})')
        ax.set_xlabel(xlab)
        ax.set_ylabel('units')
        ax.legend(fontsize=8)
    axs[0].set_title('Orientation selectivity across visual regions')
    axs[1].set_title('Orthogonal-pair selectivity')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/population_selectivity.png', bbox_inches='tight')
    plt.close(fig)


def fig_selectivity_fraction(res):
    """Fraction of selective good units per region (perm p<0.05) +/- bootstrap."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rng = np.random.default_rng(0)
    nboot = 200
    fracs, lo, hi = [], [], []
    for reg in REGIONS:
        m = region_mask(res, reg)
        p = res['gosi_perm_p'][m]
        fin = np.isfinite(p)
        obs = p[fin] < 0.05
        boot = np.array([np.mean(rng.choice(obs, size=len(obs), replace=True))
                         for _ in range(nboot)])
        fracs.append(obs.mean())
        lo.append(np.percentile(boot, 2.5))
        hi.append(np.percentile(boot, 97.5))
    fracs, lo, hi = map(np.array, (fracs, lo, hi))
    x = np.arange(len(REGIONS))
    ax.bar(x, fracs, color='0.65', edgecolor='k',
           yerr=[fracs - lo, hi - fracs], capsize=4)
    for xi, f in zip(x, fracs):
        ax.text(xi, f + 0.03, f'{f:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('fraction orientation-selective')
    ax.set_title('Permutation test on full gQSI statistic (p < 0.05); 95% bootstrap CI')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/selectivity_fraction.png', bbox_inches='tight')
    plt.close(fig)


def fig_pref_hist(res):
    """Circular histograms of preferred orientation for selected regions."""
    regs = ['VISp', 'VISl', 'LGd']
    fig, axs = plt.subplots(1, len(regs), figsize=(11, 4),
                            subplot_kw=dict(projection='polar'))
    for ax, reg in zip(axs, regs):
        m = region_mask(res, reg)
        pref = res['pref_ori'][m]
        pref = pref[np.isfinite(pref)]
        bins = np.linspace(0, 180, 19)
        counts, _ = np.histogram(pref % 180, bins=bins)
        theta = np.deg2rad(bins[:-1] + 9)
        ax.bar(theta, counts, width=np.deg2rad(18), alpha=0.7)
        ax.set_title(f'{reg} n={len(pref)}')
        ax.set_xticks(np.deg2rad([0, 45, 90, 135]))
        ax.set_xticklabels(['0', '45', '90', '135'])
    fig.suptitle('Preferred orientations cover all angles', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/pref_orientation_hist.png', bbox_inches='tight')
    plt.close(fig)


def fig_gosi_vs_rate(res, d):
    """Scatter: gQSI vs peak rate and vs blank rate for VISp units."""
    m = region_mask(res, 'VISp')
    gosi = res['gosi'][m]
    sel = res['gosi_perm_p'][m] < 0.05
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, xlab in [(axs[0], 'total_rate', 'rate at pref SF, summed over '
                                          'orientations (Hz)'),
                          (axs[1], 'blank_rate', 'blank (gray) rate (Hz)')]:
        v = res[key][m]
        ax.scatter(v[sel], gosi[sel], s=14, alpha=0.8, c='#d62728',
                   label='selective (perm p<0.05)')
        ax.scatter(v[~sel], gosi[~sel], s=14, alpha=0.6, c='#1f77b4',
                   label='not selective')
        ax.set_xlabel(xlab)
        ax.set_ylabel('global orientation index')
        ax.legend(fontsize=8)
    fig.suptitle('Selectivity vs firing rate, VISp good units')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/gosi_vs_rate.png', bbox_inches='tight')
    plt.close(fig)


def main():
    res = load_results()
    d = load_static()
    print('loaded results + static counts')
    fig_activity_timeseries(res, d)
    print('activity_timeseries.png')
    fig_example_tuning(res, d)
    print('example_tuning_curves.png')
    fig_population_selectivity(res)
    print('population_selectivity.png')
    fig_selectivity_fraction(res)
    print('selectivity_fraction.png')
    fig_pref_hist(res)
    print('pref_orientation_hist.png')
    fig_gosi_vs_rate(res, d)
    print('gosi_vs_rate.png')

    nwb, w = stream_session()
    fig_example_raster_psth(nwb, w, res)
    print('example_raster_psth.png')


if __name__ == '__main__':
    main()