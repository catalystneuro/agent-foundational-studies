"""Generate all figures for the orientation selectivity analysis.

Reads orientation_results.csv; reloads the NWB session (cached) for rasters
and per-trial rates of example units.

Figures:
  fig1_raw_data.png        - raster + PSTH for a tuned and an untuned VISp unit
  fig2_example_tuning.png  - tuning curves (cartesian + polar) for 6 units
  fig3_population.png      - fraction selective per area, gOSI distributions,
                             OSI-vs-DSI scatter, preferred orientation histogram
  fig4_tuning_heatmap.png  - normalized direction curves of selective VISp units
  fig5_permutation.png     - permutation null vs observed gOSI for examples
"""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

AREA_ORDER = ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'LGd', 'CA1']
AREA_COLORS = {'VISp': '#c0392b', 'VISl': '#e67e22', 'VISpm': '#f1c40f',
               'VISam': '#27ae60', 'VISrl': '#2980b9', 'LGd': '#8e44ad',
               'CA1': '#7f8c8d'}

res = pd.read_csv('orientation_results.csv', index_col=0)
dir_cols = [c for c in res.columns if c.startswith('dir_')]
directions = np.array([int(c.split('_')[1]) for c in dir_cols])
order = np.argsort(directions)
dir_cols = [dir_cols[i] for i in order]
directions = directions[order]

# ---------------- Load NWB (cached) ----------------
disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
inv = nwb['invalid_times'].values
overlap = np.zeros(len(dg), dtype=bool)
for a, b in inv:
    overlap |= (dg['start_time'].values < b) & (dg['stop_time'].values > a)
dg = dg[~overlap].reset_index(drop=True)
dg_dir = dg.dropna(subset=['orientation'])
units = nwb['units']


def trial_rates(uid):
    """Per-trial firing rates for one unit over valid dg presentations."""
    ts = units[uid].t
    starts = dg['start_time'].values
    stops = dg['stop_time'].values
    counts = np.searchsorted(ts, stops) - np.searchsorted(ts, starts)
    return counts / (stops - starts)


def mean_sem_by_direction(uid):
    rates = trial_rates(uid)
    means, sems = [], []
    for d in directions:
        r = rates[dg['orientation'].values == d]
        means.append(r.mean())
        sems.append(r.std(ddof=1) / np.sqrt(len(r)))
    return np.array(means), np.array(sems)


# ================= FIG 1: raw data =================
star = 950930237   # strongly tuned VISp unit, high rate
untuned_visp = res[(res['structure'] == 'VISp') & (~res['selective']) &
                   (res['firing_rate'] > 5) & (res['peak_response'] > 2)].index[0]
print("fig1 units:", star, untuned_visp)

fig1_units = [star, untuned_visp]
fig1_labels = [f'Unit {star} (VISp, orientation-selective)',
               f'Unit {untuned_visp} (VISp, not selective)']
colors = plt.cm.hsv(np.linspace(0, 1, len(directions), endpoint=False))
t_pre, t_post, bin_size = 0.5, 2.0, 0.05

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for row, (uid, lab) in enumerate(zip(fig1_units, fig1_labels)):
    ts = units[uid]
    ax = axes[row, 0]
    y = 0
    yticks, ylabels = [], []
    for di, d in enumerate(directions):
        onsets = dg_dir.loc[dg_dir['orientation'] == d, 'start_time'].values
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets), window=(-t_pre, t_post))
        for tr in peri.values():
            ax.plot(tr.t, np.full_like(tr.t, y), '|', color=colors[di],
                    markersize=3, alpha=0.85)
            y += 1
        yticks.append(y - len(onsets) / 2)
        ylabels.append(f"{int(d)}°")
        y += 3
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvspan(0, 2.0, color='gray', alpha=0.12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylabel('Grating direction')
    ax.set_xlabel('Time from grating onset (s)')
    ax.set_title(lab + ' — raster', fontsize=10)
    ax.set_xlim(-t_pre, t_post + 0.2)
    ax.set_ylim(-2, y)

    ax = axes[row, 1]
    for di, d in enumerate(directions):
        onsets = dg_dir.loc[dg_dir['orientation'] == d, 'start_time'].values
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets), window=(-t_pre, t_post))
        all_spikes = np.concatenate([p.t for p in peri.values()]) if len(peri) \
            else np.array([])
        bins = np.arange(-t_pre, t_post + bin_size, bin_size)
        counts, edges = np.histogram(all_spikes, bins=bins)
        rate = counts / (len(onsets) * bin_size)
        # light smoothing
        kernel = np.ones(3) / 3
        rate = np.convolve(rate, kernel, mode='same')
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, rate, color=colors[di], lw=1.4, label=f"{int(d)}°")
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvspan(0, 2.0, color='gray', alpha=0.12)
    ax.set_xlabel('Time from grating onset (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(lab + ' — PSTH', fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_xlim(-t_pre, t_post + 0.2)

plt.tight_layout()
plt.savefig('fig1_raw_data.png', dpi=150)
plt.close()
print("Saved fig1_raw_data.png")

# ================= FIG 2: example tuning curves =================
examples = [
    (950930295, 'orientation-selective'),
    (950932563, 'direction-selective'),
    (950930237, 'orientation-selective'),
]
# one moderately tuned VISp
mod = res[(res['structure'] == 'VISp') & res['selective'] &
          (res['gosi'] > 0.4) & (res['gosi'] < 0.55)].sort_values('gosi').index[0]
examples.append((mod, 'moderately tuned'))
# one non-selective VISp
non = res[(res['structure'] == 'VISp') & (~res['selective']) &
          (res['peak_response'] > 3)].index[0]
examples.append((non, 'not selective'))
# one CA1 unit
ca1 = res[(res['structure'] == 'CA1') & (~res['selective']) &
          (res['firing_rate'] > 2)].index[0]
examples.append((ca1, 'control'))
print("fig2 examples:", examples)

fig = plt.figure(figsize=(15, 10))
for i, (uid, tag) in enumerate(examples):
    means, sems = mean_sem_by_direction(uid)
    r = res.loc[uid]
    # cartesian
    ax = fig.add_subplot(2, 6, i + 1)
    x = np.append(directions, 360.0)
    m2 = np.append(means, means[0])
    s2 = np.append(sems, sems[0])
    ax.fill_between(x, m2 - s2, m2 + s2, color='#2980b9', alpha=0.25)
    ax.plot(x, m2, 'o-', color='#2980b9', lw=1.8, ms=5)
    ax.axhline(r['blank_rate'], color='gray', ls=':', lw=1, label='blank')
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Direction (°)', fontsize=9)
    ax.set_ylabel('Rate (Hz)', fontsize=9)
    pref_txt = 'n/a' if np.isnan(r['pref_ori']) else f"{int(r['pref_ori'])}°"
    gosi_txt = 'n/a' if np.isnan(r['gosi']) else f"{r['gosi']:.2f}"
    dsi_txt = 'n/a' if np.isnan(r['gdsi']) else f"{r['gdsi']:.2f}"
    ax.set_title(f"{uid} ({r['structure']})\n{tag}\n"
                 f"gOSI={gosi_txt} DSI={dsi_txt}\n"
                 f"p={r['p_value']:.3f} pref={pref_txt}",
                 fontsize=8)
    ax.tick_params(labelsize=8)
    # polar
    axp = fig.add_subplot(2, 6, i + 7, projection='polar')
    th = np.deg2rad(np.append(directions, 360.0))
    axp.plot(th, m2, 'o-', color='#c0392b', lw=1.8, ms=4)
    axp.fill(th, m2, color='#c0392b', alpha=0.2)
    axp.set_theta_zero_location('E')
    axp.set_theta_direction(1)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(202)  # move radial labels away from data lobe
    axp.yaxis.set_major_locator(plt.MaxNLocator(3))

fig.suptitle('Drifting-grating tuning curves (mean ± SEM across trials; '
             'dotted line = blank-sweep rate)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
fig.savefig('fig2_example_tuning.png', dpi=150)
plt.close()
print("Saved fig2_example_tuning.png")

# ================= FIG 3: population =================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) fraction selective per area with binomial 95% CI
ax = axes[0, 0]
from scipy.stats import beta as beta_dist
fracs, lo, hi, ns = [], [], [], []
for a in AREA_ORDER:
    g = res[res['structure'] == a]
    k, n = int(g['selective'].sum()), len(g)
    f = k / n
    fracs.append(f)
    ns.append(n)
    lo.append(f - beta_dist.ppf(0.025, k + 1, n - k + 1))
    hi.append(beta_dist.ppf(0.975, k + 1, n - k + 1) - f)
ax.bar(range(len(AREA_ORDER)), fracs,
       color=[AREA_COLORS[a] for a in AREA_ORDER],
       yerr=[lo, hi], capsize=4, alpha=0.85)
ax.axhline(0.05, color='k', ls='--', lw=1, label='chance (α = 0.05)')
for i, (f, n) in enumerate(zip(fracs, ns)):
    ax.text(i, f + hi[i] + 0.02, f"n={n}", ha='center', fontsize=8)
ax.set_xticks(range(len(AREA_ORDER)))
ax.set_xticklabels(AREA_ORDER, fontsize=9)
ax.set_ylabel('Fraction orientation-selective')
ax.set_ylim(0, 0.75)
ax.set_title('A. Orientation-selective units per area\n(permutation test p<0.05, peak response ≥ 1 Hz)',
             fontsize=10)
ax.legend(fontsize=8, frameon=False)

# (b) gOSI distributions per area
ax = axes[0, 1]
data = [res.loc[res['structure'] == a, 'gosi'].dropna().values for a in AREA_ORDER]
parts = ax.violinplot(data, showmedians=True, showextrema=False)
for pc, a in zip(parts['bodies'], AREA_ORDER):
    pc.set_facecolor(AREA_COLORS[a])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('k')
ax.set_xticks(range(1, len(AREA_ORDER) + 1))
ax.set_xticklabels(AREA_ORDER, fontsize=9)
ax.set_ylabel('gOSI (1 − circular variance, 2θ)')
ax.set_title('B. Orientation selectivity index distributions\n(units with grating response above blank)', fontsize=10)

# (c) OSI vs DSI scatter, visual areas
ax = axes[1, 0]
for a in AREA_ORDER:
    g = res[(res['structure'] == a) & res['selective']]
    ax.scatter(g['gosi'], g['gdsi'], s=14, alpha=0.6, color=AREA_COLORS[a],
               label=f"{a} (n={len(g)})", edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('gOSI (orientation selectivity)')
ax.set_ylabel('gDSI (direction selectivity)')
ax.set_title('C. Orientation vs direction selectivity\n(selective units only)', fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)

# (d) preferred orientation histogram, selective VISp
ax = axes[1, 1]
visp_sel = res[(res['structure'] == 'VISp') & res['selective']]
oris = [0, 45, 90, 135]
counts = [int((visp_sel['pref_ori'] == o).sum()) for o in oris]
ax.bar([str(o) + '°' for o in oris], counts, color='#c0392b', alpha=0.85)
ax.set_ylabel('Number of units')
ax.set_xlabel('Preferred orientation')
ax.set_title(f'D. Preferred orientation distribution\n(VISp selective units, n={len(visp_sel)})', fontsize=10)
for i, c in enumerate(counts):
    ax.text(i, c + 0.3, str(c), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig3_population.png', dpi=150)
plt.close()
print("Saved fig3_population.png")

# ================= FIG 4: tuning heatmap =================
visp_sel = res[(res['structure'] == 'VISp') & res['selective']].copy()
curves = visp_sel[dir_cols].values  # raw mean rates per direction
blank = visp_sel['blank_rate'].values[:, None]
curves_sub = np.clip(curves - blank, 0, None)
norm = curves_sub / np.maximum(curves_sub.max(axis=1, keepdims=True), 1e-9)
# preferred direction angle (circular mean of direction vector)
th = np.deg2rad(directions)
vec = (norm * np.exp(1j * th)).sum(axis=1)
pref_angle = np.angle(vec, deg=True) % 360
sort_idx = np.argsort(pref_angle)
norm_sorted = norm[sort_idx]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(norm_sorted, aspect='auto', cmap='viridis',
               extent=[0, 360, len(norm_sorted), 0],
               interpolation='nearest')
ax.set_xticks(directions)
ax.set_xticklabels([f"{int(d)}°" for d in directions], fontsize=9)
ax.set_xlabel('Grating direction')
ax.set_ylabel('Unit (sorted by preferred direction)')
ax.set_title(f'Normalized direction tuning of orientation-selective VISp units (n={len(norm_sorted)})\n'
             'blank-subtracted, each row normalized to its peak', fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Normalized response')
plt.tight_layout()
plt.savefig('fig4_tuning_heatmap.png', dpi=150)
plt.close()
print("Saved fig4_tuning_heatmap.png")

# ================= FIG 5: permutation test illustration =================
rng = np.random.default_rng(7)
N_SHUF = 1000
theta_rad = np.deg2rad(directions)
exp2 = np.exp(2j * theta_rad)


def gosi_stat(rates, labels, blank_rate):
    sums = np.bincount(labels, weights=rates, minlength=len(directions))
    counts = np.bincount(labels, minlength=len(directions))
    means = sums / np.maximum(counts, 1)
    sub = np.clip(means - blank_rate, 0, None)
    if sub.sum() <= 0:
        return 0.0
    return np.abs(np.sum(sub * exp2)) / sub.sum()


dir_to_idx = {d: i for i, d in enumerate(directions)}
labels_all = dg['orientation'].map(dir_to_idx).values
mask = ~np.isnan(labels_all)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (uid, tag) in zip(axes, [(star, 'orientation-selective VISp unit'),
                                 (untuned_visp, 'non-selective VISp unit')]):
    rates = trial_rates(uid)
    blank_rate = rates[~mask].mean()
    r = rates[mask]
    labels = labels_all[mask].astype(int)
    obs = gosi_stat(r, labels, blank_rate)
    null = np.array([gosi_stat(r, rng.permutation(labels), blank_rate)
                     for _ in range(N_SHUF)])
    p = (np.sum(null >= obs) + 1) / (N_SHUF + 1)
    ax.hist(null, bins=40, color='#95a5a6', alpha=0.8, density=True,
            label='shuffle null')
    ax.axvline(obs, color='#c0392b', lw=2, label=f'observed gOSI = {obs:.2f}')
    ax.set_xlabel('gOSI')
    ax.set_ylabel('Density')
    ax.set_title(f'{tag}\npermutation p = {p:.4f} ({N_SHUF} shuffles)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('fig5_permutation.png', dpi=150)
plt.close()
print("Saved fig5_permutation.png")
