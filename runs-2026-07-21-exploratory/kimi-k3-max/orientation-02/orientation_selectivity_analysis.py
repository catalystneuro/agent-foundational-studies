# %% [markdown]
# # Orientation Selectivity in Mouse Visual Cortex (DANDI 000021)
#
# This notebook demonstrates orientation selectivity in the visual system using
# real data from the DANDI Archive. We use one Neuropixels session from the
# Allen Institute Visual Coding dataset (Brain Observatory 1.1 stimulus set,
# dandiset 000021), in which full-field drifting gratings were presented at 8
# directions (0-315 deg in 45 deg steps) x 5 temporal frequencies while spiking
# activity was recorded simultaneously across visual cortical areas (VISp, VISl,
# VISpm, VISam, VISrl), the lateral geniculate nucleus (LGd), and hippocampus
# (CA1), which serves as a negative control.
#
# For each well-isolated unit we compute the mean firing rate per grating
# direction, derive orientation tuning curves (pairing opposite directions),
# and quantify selectivity with the global orientation selectivity index
# (gOSI = magnitude of the 2-theta resultant vector) and the global direction
# selectivity index (gDSI). Statistical significance is assessed with a
# permutation test (1000 label shuffles per unit). We then compare the
# fraction of orientation-selective units across areas.
#
# Data access is by streaming with remfile + a local disk cache, so no full
# file download is required.

# %% [markdown]
# ## Setup

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import chi2
from tqdm import tqdm

# Session-level NWB file (all probes + stimulus tables), session 715093703
DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"  # sub-699733573_ses-715093703.nwb
URL = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
       f"{VERSION}/assets/{ASSET_ID}/download/")

AREAS = ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'LGd', 'CA1']
AREA_COLORS = {'VISp': '#c0392b', 'VISl': '#e67e22', 'VISpm': '#f1c40f',
               'VISam': '#27ae60', 'VISrl': '#2980b9', 'LGd': '#8e44ad',
               'CA1': '#7f8c8d'}
N_SHUFFLES = 1000
MIN_PEAK_RESPONSE = 1.0  # Hz above blank required to count as selective
rng = np.random.default_rng(42)

# %% [markdown]
# ## Load the NWB file by streaming
#
# The file is ~2.9 GB; remfile fetches only the chunks we read and caches
# them on disk, so re-runs are fast.

# %%
disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Stimulus table and validity filtering
#
# The `drifting_gratings_presentations` table has one row per 2 s grating
# sweep. Rows with NaN orientation are interleaved blank sweeps (mean
# luminance), which we use to estimate the spontaneous firing rate.
# Presentations overlapping the session's `invalid_times` are excluded.

# %%
dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
inv = nwb['invalid_times'].values
overlap = np.zeros(len(dg), dtype=bool)
for a, b in inv:
    overlap |= (dg['start_time'].values < b) & (dg['stop_time'].values > a)
dg = dg[~overlap].reset_index(drop=True)

is_blank = dg['orientation'].isna().values
directions = np.sort(dg['orientation'].dropna().unique())
n_dirs = len(directions)
dir_to_idx = {d: i for i, d in enumerate(directions)}
trial_dir_idx = dg['orientation'].map(dir_to_idx).values
trial_starts = dg['start_time'].values
trial_stops = dg['stop_time'].values
trial_durs = trial_stops - trial_starts

print(f"Valid presentations: {len(dg)} ({is_blank.sum()} blank)")
print(f"Directions (deg): {directions}")
print("Trials per direction:")
print(dg['orientation'].value_counts(dropna=False).sort_index())

# %% [markdown]
# ## Unit selection
#
# We keep units flagged `quality == 'good'` and assign each to a brain
# structure through its peak channel's electrode location. CA1 (hippocampus)
# is included as a negative control: grating orientation should not
# systematically modulate hippocampal firing, so about 5% of those units
# should pass the p<0.05 test by chance alone.

# %%
units = nwb['units']
meta = units.metadata.copy()
elec = nwbfile.electrodes.to_dataframe()
meta['structure'] = meta['peak_channel_id'].map(elec['location'])
sel = meta[(meta['quality'] == 'good') & (meta['structure'].isin(AREAS))]
print(f"Selected {len(sel)} good units:")
print(sel['structure'].value_counts())

# %% [markdown]
# ## Raw data validation (Figure 1)
#
# Before computing anything, look at the spikes. Unit 950930237 (VISp) fires
# vigorously during gratings drifting along the 45/225 deg axis and little
# during other directions; unit 950930145 (VISp) responds to all directions
# about equally. The rasters and per-direction PSTHs below show this directly
# in the raw data.

# %%
dg_dir = dg.dropna(subset=['orientation'])
colors = plt.cm.hsv(np.linspace(0, 1, n_dirs, endpoint=False))
t_pre, t_post, bin_size = 0.5, 2.0, 0.05

fig1_units = [950930237, 950930145]
fig1_labels = ['Unit 950930237 (VISp, orientation-selective)',
               'Unit 950930145 (VISp, not selective)']

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for row, (uid, lab) in enumerate(zip(fig1_units, fig1_labels)):
    ts = units[uid]
    ax = axes[row, 0]
    y = 0
    yticks, ylabels = [], []
    for di, d in enumerate(directions):
        onsets = dg_dir.loc[dg_dir['orientation'] == d, 'start_time'].values
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets),
                                     window=(-t_pre, t_post))
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
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets),
                                     window=(-t_pre, t_post))
        all_spikes = np.concatenate([p.t for p in peri.values()]) if len(peri) \
            else np.array([])
        bins = np.arange(-t_pre, t_post + bin_size, bin_size)
        counts, edges = np.histogram(all_spikes, bins=bins)
        rate = counts / (len(onsets) * bin_size)
        rate = np.convolve(rate, np.ones(3) / 3, mode='same')
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
plt.show()
print("Saved fig1_raw_data.png")

# %% [markdown]
# ## Orientation tuning metrics and permutation tests
#
# For every selected unit:
#
# * per-trial firing rate = spike count / duration for each valid sweep
# * direction curve = mean rate per direction (8 values)
# * blank-subtracted curve = clip(direction curve − blank-sweep rate, min 0)
# * gOSI = |sum R(theta) e^{2i theta}| / sum R(theta) on the blank-subtracted
#   curve (1 = perfectly axis-tuned, 0 = flat)
# * gDSI = same with e^{i theta} (1 = perfectly direction tuned)
# * OSI = (R_pref − R_orth) / (R_pref + R_orth) on the 4-orientation curve
# * significance: permutation test, shuffling direction labels across trials
#   1000 times and recomputing the identical gOSI statistic
#
# A unit counts as orientation-selective if p < 0.05 and its peak
# blank-subtracted response is at least 1 Hz.

# %%
theta_rad = np.deg2rad(directions)
exp2 = np.exp(2j * theta_rad)  # orientation (axis) weights
exp1 = np.exp(1j * theta_rad)  # direction weights


def curve_metrics(dir_rates, blank_rate):
    """Selectivity metrics from the mean direction curve (8 values)."""
    sub = np.clip(dir_rates - blank_rate, 0, None)
    total = sub.sum()
    if total <= 0:
        return dict(gosi=np.nan, osi=np.nan, gdsi=np.nan, dsi=np.nan,
                    pref_dir=np.nan, pref_ori=np.nan, peak_response=0.0)
    gosi = np.abs(np.sum(sub * exp2)) / total
    gdsi = np.abs(np.sum(sub * exp1)) / total
    pref_idx = int(np.argmax(sub))
    pref_dir = directions[pref_idx]
    ori_curve = np.array([sub[i] + sub[(i + n_dirs // 2) % n_dirs]
                          for i in range(n_dirs // 2)]) / 2.0
    pref_ori_idx = int(np.argmax(ori_curve))
    pref_ori = directions[pref_ori_idx]
    orth_ori_idx = (pref_ori_idx + 2) % (n_dirs // 2)
    o_pref, o_orth = ori_curve[pref_ori_idx], ori_curve[orth_ori_idx]
    osi = (o_pref - o_orth) / (o_pref + o_orth) if (o_pref + o_orth) > 0 \
        else np.nan
    anti_idx = (pref_idx + n_dirs // 2) % n_dirs
    d_pref, d_anti = sub[pref_idx], sub[anti_idx]
    dsi = (d_pref - d_anti) / (d_pref + d_anti) if (d_pref + d_anti) > 0 \
        else np.nan
    return dict(gosi=gosi, osi=osi, gdsi=gdsi, dsi=dsi,
                pref_dir=pref_dir, pref_ori=pref_ori,
                peak_response=float(dir_rates.max() - blank_rate))


def gosi_from_labels(rates, labels, blank_rate):
    """The same gOSI statistic, computed from per-trial rates and labels."""
    sums = np.bincount(labels, weights=rates, minlength=n_dirs)
    counts = np.bincount(labels, minlength=n_dirs)
    means = sums / np.maximum(counts, 1)
    sub = np.clip(means - blank_rate, 0, None)
    if sub.sum() <= 0:
        return 0.0
    return np.abs(np.sum(sub * exp2)) / sub.sum()


dir_trial_mask = ~is_blank
dir_labels_all = np.array([int(x) if not np.isnan(x) else -1
                           for x in trial_dir_idx])

results = []
dir_rate_curves = {}
for unit_id in tqdm(sel.index, desc="Units"):
    ts = units[unit_id].t
    counts = np.searchsorted(ts, trial_stops) - np.searchsorted(ts, trial_starts)
    rates = counts / trial_durs

    blank_rate = rates[is_blank].mean() if is_blank.any() else 0.0
    dir_rates = np.array([rates[(trial_dir_idx == di)].mean()
                          for di in range(n_dirs)])
    m = curve_metrics(dir_rates, blank_rate)

    r = rates[dir_trial_mask]
    labels = dir_labels_all[dir_trial_mask]
    if not np.isnan(m['gosi']):
        obs = gosi_from_labels(r, labels, blank_rate)
        null = np.empty(N_SHUFFLES)
        for s in range(N_SHUFFLES):
            null[s] = gosi_from_labels(r, rng.permutation(labels), blank_rate)
        pval = (np.sum(null >= obs) + 1) / (N_SHUFFLES + 1)
    else:
        pval = 1.0

    results.append(dict(unit_id=unit_id,
                        structure=meta.loc[unit_id, 'structure'],
                        firing_rate=meta.loc[unit_id, 'firing_rate'],
                        snr=meta.loc[unit_id, 'snr'],
                        blank_rate=blank_rate, p_value=pval, **m))
    dir_rate_curves[unit_id] = dir_rates

res = pd.DataFrame(results).set_index('unit_id')
res['selective'] = (res['p_value'] < 0.05) & \
                   (res['peak_response'] >= MIN_PEAK_RESPONSE)
curves_df = pd.DataFrame(dir_rate_curves,
                         index=[f"dir_{int(d)}" for d in directions]).T
curves_df.index.name = 'unit_id'
res = res.join(curves_df)
res.to_csv('orientation_results.csv')
print("Saved orientation_results.csv")

frac = res.groupby('structure')['selective'].agg(['mean', 'sum', 'count'])
print("\nFraction orientation-selective per area:")
print(frac)

# %% [markdown]
# ## Example tuning curves (Figure 2)
#
# Six units spanning the selectivity range: two strongly orientation-selective
# VISp units (two lobes 180 deg apart, low DSI), one direction-selective VISp
# unit (single lobe, DSI = 1), one moderately tuned VISp unit, one
# non-selective VISp unit, and one CA1 control.

# %%
dir_cols = [f"dir_{int(d)}" for d in directions]


def mean_sem_by_direction(uid):
    ts = units[uid].t
    counts = np.searchsorted(ts, trial_stops) - np.searchsorted(ts, trial_starts)
    rates = counts / trial_durs
    means, sems = [], []
    for d in directions:
        rr = rates[dg['orientation'].values == d]
        means.append(rr.mean())
        sems.append(rr.std(ddof=1) / np.sqrt(len(rr)))
    return np.array(means), np.array(sems)


examples = [
    (950930295, 'orientation-selective'),
    (950932563, 'direction-selective'),
    (950930237, 'orientation-selective'),
    (950930964, 'moderately tuned'),
    (950930145, 'not selective'),
    (950911195, 'control'),
]

fig = plt.figure(figsize=(15, 10))
for i, (uid, tag) in enumerate(examples):
    means, sems = mean_sem_by_direction(uid)
    r = res.loc[uid]
    ax = fig.add_subplot(2, 6, i + 1)
    x = np.append(directions, 360.0)
    m2 = np.append(means, means[0])
    s2 = np.append(sems, sems[0])
    ax.fill_between(x, m2 - s2, m2 + s2, color='#2980b9', alpha=0.25)
    ax.plot(x, m2, 'o-', color='#2980b9', lw=1.8, ms=5)
    ax.axhline(r['blank_rate'], color='gray', ls=':', lw=1)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Direction (°)', fontsize=9)
    ax.set_ylabel('Rate (Hz)', fontsize=9)
    pref_txt = 'n/a' if np.isnan(r['pref_ori']) else f"{int(r['pref_ori'])}°"
    gosi_txt = 'n/a' if np.isnan(r['gosi']) else f"{r['gosi']:.2f}"
    dsi_txt = 'n/a' if np.isnan(r['gdsi']) else f"{r['gdsi']:.2f}"
    ax.set_title(f"{uid} ({r['structure']})\n{tag}\n"
                 f"gOSI={gosi_txt} DSI={dsi_txt}\n"
                 f"p={r['p_value']:.3f} pref={pref_txt}", fontsize=8)
    ax.tick_params(labelsize=8)
    axp = fig.add_subplot(2, 6, i + 7, projection='polar')
    th = np.deg2rad(np.append(directions, 360.0))
    axp.plot(th, m2, 'o-', color='#c0392b', lw=1.8, ms=4)
    axp.fill(th, m2, color='#c0392b', alpha=0.2)
    axp.set_theta_zero_location('E')
    axp.set_theta_direction(1)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(202)
    axp.yaxis.set_major_locator(plt.MaxNLocator(3))

fig.suptitle('Drifting-grating tuning curves (mean ± SEM across trials; '
             'dotted line = blank-sweep rate)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
fig.savefig('fig2_example_tuning.png', dpi=150)
plt.show()
print("Saved fig2_example_tuning.png")

# %% [markdown]
# ## Population results (Figure 3)
#
# * **A.** Fraction of orientation-selective units per area (binomial 95% CI).
#   Visual areas reach 38-49%, LGd 29%, and CA1 sits at the 5% chance level.
# * **B.** gOSI distributions per area (units responding above blank).
# * **C.** gOSI vs gDSI for selective units: nearly all points lie below the
#   diagonal, i.e. most tuned neurons in these areas are orientation- rather
#   than direction-selective.
# * **D.** Preferred-orientation histogram for selective VISp units.

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
fracs, lo, hi, ns = [], [], [], []
for a in AREAS:
    g = res[res['structure'] == a]
    k, n = int(g['selective'].sum()), len(g)
    f = k / n
    fracs.append(f)
    ns.append(n)
    lo.append(f - beta_dist.ppf(0.025, k + 1, n - k + 1))
    hi.append(beta_dist.ppf(0.975, k + 1, n - k + 1) - f)
ax.bar(range(len(AREAS)), fracs, color=[AREA_COLORS[a] for a in AREAS],
       yerr=[lo, hi], capsize=4, alpha=0.85)
ax.axhline(0.05, color='k', ls='--', lw=1, label='chance (α = 0.05)')
for i, (f, n) in enumerate(zip(fracs, ns)):
    ax.text(i, f + hi[i] + 0.02, f"n={n}", ha='center', fontsize=8)
ax.set_xticks(range(len(AREAS)))
ax.set_xticklabels(AREAS, fontsize=9)
ax.set_ylabel('Fraction orientation-selective')
ax.set_ylim(0, 0.75)
ax.set_title('A. Orientation-selective units per area\n'
             '(permutation test p<0.05, peak response ≥ 1 Hz)', fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[0, 1]
data = [res.loc[res['structure'] == a, 'gosi'].dropna().values for a in AREAS]
parts = ax.violinplot(data, showmedians=True, showextrema=False)
for pc, a in zip(parts['bodies'], AREAS):
    pc.set_facecolor(AREA_COLORS[a])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('k')
ax.set_xticks(range(1, len(AREAS) + 1))
ax.set_xticklabels(AREAS, fontsize=9)
ax.set_ylabel('gOSI (1 − circular variance, 2θ)')
ax.set_title('B. Orientation selectivity index distributions\n'
             '(units with grating response above blank)', fontsize=10)

ax = axes[1, 0]
for a in AREAS:
    g = res[(res['structure'] == a) & res['selective']]
    ax.scatter(g['gosi'], g['gdsi'], s=14, alpha=0.6, color=AREA_COLORS[a],
               label=f"{a} (n={len(g)})", edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('gOSI (orientation selectivity)')
ax.set_ylabel('gDSI (direction selectivity)')
ax.set_title('C. Orientation vs direction selectivity\n'
             '(selective units only)', fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)

ax = axes[1, 1]
visp_sel = res[(res['structure'] == 'VISp') & res['selective']]
oris = [0, 45, 90, 135]
counts_ori = [int((visp_sel['pref_ori'] == o).sum()) for o in oris]
ax.bar([str(o) + '°' for o in oris], counts_ori, color='#c0392b', alpha=0.85)
ax.set_ylabel('Number of units')
ax.set_xlabel('Preferred orientation')
chi2_stat = ((np.array(counts_ori) - len(visp_sel) / 4) ** 2).sum() / \
            (len(visp_sel) / 4)
p_chi2 = 1 - chi2.cdf(chi2_stat, df=3)
ax.set_title(f'D. Preferred orientation distribution\n'
             f'(VISp selective units, n={len(visp_sel)}; '
             f'chi² p={p_chi2:.2f})', fontsize=10)
for i, c in enumerate(counts_ori):
    ax.text(i, c + 0.3, str(c), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig3_population.png', dpi=150)
plt.show()
print("Saved fig3_population.png")

# %% [markdown]
# ## Population tuning heatmap (Figure 4)
#
# Blank-subtracted, peak-normalized direction curves of all
# orientation-selective VISp units, sorted by preferred direction. The two
# diagonal bands 180 deg apart reflect axis (orientation) tuning: most units
# respond to both drift directions along their preferred axis.

# %%
curves = visp_sel[dir_cols].values
blank = visp_sel['blank_rate'].values[:, None]
curves_sub = np.clip(curves - blank, 0, None)
norm = curves_sub / np.maximum(curves_sub.max(axis=1, keepdims=True), 1e-9)
vec = (norm * np.exp(1j * theta_rad)).sum(axis=1)
pref_angle = np.angle(vec, deg=True) % 360
norm_sorted = norm[np.argsort(pref_angle)]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(norm_sorted, aspect='auto', cmap='viridis',
               extent=[0, 360, len(norm_sorted), 0], interpolation='nearest')
ax.set_xticks(directions)
ax.set_xticklabels([f"{int(d)}°" for d in directions], fontsize=9)
ax.set_xlabel('Grating direction')
ax.set_ylabel('Unit (sorted by preferred direction)')
ax.set_title(f'Normalized direction tuning of orientation-selective VISp units '
             f'(n={len(norm_sorted)})\nblank-subtracted, each row normalized to '
             'its peak', fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Normalized response')
plt.tight_layout()
plt.savefig('fig4_tuning_heatmap.png', dpi=150)
plt.show()
print("Saved fig4_tuning_heatmap.png")

# %% [markdown]
# ## Permutation test illustration (Figure 5)
#
# Null distributions of gOSI under label shuffling for the two Figure 1 units.
# The tuned unit's observed gOSI lies far outside its null (p = 0.001); the
# non-selective unit's gOSI is unremarkable under the null (p = 0.35). This
# also shows why a normalized index alone is not enough: with ~60 trials per
# direction the null gOSI has a floor around 0.1-0.3, so significance must be
# judged against the shuffle distribution.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (uid, tag) in zip(axes, [(950930237, 'orientation-selective VISp unit'),
                                 (950930145, 'non-selective VISp unit')]):
    ts = units[uid].t
    counts = np.searchsorted(ts, trial_stops) - np.searchsorted(ts, trial_starts)
    rates = counts / trial_durs
    blank_rate = rates[is_blank].mean()
    r = rates[dir_trial_mask]
    labels = dir_labels_all[dir_trial_mask]
    obs = gosi_from_labels(r, labels, blank_rate)
    null = np.array([gosi_from_labels(r, rng.permutation(labels), blank_rate)
                     for _ in range(N_SHUFFLES)])
    p = (np.sum(null >= obs) + 1) / (N_SHUFFLES + 1)
    ax.hist(null, bins=40, color='#95a5a6', alpha=0.8, density=True,
            label='shuffle null')
    ax.axvline(obs, color='#c0392b', lw=2, label=f'observed gOSI = {obs:.2f}')
    ax.set_xlabel('gOSI')
    ax.set_ylabel('Density')
    ax.set_title(f'{tag}\npermutation p = {p:.4f} ({N_SHUFFLES} shuffles)',
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('fig5_permutation.png', dpi=150)
plt.show()
print("Saved fig5_permutation.png")

# %% [markdown]
# ## Summary
#
# In this session (DANDI 000021, session 715093703), drifting gratings
# modulated a large fraction of neurons in every recorded visual area:
# 38-49% of good units in cortical visual areas and 29% in LGd were
# significantly orientation-selective (permutation test p < 0.05 with a
# >= 1 Hz peak response), against 4.5% in hippocampal CA1, matching the 5%
# false-positive rate of the test. Among selective units, orientation
# selectivity (gOSI) typically exceeded direction selectivity (gDSI), the
# classic signature of orientation-tuned simple/complex cells. Preferred
# orientations in VISp spanned all four sampled axes, with a modest,
# non-significant excess around 90 deg (chi² test, p ≈ 0.07).
