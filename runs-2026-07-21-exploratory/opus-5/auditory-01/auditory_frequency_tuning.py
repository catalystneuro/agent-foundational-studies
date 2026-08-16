# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auditory frequency tuning in mouse auditory cortex (DANDI:000986)
#
# This notebook demonstrates frequency tuning, the defining response property of neurons in the
# auditory system, using Neuropixels recordings from mouse auditory cortex in
# [DANDI:000986](https://dandiarchive.org/dandiset/000986), *"Auditory cortex Neuropixels
# recordings and pupil diameter traces from mice during passive exposure to pure tones"*
# (Jo & McCormick, University of Oregon; related preprint
# [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# Head-fixed, passively listening mice were presented with 25 ms pure tones at 2, 4, 8, 16 and
# 32 kHz, all at 60 dB SPL, one tone every 805 ms. Each session contains roughly 7450 tones, so
# every frequency is repeated about 1490 times. The dandiset has 15 sessions from 5 mice and
# 1564 sorted units in total. Pupil diameter and running speed were recorded simultaneously,
# which lets us ask whether tuning depends on arousal state.
#
# The analysis proceeds from raw data to population statistics:
#
# 1. stream one session and verify every data stream,
# 2. compute per-unit tone-evoked responses and frequency tuning curves,
# 3. test each unit for frequency tuning and quantify best frequency, latency and selectivity,
# 4. repeat over all 15 sessions and pool 1564 units,
# 5. decode which tone was played from single-trial population activity,
# 6. fit NeMoS Poisson GLMs that give each frequency its own temporal kernel, and compare them
#    against a frequency-blind model on held-out data,
# 7. split trials by pupil diameter to check that the tuning is not an artefact of arousal.
#
# All data are streamed from the DANDI S3 mirror with `remfile` and a local disk cache; nothing
# is downloaded in full.

# %% [markdown]
# ## Setup

# %%
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import stats as sstats
from tqdm import tqdm

DANDISET = '000986'
CACHE_DIR = os.environ.get('REMFILE_CACHE', '/tmp/remfile_cache')
FIG_DIR = 'figures'
RESULTS_PKL = 'results_000986.pkl'
os.makedirs(FIG_DIR, exist_ok=True)

# Analysis windows. The tone is 25 ms long and cortical onset latencies are 10-40 ms, so a
# 100 ms window starting 5 ms after onset captures the evoked response. The baseline window is
# the mirror image before onset; the inter-tone interval of 805 ms keeps it clear of the
# previous tone's response.
WIN_EV = (0.005, 0.105)
WIN_BL = (-0.105, -0.005)
PSTH_BIN = 0.005
PSTH_RANGE = (-0.1, 0.3)

# %% [markdown]
# ## Streaming access to DANDI
#
# `remfile` fetches only the byte ranges HDF5 asks for and keeps them in a disk cache, so
# repeated reads of the same session are fast and the 3.9 GB dandiset is never downloaded.

# %%
def list_assets(dandiset_id, version='draft'):
    """Return {path: asset_id} for every asset in a dandiset."""
    url = f'https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/assets/'
    out, params = {}, {'page_size': 200}
    while url:
        r = requests.get(url, params=params).json()
        out.update({a['path']: a['asset_id'] for a in r['results']})
        url, params = r.get('next'), None
    return out


def open_nwb(dandiset_id, asset_id, version='draft'):
    """Open one NWB asset by streaming it from the DANDI S3 mirror."""
    url = (f'https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/'
           f'{version}/assets/{asset_id}/download/')
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    return NWBHDF5IO(file=h5py.File(rem, 'r'), load_namespaces=True).read()


assets = list_assets(DANDISET)
print(f'{len(assets)} assets in DANDI:{DANDISET}')
for p in sorted(assets):
    print('  ', p)

# %% [markdown]
# ## Load one session and look at every data stream
#
# Prototyping starts on a single session. Pynapple wraps the NWB file so that spike trains
# become a `TsGroup` and the behavioural traces become `Tsd` objects on a common clock.

# %%
SESSION = 'sub-LA9/sub-LA9_ses-1_behavior.nwb'
nwbfile = open_nwb(DANDISET, assets[SESSION])
nwb = nap.NWBFile(nwbfile)
print(nwb)

units = nwb['units']
trials = nwbfile.trials.to_dataframe()
pupil, running = nwb['pupil_diameter'], nwb['running_speed']
onsets, freqs = trials.start_time.values, trials.stim_frequency.values
ufreq = np.sort(np.unique(freqs))

print(f'\n{len(units)} units, {len(trials)} tones')
print('tone frequencies (Hz):', ufreq)
print('tone duration (s):', trials.stim_duration.unique(),
      ' level (dB SPL):', trials.stim_amplitude.unique())
print('inter-tone interval (s): %.3f' % np.median(np.diff(onsets)))
print('trials per frequency:\n', trials.stim_frequency.value_counts().sort_index())
print('firing rates (Hz), percentiles 0/25/50/75/100:',
      np.round(np.percentile(np.asarray(units.rates), [0, 25, 50, 75, 100]), 2))
print('pupil samples: %d, NaN fraction %.3f'
      % (len(pupil), np.isnan(np.asarray(pupil)).mean()))

# %% [markdown]
# ### Figure 1: raw spiking with the tone sequence overlaid
#
# A 12 s excerpt. Each vertical coloured bar is one 25 ms tone. Several units already show
# visible tone locking, and the pupil and running traces are continuous and well sampled.

# %%
cmap = plt.get_cmap('viridis')
fcol = {f: cmap(i / (len(ufreq) - 1)) for i, f in enumerate(ufreq)}

win = nap.IntervalSet(500.0, 512.0)
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1, 1], hspace=0.12))
ax = axes[0]
for i, u in enumerate(units.index):
    st = units[u].restrict(win).t
    ax.plot(st, np.full_like(st, i), '|', color='k', ms=3, mew=0.6)
sel = trials[(trials.start_time > win.start[0]) & (trials.start_time < win.end[0])]
for _, r in sel.iterrows():
    ax.axvspan(r.start_time, r.start_time + r.stim_duration, color=fcol[r.stim_frequency],
               alpha=0.85, lw=0, zorder=0)
ax.set_ylabel('unit #')
ax.set_title(f'DANDI:000986  {SESSION}  —  raw spiking, {len(units)} units\n'
             'shaded bars = 25 ms pure tones (colour = frequency, 60 dB SPL)')
ax.legend(handles=[plt.Line2D([], [], color=fcol[f], lw=6, label=f'{f/1000:g} kHz')
                   for f in ufreq],
          loc='upper left', bbox_to_anchor=(1.005, 1.0), frameon=False, fontsize=9, title='tone')
axes[1].plot(running.restrict(win).t, np.asarray(running.restrict(win)), color='tab:green', lw=1)
axes[1].set_ylabel('running\n(cm/s)')
axes[2].plot(pupil.restrict(win).t, np.asarray(pupil.restrict(win)), color='tab:purple', lw=1)
axes[2].set_ylabel('pupil\ndiameter')
axes[2].set_xlabel('time in session (s)')
for a in axes:
    a.margins(x=0)
fig.savefig(f'{FIG_DIR}/fig01_raw_streams.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ### Figure 2: whole-session overview
#
# Firing rates in 1 s bins for the whole session, with the interleaved silent ("spontaneous")
# blocks shaded. Tones are delivered in four ~25 min blocks separated by 5 min of silence.

# %%
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.2, 1, 1], hspace=0.15))
cnt = units.count(1.0)
z = (np.asarray(cnt) - np.asarray(cnt).mean(0)) / (np.asarray(cnt).std(0) + 1e-9)
im = axes[0].imshow(z.T, aspect='auto', origin='lower', vmin=-2, vmax=4, cmap='magma',
                    extent=[cnt.t[0], cnt.t[-1], 0, len(units)])
axes[0].set_ylabel('unit #')
axes[0].set_title('Whole-session overview: z-scored firing rate (1 s bins)')
plt.colorbar(im, ax=axes[0], pad=0.01, label='z')
axes[1].plot(running.t[::50], np.asarray(running)[::50], color='tab:green', lw=0.5)
axes[1].set_ylabel('running\n(cm/s)')
axes[2].plot(pupil.t[::50], np.asarray(pupil)[::50], color='tab:purple', lw=0.5)
axes[2].set_ylabel('pupil\ndiameter')
axes[2].set_xlabel('time in session (s)')
for _, r in nwbfile.intervals['spontaneous_blocks'].to_dataframe().iterrows():
    for a in axes:
        a.axvspan(r.start_time, r.stop_time, color='0.6', alpha=0.25, lw=0, zorder=0)
axes[2].text(0.01, 0.9, 'grey = spontaneous (no tones) blocks', transform=axes[2].transAxes,
             fontsize=9, va='top')
fig.savefig(f'{FIG_DIR}/fig02_session_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ## Quantifying tone-evoked responses
#
# For every (trial, unit) pair we count spikes in the evoked and baseline windows. The
# baseline-subtracted rate averaged over trials of one frequency is that unit's response to
# that frequency; the five values form its tuning curve.
#
# Two statistical tests are applied per unit and corrected across units with
# Benjamini-Hochberg at q = 0.05:
#
# * **responsive**: Wilcoxon signed-rank on evoked versus baseline counts across all trials,
# * **frequency-tuned**: Kruskal-Wallis on the baseline-subtracted trial-wise rates grouped by
#   frequency, which asks whether the response depends on *which* tone was played.

# %%
def trial_counts(units, onsets, win):
    """Spike counts for every (trial, unit) in a window relative to tone onset."""
    a, b = np.asarray(onsets) + win[0], np.asarray(onsets) + win[1]
    out = np.empty((len(a), len(units)), dtype=np.int32)
    for j, u in enumerate(units.index):
        st = units[u].t
        out[:, j] = np.searchsorted(st, b) - np.searchsorted(st, a)
    return out


def psth(units, onsets, tmin=-0.1, tmax=0.3, bin_size=0.005):
    """Trial-averaged firing rate (Hz) per unit on a common time axis."""
    edges = np.arange(tmin, tmax + bin_size / 2, bin_size)
    centers = edges[:-1] + bin_size / 2
    rates = np.zeros((len(units), len(centers)))
    onsets = np.asarray(onsets)
    for j, u in enumerate(units.index):
        st = units[u].t
        lo = np.searchsorted(st, onsets + tmin)
        hi = np.searchsorted(st, onsets + tmax)
        rel = np.concatenate([st[lo[k]:hi[k]] - onsets[k] for k in range(len(onsets))])
        rates[j] = np.histogram(rel, bins=edges)[0] / (len(onsets) * bin_size)
    return centers, rates


def tuning_table(counts_ev, counts_bl, freqs, win_ev, win_bl, unit_ids):
    """Per-unit tuning curves, SEMs and statistics."""
    ufreq = np.unique(freqs)
    rate_ev = counts_ev / (win_ev[1] - win_ev[0])
    rate_bl = counts_bl / (win_bl[1] - win_bl[0])
    delta = rate_ev - rate_bl

    curves = np.zeros((counts_ev.shape[1], len(ufreq)))
    sems = np.zeros_like(curves)
    for i, f in enumerate(ufreq):
        m = freqs == f
        curves[:, i] = delta[m].mean(0)
        sems[:, i] = delta[m].std(0) / np.sqrt(m.sum())

    rows = []
    for j in range(counts_ev.shape[1]):
        try:
            _, p_resp = sstats.wilcoxon(counts_ev[:, j], counts_bl[:, j])
        except ValueError:                       # all differences are zero
            p_resp = 1.0
        _, p_tune = sstats.kruskal(*[delta[freqs == f, j] for f in ufreq])
        c = curves[j]
        pos = np.clip(c, 0, None)
        n = len(pos)
        # lifetime sparseness (Rolls & Tovee 1995) on the rectified tuning curve:
        # 0 = equal response to all tones, 1 = responds to a single tone
        sparseness = ((1 - (pos.sum() / n) ** 2 / max((pos ** 2).sum() / n, 1e-12)) /
                      (1 - 1.0 / n)) if pos.sum() > 0 else np.nan
        rows.append(dict(unit=unit_ids[j], baseline_hz=rate_bl[:, j].mean(),
                         peak_evoked_hz=c.max(), min_evoked_hz=c.min(),
                         best_frequency=ufreq[np.argmax(c)], p_responsive=p_resp,
                         p_tuned=p_tune, sparseness=sparseness,
                         depth_of_tuning=(c.max() - c.min()) /
                                         (abs(c.max()) + abs(c.min()) + 1e-12)))
    return (pd.DataFrame(curves, index=unit_ids, columns=ufreq),
            pd.DataFrame(sems, index=unit_ids, columns=ufreq),
            pd.DataFrame(rows).set_index('unit'))


def fdr(pvals, q=0.05):
    """Benjamini-Hochberg correction; returns a boolean array of rejections."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej


def onset_latency(t, rate, bl_lo=-0.1, bl_hi=-0.005, n_sd=3.0, n_consec=2):
    """First post-onset time at which the PSTH exceeds baseline + n_sd SD for n_consec bins."""
    bl = rate[(t >= bl_lo) & (t < bl_hi)]
    thr = bl.mean() + n_sd * bl.std()
    run = 0
    for i in np.where(t >= 0)[0]:
        run = run + 1 if rate[i] > thr else 0
        if run >= n_consec:
            return t[i - n_consec + 1]
    return np.nan


def analyze_session(path):
    """Full per-session pipeline: load, count, fit tuning curves, test, measure latency."""
    nwbfile = open_nwb(DANDISET, assets[path])
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']
    trials = nwbfile.trials.to_dataframe()
    onsets, freqs = trials.start_time.values, trials.stim_frequency.values
    ufreq = np.sort(np.unique(freqs))
    uid = np.asarray(units.index)

    ev = trial_counts(units, onsets, WIN_EV)
    bl = trial_counts(units, onsets, WIN_BL)
    curves, sems, stats_df = tuning_table(ev, bl, freqs, WIN_EV, WIN_BL, uid)

    psths = None
    for i, f in enumerate(ufreq):
        t, r = psth(units, onsets[freqs == f], *PSTH_RANGE, PSTH_BIN)
        if psths is None:
            psths = np.zeros((len(uid), len(ufreq), len(t)))
        psths[:, i, :] = r

    stats_df['latency_s'] = [
        onset_latency(t, psths[j, list(ufreq).index(stats_df.best_frequency.iloc[j])])
        for j in range(len(uid))]
    stats_df['session'] = path
    stats_df['subject'] = path.split('/')[0].replace('sub-', '')
    stats_df['responsive'] = fdr(stats_df.p_responsive.values)
    stats_df['tuned'] = fdr(stats_df.p_tuned.values)
    return dict(session=path, ufreq=ufreq, curves=curves, sems=sems, stats=stats_df,
                psth_t=t, psths=psths, counts_ev=ev, counts_bl=bl, freqs=freqs,
                onsets=onsets, unit_ids=uid, n_trials=len(trials))


res_single = analyze_session(SESSION)
s = res_single['stats']
print(f'{len(s)} units: {int(s.responsive.sum())} tone-responsive, '
      f'{int(s.tuned.sum())} frequency-tuned (FDR q=0.05)')
print(s[s.tuned].best_frequency.value_counts().sort_index())
print(res_single['curves'].loc[s.sort_values('peak_evoked_hz', ascending=False)
                               .index[:6]].round(1).to_string())

# %% [markdown]
# ### Figure 3: example units
#
# One example per best frequency, chosen as the most strongly driven tuned unit for each. Left:
# spike raster with trials grouped by tone frequency. Middle: PSTH per frequency. Right: tuning
# curve of baseline-subtracted evoked rate, with the best frequency circled.
#
# The units span the range of behaviour seen in auditory cortex: sharply tuned excitation
# (unit at 8 kHz), broad high-frequency preference, and units that are *suppressed* by tones
# away from their best frequency (the 4 kHz unit).

# %%
tuned = s[s.tuned & s.responsive]
examples = [tuned[tuned.best_frequency == f].peak_evoked_hz.idxmax()
            for f in ufreq if (tuned.best_frequency == f).any()]

fig, axes = plt.subplots(len(examples), 3, figsize=(13, 2.6 * len(examples)),
                         gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.9], hspace=0.55,
                                          wspace=0.28))
RAST_N = 40
for r, uid in enumerate(examples):
    axR, axP, axT = axes[r]
    y = 0
    for f in ufreq:
        sel = onsets[freqs == f][:RAST_N]
        pe = nap.compute_perievent(units[uid], nap.Ts(sel), window=(-0.05, 0.2))
        for k in range(len(sel)):
            st = pe[k].t
            axR.plot(st, np.full_like(st, y), '|', color=fcol[f], ms=3.5, mew=0.8)
            y += 1
        axR.axhline(y - 0.5, color='0.85', lw=0.5)
    axR.axvspan(0, 0.025, color='0.85', zorder=0, lw=0)
    axR.set_ylim(0, y)
    axR.set_xlim(-0.05, 0.2)
    axR.set_ylabel('trial (grouped\nby frequency)', fontsize=9)
    axR.set_title(f'unit {uid} — raster ({RAST_N} trials/freq)', fontsize=10)

    t = res_single['psth_t']
    for i, f in enumerate(ufreq):
        axP.plot(t, res_single['psths'][list(res_single['unit_ids']).index(uid), i],
                 color=fcol[f], lw=1.4, label=f'{f/1000:g} kHz')
    axP.axvspan(0, 0.025, color='0.85', zorder=0, lw=0)
    axP.set_xlim(-0.1, 0.3)
    axP.set_xlabel('time from tone onset (s)', fontsize=9)
    axP.set_ylabel('firing rate (Hz)', fontsize=9)
    axP.set_title('PSTH by tone frequency', fontsize=10)
    if r == 0:
        axP.legend(fontsize=8, frameon=False, ncol=2)

    c = res_single['curves'].loc[uid].values
    axT.errorbar(ufreq / 1000, c, yerr=res_single['sems'].loc[uid].values, marker='o',
                 color='k', lw=1.5, capsize=3)
    axT.axhline(0, color='0.6', lw=0.8, ls=':')
    bf = s.loc[uid, 'best_frequency']
    axT.plot(bf / 1000, c[list(ufreq).index(bf)], 'o', ms=12, mfc='none', mec='crimson', mew=2)
    axT.set_xscale('log', base=2)
    axT.set_xticks(ufreq / 1000)
    axT.set_xticklabels([f'{f/1000:g}' for f in ufreq])
    axT.set_xlabel('tone frequency (kHz)', fontsize=9)
    axT.set_ylabel('evoked rate (Hz)\n(baseline subtracted)', fontsize=9)
    axT.set_title(f'tuning curve — BF {bf/1000:g} kHz, p={s.loc[uid,"p_tuned"]:.1e}', fontsize=10)
axes[-1][0].set_xlabel('time from tone onset (s)', fontsize=9)
fig.suptitle(f'DANDI:000986 {SESSION} — single-unit frequency tuning in mouse auditory cortex\n'
             'grey bar = 25 ms tone, 60 dB SPL', y=0.995, fontsize=12)
fig.savefig(f'{FIG_DIR}/fig03_example_units.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('example units:', examples)

# %% [markdown]
# ## All 15 sessions
#
# The same pipeline is run on every session in the dandiset (about 10 s per session once the
# byte ranges are cached) and the results are pooled over 1564 units from 5 mice.

# %%
if os.path.exists(RESULTS_PKL):
    results = pickle.load(open(RESULTS_PKL, 'rb'))
else:
    results = {}
    for p in tqdm(sorted(assets), desc='sessions'):
        results[p] = analyze_session(p)
        st = results[p]['stats']
        tqdm.write(f'{p}: {len(st)} units, {int(st.responsive.sum())} responsive, '
                   f'{int(st.tuned.sum())} freq-tuned, {results[p]["n_trials"]} trials')
    pickle.dump(results, open(RESULTS_PKL, 'wb'))

sessions = sorted(results)
stats_all = pd.concat([results[k]['stats'] for k in sessions], keys=sessions,
                      names=['file', 'unit'])
curves_all = pd.concat([results[k]['curves'] for k in sessions], keys=sessions,
                       names=['file', 'unit'])
psth_t = results[sessions[0]]['psth_t']
psth_all = np.concatenate([results[k]['psths'] for k in sessions], axis=0)
print(f'{len(stats_all)} units from {stats_all.subject.nunique()} mice: '
      f'{int(stats_all.responsive.sum())} tone-responsive, '
      f'{int((stats_all.responsive & stats_all.tuned).sum())} frequency-tuned')

# %% [markdown]
# ### Figure 4: population summary
#
# Six views of the pooled population:
#
# * **Peak-normalised tuning curves** sorted by best frequency. The diagonal band is the
#   signature of frequency tuning: each unit responds most to one part of the spectrum. Blue
#   entries are frequencies that suppress the unit below its baseline rate.
# * **BF-aligned population tuning** collapses every unit onto a common axis of octaves from
#   its own best frequency, giving a sharply peaked average tuning curve.
# * **Responses at each unit's best frequency**, z-scored against the pre-tone period and
#   sorted by peak time.
# * **Best-frequency distribution per mouse**. All five frequencies are represented in every
#   mouse, with the mixture depending on where the probe landed in the tonotopic map.
# * **Split-half reliability**: tuning curves computed from odd and even trials separately.
# * **Onset latency** at the best frequency, and lifetime sparseness in the inset.

# %%
tun_mask = (stats_all.tuned & stats_all.responsive).values
C = curves_all.values[tun_mask]
S = stats_all[tun_mask]
P = psth_all[tun_mask]

# Units whose peak evoked rate is below 1 Hz are left out of the normalised displays:
# dividing by a near-zero peak inflates the curve without adding information.
MIN_PEAK_HZ = 1.0
pos = C.max(1) >= MIN_PEAK_HZ
Cn = C[pos] / C[pos].max(1, keepdims=True)
bf_idx = C[pos].argmax(1)
centroid = (Cn * np.arange(len(ufreq))).sum(1) / np.maximum(Cn.sum(1), 1e-9)
order = np.lexsort((centroid, bf_idx))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Cn[order], aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[-0.5, len(ufreq) - 0.5, 0, pos.sum()])
ax.set_xticks(range(len(ufreq)))
ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('tone frequency (kHz)')
ax.set_ylabel('frequency-tuned unit (sorted by BF)')
ax.set_title(f'Peak-normalised tuning curves\n{pos.sum()} tuned units with peak > '
             f'{MIN_PEAK_HZ:g} Hz')
plt.colorbar(im, ax=ax, pad=0.02, label='norm. rate')

ax = fig.add_subplot(gs[0, 1])
offsets = np.arange(-(len(ufreq) - 1), len(ufreq))
acc = {o: [] for o in offsets}
for i in range(Cn.shape[0]):
    for j in range(len(ufreq)):
        acc[j - bf_idx[i]].append(Cn[i, j])
m = np.array([np.mean(acc[o]) if acc[o] else np.nan for o in offsets])
se = np.array([np.std(acc[o]) / np.sqrt(len(acc[o])) if acc[o] else np.nan for o in offsets])
ax.errorbar(offsets, m, yerr=se, marker='o', color='crimson', lw=2, capsize=3)
ax.axvline(0, color='0.6', ls=':')
ax.axhline(0, color='0.6', ls=':')
ax.set_xlabel('octaves from best frequency')
ax.set_ylabel('normalised evoked rate')
ax.set_title("Population tuning aligned to each unit's BF")
for o, yv in zip(offsets, m):
    k = len(acc[o])
    if k < 0.1 * Cn.shape[0]:
        ax.annotate(f'n={k}', (o, yv), fontsize=7, textcoords='offset points', xytext=(0, -16),
                    ha='center', color='0.4')

ax = fig.add_subplot(gs[0, 2])
bf_psth = np.array([P[i, C[i].argmax()] for i in np.where(pos)[0]])
base = bf_psth[:, psth_t < 0].mean(1, keepdims=True)
sd = bf_psth[:, psth_t < 0].std(1, keepdims=True) + 1e-6
Z = (bf_psth - base) / sd
lat_order = np.argsort(np.argmax(Z[:, psth_t >= 0], axis=1))
im = ax.imshow(Z[lat_order], aspect='auto', origin='lower', cmap='magma', vmin=-2, vmax=10,
               extent=[psth_t[0], psth_t[-1], 0, len(Z)])
ax.axvline(0, color='w', lw=0.8)
ax.axvline(0.025, color='w', lw=0.8, ls=':')
ax.set_xlabel('time from tone onset (s)')
ax.set_ylabel('tuned unit (sorted by peak time)')
ax.set_title("Response at each unit's best frequency")
plt.colorbar(im, ax=ax, pad=0.02, label='z-score vs pre-tone')

ax = fig.add_subplot(gs[1, 0])
tab = S.groupby(['subject', 'best_frequency']).size().unstack(fill_value=0)
tab = tab.div(tab.sum(1), axis=0)
x = np.arange(len(ufreq))
w = 0.15
for i, (subj, row) in enumerate(tab.iterrows()):
    ax.bar(x + (i - len(tab) / 2) * w, [row.get(f, 0) for f in ufreq], w, label=subj)
ax.set_xticks(x)
ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('best frequency (kHz)')
ax.set_ylabel('fraction of tuned units')
ax.set_title('Best-frequency distribution per mouse')
ax.legend(fontsize=8, frameon=False, ncol=2, title='subject')

rel, rel_shuf = [], []
rng = np.random.default_rng(0)
for k in sessions:
    r = results[k]
    d = r['counts_ev'] / (WIN_EV[1] - WIN_EV[0]) - r['counts_bl'] / (WIN_BL[1] - WIN_BL[0])
    keep = (r['stats'].tuned & r['stats'].responsive).values
    half = np.arange(len(r['freqs'])) % 2
    for shuffle in (False, True):
        f = rng.permutation(r['freqs']) if shuffle else r['freqs']
        a = np.array([[d[(half == 0) & (f == q), j].mean() for q in ufreq]
                      for j in np.where(keep)[0]])
        b = np.array([[d[(half == 1) & (f == q), j].mean() for q in ufreq]
                      for j in np.where(keep)[0]])
        rr = [sstats.pearsonr(a[i], b[i])[0] for i in range(len(a))]
        (rel_shuf if shuffle else rel).extend(rr)
rel, rel_shuf = np.array(rel), np.array(rel_shuf)

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(-1, 1, 41)
ax.hist(rel_shuf, bins, color='0.7',
        label=f'frequency labels shuffled\n(median {np.nanmedian(rel_shuf):.2f})')
ax.hist(rel, bins, color='crimson', alpha=0.75,
        label=f'observed\n(median {np.nanmedian(rel):.2f})')
ax.set_xlabel('split-half correlation of tuning curve (odd vs even trials)')
ax.set_ylabel('units')
ax.set_title('Tuning curves are reproducible within session')
ax.legend(fontsize=8, frameon=False, loc='upper left')

ax = fig.add_subplot(gs[1, 2])
lat = S.latency_s.dropna().values * 1000
ax.hist(lat, bins=np.arange(0, 105, 5), color='steelblue')
ax.set_xlabel('onset latency at BF (ms)')
ax.set_ylabel('units')
ax.set_title(f'Tone-onset latency\nmedian {np.median(lat):.0f} ms  (n={len(lat)})')
ax2 = ax.inset_axes([0.52, 0.45, 0.45, 0.5])
ax2.hist(S.sparseness.dropna(), bins=20, color='seagreen')
ax2.set_xlabel('lifetime sparseness', fontsize=8)
ax2.set_ylabel('units', fontsize=8)
ax2.tick_params(labelsize=7)

fig.suptitle('Frequency tuning across the DANDI:000986 population '
             f'({len(stats_all)} units, {len(sessions)} sessions, '
             f'{stats_all.subject.nunique()} mice)', fontsize=13, y=0.96)
fig.savefig(f'{FIG_DIR}/fig04_population.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('median split-half r: %.3f (shuffled %.3f)' % (np.nanmedian(rel), np.nanmedian(rel_shuf)))
print('median onset latency: %.1f ms' % np.median(lat))
print('best frequencies of tuned units:\n', S.best_frequency.value_counts().sort_index())

# %% [markdown]
# ## Decoding the tone from population activity
#
# If frequency tuning is real and distributed, a downstream reader should be able to tell which
# of the five tones was played from a single trial. We fit a multinomial logistic regression on
# the 100 ms population spike-count vector with 5-fold cross-validation.
#
# The essential control is to run exactly the same decoder on the *pre-tone* window. Any
# above-chance accuracy there would mean that slow drift or structure in the stimulus sequence,
# rather than the tone itself, was carrying the information.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)


def decode(X, y, n_splits=5):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, multi_class='multinomial', C=0.1))
    return cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True,
                                                           random_state=0), n_jobs=-1)


main = max(sessions, key=lambda k: results[k]['stats'].tuned.sum())
X, y = results[main]['counts_ev'].astype(float), results[main]['freqs']
pred = decode(X, y)
acc = (pred == y).mean()
cm = confusion_matrix(y, pred, labels=ufreq, normalize='true')

sizes = sorted({n for n in [1, 2, 5, 10, 20, 40, 80, 160, X.shape[1]] if n <= X.shape[1]})
curve_mean, curve_sd = [], []
for n in tqdm(sizes, desc='subsampling units'):
    accs = [(decode(X[:, rng.choice(X.shape[1], n, replace=False)], y) == y).mean()
            for _ in range(8 if n < X.shape[1] else 1)]
    curve_mean.append(np.mean(accs))
    curve_sd.append(np.std(accs))

per_session, per_session_bl = {}, {}
for k in tqdm(sessions, desc='decoding each session'):
    r = results[k]
    per_session[k] = (decode(r['counts_ev'].astype(float), r['freqs']) == r['freqs']).mean()
    per_session_bl[k] = (decode(r['counts_bl'].astype(float), r['freqs']) == r['freqs']).mean()

# %% [markdown]
# ### Figure 5: single-trial decoding

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.4))
ax = axes[0]
im = ax.imshow(cm, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_yticks(range(5))
ax.set_yticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('decoded frequency (kHz)')
ax.set_ylabel('presented frequency (kHz)')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='w' if cm[i, j] < 0.6 else 'k')
ax.set_title(f'{main.split("/")[-1].replace("_behavior.nwb","")}\n'
             f'{X.shape[1]} units, 5-fold CV, accuracy {acc:.2f}', fontsize=11)
plt.colorbar(im, ax=ax, pad=0.02, label='P(decoded | presented)')

ax = axes[1]
ax.errorbar(sizes, curve_mean, yerr=curve_sd, marker='o', color='crimson', capsize=3)
ax.axhline(0.2, color='0.5', ls='--', label='chance (1/5)')
ax.set_xscale('log')
ax.set_xlabel('number of units in the population')
ax.set_ylabel('decoding accuracy')
ax.set_title('Accuracy grows with population size')
ax.legend(frameon=False)

ax = axes[2]
labels = [k.split('/')[-1].replace('_behavior.nwb', '').replace('sub-', '') for k in sessions]
yy = np.arange(len(sessions))
ax.barh(yy + 0.2, [per_session[k] for k in sessions], 0.4, color='steelblue',
        label='tone-evoked window (5-105 ms)')
ax.barh(yy - 0.2, [per_session_bl[k] for k in sessions], 0.4, color='0.75',
        label='pre-tone window (-105 to -5 ms)')
ax.axvline(0.2, color='crimson', ls='--')
ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
ax.set_yticks(yy)
ax.set_yticklabels([f'{l} ({len(results[k]["stats"])}u)' for l, k in zip(labels, sessions)],
                   fontsize=7.5)
ax.set_xlabel('decoding accuracy')
ax.set_title('Single-trial decoding in every session\n(red dashed = chance, 1/5)')
fig.suptitle('Tone frequency is decodable from single-trial auditory-cortex population '
             'activity (5-105 ms spike counts)', y=1.03, fontsize=12)
fig.savefig(f'{FIG_DIR}/fig05_decoding.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('best session accuracy %.3f' % acc)
print('accuracy per session: ', {k.split("/")[-1][:-4]: round(v, 3)
                                 for k, v in per_session.items()})
print('pre-tone control:     ', {k.split("/")[-1][:-4]: round(v, 3)
                                 for k, v in per_session_bl.items()})

# %% [markdown]
# ## A NeMoS GLM with one temporal kernel per frequency
#
# The analyses so far summarise the response with a single number per trial. A Poisson GLM
# instead models the spike train in 5 ms bins as a function of the tone sequence, and gives
# each frequency its own 200 ms kernel, so the *time course* of the response to each frequency
# is estimated directly. A spike-history kernel absorbs refractoriness and bursting.
#
# The comparison model is identical except that all five tones share a single kernel, so it
# knows that a tone occurred but not which one. Fitting on the first three tone blocks and
# scoring on the fourth turns frequency tuning into a cross-validated prediction question.

# %%
import nemos as nmo

BIN = 0.005
WIN_BINS = 40      # 200 ms stimulus kernel
HIST_BINS = 30     # 150 ms spike-history kernel
N_BASIS = 8

brk = np.where(np.diff(onsets) > 5)[0]
starts = np.r_[onsets[0], onsets[brk + 1]] - 0.5
ends = np.r_[onsets[brk], onsets[-1]] + 1.0
tone_ep = nap.IntervalSet(start=starts, end=ends)
print(tone_ep, 'total %.0f s of tone blocks' % tone_ep.tot_length())

counts = units.count(BIN, ep=tone_ep)
tt = counts.t
stim = np.zeros((len(tt), len(ufreq)), dtype=np.float32)
idx = np.searchsorted(tt, onsets) - 1
valid = (idx >= 0) & (idx < len(tt))
for i, f in enumerate(ufreq):
    np.add.at(stim[:, i], idx[valid & (freqs == f)], 1.0)
print('tone events per frequency placed on the %.0f ms grid:' % (BIN * 1000), stim.sum(0))

stim_tsd = nap.TsdFrame(t=tt, d=stim, time_support=counts.time_support)
stim_any = nap.Tsd(t=tt, d=stim.sum(1), time_support=counts.time_support)
stim_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=WIN_BINS, label='tone')
hist_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=HIST_BINS, label='history')
Xstim = np.concatenate([np.asarray(stim_basis.compute_features(stim_tsd[:, i]))
                        for i in range(len(ufreq))], axis=1)
Xstim_shared = np.asarray(stim_basis.compute_features(stim_any))

split_t = starts[-1]
train, test = tt < split_t, tt >= split_t
_, kern = stim_basis.evaluate_on_grid(WIN_BINS)
lag = np.arange(WIN_BINS) * BIN


def event_average(sig, ev_idx, pre=10, post=40):
    """Average a per-bin signal in a window around event bin indices."""
    keep = (ev_idx >= pre) & (ev_idx + post < len(sig))
    return np.stack([sig[i - pre:i + post] for i in ev_idx[keep]]).mean(0)


PRE, POST = 10, 40
lag_psth = (np.arange(-PRE, POST) + 0.5) * BIN
rows, filters, pred_psth, meas_psth = [], {}, {}, {}
for uid in tqdm(examples, desc='GLM fits'):
    yv = np.asarray(counts.loc[uid]).astype(np.float32)
    Xh = np.asarray(hist_basis.compute_features(nap.Tsd(t=tt, d=yv)))
    Xfull = np.concatenate([Xstim, Xh], axis=1)
    Xred = np.concatenate([Xstim_shared, Xh], axis=1)
    ok_tr = train & ~np.isnan(Xfull).any(1)
    ok_te = test & ~np.isnan(Xfull).any(1)

    m_full = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-4,
                         solver_name='LBFGS').fit(Xfull[ok_tr], yv[ok_tr])
    m_red = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-4,
                        solver_name='LBFGS').fit(Xred[ok_tr], yv[ok_tr])
    sc_full = float(m_full.score(Xfull[ok_te], yv[ok_te], score_type='pseudo-r2-McFadden'))
    sc_red = float(m_red.score(Xred[ok_te], yv[ok_te], score_type='pseudo-r2-McFadden'))
    ll_full = float(m_full.score(Xfull[ok_te], yv[ok_te], score_type='log-likelihood'))
    ll_red = float(m_red.score(Xred[ok_te], yv[ok_te], score_type='log-likelihood'))

    coef = np.asarray(m_full.coef_)[:len(ufreq) * N_BASIS].reshape(len(ufreq), N_BASIS)
    filters[uid] = kern @ coef.T
    rate_hat = np.asarray(m_full.predict(np.nan_to_num(Xfull))) / BIN
    te_onset = idx[valid & (tt[idx[valid]] >= split_t)]
    te_freq = freqs[valid][tt[idx[valid]] >= split_t]
    pred_psth[uid] = np.stack([event_average(rate_hat, te_onset[te_freq == f], PRE, POST)
                               for f in ufreq])
    meas_psth[uid] = np.stack([event_average(yv / BIN, te_onset[te_freq == f], PRE, POST)
                               for f in ufreq])
    rows.append(dict(unit=uid, pr2_full=sc_full, pr2_reduced=sc_red,
                     dll_per_bin=ll_full - ll_red,
                     bf=float(s.loc[uid, 'best_frequency'])))
    tqdm.write(f'unit {uid}: held-out pseudo-R2 {sc_full:.4f} (frequency-specific) vs '
               f'{sc_red:.4f} (frequency-blind)')

glm_table = pd.DataFrame(rows).set_index('unit')
glm_table.to_csv('glm_model_comparison.csv')
print(glm_table.to_string())

# %% [markdown]
# ### Figures 6 and 7: GLM kernels and held-out model comparison
#
# The kernels recover the shape of the response to each frequency, and the model's predictions
# on the held-out block track the measured PSTHs including their frequency ordering. Giving the
# model access to tone identity improves held-out prediction for four of the five example units;
# the exception is the 8 kHz unit, which fires at well under 1 Hz and contributes too few spikes
# in the held-out block to pay for the extra parameters.

# %%
fig, axes = plt.subplots(2, len(examples), figsize=(3.1 * len(examples), 6.4),
                         gridspec_kw=dict(hspace=0.45, wspace=0.35))
for c, uid in enumerate(examples):
    ax = axes[0, c]
    for i, f in enumerate(ufreq):
        ax.plot(lag * 1000, filters[uid][:, i], color=fcol[f], lw=1.6, label=f'{f/1000:g} kHz')
    ax.axhline(0, color='0.6', lw=0.8, ls=':')
    ax.set_title(f'unit {uid} (BF {s.loc[uid,"best_frequency"]/1000:g} kHz)', fontsize=10)
    ax.set_xlabel('lag from tone onset (ms)', fontsize=9)
    if c == 0:
        ax.set_ylabel('GLM gain (log rate)', fontsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1, c]
    for i, f in enumerate(ufreq):
        ax.plot(lag_psth * 1000, meas_psth[uid][i], color=fcol[f], lw=1.0, alpha=0.55)
        ax.plot(lag_psth * 1000, pred_psth[uid][i], color=fcol[f], lw=1.8, ls='--')
    ax.set_xlabel('time from onset (ms)', fontsize=9)
    if c == 0:
        ax.set_ylabel('firing rate (Hz)\nheld-out block', fontsize=9)
        ax.plot([], [], color='k', lw=1.0, alpha=0.55, label='measured')
        ax.plot([], [], color='k', lw=1.8, ls='--', label='GLM prediction')
        ax.legend(fontsize=7, frameon=False)
fig.suptitle('NeMoS Poisson GLM — frequency-specific tone kernels (top) and held-out '
             'predictions (bottom)\n'
             'fitted on tone blocks 1-3; all predictions and scores are from held-out block 4',
             fontsize=11, y=1.0)
fig.savefig(f'{FIG_DIR}/fig06_glm_kernels.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(wspace=0.3))
x = np.arange(len(glm_table))
lbl = [f'unit {u}\nBF {b/1000:g} kHz' for u, b in zip(glm_table.index, glm_table.bf)]
axs[0].bar(x - 0.2, glm_table.pr2_full, 0.4, color='crimson',
           label='frequency-specific kernels')
axs[0].bar(x + 0.2, glm_table.pr2_reduced, 0.4, color='0.7',
           label='frequency-blind (single kernel)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(lbl, fontsize=8)
axs[0].set_ylabel('held-out pseudo-$R^2$ (McFadden)')
axs[0].set_title('Goodness of fit on the held-out block')
axs[0].legend(frameon=False, fontsize=9)

diff = glm_table.pr2_full - glm_table.pr2_reduced
axs[1].bar(x, diff, 0.55, color=['crimson' if d > 0 else '0.5' for d in diff])
axs[1].axhline(0, color='k', lw=0.8)
axs[1].set_xticks(x)
axs[1].set_xticklabels(lbl, fontsize=8)
axs[1].set_ylabel('$\\Delta$ pseudo-$R^2$ (specific - blind)')
axs[1].set_title('Improvement from knowing which tone was played')
for xi, d in zip(x, diff):
    axs[1].annotate(f'{d:+.4f}', (xi, d), fontsize=8, ha='center',
                    va='bottom' if d > 0 else 'top',
                    textcoords='offset points', xytext=(0, 3 if d > 0 else -3))
fig.suptitle('Frequency information improves held-out prediction', fontsize=12, y=1.0)
fig.savefig(f'{FIG_DIR}/fig07_glm_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ## Does tuning depend on arousal?
#
# DANDI:000986 was collected to relate auditory responses to arousal, so the pupil trace
# provides a robustness check. Trials are split at the session median of the pupil diameter
# averaged over the 500 ms before tone onset, and tuning curves are recomputed within each half
# for one session per mouse. Curves are normalised and aligned using the all-trial best
# frequency so that neither state gets to pick its own peak.

# %%
by_subj = {}
for k, r in results.items():
    subj = r['stats'].subject.iloc[0]
    if subj not in by_subj or len(r['stats']) > len(results[by_subj[subj]]['stats']):
        by_subj[subj] = k
ar_sessions = [by_subj[k] for k in sorted(by_subj)]

rows, curves_lo, curves_hi = [], [], []
for k in tqdm(ar_sessions, desc='arousal split'):
    r = results[k]
    nwb_k = nap.NWBFile(open_nwb(DANDISET, assets[k]))
    pup = nwb_k['pupil_diameter']
    pt, pdiam = pup.t, np.asarray(pup)
    lo_i = np.searchsorted(pt, r['onsets'] - 0.5)
    hi_i = np.searchsorted(pt, r['onsets'])
    pre_pupil = np.array([np.nanmean(pdiam[a:b]) if b > a else np.nan
                          for a, b in zip(lo_i, hi_i)])
    okp = ~np.isnan(pre_pupil)
    med = np.nanmedian(pre_pupil)
    low, high = okp & (pre_pupil <= med), okp & (pre_pupil > med)
    d = r['counts_ev'] / (WIN_EV[1] - WIN_EV[0]) - r['counts_bl'] / (WIN_BL[1] - WIN_BL[0])
    keep = (r['stats'].tuned & r['stats'].responsive).values
    for mask, store in ((low, curves_lo), (high, curves_hi)):
        store.append(np.array([[d[mask & (r['freqs'] == f), j].mean() for f in ufreq]
                               for j in np.where(keep)[0]]))
    rows.append(dict(session=k, subject=r['stats'].subject.iloc[0], n_units=int(keep.sum()),
                     pupil_median=float(med), n_low=int(low.sum()), n_high=int(high.sum())))

Clo, Chi = np.concatenate(curves_lo), np.concatenate(curves_hi)
Cpool = np.concatenate([results[k]['curves'].values[(results[k]['stats'].tuned &
                                                     results[k]['stats'].responsive).values]
                        for k in ar_sessions])
bf_pool = Cpool.argmax(1)
big = Cpool.max(1) >= 1.0
bf_lo, bf_hi = Clo.argmax(1), Chi.argmax(1)
agree = (bf_lo[big] == bf_hi[big]).mean()
peak_lo = Clo[np.arange(len(Clo)), bf_pool]
peak_hi = Chi[np.arange(len(Chi)), bf_pool]
w = sstats.wilcoxon(peak_lo[big], peak_hi[big])


def aligned(Cx):
    acc = {o: [] for o in offsets}
    for i in np.where(big)[0]:
        norm = Cx[i] / max(Cpool[i].max(), 1e-9)
        for j in range(len(ufreq)):
            acc[j - bf_pool[i]].append(norm[j])
    mm = np.array([np.mean(acc[o]) if acc[o] else np.nan for o in offsets])
    ss = np.array([np.std(acc[o]) / np.sqrt(len(acc[o])) if acc[o] else np.nan for o in offsets])
    return mm, ss


mlo, slo = aligned(Clo)
mhi, shi = aligned(Chi)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.33))
ax = axes[0]
ax.errorbar(offsets, mlo, yerr=slo, marker='o', color='navy', lw=2, capsize=3,
            label='low arousal')
ax.errorbar(offsets, mhi, yerr=shi, marker='s', color='darkorange', lw=2, capsize=3,
            label='high arousal')
ax.axvline(0, color='0.6', ls=':')
ax.axhline(0, color='0.6', ls=':')
ax.set_xlabel('octaves from best frequency')
ax.set_ylabel('normalised evoked rate')
ax.set_title('Tuning shape is preserved across arousal')
ax.legend(frameon=False)

ax = axes[1]
lim = np.percentile(np.r_[peak_lo[big], peak_hi[big]], 99)
ax.plot(peak_lo[big], peak_hi[big], '.', ms=4, color='0.35', alpha=0.6)
ax.plot([0, lim], [0, lim], 'r--', lw=1)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel('evoked rate at BF, low arousal (Hz)')
ax.set_ylabel('evoked rate at BF, high arousal (Hz)')
ax.set_title(f'Response gain (n={int(big.sum())} units)\nWilcoxon p={w.pvalue:.1e}')

ax = axes[2]
cmx = np.zeros((len(ufreq), len(ufreq)))
for a, b in zip(bf_lo[big], bf_hi[big]):
    cmx[a, b] += 1
cmx = cmx / cmx.sum(1, keepdims=True)
im = ax.imshow(cmx, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_yticks(range(5))
ax.set_yticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('BF, high arousal (kHz)')
ax.set_ylabel('BF, low arousal (kHz)')
ax.set_title(f'Best frequency is stable\n{agree*100:.0f}% of units keep the same BF',
             fontsize=10)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cmx[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='w' if cmx[i, j] < 0.6 else 'k')
plt.colorbar(im, ax=ax, pad=0.02, label='fraction')
fig.suptitle('Frequency tuning under low vs high arousal (median split on pre-tone pupil '
             f'diameter, {len(ar_sessions)} sessions, one per mouse)', y=1.04, fontsize=12)
fig.savefig(f'{FIG_DIR}/fig08_arousal.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(pd.DataFrame(rows).to_string())
print(f'BF preserved across arousal states in {agree*100:.1f}% of {int(big.sum())} units')
print('median evoked rate at BF: low %.2f Hz, high %.2f Hz (Wilcoxon p=%.2e)'
      % (np.median(peak_lo[big]), np.median(peak_hi[big]), w.pvalue))

# %% [markdown]
# ## Summary
#
# Every analysis points the same way. Of 1564 units recorded across 15 sessions in 5 mice,
# 81% respond to tones and 75% respond differently to different tone frequencies at
# FDR q = 0.05. Responses begin about 18 ms after tone onset, the expected latency for mouse
# auditory cortex. Tuning curves computed from odd and even trials agree almost perfectly
# (median r = 0.98) while the same measurement on frequency-shuffled trials gives r = 0.0,
# so the tuning is a property of the neurons and not of the estimator.
#
# The tuning is strong enough to read out: a linear decoder recovers which of the five tones
# was played on a single trial with up to 95% accuracy from 100 ms of population spike counts,
# against a chance level of 20%, and the identical decoder applied to the pre-tone window sits
# at chance in all 15 sessions. A Poisson GLM that gives each frequency its own temporal kernel
# predicts held-out spike trains better than a frequency-blind model, and the tuning survives a
# median split on pupil diameter with 87% of units keeping the same best frequency in both
# arousal states.
#
# The main limitation is the stimulus set: 000986 uses five frequencies one octave apart at a
# single sound level, so tuning width cannot be measured with any precision and the tuning
# curves are five-point samples of what is really a continuous frequency response area. The
# conclusions here concern the existence, reliability and readability of frequency tuning, not
# its bandwidth.
