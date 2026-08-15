# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta Phase Precession in Hippocampal CA1 Place Cells
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark & Buzsaki,
# *"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences."*
# Bilateral silicon-probe recordings from dorsal CA1 while rats ran back and forth on a 1.6 m
# linear track for water reward.
#
# **Phenomenon.** As a rat traverses a place cell's firing field, the cell fires progressively
# earlier within each cycle of the ~8 Hz hippocampal theta rhythm. Spikes shift from late theta
# phases at field entry to early phases at field exit. This *theta phase precession*
# (O'Keefe & Recce, 1993) converts the animal's position within the field into a temporal code
# relative to the ongoing theta oscillation.
#
# **Approach.**
# 1. Stream one session from S3 (remfile chunk caching); extract CA1 pyramidal spikes,
#    linearized position, and one CA1 LFP channel over the running (Maze) epoch.
# 2. Extract theta phase (6–12 Hz band-pass + Hilbert transform).
# 3. Build directional place fields and select place cells.
# 4. For each place cell, relate theta phase at each in-field spike to the animal's
#    position within the field, quantified by circular-linear regression (Kempter et al., 2012).
# 5. Replicate across three sessions from two animals.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from pipeline import (load_session, analyze_session, circ_lin_regress,
                      LFP_RATE, THETA_BAND, MAZE_LEN)

SESSIONS = {
    'Achilles-10252013': 'https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028',
    'Achilles-11012013': 'https://dandiarchive.s3.amazonaws.com/blobs/4a9/ef6/4a9ef66e-4d89-44cf-ab50-ad029a665d50',
    'Cicero-09172014':   'https://dandiarchive.s3.amazonaws.com/blobs/aab/623/aab6235a-0144-4072-9a35-c7a3008b9870',
}

# %% [markdown]
# ## 1. Load one session (streaming)
#
# `load_session` streams the ~8.7 GB NWB file, restricts to the running (Maze) epoch,
# keeps excitatory CA1 units, linearizes position, and picks the LFP channel with the
# strongest theta (6–12 Hz) / delta (2–4 Hz) power ratio.

# %%
name0 = 'Achilles-10252013'
sess = load_session(SESSIONS[name0])
print(f"Maze epoch: {sess['t1']-sess['t0']:.0f} s")
print(f"Excitatory CA1 units: {len(sess['spikes'])}")
print(f"LFP channel {sess['best_ch']} (theta/delta = {sess['theta_ratio']:.2f})")

# %% [markdown]
# ## 2. Raw streams and theta extraction
#
# The CA1 LFP shows a clear ~8 Hz theta oscillation during running; the band-pass
# filtered signal (used for the Hilbert phase) tracks it cleanly.

# %%
from scipy.signal import butter, filtfilt
b, a = butter(3, [THETA_BAND[0]/(LFP_RATE/2), THETA_BAND[1]/(LFP_RATE/2)], btype='band')
theta_filt = filtfilt(b, a, sess['lfp']) if 'lfp' in sess else None

pos = nap.Tsd(t=sess['pos_t'], d=sess['pos_data'])
pos_s = pos.smooth(0.1)
speed = np.abs(np.gradient(pos_s.values, sess['pos_t']))

fig, ax = plt.subplots(2, 1, figsize=(12, 5.5))
w = (sess['pos_t'] > sess['t0']+600) & (sess['pos_t'] < sess['t0']+660)
ax[0].plot(sess['pos_t'][w], sess['pos_data'][w], 'k')
ax[0].set_ylabel('Position (m)'); ax[0].set_title(f'{name0}: linearized position (60 s)')
lw = (sess['lfp_t'] > sess['t0']+610) & (sess['lfp_t'] < sess['t0']+612)
# recover LFP for display from a quick re-band-pass of the stored phase is not possible;
# reconstruct theta wave from phase for illustration
ax[1].plot(sess['lfp_t'][lw], np.cos(sess['theta_phase'][lw]), 'r', lw=1.3, label='theta (cos phase)')
ax[1].set_ylabel('theta'); ax[1].set_xlabel('Time (s)')
ax[1].set_title('Reconstructed theta oscillation (2 s)'); ax[1].legend(loc='upper right')
plt.tight_layout()
plt.savefig('fig_06_streams_notebook.png', dpi=130)
print('saved fig_06_streams_notebook.png')

# %% [markdown]
# (The primary raw-signal validation figure, showing the unfiltered LFP overlaid with the
# theta band-pass, is `fig_01_raw_streams.png`, produced by `02_preprocess.py`.)

# %% [markdown]
# ## 3. Directional place fields
#
# Place fields on a linear track are direction-specific, so we split running into
# rightward and leftward epochs and compute a 1D tuning curve per direction.
# `analyze_session` returns the per-cell phase-precession fit; here we also show the
# place-field tiling directly.

# %%
units = nap.TsGroup({u: nap.Ts(t=st) for u, st in sess['spikes'].items()})
vel = np.gradient(pos_s.values, sess['pos_t'])
speed_tsd = nap.Tsd(t=sess['pos_t'], d=np.abs(vel))
run_ep = speed_tsd.threshold(0.03, 'above').time_support.drop_short_intervals(0.5).merge_close_intervals(0.2)
direction = nap.Tsd(t=sess['pos_t'], d=np.sign(vel))
right = direction.threshold(0.0, 'above').time_support.intersect(run_ep).drop_short_intervals(0.3)
left = direction.threshold(0.0, 'below').time_support.intersect(run_ep).drop_short_intervals(0.3)
tc_r = nap.compute_1d_tuning_curves(units, pos, 50, ep=right, minmax=(0, MAZE_LEN))
tc_l = nap.compute_1d_tuning_curves(units, pos, 50, ep=left, minmax=(0, MAZE_LEN))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, tc, ttl in [(axes[0], tc_r, 'rightward'), (axes[1], tc_l, 'leftward')]:
    keep = [u for u in units.keys() if tc[u].max() > 5]
    order = sorted(keep, key=lambda u: tc[u].idxmax())
    M = np.stack([tc[u].values/tc[u].max() for u in order])
    im = ax.imshow(M, aspect='auto', origin='lower', extent=[0, MAZE_LEN, 0, len(order)],
                   cmap='viridis')
    ax.set_title(f'{ttl} place fields (n={len(order)})')
    ax.set_xlabel('Position (m)'); ax.set_ylabel('Cell (sorted by peak)')
    plt.colorbar(im, ax=ax, label='norm. rate')
plt.tight_layout()
plt.savefig('fig_07_placefields_notebook.png', dpi=130)
print('saved fig_07_placefields_notebook.png')

# %% [markdown]
# ## 4. Phase precession, single session
#
# `analyze_session` fits every place cell: it takes in-field spikes during running in the
# preferred direction, reads position-in-field and theta phase at each spike, and runs a
# circular-linear regression. A negative slope means spikes advance to earlier theta phases
# as the animal moves through the field — the signature of phase precession.

# %%
res = analyze_session(sess)
slopes = np.array([r['slope'] for r in res])
rhos = np.array([r['rho'] for r in res])
pvals = np.array([r['pval'] for r in res])
sig = pvals < 0.05
print(f"Place cells fit: {len(res)}")
print(f"Negative slope (precessing): {(slopes<0).sum()}/{len(res)} ({100*(slopes<0).mean():.0f}%)")
print(f"Significant (p<0.05): {sig.sum()}; of those negative: {(slopes[sig]<0).sum()}/{sig.sum()}")
print(f"Median slope: {np.median(slopes):.2f} cycles/field; median rho: {np.median(rhos):.2f}")

# %% [markdown]
# ### Example precessing cells
# Spikes are plotted over two theta cycles (0–720°) for clarity; the red line is the
# circular-linear fit.

# %%
cand = [i for i in range(len(res)) if pvals[i] < 0.05 and slopes[i] < 0]
order = sorted(cand, key=lambda i: -abs(rhos[i]))[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, i in zip(axes.ravel(), order):
    r = res[i]; ph = np.degrees(r['phase'])
    for rep in (0, 360):
        ax.plot(r['xn'], ph+rep, '.', ms=4, color='0.3', alpha=0.5)
    xx = np.linspace(0, 1, 100); yc = np.degrees(r['phi0']) + 360*r['slope']*xx
    for rep in (-360, 0, 360, 720):
        ax.plot(xx, yc+rep, 'r-', lw=1.5)
    ax.set_ylim(0, 720); ax.set_xlim(0, 1)
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_title(f"unit {r['uid']} ({r['bdir']}): slope={r['slope']:.2f} cyc, "
                 f"ρ={r['rho']:.2f}, p={r['pval']:.0e}", fontsize=9)
    ax.set_xlabel('position in field'); ax.set_ylabel('theta phase (deg)')
fig.suptitle('Theta phase precession in individual CA1 place cells', fontsize=13)
plt.tight_layout()
plt.savefig('fig_08_examples_notebook.png', dpi=130)
print('saved fig_08_examples_notebook.png')

# %% [markdown]
# ### Population summary

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].hist(slopes, bins=np.arange(-3, 1.6, 0.25), color='steelblue', edgecolor='k')
ax[0].axvline(0, color='r', ls='--'); ax[0].axvline(np.median(slopes), color='k',
             label=f'median={np.median(slopes):.2f}')
ax[0].set_xlabel('slope (theta cycles across field)'); ax[0].set_ylabel('# place cells')
ax[0].legend(); ax[0].set_title(f'(a) Slopes: {100*(slopes<0).mean():.0f}% negative')
ax[1].hist(rhos, bins=np.arange(-0.8, 0.85, 0.1), color='0.8', edgecolor='k', label='all')
ax[1].hist(rhos[sig], bins=np.arange(-0.8, 0.85, 0.1), color='crimson', edgecolor='k', label='p<0.05')
ax[1].axvline(0, color='k', ls='--'); ax[1].set_xlabel('circular-linear correlation ρ')
ax[1].set_ylabel('# place cells'); ax[1].legend(); ax[1].set_title('(b) Phase-position correlation')
axn = [i for i in range(len(res)) if pvals[i] < 0.05 and slopes[i] < 0]
allxn = np.concatenate([res[i]['xn'] for i in axn])
allph = np.concatenate([np.degrees(res[i]['phase']) for i in axn])
xn2 = np.concatenate([allxn, allxn]); ph2 = np.concatenate([allph, allph+360])
h = ax[2].hist2d(xn2, ph2, bins=[25, 40], cmap='magma')
plt.colorbar(h[3], ax=ax[2], label='# spikes')
ax[2].set_xlabel('position in field'); ax[2].set_ylabel('theta phase (deg)')
ax[2].set_yticks([0, 180, 360, 540, 720]); ax[2].set_title('(c) Pooled spikes (precessing cells)')
fig.suptitle(f'Population phase precession — {name0} (CA1, DANDI:000044)', fontsize=13)
plt.tight_layout()
plt.savefig('fig_09_population_notebook.png', dpi=130)
print('saved fig_09_population_notebook.png')

# %% [markdown]
# ## 5. Replication across sessions and animals
#
# Running the same pipeline on three sessions (two animals) shows that the large majority
# of place cells precess in every session.

# %%
import pandas as pd
from tqdm import tqdm
rows = []
for nm, url in tqdm(SESSIONS.items(), desc='sessions'):
    s = sess if nm == name0 else load_session(url)
    r = res if nm == name0 else analyze_session(s)
    sl = np.array([x['slope'] for x in r]); pv = np.array([x['pval'] for x in r])
    rows.append(dict(session=nm, n_place=len(r),
                     pct_negative=round(100*float((sl < 0).mean()), 1),
                     n_sig=int((pv < 0.05).sum()),
                     median_slope=round(float(np.median(sl)), 2)))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(range(len(summary)), summary['pct_negative'], color='seagreen', edgecolor='k')
ax.axhline(50, color='r', ls='--', label='chance')
ax.set_xticks(range(len(summary)))
ax.set_xticklabels(summary['session'], rotation=15, fontsize=9)
ax.set_ylabel('% place cells with negative slope'); ax.set_ylim(0, 100)
ax.legend(); ax.set_title('Phase precession replicates across sessions/animals')
plt.tight_layout()
plt.savefig('fig_10_replication_notebook.png', dpi=130)
print('saved fig_10_replication_notebook.png')

# %% [markdown]
# ## Conclusion
#
# CA1 place cells in DANDI:000044 show robust theta phase precession: in the Achilles
# 10-25-2013 session, ~92% of place cells have a negative phase-position slope
# (median ≈ −0.5 theta cycles across the field), and the effect replicates across
# three sessions from two animals (78–95% negative slopes, all far above the 50% chance
# level). The pooled spike density reproduces the classic diagonal band of spikes
# advancing from late to early theta phase as the animal crosses the place field.
