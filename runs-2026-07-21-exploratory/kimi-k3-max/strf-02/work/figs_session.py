"""Validation figures for one session."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from session_strf import (load_session, trial_bin_counts, FREQ_VALS, BIN_C,
                          BASE_MASK, ASSETS)

label = 'LA11_ses2'
d = np.load(f'session_{label}.npz')
strfs, glm_rate, pr2 = d['strfs'], d['glm_rate'], d['pseudo_r2']
tuned = d['tuned']
bf, peak_lat, sep = d['bf'], d['peak_lat'], d['separability']

unit_spikes, onsets, freqs, nwbfile = load_session(ASSETS[label])

# ---- Fig 1: raw data validation for an example tuned unit
ex = 50
counts = trial_bin_counts(unit_spikes[ex], onsets)
fig = plt.figure(figsize=(13, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
ax = fig.add_subplot(gs[0, :])
# raster: trials sorted by frequency
order = np.argsort(freqs, kind='stable')
cmap_f = {fv: c for fv, c in zip(FREQ_VALS, plt.cm.viridis(np.linspace(0, 0.9, 5)))}
spk = unit_spikes[ex]
win = (-0.05, 0.15)
for row, ti in enumerate(order):
    t0 = onsets[ti]
    lo, hi = np.searchsorted(spk, [t0 + win[0], t0 + win[1]])
    ts = spk[lo:hi] - t0
    ax.scatter(ts * 1000, np.full_like(ts, row), s=1, color=cmap_f[freqs[ti]], lw=0)
ax.set_xlim(win[0]*1000, win[1]*1000)
ax.axvline(0, color='k', lw=0.5); ax.axvline(25, color='k', lw=0.5, ls='--')
ax.set_xlabel('time from tone onset (ms)'); ax.set_ylabel('trial (sorted by frequency)')
ax.set_title(f'unit {ex} spike raster (color = tone frequency)')
# PSTH per frequency
ax2 = fig.add_subplot(gs[1, 0])
for fv in FREQ_VALS:
    m = freqs == fv
    rate = counts[m].mean(axis=0) / 0.005
    ax2.plot(BIN_C*1000, rate, color=cmap_f[fv], label=f'{int(fv/1000)} kHz')
ax2.axvline(0, color='k', lw=0.5); ax2.axvline(25, color='k', lw=0.5, ls='--')
ax2.set_xlabel('time from onset (ms)'); ax2.set_ylabel('rate (Hz)')
ax2.legend(frameon=False, fontsize=8); ax2.set_title('per-frequency PSTH')
# session-wide spike rate context
ax3 = fig.add_subplot(gs[1, 1])
t_edges = np.arange(onsets[0], onsets[-1], 1.0)
ax3.plot(0.5*(t_edges[:-1]+t_edges[1:]), np.histogram(spk, t_edges)[0], lw=0.5)
ax3.set_xlabel('session time (s)'); ax3.set_ylabel('spikes / s')
ax3.set_title('unit firing rate across session')
fig.savefig('fig1_data_validation.png', dpi=150)
print('fig1 saved')

# ---- Fig 2: grid of example rate-map STRFs (diverse BFs, strong units)
tu = np.where(tuned)[0]
strong = tu[np.argsort(-d['peak_hz'][tu])]
# pick diverse: best unit per BF + a couple with suppression
picks = []
for fv in FREQ_VALS:
    cand = [u for u in strong if bf[u] == fv]
    if cand: picks.append(cand[0])
picks = picks[:5]
# add 4 more strong diverse ones not already picked
for u in strong:
    if len(picks) >= 9: break
    if u not in picks: picks.append(u)
fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
vmax = np.percentile(np.abs(strfs[picks]), 99)
for ax, u in zip(axes.flat, picks):
    im = ax.imshow(strfs[u], aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[BIN_C[0]*1000, BIN_C[-1]*1000, -0.5, 4.5], vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(5)); ax.set_yticklabels([str(int(fv/1000)) for fv in FREQ_VALS], fontsize=8)
    ax.axvline(0, color='k', lw=0.4); ax.axvline(25, color='k', lw=0.4, ls='--')
    ax.set_title(f'unit {u} | BF {int(bf[u]/1000)} kHz | lat {peak_lat[u]*1000:.0f} ms', fontsize=9)
    ax.set_xlabel('time (ms)', fontsize=8)
    if ax in axes[:, 0]: ax.set_ylabel('freq (kHz)', fontsize=8)
fig.suptitle('Rate-map STRFs (baseline-subtracted evoked rate), example tuned units')
fig.colorbar(im, ax=axes, label='evoked - baseline (Hz)', shrink=0.8)
fig.savefig('fig2_example_strfs.png', dpi=150)
print('fig2 saved')

# ---- Fig 3: GLM vs data STRF for 3 units with high pseudo-R2
tu2 = np.where(tuned & ~np.isnan(pr2))[0]
top = tu2[np.argsort(-pr2[tu2])[:3]]
fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
for r, u in enumerate(top):
    g = glm_rate[u].copy()
    g_base = g[:, BASE_MASK].mean()
    g = g - g_base
    vm = np.abs(strfs[u]).max()
    for c, (M, ttl) in enumerate([(strfs[u], 'data rate-map'), (g, 'GLM STRF')]):
        ax = axes[r, c]
        im = ax.imshow(M, aspect='auto', origin='lower', cmap='RdBu_r',
                       extent=[BIN_C[0]*1000, BIN_C[-1]*1000, -0.5, 4.5], vmin=-vm, vmax=vm)
        ax.set_yticks(range(5)); ax.set_yticklabels([str(int(fv/1000)) for fv in FREQ_VALS], fontsize=8)
        ax.axvline(0, color='k', lw=0.4); ax.axvline(25, color='k', lw=0.4, ls='--')
        ax.set_title(f'unit {u} {ttl}', fontsize=9)
    # PSTH at BF: data vs GLM prediction
    ax = axes[r, 2]
    bfi = int(np.argmin(np.abs(FREQ_VALS - bf[u])))
    ax.plot(BIN_C*1000, strfs[u][bfi], 'k', label='data')
    ax.plot(BIN_C*1000, g[bfi], 'r', label='GLM')
    ax.axvline(0, color='k', lw=0.4); ax.axvline(25, color='k', lw=0.4, ls='--')
    ax.set_title(f'BF={int(bf[u]/1000)} kHz slice | pseudo-$R^2$={pr2[u]:.3f}', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('time (ms)', fontsize=8)
fig.suptitle('Poisson GLM STRF (2-D B-spline basis, ridge) vs raw rate map')
fig.savefig('fig3_glm_strf.png', dpi=150)
print('fig3 saved')
