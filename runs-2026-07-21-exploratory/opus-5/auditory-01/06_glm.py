"""NeMoS Poisson GLM: time-resolved frequency tuning and a held-out model comparison.

The full model gives every tone frequency its own temporal kernel; the reduced model
gives all tones a single shared kernel (frequency-blind). The difference in held-out
log-likelihood is a direct, cross-validated test of frequency tuning.
"""
import pickle
import numpy as np
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt
from tqdm import tqdm

import loaders

SESSION = 'sub-LA9/sub-LA9_ses-1_behavior.nwb'
BIN = 0.005
WIN_BINS = 40            # 200 ms stimulus kernel
HIST_BINS = 30           # 150 ms spike-history kernel
N_BASIS = 8

res = pickle.load(open('results_000986.pkl', 'rb'))[SESSION]
assets = loaders.list_assets('000986')
nwbfile = loaders.open_nwb('000986', assets[SESSION])
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
trials = nwbfile.trials.to_dataframe()
onsets, freqs = trials.start_time.values, trials.stim_frequency.values
ufreq = res['ufreq']

# tone-presentation blocks (trials come in 4 blocks separated by silent periods)
brk = np.where(np.diff(onsets) > 5)[0]
starts = np.r_[onsets[0], onsets[brk + 1]] - 0.5
ends = np.r_[onsets[brk], onsets[-1]] + 1.0
tone_ep = nap.IntervalSet(start=starts, end=ends)
print('tone blocks:', tone_ep, 'total %.0f s' % tone_ep.tot_length())

counts = units.count(BIN, ep=tone_ep)          # TsdFrame (n_bins, n_units)
tt = counts.t
print('bins:', counts.shape)

# one binary event train per frequency, on the spike-count time grid
stim = np.zeros((len(tt), len(ufreq)), dtype=np.float32)
idx = np.searchsorted(tt, onsets) - 1
valid = (idx >= 0) & (idx < len(tt))
for i, f in enumerate(ufreq):
    sel = idx[valid & (freqs == f)]
    np.add.at(stim[:, i], sel, 1.0)
print('events placed per frequency:', stim.sum(0))
stim_tsd = nap.TsdFrame(t=tt, d=stim, time_support=counts.time_support)
stim_any = nap.Tsd(t=tt, d=stim.sum(1), time_support=counts.time_support)

stim_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=WIN_BINS, label='tone')
hist_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=HIST_BINS, label='history')

Xstim = np.concatenate([np.asarray(stim_basis.compute_features(stim_tsd[:, i]))
                        for i in range(len(ufreq))], axis=1)      # (n_bins, 5*N_BASIS)
Xstim_shared = np.asarray(stim_basis.compute_features(stim_any))  # (n_bins, N_BASIS)

# train / test split on whole tone blocks: first 3 blocks train, last block test
split_t = starts[-1]
train = tt < split_t
test = ~train
print('train bins %d, test bins %d' % (train.sum(), test.sum()))

stats = res['stats']
tuned = stats[stats.tuned & stats.responsive]
examples = [tuned[tuned.best_frequency == f].peak_evoked_hz.idxmax()
            for f in ufreq if (tuned.best_frequency == f).any()]

_, kern = stim_basis.evaluate_on_grid(WIN_BINS)
lag = np.arange(WIN_BINS) * BIN

def event_average(sig, ev_idx, pre=10, post=40):
    """Average a per-bin signal in a window around event bin indices."""
    keep = (ev_idx >= pre) & (ev_idx + post < len(sig))
    seg = np.stack([sig[i - pre:i + post] for i in ev_idx[keep]])
    return seg.mean(0)


PRE, POST = 10, 40
lag_psth = (np.arange(-PRE, POST) + 0.5) * BIN
rows = []
filters, pred_psth, meas_psth = {}, {}, {}
for uid in tqdm(examples, desc='GLM fits'):
    y = np.asarray(counts.loc[uid]).astype(np.float32)
    Xh = np.asarray(hist_basis.compute_features(nap.Tsd(t=tt, d=y)))
    Xfull = np.concatenate([Xstim, Xh], axis=1)
    Xred = np.concatenate([Xstim_shared, Xh], axis=1)

    ok_tr = train & ~np.isnan(Xfull).any(1)
    ok_te = test & ~np.isnan(Xfull).any(1)

    m_full = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-4,
                         solver_name='LBFGS').fit(Xfull[ok_tr], y[ok_tr])
    m_red = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=1e-4,
                        solver_name='LBFGS').fit(Xred[ok_tr], y[ok_tr])
    sc_full = m_full.score(Xfull[ok_te], y[ok_te], score_type='pseudo-r2-McFadden')
    sc_red = m_red.score(Xred[ok_te], y[ok_te], score_type='pseudo-r2-McFadden')
    ll_full = m_full.score(Xfull[ok_te], y[ok_te], score_type='log-likelihood')
    ll_red = m_red.score(Xred[ok_te], y[ok_te], score_type='log-likelihood')
    coef = np.asarray(m_full.coef_)[:len(ufreq) * N_BASIS].reshape(len(ufreq), N_BASIS)
    filters[uid] = kern @ coef.T                     # (WIN_BINS, n_freq)

    # held-out PSTHs: model prediction vs measurement, same trials
    Xte = np.nan_to_num(Xfull)
    rate_hat = np.asarray(m_full.predict(Xte)) / BIN
    te_onset = idx[valid & (tt[idx[valid]] >= split_t)]
    te_freq = freqs[valid][tt[idx[valid]] >= split_t]
    pred_psth[uid] = np.stack([event_average(rate_hat, te_onset[te_freq == f], PRE, POST)
                               for f in ufreq])
    meas_psth[uid] = np.stack([event_average(y / BIN, te_onset[te_freq == f], PRE, POST)
                               for f in ufreq])
    rows.append(dict(unit=uid, pr2_full=float(sc_full), pr2_reduced=float(sc_red),
                     dll_per_bin=float(ll_full - ll_red),
                     bf=float(stats.loc[uid, 'best_frequency'])))
    tqdm.write(f"unit {uid}: held-out pseudo-R2 full {sc_full:.4f} vs frequency-blind "
               f"{sc_red:.4f}; ΔLL/bin {ll_full-ll_red:+.5f}")

cmap = plt.get_cmap('viridis')
fcol = {f: cmap(i / (len(ufreq) - 1)) for i, f in enumerate(ufreq)}
n = len(examples)
fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.4),
                         gridspec_kw=dict(hspace=0.45, wspace=0.35))
for c, uid in enumerate(examples):
    ax = axes[0, c]
    for i, f in enumerate(ufreq):
        ax.plot(lag * 1000, filters[uid][:, i], color=fcol[f], lw=1.6, label=f'{f/1000:g} kHz')
    ax.axhline(0, color='0.6', lw=0.8, ls=':')
    ax.set_title(f'unit {uid} (BF {stats.loc[uid,"best_frequency"]/1000:g} kHz)', fontsize=10)
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
fig.savefig('figures/fig06_glm_kernels.png', dpi=150, bbox_inches='tight')
plt.close(fig)

import pandas as pd
tab = pd.DataFrame(rows).set_index('unit')
print(tab.to_string())
tab.to_csv('glm_model_comparison.csv')

fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(wspace=0.3))
x = np.arange(len(tab))
lbl = [f'unit {u}\nBF {b/1000:g} kHz' for u, b in zip(tab.index, tab.bf)]
axs[0].bar(x - 0.2, tab.pr2_full, 0.4, color='crimson',
           label='frequency-specific kernels')
axs[0].bar(x + 0.2, tab.pr2_reduced, 0.4, color='0.7',
           label='frequency-blind (single kernel)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(lbl, fontsize=8)
axs[0].set_ylabel('held-out pseudo-$R^2$ (McFadden)')
axs[0].set_title('Goodness of fit on the held-out block')
axs[0].legend(frameon=False, fontsize=9)

diff = tab.pr2_full - tab.pr2_reduced
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
fig.savefig('figures/fig07_glm_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
