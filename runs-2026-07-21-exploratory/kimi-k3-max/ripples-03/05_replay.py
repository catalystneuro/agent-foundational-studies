"""Replay: Bayesian decoding of ripple events with the place-field template,
weighted-correlation score vs cell-ID shuffle null, PRE vs POST comparison."""
import numpy as np
import h5py
import remfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

rng = np.random.default_rng(7)

with open('/tmp/achilles_url.txt') as f:
    bare = f.read().strip().split('?')[0]
disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
h5py_file = h5py.File(remfile.File(bare, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
units = nwb['units']

pf = np.load('placefields.npz')
rp = np.load('ripples.npz')
unit_keys = pf['unit_keys']
place_keys = pf['place_keys']
tuning = pf['tuning']          # (n_units, 50)
centers = pf['centers']        # (50,)

# template: place cells only, rows sum-normalized? No: raw rates (Hz) for Poisson decode
key_to_idx = {k: i for i, k in enumerate(unit_keys)}
place_idx = np.array([key_to_idx[k] for k in place_keys])
F = tuning[place_idx]          # (C, P) Hz
C, P = F.shape
print(f'template: {C} place cells x {P} position bins')
logF = np.log(np.maximum(F, 1e-6))
sumF = F.sum(axis=0)           # (P,)

rip_start, rip_end, rip_peak = rp['rip_start'], rp['rip_end'], rp['rip_peak']
is_pre, is_post = rp['is_pre'], rp['is_post']

# spike trains of place cells
spikes = [units[k].t for k in place_keys]

BIN = 0.020
MIN_SPIKED_BINS = 5
MIN_ACTIVE_CELLS = 5
NSHUF = 500

def decode_event(t0, t1):
    """bin spikes, return count matrix (T,C) and bin centers"""
    nbins = int(np.ceil((t1 - t0) / BIN))
    if nbins < MIN_SPIKED_BINS:
        return None, None
    N = np.zeros((nbins, C), dtype=np.float32)
    for c, sp in enumerate(spikes):
        sp_ev = sp[(sp >= t0) & (sp < t1)]
        if sp_ev.size:
            b = ((sp_ev - t0) / BIN).astype(int)
            np.add.at(N[:, c], b, 1)
    return N, t0 + (np.arange(nbins) + 0.5) * BIN

def posterior_from_counts(N, logF_, sumF_):
    """(T,C) counts -> (T,P) posterior over position"""
    logL = N @ logF_ - BIN * sumF_[None, :]
    logL -= logL.max(axis=1, keepdims=True)
    Ppost = np.exp(logL)
    Ppost /= Ppost.sum(axis=1, keepdims=True)
    return Ppost

def weighted_corr(post, tbins, xbins):
    """posterior-mass-weighted correlation between time and position"""
    T = post.shape[0]
    w = post.ravel()
    tt = np.repeat(tbins, P)
    xx = np.tile(xbins, T)
    mt = (w * tt).sum()
    mx = (w * xx).sum()
    cov = (w * (tt - mt) * (xx - mx)).sum()
    vt = (w * (tt - mt) ** 2).sum()
    vx = (w * (xx - mx) ** 2).sum()
    if vt <= 0 or vx <= 0:
        return 0.0
    return cov / np.sqrt(vt * vx)

def analyze_events(mask, label):
    starts = rip_start[mask]
    ends = rip_end[mask]
    results = []
    for t0, t1 in tqdm(zip(starts, ends), total=len(starts), desc=f'{label} decode'):
        N, tbins = decode_event(t0, t1)
        if N is None:
            continue
        spiked_bins = (N.sum(axis=1) > 0).sum()
        active_cells = (N.sum(axis=0) > 0).sum()
        if spiked_bins < MIN_SPIKED_BINS or active_cells < MIN_ACTIVE_CELLS:
            continue
        post = posterior_from_counts(N, logF, sumF)
        score = weighted_corr(post, tbins - tbins.mean(), centers)
        # cell-ID shuffle null
        sh_scores = np.empty(NSHUF)
        for s in range(NSHUF):
            perm = rng.permutation(C)
            post_sh = posterior_from_counts(N, logF[perm], sumF)
            sh_scores[s] = weighted_corr(post_sh, tbins - tbins.mean(), centers)
        p = (np.sum(np.abs(sh_scores) >= np.abs(score)) + 1) / (NSHUF + 1)
        results.append(dict(t0=t0, t1=t1, peak=(t0 + t1) / 2, score=score, p=p,
                            n_bins=len(tbins), n_cells=active_cells,
                            n_spikes=int(N.sum())))
    return results

print('POST events:', is_post.sum(), ' PRE events:', is_pre.sum())
res_post = analyze_events(is_post, 'POST')
res_pre = analyze_events(is_pre, 'PRE')
print(f'qualified events: POST {len(res_post)}, PRE {len(res_pre)}')

sig_post = np.array([r['p'] < 0.05 for r in res_post])
sig_pre = np.array([r['p'] < 0.05 for r in res_pre])
scores_post = np.array([r['score'] for r in res_post])
scores_pre = np.array([r['score'] for r in res_pre])
print(f'POST significant: {sig_post.sum()}/{len(res_post)} = {sig_post.mean()*100:.1f}%  '
      f'(fwd {np.sum(sig_post & (scores_post > 0))}, rev {np.sum(sig_post & (scores_post < 0))})')
print(f'PRE  significant: {sig_pre.sum()}/{len(res_pre)} = {sig_pre.mean()*100:.1f}%')

np.savez('replay.npz',
         post_scores=scores_post, post_p=np.array([r['p'] for r in res_post]),
         post_peak=np.array([r['peak'] for r in res_post]),
         post_t0=np.array([r['t0'] for r in res_post]),
         post_t1=np.array([r['t1'] for r in res_post]),
         pre_scores=scores_pre, pre_p=np.array([r['p'] for r in res_pre]),
         pre_peak=np.array([r['peak'] for r in res_pre]),
         pre_t0=np.array([r['t0'] for r in res_pre]),
         pre_t1=np.array([r['t1'] for r in res_pre]))

# ---------------- fig05: example replay events ----------------
# pick top-4 POST events by |score| among significant
order = np.argsort(-np.abs(scores_post))
shown = 0
fig, axes = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw=dict(hspace=0.6, wspace=0.35))
for idx in order:
    r = res_post[idx]
    if r['p'] >= 0.01:
        continue
    N, tbins = decode_event(r['t0'], r['t1'])
    post = posterior_from_counts(N, logF, sumF)
    axr = axes[0, shown]
    axp = axes[1, shown]
    # raster sorted by place-field peak location
    peak_pos = centers[np.argmax(F, axis=1)]
    cell_order = np.argsort(peak_pos)
    for rank, c in enumerate(cell_order):
        sp = spikes[c]
        sp = sp[(sp >= r['t0']) & (sp < r['t1'])]
        axr.plot((sp - r['t0']) * 1000, np.full_like(sp, rank), '|', color='k', ms=3)
    axr.set_title(f"event @{r['t0']:.1f}s  score={r['score']:.2f} p={r['p']:.3f}", fontsize=9)
    axr.set_xlabel('time (ms)')
    axr.set_ylabel('cell (sorted by field)')
    axr.set_xlim(0, (r['t1'] - r['t0']) * 1000)
    # posterior
    axp.imshow(post.T, aspect='auto', origin='lower', cmap='hot_r', vmin=0, vmax=0.3,
               extent=[0, (r['t1'] - r['t0']) * 1000, centers[0], centers[-1]])
    # weighted least-squares trajectory line through the posterior
    tt = (tbins - r['t0']) * 1000
    Tn = post.shape[0]
    w = post.ravel()
    ttall = np.repeat(tt, P)
    xxall = np.tile(centers, Tn)
    mt = (w * ttall).sum() / w.sum()
    mx = (w * xxall).sum() / w.sum()
    b = (w * (ttall - mt) * (xxall - mx)).sum() / (w * (ttall - mt) ** 2).sum()
    a = mx - b * mt
    axp.plot([tt[0], tt[-1]], [a + b * tt[0], a + b * tt[-1]], color='cyan', lw=1.5)
    axp.set_xlabel('time (ms)')
    axp.set_ylabel('position (m)')
    shown += 1
    if shown == 4:
        break
fig.suptitle('Example POST-sleep replay events (raster + decoded posterior)', y=0.98)
fig.savefig('fig05_replay_examples.png', dpi=150)
print('saved fig05_replay_examples.png, shown:', shown)

# ---------------- fig06: replay statistics ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw=dict(wspace=0.3))

ax = axes[0]
bins = np.linspace(-0.35, 0.35, 41)
ax.hist(scores_pre, bins=bins, density=True, color='tab:gray', alpha=0.6, label='PRE')
ax.hist(scores_post, bins=bins, density=True, color='tab:blue', alpha=0.7, label='POST')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('weighted correlation (time vs position)')
ax.set_ylabel('density')
ax.set_title('Replay score distribution')
ax.legend()

ax = axes[1]
from scipy import stats as sstats
binom_p = sstats.binomtest(int(sig_post.sum()), len(res_post), 0.05, alternative='greater').pvalue
fisher_p = sstats.fisher_exact([[int(sig_post.sum()), int((~sig_post).sum())],
                                [int(sig_pre.sum()), int((~sig_pre).sum())]],
                               alternative='greater').pvalue
frac = [sig_pre.mean() * 100, sig_post.mean() * 100]
bars = ax.bar(['PRE sleep', 'POST sleep'], frac, color=['tab:gray', 'tab:blue'])
ax.axhline(5, color='k', ls='--', lw=0.8, label='chance (5%)')
for b, f_, n in zip(bars, frac, [len(res_pre), len(res_post)]):
    ax.text(b.get_x() + b.get_width() / 2, f_ + 0.3, f'{f_:.1f}%\n(n={n})',
            ha='center', fontsize=9)
ax.text(1, frac[1] * 0.5, f'vs chance\np={binom_p:.1e}\nvs PRE\np={fisher_p:.1e}',
        ha='center', fontsize=8, color='white')
nfwd = int(np.sum(sig_post & (scores_post > 0)))
nrev = int(np.sum(sig_post & (scores_post < 0)))
ax.text(0.02, 0.95, f'POST: {nfwd} forward, {nrev} reverse', transform=ax.transAxes,
        fontsize=9, va='top')
ax.set_ylabel('% events significant (p<0.05)')
ax.set_title('Significant replay fraction')
ax.legend(loc='upper right')
ax.set_ylim(0, max(frac) * 1.35)

ax = axes[2]
# |score| vs shuffle percentile for POST: empirical CDF comparison
abs_post = np.sort(np.abs(scores_post))
ax.plot(abs_post, np.arange(1, len(abs_post) + 1) / len(abs_post),
        color='tab:blue', label='POST |score|')
abs_pre = np.sort(np.abs(scores_pre))
ax.plot(abs_pre, np.arange(1, len(abs_pre) + 1) / len(abs_pre),
        color='tab:gray', label='PRE |score|')
ax.set_xlabel('|weighted correlation|')
ax.set_ylabel('CDF')
ax.set_title('Score magnitude, PRE vs POST')
ax.legend()

fig.savefig('fig06_replay_stats.png', dpi=150)
print('saved fig06_replay_stats.png')

from scipy import stats as sstats
u = sstats.mannwhitneyu(np.abs(scores_post), np.abs(scores_pre), alternative='greater')
print('MW |score| POST > PRE: p =', u.pvalue)
