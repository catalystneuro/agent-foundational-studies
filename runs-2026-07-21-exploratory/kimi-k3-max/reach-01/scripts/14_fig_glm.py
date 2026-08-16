import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

with open('data/glm_stats.pkl','rb') as f:
    G = pickle.load(f)
with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
un, tr = C['units'], C['trials']
res = D['res']; sig = res['anova_p_move'] < 0.01
mean_ll = G['mean_ll']; mean_ll_move = G['mean_ll_move']

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35)

# A: example unit filters
ex = int(np.nanargmax(res['depth_move'] * np.sqrt(res['mean_rate_move']) * res['cos_r2_move']))
ax = fig.add_subplot(gs[0, 0])
ax.plot(G['lag_axis'], G['fx'][:, ex], 'o-', color='tab:blue', ms=4, label='vx filter')
ax.plot(G['lag_axis'], G['fy'][:, ex], 'o-', color='tab:orange', ms=4, label='vy filter')
fn = np.hypot(G['fx'][:, ex], G['fy'][:, ex])
ax.plot(G['lag_axis'], fn, 'k--', lw=1, label='norm')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('lag (ms; + = neural leads)'); ax.set_ylabel('filter weight')
ax.legend(fontsize=8)
ax.set_title(f"A  Velocity filters, unit {un['unit_ids'][ex]}\n(GLM PD={G['pd_glm'][ex]:.0f}°)", loc='left', fontsize=10)

# B: predicted vs actual rate for example unit on held-out trials
ax = fig.add_subplot(gs[0, 1:3])
# refit single-unit GLM quickly on 4/5 of trials, predict rest
import jax; jax.config.update('jax_enable_x64', True)
import nemos as nmo
counts, BIN = V['counts'], V['BIN']
bin_times, trial_of_bin = V['bin_times'], V['trial_of_bin']
vx, vy, spd = V['vx'], V['vy'], V['spd']
SHIFT, WIN, NB = 5, 10, 4
def shift_sig(x, k):
    y = np.empty_like(x); y[:-k] = x[k:]; y[-k:] = x[-k:]; return y
basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=NB, window_size=WIN)
X_vel = np.concatenate([np.asarray(basis.compute_features(shift_sig(vx, SHIFT))),
                        np.asarray(basis.compute_features(shift_sig(vy, SHIFT))),
                        np.asarray(basis.compute_features(shift_sig(spd, SHIFT)))], axis=1)
_, widx = np.unique(trial_of_bin, return_index=True)
within = np.arange(len(bin_times)) - widx[trial_of_bin]
tlen = np.bincount(trial_of_bin)[trial_of_bin]
valid = (within >= WIN) & (within < tlen - SHIFT) & ~np.isnan(X_vel).any(1)
trials_all = np.unique(trial_of_bin)
te_tr = trials_all[trials_all % 5 == 0]
te = valid & np.isin(trial_of_bin, te_tr)
trn = valid & ~np.isin(trial_of_bin, te_tr)
g1 = nmo.glm.GLM(regularizer='Ridge', regularizer_strength=0.01, solver_name='LBFGS',
                 solver_kwargs={'tol':1e-10, 'maxiter':1000})
g1.fit(X_vel[trn], counts[trn, ex])
mu = np.asarray(g1.predict(X_vel[te]))/BIN
t_te = bin_times[te]
ra = gaussian_filter1d(V['counts'][te, ex]/BIN, sigma=2)
rp = gaussian_filter1d(mu, sigma=2)
seg = (t_te > t_te[0]+40) & (t_te < t_te[0]+58)
ax.plot(t_te[seg], ra[seg], 'k', lw=1.3, label='actual (smoothed)')
ax.plot(t_te[seg], rp[seg], 'tab:red', lw=1.3, label='GLM prediction')
ax.set_xlabel('time (s)'); ax.set_ylabel('firing rate (sp/s)')
ax.legend(fontsize=9)
r2 = np.corrcoef(ra, rp)[0,1]**2
ax.set_title(f"B  Encoding quality, unit {un['unit_ids'][ex]}, held-out trials (R²={r2:.2f})", loc='left', fontsize=10)

# C: model comparison, all bins
ax = fig.add_subplot(gs[0, 3])
ax.scatter(mean_ll['speed'], mean_ll['velocity'], s=12, alpha=0.6,
           c=['tab:red' if g=='M1' else 'tab:blue' for g in un['group']])
lims = [min(mean_ll['speed'].min(), mean_ll['velocity'].min()),
        max(mean_ll['speed'].max(), mean_ll['velocity'].max())]
ax.plot(lims, lims, 'k--', lw=0.8)
beat = mean_ll['velocity'] > mean_ll['speed']
ax.set_xlabel('test LL, speed-only'); ax.set_ylabel('test LL, velocity')
ax.set_title(f'C  All bins: velocity vs speed\n({beat.sum()}/{len(beat)} above diagonal)', loc='left', fontsize=10)

# D: model comparison, movement bins
ax = fig.add_subplot(gs[1, 0])
ax.scatter(mean_ll_move['speed'], mean_ll_move['velocity'], s=12, alpha=0.6,
           c=['tab:red' if g=='M1' else 'tab:blue' for g in un['group']])
lims = [min(mean_ll_move['speed'].min(), mean_ll_move['velocity'].min()),
        max(mean_ll_move['speed'].max(), mean_ll_move['velocity'].max())]
ax.plot(lims, lims, 'k--', lw=0.8)
beatm = mean_ll_move['velocity'] > mean_ll_move['speed']
ax.set_xlabel('test LL, speed-only'); ax.set_ylabel('test LL, velocity')
ax.set_title(f'D  Movement bins: velocity vs speed\n({beatm.sum()}/{len(beatm)} above diagonal)', loc='left', fontsize=10)

# E: what each signal adds beyond the other (movement bins)
ax = fig.add_subplot(gs[1, 1])
d_dir = mean_ll_move['velocity+speed'] - mean_ll_move['speed']      # direction beyond speed
d_spd = mean_ll_move['velocity+speed'] - mean_ll_move['velocity']   # speed beyond direction
ax.hist(d_dir, bins=40, color='tab:red', alpha=0.65,
        label=f'direction beyond speed ({(d_dir>0).sum()}/182)')
ax.hist(d_spd, bins=40, color='tab:blue', alpha=0.65,
        label=f'speed beyond direction ({(d_spd>0).sum()}/182)')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Δ test LL (movement bins)'); ax.set_ylabel('units')
ax.legend(fontsize=8)
from scipy import stats as sst
p_dir = sst.wilcoxon(d_dir).pvalue; p_spd = sst.wilcoxon(d_spd).pvalue
ax.set_title(f'E  Independent direction & speed info\n(Wilcoxon p={p_dir:.0e} / {p_spd:.0e})', loc='left', fontsize=9)

# F: encoding quality across the population (held-out movement bins)
ax = fig.add_subplot(gs[1, 2])
r_enc = G['r_enc_move']
ax.hist(r_enc[sig], bins=30, color='tab:purple', alpha=0.8, label=f'direction-tuned (n={sig.sum()})')
ax.hist(r_enc[~sig], bins=30, color='gray', alpha=0.55, label=f'not tuned (n={(~sig).sum()})')
ax.axvline(0, color='k', lw=0.8)
ax.axvline(np.nanmedian(r_enc), color='tab:red', ls='--', lw=1.2,
           label=f'median {np.nanmedian(r_enc):.2f}')
ax.set_xlabel('predicted vs actual rate correlation (r)'); ax.set_ylabel('units')
ax.legend(fontsize=8)
ax.set_title('F  Encoding quality, velocity+speed model\n(held-out movement bins)', loc='left', fontsize=9)

# G: GLM PD vs tuning-curve PD
ax = fig.add_subplot(gs[1, 3])
fnorm = np.hypot(G['fx'], G['fy'])
okg = sig & (fnorm.max(0) > np.percentile(fnorm.max(0), 25))
d_glm = (res['pd_move'][okg] - G['pd_glm'][okg] + 180) % 360 - 180
ax.scatter(res['pd_move'][okg], G['pd_glm'][okg], s=12, alpha=0.6, c='tab:green')
ax.plot([0,360],[0,360],'k--',lw=0.8)
ax.set_xlabel('PD from tuning curve (deg)'); ax.set_ylabel('PD from GLM filters (deg)')
ax.set_xlim(0,360); ax.set_ylim(0,360)
ax.set_title(f'G  GLM vs tuning-curve PD (n={okg.sum()}, med |Δ|={np.median(np.abs(d_glm)):.0f}°)', loc='left', fontsize=10)

fig.suptitle('fig07  Poisson GLM encoding of hand velocity (nemos PopulationGLM, 5-fold blocked CV)', fontsize=13)
fig.savefig('figures/fig07_glm_encoding.png', dpi=150, bbox_inches='tight')
print("fig07 saved")
print("all bins: vel>speed", beat.sum(), "/", len(beat))
print("movement bins: vel>speed", beatm.sum(), "/", len(beatm))
