"""Direction tuning: per-direction rates in movement & delay windows, PD, depth, ANOVA, cosine fit."""
import numpy as np, pickle
from scipy import stats

with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
tr, un = C['trials'], C['units']

ang = tr['target_angle']                      # 0..360
# 30-degree bins centered at 0,30,...,330
bin_centers = np.arange(0, 360, 30)
bin_id = np.floor(((ang + 15) % 360) / 30).astype(int)
occupied = np.array([b for b in range(12) if (bin_id == b).sum() >= 20])
print("occupied bins (center deg):", bin_centers[occupied], "counts:", [(bin_id==b).sum() for b in occupied])

move_onset = tr['move_onset']; go_cue = tr['go_cue']; delay = tr['delay']
WIN_MOVE = (-0.05, 0.40)     # relative to move onset
WIN_DELAY = (-0.30, 0.0)     # relative to go cue (planning), only trials with delay>=350ms
delay_ok = delay >= 350

def window_rates(spk, ev, win):
    """rate per event in window (Hz)"""
    lo, hi = win
    dur = hi - lo
    idx = np.searchsorted(spk, ev + lo)
    idx2 = np.searchsorted(spk, ev + hi)
    return (idx2 - idx) / dur

def circ_pd(angles_deg, rates):
    th = np.deg2rad(angles_deg)
    z = np.sum(rates * np.exp(1j*th))
    return (np.rad2deg(np.angle(z)) % 360), np.abs(z)/np.sum(rates)

def cosine_fit(angles_deg, rates):
    th = np.deg2rad(angles_deg)
    X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    pred = X @ beta
    ss_res = np.sum((rates-pred)**2); ss_tot = np.sum((rates-rates.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    pd = np.rad2deg(np.arctan2(beta[2], beta[1])) % 360
    amp = np.hypot(beta[1], beta[2])
    return beta[0], amp, pd, r2

n_units = len(un['spike_times'])
res = {k: np.full(n_units, np.nan) for k in
       ['pd_move','R_move','depth_move','anova_p_move','cos_r2_move','cos_pd_move',
        'pd_delay','R_delay','depth_delay','anova_p_delay','mean_rate_move','mean_rate_delay']}
bin_means_all = np.full((n_units, 12), np.nan)

for u in range(n_units):
    spk = un['spike_times'][u]
    r_move = window_rates(spk, move_onset, WIN_MOVE)
    r_delay = window_rates(spk, go_cue, WIN_DELAY)
    res['mean_rate_move'][u] = r_move.mean()
    res['mean_rate_delay'][u] = r_delay[delay_ok].mean()
    for tag, r, mask in [('move', r_move, np.ones(len(r_move), bool)), ('delay', r_delay, delay_ok)]:
        bm = np.full(12, np.nan)
        for b in occupied:
            m = (bin_id == b) & mask
            bm[b] = r[m].mean()
        if tag == 'move':
            bin_means_all[u] = bm
        vals = bm[occupied]
        if np.all(vals == 0) or np.isnan(vals).any():
            continue
        pd, R = circ_pd(bin_centers[occupied], vals)
        res[f'pd_{tag}'][u] = pd
        res[f'R_{tag}'][u] = R
        res[f'depth_{tag}'][u] = vals.max() - vals.min()
        groups = [r[(bin_id == b) & mask] for b in occupied]
        F, p = stats.f_oneway(*groups)
        res[f'anova_p_{tag}'][u] = p
        if tag == 'move':
            b0, amp, cpd, r2 = cosine_fit(bin_centers[occupied], vals)
            res['cos_r2_move'][u] = r2
            res['cos_pd_move'][u] = cpd

sig_move = res['anova_p_move'] < 0.01
sig_delay = res['anova_p_delay'] < 0.01
print(f"\nunits direction-tuned (move, ANOVA p<0.01): {sig_move.sum()}/{n_units}")
print(f"units direction-tuned (delay): {np.nansum(sig_delay)}/{int(delay_ok.sum())} trials used, {np.sum(~np.isnan(res['anova_p_delay']))} units")
for grp in ['M1','PMd']:
    m = un['group'] == grp
    print(f"  {grp}: tuned move {np.sum(sig_move & m)}/{m.sum()}, delay {np.nansum(sig_delay & m)}/{np.sum(m & ~np.isnan(res['anova_p_delay']))}")

out = dict(res=res, bin_means=bin_means_all, bin_centers=bin_centers, occupied=occupied,
           bin_id=bin_id, group=un['group'], heldout=un['heldout'])
with open('data/direction_stats.pkl','wb') as f:
    pickle.dump(out, f)
print("saved data/direction_stats.pkl")
