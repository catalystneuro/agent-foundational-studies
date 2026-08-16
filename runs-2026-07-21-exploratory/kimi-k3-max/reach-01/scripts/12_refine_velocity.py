import numpy as np, pickle
from scipy import stats

with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
tr = C['trials']
res = D['res']
sig = res['anova_p_move'] < 0.01

print("peak_cc quantiles:", np.round(np.quantile(V['peak_cc'], [0.1,0.25,0.5,0.75,0.9]),3))
edge = np.abs(V['peak_lag']) >= 0.4
print("frac at edge:", edge.mean().round(3), " frac at edge | peak_cc>0.05:", edge[V['peak_cc']>0.05].mean().round(3))

# --- velocity-direction PD using only early reach bins [move_onset, move_onset+0.25] ---
bin_times = V['bin_times']; vx, vy, spd = V['vx'], V['vy'], V['spd']
rates = V['counts']/V['BIN']
mo_of_trial = tr['move_onset']
trel = bin_times - mo_of_trial[V['trial_of_bin']]
early = (trel >= 0) & (trel <= 0.25) & (spd > 10)
vdir = (np.degrees(np.arctan2(vy, vx)) + 360) % 360
vbin = np.floor(((vdir + 15) % 360)/30).astype(int)
bc = np.arange(0, 360, 30)
n_units = rates.shape[1]
occ_v = np.array([b for b in range(12) if ((vbin==b)&early).sum() >= 30])
vt = np.full((12, n_units), np.nan)
for b in occ_v:
    m = (vbin == b) & early
    vt[b] = rates[m].mean(0)
th = np.deg2rad(bc[occ_v])
z = np.nansum(vt[occ_v]*np.exp(1j*th)[:,None], axis=0)
pd_vel_early = (np.degrees(np.angle(z))+360)%360
R_vel_early = np.abs(z)/np.nansum(vt[occ_v], axis=0)
ok = sig & (R_vel_early > 0.1)
d_pd = ((res['pd_move'][ok] - pd_vel_early[ok] + 180) % 360 - 180)
print("early-reach velocity PD: n=", ok.sum(), " median |dPD| =", round(np.median(np.abs(d_pd)),1))

V['pd_vel_early'] = pd_vel_early
V['R_vel_early'] = R_vel_early
with open('data/velocity_stats.pkl','wb') as f:
    pickle.dump(V, f)
