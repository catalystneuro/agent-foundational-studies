"""Velocity/speed tuning: binned rates vs hand kinematics within trials."""
import numpy as np, pickle
from scipy import stats
from tqdm import tqdm

with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
tr, un, bh = C['trials'], C['units'], C['behav']

BIN = 0.05  # 50 ms
t_beh = bh['hand_vel_t']; vel = bh['hand_vel'].astype(np.float64)/10.0  # cm/s
speed = np.hypot(vel[:,0], vel[:,1])

# ---- build per-trial binned data ----
bin_times_list, bin_vx, bin_vy, bin_spd = [], [], [], []
trial_idx_list = []
for i in range(len(tr['start'])):
    edges = np.arange(tr['start'][i], tr['stop'][i], BIN)
    if len(edges) < 2: continue
    ct = (edges[:-1]+edges[1:])/2
    bi = np.searchsorted(t_beh, ct)
    bi = np.clip(bi, 0, len(t_beh)-1)
    bin_times_list.append(ct)
    bin_vx.append(vel[bi,0]); bin_vy.append(vel[bi,1]); bin_spd.append(speed[bi])
    trial_idx_list.append(np.full(len(ct), i))
bin_times = np.concatenate(bin_times_list)
vx = np.concatenate(bin_vx); vy = np.concatenate(bin_vy); spd = np.concatenate(bin_spd)
trial_of_bin = np.concatenate(trial_idx_list)
n_bins = len(bin_times)
print("bins:", n_bins, "bin dur:", BIN)

# spike counts per unit per bin
n_units = len(un['spike_times'])
counts = np.zeros((n_bins, n_units), dtype=np.float32)
left_edges = bin_times - BIN/2  # globally sorted; bins tile trials, gaps between trials
for u in range(n_units):
    spk = un['spike_times'][u]
    bi = np.searchsorted(left_edges, spk, side='right') - 1
    ok = (bi >= 0) & (bi < n_bins) & (spk < left_edges[np.clip(bi,0,n_bins-1)] + BIN - 1e-9)
    np.add.at(counts[:, u], bi[ok], 1)
rates = counts / BIN  # Hz
print("mean rate (Hz):", rates.mean().round(2), " total spikes placed:", int(counts.sum()))

# ---- 1. speed tuning ----
spd_edges = np.quantile(spd, np.linspace(0, 1, 21))
spd_centers = (spd_edges[:-1]+spd_edges[1:])/2
spd_bin = np.clip(np.searchsorted(spd_edges, spd)-1, 0, 19)
speed_tuning = np.zeros((20, n_units))
for b in range(20):
    m = spd_bin == b
    speed_tuning[b] = rates[m].mean(0)
rho_speed = np.array([stats.spearmanr(spd, rates[:,u]).statistic for u in range(n_units)])

# ---- 2. velocity-direction tuning (speed > 15 cm/s) ----
fast = spd > 15
vdir = (np.degrees(np.arctan2(vy, vx)) + 360) % 360
vbin = np.floor(((vdir + 15) % 360)/30).astype(int)
occ_v = np.array([b for b in range(12) if ((vbin==b)&fast).sum() >= 30])
print("occupied velocity-dir bins:", occ_v, "fast frac:", fast.mean().round(3))
vdir_tuning = np.full((12, n_units), np.nan)
for b in occ_v:
    m = (vbin == b) & fast
    vdir_tuning[b] = rates[m].mean(0)
bc = np.arange(0, 360, 30)
th = np.deg2rad(bc[occ_v])
z = np.nansum(vdir_tuning[occ_v] * np.exp(1j*th)[:,None], axis=0)
pd_vel = (np.degrees(np.angle(z)) + 360) % 360
R_vel = np.abs(z) / np.nansum(vdir_tuning[occ_v], axis=0)

# ---- 3. lag analysis: corr(rate[t], speed[t+lag]) per unit ----
lags = np.arange(-0.4, 0.401, BIN)  # s; positive = neural leads
lag_bins = (lags/BIN).round().astype(int)
# work within trials to avoid cross-trial correlations
cc = np.zeros((len(lags), n_units))
# z-score speed and rates within each trial, then average product across matching bins
spd_z = np.zeros_like(spd)
rate_z = np.zeros_like(rates)
for i in np.unique(trial_of_bin):
    m = trial_of_bin == i
    s = spd[m]; spd_z[m] = (s - s.mean())/ (s.std()+1e-9)
    r = rates[m]; rate_z[m] = (r - r.mean(0)) / (r.std(0)+1e-9)
for li, lb in enumerate(lag_bins):
    prod_sum = np.zeros(n_units); n_sum = 0
    # shift within trials
    for i in np.unique(trial_of_bin):
        m = np.where(trial_of_bin == i)[0]
        if lb >= 0:
            a, b = m[:-lb] if lb>0 else m, m[lb:] if lb>0 else m
        else:
            a, b = m[-lb:], m[:lb]
        if len(a) == 0: continue
        prod_sum += (rate_z[a] * spd_z[b][:,None]).sum(0)
        n_sum += len(a)
    cc[li] = prod_sum / n_sum
peak_lag = lags[np.argmax(cc, axis=0)]
peak_cc = cc.max(axis=0)

# ---- 4. PD-aligned speed tuning: rate vs speed for moves toward/away from unit PD ----
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
pd_move = D['res']['pd_move']
# angular distance between instantaneous velocity direction and unit PD
dtheta = np.abs(((vdir[None,:] - pd_move[:,None]) + 180) % 360 - 180)  # (units, bins)
toward = (dtheta <= 60) & fast[None,:]
away = (dtheta >= 120) & fast[None,:]
spd_q = np.quantile(spd[fast], np.linspace(0,1,11))
spd_qc = (spd_q[:-1]+spd_q[1:])/2
sq_bin = np.clip(np.searchsorted(spd_q, spd)-1, 0, 9)
rate_toward = np.full((10, n_units), np.nan); rate_away = np.full((10, n_units), np.nan)
for b in range(10):
    mb = sq_bin == b
    rate_toward[b] = np.array([rates[toward[u] & mb, u].mean() if (toward[u]&mb).sum()>5 else np.nan for u in range(n_units)])
    rate_away[b]   = np.array([rates[away[u] & mb, u].mean() if (away[u]&mb).sum()>5 else np.nan for u in range(n_units)])
# per-unit speed slope toward PD (linear regression on bin means)
slope_toward = np.array([stats.linregress(spd_qc[~np.isnan(rate_toward[:,u])], rate_toward[~np.isnan(rate_toward[:,u]),u]).slope
                         if (~np.isnan(rate_toward[:,u])).sum()>4 else np.nan for u in range(n_units)])
slope_away = np.array([stats.linregress(spd_qc[~np.isnan(rate_away[:,u])], rate_away[~np.isnan(rate_away[:,u]),u]).slope
                       if (~np.isnan(rate_away[:,u])).sum()>4 else np.nan for u in range(n_units)])

with open('data/velocity_stats.pkl','wb') as f:
    pickle.dump(dict(spd_centers=spd_centers, speed_tuning=speed_tuning, rho_speed=rho_speed,
                     vdir_tuning=vdir_tuning, occ_v=occ_v, pd_vel=pd_vel, R_vel=R_vel,
                     lags=lags, cc=cc, peak_lag=peak_lag, peak_cc=peak_cc,
                     bin_times=bin_times, vx=vx, vy=vy, spd=spd, counts=counts,
                     trial_of_bin=trial_of_bin, BIN=BIN, spd_qc=spd_qc, rate_toward=rate_toward,
                     rate_away=rate_away, slope_toward=slope_toward, slope_away=slope_away), f)
print("saved velocity_stats.pkl")
print("median speed-rate Spearman rho:", np.median(rho_speed).round(3))
print("frac units rho>0:", (rho_speed>0).mean().round(2))
print("median peak lag (s):", np.median(peak_lag).round(3))
print("median speed slope toward PD (Hz per cm/s):", np.nanmedian(slope_toward).round(4))
print("median speed slope away from PD:", np.nanmedian(slope_away).round(4))
