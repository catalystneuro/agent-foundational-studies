import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
un = C['units']
res = D['res']
sig = res['anova_p_move'] < 0.01

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# (a) example units: rate vs speed (toward PD / away from PD)
ax = fig.add_subplot(gs[0, 0])
depth = res['depth_move']
ex_units = np.argsort(np.nan_to_num(depth))[::-1][:6]
for u in ex_units:
    ax.plot(V['spd_qc'], V['rate_toward'][:,u], '-', lw=1.5, alpha=0.8, label=f'u{un["unit_ids"][u]}')
ax.set_xlabel('hand speed (cm/s)'); ax.set_ylabel('firing rate (sp/s)')
ax.legend(fontsize=7)
ax.set_title('A  Rate vs speed, moving toward PD', loc='left', fontsize=11)

# (b) population: mean normalized rate vs speed toward/away
ax = fig.add_subplot(gs[0, 1])
rt = V['rate_toward']; ra = V['rate_away']
norm = np.nanmean(np.concatenate([rt, ra], axis=0), axis=0, keepdims=True)
rtn = rt/norm; ran = ra/norm
ax.errorbar(V['spd_qc'], np.nanmean(rtn,1), yerr=np.nanstd(rtn,1)/np.sqrt(rtn.shape[1]), color='tab:red', lw=1.5, label='toward PD (±60°)')
ax.errorbar(V['spd_qc'], np.nanmean(ran,1), yerr=np.nanstd(ran,1)/np.sqrt(ran.shape[1]), color='tab:blue', lw=1.5, label='away from PD (>120°)')
ax.set_xlabel('hand speed (cm/s)'); ax.set_ylabel('normalized rate')
ax.legend(fontsize=9)
ax.set_title('B  Population speed modulation by direction', loc='left', fontsize=11)

# (c) slope distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(V['slope_toward'][sig]*10, bins=40, alpha=0.7, color='tab:red', label='toward PD')
ax.hist(V['slope_away'][sig]*10, bins=40, alpha=0.7, color='tab:blue', label='away from PD')
ax.axvline(0, color='k', lw=0.8)
wt = stats.wilcoxon(V['slope_toward'][sig] - V['slope_away'][sig])
ax.set_xlabel('speed slope (sp/s per 10 cm/s)'); ax.set_ylabel('units')
ax.legend(fontsize=9)
ax.set_title(f'C  Speed slopes (Wilcoxon p={wt.pvalue:.1e})', loc='left', fontsize=11)

# (d) lag cross-correlation: population mean + examples
ax = fig.add_subplot(gs[1, 0])
for u in ex_units[:3]:
    ax.plot(V['lags']*1000, V['cc'][:,u], lw=1, alpha=0.7)
ax.plot(V['lags']*1000, V['cc'][:,sig].mean(1), 'k', lw=2.5, label='population mean')
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('lag (ms; + = neural leads hand speed)'); ax.set_ylabel('rate–speed correlation (z)')
ax.legend(fontsize=8)
ax.set_title('D  Rate–speed cross-correlation vs lag', loc='left', fontsize=11)

# (e) peak lag histogram
ax = fig.add_subplot(gs[1, 1])
coupled = sig & (V['peak_cc'] > 0.05)
ax.hist(V['peak_lag'][coupled]*1000, bins=np.arange(-400,401,50), color='tab:purple', alpha=0.8)
ax.axvline(0, color='k', lw=0.8)
ax.axvline(np.median(V['peak_lag'][coupled])*1000, color='tab:red', ls='--', lw=1.2,
           label=f"median {np.median(V['peak_lag'][coupled])*1000:.0f} ms (n={coupled.sum()})")
ax.set_xlabel('peak lag (ms)'); ax.set_ylabel('units')
ax.legend(fontsize=9)
ax.set_title('E  Distribution of optimal lags', loc='left', fontsize=11)

# (f) PD from instantaneous velocity vs PD from target direction
ax = fig.add_subplot(gs[1, 2])
ok = sig & (V['R_vel_early'] > 0.1)
d_pd = ((res['pd_move'][ok] - V['pd_vel_early'][ok] + 180) % 360 - 180)
ax.scatter(res['pd_move'][ok], V['pd_vel_early'][ok], s=12, alpha=0.6, c='tab:green')
ax.plot([0,360],[0,360],'k--',lw=0.8)
ax.set_xlabel('PD from target direction (deg)'); ax.set_ylabel('PD from hand velocity (deg)')
ax.set_title(f'F  PD: target dir vs early-reach velocity (n={ok.sum()}, med |Δ|={np.median(np.abs(d_pd)):.0f}°)', loc='left', fontsize=10)
ax.set_xlim(0,360); ax.set_ylim(0,360)

fig.suptitle('fig06  Speed and velocity tuning: 182 units, MC_Maze sub-Jenkins', fontsize=13)
fig.savefig('figures/fig06_speed_tuning.png', dpi=150, bbox_inches='tight')
print("fig06 saved")
print("toward>away wilcoxon p:", wt.pvalue)
print("PD consistency: n=", ok.sum(), "median |dPD| =", round(np.median(np.abs(d_pd)),1))
