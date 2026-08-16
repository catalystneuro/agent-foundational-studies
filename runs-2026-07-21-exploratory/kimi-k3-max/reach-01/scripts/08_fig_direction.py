import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
tr, un = C['trials'], C['units']
res, bm, bc, occ = D['res'], D['bin_means'], D['bin_centers'], D['occupied']
bin_id = D['bin_id']
cmap = plt.cm.hsv
bincolors = {b: cmap(bc[b]/360) for b in occ}

# ---------- fig03: example unit raster+PETH by direction ----------
score = res['depth_move'] * np.sqrt(res['mean_rate_move']) * res['cos_r2_move']
ex = int(np.nanargmax(score))
spk = un['spike_times'][ex]
mo = tr['move_onset']
T0, T1 = -0.3, 0.6
bin_s = 0.01
edges = np.arange(T0, T1+bin_s, bin_s)
centers_t = (edges[:-1]+edges[1:])/2

fig = plt.figure(figsize=(15, 9))
order = np.argsort(bc[occ])
for k, b in enumerate(occ[order]):
    ax = fig.add_subplot(3, 4, k+1)
    trials_b = np.where(bin_id == b)[0]
    # raster
    for j, ti in enumerate(trials_b):
        s = spk[(spk >= mo[ti]+T0) & (spk < mo[ti]+T1)] - mo[ti]
        ax.vlines(s, j+0.5, j+1.5, color=bincolors[b], lw=0.4)
    # peth
    counts = np.zeros(len(edges)-1)
    for ti in trials_b:
        s = spk[(spk >= mo[ti]+T0) & (spk < mo[ti]+T1)] - mo[ti]
        counts += np.histogram(s, edges)[0]
    peth = gaussian_filter1d(counts/len(trials_b)/bin_s, sigma=2)
    ax2 = ax.twinx()
    ax2.plot(centers_t, peth, 'k', lw=1.5)
    ax2.set_ylim(0, np.nanmax(peth)*1.3+1)
    ax2.set_yticks([])
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlim(T0, T1)
    ax.set_ylim(0.5, len(trials_b)+0.5)
    ax.set_yticks([])
    if k >= 8 - 2: ax.set_xlabel('time from move onset (s)', fontsize=9)
    ax.set_title(f"{bc[b]}°  (n={len(trials_b)})", fontsize=10, color=bincolors[b])
fig.suptitle(f"fig03  Unit {un['unit_ids'][ex]} ({un['group'][ex]}): rasters & PETHs by reach direction", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig('figures/fig03_direction_rasters.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("fig03 saved, example unit", un['unit_ids'][ex], un['group'][ex],
      "depth", round(res['depth_move'][ex],1), "PD", round(res['pd_move'][ex],1))

# ---------- fig04: polar tuning curves, top 12 units ----------
top = np.argsort(np.nan_to_num(score))[::-1][:12]
fig = plt.figure(figsize=(13, 10))
for k, u in enumerate(top):
    ax = fig.add_subplot(3, 4, k+1, projection='polar')
    vals = bm[u][occ]
    th = np.deg2rad(bc[occ])
    # close the curve
    th_c = np.concatenate([th, th[:1]]); v_c = np.concatenate([vals, vals[:1]])
    ax.plot(th_c, v_c, 'o-', color='tab:blue', lw=1.5, ms=4)
    pd = res['pd_move'][u]
    ax.plot([np.deg2rad(pd), np.deg2rad(pd)], [0, vals.max()*1.1], color='tab:red', lw=1.5, ls='--')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_title(f"u{un['unit_ids'][u]} {un['group'][u]}\nPD={pd:.0f}°, depth={res['depth_move'][u]:.0f} sp/s", fontsize=9, pad=22)
    ax.set_rlabel_position(0)
    ax.tick_params(labelsize=7)
fig.suptitle("fig04  Movement-period direction tuning (polar): top 12 units by modulation", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('figures/fig04_tuning_curves_examples.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("fig04 saved")
