"""Generate all figures for the orientation-selectivity analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
import scipy.stats as st
from analysis_core import load_nwb, count_in_intervals, ORIS_DEG, THETA, TFS

res = np.load('results.npz', allow_pickle=True)
unit_ids = res['unit_ids']; unit_area = res['unit_area']; quality = res['quality']
gosi = res['gosi']; peak = res['peak_above_blank']
pref_ori = res['pref_ori_deg']; pref_tf = res['pref_tf_deg']
p_value = res['p_value']; blank = res['blank']
means = res['means_response']   # n_units x tf x ori

ALL_ORI_DEG = np.array([0., 45., 90., 135.])

def good_in(area):
    return (unit_area == area) & (quality == 'good')

visp = good_in('VISp')
selective = (p_value < 0.05) & (peak >= 1.0)
print("VISp good:", visp.sum(), "selective:", (selective & visp).sum())

# area list for summaries (visual hierarchy + control CA1)
areas = ['VISp', 'VISl', 'VISam', 'VISpm', 'LGd', 'CA1']

# ============ FIG 1: raster + psth for a strongly tuned VISp unit ============
# stream spike data + trial structure
nwbfile = load_nwb("58703c97-c0a9-4736-b684-73c85c1a444a")
nwb = nap.NWBFile(nwbfile)
units_obj = nwb['units']
dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
starts = dg['start_time'].values; stops = dg['stop_time'].values
ori_raw = dg['orientation'].values.astype(float)
inv = nwbfile.intervals['invalid_times'].to_dataframe()
ov = np.zeros(len(dg), bool)
for _, r in inv.iterrows():
    ov |= (starts < r['stop_time']) & (stops > r['start_time'])
valid = ~ov
vstarts = starts[valid]; vstops = stops[valid]; vori = ori_raw[valid]
nb = ~np.isnan(vori)
vstarts = vstarts[nb]; vori = vori[nb]
vori_fold = np.mod(vori, 180.0)
t0 = vstarts - 0.08

# pick example strong unit among VISp selective with plenty of spikes
sel_ids = unit_ids[selective & visp]
pref_order = np.argsort(-gosi[selective & visp])
ex_ids = unit_ids[(selective & visp)][pref_order]
# find one with enough total spikes
ex_id = ex_ids[0]
ts_ex = units_obj[ex_id].t
if len(ts_ex) < 2000:
    for i in range(1, min(30, len(ex_ids))):
        c = units_obj[ex_ids[i]].t
        if len(c) >= 2000:
            ex_id = ex_ids[i]; ts_ex = c; break

# raster: spikes per trial (0-2 s), grouped by folded orientation
bin_t = np.arange(0, 2.05, 0.01)
psth_arr = np.zeros((4, len(bin_t)-1))
order = np.zeros(4, int)
raster_times = []; raster_pos = []; raster_dirs = []
for k, o in enumerate(ALL_ORI_DEG):
    trials = np.where(vori_fold == o)[0]
    sel = np.argsort(vstarts[trials])
    trials = trials[sel]
    order[k] = trials.size
    for tr in trials:
        sp = ts_ex[(ts_ex >= t0[tr]) & (ts_ex < t0[tr] + 2.0)] - t0[tr]
        raster_times.append(sp)
        raster_pos.append(len(raster_pos))
        raster_dirs.append(k)
        psth_arr[k] += np.histogram(sp, bin_t)[0]
psth_arr /= np.maximum(order[:, None], 1)  # mean spikes/bin -> rate

fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                       gridspec_kw={'height_ratios': [2.2, 1]})
# orientation separators
counts = np.bincount(raster_dirs)
cum = np.cumsum(np.hstack([[0], counts]))
for c in cum[1:-1]:
    ax[0].axhline(c - 0.5, color='red', lw=1, alpha=0.7)
ax[0].eventplot(raster_times, colors='k', linewidths=0.5)
ax[0].set_ylabel('trial  (sorted by orientation)')
ax[0].set_title(f'Raster: VISp good unit {ex_id} during drifting gratings\n'
                f'gOSI={gosi[np.where(unit_ids==ex_id)[0][0]]:.2f}, p={p_value[np.where(unit_ids==ex_id)[0][0]]:.3g}')
# orientation labels
cc = np.cumsum(counts)
mid = (cc - counts) + counts/2
for m, o in zip(mid, ALL_ORI_DEG):
    ax[0].text(2.06, m, f'{int(o)}°', va='center', color='red', fontsize=8)
tmid = 0.5*(bin_t[:-1]+bin_t[1:])
for k in range(4):
    ax[1].plot(tmid, psth_arr[k]* (1000/0.01), label=f'{int(ALL_ORI_DEG[k])}°')
ax[1].set_xlabel('time from onset (s)'); ax[1].set_ylabel('firing rate (Hz)')
ax[1].legend(fontsize=8, ncol=4, loc='upper right')
ax[1].set_title('PSTH per orientation')
ax[0].set_xlim(0, 2.1)
fig.tight_layout()
fig.savefig('figures/fig1_raster_psth.png', dpi=150)
plt.close(fig)
print("fig1 saved")

# ============ FIG 2: example tuning curves (linear + polar) ============
# choose strong, moderate, untuned VISp good units
visp_sel_idx = np.where(selective & visp)[0]
visp_non_idx = np.where((~selective) & visp & (gosi < 0.15) & (peak > 0))[0]
strong_i = visp_sel_idx[np.argsort(-gosi[visp_sel_idx])[0]]
if len(visp_sel_idx) > 10:
    mid_i = visp_sel_idx[np.argsort(gosi[visp_sel_idx])[len(visp_sel_idx)//2]]
untuned_i = visp_non_idx[np.argsort(-peak[visp_non_idx])[0]]  # highest rate untuned
examples = [(strong_i, 'strongly tuned'), (mid_i, 'moderately tuned'), (untuned_i, 'untuned')]

def resp_at_pref(i):
    ptf = np.argmin(np.abs(TFS - pref_tf[i]))
    return means[i, ptf, :] - blank[i]

fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': 'polar'})
for a, (i, lab) in zip(axs, examples):
    r = np.maximum(resp_at_pref(i), 0)
    ang = np.deg2rad(np.hstack([ALL_ORI_DEG, ALL_ORI_DEG[0]]))
    rr = np.hstack([r, r[0]])
    a.plot(ang, rr, 'o-', color='navy')
    a.fill(ang, rr, color='navy', alpha=0.2)
    a.set_xticks(np.deg2rad(ALL_ORI_DEG))
    a.set_xticklabels([f'{int(o)}°' for o in ALL_ORI_DEG])
    a.set_title(f'{lab}\nunit {unit_ids[i]}\ngOSI={gosi[i]:.2f}', fontsize=9)
axs[0].set_ylabel('baseline-subtracted firing rate')
fig.suptitle('Example VISp orientation tuning curves (drifting gratings, preferred TF)')
fig.tight_layout()
fig.savefig('figures/fig2_example_tuning_polar.png', dpi=150)
plt.close(fig)

# linear-only version with stronger modulation contrast
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (i, lab) in zip(axs, examples):
    r = np.maximum(resp_at_pref(i), 0)
    a.plot(ALL_ORI_DEG, r, 'o-', color='navy')
    a.axhline(blank[i], color='gray', ls='--', label=f'blank {blank[i]:.1f} Hz')
    a.set_xlabel('orientation (°)')
    a.set_ylim(0)
    a.set_title(f'{lab} (unit {unit_ids[i]}, gOSI={gosi[i]:.2f})', fontsize=9)
    a.legend(fontsize=7)
axs[0].set_ylabel('firing rate (Hz)')
fig.suptitle('Example VISp orientation tuning curves (preferred temporal frequency)')
fig.tight_layout()
fig.savefig('figures/fig2_example_tuning_linear.png', dpi=150)
plt.close(fig)
print("fig2 saved")

# ============ FIG 3: population gOSI distribution (VISp) ============
gs_visp = gosi[visp]
sel_v = selective[visp]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(gs_visp[~sel_v], bins=np.linspace(0, 1, 41), color='gray', alpha=0.7, label='non-selective')
ax[0].hist(gs_visp[sel_v], bins=np.linspace(0, 1, 41), color='crimson', alpha=0.7, label='selective')
ax[0].set_xlabel('gOSI'); ax[0].set_ylabel('# VISp units'); ax[0].legend()
ax[0].set_title(f'VISp good units: {sel_v.sum()}/{gs_visp.size} orientation-selective ({100*sel_v.mean():.0f}%)')
ax[1].hist(gosi[good_in('CA1')], bins=np.linspace(0, 1, 41),
           color='dimgray', alpha=0.7, label='CA1 (control)')
ax[1].set_xlabel('gOSI'); ax[1].set_ylabel('# units'); ax[1].legend()
ax[1].set_title('CA1 (hippocampus, control) gOSI')
for a in ax:
    a.axvline(np.nanmedian(gs_visp), color='navy', ls='--', lw=1)
fig.tight_layout()
fig.savefig('figures/fig3_population_gosi.png', dpi=150)
plt.close(fig)
print("fig3 saved")

# ============ FIG 4: selectivity by area ============
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
fracs = []; ns = []
for ar in areas:
    m = good_in(ar)
    ns.append(m.sum())
    fracs.append((selective & m).sum() / m.sum() if m.sum() else np.nan)
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b']
axs[0].bar(np.arange(len(areas)), fracs, color=colors)
axs[0].set_xticks(np.arange(len(areas))); axs[0].set_xticklabels(areas)
axs[0].set_ylabel('fraction orientation-selective')
axs[0].set_title('Fraction of good units selective to orientation')
for i, (f, n) in enumerate(zip(fracs, ns)):
    axs[0].text(i, f + 0.01, f'{100*f:.0f}% (n={n})', ha='center', fontsize=8)
axs[0].axhline(0.05, color='gray', ls='--', label='chance (5%)')
axs[0].legend()
# gOSI distribution by area
bp = axs[1].boxplot([gosi[good_in(a)][~np.isnan(gosi[good_in(a)])] for a in areas],
                    labels=areas, patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
axs[1].set_ylabel('gOSI'); axs[1].set_title('gOSI distribution across regions')
fig.tight_layout()
fig.savefig('figures/fig4_area_selectivity.png', dpi=150)
plt.close(fig)
print("fig4 saved")

# ============ FIG 5: preferred-orientation heatmap (selective VISp) ============
sel_idx = np.where(selective & visp)[0]
if sel_idx.size:
    data = means[sel_idx, 0, :]  # placeholder pref-tf per unit
    # baseline-subtracted at each unit's preferred tf
    hm = np.array([means[i, np.argmin(np.abs(TFS - pref_tf[i])), :] - blank[i] for i in sel_idx])
    hmn = hm / np.nanmax(np.abs(hm), axis=1, keepdims=True)
    order_idx = np.argsort(pref_ori[sel_idx] + 0.001 * gosi[sel_idx])
    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(hmn[order_idx], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{int(o)}°' for o in ALL_ORI_DEG])
    ax.set_yticks([]); ax.set_xlabel('orientation (°)'); ax.set_ylabel('unit (sorted by preferred orientation)')
    ax.set_title(f'Preferred-orientation map: {sel_idx.size} selective VISp units')
    cb = fig.colorbar(im, ax=ax); cb.set_label('normalized response')
    fig.tight_layout()
    fig.savefig('figures/fig5_pref_ori_heatmap.png', dpi=150)
    plt.close(fig)
    print("fig5 saved")

# ============ FIG 6: permutation null for the example unit ============
# recompute null gOSI (N=2000) for the strong example unit only
ts_ex = units_obj[ex_id].t
fr_ex = count_in_intervals(ts_ex, t0, t0 + 2.0)
# reproduce per-trial grouping labels (folded orientation, temporal frequency)
ori_idx = np.array([np.argmin(np.abs(ORIS_DEG - o)) for o in vori_fold])
tfs_all = dg['temporal_frequency'].values.astype(float)[valid][nb]
tf_idx = np.array([np.argmin(np.abs(TFS - t)) for t in tfs_all])
rng = np.random.default_rng(1)
N = 2000
nulls = np.zeros(N)
for s in range(N):
    ori_p = ori_idx.copy()
    for t in range(5):
        selm = np.where(tf_idx == t)[0]
        ori_p[selm] = rng.permutation(ori_idx[selm])
    ok = np.zeros((5, 4))
    for tt in range(5):
        for oo in range(4):
            selm = np.where((tf_idx == tt) & (ori_p == oo))[0]
            ok[tt, oo] = fr_ex[selm].mean() if selm.size else np.nan
    preft = np.nanargmax(np.nanmean(ok, axis=1))
    r = ok[preft]
    deno = r.sum(); numo = np.sqrt((np.sum(r*np.cos(2*THETA)))**2 + (np.sum(r*np.sin(2*THETA)))**2)
    nulls[s] = numo/deno if deno > 0 else np.nan
# observed gOSI and mean response at observed pref tf
oki = np.zeros((5, 4))
for tt in range(5):
    for oo in range(4):
        selm = np.where((tf_idx == tt) & (ori_idx == oo))[0]
        oki[tt, oo] = fr_ex[selm].mean() if selm.size else np.nan
preft_o = np.nanargmax(np.nanmean(oki, axis=1))
r_o = oki[preft_o]
deno = r_o.sum(); numo = np.sqrt((np.sum(r_o*np.cos(2*THETA)))**2 + (np.sum(r_o*np.sin(2*THETA)))**2)
gosi_o = numo/deno if deno > 0 else np.nan
p_ex = (np.sum(nulls >= gosi_o) + 1) / (N + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(nulls, bins=60, color='gray', alpha=0.8, label='null gOSI (permitted)')
ax.axvline(gosi_o, color='crimson', lw=2, label=f'observed gOSI={gosi_o:.2f}')
ax.set_xlabel('gOSI'); ax.set_ylabel('count')
ax.set_title(f'Permutation test for VISp unit {ex_id}\np = {p_ex:.3g} (vs 2000 shuffles)')
ax.legend()
fig.tight_layout()
fig.savefig('figures/fig6_permutation_null.png', dpi=150)
plt.close(fig)
print("fig6 saved")

# ============ FIG 7: nemos GLM encoding ============
try:
    nr = np.load('nemos_r2.npz', allow_pickle=True)
    r2 = nr['r2']; beta = nr['beta']; sela = nr['selective']
    gain = np.linalg.norm(beta, axis=1)
    okk = ~np.isnan(r2)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(r2[okk & sela], bins=30, alpha=0.7, color='crimson', label='selective')
    axs[0].hist(r2[okk & ~sela], bins=30, alpha=0.7, color='gray', label='non-selective')
    axs[0].axvline(np.nanmean(r2[okk & sela]), color='crimson', ls='--')
    axs[0].axvline(np.nanmean(r2[okk & ~sela]), color='gray', ls='--')
    axs[0].set_xlabel('pseudo-R² (orientation GLM)'); axs[0].set_ylabel('# VISp units')
    axs[0].set_title('Single-trial orientation encoding (nemos GLM)')
    axs[0].legend()
    gv = gosi[visp]
    axs[1].scatter(gain, gv, s=12, alpha=0.5, color='navy')
    axs[1].set_xlabel('|β| = orientation gain (GLM)'); axs[1].set_ylabel('gOSI')
    axs[1].set_title(f'GLM gain vs gOSI  (r={np.corrcoef(gain[~np.isnan(gain)], gv[~np.isnan(gain)])[0,1]:.2f})')
    fig.tight_layout()
    fig.savefig('figures/fig7_nemos_glm.png', dpi=150)
    plt.close(fig)
    print("fig7 saved")
except Exception as e:
    print("fig7 skipped:", e)

print("ALL FIGURES DONE")
