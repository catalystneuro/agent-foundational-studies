"""Generate final figures from results/ produced by run_analysis.py."""
import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('results/grid_metrics.csv')
maps = pickle.load(open('results/maps.pkl', 'rb'))
null = np.load('results/pooled_null.npy')
thr = float(open('results/threshold.txt').read())
grid = df[df.is_grid].sort_values('grid_score', ascending=False)

# ---- Figure 1: gallery of grid cells (rate map over autocorr) ----
n = min(12, len(grid))
sel = grid.head(n)
fig, axes = plt.subplots(2, n, figsize=(1.7*n, 4.6))
for j, (_, row) in enumerate(sel.iterrows()):
    rm, ac = maps[row.label]
    axes[0, j].imshow(rm, origin='lower', cmap='jet'); axes[0, j].set_aspect('equal')
    axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    axes[0, j].set_title(f'{row.peak_rate:.0f} Hz | {row.spacing_cm:.0f} cm', fontsize=8)
    axes[1, j].imshow(ac, origin='lower', cmap='jet', vmin=-0.5, vmax=1); axes[1, j].set_aspect('equal')
    axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    axes[1, j].set_title(f'grid score {row.grid_score:.2f}', fontsize=8)
axes[0, 0].set_ylabel('rate map', fontsize=10)
axes[1, 0].set_ylabel('autocorrelogram', fontsize=10)
fig.suptitle(f'Grid cells in MEC (DANDI:000582) — top {n} of {int(df.is_grid.sum())} classified grid cells',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.subplots_adjust(hspace=0.25)
plt.savefig('figures/01_grid_cell_gallery.png', dpi=120); plt.close()

# ---- Figure 2: grid-score distribution vs shuffle null ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = np.linspace(-0.9, 2.0, 40)
ax[0].hist(null, bins=bins, density=True, color='0.6', alpha=0.8, label='shuffle null')
ax[0].hist(df.grid_score, bins=bins, density=True, color='steelblue', alpha=0.6, label='observed cells')
ax[0].axvline(thr, color='red', ls='--', label=f'95th pct = {thr:.2f}')
ax[0].set_xlabel('grid score'); ax[0].set_ylabel('density'); ax[0].legend()
ax[0].set_title('Observed grid scores vs. shuffled null')
# ecdf
xs = np.sort(df.grid_score.values)
ax[1].plot(xs, np.linspace(0, 1, len(xs)), marker='o', ms=3)
ax[1].axvline(thr, color='red', ls='--')
frac = df.is_grid.mean()
ax[1].set_xlabel('grid score'); ax[1].set_ylabel('cumulative fraction of cells')
ax[1].set_title(f'{int(df.is_grid.sum())}/{len(df)} cells ({100*frac:.0f}%) exceed threshold')
plt.tight_layout(); plt.savefig('figures/02_grid_score_significance.png', dpi=120); plt.close()

# ---- Figure 3: population properties of grid cells ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
g = df[df.is_grid]
ax[0].hist(g.spacing_cm.dropna(), bins=np.arange(20, 90, 6), color='seagreen', edgecolor='k')
ax[0].set_xlabel('grid spacing (cm)'); ax[0].set_ylabel('# grid cells')
ax[0].set_title(f'Grid spacing (median {g.spacing_cm.median():.0f} cm)')
ax[1].hist(g.orient_deg.dropna(), bins=np.arange(0, 61, 6), color='goldenrod', edgecolor='k')
ax[1].set_xlabel('grid orientation (deg, mod 60)'); ax[1].set_ylabel('# grid cells')
ax[1].set_title('Grid orientation')
ax[2].scatter(df.spatial_info, df.grid_score, c=df.is_grid.map({True:'crimson', False:'0.6'}), s=25)
ax[2].axhline(thr, color='red', ls='--', lw=1)
ax[2].set_xlabel('spatial information (bits/spike)'); ax[2].set_ylabel('grid score')
ax[2].set_title('Grid score vs spatial information')
plt.tight_layout(); plt.savefig('figures/03_population_properties.png', dpi=120); plt.close()

print('figures written')
print(df.groupby('subject').agg(n_units=('label','count'), n_grid=('is_grid','sum')).to_string())
