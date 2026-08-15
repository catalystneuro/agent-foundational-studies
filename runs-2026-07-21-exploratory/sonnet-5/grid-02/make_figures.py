import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

with open('grid_analysis_results.pkl', 'rb') as f:
    all_data = pickle.load(f)

# Flatten all unit results across sessions
all_results = []
for session_name, d in all_data.items():
    all_results.extend(d['results'])

print(f"Total units: {len(all_results)}")
n_grid = sum(r['is_grid_cell'] for r in all_results)
print(f"Grid cells (gridness > shuffle 95th pct): {n_grid}")

# ---- Figure 1: raw trajectory + spike raster for one session ----
first_session = list(all_data.keys())[0]
position = all_data[first_session]['position']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(position['x'].values, position['y'].values, lw=0.3, color='gray', alpha=0.7)
axes[0].set_title(f"Rat trajectory in open field ({first_session})")
axes[0].set_xlabel("x (cm)")
axes[0].set_ylabel("y (cm)")
axes[0].set_aspect('equal')

axes[1].plot(position.t, position['x'].values, label='x', lw=0.5)
axes[1].plot(position.t, position['y'].values, label='y', lw=0.5)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("position (cm)")
axes[1].set_title("Position over time")
axes[1].legend()
plt.tight_layout()
plt.savefig('fig1_trajectory.png', dpi=130)
plt.close()
print("saved fig1_trajectory.png")

# ---- Figure 2: rate maps + autocorrelograms for example grid cells ----
grid_results = sorted(
    [r for r in all_results if r['is_grid_cell']],
    key=lambda r: -r['gridness']
)
n_show = min(6, len(grid_results))

fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 7))
if n_show == 1:
    axes = axes.reshape(2, 1)
for i in range(n_show):
    r = grid_results[i]
    rate_map = r['rate_map']
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    smoothed[np.isnan(rate_map)] = np.nan
    ax = axes[0, i]
    im = ax.imshow(smoothed, origin='lower', cmap='jet')
    ax.set_title(f"{r['session']}\nunit {r['unit_id']}, g={r['gridness']:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax2 = axes[1, i]
    im2 = ax2.imshow(r['autocorr'], origin='lower', cmap='jet', vmin=-1, vmax=1)
    ax2.set_title("spatial autocorr", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046)

axes[0, 0].set_ylabel("Rate map", fontsize=11)
axes[1, 0].set_ylabel("Autocorrelogram", fontsize=11)
plt.suptitle("Top grid cells: firing rate maps and spatial autocorrelograms", y=1.02)
plt.tight_layout()
plt.savefig('fig2_grid_cell_examples.png', dpi=130, bbox_inches='tight')
plt.close()
print("saved fig2_grid_cell_examples.png")

# ---- Figure 3: non-grid cell examples for comparison ----
non_grid_results = sorted(
    [r for r in all_results if not r['is_grid_cell'] and not np.isnan(r['gridness'])],
    key=lambda r: r['gridness']
)
n_show2 = min(4, len(non_grid_results))
fig, axes = plt.subplots(1, n_show2, figsize=(3.5 * n_show2, 3.8))
if n_show2 == 1:
    axes = [axes]
for i in range(n_show2):
    r = non_grid_results[i]
    rate_map = r['rate_map']
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    smoothed[np.isnan(rate_map)] = np.nan
    ax = axes[i]
    im = ax.imshow(smoothed, origin='lower', cmap='jet')
    ax.set_title(f"{r['session']} unit {r['unit_id']}\ng={r['gridness']:.2f} (not grid)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle("Example non-grid MEC units", y=1.05)
plt.tight_layout()
plt.savefig('fig3_non_grid_examples.png', dpi=130, bbox_inches='tight')
plt.close()
print("saved fig3_non_grid_examples.png")

# ---- Figure 4: population summary - gridness distribution vs shuffle null ----
all_gridness = np.array([r['gridness'] for r in all_results])
all_null = np.concatenate([r['shuffle_null'] for r in all_results])
all_95pct = np.array([r['shuffle_95pct'] for r in all_results])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
bins = np.linspace(-1.5, 2.0, 30)
ax.hist(all_null, bins=bins, density=True, alpha=0.5, label='shuffled null (all units pooled)', color='gray')
ax.hist(all_gridness[~np.isnan(all_gridness)], bins=bins, density=True, alpha=0.6, label='observed gridness', color='C1')
ax.axvline(np.nanmedian(all_95pct), color='k', linestyle='--', label='median shuffle 95th pct threshold')
ax.set_xlabel("Gridness score")
ax.set_ylabel("Density")
ax.set_title("Observed vs. shuffled gridness scores")
ax.legend(fontsize=9)

ax = axes[1]
sessions = list(all_data.keys())
session_labels = []
session_counts = []
session_grid = []
for s in sessions:
    res = all_data[s]['results']
    session_labels.append(s)
    session_counts.append(len(res))
    session_grid.append(sum(r['is_grid_cell'] for r in res))
x = np.arange(len(sessions))
ax.bar(x, session_counts, color='lightgray', label='total units')
ax.bar(x, session_grid, color='C1', label='grid cells')
ax.set_xticks(x)
ax.set_xticklabels(session_labels, rotation=30, ha='right')
ax.set_ylabel("Number of units")
ax.set_title("Grid cells identified per session")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig4_population_summary.png', dpi=130)
plt.close()
print("saved fig4_population_summary.png")

# Print summary table
print("\n=== SUMMARY TABLE ===")
for r in sorted(all_results, key=lambda r: (r['session'], r['unit_id'])):
    print(f"{r['session']:15s} unit {r['unit_id']:2d} ({r['histology']:10s}): "
          f"gridness={r['gridness']:.3f}  95pct_null={r['shuffle_95pct']:.3f}  "
          f"grid_cell={r['is_grid_cell']}  n_spikes={r['n_spikes']}")
