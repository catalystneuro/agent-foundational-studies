"""Generate population-level figures from population_results.pkl."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("population_results.pkl", "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(
    [
        {
            "session": r["session"],
            "unit_id": r["unit_id"],
            "histology": r["histology"],
            "n_spikes": r["n_spikes"],
            "mean_rate": r["mean_rate"],
            "gridness": r["gridness"],
            "null_95th": r["null_95th"],
            "is_grid_cell": r["is_grid_cell"],
        }
        for r in results
    ]
)
df.to_csv("population_summary.csv", index=False)
print(df)
print()
print("N units:", len(df))
print("N grid cells:", df["is_grid_cell"].sum())
print(df.groupby("histology")["is_grid_cell"].agg(["sum", "count"]))

# --- Figure 1: population histogram of gridness scores ---
fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(-1.5, 1.5, 31)
ax.hist(df.loc[~df["is_grid_cell"], "gridness"], bins=bins, color="0.7", label="not classified as grid cell")
ax.hist(df.loc[df["is_grid_cell"], "gridness"], bins=bins, color="crimson", label="grid cell (gridness > shuffled 95th pct)")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Gridness score")
ax.set_ylabel("Number of units")
ax.set_title(f"Gridness score distribution across MEC units (n={len(df)}, {df['is_grid_cell'].sum()} classified as grid cells)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/population_gridness_histogram.png", dpi=150)
plt.close()

# --- Figure 2: gridness by histological layer ---
fig, ax = plt.subplots(figsize=(8, 5))
layers = sorted(df["histology"].unique())
data_by_layer = [df.loc[df["histology"] == layer, "gridness"].values for layer in layers]
bp = ax.boxplot(data_by_layer, labels=layers, showfliers=False)
for i, layer in enumerate(layers):
    y = df.loc[df["histology"] == layer, "gridness"].values
    colors = ["crimson" if g else "0.4" for g in df.loc[df["histology"] == layer, "is_grid_cell"].values]
    x = np.random.default_rng(0).normal(i + 1, 0.05, size=len(y))
    ax.scatter(x, y, c=colors, s=25, zorder=3)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_ylabel("Gridness score")
ax.set_title("Gridness score by MEC layer (red = classified grid cell)")
plt.tight_layout()
plt.savefig("figures/gridness_by_layer.png", dpi=150)
plt.close()

# --- Figure 3: example grid cells (top scoring) ---
grid_cells = [r for r in results if r["is_grid_cell"]]
grid_cells.sort(key=lambda r: r["gridness"], reverse=True)
n_examples = min(6, len(grid_cells))

fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6))
if n_examples == 1:
    axes = axes.reshape(2, 1)
for i in range(n_examples):
    r = grid_cells[i]
    axes[0, i].imshow(r["smoothed_ratemap"].T, origin="lower", cmap="jet")
    axes[0, i].set_title(f"{r['session'].split('/')[0]}\nunit {r['unit_id']}", fontsize=9)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(r["autocorr"].T, origin="lower", cmap="jet")
    axes[1, i].set_title(f"gridness={r['gridness']:.2f}", fontsize=9)
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Rate map", fontsize=10)
axes[1, 0].set_ylabel("Autocorrelogram", fontsize=10)
fig.suptitle("Top-scoring grid cells across sessions (DANDI 000582, Sargolini et al. 2006)")
plt.tight_layout()
plt.savefig("figures/example_grid_cells.png", dpi=150)
plt.close()

# --- Figure 4: shuffle null distribution example for one grid cell ---
example = grid_cells[0]
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(example["null_scores"], bins=20, color="0.7", label="shuffled null distribution")
ax.axvline(example["gridness"], color="crimson", lw=2, label=f"observed gridness = {example['gridness']:.2f}")
ax.axvline(example["null_95th"], color="k", ls="--", label="95th percentile of null")
ax.set_xlabel("Gridness score")
ax.set_ylabel("Count (shuffles)")
ax.set_title(f"Shuffle test: {example['session'].split('/')[0]} unit {example['unit_id']}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/shuffle_null_example.png", dpi=150)
plt.close()

print("Figures saved.")
