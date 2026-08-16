"""07_fig_placefields.py — place-field template figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = np.load("scripts/place_fields.npz")
template, centers = P["template"], P["centers"]
rm_pos, rm_neg = P["rm_pos"], P["rm_neg"]
order = P["order"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5),
                         gridspec_kw={"width_ratios": [1.2, 1, 1]})

# (a) direction-pooled template, z-scored per cell, sorted by peak
ax = axes[0]
z = (template - np.nanmean(template, axis=1, keepdims=True)) / np.nanstd(template, axis=1, keepdims=True)
im = ax.imshow(z, aspect="auto", cmap="viridis",
               extent=[centers[0], centers[-1], len(z), 0])
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"A  place-field template (n={len(z)} cells)", loc="left")
fig.colorbar(im, ax=ax, label="z-scored rate", shrink=0.8)

# (b) six example cells: both directions
ax = axes[1]
picks = np.linspace(4, len(order) - 4, 6).astype(int)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(picks)))
for c, pi in zip(colors, picks):
    ax.plot(centers, rm_pos[pi] + 0, color=c, lw=1.2)
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("B  example ratemaps (+dir)", loc="left")

ax = axes[2]
for c, pi in zip(colors, picks):
    ax.plot(centers, rm_neg[pi] + 0, color=c, lw=1.2)
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("C  same cells (-dir)", loc="left")

fig.tight_layout()
fig.savefig("figures/fig2_place_fields.png", dpi=150)
print("saved figures/fig2_place_fields.png")
