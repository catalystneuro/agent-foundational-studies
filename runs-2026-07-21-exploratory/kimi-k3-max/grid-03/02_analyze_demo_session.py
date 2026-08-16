"""02_analyze_demo_session.py — grid-cell analysis prototype on the demo session.

Computes occupancy, smoothed rate maps, spatial autocorrelations, and grid
scores for all units of sub-11265_ses-16030604, and saves a gallery figure.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from grid_utils import analyze_session_units, load_session

DEMO_PATH = "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb"

if __name__ == "__main__":
    with open("asset_manifest.json") as f:
        manifest = json.load(f)

    nwb, io = load_session(manifest[DEMO_PATH])
    results, meta = analyze_session_units(nwb, run_shuffles=False)
    print(f"arena size: {meta['arena_size'][0]:.0f} x {meta['arena_size'][1]:.0f} cm, "
          f"duration {meta['duration']:.0f} s, coverage {meta['coverage']:.3f}")

    for uid, res in sorted(results.items()):
        print(f"unit {uid:2d} ({res['unit_name']:6s}, {res['layer']}): "
              f"{res['n_spikes']:5d} spikes, grid score {res['grid_score']:+.3f}, "
              f"spacing {res['grid_spacing']:.0f} cm")

    # --- Gallery: rate map + autocorrelation for every analyzed unit ---
    # Layout: blocks of (rate-map row, autocorr row), 7 units per block.
    n = len(results)
    ncols = 7
    nblocks = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nblocks * 2, ncols, figsize=(2.2 * ncols, 2.2 * nblocks * 2))
    for i, (uid, res) in enumerate(sorted(results.items())):
        block, col = divmod(i, ncols)
        ax = axes[block * 2, col]
        ax.imshow(res["rate_map"], origin="lower", cmap="jet")
        ax.set_title(f"u{uid} {res['unit_name']}, {res['n_spikes']} spk", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[block * 2 + 1, col]
        ax.imshow(res["acorr"], origin="lower", cmap="jet")
        ax.set_title(f"gs={res['grid_score']:.2f}, d={res['grid_spacing']:.0f} cm", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.flat[n * 2:]:
        ax.axis("off")
    for block in range(nblocks):
        axes[block * 2, 0].set_ylabel("rate map", fontsize=9)
        axes[block * 2 + 1, 0].set_ylabel("autocorr", fontsize=9)
    fig.suptitle("sub-11265_ses-16030604 — all units: smoothed rate maps and spatial autocorrelations")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("fig_02_demo_session_gallery.png", dpi=150)
    print("saved fig_02_demo_session_gallery.png")
    io.close()
