"""04_visualize.py — population figures for the DANDI 000582 grid-cell analysis.

Reads results/grid_scores.csv and results/maps/*.npz produced by
03_run_all_sessions.py and generates:
  fig_03_example_grid_cells.png   gallery of the top significant grid cells
  fig_04_grid_score_distributions.png   observed vs shuffle-null distributions
  fig_05_layer_breakdown.png      grid-cell prevalence and scores by MEC layer
  fig_06_grid_spacing.png         grid spacing / orientation of significant cells
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
P_SIG = 0.05
LAYER_ORDER = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]


def load_maps(session):
    safe = session.replace("/", "_").replace(".nwb", "")
    return np.load(os.path.join(MAPS_DIR, safe + ".npz"))


def fig_example_grid_cells(df):
    """Gallery: top 12 significant grid cells (rate map + autocorrelation)."""
    sig = df[(df["shuffle_p"] < P_SIG)].sort_values("grid_score", ascending=False)
    # spread examples across sessions and layers for variety
    seen_sessions, picks = set(), []
    for _, row in sig.iterrows():
        if row["session"] not in seen_sessions or len(picks) >= 8:
            picks.append(row)
            seen_sessions.add(row["session"])
        if len(picks) >= 12:
            break

    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    for i, row in enumerate(picks):
        maps = load_maps(row["session"])
        uid = row["unit_id"]
        r, c = divmod(i, 6)
        ax = axes[r * 2, c]
        ax.imshow(maps[f"rate_map_{uid}"], origin="lower", cmap="jet")
        sub = row["session"].split("/")[0].replace("sub-", "rat ")
        ax.set_title(f"{sub} {row['unit_name']} ({row['layer'].replace('MEC ', '')})",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[r * 2 + 1, c]
        ax.imshow(maps[f"acorr_{uid}"], origin="lower", cmap="jet")
        ax.set_title(f"gs={row['grid_score']:.2f}, d={row['grid_spacing']:.0f} cm, "
                     f"p={row['shuffle_p']:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(picks) * 2:]:
        ax.axis("off")
    for r in range(2):
        axes[r * 2, 0].set_ylabel("rate map", fontsize=9)
        axes[r * 2 + 1, 0].set_ylabel("autocorr", fontsize=9)
    fig.suptitle("Significant grid cells from DANDI 000582 (Sargolini et al. 2006) — "
                 "top grid scores across sessions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fig_03_example_grid_cells.png", dpi=150)
    plt.close(fig)
    print("saved fig_03_example_grid_cells.png")


def fig_score_distributions(df):
    """Observed grid scores vs pooled shuffle null; significance scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    bins = np.linspace(-1.5, 2.0, 60)
    ax.hist(df["grid_score"], bins=bins, color="0.4", label=f"all units (n={len(df)})")
    sig = df[df["shuffle_p"] < P_SIG]
    ax.hist(sig["grid_score"], bins=bins, color="tab:red",
            label=f"significant (n={len(sig)})")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("grid score")
    ax.set_ylabel("number of units")
    ax.set_title("Grid score distribution")
    ax.legend(fontsize=8)

    # pooled shuffle null for all shuffled (candidate) units
    ax = axes[1]
    null_scores = []
    for session in df.loc[df["shuffle_p"].notna(), "session"].unique():
        maps = load_maps(session)
        for k in maps.files:
            if k.startswith("shuffle_"):
                null_scores.append(maps[k])
    null_scores = np.concatenate(null_scores)
    null_scores = null_scores[~np.isnan(null_scores)]
    ax.hist(null_scores, bins=np.linspace(-1.5, 2.0, 80), density=True, color="0.6",
            label=f"shuffle null (n={len(null_scores)})")
    ax.hist(df.loc[df["shuffle_p"].notna(), "grid_score"], bins=np.linspace(-1.5, 2.0, 80),
            density=True, histtype="step", lw=2, color="tab:red", label="observed (candidates)")
    ax.set_xlabel("grid score")
    ax.set_ylabel("density")
    ax.set_title("Observed vs circular-shift null")
    ax.legend(fontsize=8)

    ax = axes[2]
    cand = df[df["shuffle_p"].notna()]
    ax.scatter(cand["grid_score"], cand["shuffle_z"], s=8, alpha=0.4,
               c=np.where(cand["shuffle_p"] < P_SIG, "tab:red", "0.5"))
    ax.axhline(1.645, color="k", ls="--", lw=0.8, label="z = 1.645 (p=0.05, one-sided)")
    ax.set_xlabel("observed grid score")
    ax.set_ylabel("shuffle z-score")
    ax.set_title("Significance of candidate units")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("fig_04_grid_score_distributions.png", dpi=150)
    plt.close(fig)
    print("saved fig_04_grid_score_distributions.png")


def fig_layer_breakdown(df):
    """Grid-cell prevalence and score distributions by MEC layer."""
    df = df.copy()
    df["layer"] = df["layer"].replace("", np.nan)
    dfl = df.dropna(subset=["layer"])
    dfl = dfl[dfl["layer"].isin(LAYER_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    frac = [((dfl["layer"] == layer) & (dfl["shuffle_p"] < P_SIG)).sum() /
            max((dfl["layer"] == layer).sum(), 1) for layer in LAYER_ORDER]
    ns = [(dfl["layer"] == layer).sum() for layer in LAYER_ORDER]
    bars = ax.bar(range(len(LAYER_ORDER)), frac, color="steelblue")
    ax.set_xticks(range(len(LAYER_ORDER)))
    ax.set_xticklabels([l.replace("MEC ", "") for l in LAYER_ORDER])
    for rect, f, n in zip(bars, frac, ns):
        ax.text(rect.get_x() + rect.get_width() / 2, f + 0.01,
                f"{f * 100:.0f}%\n(n={n})", ha="center", fontsize=9)
    ax.set_ylabel("fraction significant grid cells")
    ax.set_title("Grid-cell prevalence by layer")
    ax.set_ylim(0, max(frac) * 1.35)

    ax = axes[1]
    data = [dfl.loc[dfl["layer"] == layer, "grid_score"].values for layer in LAYER_ORDER]
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
    sig_data = [dfl.loc[(dfl["layer"] == layer) & (dfl["shuffle_p"] < P_SIG),
                        "grid_score"].values for layer in LAYER_ORDER]
    for i, vals in enumerate(sig_data):
        ax.scatter(np.full(len(vals), i + 1) + np.random.default_rng(0).normal(0, 0.04, len(vals)),
                   vals, s=10, color="tab:red", zorder=3, label="significant" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(1, len(LAYER_ORDER) + 1))
    ax.set_xticklabels([l.replace("MEC ", "") for l in LAYER_ORDER])
    ax.set_ylabel("grid score")
    ax.set_title("Grid scores by layer")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("fig_05_layer_breakdown.png", dpi=150)
    plt.close(fig)
    print("saved fig_05_layer_breakdown.png")


def fig_grid_spacing(df):
    """Grid spacing and orientation of significant grid cells."""
    sig = df[(df["shuffle_p"] < P_SIG)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(sig["grid_spacing"], bins=np.arange(20, 200, 10), color="steelblue")
    med = sig["grid_spacing"].median()
    ax.axvline(med, color="tab:red", ls="--", label=f"median = {med:.0f} cm")
    ax.set_xlabel("grid spacing (cm)")
    ax.set_ylabel("number of units")
    ax.set_title(f"Grid spacing of significant cells (n={len(sig)})")
    ax.legend(fontsize=8)

    ax = axes[1]
    orient = sig["grid_orientation"].dropna().values % 60  # 60-deg periodicity of hex grid
    ax.hist(orient, bins=np.arange(0, 65, 5), color="steelblue")
    ax.set_xlabel("grid orientation (deg, modulo 60)")
    ax.set_ylabel("number of units")
    ax.set_title("Grid orientation distribution")

    fig.tight_layout()
    fig.savefig("fig_06_grid_spacing.png", dpi=150)
    plt.close(fig)
    print("saved fig_06_grid_spacing.png")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RESULTS_DIR, "grid_scores.csv"))
    print(f"{len(df)} units, {(df['shuffle_p'] < P_SIG).sum()} significant grid cells "
          f"(p<{P_SIG}, 100 shuffles)")
    fig_example_grid_cells(df)
    fig_score_distributions(df)
    fig_layer_breakdown(df)
    fig_grid_spacing(df)
