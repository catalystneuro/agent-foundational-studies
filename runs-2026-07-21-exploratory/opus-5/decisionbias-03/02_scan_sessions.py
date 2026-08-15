"""Scan a sample of DANDI:000409 sessions and rank them for this analysis.

We want sessions with (a) plenty of trials including biased blocks, (b) a decent
yield of well-isolated units, and (c) coverage of areas where the IBL prior is
known to be represented (frontal cortex, striatum, thalamus, midbrain, hippocampus).
"""
import json
import numpy as np
from tqdm import tqdm

import ibl_io as io

# Coarse Allen groupings, matched against the full region names stored in the
# electrodes table.
GROUPS = {
    "frontal": ["Secondary motor", "Anterior cingulate", "Prelimbic", "Infralimbic",
                "Orbital area", "Primary motor", "Frontal pole"],
    "striatum": ["Caudoputamen", "Nucleus accumbens", "Striatum", "Olfactory tubercle",
                 "Globus pallidus"],
    "thalamus": ["thalamus", "Thalamus", "geniculate", "Lateral posterior nucleus",
                 "Posterior complex", "nucleus of the thalamus"],
    "midbrain": ["Superior colliculus", "Midbrain reticular", "Periaqueductal",
                 "Substantia nigra", "Ventral tegmental", "Anterior pretectal",
                 "Nucleus of the optic tract"],
    "hippocampus": ["Field CA1", "Field CA2", "Field CA3", "Dentate gyrus",
                    "Subiculum", "Entorhinal"],
    "visual": ["Primary visual", "visual area", "Visual area"],
}


def group_of(name):
    for g, keys in GROUPS.items():
        if any(k in name for k in keys):
            return g
    return "other"


def scan(asset):
    path, size, aid = asset
    hf = io.open_h5(aid)
    tr = io.read_trials(hf)
    reg = io.read_unit_regions(hf)
    small = io.read_unit_table_small(hf)
    good = small["kilosort2_label"] == "good"
    biased = tr["probability_left"] != 0.5
    groups = np.array([group_of(r) for r in reg])
    counts = {g: int(((groups == g) & good).sum()) for g in list(GROUPS) + ["other"]}
    zero = tr["gabor_stimulus_contrast"] == 0
    return dict(
        path=path, asset_id=aid, size=size,
        subject=path.split("/")[0].replace("sub-", ""),
        n_trials=int(len(tr["start_time"])),
        n_biased=int(biased.sum()),
        n_blocks=int((np.diff(tr["probability_left"]) != 0).sum() + 1),
        n_zero_contrast=int((zero & biased).sum()),
        n_units=int(len(reg)), n_good=int(good.sum()),
        counts=counts,
        n_target=int(sum(v for k, v in counts.items() if k != "other")),
    )


if __name__ == "__main__":
    assets = io.list_processed_assets()
    rng = np.random.default_rng(0)
    # Sample across subjects to avoid over-weighting any one animal.
    idx = rng.permutation(len(assets))[:90]
    rows = []
    for i in tqdm(sorted(idx), desc="scanning"):
        rows.append(scan(assets[i]))
    rows.sort(key=lambda r: -(r["n_target"] * min(r["n_biased"], 400)))
    with open("session_scan.json", "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows[:25]:
        print(f"{r['subject']:>16s} tr={r['n_trials']:4d} bias={r['n_biased']:4d} "
              f"zc={r['n_zero_contrast']:3d} good={r['n_good']:4d} "
              f"tgt={r['n_target']:4d} {r['counts']}")
