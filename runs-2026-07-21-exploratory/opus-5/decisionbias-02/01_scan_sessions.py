"""Scan a subset of DANDI 000409 sessions and record what each one offers.

We only touch the trials table, the units metadata and the electrode locations,
so this stays cheap even though the files are ~1 GB each.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import ibl_common as ic

assets = pd.read_csv("processed_assets.csv")
# Deterministic spread over the (size-sorted) session list.
cand = assets.iloc[np.linspace(20, len(assets) - 1, 44).astype(int)].reset_index(drop=True)

rows = []
for _, a in tqdm(list(cand.iterrows()), desc="scanning sessions"):
    nwb = ic.open_nwb(a.asset_id)
    tr = ic.get_trials(nwb)
    v = ic.valid_trials(tr)
    u = ic.get_units(nwb)
    reg = u.region.value_counts()
    zero = v[v.abs_contrast == 0]
    # Block-structure integrity.  A minority of files in this dandiset carry a
    # ``probability_left`` that flips on every trial, which is not the IBL block
    # schedule; those sessions have to be dropped, not analysed.
    pl = v.probability_left.to_numpy()
    runs = np.diff(np.flatnonzero(np.r_[True, pl[1:] != pl[:-1], True]))
    rows.append(
        dict(
            asset_id=a.asset_id,
            path=a.path,
            size_gb=a["size"] / 1e9,
            session_id=a.path.split("ses-")[1][:36],
            n_units=len(u),
            n_trials=len(tr),
            n_biased=len(v),
            n_zero=len(zero),
            n_regions=int((reg >= 10).sum()),
            n_blocks=len(runs),
            median_block_len=float(np.median(runs)),
            top_regions=";".join(f"{k}:{v_}" for k, v_ in reg.items() if v_ >= 10),
            # Behavioural bias: difference in P(choose right) between blocks,
            # restricted to |contrast| <= 6.25 % where the prior dominates.
            bias=float(
                v[v.abs_contrast <= 0.0625].groupby("block_right").choice_right.mean().diff().iloc[-1]
            )
            if v[v.abs_contrast <= 0.0625].block_right.nunique() == 2
            else np.nan,
        )
    )
    print(rows[-1]["session_id"], rows[-1]["n_units"], rows[-1]["n_biased"], rows[-1]["top_regions"])

scan = pd.DataFrame(rows)
scan.to_csv("session_scan.csv", index=False)
print(scan[["session_id", "n_units", "n_biased", "n_zero", "n_regions",
            "median_block_len", "bias"]].to_string())

# Session selection.  Data-quality criteria only, plus a stratification on the
# *behavioural* bias so the sample is not enriched for strongly biased animals
# (the neural analysis is then not selected on anything neural).
good = scan[(scan.median_block_len >= 20) & (scan.n_blocks >= 4)
            & (scan.n_biased >= 300) & (scan.n_units >= 200)].copy()
print(f"\n{len(good)}/{len(scan)} sessions pass the block-integrity and size criteria")
good["bias_quartile"] = pd.qcut(good.bias, 4, labels=False)
sel = (good.sort_values(["bias_quartile", "n_units"], ascending=[True, False])
           .groupby("bias_quartile").head(5).sort_values("bias").reset_index(drop=True))
sel.to_csv("selected_sessions.csv", index=False)
print(f"selected {len(sel)} sessions for extraction")
