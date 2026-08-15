"""Scan candidate DANDI:000409 sessions and pick the ones that meet pre-specified criteria.

Only the trials table and the units metadata columns are read here (a few MB per session);
spike times are not touched.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import ibl_common as ic

N_CANDIDATES = 30
MIN_BIASED_TRIALS = 300
MIN_ZERO_CONTRAST = 40
MIN_UNITS = 60
MIN_BLOCKS = 6

sessions = ic.list_processed_sessions()
print(f"{len(sessions)} processed sessions in DANDI:{ic.DANDISET}")

# Deterministic candidate pool: one session per subject first (maximises animal diversity),
# then fill up in listing order.
by_subject = {}
for s in sessions:
    sub = s["path"].split("/")[0]
    by_subject.setdefault(sub, []).append(s)
pool = [v[0] for v in by_subject.values()]
pool = sorted(pool, key=lambda s: s["path"])[:N_CANDIDATES]

rows = []
for s in tqdm(pool, desc="scanning"):
    nwbfile, nwb, io = ic.open_session(s["path"])
    tr = ic.get_trials(nwbfile)
    units = ic.good_units(nwb)
    meta = ic.unit_metadata(units)
    n_units = len(units)

    biased = tr[tr.block_left.notna() & tr.choice_left.notna()]
    zero = biased[biased.gabor_stimulus_contrast == 0]
    n_blocks = int(tr.loc[tr.block_left.notna(), "block_index"].nunique())
    frac_agree, n_hi = ic.check_choice_convention(tr)

    # behavioural bias on zero-contrast trials: P(left choice | left block) - P(| right block)
    if len(zero) > 0:
        pl = zero.groupby("block_left")["choice_left"].mean()
        bias = float(pl.get(1.0, np.nan) - pl.get(0.0, np.nan))
    else:
        bias = np.nan

    rows.append(
        dict(
            path=s["path"],
            subject=s["path"].split("/")[0],
            n_trials=len(tr),
            n_biased=len(biased),
            n_zero=len(zero),
            n_blocks=n_blocks,
            n_units=n_units,
            n_regions=int(meta.region.nunique()),
            zero_contrast_bias=bias,
            choice_convention_agreement=frac_agree,
            n_high_contrast=n_hi,
        )
    )
    io.close()

df = pd.DataFrame(rows)
df["passes"] = (
    (df.n_biased >= MIN_BIASED_TRIALS)
    & (df.n_zero >= MIN_ZERO_CONTRAST)
    & (df.n_units >= MIN_UNITS)
    & (df.n_blocks >= MIN_BLOCKS)
)
df = df.sort_values(["passes", "n_units"], ascending=False)
df.to_csv("session_scan.csv", index=False)
print(df.to_string(index=False))
print("\npassing:", int(df.passes.sum()))
json.dump(
    df[df.passes].path.tolist(), open("selected_sessions.json", "w"), indent=1
)
