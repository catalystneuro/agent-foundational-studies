"""Run the HD-cell pipeline across multiple sessions of DANDI 000056.

Eight sessions from six mice (skipping the two ~30 GB Mouse12 raw files
for speed). Per session: HD-cell identification, then preservation of
the HD ensemble correlation structure in REM/NREM sleep.
"""
import os
import numpy as np
from tqdm import tqdm

from hd_analysis import (load_session, get_epochs, analyze_session,
                         state_correlations)

SESSIONS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse17-130125": "e62179fc-cd8d-4901-98aa-ccb35086f9f7",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse25-140205": "48dae2dc-35f7-4a9e-bcdb-883a7d5e2d32",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
    "Mouse32-140820": "43be9315-c73f-4e83-adea-20e73ce75466",
}

OUT = "results_multisession"
os.makedirs(OUT, exist_ok=True)

N_SHUF = 500
MIN_CELLS_CORR = 4


def corr_of_corr(Cw, Cs):
    iu = np.triu_indices(Cw.shape[0], k=1)
    m = ~np.isnan(Cw[iu]) & ~np.isnan(Cs[iu])
    if m.sum() < 6:
        return np.nan, m.sum(), None, None
    r = np.corrcoef(Cw[iu][m], Cs[iu][m])[0, 1]
    return r, m.sum(), Cw[iu][m], Cs[iu][m]


def shuffle_corr_of_corr(Cw, Cs, n_shuf=200, seed=0):
    """Cell-label permutation null for the correlation of correlations."""
    rng = np.random.default_rng(seed)
    n = Cw.shape[0]
    iu = np.triu_indices(n, k=1)
    out = []
    for _ in range(n_shuf):
        p = rng.permutation(n)
        Css = Cs[np.ix_(p, p)]
        m = ~np.isnan(Cw[iu]) & ~np.isnan(Css[iu])
        if m.sum() >= 6:
            out.append(np.corrcoef(Cw[iu][m], Css[iu][m])[0, 1])
    return np.array(out)


summary = []
for name, asset_id in tqdm(SESSIONS.items(), desc="sessions"):
    print(f"\n=== {name} ===")
    nwb, io = load_session(asset_id)
    wake, rem, nrem = get_epochs(nwb)
    res = analyze_session(nwb, n_shuf=N_SHUF, seed=0)

    units = nwb["units"]
    units_f = units[units.metadata["rate"].values > 0.1]
    hd_keys = res["keys"][res["is_hd"]]

    out = {k: v for k, v in res.items() if v is not None}
    out["session"] = name

    if len(hd_keys) >= MIN_CELLS_CORR:
        Cw = state_correlations(units_f, hd_keys, wake)
        Cr = state_correlations(units_f, hd_keys, rem)
        Cn = state_correlations(units_f, hd_keys, nrem)

        r_rem, npairs_rem, pw_rem, pr_rem = corr_of_corr(Cw, Cr)
        r_nrem, npairs_nrem, pw_nrem, pr_nrem = corr_of_corr(Cw, Cn)
        null_rem = shuffle_corr_of_corr(Cw, Cr, seed=1)
        null_nrem = shuffle_corr_of_corr(Cw, Cn, seed=2)

        out.update(Cw=Cw, Cr=Cr, Cn=Cn,
                   r_rem=r_rem, r_nrem=r_nrem,
                   npairs_rem=npairs_rem, npairs_nrem=npairs_nrem,
                   pairs_wake_rem=pw_rem, pairs_rem=pr_rem,
                   pairs_wake_nrem=pw_nrem, pairs_nrem=pr_nrem,
                   null_rem=null_rem, null_nrem=null_nrem)
        p_rem = (np.sum(null_rem >= r_rem) + 1) / (len(null_rem) + 1)
        p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (len(null_nrem) + 1)
        print(f"  corr-of-corr REM: {r_rem:.3f} (p={p_rem:.3f}), "
              f"NREM: {r_nrem:.3f} (p={p_nrem:.3f})")
    else:
        print(f"  only {len(hd_keys)} HD cells, skipping correlations")
        out.update(r_rem=np.nan, r_nrem=np.nan, npairs_rem=0, npairs_nrem=0)

    np.savez(os.path.join(OUT, f"results_{name}.npz"),
             **{k: v for k, v in out.items() if v is not None})
    summary.append(dict(session=name, n_units=len(res["keys"]),
                        n_hd=int(res["is_hd"].sum()),
                        wake_dur=res["wake_dur"], rem_dur=res["rem_dur"],
                        nrem_dur=res["nrem_dur"],
                        r_rem=out["r_rem"], r_nrem=out["r_nrem"]))
    io.close()

import pandas as pd
df = pd.DataFrame(summary)
df.to_csv(os.path.join(OUT, "session_summary.csv"), index=False)
print("\n", df.to_string(index=False))
print("\ntotal HD cells:", df.n_hd.sum(), "/", df.n_units.sum())
