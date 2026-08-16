"""Run HD pipeline on the two replacement sessions."""
import os
import numpy as np
import pandas as pd
from hd_analysis import (load_session, get_epochs, analyze_session,
                         state_correlations)

SESSIONS = {
    "Mouse24-131217": "c6cfc0f2-dda6-4136-be2c-be4a4de5a50e",
    "Mouse20-130515": "f92f2709-4469-4c1d-a883-75eb66ee898a",
}
OUT = "results_multisession"
N_SHUF = 500
MIN_CELLS_CORR = 4


def corr_of_corr(Cw, Cs):
    iu = np.triu_indices(Cw.shape[0], k=1)
    m = ~np.isnan(Cw[iu]) & ~np.isnan(Cs[iu])
    if m.sum() < 6:
        return np.nan, m.sum(), None, None
    return np.corrcoef(Cw[iu][m], Cs[iu][m])[0, 1], m.sum(), Cw[iu][m], Cs[iu][m]


def shuffle_corr_of_corr(Cw, Cs, n_shuf=200, seed=0):
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
for name, asset_id in SESSIONS.items():
    print(f"\n=== {name} ===")
    nwb, io = load_session(asset_id)
    wake, rem, nrem = get_epochs(nwb)
    res = analyze_session(nwb, n_shuf=N_SHUF, seed=0)

    units = nwb["units"]
    units_f = units[units.metadata["rate"].values > 0.1]
    hd_keys = res["keys"][res["is_hd"]]

    out = {k: v for k, v in res.items() if v is not None}
    out["session"] = name

    r_rem = r_nrem = np.nan
    p_rem = p_nrem = np.nan
    npairs = 0
    if len(hd_keys) >= MIN_CELLS_CORR:
        Cw = state_correlations(units_f, hd_keys, wake)
        Cr = state_correlations(units_f, hd_keys, rem)
        Cn = state_correlations(units_f, hd_keys, nrem)
        r_rem, npairs, pw_rem, pr_rem = corr_of_corr(Cw, Cr)
        r_nrem, _, pw_nrem, pr_nrem = corr_of_corr(Cw, Cn)
        null_rem = shuffle_corr_of_corr(Cw, Cr, seed=1)
        null_nrem = shuffle_corr_of_corr(Cw, Cn, seed=2)
        p_rem = (np.sum(null_rem >= r_rem) + 1) / (len(null_rem) + 1)
        p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (len(null_nrem) + 1)
        out.update(Cw=Cw, Cr=Cr, Cn=Cn, r_rem=r_rem, r_nrem=r_nrem,
                   npairs_rem=npairs, pairs_wake_rem=pw_rem,
                   pairs_rem=pr_rem, pairs_wake_nrem=pw_nrem,
                   pairs_nrem=pr_nrem, null_rem=null_rem,
                   null_nrem=null_nrem)
        print(f"  corr-of-corr REM: {r_rem:.3f} (p={p_rem:.3f}), "
              f"NREM: {r_nrem:.3f} (p={p_nrem:.3f})")
    else:
        print(f"  only {len(hd_keys)} HD cells, skipping correlations")

    np.savez(os.path.join(OUT, f"results_{name}.npz"),
             **{k: v for k, v in out.items() if v is not None})
    summary.append(dict(session=name, mouse=name.split("-")[0],
                        n_units=len(res["keys"]),
                        n_hd=int(res["is_hd"].sum()),
                        wake_dur=res["wake_dur"], rem_dur=res["rem_dur"],
                        nrem_dur=res["nrem_dur"],
                        r_rem=r_rem, r_nrem=r_nrem,
                        p_rem=p_rem, p_nrem=p_nrem, npairs=npairs))
    io.close()

# merge into the shared summary csv
csv_path = os.path.join(OUT, "session_summary.csv")
df_old = pd.read_csv(csv_path)
df_new = pd.DataFrame(summary)
df = pd.concat([df_old, df_new], ignore_index=True)
df = df.drop_duplicates(subset="session", keep="last")
for c in ["p_rem", "p_nrem", "npairs", "mouse"]:
    if c not in df_old.columns:
        pass
df.to_csv(csv_path, index=False)
print("\n", df.to_string(index=False))
print("\ntotal HD cells:", df.n_hd.sum(), "/", df.n_units.sum())
