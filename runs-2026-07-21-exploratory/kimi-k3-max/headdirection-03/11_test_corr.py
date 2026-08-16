"""Test pairwise correlation preservation across states."""
import numpy as np
from hd_analysis import (load_session, get_epochs, analyze_session,
                         state_correlations)

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
nwb, io = load_session(ASSET_ID)
wake, rem, nrem = get_epochs(nwb)
res = analyze_session(nwb, n_shuf=100, seed=0, verbose=False)

units = nwb["units"]
units_f = units[units.metadata["rate"].values > 0.1]
hd_keys = res["keys"][res["is_hd"]]
print("HD cell keys:", hd_keys)

Cw = state_correlations(units_f, hd_keys, wake, bin_size=0.1)
Cr = state_correlations(units_f, hd_keys, rem, bin_size=0.1)
Cn = state_correlations(units_f, hd_keys, nrem, bin_size=0.1)

iu = np.triu_indices(len(hd_keys), k=1)
m = ~np.isnan(Cw[iu]) & ~np.isnan(Cr[iu])
print("wake-REM corr-of-corr:", np.corrcoef(Cw[iu][m], Cr[iu][m])[0, 1],
      f"(n={m.sum()} pairs)")
m = ~np.isnan(Cw[iu]) & ~np.isnan(Cn[iu])
print("wake-NREM corr-of-corr:", np.corrcoef(Cw[iu][m], Cn[iu][m])[0, 1],
      f"(n={m.sum()} pairs)")

# sanity: cells with similar wake prefs should be positively correlated in wake
prefs = res["pref_wake"][res["is_hd"]]
order = np.argsort(prefs)
print("wake corr matrix sorted by pref dir:")
print(np.round(Cw[np.ix_(order, order)], 2))
