# Single-unit pre-stimulus choice selectivity: auROC + permutation tests, all sessions
import numpy as np
import decoding_lib as dl

SESSIONS = ["sub-92130c1b", "sub-70bf8cbd", "sub-9bebfe0b", "sub-c6e8125f"]
PRE = (-0.4, 0.0)
POST = (0.05, 0.45)
N_SHUF = 200

def auroc_mat(X, y):
    return np.array([dl.unit_auroc(X[:, u], y) for u in range(X.shape[1])])

all_rows = []
for sess in SESSIONS:
    trials, unit_spikes, ks2, regions = dl.load_session(f"cache_{sess}.npz")
    mask, _ = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
    units = [s for s, m in zip(unit_spikes, mask) if m]
    regs = regions[mask]
    stim_on = trials["stimOn_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)

    for tag, m, win in [("pre_all", valid, PRE), ("pre_zero", zero_c, PRE),
                        ("post_all", valid, POST)]:
        y = (choice[m] == 1).astype(int)
        t0s = stim_on[m]
        X = dl.count_in_windows(units, t0s + win[0], t0s + win[1])
        ar = auroc_mat(X, y)
        rng = np.random.default_rng(0)
        null = np.empty((N_SHUF, X.shape[1]))
        for s in range(N_SHUF):
            null[s] = auroc_mat(X, rng.permutation(y))
        # two-sided p per unit
        p = (np.sum(np.abs(null - 0.5) >= np.abs(ar - 0.5), axis=0) + 1) / (N_SHUF + 1)
        for u in range(X.shape[1]):
            all_rows.append(dict(session=sess, cond=tag, unit=u, auroc=ar[u],
                                 p=p[u], region=regs[u]))
        frac = (p < 0.05).mean()
        print(f"{sess} [{tag}]: frac sig units p<0.05 = {frac:.3f} "
              f"(expected 0.05), mean|auroc-.5| = {np.mean(np.abs(ar-0.5)):.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(all_rows)
df.to_csv("single_unit_auroc.csv", index=False)
print("saved single_unit_auroc.csv")

# region breakdown for pre_zero, annotated sessions
sub = df[(df.cond == "pre_zero") & (df.region != "unknown") & (df.p < 0.05)]
print("\nsignificant pre-stim 0%-trial choice units by region:")
print(sub.groupby("region").size().sort_values(ascending=False).head(15))
