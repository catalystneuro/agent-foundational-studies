"""Test aggregate statistics across the 5 cached sessions."""
import numpy as np

SESSIONS = ["Mouse17-130128", "Mouse20-130514", "Mouse24-131213",
            "Mouse25-140123", "Mouse28-140310"]
rng = np.random.default_rng(7)

per_session = {}
for s in SESSIONS:
    d = np.load(f"cache/results_{s}.npz", allow_pickle=True)
    per_session[s] = {k: d[k] for k in d.files}
    n = len(per_session[s]["keys"])
    nh = int(per_session[s]["is_hd"].sum())
    has = [k for k in per_session[s] if k.startswith("pairs_")]
    print(f"{s}: {nh}/{n} HD cells; pair sets: {has}; "
          f"occ_mvl={float(per_session[s]['occ_mvl']):.3f}")

# pooled wake-sleep with NaN handling
wake_pairs, rem_pairs, nrem_pairs, sess_id = [], [], [], []
for i, s in enumerate(SESSIONS):
    d = per_session[s]
    if all(k in d for k in ["pairs_Awake", "pairs_REM", "pairs_Non-REM"]):
        w, r, n_ = d["pairs_Awake"], d["pairs_REM"], d["pairs_Non-REM"]
        ok = ~(np.isnan(w) | np.isnan(r) | np.isnan(n_))
        print(f"{s}: {ok.sum()}/{len(ok)} valid pairs")
        wake_pairs.append(w[ok]); rem_pairs.append(r[ok]); nrem_pairs.append(n_[ok])
        sess_id.append(np.full(ok.sum(), i))
wake_pairs = np.concatenate(wake_pairs)
rem_pairs = np.concatenate(rem_pairs)
nrem_pairs = np.concatenate(nrem_pairs)
sess_id = np.concatenate(sess_id)

r_rem = np.corrcoef(wake_pairs, rem_pairs)[0, 1]
r_nrem = np.corrcoef(wake_pairs, nrem_pairs)[0, 1]
n_perm = 10_000
null_rem = np.empty(n_perm)
null_nrem = np.empty(n_perm)
for p in range(n_perm):
    perm = np.empty(len(wake_pairs), dtype=int)
    for i in np.unique(sess_id):
        idx = np.where(sess_id == i)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    null_rem[p] = np.corrcoef(wake_pairs, rem_pairs[perm])[0, 1]
    null_nrem[p] = np.corrcoef(wake_pairs, nrem_pairs[perm])[0, 1]
p_rem = (np.sum(null_rem >= r_rem) + 1) / (n_perm + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (n_perm + 1)
print(f"\npooled: {len(wake_pairs)} pairs from {len(np.unique(sess_id))} sessions")
print(f"wake vs REM:  r = {r_rem:.3f} (p = {p_rem:.5f})")
print(f"wake vs NREM: r = {r_nrem:.3f} (p = {p_nrem:.5f})")

# preferred directions uniformity
pref_all = np.concatenate([per_session[s]["pref_angle"][per_session[s]["is_hd"]]
                           for s in SESSIONS])
pref_all = pref_all[~np.isnan(pref_all)]
R_obs = np.abs(np.exp(1j * pref_all).mean())
null_R = np.abs(np.exp(1j * rng.uniform(0, 2 * np.pi, size=(10_000, len(pref_all)))).mean(axis=1))
p_unif = (np.sum(null_R >= R_obs) + 1) / 10_001
print(f"\npref directions: n={len(pref_all)}, R={R_obs:.3f}, uniformity p={p_unif:.3f}")

mvl_all = np.concatenate([per_session[s]["mvl"] for s in SESSIONS])
is_hd_all = np.concatenate([per_session[s]["is_hd"] for s in SESSIONS])
print(f"total: {is_hd_all.sum()}/{len(mvl_all)} HD cells "
      f"({100 * is_hd_all.mean():.0f}%)")
print(f"HD cell MVL: median {np.nanmedian(mvl_all[is_hd_all]):.2f}, "
      f"range [{np.nanmin(mvl_all[is_hd_all]):.2f}, {np.nanmax(mvl_all[is_hd_all]):.2f}]")
