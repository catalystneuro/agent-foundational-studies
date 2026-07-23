"""Prototype the Bayesian decoding section against the main script's own definitions."""
import time

t0 = time.time()
src = open("hippocampal_place_cells.py").read().split("# ### Figure 2")[0]
exec(compile(src, "hippocampal_place_cells.py", "exec"), globals())
print(f"[setup {time.time() - t0:.1f}s]")

import numpy as np
import pynapple as nap

BIN_SIZE = 0.25


def decode_cv(units, beh, direction, bin_size=BIN_SIZE, permute_maps=False, seed=0):
    """Decode position on held-out laps: maps from odd laps decode even laps, and back."""
    rng = np.random.default_rng(seed)
    odd, even = odd_even_laps(beh, direction)
    dec_t, dec_x, true_x, posts, ep_used = [], [], [], [], []
    for train_ep, test_ep in [(odd, even), (even, odd)]:
        tc = rate_maps(units, beh, train_ep)
        if permute_maps:
            tc = tc.copy(data=np.asarray(tc.values)[rng.permutation(tc.shape[0])])
        decoded, proba = nap.decode_bayes(tc, units, test_ep, bin_size)
        truth = beh["position"].interpolate(decoded, ep=test_ep)
        dec_t.append(decoded.t); dec_x.append(decoded.d)
        true_x.append(truth.d); posts.append(proba); ep_used.append(test_ep)
    order = np.argsort(np.concatenate(dec_t))
    return dict(
        t=np.concatenate(dec_t)[order],
        decoded=np.concatenate(dec_x)[order],
        true=np.concatenate(true_x)[order],
        posteriors=posts, test_eps=ep_used,
    )


for d in ("right", "left"):
    r = decode_cv(pyr, beh, d)
    ctl = decode_cv(pyr, beh, d, permute_maps=True)
    err = np.abs(r["decoded"] - r["true"])
    cerr = np.abs(ctl["decoded"] - ctl["true"])
    ok = np.isfinite(err)
    print(f"{d:5s}: n_bins={ok.sum():5d}  median err {100 * np.median(err[ok]):5.1f} cm  "
          f"| shuffled-map control {100 * np.nanmedian(cerr):5.1f} cm  "
          f"| r={np.corrcoef(r['decoded'][ok], r['true'][ok])[0, 1]:.3f}")

r = decode_cv(pyr, beh, "right")
p = r["posteriors"][0]
print("posterior type", type(p), getattr(p, "shape", None))
print("coords:", getattr(p, "coords", None))
print(f"[total {time.time() - t0:.1f}s]")
