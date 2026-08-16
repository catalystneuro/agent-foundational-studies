"""Block-shift null for the instantaneous-velocity-direction tuning.

The per-bin cosine F test treats 20 ms bins as independent samples. They are not: both firing
rate and hand velocity are strongly autocorrelated within a reach, so the nominal p-values are
anticonservative. This builds the honest null by circularly shifting the neural series against
the kinematics by a large offset, which preserves the autocorrelation of both signals exactly
while destroying the true pairing.
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

CACHE = "./cache"


def cosine_fit(rates, angles):
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    ss_res = ((rates - X @ beta) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    n = len(angles)
    fstat = (ss_tot - ss_res) / 2 / (ss_res / (n - 3))
    return pd.DataFrame(dict(b0=beta[0], pd=np.arctan2(beta[2], beta[1]),
                             mod_depth=np.hypot(beta[1], beta[2]),
                             r2=1 - ss_res / ss_tot, f=fstat, p=stats.f.sf(fstat, 2, n - 3)))


def shift_by_lag(M, lag_bins, key="trial"):
    n = len(M["t"])
    src = np.arange(n) + lag_bins
    ok = (src >= 0) & (src < n)
    ok[ok] &= M[key][src[ok]] == M[key][np.arange(n)[ok]]
    return np.arange(n)[ok], src[ok]


def block_shift_null(M, best_lag, n_shuffles=100, speed_thresh=100.0, key="trial", seed=0):
    """Observed vs circularly-shifted cosine R^2 per unit.

    Returns (observed_fit, null_r2) where null_r2 is (n_shuffles, n_units).
    """
    bin_size = M.get("bin_size", 0.02)
    i_n, i_k = shift_by_lag(M, int(round(best_lag / bin_size)), key=key)
    moving = M["speed"][i_k] > speed_thresh
    i_n, i_k = i_n[moving], i_k[moving]
    A = M["vel_angle"][i_k]
    obs = cosine_fit(M["counts"][i_n] / bin_size, A)

    n = len(M["t"])
    rng = np.random.default_rng(seed)
    # offsets at least 40 s away from zero (and from a full wrap) so the pairing is genuinely broken
    guard = int(round(40.0 / bin_size))
    offsets = rng.integers(guard, n - guard, size=n_shuffles)
    null = np.empty((n_shuffles, M["counts"].shape[1]))
    for j, k in enumerate(tqdm(offsets, desc="block shifts", leave=False)):
        rolled = np.roll(M["counts"], k, axis=0)
        null[j] = cosine_fit(rolled[i_n] / bin_size, A)["r2"].values
    return obs, null


def summarise(name, obs, null, alpha_nominal=0.01):
    n_shuf = null.shape[0]
    # empirical p: fraction of shifts whose R^2 reaches the observed R^2
    p_emp = (null >= obs["r2"].values[None, :]).sum(0) / (n_shuf + 1)
    n_units = len(obs)
    print(f"{name}: {(obs['p'] < alpha_nominal).sum()}/{n_units} tuned by the nominal F test, "
          f"{(p_emp < 0.05).sum()}/{n_units} by the block-shift null (p_emp < 0.05)")
    print(f"   observed R^2  median {np.median(obs['r2']):.4f}  max {obs['r2'].max():.4f}")
    print(f"   shifted  R^2  median {np.median(null):.4f}  99.9th pct {np.percentile(null, 99.9):.4f}")
    return p_emp


if __name__ == "__main__":
    with open(f"{CACHE}/binned.pkl", "rb") as fh:
        maze = pickle.load(fh)
    M = maze["M"] if "M" in maze else maze
    with open(f"{CACHE}/lagsweep.pkl", "rb") as fh:
        ls = pickle.load(fh)
    print("lagsweep keys:", list(ls))
    best_lag = ls["best_lag"] if "best_lag" in ls else 0.10
    obs_m, null_m = block_shift_null(M, best_lag)
    p_m = summarise("MC_Maze", obs_m, null_m)

    with open(f"{CACHE}/rtt_binned.pkl", "rb") as fh:
        rtt = pickle.load(fh)
    Mx = rtt["M"]
    obs_r, null_r = block_shift_null(Mx, 0.08, key="seg")
    p_r = summarise("MC_RTT", obs_r, null_r)

    with open(f"{CACHE}/shuffle_control.pkl", "wb") as fh:
        pickle.dump(dict(obs_maze=obs_m, null_maze=null_m, p_maze=p_m,
                         obs_rtt=obs_r, null_rtt=null_r, p_rtt=p_r), fh)
    print("wrote shuffle_control.pkl")
