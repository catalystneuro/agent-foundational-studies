"""Trial-based reach-direction tuning of M1/PMd units (cosine tuning)."""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import pynapple as nap
import matplotlib.pyplot as plt
from s01_load import load_all, to_pynapple

CACHE, FIG = "cache", "figures"

MOVE_WIN = (-0.10, 0.30)   # s relative to movement onset (M1 leads movement)
PLAN_WIN = (-0.40, -0.05)  # s relative to go cue = late delay / planning period


def epoch_rates(spk, starts, win):
    ep = nap.IntervalSet(start=starts + win[0], end=starts + win[1])
    cnt = spk.restrict(ep).count(ep=ep, bin_size=win[1] - win[0])
    # count with bin_size == epoch length gives one bin per epoch
    return np.asarray(cnt.values) / (win[1] - win[0]), ep


def cosine_fit(rate, theta):
    """rate ~ b0 + b1 cos(th) + b2 sin(th).  Returns dict per unit."""
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, rate, rcond=None)
    pred = X @ beta
    ss_res = np.sum((rate - pred) ** 2, 0)
    ss_tot = np.sum((rate - rate.mean(0)) ** 2, 0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    pd_ = np.arctan2(beta[2], beta[1])
    depth = np.hypot(beta[1], beta[2])
    return dict(b0=beta[0], pref_dir=pd_, depth=depth, r2=r2,
                mdi=depth / np.where(beta[0] > 0, beta[0], np.nan))


def permutation_p(rate, theta, nperm=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = cosine_fit(rate, theta)['r2']
    cnt = np.zeros_like(obs)
    for _ in range(nperm):
        cnt += cosine_fit(rate, theta[rng.permutation(len(theta))])['r2'] >= obs
    return (cnt + 1) / (nperm + 1)


if __name__ == "__main__":
    d = load_all()
    spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    theta = K.dir_end.values

    rate_mv, ep_mv = epoch_rates(spk, K.t_onset.values, MOVE_WIN)
    rate_pl, ep_pl = epoch_rates(spk, d['go_cue_time'], PLAN_WIN)
    long_delay = d['delay'] >= 450          # window must lie inside the delay period
    print("long-delay trials:", long_delay.sum())
    print("rate matrices", rate_mv.shape, rate_pl.shape)

    straight = d['num_barriers'] == 0   # unobstructed, near-straight reaches
    print("straight trials:", straight.sum())

    res_st = cosine_fit(rate_mv[straight], theta[straight])
    p_st = permutation_p(rate_mv[straight], theta[straight], 500)
    res_mv = cosine_fit(rate_mv, theta)
    res_pl = cosine_fit(rate_pl[long_delay], theta[long_delay])
    p_mv = permutation_p(rate_mv, theta, 500)
    p_pl = permutation_p(rate_pl[long_delay], theta[long_delay], 500)

    df = pd.DataFrame(dict(
        unit=np.arange(rate_mv.shape[1]), area=d['area'],
        mean_rate=rate_mv.mean(0),
        pref_dir=res_mv['pref_dir'], depth=res_mv['depth'], r2=res_mv['r2'],
        mdi=res_mv['mdi'], p=p_mv,
        pref_dir_plan=res_pl['pref_dir'], r2_plan=res_pl['r2'], p_plan=p_pl,
        pref_dir_str=res_st['pref_dir'], depth_str=res_st['depth'],
        r2_str=res_st['r2'], mdi_str=res_st['mdi'], p_str=p_st,
    ))
    df.to_pickle(f"{CACHE}/dir_tuning.pkl")
    np.save(f"{CACHE}/rate_mv.npy", rate_mv); np.save(f"{CACHE}/rate_pl.npy", rate_pl)
    np.save(f"{CACHE}/straight.npy", straight)
    np.save(f"{CACHE}/long_delay.npy", long_delay)
    for lbl, r2c, pc in [('move/all', 'r2', 'p'), ('move/straight', 'r2_str', 'p_str'),
                         ('plan/all', 'r2_plan', 'p_plan')]:
        print(f"{lbl:15s} tuned(p<0.01)={np.mean(df[pc]<0.01):.1%}  "
              f"median R2={df[r2c].median():.3f}  max R2={df[r2c].max():.3f}")
    print(df.sort_values('r2_str', ascending=False).head(8)
          [['unit','mean_rate','pref_dir_str','depth_str','r2_str','mdi_str']].to_string())
