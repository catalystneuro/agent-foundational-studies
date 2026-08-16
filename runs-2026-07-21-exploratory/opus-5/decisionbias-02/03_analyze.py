"""Main analysis: can the upcoming decision bias be read out before the stimulus?

Three questions, in order of how directly they answer the title:

A. Is the *prior block* (which biases the upcoming decision) decodable from
   spike counts in the 400 ms before Gabor onset?
B. Is the *upcoming choice on zero-evidence trials* -- where the choice is
   nothing but the bias -- decodable from that same pre-stimulus window?
C. Does the pre-stimulus neural read-out predict the animal's choice on
   near-zero-contrast trials over and above the block identity and the
   behavioural history, i.e. does it track the bias trial by trial?

Plus the controls that decide whether any of this is interesting: movement and
arousal in the pre-stimulus window, and behavioural history.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import decoding as dec

DATA = "session_data"
RES = "results"
MIN_TRIALS = 150
MIN_UNITS = 10
N_NULL_MAIN = 500
N_NULL_TIME = 100


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_sessions():
    out = {}
    for f in sorted(glob.glob(f"{DATA}/*.npz")):
        sid = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=True)
        tr = pd.read_csv(f"{DATA}/{sid}_trials.csv")
        out[sid] = dict(
            counts=z["counts"], centers=z["centers"], regions=z["regions"],
            locations=z["locations"], firing_rate=z["firing_rate"], trials=tr,
        )
    return out


def window_sum(d, lo, hi):
    """Sum spike counts over the non-overlapping 200 ms windows tiling [lo, hi]."""
    c, half = d["centers"], 0.10
    step = int(round(2 * half / (c[1] - c[0])))  # stride giving abutting windows
    sel = [i for i, x in enumerate(c) if x - half >= lo - 1e-6 and x + half <= hi + 1e-6]
    sel = sel[::step]
    assert sel, f"no window fits inside [{lo}, {hi}]"
    return d["counts"][:, :, sel].sum(-1)


def history_matrix(tr):
    """Behavioural-history regressors available before the stimulus appears."""
    prev_choice = tr["prev_choice_right"].to_numpy()
    prev_rew = tr["prev_rewarded"].to_numpy()
    prev_stim = tr["prev_stim_right"].to_numpy()
    # running average of the last 10 choices: the best purely behavioural
    # estimate of which block the animal is in
    ch = tr["choice_right"].to_numpy().astype(float)
    run = pd.Series(ch).shift(1).rolling(10, min_periods=1).mean().to_numpy()
    H = np.column_stack([prev_choice, prev_rew, prev_stim, prev_choice * prev_rew, run])
    return np.nan_to_num(H, nan=0.5)


# --------------------------------------------------------------------------
# A. Block decoding, per session and per region
# --------------------------------------------------------------------------

def analyse_population(X, y, tag, n_null=N_NULL_MAIN, null_kind="pseudo", seed=0):
    real = dec.cv_auc(X, y)
    null = dec.null_aucs(X, y, n_null=n_null, kind=null_kind, rng=seed)
    return dict(
        **tag, n_trials=len(y), n_units=X.shape[1], auc=real,
        null_mean=float(null.mean()), null_sd=float(null.std()),
        z=dec.null_z(real, null), p=dec.perm_p(real, null), null=null,
    )


def group_permutation(res):
    """Group-level test on the mean AUC across sessions.

    Averaging the i-th surrogate draw across sessions gives a null distribution
    for the group mean.  This has more power than combining noisy per-session
    z-scores, because the surrogate draws are independent across sessions and
    the averaging cancels their variance the same way it does for the real data.
    """
    nulls = [np.asarray(r) for r in res["null"]]
    n = min(len(x) for x in nulls)
    mat = np.stack([x[:n] for x in nulls])          # sessions x draws
    group_null = mat.mean(0)
    real = res["auc"].mean()
    return dict(mean_auc=real, null_mean=float(group_null.mean()),
                null_sd=float(group_null.std()), z=dec.null_z(real, group_null),
                p=dec.perm_p(real, group_null), n_sessions=len(nulls), n_draws=n)


def block_decoding(sessions):
    jobs = []
    for sid, d in sessions.items():
        tr = d["trials"]
        y = tr["block_right"].to_numpy()
        if len(y) < MIN_TRIALS or len(np.unique(y)) < 2:
            continue
        X_all = window_sum(d, -0.4, 0.0)
        jobs.append((X_all, y, dict(session=sid, region="ALL", window="pre")))
        # an even earlier window, to show this is not a stimulus-anticipation edge effect
        jobs.append((window_sum(d, -1.0, -0.6), y,
                     dict(session=sid, region="ALL", window="early")))
        for reg in np.unique(d["regions"]):
            if reg in ("unknown", "other"):
                continue
            m = d["regions"] == reg
            if m.sum() < MIN_UNITS:
                continue
            jobs.append((X_all[:, m], y, dict(session=sid, region=reg, window="pre")))

    res = Parallel(n_jobs=-1, verbose=5)(
        delayed(analyse_population)(X, y, tag, seed=i) for i, (X, y, tag) in enumerate(jobs)
    )
    return pd.DataFrame(res)


# --------------------------------------------------------------------------
# Time-resolved block decoding
# --------------------------------------------------------------------------

def time_resolved(sessions):
    rows = []
    jobs = []
    for sid, d in sessions.items():
        tr = d["trials"]
        y = tr["block_right"].to_numpy()
        if len(y) < MIN_TRIALS or len(np.unique(y)) < 2:
            continue
        for wi, c in enumerate(d["centers"]):
            jobs.append((sid, float(c), d["counts"][:, :, wi], y))

    def one(sid, c, X, y, seed):
        real = dec.cv_auc(X, y)
        null = dec.null_aucs(X, y, n_null=N_NULL_TIME, kind="pseudo", rng=seed)
        return dict(session=sid, center=c, auc=real, null_mean=float(null.mean()),
                    z=dec.null_z(real, null))

    res = Parallel(n_jobs=-1, verbose=5)(
        delayed(one)(sid, c, X, y, i) for i, (sid, c, X, y) in enumerate(jobs)
    )
    return pd.DataFrame(res)


# --------------------------------------------------------------------------
# B. Upcoming-choice decoding on low-evidence trials
# --------------------------------------------------------------------------

def choice_decoding(sessions, max_contrast=0.0625):
    rows = []
    for sid, d in tqdm(sessions.items(), desc="choice decoding"):
        tr = d["trials"]
        m = (tr["abs_contrast"] <= max_contrast).to_numpy()
        y = tr.loc[m, "choice_right"].to_numpy()
        if m.sum() < 50 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 12:
            continue
        X = window_sum(d, -0.4, 0.0)[m]
        real = dec.cv_auc(X, y)
        null = dec.null_aucs(X, y, n_null=N_NULL_MAIN, kind="shift", rng=hash(sid) % 1000)
        rows.append(dict(session=sid, n_trials=int(m.sum()), n_units=X.shape[1],
                         auc=real, null_mean=float(null.mean()), z=dec.null_z(real, null),
                         p=dec.perm_p(real, null), null=null))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Controls: movement / arousal, and behavioural history
# --------------------------------------------------------------------------

def negative_control(sessions):
    """Decode something the animal cannot know: the upcoming stimulus contrast.

    Contrast is drawn independently of the block on every trial, so a pipeline
    that is not manufacturing above-chance performance must return AUC ~ 0.5
    here while returning AUC > 0.5 for the block.
    """
    rows = []
    for sid, d in tqdm(sessions.items(), desc="negative control"):
        tr = d["trials"]
        y = (tr["abs_contrast"] >= 0.25).astype(int).to_numpy()
        if len(y) < MIN_TRIALS or min(np.bincount(y)) < 40:
            continue
        X = window_sum(d, -0.4, 0.0)
        real = dec.cv_auc(X, y)
        null = dec.null_aucs(X, y, n_null=100, kind="shift", rng=2)
        rows.append(dict(session=sid, auc=real, null_mean=float(null.mean()),
                         z=dec.null_z(real, null), p=dec.perm_p(real, null)))
    return pd.DataFrame(rows)


def control_decoding(sessions):
    rows = []
    for sid, d in tqdm(sessions.items(), desc="controls"):
        tr = d["trials"]
        y = tr["block_right"].to_numpy()
        if len(y) < MIN_TRIALS or len(np.unique(y)) < 2:
            continue
        beh = tr[["wheel_abs_vel", "wheel_range", "motion_energy", "pupil"]]
        beh = beh.fillna(beh.mean()).fillna(0.0).to_numpy()
        H = history_matrix(tr)
        Xn = window_sum(d, -0.4, 0.0)

        auc_beh = dec.cv_auc(beh, y)
        auc_hist = dec.cv_auc(H, y)
        auc_neu = dec.cv_auc(Xn, y)

        # Does the population add to the behavioural-history model?  Adding
        # several hundred raw spike-count features to a five-feature model would
        # lose on regularisation alone, so the population enters as a single
        # stacked feature: its out-of-fold decoder read-out.
        p_neu = dec.oof_predict(Xn, y)
        p_neu = np.nan_to_num(p_neu, nan=np.nanmean(p_neu))
        auc_hist_neu = dec.cv_auc(np.column_stack([H, p_neu]), y)

        # Null: circularly shift the neural read-out only, which breaks its
        # relation to the block while leaving the history features aligned.
        d_real = auc_hist_neu - auc_hist
        d_null = np.array([
            dec.cv_auc(np.column_stack([H, np.roll(p_neu, s)]), y) - auc_hist
            for s in dec.circular_shifts(len(y), 200, rng=1)
        ])
        rows.append(dict(session=sid, auc_movement=auc_beh, auc_history=auc_hist,
                         auc_neural=auc_neu, auc_history_neural=auc_hist_neu,
                         delta=d_real, delta_null_mean=float(d_null.mean()),
                         delta_null_sd=float(d_null.std()),
                         z_delta=dec.null_z(d_real, d_null),
                         p_delta=dec.perm_p(d_real, d_null)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. Does the pre-stimulus read-out track the bias trial by trial?
# --------------------------------------------------------------------------

def trialwise_bias(sessions, max_contrast=0.0625):
    """Logistic model of choice on low-evidence trials: block + history + neural.

    The neural regressor is the out-of-fold pre-stimulus block-decoder output,
    so it cannot be fit to the choices it is being used to predict.
    """
    rows, psycho = [], []
    for sid, d in tqdm(sessions.items(), desc="trialwise"):
        tr = d["trials"]
        y_block = tr["block_right"].to_numpy()
        if len(y_block) < MIN_TRIALS or len(np.unique(y_block)) < 2:
            continue
        X = window_sum(d, -0.4, 0.0)
        p_neu = dec.oof_predict(X, y_block)

        m = ((tr["abs_contrast"] <= max_contrast) & np.isfinite(p_neu)).to_numpy()
        if m.sum() < 60:
            continue
        ch = tr.loc[m, "choice_right"].to_numpy()
        if len(np.unique(ch)) < 2:
            continue
        prev = np.nan_to_num(tr.loc[m, "prev_choice_right"].to_numpy(), nan=0.5)
        sc = tr.loc[m, "signed_contrast"].to_numpy()
        design = pd.DataFrame({
            "block": y_block[m].astype(float),
            "prev_choice": prev,
            "signed_contrast": sc,
            "neural": stats.zscore(p_neu[m]),
        })
        import statsmodels.api as sm
        model = sm.Logit(ch, sm.add_constant(design)).fit(disp=0, method="bfgs", maxiter=200)
        if abs(model.params["block"]) > 10:
            # quasi-separation: the block predicts choice perfectly on these
            # trials and the fit is not interpretable
            continue
        rows.append(dict(session=sid, n=int(m.sum()),
                         beta_neural=model.params["neural"],
                         se_neural=model.bse["neural"],
                         p_neural=model.pvalues["neural"],
                         beta_block=model.params["block"]))

        # Psychometric curves split by the pre-stimulus read-out.  The read-out
        # is z-scored *within block*, so the split is orthogonal to block
        # identity and reflects trial-to-trial fluctuation of the signal.
        p_within = pd.Series(p_neu).groupby(pd.Series(y_block)).transform(
            lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
        ).to_numpy()
        sel = np.isfinite(p_within) & (tr["abs_contrast"] <= 0.25).to_numpy()
        q = pd.qcut(p_within[sel], 3, labels=["low", "mid", "high"])
        psycho.append(pd.DataFrame(dict(
            session=sid, signed_contrast=tr.loc[sel, "signed_contrast"].to_numpy(),
            choice_right=tr.loc[sel, "choice_right"].to_numpy(),
            block_right=y_block[sel], neural_group=q.astype(str))))
    return pd.DataFrame(rows), pd.concat(psycho, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    sessions = load_sessions()
    print(f"{len(sessions)} sessions loaded")

    print("\n== A. block decoding ==")
    block = block_decoding(sessions)
    allpre = block[(block.region == "ALL") & (block.window == "pre")]
    early = block[(block.region == "ALL") & (block.window == "early")]
    print(allpre[["session", "n_units", "n_trials", "auc", "null_mean", "z", "p"]].to_string())
    print(f"Stouffer Z (all-unit populations, pre window): {dec.stouffer(allpre.z):.2f}")
    grp = pd.DataFrame([dict(analysis="block_pre", **group_permutation(allpre)),
                        dict(analysis="block_early", **group_permutation(early))])
    print(grp.to_string())
    block.drop(columns="null").to_csv(f"{RES}/block_decoding.csv", index=False)

    print("\n== time-resolved ==")
    tres = time_resolved(sessions)
    tres.to_csv(f"{RES}/time_resolved.csv", index=False)

    print("\n== B. choice decoding on low-evidence trials ==")
    ch = choice_decoding(sessions)
    print(ch.drop(columns="null").to_string())
    print(f"Stouffer Z: {dec.stouffer(ch.z):.2f}")
    grp = pd.concat([grp, pd.DataFrame([dict(analysis="choice_low_contrast",
                                             **group_permutation(ch))])],
                    ignore_index=True)
    ch.drop(columns="null").to_csv(f"{RES}/choice_decoding.csv", index=False)

    print("\n== B2. choice decoding, zero-contrast trials only ==")
    ch0 = choice_decoding(sessions, max_contrast=0.0)
    print(f"{len(ch0)} sessions with enough zero-contrast trials, "
          f"mean AUC {ch0.auc.mean():.3f}, Stouffer Z {dec.stouffer(ch0.z):.2f}")
    grp = pd.concat([grp, pd.DataFrame([dict(analysis="choice_zero_contrast",
                                             **group_permutation(ch0))])],
                    ignore_index=True)
    ch0.drop(columns="null").to_csv(f"{RES}/choice_decoding_zero.csv", index=False)
    grp.to_csv(f"{RES}/group_tests.csv", index=False)
    print(grp.to_string())

    print("\n== negative control: upcoming contrast ==")
    neg = negative_control(sessions)
    neg.to_csv(f"{RES}/negative_control.csv", index=False)
    print(neg.to_string())
    print(f"mean AUC {neg.auc.mean():.3f}, Stouffer Z {dec.stouffer(neg.z):.2f}")

    print("\n== controls ==")
    ctl = control_decoding(sessions)
    ctl.to_csv(f"{RES}/controls.csv", index=False)
    print(ctl.to_string())

    print("\n== C. trial-wise bias ==")
    tw, psycho = trialwise_bias(sessions)
    tw.to_csv(f"{RES}/trialwise_bias.csv", index=False)
    psycho.to_csv(f"{RES}/psycho_by_readout.csv", index=False)
    print(tw.to_string())
    if len(tw):
        t, p = stats.ttest_1samp(tw.beta_neural, 0)
        print(f"beta_neural across sessions: t={t:.2f}, p={p:.4f}")
