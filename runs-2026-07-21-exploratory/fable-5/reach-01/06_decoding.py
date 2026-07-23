"""Population-level readout: Georgopoulos population vector + continuous velocity decoding."""
import numpy as np
import pynapple as nap
import mcmaze_io as mio

nap.nap_config.suppress_conversion_warnings = True

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)

td = np.load("trial_data.npz")
fits = np.load("tuning_fits.npz")
LAG = float(td["best_lag"])
trial_rate, base_rate = td["trial_rate"], td["base_rate"]
reach_dir, peak_speed = td["reach_dir"], td["peak_speed"]
pdir, b1, pval = fits["pdir"], fits["b1"], fits["pval"]

# ------------------------------------------------- 1. population vector (trials)
# Leave-one-trial-out: preferred directions are refit without the test trial so
# the decode is honest.
sel = pval < 0.01
print(f"using {sel.sum()} direction-tuned units for the population vector")


def cosine_pd(theta, R):
    """Vectorised cosine fit across units; returns (pd, b1)."""
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, R, rcond=None)
    return np.arctan2(beta[2], beta[1]) % (2 * np.pi), np.hypot(beta[1], beta[2]), beta[0]


R = trial_rate[:, sel]
n_trials = len(reach_dir)
# 10-fold cross-validation over trials
folds = np.arange(n_trials) % 10
dec_angle = np.full(n_trials, np.nan)
dec_len = np.full(n_trials, np.nan)
for f in range(10):
    tr, te = folds != f, folds == f
    pd_tr, b1_tr, b0_tr = cosine_pd(reach_dir[tr], R[tr])
    # normalised, baseline-subtracted activity weighted by each unit's PD vector
    w = (R[te] - b0_tr) / np.where(b1_tr > 0, b1_tr, np.nan)
    vx = np.nansum(w * np.cos(pd_tr), axis=1)
    vy = np.nansum(w * np.sin(pd_tr), axis=1)
    dec_angle[te] = np.arctan2(vy, vx) % (2 * np.pi)
    dec_len[te] = np.hypot(vx, vy)

err = np.degrees((dec_angle - reach_dir + np.pi) % (2 * np.pi) - np.pi)
print(f"population vector: median |error| = {np.median(np.abs(err)):.1f} deg, "
      f"{100*np.mean(np.abs(err) < 45):.0f}% within 45 deg")

# does population vector length track reach speed?
r_len = np.corrcoef(dec_len, peak_speed)[0, 1]
print(f"corr(population vector length, peak speed) = {r_len:.3f}")

# -------------------------------------- 2. continuous velocity decoding (ridge)
BIN = 0.02
counts = spikes.count(BIN, ep=trials_ep)
rate_sm = counts.smooth(std=0.05, size_factor=10) / BIN
vel = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values, columns=["vx", "vy"]).restrict(trials_ep)
Vtrue = vel.interpolate(counts, ep=counts.time_support).values
Xall = rate_sm.values
ok = np.isfinite(Xall).all(1) & np.isfinite(Vtrue).all(1)
Xall, Vtrue = Xall[ok], Vtrue[ok]
tvec = counts.t[ok]

block = np.arange(len(Xall)) // 500
fold = block % 5
pred = np.zeros_like(Vtrue)
for f in range(5):
    tr, te = fold != f, fold == f
    Xt = np.column_stack([np.ones(tr.sum()), Xall[tr]])
    lam = 1.0
    A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
    A[0, 0] -= lam
    beta = np.linalg.solve(A, Xt.T @ Vtrue[tr])
    pred[te] = np.column_stack([np.ones(te.sum()), Xall[te]]) @ beta

r2 = 1 - ((Vtrue - pred) ** 2).sum(0) / ((Vtrue - Vtrue.mean(0)) ** 2).sum(0)
cc = [np.corrcoef(Vtrue[:, i], pred[:, i])[0, 1] for i in range(2)]
print(f"continuous decoding R2: vx={r2[0]:.3f} vy={r2[1]:.3f}; r: {cc[0]:.3f}, {cc[1]:.3f}")

# decoded direction/speed error during movement
sp_true = np.hypot(*Vtrue.T)
mv = sp_true > 100
ang_t = np.arctan2(Vtrue[:, 1], Vtrue[:, 0])
ang_p = np.arctan2(pred[:, 1], pred[:, 0])
ang_err = np.degrees((ang_p - ang_t + np.pi) % (2 * np.pi) - np.pi)
sp_pred = np.hypot(*pred.T)
r_speed = np.corrcoef(sp_true[mv], sp_pred[mv])[0, 1]
print(f"moving bins: median |angle error| = {np.median(np.abs(ang_err[mv])):.1f} deg; "
      f"corr(true speed, decoded speed) = {r_speed:.3f}")

np.savez("decoding_results.npz", dec_angle=dec_angle, dec_len=dec_len, err=err,
         reach_dir=reach_dir, peak_speed=peak_speed, r_len=r_len,
         t=tvec, Vtrue=Vtrue, pred=pred, r2=r2, ang_err=ang_err, mv=mv,
         sp_true=sp_true, sp_pred=sp_pred, r_speed=r_speed)
print("saved decoding_results.npz")
