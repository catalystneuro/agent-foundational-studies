"""Poisson-GLM STRFs for one 000986 session, fitted with NeMoS.

The reverse-correlation STRF is an unregularised, unconstrained estimate.  A
Poisson GLM with a raised-cosine temporal basis gives a smooth, regularised
STRF, separates the stimulus drive from the unit's own spike history, and can
be scored on held-out data - which is the test of whether the STRF actually
predicts the spike train.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
from tqdm import tqdm

import strf_lib as S

DT = 0.005
WINDOW = 50          # 250 ms stimulus window
N_BASIS = 9
HIST_WINDOW = 40
N_HIST = 6
SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
OUT = "glm_strf.npz"


def main():
    d = S.load_ac_session(S.asset_url("000986", SESSION))
    units, trials = d["units"], d["trials"]
    freqs = S.TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    t0, t1 = onsets[0] - 1.0, onsets[-1] + 2.0
    t_bins = np.arange(t0, t1, DT)
    # one impulse per tone onset: the GLM kernel is then directly comparable
    # with the onset-triggered reverse-correlation STRF (all tones are 25 ms)
    Sm = S.build_tone_stimulus(onsets, fidx, len(freqs), t_bins, DT)

    stim_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=WINDOW,
                                               label="tone")
    hist_basis = nmo.basis.RaisedCosineLogConv(N_HIST, window_size=HIST_WINDOW,
                                               label="history")
    Xs = np.asarray(stim_basis.compute_features(Sm))
    _, Kstim = stim_basis.evaluate_on_grid(WINDOW)

    rate = np.array([len(units[i].t) / (t1 - t0) for i in units.index])
    order = np.argsort(rate)[::-1]
    sel = [units.index[i] for i in order[:24]]

    n_t = len(t_bins)
    split = int(0.7 * n_t)
    # bins falling within 150 ms of any tone onset; the tone ensemble is sparse
    # (25 ms every ~805 ms) so a whole-session score is dominated by silence
    tone_win = Sm.sum(axis=1) > 0
    tone_win = np.convolve(tone_win, np.ones(30), mode="full")[: n_t] > 0
    print(f"tone window covers {tone_win.mean()*100:.1f}% of bins")
    results = []
    for uid in tqdm(sel, desc="GLM"):
        counts = S.bin_spikes(units[uid].t, t_bins)
        Xh = np.asarray(hist_basis.compute_features(counts[:, None]))
        X = np.hstack([Xs, Xh])
        ok = np.all(np.isfinite(X), axis=1)

        tr = ok.copy(); tr[split:] = False
        te = ok.copy(); te[:split] = False

        model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                            solver_name="LBFGS")
        model.fit(X[tr], counts[tr])
        r2 = model.score(X[te], counts[te], score_type="pseudo-r2-McFadden")
        tw = te & tone_win
        r2_win = model.score(X[tw], counts[tw], score_type="pseudo-r2-McFadden")

        # null model: spike history only
        null = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                           solver_name="LBFGS")
        null.fit(Xh[ok][: tr.sum()], counts[tr])
        r2_null = null.score(Xh[te], counts[te], score_type="pseudo-r2-McFadden")
        r2_null_win = null.score(Xh[tw], counts[tw], score_type="pseudo-r2-McFadden")

        w = np.asarray(model.coef_)[: len(freqs) * N_BASIS].reshape(len(freqs), N_BASIS)
        strf = Kstim @ w.T                      # (window, n_freq), log-rate units
        results.append(dict(unit=int(uid), strf=strf, r2=float(r2),
                            r2_null=float(r2_null), r2_win=float(r2_win),
                            r2_null_win=float(r2_null_win)))

    np.savez_compressed(
        OUT,
        strf=np.stack([r["strf"] for r in results]).astype(np.float32),
        unit=np.array([r["unit"] for r in results]),
        r2=np.array([r["r2"] for r in results]),
        r2_null=np.array([r["r2_null"] for r in results]),
        r2_win=np.array([r["r2_win"] for r in results]),
        r2_null_win=np.array([r["r2_null_win"] for r in results]),
        freqs=freqs, dt=DT, window=WINDOW,
    )
    print("saved", OUT)
    print("pseudo-R2 full  ", np.round([r["r2"] for r in results], 4))
    print("pseudo-R2 hist  ", np.round([r["r2_null"] for r in results], 4))
    print("tone-window ΔR2 ", np.round([r["r2_win"] - r["r2_null_win"] for r in results], 4))


if __name__ == "__main__":
    main()
