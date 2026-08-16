"""Step 5: NeMoS Poisson GLM of the tone response for one session.

Writes results/glm_LA11_ses-1.csv and results/glm_filters.npz
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import EXAMPLE_SESSION, FIGDIR, FREQS, FREQ_LABELS, RESDIR, load_session
from glm import BIN, WINDOW, design, filters, fit_unit, glm_tuning
from tuning import analyze_session

N_FIT = 30      # units fit with the GLM (most strongly tuned first)

os.makedirs(FIGDIR, exist_ok=True)
name, asset = EXAMPLE_SESSION
sess = load_session(name, asset, with_behavior=False)
units, tmean, _, _ = analyze_session(name, asset, sess=sess)
counts, X_stim, stim_basis, hist_basis = design(sess["spikes"], sess["trials"])
print(f"{counts.shape[0]} bins of {BIN * 1e3:.0f} ms, "
      f"{X_stim.shape[1]} stimulus features")

tuned = units[units.freq_tuned].sort_values("q_frequency")
fit_ids = list(tuned.unit.values[:N_FIT])
# Keep the three example units from figure 2 in the set.
for b in [0, 2, 4]:
    cand = tuned[tuned.bf_idx == b].sort_values("best_hz", ascending=False)
    if len(cand) and int(cand.iloc[0].unit) not in fit_ids:
        fit_ids.append(int(cand.iloc[0].unit))

rows, all_filters, hist_filters = [], {}, {}
for u in tqdm(fit_ids, desc="GLM fits"):
    col = int(np.flatnonzero(np.asarray(sess["spikes"].index) == u)[0])
    model, X, ok = fit_unit(counts[:, col], X_stim, hist_basis)
    kf, kh = filters(model, stim_basis, hist_basis)
    tune, base, lags = glm_tuning(model, kf)
    all_filters[u], hist_filters[u] = kf, kh
    y = np.asarray(counts[:, col], dtype=float)
    r2 = float(model.score(X[ok], y[ok], score_type="pseudo-r2-McFadden"))
    emp = tmean[:, col]
    rows.append(dict(unit=u, pseudo_r2=r2, glm_baseline_hz=base,
                     glm_bf_hz=FREQS[int(np.argmax(tune))],
                     emp_bf_hz=FREQS[int(np.argmax(emp))],
                     r_glm_emp=float(np.corrcoef(tune, emp)[0, 1]),
                     **{f"glm_{int(f)}": t for f, t in zip(FREQS, tune)},
                     **{f"emp_{int(f)}": e for f, e in zip(FREQS, emp)}))

glm_df = pd.DataFrame(rows)
glm_df.to_csv(f"{RESDIR}/glm_{name}.csv", index=False)
np.savez(f"{RESDIR}/glm_filters.npz",
         units=np.array(fit_ids),
         freq_filters=np.stack([all_filters[u] for u in fit_ids]),
         hist_filters=np.stack([hist_filters[u] for u in fit_ids]))
match = (glm_df.glm_bf_hz == glm_df.emp_bf_hz).mean()
print(f"median pseudo-R2 {glm_df.pseudo_r2.median():.3f}")
print(f"GLM and window-average BF agree for {100 * match:.0f}% of {len(glm_df)} units")
print(f"median r(GLM, empirical tuning) = {glm_df.r_glm_emp.median():.2f}")
