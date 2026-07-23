"""Build all publication figures from results.pkl."""
import pickle
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import welch

import theta_lib as T
import theta_analysis as A
import theta_figures as F

EXAMPLE = "Achilles_10252013"

with open("results.pkl", "rb") as fh:
    results = pickle.load(fh)
res_by_session = {r["session"]: r for r in results}
ex = res_by_session[EXAMPLE]

all_cells = pd.concat([r["cells"] for r in results], ignore_index=True)
all_spikes = pd.concat([r["spikes"] for r in results], ignore_index=True)
all_entrain = pd.concat([r["entrain"] for r in results], ignore_index=True)
hists = {(r["session"], u): h for r in results for u, h in r["phase_hists"].items()}
edges = results[0]["phase_edges"]

print(f"{len(all_entrain)} units across {len(results)} sessions; "
      f"{int((all_entrain.rayleigh_p < 0.01).sum())} theta-locked (p<0.01)")
pcs = all_cells[all_cells.is_place_cell]
print(f"{len(pcs)} place fields; {int((pcs.perm_p < 0.05).sum())} with significant "
      f"precession; {100*(pcs.slope<0).mean():.0f}% negative slope")

# ---------------------------------------------------------------- figure 1
nwbfile = T.open_session(EXAMPLE)
maze = T.get_epochs(nwbfile)["MazeEpoch"]
lfp, fs = T.get_lfp(nwbfile, ex["channel"], maze)
phase, amp, filt = T.theta_phase_amp(lfp, fs)
position, dt, track_len = T.get_position(nwbfile)
runs, direction = T.run_epochs(position, dt, track_len)
units = T.get_units(nwbfile).restrict(maze)

run_ep = nap.IntervalSet(start=np.concatenate([ex["dir_eps"]["R"][:, 0],
                                               ex["dir_eps"]["L"][:, 0]]),
                         end=np.concatenate([ex["dir_eps"]["R"][:, 1],
                                             ex["dir_eps"]["L"][:, 1]]))
lfp_run = lfp.restrict(run_ep)
psd_f, psd_p = welch(lfp_run.values, fs=fs, nperseg=int(4 * fs))

F.fig_raw_streams(EXAMPLE, position, lfp, filt, phase, units, runs, direction,
                  psd_f, psd_p, 10, "fig01_raw_data_and_theta.png")
print("fig01 done")

# ---------------------------------------------------------------- figure 2
F.fig_place_fields(ex, "fig02_place_fields.png")
print("fig02 done")

# ---------------------------------------------------------------- figure 3
F.fig_entrainment(all_entrain, hists, edges, ex["wave_centers"], ex["mean_wave"],
                  "fig03_theta_entrainment.png")
print("fig03 done")

# ---------------------------------------------------------------- figure 4
F.fig_precession_examples(ex, all_spikes, "fig04_precession_examples.png", n=8)
print("fig04 done")

# ---------------------------------------------------------------- figure 5
# shuffled null for the circular-linear correlation, matched per field
rng = np.random.default_rng(1)
null_rhos = []
for (sess, uid, d), s in all_spikes.groupby(["session", "unit", "direction"]):
    ph = rng.permutation(s.phase.values)
    null_rhos.append(abs(A.circlin_fit(s.x_norm.values, ph)[2]))
null_rhos = np.array(null_rhos)
F.fig_precession_population(all_cells, all_spikes, null_rhos,
                            "fig05_precession_population.png")
print("fig05 done")

# ---------------------------------------------------------------- summary table
summary = pd.DataFrame([{
    "session": r["session"],
    "track_len_m": r["track_len"],
    "theta_ch": r["channel"],
    "theta_freq_Hz": round(r["mean_theta_freq"], 2),
    "laps": r["n_laps"],
    "run_time_s": round(r["run_time"], 1),
    "units": r["n_units"],
    "units_tested": len(r["entrain"]),
    "theta_locked_p01": int((r["entrain"].rayleigh_p < 0.01).sum()),
    "place_fields": int(r["cells"].is_place_cell.sum()),
    "precessing_p05": int((r["cells"].perm_p < 0.05).sum()),
    "median_slope": round(float(
        r["cells"][r["cells"].is_place_cell].slope.median()), 3),
} for r in results])
summary.to_csv("session_summary.csv", index=False)
pcs.to_csv("place_field_precession.csv", index=False)
all_entrain.to_csv("theta_entrainment.csv", index=False)
print(summary.to_string(index=False))
