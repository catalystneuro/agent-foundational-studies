"""Stage 3: single-session theta phase entrainment of CA1 units.

Convention: theta phase is the Hilbert phase of the 6-10 Hz filtered LFP,
0 deg = peak of the filtered signal, 180 deg = trough.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm

import theta_utils as tu

SESSION = "Achilles-10252013"
BEST_CH = 117          # from stage 2 (max theta/delta on the maze)
MIN_SPIKES = 100
N_SHUFFLES = 200

h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)
units = meta["units"]
maze = meta["epochs"]["MazeEpoch"]

# -------------------------------------------------------------- state epochs
pos = meta["position"].restrict(maze)
speed = tu.compute_speed(pos, rate=meta["position_rate"])
run = speed.threshold(10.0).time_support.drop_short_intervals(1.0)
post = meta["epochs"]["POSTEpoch"]
rem = meta["states"]["REM"].intersect(post).drop_short_intervals(20.0)
nrem = meta["states"]["Non-REM"].intersect(post).drop_short_intervals(20.0)

print(f"RUN {run.tot_length():.0f}s / REM {rem.tot_length():.0f}s / "
      f"nonREM {nrem.tot_length():.0f}s available")

state_epochs = {"RUN": run, "REM": rem, "nonREM": nrem}
phases, amps, raws, epochs = {}, {}, {}, {}
for name, ep in state_epochs.items():
    ph, am, raw, kept = tu.phase_by_interval(h, BEST_CH, ep, max_total=1200.0)
    phases[name], amps[name], raws[name], epochs[name] = ph, am, raw, kept
    print(f"{name}: {kept.tot_length():.0f} s of LFP over {len(kept)} segments")

lookups = {k: tu.PhaseLookup(v) for k, v in phases.items()}

# ----------------------------------------------------- per-unit phase locking
rows = []
rng = np.random.default_rng(1)
cell_type = units.cell_type.values
for i, uid in enumerate(tqdm(units.index, desc="units")):
    st_all = units[uid].t
    row = {"session": SESSION, "unit": uid, "cell_type": cell_type[i]}
    for name in state_epochs:
        ph = lookups[name](st_all)
        n = len(ph)
        r, z, p = tu.rayleigh_test(ph) if n >= MIN_SPIKES else (np.nan,) * 3
        row[f"n_{name}"] = n
        row[f"mrl_{name}"] = r
        row[f"p_{name}"] = p
        row[f"ppc_{name}"] = tu.ppc(ph) if n >= MIN_SPIKES else np.nan
        row[f"pref_{name}"] = tu.circ_mean(ph) if n >= MIN_SPIKES else np.nan
    # jitter control on RUN only
    ph_run = lookups["RUN"](st_all)
    if len(ph_run) >= MIN_SPIKES:
        null = tu.jitter_null_mrl(st_all, lookups["RUN"], n_shuffles=N_SHUFFLES,
                                  rng=rng)
        row["mrl_null_mean"] = null.mean()
        row["mrl_null_p95"] = np.percentile(null, 95)
        row["mrl_z"] = (row["mrl_RUN"] - null.mean()) / null.std()
        row["p_shuffle"] = (np.sum(null >= row["mrl_RUN"]) + 1) / (N_SHUFFLES + 1)
    rows.append(row)

res = pd.DataFrame(rows)
for name in state_epochs:
    ok = res[f"n_{name}"] >= MIN_SPIKES
    res[f"sig_{name}"] = False
    res.loc[ok, f"sig_{name}"] = tu.benjamini_hochberg(res.loc[ok, f"p_{name}"].values)
res.to_csv("phase_locking_single_session.csv", index=False)

incl = res["n_RUN"] >= MIN_SPIKES
print("\n=== RUN epochs, %d units with >=%d spikes ===" % (incl.sum(), MIN_SPIKES))
print("significantly phase locked (Rayleigh, FDR 0.05): %d/%d (%.0f%%)"
      % (res.loc[incl, "sig_RUN"].sum(), incl.sum(),
         100 * res.loc[incl, "sig_RUN"].mean()))
print("beats jitter null (p<0.05):                      %d/%d"
      % ((res.loc[incl, "p_shuffle"] < 0.05).sum(), incl.sum()))
for ct in ["excitatory", "inhibitory"]:
    m = incl & (res.cell_type == ct)
    print(f"  {ct:11s} n={m.sum():3d}  median MRL={res.loc[m,'mrl_RUN'].median():.3f}"
          f"  preferred phase={np.degrees(tu.circ_mean(res.loc[m & res.sig_RUN,'pref_RUN'])):6.1f} deg")
for name in state_epochs:
    m = res[f"n_{name}"] >= MIN_SPIKES
    print("%-7s median MRL=%.3f  locked=%d/%d" %
          (name, res.loc[m, f"mrl_{name}"].median(),
           res.loc[m, f"sig_{name}"].sum(), m.sum()))

np.save("run_epochs.npy", np.c_[run.start, run.end])
print("\nsaved phase_locking_single_session.csv")
