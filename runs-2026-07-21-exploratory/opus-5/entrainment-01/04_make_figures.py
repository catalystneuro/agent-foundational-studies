"""Build every figure and print the headline numbers, pooling all sessions."""

import numpy as np
import pandas as pd

import compute_session as cs
import figures as F
import hc11_io as io

results = [cs.compute(s) for s in io.SESSIONS]
df = pd.concat([r["df"] for r in results], ignore_index=True)
df.to_csv("phase_locking_all_sessions.csv", index=False)

prototype = results[0]
print(F.fig_lfp_states(prototype))
print(F.fig_spike_phase_excerpt(prototype))
print(F.fig_example_units(prototype))
print(F.fig_population_map(prototype))
print(F.fig_population_stats(df))
print(F.fig_specificity(results))
print(F.fig_sessions(df))

# ---------------------------------------------------------------- statistics
run = df[(df.state == "run") & (df.n_spikes >= F.MIN_SPIKES)]
print(f"\nsessions: {df.session.nunique()}, units analysed during running: {len(run)}")
print(f"significantly locked (circular-shift p<0.05): {(run.shuffle_p<0.05).sum()} "
      f"({(run.shuffle_p<0.05).mean()*100:.0f}%)")
bonf = 0.05 / len(run)
print(f"Rayleigh p < {bonf:.2e} (Bonferroni): "
      f"{(run.rayleigh_p < bonf).sum()} units")
for ct, sub in run.groupby("cell_type"):
    print(f"  {ct}: n={len(sub)}, median MRL={sub.mrl.median():.3f}, "
          f"locked={100*(sub.shuffle_p<0.05).mean():.0f}%, "
          f"median preferred phase="
          f"{np.degrees(np.mod(np.angle(np.exp(1j*sub.pref_phase).mean()),2*np.pi)):.0f}°")

from scipy.stats import mannwhitneyu
e = run[run.cell_type == "excitatory"].mrl
i = run[run.cell_type == "inhibitory"].mrl
u, p = mannwhitneyu(e, i)
print(f"excitatory vs inhibitory MRL: Mann-Whitney U={u:.0f}, p={p:.2e}")

amp = pd.concat([r["amp_split"] for r in results])
amp = amp[(amp.n_high >= F.MIN_SPIKES) & (amp.n_low >= F.MIN_SPIKES)]
from scipy.stats import wilcoxon
w, pw = wilcoxon(amp.mrl_high, amp.mrl_low)
print(f"MRL high- vs low-amplitude theta: n={len(amp)}, "
      f"{100*(amp.mrl_high>amp.mrl_low).mean():.0f}% higher, "
      f"Wilcoxon p={pw:.2e}")

piv = F.locked_both(df)
d = np.angle(np.exp(1j * (piv["REM"] - piv["run"])))
rho = np.abs(np.exp(1j * d).mean())
print(f"run vs REM preferred phase (units locked in both): n={len(piv)}, "
      f"concentration={rho:.2f}, median offset={np.degrees(np.median(d)):+.0f}°, "
      f"Rayleigh z={len(piv)*rho**2:.1f}")

matched = df[(df.n_matched >= F.MIN_SPIKES)]
print("\nspike-count-matched median MRL by state:")
print(matched.groupby("state").mrl_matched.median().round(3).to_string())
