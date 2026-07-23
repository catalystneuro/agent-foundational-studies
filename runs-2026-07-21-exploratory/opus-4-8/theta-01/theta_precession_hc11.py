# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta phase entrainment and phase precession in hippocampal CA1
#
# **Data:** [DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki
# (2016), *Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences* (the `hc-11` dataset). Bilateral silicon-probe recordings from
# dorsal CA1 of freely moving rats, with 1250 Hz LFP on 128 channels, spike-sorted units
# labelled as excitatory or inhibitory, and tracked position on a 1.6 m linear track.
#
# **What this notebook demonstrates.** Two related phenomena that together make up the
# theta-phase code of the hippocampus:
#
# 1. **Theta phase entrainment.** During running, CA1 spiking is locked to the ongoing
#    6-12 Hz theta rhythm in the local field potential. Individual cells fire
#    preferentially at a particular phase of the cycle, and their spike trains are
#    themselves rhythmic at theta frequency.
# 2. **Theta phase precession.** As the animal crosses a place field, the cell's spikes
#    occur at systematically earlier theta phases. The firing phase therefore carries
#    information about position *within* the field that the firing rate alone does not.
#
# The analysis runs over the four sessions of the dandiset recorded on the 1.6 m linear
# track (the other two assets use a circular maze whose linearization is not comparable).
# Data are streamed from the DANDI S3 bucket with LINDI and a local cache; nothing is
# downloaded in full. All time-series handling is done with Pynapple.
#
# **Phase convention throughout:** theta phase is the Hilbert phase of the 6-12 Hz
# bandpassed LFP, so 0/360 degrees is the *peak* of the filtered signal and 180 degrees is
# the trough.

# %%
import pickle

import matplotlib
matplotlib.use("Agg")            # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import theta_lib as tl
import analysis
import plots

pd.set_option("display.width", 160)
print("sessions to analyze:")
for subject, label, aid in tl.SESSIONS:
    print(f"  {subject:10s} {label}")

# %% [markdown]
# ## 1. Loading one session and checking every data stream
#
# `analysis.analyze_session` performs the whole per-session pipeline. Before trusting any
# of it, we look at what it loaded.
#
# Two properties of this particular NWB conversion are worth flagging, because they are
# easy to get wrong:
#
# * The behaviour `SpatialSeries` stores the sampling **period** (0.0256 s) in its `rate`
#   field rather than the rate. Taking it at face value would place the position samples
#   several hours away from the spikes. Reading it as a period reproduces the
#   `MazeEpoch` duration exactly, which is how we verified the interpretation.
# * The linearized position is only defined (non-NaN) while the animal is actually
#   traversing the track; it is NaN while the animal sits at the reward ends. The
#   contiguous non-NaN blocks therefore *are* the individual runs, and we use them
#   directly as the run epochs after requiring a near-monotonic traversal of at least 1 m.

# %%
subject, label, aid = tl.SESSIONS[0]
res = analysis.analyze_session(subject, label, aid)
df = res["df"]
pc = analysis.place_cell_mask(df)
df["is_place_cell"] = pc

print(f"\nunits: {len(res['units'])}  ({df.cell_type.value_counts().to_dict()})")
print(f"track traversals: {len(res['runs'])}  "
      f"(rightward {(res['direction'] > 0).sum()}, leftward {(res['direction'] < 0).sum()})")
print(f"selected LFP channel: {res['best_ch']};  LFP theta peak: {res['lfp_theta_freq']:.2f} Hz")
print(f"place cells: {int(pc.sum())}")

# %% [markdown]
# ### Choosing an LFP channel
#
# The probes span the CA1 layers and beyond, so theta amplitude varies a lot across the
# 128 channels. We pick the channel with the largest theta (6-12 Hz) to delta (1-4 Hz)
# power ratio over a 60 s window of running, which reliably lands near the pyramidal
# layer / stratum lacunosum-moleculare where theta is strongest.

# %%
plots.fig_channel_selection(res)
plt.close("all")

# %% [markdown]
# ![channel selection](figures/00_channel_selection.png)

# %% [markdown]
# ### Raw data check
#
# Raw LFP with the theta-filtered signal on top, the linearized position with the detected
# traversals shaded, and a two-second zoom showing individual theta cycles and the Hilbert
# phase ramp. Theta is unmistakable during running.

# %%
plots.fig_raw_streams(res)
plt.close("all")

# %% [markdown]
# ![raw streams](figures/01_raw_streams.png)

# %% [markdown]
# ## 2. Place fields
#
# Tuning curves are computed with `nap.compute_1d_tuning_curves` over 50 spatial bins,
# separately for rightward and leftward runs, since CA1 fields on a linear track are
# strongly direction-selective. A unit counts as a place cell if it is labelled
# excitatory, has Skaggs spatial information of at least 0.5 bits/spike, a peak rate of at
# least 1 Hz in a bin the animal actually occupied for 0.5 s or more, at least 50
# in-field spikes, and a mean on-track rate below 8 Hz.
#
# The place-field boundaries used for the precession analysis are the contiguous bins
# around the peak that stay above 20% of the peak rate.

# %%
plots.fig_place_fields(res, pc)
plt.close("all")
print(df[pc][["unit", "direction", "peak_pos", "field_width", "spatial_info",
              "peak_rate", "n_field_spikes"]].head(10).to_string(index=False))

# %% [markdown]
# ![place fields](figures/02_place_fields.png)
#
# The population tiles the track: sorted by peak position, the place fields form a clean
# diagonal in both running directions. Many cells have a field in only one direction.

# %% [markdown]
# ## 3. Theta phase entrainment
#
# For every unit we take the theta phase at each of its spike times during running
# (`TsGroup.value_from` on the phase `Tsd`), then compute the mean resultant length (MRL),
# the preferred phase, and a Rayleigh test of non-uniformity.

# %%
plots.fig_phase_locking(res, pc)
plt.close("all")

# %% [markdown]
# ![phase locking](figures/03_theta_phase_locking.png)
#
# Both the example place cell and the example interneuron fire on a restricted portion of
# the theta cycle. Across the session, almost all interneurons and most pyramidal cells
# are significantly locked, with pyramidal preferred phases clustered near the descending
# phase / trough of the filtered LFP.

# %% [markdown]
# ### Theta rhythmicity of the spike trains
#
# Entrainment should also show up without any reference to the LFP: the spike
# autocorrelogram of a theta-modulated cell has side peaks at multiples of the theta
# period. Comparing where those peaks fall against the LFP theta period is informative,
# because the dual-oscillator account of phase precession predicts that place cells
# oscillate slightly *faster* than the LFP.

# %%
mods, intrinsic, f_lfp = plots.fig_theta_rhythmicity(
    res, pc, fname=f"04_theta_rhythmicity_{label}.png")
plt.close("all")
print(f"LFP theta: {f_lfp:.2f} Hz;  median intrinsic place-cell frequency: "
      f"{np.median(intrinsic['place cells']):.2f} Hz")

# %% [markdown]
# ![rhythmicity](figures/04_theta_rhythmicity_Achilles-10252013.png)

# %% [markdown]
# ## 4. Theta phase precession
#
# For each place cell we take the spikes that fall inside its field on runs in its
# preferred direction, express position as a fraction of field width *along the direction
# of travel*, and fit the circular-linear regression of Kempter et al. (2012): the slope
# is chosen to maximize the resultant length of the phase residuals, and the strength of
# the relationship is summarized by the circular-linear correlation coefficient rho.
#
# Significance is assessed against a permutation null in which spike phases are shuffled
# against positions 200 times, preserving both marginal distributions. A cell counts as
# precessing if its observed |rho| exceeds the shuffled |rho| in at least 95% of
# permutations.

# %%
plots.fig_precession_examples(res, pc)
plt.close("all")

# %% [markdown]
# ![examples](figures/05_precession_examples.png)
#
# Each column is one cell: the place field on top, and the phase of every in-field spike
# against position below, plotted over two theta cycles so the wrap-around is visible.
# The red lines are the fitted circular-linear regression, repeated at 360 degree offsets.
# Phase falls by roughly 200-360 degrees over a single field traversal.

# %% [markdown]
# ### Precession within single traversals
#
# The pooled scatter above could in principle arise from a slow phase drift across the
# session rather than from a within-pass effect. Looking at individual traversals rules
# that out: each pass through the field shows its own descending phase ramp.

# %%
plots.fig_single_runs(res, pc)
plt.close("all")

# %% [markdown]
# ![single runs](figures/07_single_run_precession.png)

# %% [markdown]
# ## 5. Running the pipeline over all four linear-track sessions
#
# The prototype above is now applied to every session. This streams roughly 2000 s of
# single-channel LFP plus all spike times per session and takes a few minutes.

# %%
all_df, all_prec = [], []
mods_pooled = {"place cells": [], "interneurons": []}
intr_pooled = {"place cells": [], "interneurons": []}
lfp_freqs = []

for subject, label, aid in tqdm(tl.SESSIONS, desc="sessions"):
    r = res if label == tl.SESSIONS[0][1] else analysis.analyze_session(subject, label, aid)
    d = r["df"]
    m = analysis.place_cell_mask(d)
    d["is_place_cell"] = m
    for p in r["precession"]:
        p["session"] = label
    keep = set(d.loc[m, "unit"])
    all_prec += [p for p in r["precession"] if p["unit"] in keep]
    all_df.append(d)

    mm, ii, ff = plots.fig_theta_rhythmicity(r, m, fname=f"04_theta_rhythmicity_{label}.png")
    plt.close("all")
    for k in mods_pooled:
        mods_pooled[k].append(mm[k])
        intr_pooled[k].append(ii[k])
    lfp_freqs.append(ff)
    print(f"  {label}: {int(m.sum())} place cells, "
          f"{int((d[m].prec_p_shuffle < 0.05).sum())} with significant precession")

dfall = pd.concat(all_df, ignore_index=True)
dfpc = dfall[dfall.is_place_cell].copy()
mods_pooled = {k: np.concatenate(v) for k, v in mods_pooled.items()}
intr_pooled = {k: np.concatenate(v) for k, v in intr_pooled.items()}
dfall.to_csv("unit_table_all_sessions.csv", index=False)
with open("results.pkl", "wb") as fh:
    pickle.dump(dict(dfall=dfall, prec=all_prec, mods=mods_pooled,
                     intrinsic=intr_pooled, lfp_freqs=lfp_freqs), fh)

# %% [markdown]
# ## 6. Population results

# %%
plots.fig_entrainment_population(dfall, mods_pooled, intr_pooled, lfp_freqs)
plots.fig_precession_population(dfpc, all_prec)
plt.close("all")

# %% [markdown]
# ![entrainment population](figures/08_entrainment_population.png)
#
# ![precession population](figures/06_precession_population.png)

# %%
sig = dfpc[dfpc.prec_p_shuffle < 0.05]
exc = dfall[(dfall.cell_type == "excitatory") & (dfall.n_spikes_run > 100)]
inh = dfall[(dfall.cell_type == "inhibitory") & (dfall.n_spikes_run > 100)]

print(f"sessions {dfall.session.nunique()}   units {len(dfall)}   place cells {len(dfpc)}")
print()
print("--- theta phase entrainment ---")
print(f"significantly locked (Rayleigh p<0.05): "
      f"pyramidal {100 * (exc.p_rayleigh < 0.05).mean():.0f}% of {len(exc)}, "
      f"interneurons {100 * (inh.p_rayleigh < 0.05).mean():.0f}% of {len(inh)}")
print(f"median MRL: pyramidal {exc.mrl.median():.3f}, interneuron {inh.mrl.median():.3f}")
print(f"LFP theta frequency per session: {[round(f, 2) for f in lfp_freqs]} Hz")
print(f"median intrinsic place-cell oscillation: "
      f"{np.median(intr_pooled['place cells']):.2f} Hz")
print()
print("--- theta phase precession ---")
print(f"significant (shuffle p<0.05): {len(sig)}/{len(dfpc)} "
      f"({100 * len(sig) / len(dfpc):.0f}%)")
print(f"of those, negative slope: {int((sig.prec_slope < 0).sum())}/{len(sig)}")
print(f"median slope {np.degrees(sig.prec_slope.median()):.0f} deg per field traversal; "
      f"median |rho| {sig.prec_rho.abs().median():.2f}")
print()
print(dfpc.groupby("session").apply(
    lambda g: pd.Series({"place_cells": len(g),
                         "precessing": int((g.prec_p_shuffle < 0.05).sum()),
                         "median_slope_deg": np.degrees(
                             g.loc[g.prec_p_shuffle < 0.05, "prec_slope"].median())}),
    include_groups=False).to_string())

# %% [markdown]
# ## 7. Summary
#
# Both phenomena are present and robust in this dataset.
#
# **Entrainment.** During track running, the great majority of CA1 units fire at a
# preferred phase of the LFP theta cycle, with interneurons locked more tightly than
# pyramidal cells. The effect is visible without any LFP reference as well: spike
# autocorrelograms carry clear side peaks at the theta period. Those peaks sit at
# slightly *shorter* lags than the LFP theta period, meaning place cells oscillate a few
# tenths of a hertz faster than the field potential. That frequency offset is exactly the
# dual-oscillator signature that produces a steady phase advance.
#
# **Precession.** About two thirds of place cells show a circular-linear relationship
# between theta phase and within-field position that survives a phase-shuffling
# permutation test, and essentially all of those have negative slope: phase advances as
# the animal moves through the field. The median advance is close to 200 degrees over a
# single traversal, in line with the published range, and it is visible pass by pass
# rather than only in the pooled data.
#
# Together these say that CA1 spike timing is organized by theta twice over: the rhythm
# sets *when in the cycle* a cell may fire at all, and the animal's progress through the
# place field sets *where in that window* the spikes land.
