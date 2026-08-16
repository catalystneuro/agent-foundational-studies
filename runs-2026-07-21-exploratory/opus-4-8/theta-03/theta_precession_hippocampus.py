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
# # Theta phase entrainment and phase precession of hippocampal place cells
#
# **Dataset: [DANDI:000044](https://dandiarchive.org/dandiset/000044)** — Grosmark, Long
# and Buzsáki, *Recordings from hippocampal area CA1, PRE, during and POST novel spatial
# learning* (the dataset behind Grosmark & Buzsáki, Science 2016). Bilateral silicon
# probe recordings from dorsal CA1 of freely moving rats, with sorted units, 128-channel
# LFP at 1250 Hz, and tracked position on a linear track flanked by pre- and post-run
# sleep.
#
# This notebook demonstrates two related properties of hippocampal place cells:
#
# 1. **Theta phase entrainment.** During locomotion the CA1 LFP is dominated by a
#    6-12 Hz theta rhythm, and pyramidal-cell spikes are not uniformly distributed
#    across the theta cycle: they cluster near a preferred phase.
# 2. **Theta phase precession.** Within a single place field the preferred phase is not
#    fixed. As the animal traverses the field, successive spikes occur at
#    systematically earlier phases of theta, so that phase carries information about
#    position within the field over and above firing rate (O'Keefe & Recce 1993;
#    Skaggs et al. 1996).
#
# Everything is computed from the archive by streaming: the NWB files are 5-9 GB each
# and are never downloaded in full. Position, spikes and one LFP channel per session are
# read over HTTP through LINDI, which resolves into byte-range requests against the
# DANDI S3 bucket.
#
# **Analysis outline**
#
# * split the maze epoch into single track traversals and separate the two running
#   directions, because place fields on a linear track are directional;
# * pick the LFP channel with the highest theta/delta power ratio during running and
#   define theta phase by the Hilbert transform of the 6-12 Hz bandpassed signal
#   (0 deg = theta peak, 180 deg = theta trough);
# * build directional rate maps, keep excitatory units with a well-defined field and
#   at least 0.3 bits/spike of spatial information;
# * for each field, test phase locking with a Rayleigh test against a spike-jitter
#   control, and test precession with the Kempter et al. (2012) circular-linear
#   regression of theta phase on normalized position, with a permutation p-value.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive: figures are written to disk
import matplotlib.pyplot as plt

import pynapple as nap

from hc11_io import (
    SESSIONS,
    open_session,
    get_maze_epoch,
    get_position,
    get_units,
    maze_type,
    is_linear_track,
)
from theta_analysis import (
    THETA_BAND,
    lap_intervals,
    running_speed,
    theta_phase,
    rayleigh,
    circ_lin_fit,
    circ_lin_shuffle_p,
)
from run_session import analyze_session
import make_figures as F

PRIMARY = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # sub-Achilles, 10/25/2013
os.makedirs("figures", exist_ok=True)
print("pynapple", nap.__version__)

# %% [markdown]
# ## 1. What is in the file
#
# The NWB file holds a `behavior` module with the raw and linearized position, an
# `ecephys` module with the 128-channel LFP, a units table with cell-type labels, and an
# epoch table separating pre-sleep, maze running and post-sleep.

# %%
h5, nwbfile = open_session(PRIMARY)
print(nwbfile.intervals["epochs"].to_dataframe())
print("\nbehavior interfaces:", list(nwbfile.processing["behavior"].data_interfaces))
print("maze:", maze_type(nwbfile))
units_tbl = nwbfile.units.to_dataframe()
print("\nunits:", len(units_tbl))
print(units_tbl.groupby(["location", "cell_type"]).size())
es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
print("\nLFP:", es.data.shape, "at", es.rate, "Hz")

# %% [markdown]
# ## 2. Behaviour: single traversals of the linear track
#
# The linearized position is stored only while the animal is on the track proper, so
# each contiguous block of finite samples is one traversal. One quirk worth noting: the
# `SpatialSeries` stores the sampling *period* (0.0256 s, i.e. 39.06 Hz) in its `rate`
# field, so timestamps have to be reconstructed by hand. The reconstruction is checked
# against the epoch table in `hc11_io.get_position`.

# %%
maze = get_maze_epoch(nwbfile)
pos = get_position(nwbfile).restrict(maze)
units = get_units(nwbfile)
track_len = float(np.ceil(np.nanmax(pos.d) / 10.0) * 10.0)
laps, lap_dir = lap_intervals(pos, min_duration=0.5, min_extent=0.6 * track_len)
posf = pos[np.isfinite(pos.d)]
speed = running_speed(posf, laps)

print(f"track length      {track_len:.0f} cm")
print(f"maze epoch        {maze.tot_length():.0f} s")
print(f"traversals        {len(laps)} "
      f"({(lap_dir==1).sum()} left-to-right, {(lap_dir==-1).sum()} right-to-left)")
print(f"time running      {laps.tot_length():.0f} s")
print(f"median speed      {np.median(speed.restrict(laps).d):.0f} cm/s")

# %% [markdown]
# ## 3. Theta in the CA1 LFP
#
# Theta amplitude varies a lot across the 128 recording sites, so the channel is chosen
# empirically: for every channel the power spectrum is computed over the running periods
# only and the channel with the largest 6-12 Hz / 1-4 Hz ratio is kept. Theta phase then
# comes from the Hilbert transform of the bandpassed signal, with 0 deg at the peak and
# 180 deg at the trough of the filtered wave.

# %%
lfp_demo = nap.Tsd(
    t=np.arange(int(maze.start[0] * 1250), int(maze.start[0] * 1250) + 1250 * 10) / 1250,
    d=np.asarray(es.data[int(maze.start[0] * 1250):int(maze.start[0] * 1250) + 1250 * 10, 65],
                 float) * float(es.conversion) * 1e3,
)
theta_demo, phase_demo = theta_phase(lfp_demo)
print("theta phase range:", phase_demo.d.min(), "-", phase_demo.d.max(), "deg")

# %% [markdown]
# ## 4. Run the full per-session pipeline
#
# `analyze_session` puts the above together: it selects the theta channel, reads that
# channel across the whole maze epoch, computes directional rate maps, identifies place
# fields, assigns a theta phase to every in-field spike, and runs the entrainment and
# precession statistics. Results are cached under `results/` so re-running is cheap.
#
# Field criteria: excitatory cell type, peak rate >= 1 Hz, spatial information >= 0.3
# bits/spike, and a contiguous region above 25% of the peak rate spanning at least 3
# position bins. Precession is only fitted for fields with at least 40 in-field spikes.

# %%
S = analyze_session(PRIMARY)
df = S["df"]
print(df.head())

# %% [markdown]
# ### Behaviour and spiking overview

# %%
F.fig_behavior(S)

# %% [markdown]
# ![](figures/fig01_behaviour_and_spiking.png)
#
# The animal runs 81 traversals of the 1.6 m track. Ordering place cells by the position
# of their field peak and plotting their spikes during one traversal shows the expected
# sequential activation: the population follows the animal's position across the track.

# %% [markdown]
# ### Theta rhythm

# %%
F.fig_theta(S)

# %% [markdown]
# ![](figures/fig02_theta_lfp.png)
#
# During running the LFP shows a clean ~9 Hz theta oscillation with a visible harmonic
# near 18 Hz, and theta power is markedly larger than during the rest of the maze epoch
# (when the animal is mostly stationary at the reward ends). The theta/delta ratio on the
# selected channel is above 13.

# %% [markdown]
# ### Place fields

# %%
F.fig_place_fields(S)

# %% [markdown]
# ![](figures/fig03_place_fields.png)
#
# Fields tile the track in both running directions, as expected for CA1 on a linear
# track.

# %% [markdown]
# ## 5. Theta phase entrainment
#
# For each field, in-field spikes are assigned the theta phase of the nearest LFP sample
# (0.8 ms resolution, about 3 deg of theta at 9 Hz). Non-uniformity is tested with a
# Rayleigh test. As a control, each spike train is jittered by a uniform offset of up to
# +/- 400 ms, which preserves firing rate and field position but destroys any
# relationship to the ongoing theta cycle.

# %%
sig_lock = (df.p_rayleigh < 0.05).sum()
print(f"fields with significant phase locking: {sig_lock}/{len(df)} "
      f"({100*sig_lock/len(df):.0f}%)")
print(f"median MRL observed {df.mrl.median():.3f} vs jitter control "
      f"{df.mrl_jitter.median():.3f}")
mu, R, p, n = rayleigh(np.deg2rad(df.loc[df.p_rayleigh < 0.05, "pref_phase"].values))
print(f"preferred phases cluster at {np.rad2deg(mu)%360:.0f} deg "
      f"(R={R:.2f}, Rayleigh p={p:.1e}, n={n})")

# %%
F.fig_entrainment(S)

# %% [markdown]
# ![](figures/fig04_theta_entrainment.png)
#
# Individual place fields are strongly locked to theta, with mean resultant lengths far
# above the jitter control for essentially every field. Preferred phases are themselves
# clustered across the population, near the trough of the theta wave recorded on the
# selected channel. The absolute preferred phase depends on the recording depth (the
# theta wave reverses across the CA1 layers), so it is the clustering, not the numerical
# value, that is the result here.

# %% [markdown]
# ## 6. Phase precession
#
# For every field, spike phase is regressed on the animal's normalized position within
# the field using the circular-linear method of Kempter et al. (2012): the slope `a` is
# the value that maximizes the resultant length of `phi - 2*pi*a*x`, and the
# circular-linear correlation `rho` is computed at that slope. Position is normalized so
# the slope is in theta cycles per field, and is oriented along the direction of travel
# so that negative slopes always mean phase advance.
#
# Significance uses a permutation test: the phase-position pairing is shuffled 1000
# times and the observed `|rho|` is compared against the null.
#
# Here is the fit for one example field, done explicitly:

# %%
best = df.dropna(subset=["rho"]).sort_values("rho").iloc[0]
e = S["examples"][(best.unit, int(best.direction))]
fit = circ_lin_shuffle_p(e["x"], np.deg2rad(e["phase"]), n_shuffle=1000,
                         rng=np.random.default_rng(0))
print(f"unit {int(best.unit)}, direction {int(best.direction):+d}, "
      f"{fit['n']} in-field spikes")
print(f"  slope {fit['slope']:.2f} cycles/field = {fit['slope']*360:.0f} deg/field")
print(f"  rho   {fit['rho']:.3f}")
print(f"  p     {fit['p_shuffle']:.4f} (permutation), {fit['p']:.2e} (parametric)")

# %%
F.fig_precession_examples(S)

# %% [markdown]
# ![](figures/fig05_precession_examples.png)
#
# Each panel plots the theta phase of every in-field spike against the animal's
# normalized position in the field, repeated over two theta cycles so the wrap-around is
# visible. The black lines are the fitted circular-linear regression. In every case
# spikes drift to earlier phases as the animal moves through the field.

# %%
F.fig_precession_population(S)

# %% [markdown]
# ![](figures/fig06_precession_population.png)

# %%
fit_df = df.dropna(subset=["rho"])
sig = fit_df[fit_df.p_shuffle < 0.05]
print(f"fields fitted:            {len(fit_df)}")
print(f"significant precession:   {len(sig)} ({100*len(sig)/len(fit_df):.0f}%)")
print(f"  of which negative slope: {(sig.slope<0).sum()} "
      f"({100*(sig.slope<0).mean():.0f}%)")
print(f"median slope (significant): {sig.slope.median()*360:.0f} deg per field")
print(f"median rho   (significant): {sig.rho.median():.2f}")

# %% [markdown]
# ## 7. Across sessions
#
# The dandiset has eight sessions. Three of them (`Achilles-11012013`,
# `Cicero-09102014`, `Gatsby-08282013`) used a circular maze whose linearized coordinate
# wraps around, which would need different field-detection logic, so they are skipped
# here; the remaining five linear-track sessions from four rats are pooled.

# %%
results = {S["label"]: S}
for aid, label in SESSIONS.items():
    if aid == PRIMARY:
        continue
    out = analyze_session(aid)
    if out is not None:
        results[label] = out

for label, r in results.items():
    print(f"{label:20s} theta/delta on chosen channel "
          f"{r['chan_scan'][:,2].max():5.1f} (ch {r['best_ch']})")

all_df = pd.concat([r["df"] for r in results.values()], ignore_index=True)
all_df.to_csv("all_sessions_fields.csv", index=False)
print(all_df.groupby("session").size())

# %%
F.fig_sessions(all_df)

# %% [markdown]
# ![](figures/fig07_across_sessions.png)

# %%
fit_all = all_df.dropna(subset=["rho"])
sig_all = fit_all[fit_all.p_shuffle < 0.05]
lock_all = all_df[all_df.p_rayleigh < 0.05]
mu, R, p, n = rayleigh(np.deg2rad(lock_all.pref_phase.values))

print(f"sessions                     {all_df.session.nunique()}")
print(f"place fields                 {len(all_df)}")
print(f"  phase locked (Rayleigh)    {len(lock_all)} "
      f"({100*len(lock_all)/len(all_df):.0f}%)")
print(f"  MRL observed / jittered    {all_df.mrl.median():.3f} / "
      f"{all_df.mrl_jitter.median():.3f}")
print(f"fields with precession fit   {len(fit_all)}")
print(f"  significant (perm p<0.05)  {len(sig_all)} "
      f"({100*len(sig_all)/len(fit_all):.0f}%)")
print(f"  negative slope             {(sig_all.slope<0).sum()} "
      f"({100*(sig_all.slope<0).mean():.0f}%)")
print(f"median slope                 {sig_all.slope.median()*360:.0f} deg/field")
print(f"median rho                   {sig_all.rho.median():.2f}")

# %% [markdown]
# ## Conclusion
#
# Both phenomena are clearly present in this dataset. CA1 place cells fire
# preferentially near one phase of the ongoing theta rhythm, with mean resultant lengths
# well above a spike-jitter control, and preferred phases that cluster across the
# population. Within a field, spike phase advances systematically with position: the
# large majority of fields with a significant circular-linear correlation have a negative
# slope, of roughly a half to two thirds of a theta cycle across the field, which matches
# the classical description of phase precession.
#
# The result is robust to the direction of travel, holds in every linear-track session
# analysed, and survives a permutation test that keeps the marginal distributions of
# phase and position intact.
