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
# # The head-direction system is a continuous ring attractor that keeps its 1-D structure during sleep
#
# **Data:** [DANDI:000939](https://dandiarchive.org/dandiset/000939), *Large-scale recordings of
# head direction cells in mouse postsubiculum* (Duszkiewicz, Peyrache and colleagues).  Each
# session contains a silicon-probe recording from the postsubiculum (PoSub) of one mouse while it
# forages in an open arena (head direction tracked) and while it sleeps in its home cage (no
# behavioural reference at all), together with REM/NREM scoring and a `is_head_direction` flag
# for each unit.
#
# **The claim to be demonstrated.** The head-direction (HD) system is usually described as a
# *continuous attractor*: the population activity of a large group of HD cells is confined to a
# one-dimensional, closed (ring-shaped) manifold, on which a single bump of activity moves
# smoothly.  If that ring is a property of the *network* rather than of the sensory input, it must
# still be there when the animal is asleep and the input is gone.  Three things then have to be
# true during sleep:
#
# 1. **Geometry.** Population activity still lies on a ring: a one-dimensional, closed curve, not
#    a filled blob and not a set of unrelated clusters.
# 2. **Correspondence.** The coordinate along that ring is the *same* coordinate as during waking:
#    the cells' positions around the sleeping ring reproduce their waking preferred directions.
# 3. **Dynamics.** A single sharp bump moves continuously along the ring, rather than the
#    population flickering between random states.
#
# **How each is tested here.** Everything below uses only spike times.  We fit HD tuning curves
# during waking, then throw the behaviour away and analyse sleep blind:
#
# | Question | Analysis | Control |
# |---|---|---|
# | Ring geometry | Isomap embedding of population vectors; annularity ("ring score") | per-cell circular shift of spike trains |
# | Ring topology | persistent homology (H1) of the point cloud | same shuffle |
# | Same coordinate as waking | tuning curves recomputed against the *internal* manifold angle | same shuffle |
# | Coherent single bump | two disjoint halves of the population decode independently and are compared | same shuffle |
# | Continuous motion | angular velocity of the decoded bump | same shuffle |
# | The internal angle drives the cells | Poisson GLM (NeMoS) on the internal angle, cross-validated | time-shifted covariate |
#
# The shuffle is the key control: it circularly shifts each cell's spike train independently
# within the same epochs, so every cell keeps its firing rate and its own temporal statistics,
# and only the coordination *between* cells is destroyed.
#
# The analysis is prototyped on one session and then repeated on every session in the dandiset
# with at least 20 well-tuned HD cells (19 sessions, 19 mice).

# %% [markdown]
# ## Setup
#
# Data are streamed from the DANDI S3 bucket with `remfile` (disk-cached); nothing is downloaded
# in full.  `hd_io.py`, `hd_core.py`, `hd_glm.py` and `hd_figs.py` sit next to this notebook.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")          # headless: figures are written to disk
import matplotlib.pyplot as plt
import pynapple as nap

import hd_io
import hd_core as hc
import hd_glm as hg
import hd_figs

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
SESSION = "sub-A3716/sub-A3716_ses-201015b_behavior+ecephys.nwb"
BS = hc.EMBED["bin_size"]      # 0.2 s bins, used for every state
rng = np.random.default_rng(0)
print("pynapple", nap.__version__, "| bin size", BS, "s")

# %% [markdown]
# ## 1. Load one session and look at the raw data
#
# The recording alternates home cage (sleep) and arena (foraging) blocks.  Sleep states are only
# used inside the home cage; the arena epochs are waking by construction.

# %%
d = hd_io.load_session(SESSION)
beh = hc.behaviour_epoch(d)                 # arena exploration, HD tracked
sleep = hc.sleep_epochs(d)                  # REM / NREM inside the home cage
hd = hc.clean_hd(d["hd"])

print(f"session {d['subject']}/{d['session']}: {len(d['units'])} units, "
      f"{int(np.sum(d['units'].is_hd))} flagged as head-direction cells")
print(f"arena epochs: {float(beh.tot_length()):.0f} s")
for s, ep in sleep.items():
    print(f"{s.upper():5s}: {float(ep.tot_length()):.0f} s in {len(ep)} bouts")

# %% [markdown]
# The raster below is the whole demonstration in miniature: with the cells sorted by their
# preferred direction, the active subset is a compact "bump" that slides up and down in step with
# the measured head direction.

# %%
tc, st = hc.select_hd_cells(d, beh)         # tuning curves + selection statistics for all units
keep = st.index[st["keep"]].values           # HD cells that pass rate / strength / stability
units = d["units"][list(keep)]
tck = tc[keep]
pref = st.loc[keep, "pref"].values

R = dict(session=d["session"], subject=d["subject"], beh=beh, sleep=sleep, tc=tc, st=st,
         keep=keep, pref=pref, hd=hd, pos=d["pos"], epochs=d["epochs"], states=d["states"],
         spikes={int(k): units[k].t for k in units.keys()}, tck=tck)
print(hd_figs.fig01(R, FIGDIR))

# %% [markdown]
# ![](figures/fig01_session_overview.png)

# %% [markdown]
# ## 2. Head-direction tuning during waking
#
# Tuning curves are computed with pynapple over the arena epochs.  We keep a unit if the dataset
# flags it as an HD cell *and* it fires at least 0.5 Hz, has a mean vector length above 0.25, and
# its tuning curve computed on the first half of the arena epochs correlates above 0.5 with the
# one computed on the second half.

# %%
print(f"{int(st['keep'].sum())} HD cells kept out of {int(np.sum(d['units'].is_hd))} flagged "
      f"({len(st)} units total)")
print(st.loc[keep, ["rate", "mvl", "stability", "peak"]].describe().round(2).to_string())
print(hd_figs.fig02(R, FIGDIR))

# %% [markdown]
# ![](figures/fig02_hd_tuning.png)
#
# Preferred directions tile the circle, which is what makes a ring visible in the population
# activity at all.

# %% [markdown]
# ## 3. One preprocessing recipe for every brain state
#
# For each epoch the spikes are binned at 200 ms, square-root transformed, smoothed with a
# 1-bin Gaussian *within* each contiguous interval, z-scored per cell, and each population vector
# is normalised to unit length (so the analysis is about the *pattern* of activity, not the
# overall amount).  The embedding is a 2-D Isomap with 25 neighbours on up to 3000 randomly
# chosen bins.
#
# During NREM the cortex passes through DOWN states in which the whole population falls silent
# and no bump exists; the 20 % of bins with the lowest population activity are therefore excluded
# in that state only.  Wake and REM use every bin.
#
# The same pipeline is applied to the data and to the per-cell circular shuffle.

# %%
STATES = [("wake", beh, 0), ("rem", sleep["rem"], 0),
          ("nrem", sleep["nrem"], hc.EMBED["nrem_activity_pct"])]

res = {}
for name, ep, pct in STATES:
    out = {}
    for cond in ["real", "shuffle"]:
        U = units if cond == "real" else hc.shift_shuffle(units, ep, rng)

        # --- geometry -------------------------------------------------------
        Y, cnt, mask = hc.population_matrix(U, ep, activity_pct=pct)
        emb, sel = hc.isomap_embed(Y, n_components=2, n_neighbors=hc.EMBED["n_neighbors"],
                                   max_points=hc.EMBED["max_points"])
        ring = hc.ring_metrics(emb)
        ph = hc.ph_h1(Y)                                   # persistent homology (H1)

        # --- the internal coordinate, and what the cells do along it ---------
        itc = hc.internal_tuning(cnt.values[mask][sel], ring["angle"])
        itc.columns = tck.columns
        ipref = hc.tuning_stats(itc)["pref"].values
        cc_pref, off, sgn = hc.best_circ_align(ipref, pref)

        # --- decoding with the *waking* tuning curves ------------------------
        dec, prob = hc.decode(tck, U, ep, BS)
        split = hc.split_half_decode(tck, U, ep, BS, activity_mask=mask)
        speed = hc.angular_speed(dec, BS)
        offs, corroff, C = hc.pair_corr_vs_offset(Y, pref)

        out[cond] = dict(emb=emb, sel=sel, ring=ring, ph=ph, mask=mask,
                         dec_t=dec.t, dec=np.mod(dec.values, hc.TWOPI),
                         dv=np.mod(dec.values[mask], hc.TWOPI),
                         prob_t=prob.t, prob=prob.values,
                         itc=itc.values, itc_idx=itc.index.values, ipref=ipref,
                         cc_pref=cc_pref, align=(off, sgn), split=split, speed=speed,
                         offs=offs, corroff=corroff, C=C,
                         counts=cnt.values[mask], t=cnt.t[mask], ep=ep,
                         cc_iso=hc.best_circ_align(ring["angle"], np.mod(dec.values[mask],
                                                                        hc.TWOPI)[sel])[0],
                         Y=Y if cond == "real" else None)
        if name == "wake" and cond == "real":     # ground truth, for validation only
            true = np.mod(np.angle(np.interp(dec.t, hd.t, np.cos(hd.d))
                                   + 1j * np.interp(dec.t, hd.t, np.sin(hd.d))), hc.TWOPI)
            out[cond]["true"] = true
            out[cond]["err"] = hc.circ_dist(np.mod(dec.values, hc.TWOPI), true)
    res[name] = out
    r, s = out["real"], out["shuffle"]
    print(f"{name:5s}  ring {r['ring']['ring_score']:.2f} (shuffle {s['ring']['ring_score']:.2f})"
          f"   H1 {r['ph']['top']:.2f} ({s['ph']['top']:.2f})"
          f"   split-half r {r['split']['cc']:.2f} ({s['split']['cc']:.2f})"
          f"   internal-vs-wake pref r {r['cc_pref']:.2f} ({s['cc_pref']:.2f})")
R["res"] = res

# %% [markdown]
# A larger null: five independent shuffles per state, for error bars on the summary panels.

# %%
null = {}
for name, ep, pct in STATES:
    vals = []
    for i in range(5):
        U = hc.shift_shuffle(units, ep, rng)
        Y, cnt, mask = hc.population_matrix(U, ep, activity_pct=pct)
        emb, sel = hc.isomap_embed(Y, n_components=2, max_points=2000, seed=i)
        vals.append((hc.ring_metrics(emb)["ring_score"], hc.ph_h1(Y, seed=i)["top"],
                     hc.split_half_decode(tck, U, ep, BS, seed=i, activity_mask=mask)["cc"]))
    null[name] = np.array(vals)
R["null"] = null
print(pd.DataFrame({s: null[s].mean(0) for s in null},
                   index=["ring score", "H1 lifetime", "split-half r"]).round(3).to_string())

# %% [markdown]
# ## 4. Waking: the manifold is a ring and its coordinate is head direction
#
# This is the sanity check that the method works when the answer is known.  The Isomap embedding
# of waking population vectors is a clean annulus; colouring it by the *measured* head direction
# shows that going once around the ring corresponds to turning the head once around.  Bayesian
# decoding from the same spikes recovers head direction with a median error of about 12°.
#
# The right-hand panel makes the "ring" statement concrete in the raw (un-embedded) data: the
# first two principal components of the population activity assign each cell a position on a
# circle, and that position is its preferred direction.

# %%
w = res["wake"]["real"]
print(f"median decoding error: {np.degrees(np.median(np.abs(w['err']))):.1f} deg")
print(f"circular r(decoded, measured) = "
      f"{hc.circ_corr(np.mod(w['dec'], hc.TWOPI), w['true']):.3f}")
print(hd_figs.fig03(R, FIGDIR))

# %% [markdown]
# ![](figures/fig03_wake_ring.png)

# %% [markdown]
# ## 5. Sleep: the ring is still there
#
# Now the same pipeline on REM and NREM, with no behavioural variable anywhere in it.  The
# embeddings are coloured by the head direction decoded from the *waking* tuning curves, which is
# the network's internally maintained estimate.
#
# Both sleep states give an annulus with the decoded angle running smoothly around it.  The
# shuffled controls, which keep every single-cell property, give a featureless blob.

# %%
print(hd_figs.fig04(R, FIGDIR))

# %% [markdown]
# ![](figures/fig04_sleep_ring.png)
#
# **Ring score** is `1 - IQR(radius)/median(radius)` of the embedded cloud: a thin annulus gives
# a value near 1, a filled blob near 0.  **Centre-filling** is the fraction of points inside half
# the median radius, which a ring must keep near zero.
#
# **Split-half decoder agreement** is the strongest of the state-free measures: the cells are
# split into two disjoint halves, each half decodes head direction with its own waking tuning
# curves, and the two independent estimates are compared.  Two halves of a population can only
# agree bin by bin if a single coherent bump is present.

# %% [markdown]
# ## 6. The topology is a circle, not just a "roughly round cloud"
#
# Persistent homology asks a question that no amount of visual roundness can answer: does the
# point cloud have exactly one hole?  We reduce the population vectors to 6 principal components,
# keep the densest 60 % of points (standard denoising before persistence computations), and
# compute the H1 (loop) persistence diagram with Ripser.  Lifetimes are normalised by the scale
# at which the cloud becomes connected, so states with different noise levels are comparable.
#
# Every state has exactly one loop that lives far longer than all the others.  The shuffles have
# none.

# %%
for s in ["wake", "rem", "nrem"]:
    r, sh = res[s]["real"]["ph"], res[s]["shuffle"]["ph"]
    print(f"{s:5s} longest H1 lifetime {r['top']:.2f} (2nd longest {r['life'][1]:.2f}, "
          f"ratio {r['ratio']:.1f})   shuffle {sh['top']:.2f} (ratio {sh['ratio']:.1f})")
print(hd_figs.fig05(R, FIGDIR))

# %% [markdown]
# ![](figures/fig05_topology.png)

# %% [markdown]
# ## 7. The sleeping ring carries the waking map
#
# A ring by itself is not enough: it has to be the *head-direction* ring.  So we forget the
# waking tuning curves and use only the angle each population vector has on the sleep manifold.
# Recomputing every cell's tuning curve against that purely internal angle reproduces the waking
# tuning curves, and the cells' preferred angles on the sleeping ring line up with their waking
# preferred directions (circular r ≈ 0.9 in both REM and NREM for this session).
#
# The internal angle is defined only up to a rotation and a reflection (the embedding has no way
# to know which way is north), so both are fitted before comparison; that is two parameters for
# 77 cells.

# %%
for s in ["wake", "rem", "nrem"]:
    print(f"{s:5s} internal ring order vs waking preferred directions: "
          f"circular r = {res[s]['real']['cc_pref']:.3f} "
          f"(shuffle {res[s]['shuffle']['cc_pref']:.3f})")
print(hd_figs.fig06(R, FIGDIR))

# %% [markdown]
# ![](figures/fig06_internal_tuning.png)

# %% [markdown]
# ## 8. A single bump that moves continuously
#
# Decoding with the waking tuning curves gives a posterior over head direction in every 200 ms
# bin.  In REM the posterior is a single sharp bump that drifts smoothly, at roughly the angular
# speed of real head movements.  In NREM the bump is equally sharp but moves several times
# faster, and is interrupted by the DOWN states in which the population is silent (the dark
# vertical bands).  In both cases the step-to-step angular displacement is far smaller than for
# the shuffle, i.e. the state moves along the ring rather than jumping across it.

# %%
for s in ["wake", "rem", "nrem"]:
    r, sh = res[s]["real"], res[s]["shuffle"]
    print(f"{s:5s} median |angular velocity| {np.median(r['speed']):6.0f} deg/s "
          f"(shuffle {np.median(sh['speed']):.0f});  split-half disagreement "
          f"{np.degrees(r['split']['median_abs_err']):.0f} deg "
          f"(shuffle {np.degrees(sh['split']['median_abs_err']):.0f})")
print(hd_figs.fig07(R, FIGDIR))

# %% [markdown]
# ![](figures/fig07_bump_dynamics.png)

# %% [markdown]
# ## 9. Cell-cell coordination during sleep follows the waking map
#
# The classical signature of an internally organised HD system (Peyrache et al., 2015) is that
# pairs of HD cells with similar preferred directions stay correlated in sleep.  Sorting the
# correlation matrix by waking preferred direction gives the same banded structure in wake, REM
# and NREM, and the correlation-versus-offset curves fall almost on top of each other.

# %%
print(hd_figs.fig08(R, FIGDIR))

# %% [markdown]
# ![](figures/fig08_pairwise_correlation.png)

# %% [markdown]
# ## 10. A GLM driven by the internal angle predicts sleep spiking
#
# Finally a quantitative, cross-validated version of section 7, fitted with NeMoS: a Poisson GLM
# with a cyclic B-spline basis over an angular covariate, one model per population, 5-fold
# cross-validation, scored as McFadden pseudo-R² per cell.
#
# During waking the covariate can be the measured head direction.  During sleep the only
# covariate available is the angle the population defines on its own manifold, and it predicts
# held-out sleep spiking almost as well as real head direction predicts waking spiking.  The
# fitted sleep tuning curves are the waking tuning curves.  Shifting the covariate in time
# destroys the prediction.

# %%
conds = [("wake / measured HD", w["true"][w["mask"]][w["sel"]], w["counts"][w["sel"]])]
for s in ["wake", "rem", "nrem"]:
    r = res[s]["real"]
    _, off, sgn = hc.best_circ_align(r["ipref"], pref)
    conds.append((f"{s} / internal angle", np.mod(sgn * r["ring"]["angle"] + off, hc.TWOPI),
                  r["counts"][r["sel"]]))
for lab, ang, cnt in conds:
    pr = hg.cv_pseudo_r2(ang, cnt)
    pn = hg.cv_pseudo_r2(hg.shift_null(ang, rng), cnt)
    print(f"{lab:24s} median pseudo-R2 = {np.median(pr):.3f}   (time-shifted null "
          f"{np.median(pn):.3f})")
print(hd_figs.fig09(R, FIGDIR))

# %% [markdown]
# ![](figures/fig09_glm_internal_coordinate.png)

# %% [markdown]
# ## 11. Every session in the dandiset
#
# The whole pipeline (including one shuffle per state) is repeated on every session with at least
# 20 well-tuned HD cells.  `sweep_sessions.py` writes `cross_session_metrics.csv`; if the file is
# already present it is reused.

# %%
if not os.path.exists("cross_session_metrics.csv"):
    import sweep_sessions
    sweep_sessions.main()
cs = pd.read_csv("cross_session_metrics.csv")
print(f"{cs['session'].nunique()} sessions, {cs['subject'].nunique()} mice")
summary = (cs.assign(cond=np.where(cs["cond"] == "real", "data", "shuffle"))
             .groupby(["state", "cond"])[["ring_score", "h1_top", "cc_internal_pref",
                                          "split_cc", "split_err_deg", "med_speed"]]
             .median().round(3))
print(summary.to_string())
print(hd_figs.fig10(cs, FIGDIR))

# %% [markdown]
# ![](figures/fig10_cross_session.png)

# %% [markdown]
# ## 12. What the numbers say
#
# Across 19 sessions from 19 mice, with the per-cell shuffle as the null (medians):
#
# | | wake | REM | NREM | shuffle (wake/REM/NREM) |
# |---|---|---|---|---|
# | ring score | 0.65 | 0.64 | 0.54 | 0.21 / 0.39 / 0.22 |
# | longest H1 lifetime | 1.81 | 0.74 | 0.54 | 0.32 / 0.46 / 0.32 |
# | internal ring order vs waking preferred direction (circ. r) | 0.94 | 0.76 | 0.62 | 0.10 / 0.07 / 0.10 |
# | split-half decoder agreement (circ. r) | 0.83 | 0.64 | 0.17 | 0.00 / 0.01 / 0.00 |
# | split-half disagreement | 9° | 15° | 36° | 90° / 87° / 87° |
# | bump angular speed | 45°/s | 30°/s | 255°/s | 135 / 105 / 345°/s |
#
# (Median head-direction decoding error during waking, across sessions: 9.4°.)
#
# **The three requirements are met.**  During both REM and NREM the population activity of
# postsubicular HD cells stays on a one-dimensional closed manifold (ring score and a single
# long-lived H1 loop, both far above the shuffle); the coordinate along that manifold is the
# waking head-direction map (internal ring order versus waking preferred direction, circular
# r ≈ 0.6 across sessions and ≈ 0.9 in the best sessions); and a single sharp bump moves
# continuously along it, so that two disjoint halves of the population report the same angle.
# Since the shuffles preserve every single-cell property and destroy only cross-cell
# coordination, none of this can be explained by firing rates or by individual cells' temporal
# statistics.
#
# **Where the evidence is weaker, and why.**  The moment-to-moment measures are much noisier in
# NREM than in REM.  Two reasons are visible in the data: the bump moves about five times faster
# in NREM (255°/s versus 30°/s), so a 200 ms bin smears it, and the population is repeatedly
# silenced by DOWN states.  Consistent with a resolution limit rather than an absent ring,
# split-half agreement in NREM depends strongly on the number of recorded HD cells (panel 5 of
# the cross-session figure): it is 0.74 and 0.76 in the two sessions with the largest
# populations, and scatters around zero when fewer than ~35 cells were recorded.  The
# *geometric* and *ordering* measures, which pool over the whole epoch instead of individual
# bins, are clearly above the shuffle in NREM in almost every session.
#
# **Caveats.**  (i) The recordings are from postsubiculum, one node of the HD circuit; the
# canonical attractor is usually attributed to the anterodorsal thalamus, with which PoSub is
# reciprocally connected, so this shows the ring is expressed in PoSub, not where it is
# generated.  (ii) The rotation and reflection relating the internal angle to the waking map are
# fitted (two parameters per session).  (iii) Isomap and the ring score are descriptive; the
# persistent-homology result is the assumption-light version of the same claim, and the
# split-half decoding and GLM results do not involve dimensionality reduction at all.
# (iv) Sleep-state scoring is taken from the dandiset and is only used inside home-cage epochs.

# %% [markdown]
# ### Figures written
#
# All figures are in `figures/`:
#
# 1. `fig01_session_overview.png`: session structure, behaviour, sorted raster
# 2. `fig02_hd_tuning.png`: waking HD tuning curves and cell selection
# 3. `fig03_wake_ring.png`: the ring during waking, decoding validation
# 4. `fig04_sleep_ring.png`: the ring in REM and NREM, with shuffles
# 5. `fig05_topology.png`: persistent homology (one loop per state)
# 6. `fig06_internal_tuning.png`: the internal coordinate is the waking map
# 7. `fig07_bump_dynamics.png`: decoded posterior, bump speed, split-half agreement
# 8. `fig08_pairwise_correlation.png`: correlation structure across states
# 9. `fig09_glm_internal_coordinate.png`: NeMoS GLM on the internal angle
# 10. `fig10_cross_session.png`: all 19 sessions
