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
# # Orientation selectivity in the mouse visual system
#
# **Dataset: DANDI:000021, "Allen Institute - Visual Coding - Neuropixels
# (Brain Observatory 1.1 Stimulus Set)".**
#
# Orientation selectivity is the property, first described by Hubel and Wiesel in
# cat area 17, that a neuron in visual cortex responds strongly to a bar or grating
# at one orientation and weakly or not at all to the orthogonal orientation. This
# notebook demonstrates the phenomenon in mouse using extracellular Neuropixels
# recordings from the Allen Brain Observatory, streamed directly from the DANDI
# Archive.
#
# The experiment presented drifting sinusoidal gratings at 8 directions
# (0 to 315 degrees in 45 degree steps) x 5 temporal frequencies, 15 repeats each,
# 2 s per trial, interleaved with mean-luminance blank sweeps. A separate block
# presented static gratings at 6 orientations x 5 spatial frequencies x 4 phases,
# 0.25 s per trial. Each session recorded simultaneously from up to six Neuropixels
# probes spanning primary and higher visual cortex, visual thalamus (LGd, LP), and
# structures with no role in early vision (hippocampus).
#
# ### What is shown here
#
# 1. Single units in visual cortex fire selectively for grating orientation, with
#    tuning curves that are well described by a two-lobed von Mises function.
# 2. Selectivity is far stronger in cortex than in visual thalamus, and absent in
#    the hippocampal control population, which is the expected anatomical ordering.
# 3. Each unit's preferred orientation is stable when estimated from independent
#    halves of the trials, and it transfers to a different stimulus class
#    (static gratings), so it is a property of the neuron and not of the fit.
# 4. An optional model-based extension (a cross-validated NeMoS Poisson GLM and a
#    population decoder) is provided in `05_glm_decoding.py`; it was not run to
#    completion here and nothing below depends on it.
#
# ### Methods notes
#
# * Data are streamed with `remfile` + a local disk cache; no file is downloaded in
#   full. Session files are 2-3 GB each and only the byte ranges holding the
#   relevant spike trains and stimulus tables are fetched.
# * Units are filtered with the Allen default quality criteria
#   (`quality == 'good'`, ISI violations < 0.5, amplitude cutoff < 0.1,
#   presence ratio > 0.9). Brain region comes from the units' peak channel, resolved
#   through the electrodes table.
# * Analysis functions live in `orientation_lib.py` next to this notebook.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pynapple as nap

import orientation_lib as ol

matplotlib.use("Agg")           # headless: figures are written to disk
os.makedirs("figures", exist_ok=True)
os.makedirs("cache", exist_ok=True)

N_SESSIONS = 8                  # sessions used for the pooled analysis
PROTOTYPE = "715093703"         # session used for all single-session figures

# %% [markdown]
# ## 1. Find the sessions in the dandiset

# %%
sessions = ol.list_session_assets()
print("%d session-level NWB files in DANDI:%s" % (len(sessions), ol.DANDISET))
print(sessions[["subject_id", "session_id", "size"]].head(10).to_string())

row = sessions[sessions.session_id == PROTOTYPE].iloc[0]
print("\nprototype session:", row["path"])

# %% [markdown]
# ## 2. Open one session and inspect every data stream
#
# The NWB file holds the sorted units, the stimulus presentation tables (one
# `TimeIntervals` table per stimulus class), the electrode table used to assign
# units to brain regions, and behavioural signals including running speed.

# %%
f = ol.open_session(row["url"])
print("session_description:", f["session_description"][()].decode())
print("genotype:", f["general"]["subject"]["genotype"][()].decode(),
      "| age:", f["general"]["subject"]["age"][()].decode(),
      "| sex:", f["general"]["subject"]["sex"][()].decode())
print("stimulus blocks:", list(f["intervals"].keys()))

units = ol.unit_table(f)
print("\n%d sorted units, %d pass the quality criteria" % (len(units), units.pass_qc.sum()))
print(units[units.pass_qc].region.value_counts().to_string())

# %%
keep = ol.VISUAL_CORTEX + ol.THALAMUS + ol.CONTROL
sel = units[units.pass_qc & units.region.isin(keep)].reset_index(drop=True)

dg, dg_blank = ol.drifting_gratings(f)
sg, sg_blank = ol.static_gratings(f)
# The `orientation` column of the drifting-grating table holds the direction of
# motion (8 values, 0 to 315 in 45 degree steps); the static-grating table holds
# the grating orientation (6 values, 0 to 150 in 30 degree steps). All angles below
# are reported in the dataset's own convention. The two conventions turn out to be
# consistent with each other: preferred angles measured from the two stimulus
# classes agree (section 7), which they could not if the labels were 90 degrees
# apart.
print("drifting gratings: %d trials (%.1f s each) + %d blank sweeps"
      % (len(dg), dg.duration.median(), len(dg_blank)))
print(dg.groupby(["direction", "temporal_frequency"]).size().unstack().to_string())
print("\nstatic gratings: %d trials (%.2f s each) + %d blank"
      % (len(sg), sg.duration.median(), len(sg_blank)))
print(sg.ori.value_counts().sort_index().to_string())

# %% [markdown]
# ### Load spike trains into a pynapple `TsGroup`
#
# One caveat worth flagging: the NWB unit ids in this dandiset are not monotonic
# with row order, and `TsGroup` sorts its keys. `ol.load_spikes` therefore returns
# the units metadata table re-ordered to match the `TsGroup`, and everything
# downstream uses that returned table. Getting this wrong silently scrambles the
# region labels.

# %%
t_end = max(dg.stop_time.max(), sg.stop_time.max()) + 60
ep = nap.IntervalSet(start=0.0, end=t_end)
spikes, sel = ol.load_spikes(f, sel, time_support=ep)
spikes.set_info(region=np.asarray(sel.region.values))
print(spikes)

run = f["processing"]["running"]["running_speed"]
speed = nap.Tsd(t=run["timestamps"][:], d=run["data"][:], time_support=ep)
print("running speed: %d samples, median %.1f cm/s" % (len(speed), np.median(speed.d)))

# %% [markdown]
# ### Raw data
#
# Before any analysis, look at the spikes. Each grey block below is one 2 s
# grating presentation, labelled with its drift direction. Several units visibly
# change their firing from one direction to the next.

# %%
vis = np.where(np.isin(spikes.region, ol.VISUAL_CORTEX))[0]
all_keys = list(spikes.keys())
vis_keys = [all_keys[i] for i in vis]
rates_all = np.array([spikes[k].rate for k in vis_keys])
# sample 30 units spanning the firing-rate range rather than the 30 fastest,
# so the raster stays legible
srt = np.argsort(rates_all)[::-1]
vis_keys = [vis_keys[i] for i in srt[np.linspace(0, len(srt) - 1, 30).astype(int)]]

t0 = dg.start_time.iloc[0] - 3
t1 = dg.start_time.iloc[0] + 27
win = nap.IntervalSet(start=t0, end=t1)

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 0.8, 1], hspace=0.18))
ax = axes[0]
for i, k in enumerate(vis_keys):
    tt = spikes[k].restrict(win).t
    ax.plot(tt, np.full_like(tt, i), "|", ms=4, color="k", alpha=0.85, mew=0.7)
ax.set_ylabel("visual cortex unit")
ax.set_ylim(-1, len(vis_keys))
ax.set_title("DANDI:000021 session %s - raw spiking during drifting gratings" % PROTOTYPE)

ax = axes[1]
for _, r in dg[(dg.start_time < t1) & (dg.stop_time > t0)].iterrows():
    ax.axvspan(r.start_time, r.stop_time, color="C0", alpha=0.25)
    ax.text(r.start_time + 1, 0.5, "%d" % r.direction, ha="center", va="center", fontsize=8)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("grating\ndirection (deg)")

ax = axes[2]
s = speed.restrict(win)
ax.plot(s.t, s.d, lw=0.8, color="C3")
ax.set_ylabel("running\n(cm/s)")
ax.set_xlabel("time (s)")
plt.savefig("figures/fig01_raw_data.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 3. Trial firing rates
#
# For every unit and every trial, count the spikes inside the presentation window
# and divide by its duration. The blank sweeps inside the drifting-grating block
# provide a matched baseline (same block, same mean luminance, no grating).

# %%
dg_rates, dg_counts = ol.trial_rates(spikes, dg)
sg_rates, _ = ol.trial_rates(spikes, sg)
dgb_rates, _ = ol.trial_rates(spikes, dg_blank)
print("drifting-grating rate matrix:", dg_rates.shape, "(trials x units)")
print("mean rate during gratings %.2f Hz, during blank sweeps %.2f Hz"
      % (dg_rates.mean(), dgb_rates.mean()))

# %% [markdown]
# ## 4. Per-unit orientation-selectivity metrics
#
# `ol.analyze_session` computes, for each unit:
#
# * **visual responsiveness** - the best direction x temporal-frequency condition
#   tested against the blank sweeps (Mann-Whitney, and a mean rate above 0.5 Hz).
#   Because that condition is picked as the maximum over the same data, the
#   p-value is Bonferroni-corrected for the 40 conditions the selection ranged
#   over, i.e. the threshold is 0.01/40;
# * the **direction tuning curve** at that unit's preferred temporal frequency
#   (15 repeats per direction);
# * the **global OSI**, `|sum_k r_k exp(2 i theta_k)| / sum_k r_k`, which is one
#   minus the circular variance of the response at twice the angle, and the
#   preferred orientation `arg(...)/2`; the analogous **global DSI** at one times
#   the angle; and the classic two-point
#   `OSI = (R_pref - R_orth)/(R_pref + R_orth)`;
# * a **permutation test**: the direction label is shuffled across trials 1000
#   times and the observed gOSI is compared to the resulting null distribution.
#   A unit counts as orientation-selective if it is responsive and p < 0.01;
# * **split-half** preferred orientations from two disjoint random halves of the
#   trials, and the preferred orientation from the independent static-grating block.

# %%
res, tun = ol.analyze_session(
    dg_rates, dgb_rates, dg.direction.values, dg.temporal_frequency.values,
    sg_rates, sg.ori.values, sel.unit_id.values, sel.region.values,
)
res["session_id"] = PROTOTYPE
res.to_csv(f"cache/res_{PROTOTYPE}.csv", index=False)
np.savez_compressed(f"cache/tc_{PROTOTYPE}.npz", **tun)
tc, tc_sem, dirs = tun["tc"], tun["tc_sem"], tun["dirs"]

print("%d units | %d visually responsive | %d orientation-selective"
      % (len(res), res.responsive.sum(), res.sig_ori.sum()))
print("\n%-6s %5s %6s %10s %10s" % ("region", "n", "resp", "med gOSI", "frac sig"))
for reg in ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]:
    s = res[res.region == reg]
    if len(s) < 5:
        continue
    print("%-6s %5d %6d %10.3f %10.2f"
          % (reg, len(s), s.responsive.sum(),
             s[s.responsive].gosi.median(), s.sig_ori.mean()))

# %% [markdown]
# ## 5. Example units
#
# Rasters are grouped by drift direction and coloured by direction; the middle
# column shows the corresponding PSTHs; the right column is the polar tuning curve
# (mean +/- SEM over 15 repeats) with a two-lobed von Mises fit. The dashed blue
# circle is the unit's blank-sweep rate.
#
# The von Mises concentration is capped so the half width at half maximum cannot
# fall below 22.5 degrees. The stimulus samples direction every 45 degrees, so
# anything narrower is not identifiable from these data and an unconstrained fit
# will place a spike between two measured points.

# %%
cand = res[res.region.isin(ol.VISUAL_CORTEX) & res.sig_ori & (res.peak_rate > 3)]
cand = pd.concat([cand[cand.region == "VISp"].sort_values("gosi", ascending=False),
                  cand[cand.region != "VISp"].sort_values("gosi", ascending=False)])
picks, used = [], []
for i, r in cand.iterrows():
    if all(ol.circ_dist_180(r.pref_ori, u) > 25 for u in used):
        picks.append(i)
        used.append(r.pref_ori)
    if len(picks) == 4:
        break
print(res.loc[picks, ["unit_id", "region", "gosi", "osi", "pref_ori", "pref_tf"]].to_string())

# %%
WIN = (-0.5, 2.5)
fig = plt.figure(figsize=(15, 14))
gs = GridSpec(4, 3, figure=fig, width_ratios=[1.5, 1.1, 1.0], hspace=0.62, wspace=0.34,
              top=0.9, bottom=0.05)
colors = plt.cm.hsv(np.linspace(0, 1, len(dirs), endpoint=False))
dg_start = dg.start_time.values
tf = dg.temporal_frequency.values
direction = dg.direction.values

for k, j in enumerate(picks):
    uid = res.unit_id.iloc[j]
    m_tf = tf == res.pref_tf.iloc[j]

    ax = fig.add_subplot(gs[k, 0])
    ytick, ylab, y = [], [], 0
    for di, dd in enumerate(dirs):
        ix = np.where(m_tf & (direction == dd))[0]
        pe = nap.compute_perievent(spikes[uid], nap.Ts(t=dg_start[ix]), WIN)
        y0 = y
        for e in pe.keys():
            tt = pe[e].t
            ax.plot(tt, np.full_like(tt, y), "|", ms=3, mew=0.8, color=colors[di])
            y += 1
        ytick.append((y0 + y) / 2)
        ylab.append("%d" % dd)
    ax.axvspan(0, 2, color="0.9", zorder=-2)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.set_ylim(-1, y)
    ax.set_xlim(*WIN)
    ax.set_ylabel("direction (deg)")
    ax.set_title("unit %d (%s)  gOSI=%.2f" % (uid, res.region.iloc[j], res.gosi.iloc[j]),
                 fontsize=10)
    if k == 3:
        ax.set_xlabel("time from grating onset (s)")

    ax = fig.add_subplot(gs[k, 1])
    bins = np.arange(WIN[0], WIN[1] + 0.05, 0.05)
    for di, dd in enumerate(dirs):
        ix = np.where(m_tf & (direction == dd))[0]
        pe = nap.compute_perievent(spikes[uid], nap.Ts(t=dg_start[ix]), WIN)
        allt = np.concatenate([pe[e].t for e in pe.keys()])
        h, _ = np.histogram(allt, bins=bins)
        ax.plot(bins[:-1] + 0.025, h / (len(ix) * 0.05), color=colors[di], lw=1.2,
                label="%d" % dd if k == 0 else None)
    ax.axvspan(0, 2, color="0.9", zorder=-2)
    ax.set_xlim(*WIN)
    ax.set_ylabel("rate (Hz)")
    if k == 0:
        ax.legend(fontsize=6.5, ncol=2, title="direction", title_fontsize=7,
                  loc="upper right", framealpha=0.85)
    if k == 3:
        ax.set_xlabel("time from onset (s)")

    ax = fig.add_subplot(gs[k, 2], projection="polar")
    th = np.deg2rad(dirs)
    r = tc[:, j]
    ax.errorbar(np.r_[th, th[0]], np.r_[r, r[0]],
                yerr=np.r_[tc_sem[:, j], tc_sem[0, j]],
                marker="o", ms=4, lw=1.4, color="k", capsize=2)
    popt, hwhm, r2 = ol.fit_von_mises(dirs, r)
    fine = np.linspace(0, 360, 361)
    ax.plot(np.deg2rad(fine), ol.double_von_mises(fine, *popt), color="C3", lw=1.6,
            alpha=0.85)
    ax.axhline(res.blank_rate.iloc[j], color="C0", ls="--", lw=1, alpha=0.7)
    ax.set_title("pref ori %.0f$\\degree$\nvon Mises HWHM %.0f$\\degree$"
                 % (res.pref_ori.iloc[j], hwhm), fontsize=9, pad=24)
    ax.tick_params(labelsize=7)

fig.suptitle("Orientation tuning of single units in mouse visual cortex "
             "(DANDI:000021, session %s)" % PROTOTYPE, fontsize=13, y=0.955)
plt.savefig("figures/fig02_example_units.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![example units](figures/fig02_example_units.png)
#
# Every example fires for two opposite directions of drift and is nearly silent
# for the orthogonal ones: the response depends on the axis of the grating, not on
# which way it moves. That is orientation selectivity.

# %% [markdown]
# ## 6. Selectivity across the visual hierarchy
#
# If the metric is measuring orientation selectivity and not an artefact, it must
# order the brain regions correctly: strong in cortex, weaker in visual thalamus,
# absent in a region that has nothing to do with early vision. CA1 units recorded
# on the same probes in the same sessions provide that negative control.

# %%
order = [r for r in ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]
         if (res.region == r).sum() >= 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
data = [res[(res.region == r) & res.responsive].gosi.values for r in order]
parts = ax.violinplot(data, showmedians=True, widths=0.85)
for pc, r in zip(parts["bodies"], order):
    pc.set_facecolor("C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6"))
    pc.set_alpha(0.7)
for i, dd in enumerate(data):
    ax.plot(np.random.normal(i + 1, 0.06, len(dd)), dd, ".", ms=2, color="k", alpha=0.35)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("global OSI")
ax.set_title("Orientation selectivity by region\n(visually responsive units)", fontsize=10)

ax = axes[1]
frac = [res[res.region == r].sig_ori.mean() for r in order]
n = [(res.region == r).sum() for r in order]
cols = ["C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6") for r in order]
ax.bar(range(len(order)), frac, color=cols)
for i, (fr, nn) in enumerate(zip(frac, n)):
    ax.text(i, fr + 0.015, "%d" % nn, ha="center", fontsize=8)
ax.axhline(0.01, color="r", ls="--", lw=1, label="test alpha (0.01)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("fraction orientation-selective")
ax.set_title("Significantly tuned units\n(permutation test, p<0.01)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
ctx = res[res.region.isin(ol.VISUAL_CORTEX) & res.responsive]
tha = res[res.region.isin(ol.THALAMUS) & res.responsive]
ax.plot(tha.gosi, tha.gdsi, "o", ms=4, color="C1", alpha=0.6, label="thalamus (LGd, LP)")
ax.plot(ctx.gosi, ctx.gdsi, "o", ms=4, color="C0", alpha=0.6, label="visual cortex")
ax.plot([0, 0.8], [0, 0.8], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.set_title("Orientation vs direction selectivity", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig03_population_by_region.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![by region](figures/fig03_population_by_region.png)
#
# Most points in the right panel sit below the diagonal: units are more selective
# for the axis of the grating than for its direction of motion, which is the
# signature of orientation rather than direction tuning.

# %% [markdown]
# ## 7. Is the tuning real? Two controls
#
# The gOSI of a noisy unit is never exactly zero, so a high value on its own is
# weak evidence. Two independent checks:
#
# * **Split-half.** Estimate the preferred orientation from a random half of the
#   trials and again from the other half. For a genuinely tuned unit the two agree;
#   for a noise-driven one they are unrelated (mean absolute difference 45 degrees
#   under a uniform null).
# * **Cross-stimulus.** Estimate it from the drifting gratings and, separately,
#   from the static-grating block, a different stimulus class with different
#   spatial frequencies, phases and a 0.25 s presentation. Agreement here cannot be
#   produced by overfitting the drifting-grating trials.

# %%
sel_ctx = res[res.region.isin(ol.VISUAL_CORTEX) & res.sig_ori]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
ax.plot(sel_ctx.pref_a, sel_ctx.pref_b, "o", ms=4, alpha=0.6, color="C0")
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
ax.set_xlabel("preferred orientation, trial half A (deg)")
ax.set_ylabel("half B (deg)")
ax.set_title("Split-half reliability\nmedian |$\\Delta$| = %.1f$\\degree$ (chance 45$\\degree$)"
             % sel_ctx.split_half_diff.median(), fontsize=10)

ax = axes[1]
ax.plot(sel_ctx.pref_ori, sel_ctx.sg_pref_ori, "o", ms=4, alpha=0.6, color="C2")
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
ax.set_xlabel("preferred orientation, drifting gratings (deg)")
ax.set_ylabel("static gratings (deg)")
ax.set_title("Cross-stimulus agreement\nmedian |$\\Delta$| = %.1f$\\degree$"
             % sel_ctx.ori_diff_dg_sg.median(), fontsize=10)

ax = axes[2]
bins = np.arange(0, 95, 7.5)
ax.hist(sel_ctx.ori_diff_dg_sg, bins=bins, color="C2", alpha=0.75, density=True,
        label="drifting vs static")
ax.hist(sel_ctx.split_half_diff, bins=bins, histtype="step", lw=2, color="C0",
        density=True, label="split-half (drifting)")
ax.axhline(1 / 90, color="k", ls="--", lw=1, label="chance (uniform)")
ax.set_xlabel("|$\\Delta$ preferred orientation| (deg)")
ax.set_ylabel("density")
ax.set_title("Consistency of preferred orientation", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig04_validation.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![validation](figures/fig04_validation.png)
#
# Points in the top-left and bottom-right corners of the scatter plots are not
# disagreements: orientation wraps at 180 degrees, so 175 and 5 are 10 degrees
# apart. The summary statistics use the wrapped circular distance.

# %% [markdown]
# ### Cross-validated population tuning
#
# Sorting units by their own peak and then plotting the same data would guarantee a
# peak even for pure noise. Here each unit's peak direction is taken from one random
# half of the trials and the curve from the **other** half is plotted, so a flat unit
# cannot manufacture a peak. Rates are divided by each unit's own mean, so an
# untuned population sits flat at 1.

# %%
tc_b_aligned = tun["tc_b_aligned"]
groups = [(res.region.isin(ol.VISUAL_CORTEX), "visual cortex"),
          (res.region.isin(ol.THALAMUS), "thalamus (LGd/LP)"),
          (res.region == "CA1", "hippocampus CA1 (control)")]
rel = np.r_[dirs[: len(dirs) // 2 + 1], dirs[len(dirs) // 2 + 1:] - 360]
order_rel = np.argsort(rel)
rel_sorted = np.sort(rel)

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5),
                         gridspec_kw=dict(height_ratios=[2, 1], hspace=0.35, wspace=0.42))
for c, (group, name) in enumerate(groups):
    m = (group & res.responsive).values
    z = tc_b_aligned[:, m]
    z = z / np.where(z.mean(0) == 0, np.nan, z.mean(0))
    z = z[order_rel]
    o = np.argsort(res.gosi.values[m])[::-1]
    ax = axes[0, c]
    im = ax.imshow(z[:, o].T, aspect="auto", cmap="magma", origin="lower",
                   vmin=0.2, vmax=2.0,
                   extent=[rel_sorted[0] - 22.5, rel_sorted[-1] + 22.5, 0, m.sum()])
    ax.set_xticks(rel_sorted)
    ax.set_ylabel("unit (sorted by gOSI)")
    ax.set_title("%s (n=%d responsive)" % (name, m.sum()), fontsize=10)
    ax.set_xlabel("direction relative to preferred (deg)", fontsize=8)
    if c == 2:
        plt.colorbar(im, ax=ax, label="rate / mean rate")

    ax = axes[1, c]
    mu = np.nanmean(z, axis=1)
    se = np.nanstd(z, axis=1) / np.sqrt(np.isfinite(z).sum(1))
    ax.errorbar(rel_sorted, mu, yerr=se, marker="o", ms=4, color="C3", capsize=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(0.3, 2.3)
    ax.set_xticks(rel_sorted)
    ax.set_xlabel("direction relative to preferred (deg)")
    if c == 0:
        ax.set_ylabel("population mean\n(rate / mean rate)")
fig.suptitle("Cross-validated tuning: peak direction from trial half A, "
             "response from half B", fontsize=12, y=0.97)
plt.savefig("figures/fig05_population_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![cross-validated tuning](figures/fig05_population_heatmap.png)
#
# The cortical population shows two peaks, at 0 and at 180 degrees relative to the
# preferred direction. A cell that preferred a direction of motion would show only
# the peak at 0. Two peaks 180 degrees apart is exactly what orientation tuning
# predicts. The thalamic population shows a much smaller single peak and CA1 is flat.

# %% [markdown]
# ## 8. Repeat across sessions and mice
#
# Everything above comes from one recording. `04_multi_session.py` runs the identical
# pipeline over the first `N_SESSIONS` sessions of the dandiset (one session per
# mouse) and pools the units. Results per session are cached, so re-running is cheap.

# %%
all_res, all_tcb = [], []
for _, srow in sessions.head(N_SESSIONS).iterrows():
    ses = srow["session_id"]
    if os.path.exists(f"cache/res_{ses}.csv"):
        r = pd.read_csv(f"cache/res_{ses}.csv", dtype={"session_id": str})
        z = np.load(f"cache/tc_{ses}.npz")
    else:
        print("processing session", ses)
        D = ol.load_and_prepare(srow["url"])
        r, z = ol.analyze_session(
            D["dg_rates"], D["dgb_rates"], D["dg"].direction.values,
            D["dg"].temporal_frequency.values, D["sg_rates"], D["sg"].ori.values,
            D["units"].unit_id.values, D["units"].region.values)
        r["session_id"] = ses
        r.to_csv(f"cache/res_{ses}.csv", index=False)
        np.savez_compressed(f"cache/tc_{ses}.npz", **z)
        del D
    r["subject_id"] = srow["subject_id"]
    all_res.append(r)
    all_tcb.append(z["tc_b_aligned"])

pooled = pd.concat(all_res, ignore_index=True)
tc_b_pool = np.hstack(all_tcb)
print("pooled: %d units, %d sessions, %d mice"
      % (len(pooled), pooled.session_id.nunique(), pooled.subject_id.nunique()))

summary = (pooled.groupby("region")
           .agg(n=("gosi", "size"), n_resp=("responsive", "sum"),
                frac_sig=("sig_ori", "mean")))
summary["median_gosi_responsive"] = pooled[pooled.responsive].groupby("region").gosi.median()
order = [r for r in ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]
         if (pooled.region == r).sum() >= 30]
print(summary.loc[order].round(3).to_string())
summary.loc[order].to_csv("results_by_region.csv")

# %%
col = {r: ("C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6"))
       for r in order}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax = axes[0]
data = [pooled[(pooled.region == r) & pooled.responsive].gosi.values for r in order]
parts = ax.violinplot(data, showmedians=True, widths=0.85)
for pc, r in zip(parts["bodies"], order):
    pc.set_facecolor(col[r])
    pc.set_alpha(0.75)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("global OSI")
ax.set_title("Orientation selectivity, all sessions pooled\n(visually responsive units)",
             fontsize=10)

ax = axes[1]
for i, r in enumerate(order):
    per_ses = pooled[pooled.region == r].groupby("session_id").sig_ori.mean()
    ax.bar(i, pooled[pooled.region == r].sig_ori.mean(), color=col[r])
    ax.plot(np.random.normal(i, 0.07, len(per_ses)), per_ses.values, "o", ms=4,
            color="k", alpha=0.7, mfc="none")
ax.axhline(0.01, color="r", ls="--", lw=1, label="test alpha (0.01)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("fraction orientation-selective")
ax.set_title("Significantly tuned units\n(bars = pooled, circles = single sessions)",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
rel = np.r_[dirs[:5], dirs[5:] - 360]
o = np.argsort(rel)
for group, name, c in [(pooled.region.isin(ol.VISUAL_CORTEX), "visual cortex", "C0"),
                       (pooled.region.isin(ol.THALAMUS), "thalamus", "C1"),
                       (pooled.region == "CA1", "CA1 (control)", "0.5")]:
    m = (group & pooled.responsive).values
    z = tc_b_pool[:, m]
    z = (z / np.where(z.mean(0) == 0, np.nan, z.mean(0)))[o]
    ax.errorbar(np.sort(rel), np.nanmean(z, 1),
                yerr=np.nanstd(z, 1) / np.sqrt(np.isfinite(z).sum(1)),
                marker="o", ms=4, color=c, capsize=3,
                label="%s (n=%d)" % (name, m.sum()))
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticks(np.sort(rel))
ax.set_xlabel("direction relative to preferred (deg)")
ax.set_ylabel("rate / mean rate")
ax.set_title("Cross-validated population tuning\n(peak from half A, response from half B)",
             fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig09_pooled_by_region.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![pooled](figures/fig09_pooled_by_region.png)

# %%
sel_pool = pooled[pooled.region.isin(ol.VISUAL_CORTEX) & pooled.sig_ori]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axes[0]
bins = np.arange(0, 181, 15)
h, _ = np.histogram(sel_pool.pref_ori, bins=bins)
ax.bar(bins[:-1] + 7.5, h, width=13, color="C0")
ax.axhline(len(sel_pool) / (len(bins) - 1), color="k", ls="--", lw=1, label="uniform")
chi2, p_uni = stats.chisquare(h)
ax.set_xlabel("preferred orientation (deg)")
ax.set_ylabel("number of units")
ax.set_xticks(np.arange(0, 181, 45))
ax.set_title("Preferred orientations across cortex\n"
             "$\\chi^2$ vs uniform: p = %.1e (n=%d)" % (p_uni, len(sel_pool)), fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
bins = np.arange(0, 95, 7.5)
ax.hist(sel_pool.ori_diff_dg_sg.dropna(), bins=bins, color="C2", alpha=0.75, density=True,
        label="drifting vs static gratings")
ax.hist(sel_pool.split_half_diff.dropna(), bins=bins, histtype="step", lw=2, color="C0",
        density=True, label="split-half (drifting)")
ax.axhline(1 / 90, color="k", ls="--", lw=1, label="chance")
ax.set_xlabel("|$\\Delta$ preferred orientation| (deg)")
ax.set_ylabel("density")
ax.set_title("Preferred orientation is stable\nacross trials and across stimuli", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
for group, name, c in [(pooled.region.isin(ol.VISUAL_CORTEX), "visual cortex", "C0"),
                       (pooled.region.isin(ol.THALAMUS), "thalamus", "C1")]:
    s = pooled[group & pooled.responsive]
    ax.plot(s.gosi, s.gdsi, "o", ms=3, alpha=0.45, color=c, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.set_title("Orientation vs direction selectivity\n(pooled)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig10_pooled_validation.png", dpi=140, bbox_inches="tight")
plt.close()

# %% [markdown]
# ![pooled validation](figures/fig10_pooled_validation.png)
#
# The distribution of preferred orientations is not uniform: cardinal orientations
# (0 and 90 degrees, i.e. horizontal and vertical gratings) are over-represented,
# a bias reported repeatedly in mouse V1.

# %% [markdown]
# ## 9. Encoding model and population decoding (optional extra)
#
# `05_glm_decoding.py`, shipped alongside this notebook, adds two model-based
# summaries: a cross-validated NeMoS Poisson GLM in which per-trial spike counts are
# regressed on the grating angle expanded in a cyclic B-spline basis and scored by
# held-out deviance explained, and a cross-validated multinomial decoder that reads
# the presented direction out of population spike counts. Fitting the population GLM
# is expensive (tens of seconds per fold per session), so it was not run to
# completion for this write-up and no numbers from it are reported below. Run
# `python 05_glm_decoding.py 4` to produce `cache/glm_results.csv`,
# `cache/decoding_results.csv` and figures 6 to 8.

# %%
GLM_PATH = "cache/glm_results.csv"
if os.path.exists(GLM_PATH):
    glm = pd.read_csv(GLM_PATH)
    dec = pd.read_csv("cache/decoding_results.csv")
    print("cross-validated deviance explained by grating angle (median):")
    print(glm.groupby("region")[["dev_dir", "dev_ori"]].median()
          .sort_values("dev_dir", ascending=False).to_string())
    print("\ndecoding accuracy (mean over sessions):")
    print(dec.groupby("group")[["acc_dir", "acc_ori"]].mean().to_string())
else:
    print("not computed in this run; see 05_glm_decoding.py")

# %% [markdown]
# ## 10. Summary
#
# Roughly half of the quality-passing units recorded in mouse visual cortex in this
# dandiset are significantly orientation-selective by a permutation test on the
# global OSI, with a median gOSI around 0.2 among visually responsive units and
# von Mises half-widths of a few tens of degrees. The same measurement applied to
# visual thalamus gives markedly weaker selectivity and, applied to hippocampal CA1
# units recorded on the same probes in the same sessions, returns a significant
# fraction at the level of the test's own false-positive rate. Preferred orientation
# is reproducible across disjoint halves of the trials and transfers to an
# independent static-grating stimulus, and the cross-validated population tuning
# curve peaks at both 0 and 180 degrees relative to the preferred direction, which
# distinguishes orientation tuning from direction tuning.

# %%
print("done - figures in ./figures")
