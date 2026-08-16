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
# # Decoding hippocampal replay during sharp-wave ripples
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark & Buzsáki (2016),
# *"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences"*
# (the `hc-11` dataset). Dorsal CA1 silicon-probe recordings from four rats, each session
# structured as PRE-sleep → novel linear track → POST-sleep, with 128-channel LFP at 1250 Hz
# and spike-sorted units labelled as putative excitatory or inhibitory.
#
# **Question.** During a sharp-wave ripple (SWR), does the CA1 population re-express an ordered
# spatial trajectory from the track the animal ran earlier?
#
# **Approach.**
#
# 1. Build direction-specific place-field templates from running on the linear track.
# 2. Validate the templates by Bayesian-decoding the animal's real position during running.
# 3. Detect SWRs in the 140–250 Hz band of the pyramidal-layer LFP during PRE and POST sleep.
# 4. Take population-burst events coincident with a ripple as candidate replay events, decode
#    each in 20 ms bins, and score the posterior with the weighted correlation between decoded
#    position and time.
# 5. Test each event against a column-cycle shuffle and a time-bin-permutation shuffle, and
#    calibrate the whole procedure with a cell-identity shuffle of the place fields.
#
# All data are streamed from the DANDI S3 bucket with `remfile` (chunk-level disk cache); no
# file is downloaded in full. Every computation on spike trains, intervals and position uses
# Pynapple.

# %%
import os
import warnings

import matplotlib
matplotlib.use("Agg")           # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.stats import chi2_contingency, mannwhitneyu
from tqdm.auto import tqdm

import replaylib as R      # streaming, ripple detection, Bayesian decoder
import pipeline as P       # the same analysis packaged for reuse across sessions

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("figures", exist_ok=True)
SESSION = "Achilles-10252013"
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Load the session and inspect every data stream
#
# `open_session` resolves the DANDI asset id to a presigned S3 URL and opens the NWB file
# through `remfile` + `h5py` + `pynwb`, then wraps it in a `pynapple.NWBFile`.

# %%
h5, nwbfile, nwb = R.open_session(SESSION)
print(nwb)
print("subject:", nwbfile.subject.subject_id, "| species:", nwbfile.subject.species)
print("\nEpochs:")
print(nwbfile.epochs.to_dataframe())

epochs = {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
          for row in nwbfile.epochs.to_dataframe().itertuples()}
states_df = nwbfile.processing["behavior"]["states"].to_dataframe()
print("\nScored brain states:", states_df.label.value_counts().to_dict())

# %%
units = nwb["units"]
cell_type = np.array(nwbfile.units["cell_type"][:])
location = np.array(nwbfile.units["location"][:])
lfp_rate, lfp_t0, n_t, n_ch = R.lfp_meta(h5)
print(f"{len(units)} units: {(cell_type == 'excitatory').sum()} excitatory, "
      f"{(cell_type == 'inhibitory').sum()} inhibitory")
print("recording sites:", {u: int((location == u).sum()) for u in np.unique(location)})
print(f"LFP: {n_ch} channels @ {lfp_rate} Hz, {n_t / lfp_rate / 3600:.2f} h")

# %% [markdown]
# ### Behaviour
#
# The archived `LinearizedPosition` is masked to the authors' own run epochs, which leaves only
# about 4 minutes of behaviour. It is, however, an exact affine function of the raw x coordinate
# (residual below 1e-15 m), so `load_position` recovers that map and applies it to every tracked
# sample that lies on the track. That recovers about 26 minutes of position and both running
# directions, without inventing a linearisation of our own.

# %%
position, posinfo = R.load_position(h5)
speed = R.compute_speed(position)
run_dirs = R.direction_intervals(position, speed, min_speed=P.MIN_SPEED)
run_all = run_dirs["right"].union(run_dirs["left"])
print("affine map recovered from the archived linearisation:", posinfo)
print(f"rightward traversals: {len(run_dirs['right'])} "
      f"({run_dirs['right'].tot_length():.0f} s)")
print(f"leftward  traversals: {len(run_dirs['left'])} "
      f"({run_dirs['left'].tot_length():.0f} s)")

# %% [markdown]
# ### Ripple-channel selection
#
# Ripples are sparse, high-amplitude transients confined to the CA1 pyramidal layer, so we score
# every channel by the fraction of time its ripple-band envelope exceeds 5 robust SD over a
# two-minute slice of POST sleep and keep the densest one.
#
# Both the score and the detector below use a median/MAD scale rather than mean/SD. That is not
# cosmetic: on one session a handful of artifacts reaching 20x the ripple amplitude inflated the
# SD enough that genuine ripples never crossed a 4 SD threshold, and the detected rate collapsed
# to 0.016 Hz against 0.25 Hz elsewhere. The median and the MAD ignore those few samples.

# %%
probe_start = float(epochs["POSTEpoch"].start[0]) + 300.0
ch_scores = []
for ch in tqdm(range(n_ch), desc="scoring channels"):
    x = R.read_lfp(h5, [ch], probe_start, probe_start + 120.0).values[:, 0]
    if np.allclose(x, 0):
        ch_scores.append(np.nan)
        continue
    _, env = R.ripple_envelope(x, lfp_rate)
    ch_scores.append(R.ripple_density(env))
ch_scores = np.array(ch_scores)
best_ch = int(np.nanargmax(ch_scores))
print(f"best ripple channel: {best_ch} (density {ch_scores[best_ch]:.4f})")

# %% [markdown]
# ### Figure 1 — session overview

# %%
nrem_rows = states_df[states_df.label == "Non-REM"]
nrem = nap.IntervalSet(start=nrem_rows.start_time.values, end=nrem_rows.stop_time.values)
maze = epochs["MazeEpoch"]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
epoch_colors = {"PREEpoch": "tab:blue", "MazeEpoch": "tab:green", "POSTEpoch": "tab:red"}
for lab, iv in epochs.items():
    ax.axvspan(iv.start[0] / 3600, iv.end[0] / 3600, alpha=0.3, color=epoch_colors[lab],
               label=f"{lab.replace('Epoch', '')} ({iv.tot_length() / 3600:.1f} h)")
for s, e in zip(nrem.start, nrem.end):
    ax.axvspan(s / 3600, e / 3600, ymin=0.0, ymax=0.25, color="k", alpha=0.55, lw=0)
ax.set_xlim(0, n_t / lfp_rate / 3600)
ax.set_yticks([])
ax.set_xlabel("time (h)")
ax.set_title(f"{SESSION}: session structure; black bars = scored non-REM")
ax.legend(loc="upper center", ncol=3, fontsize=9, framealpha=0.95)

ax = fig.add_subplot(gs[1, :])
ax.plot(position.t, position.d, "k.", ms=0.6)
ax.set_xlim(maze.start[0], maze.end[0])
ax.set_xlabel("time (s)")
ax.set_ylabel("linear position (m)")
ax.set_title("Linear-track behaviour (full maze epoch)")

ax = fig.add_subplot(gs[2, 0])
w0, w1 = maze.start[0] + 250, maze.start[0] + 400
w = (position.t > w0) & (position.t < w1)
ax.plot(position.t[w], position.d[w], "k-", lw=1)
for name, col in (("right", "tab:red"), ("left", "tab:blue")):
    for s, e in zip(run_dirs[name].start, run_dirs[name].end):
        if s > w0 and e < w1:
            m = (position.t >= s) & (position.t <= e)
            ax.plot(position.t[m], position.d[m], color=col, lw=2)
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.set_title("Traversals: rightward (red) / leftward (blue)", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
ax.hist(speed.d[np.isfinite(speed.d)], bins=60, color="0.4")
ax.axvline(P.MIN_SPEED, color="r", ls="--", label=f"run threshold {P.MIN_SPEED * 100:.0f} cm/s")
ax.set_xlabel("speed (m/s)")
ax.set_ylabel("samples")
ax.set_title("Speed distribution", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[3, 0])
ax.plot(ch_scores, "k.-", ms=3, lw=0.5)
ax.plot(best_ch, ch_scores[best_ch], "r*", ms=14)
ax.set_xlabel("LFP channel")
ax.set_ylabel("fraction of time above 5 robust SD")
ax.set_title(f"Ripple channel selection (best = ch {best_ch})", fontsize=10)

ax = fig.add_subplot(gs[3, 1])
rates = np.array([len(units[i]) / (n_t / lfp_rate) for i in units.index])
ax.hist(rates[cell_type == "excitatory"], bins=np.logspace(-2, 1.6, 30), alpha=0.7,
        label="excitatory")
ax.hist(rates[cell_type == "inhibitory"], bins=np.logspace(-2, 1.6, 30), alpha=0.7,
        label="inhibitory")
ax.set_xscale("log")
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("units")
ax.set_title("Unit firing rates", fontsize=10)
ax.legend(fontsize=8)
fig.savefig("figures/01_session_overview.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## 2. Place fields and decoder validation
#
# Tuning curves are computed separately for rightward and leftward traversals in 4 cm bins and
# smoothed with a 1.5-bin Gaussian. The decoding ensemble is every putative excitatory cell with
# a peak rate of at least 1 Hz that fired at least 50 spikes while running.
#
# Selection is deliberately on firing rather than on spatial information. A fixed Skaggs
# information threshold of 0.3 bits/spike was tried first and left as few as 11 cells in one
# session, which more than doubled the cross-validated decoding error there (19.6 cm against
# 10.5 cm); the Bayesian decoder tolerates weakly tuned cells because their near-flat tuning
# curves contribute an almost position-independent term to the likelihood. Spatial information
# is still computed and plotted, but only descriptively.

# %%
pf = P.build_templates(h5, nwb, nwbfile)
centers, edges = pf["centers"], pf["edges"]
templates, place_ids = pf["templates"], pf["place_ids"]
place_units = nwb["units"][list(place_ids)]
print(f"spatial grid: {len(centers)} bins of {P.BIN_CM} cm "
      f"spanning {edges[0]:.2f}-{edges[-1]:.2f} m")
print(f"decoding ensemble: {len(place_ids)} / {pf['n_pyr']} pyramidal cells")

# %% [markdown]
# The decoder used on ripples is the memoryless Bayesian decoder of Zhang et al. (1998):
#
# $$P(x \mid n) \propto P(x)\ \prod_i f_i(x)^{n_i} \exp\!\big(-\tau \textstyle\sum_i f_i(x)\big)$$
#
# Before trusting it on 20 ms ripple bins, we check that it recovers the animal's real position
# during running, both in-sample and with templates fit on odd laps and tested on even laps.

# %%
VAL_BIN = 0.25
counts_val = place_units.count(VAL_BIN, run_all)
true_pos = position.bin_average(VAL_BIN, run_all)
in_right = np.zeros(len(counts_val), dtype=bool)
for s, e in zip(run_dirs["right"].start, run_dirs["right"].end):
    in_right |= (counts_val.t >= s) & (counts_val.t <= e)
occ = {d: np.histogram(position.restrict(run_dirs[d]).d, bins=edges)[0]
       for d in ("right", "left")}
post_val = np.zeros((len(counts_val), len(centers)))
for d, mask in (("right", in_right), ("left", ~in_right)):
    post_val[mask] = R.bayesian_decode(counts_val.values[mask], templates[d], VAL_BIN,
                                       prior=occ[d] / occ[d].sum())
decoded = centers[np.argmax(post_val, axis=1)]
ok = np.isfinite(true_pos.values) & (counts_val.values.sum(axis=1) > 0)
err = np.abs(decoded[ok] - true_pos.values[ok])

cv_err = []
for d in ("right", "left"):
    iv = run_dirs[d]
    odd = nap.IntervalSet(start=iv.start[::2], end=iv.end[::2])
    even = nap.IntervalSet(start=iv.start[1::2], end=iv.end[1::2])
    tmpl_odd = P._tuning(place_units, position, odd, edges)
    c = place_units.count(VAL_BIN, even)
    p_even = R.bayesian_decode(c.values, tmpl_odd, VAL_BIN)
    truth = position.bin_average(VAL_BIN, even)
    good = np.isfinite(truth.values) & (c.values.sum(axis=1) > 0)
    cv_err.append(np.abs(centers[np.argmax(p_even, axis=1)][good] - truth.values[good]))
cv_err = np.concatenate(cv_err)
print(f"in-sample median decoding error   : {np.median(err) * 100:.1f} cm "
      f"(r = {np.corrcoef(decoded[ok], true_pos.values[ok])[0, 1]:.3f})")
print(f"odd-lap -> even-lap median error  : {np.median(cv_err) * 100:.1f} cm")

# %% [markdown]
# ### Figure 2 — place fields and decoder validation

# %%
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.42)
order = np.argsort(np.argmax(templates["right"], axis=1))
for k, d in enumerate(("right", "left")):
    ax = fig.add_subplot(gs[0, k])
    norm = templates[d][order] / np.clip(templates[d][order].max(axis=1, keepdims=True),
                                         1e-9, None)
    im = ax.imshow(norm, aspect="auto", origin="lower", cmap="viridis",
                   extent=[centers[0], centers[-1], 0, len(place_ids)])
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted by rightward peak)" if k == 0 else "")
    ax.set_title(f"{d}ward-run place fields (n={len(place_ids)})", fontsize=11)
    plt.colorbar(im, ax=ax, label="normalised rate", fraction=0.046)

ax = fig.add_subplot(gs[0, 2])
for i in np.linspace(0, len(place_ids) - 1, 8).astype(int):
    ax.plot(centers, templates["right"][order[i]], lw=1.5)
ax.set_xlabel("position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example place fields (rightward runs)", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
ax.hist(pf["si_all"], bins=30, color="0.4", label="all pyramidal")
ax.hist(pf["si"], bins=30, color="tab:red", alpha=0.7, label="decoding ensemble")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("cells")
ax.set_title("Spatial information", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
n_show = 400
ax.plot(np.arange(n_show) * VAL_BIN, true_pos.values[ok][:n_show], "k-", lw=2,
        label="actual")
ax.plot(np.arange(n_show) * VAL_BIN, decoded[ok][:n_show], ".", color="tab:red", ms=4,
        label="decoded")
ax.set_xlabel("time within concatenated run bins (s)")
ax.set_ylabel("position (m)")
ax.set_title(f"Decoder validation ({VAL_BIN * 1000:.0f} ms bins)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 2])
bins_e = np.linspace(0, 60, 40)
ax.hist(err * 100, bins=bins_e, color="0.4", alpha=0.7, density=True,
        label=f"in-sample (median {np.median(err) * 100:.1f} cm)")
ax.hist(cv_err * 100, bins=bins_e, color="tab:red", alpha=0.55, density=True,
        label=f"held-out laps (median {np.median(cv_err) * 100:.1f} cm)")
ax.set_xlabel("decoding error (cm)")
ax.set_ylabel("density")
ax.set_title("Decoding error during running", fontsize=11)
ax.legend(fontsize=7)
fig.savefig("figures/02_place_fields.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## 3. Sharp-wave ripple detection
#
# The chosen channel is band-passed at 140–250 Hz, Hilbert-transformed and the envelope smoothed
# over 8 ms. Events are excursions above 3 robust SD that reach 6 robust SD, lasting 30–300 ms;
# REM periods are excluded. The LFP is read in 30-minute blocks so nothing large is held in memory at once.

# %%
rem_rows = states_df[states_df.label == "REM"]
rem = nap.IntervalSet(start=rem_rows.start_time.values, end=rem_rows.stop_time.values)

ripples = {}
for name in ("PREEpoch", "POSTEpoch"):
    iv, pk_t, pk_z = P.detect_epoch_ripples(h5, epochs[name], best_ch, lfp_rate, rem)
    ripples[name] = dict(iv=iv, peak_t=pk_t, peak_z=pk_z)
    print(f"{name}: {len(iv)} ripples "
          f"({len(iv) / epochs[name].tot_length():.3f} Hz), "
          f"median duration {np.median(iv.end - iv.start) * 1000:.0f} ms")

# %% [markdown]
# ### Figure 3 — ripple validation
#
# Example events, the ripple-triggered average and the event spectrum confirm that the detector
# picks up genuine ~170 Hz oscillations riding on sharp waves, and that their rate tracks non-REM.

# %%
pk = ripples["POSTEpoch"]["peak_t"]
pk_z = ripples["POSTEpoch"]["peak_z"]
rip_iv = ripples["POSTEpoch"]["iv"]
order_z = np.argsort(pk_z)[::-1]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 3, hspace=0.65, wspace=0.3)
for k, j in enumerate(order_z[[3, 20, 60]]):
    tc = pk[j]
    seg = R.read_lfp(h5, [best_ch], tc - 0.25, tc + 0.25)
    raw = seg.values[:, 0]
    f_, e_ = R.ripple_envelope(raw, lfp_rate)
    ax = fig.add_subplot(gs[0, k])
    ax.plot((seg.t - tc) * 1000, raw, "k", lw=0.7)
    ax.axvspan((rip_iv.start[j] - tc) * 1000, (rip_iv.end[j] - tc) * 1000,
               color="tab:orange", alpha=0.25)
    ax.set_title(f"raw LFP, ripple #{j} (peak {pk_z[j]:.1f} SD)", fontsize=9)
    if k == 0:
        ax.set_ylabel("a.u.")
    ax = fig.add_subplot(gs[1, k])
    ax.plot((seg.t - tc) * 1000, f_, "k", lw=0.7)
    ax.plot((seg.t - tc) * 1000, e_, "tab:red", lw=1.2)
    ax.set_xlabel("time from ripple peak (ms)")
    if k == 0:
        ax.set_ylabel("140-250 Hz")
    ax.set_title("ripple-band + envelope", fontsize=9)

# Ripple-triggered average, computed from a one-hour slice of POST sleep.
slice_start = float(epochs["POSTEpoch"].start[0])
slice_stop = slice_start + 3600.0
seg = R.read_lfp(h5, [best_ch], slice_start, slice_stop)
filt, _ = R.ripple_envelope(seg.values[:, 0], lfp_rate)
win = int(0.15 * lfp_rate)
idx = np.searchsorted(seg.t, pk[(pk > slice_start) & (pk < slice_stop)])
idx = idx[(idx > win) & (idx < len(seg.t) - win)]
snips = np.stack([filt[i - win:i + win] for i in idx])

ax = fig.add_subplot(gs[2, 0])
ax.plot((np.arange(-win, win) / lfp_rate) * 1000, snips.mean(axis=0), "k", lw=1)
ax.set_xlabel("time from peak (ms)")
ax.set_ylabel("mean filtered LFP")
ax.set_title(f"Ripple-triggered average (n={len(snips)})", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
fw, pw = welch(snips, fs=lfp_rate, nperseg=min(256, snips.shape[1]))
ax.semilogy(fw, pw.mean(axis=0), "k")
ax.set_xlim(0, 400)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power")
ax.set_title("Spectrum of detected events", fontsize=10)

ax = fig.add_subplot(gs[2, 2])
for name, col in (("PREEpoch", "tab:blue"), ("POSTEpoch", "tab:red")):
    d = (ripples[name]["iv"].end - ripples[name]["iv"].start) * 1000
    ax.hist(d, bins=np.arange(30, 305, 5), alpha=0.55, density=True, color=col,
            label=name.replace("Epoch", ""))
ax.set_xlabel("ripple duration (ms)")
ax.set_ylabel("density")
ax.set_title("Event durations", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[3, :])
allpk = np.concatenate([ripples[n]["peak_t"] for n in ripples])
tbins = np.arange(0, n_t / lfp_rate + 60, 60)
ax.plot(tbins[:-1] / 3600, np.histogram(allpk, bins=tbins)[0] / 60, "k", lw=0.8)
for s, e in zip(nrem.start, nrem.end):
    ax.axvspan(s / 3600, e / 3600, color="tab:green", alpha=0.12, lw=0)
for name, col in epoch_colors.items():
    ax.axvline(epochs[name].start[0] / 3600, color=col, ls="--")
    ax.text(epochs[name].start[0] / 3600 + 0.05, ax.get_ylim()[1] * 0.9,
            name.replace("Epoch", ""), color=col, fontsize=9)
ax.set_xlabel("time (h)")
ax.set_ylabel("ripple rate (Hz)")
ax.set_title("Ripple rate across the session (green = scored non-REM)", fontsize=10)
fig.savefig("figures/03_ripples.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## 4. Candidate events and Bayesian decoding of the replayed trajectory
#
# A ripple alone is not a replay event. Following the standard definition, candidate events are
# **population-burst events** (smoothed multiunit rate of the place-cell ensemble crossing 3 SD,
# 100–500 ms long) that **contain a detected ripple peak**, with at least 5 active place cells.
# Each event is decoded in 20 ms bins against both direction templates.
#
# The sequence statistic is the posterior-weighted correlation between decoded position and
# time within the event, and its regression slope gives the speed of the virtual trajectory.
# Significance requires *both* shuffles to give p < 0.025 (Bonferroni for two templates):
#
# * **column-cycle shuffle** — each time bin's posterior is circularly shifted in position,
#   destroying spatial alignment across bins while preserving each bin's posterior shape;
# * **time-bin permutation** — the order of the time bins is permuted, destroying temporal order
#   while preserving the set of decoded locations.

# %%
results = {}
for name in ("POSTEpoch", "PREEpoch"):
    events, burst_peak = P.population_bursts(place_units, epochs[name],
                                             ripples[name]["peak_t"])
    df = P.score_events(place_units, events, burst_peak, templates, centers, rng,
                        desc=f"decoding {name}")
    results[name] = df
    print(f"{name}: {len(df)} candidate events, {df.significant.sum()} significant "
          f"({100 * df.significant.mean():.1f}%)")

# %% [markdown]
# ### The critical control
#
# How often would this procedure call an event significant if there were no spatial code at all?
# Reassigning the place fields to random cells destroys the spatial code while preserving every
# spike time, every event boundary and every population statistic. The fraction of "significant"
# events it yields is the empirical false-positive rate of the entire pipeline.

# %%
events_post, peak_post = P.population_bursts(place_units, epochs["POSTEpoch"],
                                             ripples["POSTEpoch"]["peak_t"])
perm = rng.permutation(len(place_ids))
tmpl_shuf = {d: templates[d][perm] for d in templates}
results["control"] = P.score_events(place_units, events_post, peak_post, tmpl_shuf,
                                    centers, rng, desc="decoding control")

post_df, pre_df, ctrl_df = results["POSTEpoch"], results["PREEpoch"], results["control"]
for lab, df in (("cell-ID shuffled", ctrl_df), ("PRE sleep", pre_df),
                ("POST sleep", post_df)):
    print(f"{lab:18s}: {100 * df.significant.mean():5.1f}% significant "
          f"({df.significant.sum()}/{len(df)}), median |r| = {df.r.abs().median():.3f}")

def sig_table(a, b):
    return [[a.significant.sum(), (~a.significant).sum()],
            [b.significant.sum(), (~b.significant).sum()]]


chi2, p_chi, _, _ = chi2_contingency(sig_table(post_df, pre_df))
chi2c, p_ctrl, _, _ = chi2_contingency(sig_table(post_df, ctrl_df))
chi2p, p_prectrl, _, _ = chi2_contingency(sig_table(pre_df, ctrl_df))
_, p_u = mannwhitneyu(post_df.r.abs(), pre_df.r.abs(), alternative="greater")
_, p_u_ctrl = mannwhitneyu(post_df.r.abs(), ctrl_df.r.abs(), alternative="greater")
_, p_u_prectrl = mannwhitneyu(pre_df.r.abs(), ctrl_df.r.abs(), alternative="greater")
print(f"\nprevalence  POST vs PRE     : chi2 = {chi2:.1f}, p = {p_chi:.2g}")
print(f"prevalence  POST vs control : chi2 = {chi2c:.1f}, p = {p_ctrl:.2g}")
print(f"prevalence  PRE  vs control : chi2 = {chi2p:.1f}, p = {p_prectrl:.2g}")
print(f"|r|         POST vs PRE     : Mann-Whitney p = {p_u:.2g}")
print(f"|r|         POST vs control : Mann-Whitney p = {p_u_ctrl:.2g}")
print(f"|r|         PRE  vs control : Mann-Whitney p = {p_u_prectrl:.2g}")

# %% [markdown]
# ### Figure 4 — individual replay events
#
# For each event: the raw LFP with the ripple visible, the place-cell raster ordered by field
# position (an ordered diagonal is replay seen directly in the spikes), and the decoded posterior.

# %%
field_peak = centers[np.argmax(pf["tc_all"], axis=1)]
sort_by_field = np.argsort(field_peak)
sig = post_df[post_df.significant].copy()
sig["absr"] = sig.r.abs()
examples = sig.sort_values("absr", ascending=False).head(6)


def decode_event(row):
    ev = nap.IntervalSet(start=row.t_start, end=row.t_end)
    c = place_units.count(P.DEC_BIN, ev)
    return R.bayesian_decode(c.values, templates[row.direction], P.DEC_BIN), c.t - c.t[0]


fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 6, height_ratios=[0.55, 1, 1.2], hspace=0.18, wspace=0.35)
for k, (_, row) in enumerate(examples.iterrows()):
    pad = 0.05
    seg = R.read_lfp(h5, [best_ch], row.t_start - pad, row.t_end + pad)
    ax = fig.add_subplot(gs[0, k])
    ax.plot((seg.t - row.t_start) * 1000, seg.values[:, 0], "k", lw=0.6)
    ax.axvspan(0, (row.t_end - row.t_start) * 1000, color="tab:orange", alpha=0.2)
    ax.set_xticks([])
    ax.set_title(f"event {int(row.event)} ({row.direction})\n"
                 f"r={row.r:+.2f}, {abs(row.slope):.1f} m/s", fontsize=9)
    if k == 0:
        ax.set_ylabel("LFP")

    ax = fig.add_subplot(gs[1, k])
    for j, ui in enumerate(np.array(place_units.index)[sort_by_field]):
        st = place_units[ui].t
        st = st[(st >= row.t_start - pad) & (st <= row.t_end + pad)]
        if len(st):
            ax.plot((st - row.t_start) * 1000, np.full(len(st), j), "|", color="k",
                    ms=3, mew=0.8)
    ax.axvspan(0, (row.t_end - row.t_start) * 1000, color="tab:orange", alpha=0.2)
    ax.set_ylim(-1, len(place_ids))
    ax.set_xticks([])
    if k == 0:
        ax.set_ylabel("cell (by field position)")

    post, tt = decode_event(row)
    ax = fig.add_subplot(gs[2, k])
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (tt[-1] + P.DEC_BIN) * 1000, centers[0], centers[-1]])
    ax.plot((tt + P.DEC_BIN / 2) * 1000, centers[np.argmax(post, axis=1)], "w.", ms=5)
    ax.set_xlabel("time in event (ms)")
    if k == 0:
        ax.set_ylabel("decoded position (m)")
fig.suptitle("Decoded trajectories during POST-sleep sharp-wave ripples "
             "(top: raw LFP; middle: place-cell raster; bottom: posterior)",
             fontsize=13, y=0.96)
fig.savefig("figures/04_replay_examples.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ### Figure 5 — replay statistics for this session

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 26)
ax.hist(ctrl_df.r.abs(), bins=bins, density=True, histtype="step", lw=2, color="k",
        label=f"cell-ID shuffled (n={len(ctrl_df)})")
ax.hist(pre_df.r.abs(), bins=bins, density=True, alpha=0.55, color="tab:blue",
        label=f"PRE (n={len(pre_df)})")
ax.hist(post_df.r.abs(), bins=bins, density=True, alpha=0.55, color="tab:red",
        label=f"POST (n={len(post_df)})")
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.set_title(f"Sequence score\nPOST > control p = {p_u_ctrl:.1e}; "
             f"PRE > control p = {p_u_prectrl:.2f}", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1])
frac = [100 * ctrl_df.significant.mean(), 100 * pre_df.significant.mean(),
        100 * post_df.significant.mean()]
ns = [len(ctrl_df), len(pre_df), len(post_df)]
errbar = [100 * np.sqrt(f / 100 * (1 - f / 100) / m) for f, m in zip(frac, ns)]
ax.bar(["cell-ID\nshuffled", "PRE", "POST"], frac, yerr=errbar,
       color=["0.6", "tab:blue", "tab:red"], capsize=6)
ax.axhline(frac[0], color="k", ls="--", lw=1, label=f"empirical chance ({frac[0]:.1f}%)")
for i, (f, m) in enumerate(zip(frac, ns)):
    ax.text(i, f + errbar[i] + 0.6, f"{f:.1f}%\n({int(round(f / 100 * m))}/{m})",
            ha="center", fontsize=9)
ax.set_ylabel("significant replay events (%)")
ax.set_title(f"Replay prevalence\nPOST vs control p = {p_ctrl:.1e}; "
             f"PRE vs control p = {p_prectrl:.2f}", fontsize=11)
ax.set_ylim(0, max(frac) * 1.5)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[0, 2])
sp = post_df.loc[post_df.significant, "slope"].abs()
ax.hist(sp, bins=np.linspace(0, 30, 31), color="tab:red", alpha=0.8)
ax.axvline(sp.median(), color="k", ls="--", label=f"median {sp.median():.1f} m/s")
ax.set_xlabel("replay speed |slope| (m/s)")
ax.set_ylabel("events")
ax.set_title("Virtual trajectory speed (POST)", fontsize=11)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(post_df.n_active, post_df.r.abs(), s=6, c="0.7", label="all candidates")
ax.scatter(sig.n_active, sig.r.abs(), s=8, c="tab:red", label="significant")
ax.set_xlabel("active place cells in event")
ax.set_ylabel("|weighted correlation|")
ax.set_title("Sequence score vs. event participation", fontsize=11)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 1])
ax.bar(["forward\n(+slope)", "reverse\n(-slope)"],
       [(sig.slope > 0).sum(), (sig.slope < 0).sum()],
       color=["tab:green", "tab:purple"])
ax.set_ylabel("significant events")
ax.set_title("Direction of the decoded trajectory (POST)", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
t0 = post_df.t_start.min()
bw = 600.0
tedges = np.arange(0, post_df.t_start.max() - t0 + bw, bw)
ax.plot(tedges[:-1] / 60, np.histogram(post_df.t_start - t0, bins=tedges)[0] / bw * 60,
        color="0.6", label="candidate events")
ax.plot(tedges[:-1] / 60,
        np.histogram(post_df.loc[post_df.significant, "t_start"] - t0, bins=tedges)[0] / bw * 60,
        color="tab:red", label="significant replay")
ax.set_xlabel("time into POST sleep (min)")
ax.set_ylabel("events / min")
ax.set_title("Replay across POST sleep", fontsize=11)
ax.legend(fontsize=9)
fig.suptitle("Replay statistics: POST-sleep ripples carry ordered spatial trajectories "
             "far above chance, PRE-sleep ripples do not", fontsize=13, y=0.97)
fig.savefig("figures/05_replay_statistics.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ### Figure 6 — a gallery of decoded replay trajectories

# %%
best = sig.sort_values("absr", ascending=False).head(24)
fig, axes = plt.subplots(4, 6, figsize=(16, 9))
for ax, (_, row) in zip(axes.ravel(), best.iterrows()):
    post, tt = decode_event(row)
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (tt[-1] + P.DEC_BIN) * 1000, centers[0], centers[-1]])
    ax.set_title(f"r={row.r:+.2f} p={row.pcyc:.3f}", fontsize=8, pad=2)
    ax.tick_params(labelsize=7)
for ax in axes[-1]:
    ax.set_xlabel("ms", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("pos (m)", fontsize=8)
fig.suptitle("Gallery of decoded POST-sleep replay events (posterior probability)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/06_replay_gallery.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## 5. Across sessions
#
# Five of the eight sessions in the dandiset use a straight track (four 1.6 m, one 2 m); the
# other three use a circular maze, whose linearisation is an arc length rather than an affine
# function of x, and are excluded. Running `06_multi_session.py` executes the identical
# pipeline on each straight-track session; the cell below loads and plots its output. It takes
# roughly half an hour, so re-run it separately rather than inline.

# %%
import multisummary as M

if os.path.exists("cache/multi_session_summary.csv"):
    S, pooled, multi_stats = M.summarize()
    M.figure(S, pooled, multi_stats)
else:
    print("run 06_multi_session.py first to produce cache/multi_session_summary.csv")

# %% [markdown]
# ## 6. What the analysis shows
#
# The decoder recovers the animal's position during running to within a few centimetres, so the
# place-field templates are sound. During POST-sleep sharp-wave ripples the same decoder returns
# posteriors that sweep smoothly and monotonically across the track within 100–200 ms, in both
# the forward and the reverse direction, at roughly ten times the animal's running speed. These
# events are the classic signature of hippocampal replay: a compressed re-expression of a spatial
# trajectory in the absence of movement.
#
# The prevalence numbers are what make the claim testable rather than anecdotal. Shuffling the
# assignment of place fields to cells, which leaves every spike time and every event boundary
# untouched, sets the empirical false-positive rate of the procedure, and POST sleep sits at
# roughly twice that rate. PRE sleep, before the animal had ever run the track, is statistically
# indistinguishable from that control on both prevalence and sequence score, which is the
# comparison that ties the POST-sleep sequences to the experience rather than to any standing
# structure in the ensemble.
#
# Two caveats worth stating. The conjunction of two shuffle tests at p < 0.025 is not exactly a
# 5% test, which is why the cell-identity control rather than the nominal level is used as the
# chance line. And the direction template that maximises |r| is selected per event before
# testing, which is why the per-shuffle threshold carries a Bonferroni correction for the two
# templates.
