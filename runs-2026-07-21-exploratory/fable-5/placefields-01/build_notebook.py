"""Assemble the self-contained jupytext notebook from the tested modules.

Keeping one copy of the analysis code and generating the notebook from it means
the notebook can never drift from what was actually run.
"""

import ast
import subprocess

OUT_PY = "place_cell_analysis.py"
OUT_IPYNB = "place_cell_analysis.ipynb"


def top_level_sources(path):
    """Map each top-level name in a module to its source text."""
    src = open(path).read()
    lines = src.splitlines()
    out = {}
    for node in ast.parse(src).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            tgt = node.targets[0]
            name = tgt.id if isinstance(tgt, ast.Name) else None
        if name is None:
            continue
        start = min([d.lineno for d in getattr(node, "decorator_list", [])]
                    + [node.lineno]) - 1
        out[name] = "\n".join(lines[start:node.end_lineno])
    return out


LIB = top_level_sources("place_cell_lib.py")
FIG = top_level_sources("figures.py")


def code(*names, source=None):
    src = source or LIB
    body = "\n\n\n".join(src[n] for n in names)
    return body.replace("pcl.nap", "nap").replace("pcl.", "")


CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def py(text):
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Hippocampal place cells in rat CA1

**Data: [DANDI:000044](https://dandiarchive.org/dandiset/000044)** — Grosmark & Buzsáki
(2016), *Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences*, Science 351:1440. Eight bilateral silicon-probe sessions
from dorsal CA1 in four rats (Achilles, Buddy, Cicero, Gatsby), each consisting
of a pre-behaviour sleep period, a period of running on a maze for water reward,
and a post-behaviour sleep period.

A **place cell** is a hippocampal pyramidal neuron that fires when, and only
when, the animal occupies a particular part of its environment (O'Keefe &
Dostrovsky, 1971). This notebook demonstrates the phenomenon end to end on real
data:

1. Stream the NWB files from the DANDI Archive and reconstruct position, running
   laps and spike trains as pynapple objects.
2. Build occupancy-normalised firing rate maps for each unit and each running
   direction.
3. Test every unit against a null distribution built by circularly shifting its
   own spike train, and against the independent criterion of split-half
   stability across laps.
4. Characterise the resulting fields: peak rate, width, coverage of the track,
   reliability lap by lap, and the direction-specific remapping that is typical
   of linear tracks.
5. Decode the animal's position from held-out laps by Bayesian inference, which
   turns the single-cell result into a statement about the population code.
6. Fit Poisson GLMs with `nemos` to check that the fields reflect position
   rather than the running-speed profile along the track.
7. Repeat the whole analysis over all eight sessions.

Everything runs from streamed data with on-disk caching; nothing is downloaded
in full and no data are simulated.
""")

py("""
import re
import time

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import remfile
import requests
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

import pynapple as nap
import nemos as nmo

nap.nap_config.suppress_conversion_warnings = True
""")

md("""
## Analysis parameters

Rate maps use 4 cm bins smoothed with a 6 cm Gaussian. A lap counts only if the
animal covered at least 60% of the track in a consistent direction, and only the
parts of a lap where it was running faster than 5 cm/s enter the maps. A unit is
called a place cell if, in at least one running direction, its spatial
information exceeds the 99th percentile of its own circular-shift null and it
fires enough spikes to make the map meaningful.
""")

py(code("DANDISET_ID", "DANDI_API", "REMFILE_CACHE", "TRACK_LENGTH_CM",
        "BIN_WIDTH_CM", "N_POS_BINS", "SMOOTH_SIGMA_BINS", "MIN_LAP_DURATION_S",
        "MIN_LAP_COVERAGE", "MIN_LAP_MONOTONICITY", "MIN_SPEED_CM_S",
        "MIN_SPIKES_ON_TRACK", "MIN_PEAK_RATE_HZ", "N_SHUFFLES",
        "SHUFFLE_ALPHA", "DIRECTIONS", "DIR_SIGN"))

md(r"""
## Streaming the data from DANDI

The eight assets are 5–9 GB each, almost all of it raw electrophysiology, so
they are streamed with `remfile` and an on-disk chunk cache rather than
downloaded. Only the spike times, the behavioural position and short snippets of
LFP are ever pulled across the network.

Three properties of these particular NWB files shape the loader:

1. The behavioural `SpatialSeries` store the sampling **period** (0.0256 s) in
   the `rate` field rather than the rate, so timestamps have to be rebuilt as
   `starting_time + arange(n) * rate`. The resulting 39.06 Hz stream spans
   exactly the maze epoch.
2. The linearised position is `NaN` whenever the animal is not on the track, so
   the contiguous non-`NaN` segments bracket its time on the track. Those
   segments include brief shuffles at the reward ports as well as full runs, so
   laps are selected by requiring a segment to cross most of the track in one
   direction.
3. The track differs between sessions (1.6 m linear, 2 m linear, or a ~2.9 m
   circular maze) and is named in the `Position` container, so bin edges are
   derived per session rather than hard-coded.
""")

py(code("list_session_urls", "open_nwb"))
py(code("load_session", "_contiguous_epochs", "_select_traversals", "_speed_cm_s"))
py(code("lap_epochs", "apply_speed_mask", "direction_epochs"))

md("""
Load the session used throughout the first half of the notebook: rat Achilles on
a 1.6 m linear track.
""")

py("""
PRIMARY = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
urls = list_session_urls()
print(f"{len(urls)} NWB assets in DANDI:{DANDISET_ID}")

S = load_session(urls[PRIMARY])
print(S["nwbfile"])
""")

py("""
print(f"session      : {S['session_id']}  (subject {S['subject_id']})")
print(f"maze         : {S['maze_name']}, {S['track_cm']:.0f} cm {S['maze_type']}, "
      f"{S['n_bins']} position bins")
print(f"epochs       :\\n{S['epochs']}")
print(f"units        : {len(S['units'])} "
      f"({(S['units'].metadata['cell_type'] == 'excitatory').sum()} excitatory, "
      f"{(S['units'].metadata['cell_type'] == 'inhibitory').sum()} inhibitory)")
print(f"position     : {len(S['position'])} samples at "
      f"{1 / S['dt']:.2f} Hz, {S['position'].values.min():.0f}-"
      f"{S['position'].values.max():.0f} cm")
print(f"laps         : {(S['lap_dir'] == 1).sum()} rightward, "
      f"{(S['lap_dir'] == -1).sum()} leftward, "
      f"{(S['run_ep'].end - S['run_ep'].start).sum():.0f} s of running")
print(f"running speed: median {np.median(S['speed'].values):.0f} cm/s")
""")

md("""
### Checking the behavioural stream before analysing it

The panels below confirm that the reconstructed timestamps, the lap segmentation
and the direction assignment are all correct: the animal shuttles end to end,
each detected lap is a clean monotonic traversal, and the two directions
alternate.
""")

py('''
fig, ax = plt.subplots(3, 1, figsize=(13, 8))
fig.subplots_adjust(hspace=0.55)

pos = S["position"]
ax[0].plot(pos.t, pos.values, ".", color="0.75", ms=1)
for name in DIRECTIONS:
    laps = lap_epochs(S["run_ep"], S["lap_dir"], name)
    for i in range(len(laps)):
        seg = pos.restrict(laps[i])
        ax[0].plot(seg.t, seg.values, lw=1,
                   color="#1b6ca8" if name == "rightward" else "#d1495b")
ax[0].set_xlim(float(S["maze_ep"].start[0]), float(S["maze_ep"].end[0]))
ax[0].set_ylabel("position (cm)")
ax[0].set_xlabel("time (s)")
ax[0].set_title("Linearised position over the maze epoch "
                "(grey = on the track, coloured = accepted laps)")

t0 = float(S["run_ep"].start[3]) - 5
zoom = nap.IntervalSet(start=t0, end=t0 + 90)
seg = pos.restrict(zoom)
ax[1].plot(seg.t - t0, seg.values, "k.-", ms=2, lw=0.5)
for i in range(len(S["run_ep"])):
    a, b = float(S["run_ep"].start[i]), float(S["run_ep"].end[i])
    if b > t0 and a < t0 + 90:
        ax[1].axvspan(a - t0, b - t0, color="gold", alpha=0.35)
ax[1].set_xlim(0, 90)
ax[1].set_ylabel("position (cm)")
ax[1].set_xlabel("time from start of window (s)")
ax[1].set_title("90 s zoom: shaded bands are the detected laps")

ax[2].hist(S["speed"].values, bins=60, color="0.4")
ax[2].axvline(MIN_SPEED_CM_S, color="crimson", lw=2)
ax[2].set_xlabel("running speed (cm/s)")
ax[2].set_ylabel("samples")
ax[2].set_title(f"Speed during laps; only samples above the red line "
                f"({MIN_SPEED_CM_S:.0f} cm/s) enter the rate maps")
fig.savefig("fig00_behaviour_check.png", dpi=150, bbox_inches="tight")
''')

md(r"""
## Rate maps and place-field statistics

A rate map is the number of spikes fired in each position bin divided by the
time spent there. The implementation below is byte-for-byte equivalent to
pynapple's `compute_1d_tuning_curves` (verified further down); it is written out
explicitly because the same binning has to be reused thousands of times inside
the shuffle control, where going through pynapple would be far too slow.

Spatial information follows Skaggs et al. (1993),

$$\mathrm{SI} = \sum_i p_i \frac{\lambda_i}{\bar\lambda}
   \log_2 \frac{\lambda_i}{\bar\lambda} \quad \text{bits/spike},$$

where $p_i$ is the fraction of time spent in bin $i$ and $\lambda_i$ the firing
rate there.
""")

py(code("_bin_edges", "rate_maps", "_spike_position_bins", "_spike_sample_index"))
py(code("spatial_information", "sparsity", "_field_extent", "field_width_cm",
        "field_contrast", "lap_participation"))

md("""
Check the rate maps against pynapple's own routine before going further.
""")

py("""
_laps = lap_epochs(S["run_ep"], S["lap_dir"], "rightward")
_ep = apply_speed_mask(_laps, S["speed"])
_mine, _, _, _ = rate_maps(S["units"], S["position"], _ep, S["dt"],
                          n_bins=S["n_bins"], track_cm=S["track_cm"], sigma=0)
_theirs = nap.compute_1d_tuning_curves(S["units"], S["position"],
                                       nb_bins=S["n_bins"],
                                       minmax=(0, S["track_cm"]), ep=_ep)
print("largest discrepancy with nap.compute_1d_tuning_curves: "
      f"{np.nanmax(np.abs(_theirs.values.T - _mine)):.2e} Hz")
""")

md(r"""
## The shuffle control

The central worry with any rate map is that a peak could arise by chance,
because the animal samples the track unevenly and neurons fire in bursts. The
control used here circularly shifts each unit's spike train by a random lag
*within the concatenated running epochs*. This preserves the exact number of
spikes, the inter-spike-interval structure including bursting, and the occupancy
map, while destroying the alignment between spiking and position. Repeating it a
thousand times gives every cell its own null distribution of spatial
information.

This matters more than it might seem. Raw bits/spike is strongly inflated by a
low spike count: a cell that fires twenty spikes on the track will have a spiky
rate map and a high spatial information even with no spatial tuning at all.
Comparing each cell only against its own null is what makes the test fair.
""")

py(code("shuffle_spatial_information"))
py(code("split_half_stability", "classify_place_cells", "field_table",
        "best_direction"))

py("""
stats, per_dir = classify_place_cells(
    S["units"], S["position"], S["run_ep"], S["lap_dir"], S["speed"], S["dt"],
    n_bins=S["n_bins"], track_cm=S["track_cm"],
    circular=S["maze_type"] == "circular", n_shuffles=N_SHUFFLES, progress=tqdm)
fields = field_table(S["units"], stats, per_dir)

n_exc = int((stats["cell_type"] == "excitatory").sum())
print(f"{stats['is_place_cell'].sum()} place cells out of {n_exc} excitatory "
      f"units ({stats['is_place_cell'].sum() / n_exc:.0%}), "
      f"{len(fields)} fields in total")
print(fields[["si", "peak_rate", "width", "participation", "stability"]]
      .describe().loc[["50%", "mean"]].round(2))
""")

md("""
## Bayesian decoding and the LFP

Decoding turns the single-cell result into a claim about the population: if
these cells really carry position, an observer with access only to their spikes
should be able to recover where the animal is. Tuning curves are fitted to the
odd laps and used to decode the even laps, and vice versa, so no lap ever
contributes to both the encoding model and the test set.
""")

py(code("_tc_xarray", "crossvalidated_decoding", "decode_lap_block"))
py(code("lfp_series", "pick_lfp_channel", "_welch", "lfp_snippet"))

md("""
## Figures
""")

py(code("DIR_COLOR", "_peak_order", "_norm_map", "MAZE_COLOR", source=FIG)
   + "\n\n\n" + """plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})""")

md("""
### Figure 1 — the raw data

The recording, the behaviour and the spikes, with nothing computed except the
ordering of the cells. Panel (d) is the phenomenon in its rawest form: with the
place cells sorted by the position of their fields, a single traversal of the
track drives a diagonal sweep of activity across the ensemble.
""")
py(code("fig_session_overview", source=FIG))
py("""
lfp_channel, theta_ratio = pick_lfp_channel(S["nwbfile"],
                                            float(S["run_ep"].start[0]))
print(f"LFP channel {lfp_channel}: {theta_ratio:.0%} of 1-100 Hz power in the "
      f"6-12 Hz theta band")
fig_session_overview(S, stats, per_dir, lfp_channel, "fig01_session_overview.png")
""")

md("""
### Figure 2 — individual place cells

Each pair of panels shows one cell: above, the position of every spike it fired,
lap by lap; below, the rate map for each direction. The spikes cluster into a
horizontal band, present on nearly every lap, that occupies a small part of the
track. Examples were chosen for compactness, stability and firing rate rather
than for spatial information, which as noted above is not a fair ranking across
cells with different spike counts.
""")
py(code("select_examples", "fig_example_cells", source=FIG))
py('fig_example_cells(S, stats, per_dir, fields, "fig02_example_place_cells.png")')

md("""
### Figure 3 — the population

Stacking the normalised rate maps and sorting by field position shows that the
fields tile the track continuously. Sorting the leftward maps by the *rightward*
peak scrambles the diagonal, which is the directional remapping characteristic
of one-dimensional tracks: the same cell generally has unrelated fields in the
two directions of travel.
""")
py(code("fig_population_maps", source=FIG))
py('fig_population_maps(S, stats, per_dir, "fig03_population_maps.png")')

md("""
### Figure 4 — is the tuning real?
""")
py(code("fig_spatial_information", source=FIG))
py('fig_spatial_information(S, stats, per_dir, "fig04_spatial_information.png")')

md("""
### Figure 5 — what the fields look like

Panel (c) is the one that actually separates place cells from cells whose rate
map merely happens to be peaked: a real field produces at least one spike on the
large majority of the passes through it, whereas an incidental peak does not.
""")
py(code("fig_field_properties", source=FIG))
py('fig_field_properties(S, stats, per_dir, fields, "fig05_field_properties.png")')

md("""
### Figure 6 — decoding position from the population
""")
py(code("decoding_results", "fig_decoding", source=FIG))
py("""
dec = decoding_results(S, stats, per_dir)
print(f"median decoding error {np.median(dec['error']):.1f} cm "
      f"(chance {np.median(dec['chance']):.0f} cm) from "
      f"{dec['n_cells']} place cells in {int(dec['bin_size'] * 1000)} ms bins")
fig_decoding(S, stats, per_dir, dec, "fig06_decoding.png")
""")

md("""
### Figure 7 — position or running speed?

Running speed is not uniform along a linear track: the animal accelerates away
from one reward port and decelerates into the other. A purely speed-tuned cell
would therefore still produce a peaked rate map. Fitting Poisson GLMs with
`nemos` separates the two: a spline basis over position, a spline basis over
speed, and both together, each scored by held-out log-likelihood relative to a
constant-rate model and expressed in bits per spike, the same units as the
Skaggs measure.
""")
py(code("glm_position_vs_speed"))
py(code("fig_glm", source=FIG))
py("""
ref = max(per_dir, key=lambda d: len(per_dir[d]["laps"]))
glm = glm_position_vs_speed(S["units"], S["position"], S["speed"],
                            per_dir[ref]["laps"], S["dt"],
                            n_bins=S["n_bins"], track_cm=S["track_cm"])
pc_mask = stats["is_place_cell"].values
for name, g in glm["gains"].items():
    print(f"{name:>16s}: {np.nanmedian(g[pc_mask]):.3f} bits/spike "
          f"(median over place cells)")
fig_glm(S, stats, per_dir, fields, glm, ref, "fig07_glm_position_vs_speed.png")
""")

md("""
## All eight sessions

The whole pipeline is now run over every session in the dandiset, including the
three in which the animal ran on a circular maze. There the linearised
coordinate wraps, so rate maps are smoothed circularly and the session is
treated as having a single running direction.
""")

py(code("COMMON_BINS", "analyse", "decode_session", "resample_map",
        source=top_level_sources("run_all.py")))
py(code("fig_multisession", source=FIG))

py("""
rng = np.random.default_rng(0)
rows, all_fields, all_maps, glm_rows = [], [], [], []

for path, url in tqdm(list(urls.items()), desc="sessions"):
    Si = S if path == PRIMARY else load_session(url)
    st, pd_, fl = (stats, per_dir, fields) if path == PRIMARY else analyse(Si)
    med_err, chance_err = decode_session(Si, st, pd_, rng)

    refi = max(pd_, key=lambda d: len(pd_[d]["laps"]))
    gi = glm_position_vs_speed(Si["units"], Si["position"], Si["speed"],
                               pd_[refi]["laps"], Si["dt"],
                               n_bins=Si["n_bins"], track_cm=Si["track_cm"])
    m = st["is_place_cell"].values & np.isfinite(gi["gains"]["position"])
    glm_rows.append(pd.DataFrame({
        "session": Si["session_id"], "maze_type": Si["maze_type"],
        "position": gi["gains"]["position"][m],
        "speed": gi["gains"]["speed"][m],
        "position+speed": gi["gains"]["position+speed"][m]}))

    n_exc = int((st["cell_type"] == "excitatory").sum())
    rows.append(dict(
        session=Si["session_id"], subject=Si["subject_id"],
        maze_type=Si["maze_type"], track_cm=Si["track_cm"],
        n_units=len(Si["units"]), n_excitatory=n_exc,
        n_laps=len(Si["run_ep"]),
        run_time_s=float((Si["run_ep"].end - Si["run_ep"].start).sum()),
        n_place_cells=int(st["is_place_cell"].sum()),
        place_cell_fraction=float(st["is_place_cell"].sum()) / n_exc,
        n_fields=len(fl),
        median_si=float(fl["si"].median()),
        median_width_cm=float(fl["width"].median()),
        median_peak_rate=float(fl["peak_rate"].median()),
        median_participation=float(fl["participation"].median()),
        decode_error_cm=med_err, chance_error_cm=chance_err))

    fl = fl.assign(session=Si["session_id"], track_cm=Si["track_cm"],
                   maze_type=Si["maze_type"])
    all_fields.append(fl)
    for name, d in pd_.items():
        sel = fl.loc[fl["direction"] == name, "k"].astype(int).values
        if sel.size:
            all_maps.append(resample_map(d["rates"][sel]))
    if path != PRIMARY:
        Si["io"].close()

summary = pd.DataFrame(rows)
pooled_fields = pd.concat(all_fields, ignore_index=True)
pooled_maps = np.concatenate(all_maps)
pooled_glm = pd.concat(glm_rows, ignore_index=True)

summary.to_csv("session_summary.csv", index=False)
pooled_fields.to_csv("place_fields_all_sessions.csv", index=False)
stats.to_csv("place_cell_stats_primary.csv")
fields.to_csv("place_fields_primary.csv", index=False)

summary[["session", "subject", "maze_type", "track_cm", "n_excitatory",
         "n_place_cells", "place_cell_fraction", "n_fields", "median_si",
         "median_width_cm", "decode_error_cm", "chance_error_cm"]].round(3)
""")

py("""
fig_multisession(summary, pooled_fields, pooled_maps, pooled_glm,
                 "fig08_across_sessions.png")
""")

md("""
## Summary of what the data show

Across all eight sessions of DANDI:000044 — four rats, three maze geometries —
the analysis recovers the defining properties of hippocampal place cells:

- **Spatial firing.** Between a third and nine tenths of the CA1 excitatory
  units in each session have a firing field that survives a per-cell
  circular-shift test at p < 0.01. Pooled over sessions this is 322 place cells
  carrying 412 fields.
- **Compact and reliable fields.** The median field is about 32 cm wide, roughly
  a fifth of the track, with a median in-field peak of a few to a few tens of Hz.
  A field produces at least one spike on about 85% of the passes through it,
  against roughly 20% for excitatory cells whose rate maps merely happen to be
  peaked, and the maps built from odd and even laps correlate at about 0.95.
- **Complete coverage and directional remapping.** Field peaks tile the entire
  track, with the usual over-representation of the reward ends. On the linear
  tracks the two directions of travel carry essentially independent maps: peak
  positions in the two directions correlate at only about 0.2.
- **A read-out-able population code.** Bayesian decoding from held-out laps
  recovers position to a median error of 4–13 cm depending on the session,
  against a chance level of 47–84 cm, and the error falls steadily as more cells
  are added to the ensemble.
- **Position, not speed.** Poisson GLMs give position roughly an order of
  magnitude more held-out likelihood than running speed, position wins for about
  90% of place cells, and adding speed to a position model buys essentially
  nothing.

Two methodological points are worth carrying away. First, raw spatial
information in bits per spike is not comparable across cells with different
spike counts: in this data set the excitatory units that *fail* the place-cell
test have a higher median raw spatial information than the ones that pass,
because sparse firing produces noisy, spiky rate maps. Only the comparison
against each cell's own null, or a reliability measure such as lap-by-lap
participation, separates them. Second, the fast-spiking interneurons in this
data set are weakly but detectably spatially modulated; excluding them by cell
type, rather than by any map statistic, is what keeps the place-cell population
clean.
""")

# ---------------------------------------------------------------------------
lines = ["""# ---
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
"""]
for kind, text in CELLS:
    if kind == "markdown":
        lines.append("# %% [markdown]\n"
                     + "\n".join("# " + ln if ln else "#"
                                 for ln in text.splitlines()))
    else:
        lines.append("# %%\n" + text)

with open(OUT_PY, "w") as f:
    f.write("\n\n".join(lines) + "\n")
print(f"wrote {OUT_PY} ({sum(1 for _ in open(OUT_PY))} lines, {len(CELLS)} cells)")

subprocess.run(["jupytext", "--to", "notebook", "--output", OUT_IPYNB, OUT_PY],
               check=True)
print(f"wrote {OUT_IPYNB}")
