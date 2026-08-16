"""Build the final jupytext script (ripples_replay_analysis.py) from the
verified dev pipeline (scratch/dev_pipeline.py).

Sections are delimited by banner lines in the dev pipeline; each section gets
a markdown cell (prose) followed by a code cell whose body is copied verbatim
from the pipeline that actually ran. This guarantees the shipped code is
byte-for-byte the code that was validated.
"""
import re

SRC = "scratch/dev_pipeline.py"
OUT = "ripples_replay_analysis.py"

lines = open(SRC).read().split("\n")

MD = {}

MD["header"] = """# Sharp-Wave Ripples and Spatial Replay in Hippocampal CA1

This notebook detects sharp-wave ripples (SWRs) in dorsal CA1 and tests for
spatial replay during those ripples using real extracellular recordings,
streamed from the DANDI Archive.

**Dataset:**
dandiset 000044, the UCLA Buzsáki "Hippocampal CA1 multi-day recordings"
collection (Grosmark and Buzsáki, 2016). Session `sub-Achilles`: a Long-Evans
rat running a 1.6 m linear-maze alternation task, flanked by PRE and POST
epochs of spontaneous sleep. The asset is
`c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d`, streamed from S3 through remfile with
a local disk cache, so no full download is needed.

The pipeline:

1. Load and inspect the NWB file.
2. Select the LFP channel with the strongest ripple-band envelope in a
   Non-REM block.
3. Detect sharp-wave ripples (100-250 Hz envelope thresholding).
4. Identify place cells from the linear-track runs (Skaggs spatial
   information against a circular random-shift null).
5. Decode population position in 20 ms bins during each ripple and score each
   event with a posterior-mass-weighted time-position correlation, tested
   against a cell-identity shuffle, to quantify forward/reverse replay.

All figures are saved to `figures/`. Run headlessly (`MPLBACKEND=Agg`)."""

MD["load"] = """## Setup and data loading

Import libraries, fix the random seed, configure plotting, resolve the DANDI
asset URL, and open the NWB file with Pynapple. Units, epochs, states,
position, and the LFP are exposed as Pynapple objects."""

MD["channel"] = """## LFP channel selection

Sharp-wave ripples are clearest on electrodes in the CA1 pyramidal layer, but
we do not assume a priori which tetrode electrode that is. Each channel is
scored on a 60 s Non-REM block of POST sleep by the product of the
99th-percentile-to-median envelope ratio and the ripple/delta power ratio; the
best channel is used for event detection."""

MD["ripple"] = """## Sharp-wave ripple detection

The selected LFP channel is band-pass filtered (100-250 Hz, 4th order
Butterworth SOS, zero-phase) and its analytic-signal envelope is smoothed.
Thresholds come from the Non-REM distribution: onset/offset at mean + 1 SD,
peak at mean + 4 SD. Detections shorter than 30 ms or longer than 500 ms are
removed, gaps up to 30 ms are merged, and events must be contained in a
Non-REM block. POST ripples are those whose center falls in the POST epoch."""

MD["fig1"] = """## Figure 1: detection on raw LFP

A 12 s window of Non-REM with the envelope, thresholds, and detected events
shaded."""

MD["fig2"] = """## Figure 2: ripple statistics

Counts, duration distribution, and ripple rate per Non-REM minute across the
session, split by epoch."""

MD["fig3"] = """## Figure 3: ripple-triggered average and spectrogram

Band-limited LFP and spectrogram averaged over POST ripple peaks."""

MD["placecells"] = """## Place cell identification

Running bouts are detected from the linearized position (speed > 0.10 m/s,
gaps <= 0.3 s merged). Spike times are mapped onto a concatenated run-time
axis (tau). Per-unit rate maps (50 bins along the 1.6 m track) and Skaggs
spatial information are computed; significance is tested against 500 circular
random time shifts. A unit is a place cell if its spatial information beats
95% of the null, peak run rate >= 1 Hz, and mean run rate >= 0.1 Hz."""

MD["fig4"] = """## Figure 4: place fields

Example fields, all accepted place-cell rate maps, and the spatial
information distribution vs. the null."""

MD["replay"] = """## Replay decoding

During every detected ripple the posterior over position is decoded in 20 ms
bins from the place-cell ensemble (normalized template Bayes, occupancy
prior). Each event is scored by the posterior-mass-weighted Pearson
correlation between time-bin index and decoded position, giving rho. A null
distribution for each event is built from 500 cell-identity shuffles (spike
trains kept, rate maps permuted). Events with fewer than 5 active cells or 5
non-empty bins are not analyzed."""

MD["fig5"] = """## Figure 5: example replay trajectories

The most significant forward and reverse POST events, drawn as decoded
posterior heatmaps over time with the decoded position overlaid."""

MD["fig6"] = """## Figure 6: replay summary

Fraction of significantly replaying events PRE vs POST (with Fisher test),
distribution of |rho| (Mann-Whitney), forward/reverse counts, and rho vs time
during POST."""


def find_banner(pat):
    for i, ln in enumerate(lines):
        if pat in ln:
            return i
    raise RuntimeError("banner not found: " + pat)


# ordered sections: (md_key, code_start_line)
order = [
    ("header", 0),
    ("load", find_banner("# ---------- load ----------")),
    ("channel", find_banner("# ---------- channel selection ----------")),
    ("ripple", find_banner("# ---------- ripple detection ----------")),
    ("fig1", find_banner("FIGURE 1")),
    ("fig2", find_banner("FIGURE 2")),
    ("fig3", find_banner("FIGURE 3")),
    ("placecells", find_banner("PLACE CELLS")),
    ("fig4", find_banner("FIGURE 4")),
    ("replay", find_banner("REPLAY DECODING")),
    ("fig5", find_banner("FIGURE 4b/5")),
    ("fig6", find_banner("FIGURE 6")),
]

divider = re.compile(r"^# [-=]{3,}")


def clean(seg):
    out = []
    for ln in seg:
        if divider.match(ln):
            continue
        out.append(ln)
    while out and out[-1].strip() == "":
        out.pop()
    while out and out[0].strip() == "":
        out.pop(0)
    return out


out = []
for i, (key, start) in enumerate(order):
    end = order[i + 1][1] if i + 1 < len(order) else len(lines)
    out.append("# %% [markdown]")
    for ml in MD[key].split("\n"):
        out.append("# " + ml)
    out.append("")
    out.append("# %%")
    out.extend(clean(lines[start:end]))
    out.append("")

text = "\n".join(out)
# keep the results file in the working directory of the notebook
text = text.replace('np.savez("scratch/replay_results.npz"',
                    'np.savez("replay_results.npz"')
# drop the dev-pipeline module docstring and the now-unused raw spike arrays
# (spike times are read from the pynapple TsGroup instead)
text = text.replace('"""Development version of the ripple/replay pipeline (debugging, not final)."""\n', "")
for dead in [
    'spike_times = np.asarray(h5py_file["units/spike_times"], dtype=np.float64)\n',
    'spike_index = np.asarray(h5py_file["units/spike_times_index"], dtype=np.int64)\n',
    'spike_cum = np.concatenate([[0], np.cumsum(spike_index)])\n',
]:
    text = text.replace(dead, "")
open(OUT, "w").write(text)
print("Wrote", OUT, "with", len(order), "sections,", text.count("# %%"), "cells")