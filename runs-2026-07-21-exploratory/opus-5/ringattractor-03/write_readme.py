"""Generate README.md from the session summary so the quoted numbers stay consistent."""
import numpy as np
import pandas as pd

PRIM = "Mouse28-140312"

df = pd.read_csv("session_summary.csv")
p = df.set_index("sid")


def f(col, sid=PRIM, nd=2):
    return f"{p.loc[sid, col]:.{nd}f}"


def rng(col, nd=2):
    v = df[col.format(st="")] if "{st}" not in col else None
    v = df[col]
    return f"{v.min():.{nd}f}-{v.max():.{nd}f}"


n_ses, n_mice = len(df), df.subject.nunique()
tot_hd = int(df.n_hd.sum())

text = f"""# A continuous ring attractor in the head-direction system, maintained internally during sleep

## Dataset

[**DANDI:000056**](https://dandiarchive.org/dandiset/000056) — *Internally organized
mechanisms of the head direction sense* (Peyrache, Lacroix, Petersen & Buzsáki,
*Nature Neuroscience* 2015). Extracellular recordings from the anterior thalamus and
post-subiculum of freely moving mice. Each session contains an open-field foraging
epoch flanked by sleep, with awake / REM / non-REM state labels and two head-mounted
LEDs from which head direction is recovered. Files were streamed from the DANDI S3
bucket with `remfile` (nothing downloaded in full) and analysed with Pynapple.

{n_ses} sessions from {n_mice} mice were analysed ({tot_hd} head-direction cells in
total); `{PRIM}` ({int(p.loc[PRIM, 'n_hd'])} HD cells of {int(p.loc[PRIM, 'n_units'])}
units) is shown in the per-session figures. "Wake" here means all awake periods with
continuous LED tracking, which includes both open-field foraging and quiet wakefulness
in the sleep box.

## What was analysed

The head-direction system is standardly described as a continuous ring attractor: the
joint activity of the population is confined to a one-dimensional closed curve, and the
position of a single activity bump on that curve is the animal's heading. The
interesting version of that claim is that the ring is a property of the *network*, not
of the sensory input, so it should survive sleep, when there is no informative
vestibular or visual heading signal. This analysis tests that, using spikes only.

HD cells were identified from wake tuning curves (mean vector length above both a
threshold and a circular-shift null, plus split-half tuning stability). The ensemble
was then binned at 100 ms in each of wake, REM and non-REM sleep, and four things were
measured: the pairwise correlation structure against the difference in preferred
direction; the shape of the Isomap embedding of the population states; how much of a
held-out half of the ensemble is explained by the heading decoded from the other half;
and whether two disjoint halves of the ensemble decode the same angle. Every sleep
measurement is computed without reference to behaviour. The null throughout is an
independent random circular time shift per neuron, which preserves all single-neuron
statistics and destroys only the coordination between neurons, and a second control
uses the simultaneously recorded non-HD units.

## Key finding

The ring is there in sleep, with the same geometry it has in waking. In `{PRIM}`, the
correlation between HD-cell pairs is a cosine function of their wake preferred-direction
difference in all three states (R² = {f('cos_r2_wake')} wake, {f('cos_r2_REM')} REM,
{f('cos_r2_nREM')} non-REM, against {f('null_cos_r2_nREM', nd=3)} for the time-shift
null and {f('ctrl_cos_r2_nREM')} for non-HD units). The Isomap embedding of the
population states is a hollow ring in all three states (hollowness
{f('hollowness_wake')} / {f('hollowness_REM')} / {f('hollowness_nREM')} versus
{f('null_hollowness_nREM')} for shuffled data and {f('ctrl_hollowness_nREM')} for
non-HD units), and its angular coordinate, obtained with no behavioural or
tuning-curve information at all, pins down the decoded heading tightly
(residual resultant {f('map_conc_REM')} in REM and {f('map_conc_nREM')} in non-REM
against {f('null_map_conc_nREM')} for shuffled labels). Two disjoint halves of the
ensemble decode the same angle during sleep (resultant length of the angular difference
{f('split_coh_REM')} in REM and {f('split_coh_nREM')} in non-REM, against
{f('null_split_coh_nREM', nd=2)} for the null), so the
population holds one coherent internal state rather than a set of independently
modulated cells. The decoder itself is validated on wake, where it recovers the
measured head direction with a median error of {f('decode_err_median_deg', nd=0)}°.

That internal state moves, and it moves independently of the animal. Within sleep
episodes of at least 30 s the measured head direction is essentially fixed (median
resultant length {f('R_head_REM')} in REM and {f('R_head_nREM')} in non-REM) while the
decoded internal heading is spread around the ring ({f('R_decoded_REM')} and
{f('R_decoded_nREM')}). Measured over 1 s the head turns at a median of
{f('head_speed_REM', nd=0)}°/s in REM and {f('head_speed_nREM', nd=0)}°/s in non-REM,
while the internal heading travels {f('internal_speed_REM', nd=0)}°/s and
{f('internal_speed_nREM', nd=0)}°/s respectively. It does so continuously rather than
by jumping, and about
{p.loc[PRIM, 'internal_speed_nREM'] / p.loc[PRIM, 'internal_speed_REM']:.1f}× faster in
non-REM than in REM, which is the time compression expected of replay-like non-REM
dynamics. All of this holds across all {n_ses} sessions (cosine R² in non-REM
{rng('cos_r2_nREM')}, hollowness {rng('hollowness_nREM')}, split-ensemble agreement
{rng('split_coh_nREM')}; see `fig08_multisession_summary.png` and
`session_summary.csv`).

## Files

| file | contents |
| --- | --- |
| `ring_attractor_hd_sleep.py` | consolidated jupytext script, runs end to end |
| `ring_attractor_hd_sleep.ipynb` | the same as a notebook |
| `fig01_data_overview.png` | state segmentation, wake and sleep rasters sorted by preferred direction |
| `fig02_hd_tuning_curves.png` | wake tuning curves and HD-cell selection |
| `fig03_pairwise_correlations.png` | correlation matrices and cosine fits per state |
| `fig04_ring_manifold.png` | Isomap embeddings, shuffle and non-HD controls |
| `fig05_dimensionality.png` | PCA spectra, PC2/PC1 ratio, hollowness, held-out R² |
| `fig06_internal_coherence.png` | split-ensemble decoder agreement and bump drift speed |
| `fig07_decoding_validation.png` | decoder validation on wake |
| `fig08_multisession_summary.png` | all metrics across {n_ses} sessions |
| `fig09_internal_vs_measured.png` | internal heading versus the animal's actual head |
| `session_summary.csv` | every metric, per session |
| `loader.py`, `hdcells.py`, `ringanalysis.py`, `run_session.py`, `plots.py`, `make_figures.py` | the modules the notebook is assembled from |

## Caveats

The units table in this NWB conversion carries only spike times, with no anatomical
label, so anterior thalamic and post-subicular cells are pooled and the analysis cannot
separate them. Isomap hollowness is a summary statistic rather than a topological
proof; persistent homology would be the stronger version of the same claim. Sleep
states are taken from the archived scoring rather than re-derived from the LFP. The
non-HD control ensemble is drawn from the same recordings and firing-rate range but is
not matched cell by cell.
"""

open("README.md", "w").write(text)
print("wrote README.md")
