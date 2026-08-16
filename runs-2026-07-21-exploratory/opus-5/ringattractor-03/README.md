# A continuous ring attractor in the head-direction system, maintained internally during sleep

## Dataset

[**DANDI:000056**](https://dandiarchive.org/dandiset/000056) — *Internally organized
mechanisms of the head direction sense* (Peyrache, Lacroix, Petersen & Buzsáki,
*Nature Neuroscience* 2015). Extracellular recordings from the anterior thalamus and
post-subiculum of freely moving mice. Each session contains an open-field foraging
epoch flanked by sleep, with awake / REM / non-REM state labels and two head-mounted
LEDs from which head direction is recovered. Files were streamed from the DANDI S3
bucket with `remfile` (nothing downloaded in full) and analysed with Pynapple.

4 sessions from 3 mice were analysed (76 head-direction cells in
total); `Mouse28-140312` (25 HD cells of 72
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

The ring is there in sleep, with the same geometry it has in waking. In `Mouse28-140312`, the
correlation between HD-cell pairs is a cosine function of their wake preferred-direction
difference in all three states (R² = 0.51 wake, 0.48 REM,
0.41 non-REM, against 0.004 for the time-shift
null and 0.01 for non-HD units). The Isomap embedding of the
population states is a hollow ring in all three states (hollowness
4.46 / 5.94 / 3.45 versus
2.15 for shuffled data and 2.65 for
non-HD units), and its angular coordinate, obtained with no behavioural or
tuning-curve information at all, pins down the decoded heading tightly
(residual resultant 0.94 in REM and 0.71 in non-REM
against 0.20 for shuffled labels). Two disjoint halves of the
ensemble decode the same angle during sleep (resultant length of the angular difference
0.67 in REM and 0.62 in non-REM, against
0.10 for the null), so the
population holds one coherent internal state rather than a set of independently
modulated cells. The decoder itself is validated on wake, where it recovers the
measured head direction with a median error of 12°.

That internal state moves, and it moves independently of the animal. Within sleep
episodes of at least 30 s the measured head direction is essentially fixed (median
resultant length 1.00 in REM and 1.00 in non-REM) while the
decoded internal heading is spread around the ring (0.35 and
0.37). Measured over 1 s the head turns at a median of
2°/s in REM and 3°/s in non-REM,
while the internal heading travels 23°/s and
68°/s respectively. It does so continuously rather than
by jumping, and about
3.0× faster in
non-REM than in REM, which is the time compression expected of replay-like non-REM
dynamics. All of this holds across all 4 sessions (cosine R² in non-REM
0.41-0.54, hollowness 3.09-3.45, split-ensemble agreement
0.32-0.62; see `fig08_multisession_summary.png` and
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
| `fig08_multisession_summary.png` | all metrics across 4 sessions |
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
