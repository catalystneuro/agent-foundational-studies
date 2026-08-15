# Theta phase precession in hippocampal CA1 place cells

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), *Diversity in neural firing
dynamics supports both rigid and learned hippocampal sequences* (Grosmark & Buzsáki,
Science 2016), the dataset also distributed as `hc-11`. Four rats ran back and forth
on linear tracks for water reward while bilateral silicon probes recorded CA1 spikes
and 1250 Hz local field potentials. Each session is a PRE sleep epoch, a maze epoch,
and a POST sleep epoch; only the maze epoch is used here.

Five of the eight sessions use a bidirectional linear track (1.6 m or 2.0 m) and all
five are analysed: `Achilles-10252013`, `Buddy-06272013`, `Cicero-09012014`,
`Cicero-09172014`, `Gatsby-08022013`. The other three use a circular maze whose
linearized coordinate wraps around, which the run-direction segmentation used here
does not handle, so they are excluded. The 5–9 GB NWB files are read over HTTP with
`remfile` and a local disk cache; only the byte ranges actually touched (the spike
times, the position, and one LFP channel over the maze epoch) are transferred.

## What was analysed

Position was taken from the stored linearized track coordinate and differentiated to
give velocity, which was thresholded at 10 cm/s and split by sign into rightward and
leftward running epochs. A theta reference LFP channel was chosen per session as the
one with the largest 6–12 Hz power relative to the 2–5 and 13–25 Hz background, then
band-passed at 6–12 Hz; the Hilbert phase of that signal defines theta phase, with 0°
at the peak of the filtered LFP (verified empirically in the notebook). Direction-specific
firing rate maps were built in 2 cm bins for the excitatory units, and a place field
was taken as the contiguous region around the peak above 25 % of the peak rate, with a
peak above 1 Hz and a width between 12 and 80 cm.

For each (unit × running direction) field with at least 50 in-field spikes, the theta
phase of every in-field spike was regressed on its normalized within-field travel
distance using the circular-linear method of Kempter et al. (2012): the slope is the
value maximizing the resultant length of the phase residuals, searched over ±2 cycles
per traversal. Because in-field spikes from one theta cycle are not independent,
significance came from a permutation test rather than the asymptotic one: phases were
shuffled against positions 500 times, which preserves both the theta locking and the
in-field position distribution and destroys only their pairing. Within-field distance
is measured in the direction the animal is travelling, which matters: without that
sign convention the fitted slopes for leftward runs come out mirrored and the
population median lands near zero.

## Key finding

Phase precession is present and quantitatively as described in the literature. Across
the five sessions, 204 unit × direction place fields were analysed; 52 of them (25 %,
against a 5 % false-positive rate) show a phase-position relationship that survives the
permutation test, and the significant fields have a median slope of **−194° of theta
phase per field traversal**, with 75 % of slopes negative (binomial p = 4 × 10⁻⁴).
Taking every field regardless of individual significance, the circular-linear
correlation is shifted negative (median ρ = −0.031, Wilcoxon p = 1.3 × 10⁻⁴). Pooling
all 52,761 in-field spikes into the position-by-phase plane (figure 6, panels A–B)
produces the diagonal band that is the classic signature of the effect: spikes near
field entry occur late in the theta cycle and spikes near field exit occur roughly
half a cycle earlier. The effect is visible within single field traversals, not only
in the pooled cloud (figure 5).

How many fields reach individual significance is limited mostly by how many spikes
each field contributes: fields with more than 400 in-field spikes are significant
about 50 % of the time, whereas fields with 50–100 spikes are near the noise floor
(figure 6, panel E). Sessions differ for the same reason and because of recording
quality, with `Achilles-10252013` (the session most commonly used with this dataset,
and the one with the most running and the most excitatory units) giving 41 % and
`Gatsby-08022013` only 7 %. Note also that the mean phase advance measured
model-free (figure 6, panel D, about −60° from field entry to exit) is much shallower
than the fitted ridge slope of −194°, because a substantial fraction of in-field
spikes sit off the precession ridge and pull the circular mean toward the overall
preferred phase.

## Files

| file | contents |
| --- | --- |
| `theta_phase_precession.py` | consolidated jupytext (percent format) script, runs end to end |
| `theta_phase_precession.ipynb` | the same script executed as a notebook |
| `precession_results.pkl` | per-field fits for all five sessions |
| `figures/fig01_track_and_runs.png` | tracking, traversals, running speed, run-epoch detection |
| `figures/fig02_theta_lfp.png` | LFP spectra, reference-channel selection, phase convention, example trace |
| `figures/fig03_place_fields.png` | direction-specific rate maps for all units, example fields |
| `figures/fig04_example_precession.png` | six strongest precessing fields, phase vs position |
| `figures/fig05_single_cell_passes.png` | one field broken down by individual traversal |
| `figures/fig06_population.png` | pooled phase density, slope distribution, per-session summary |

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `scipy`, `matplotlib`, `tqdm`. Run
with `python theta_phase_precession.py`; roughly 4 minutes on a cold cache, 40 s once
the streamed byte ranges are cached under `/tmp/remfile_cache`.
