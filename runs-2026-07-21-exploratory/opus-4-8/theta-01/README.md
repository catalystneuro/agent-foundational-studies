# Theta phase entrainment and phase precession in hippocampal CA1

This analysis demonstrates two linked properties of the hippocampal theta code, using
real extracellular recordings from the DANDI Archive: that CA1 spiking is entrained to
the local field potential theta rhythm during running, and that place cells advance their
firing phase systematically as the animal traverses a place field.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences* (the `hc-11` dataset). Bilateral silicon-probe recordings from dorsal CA1 of
freely moving Long-Evans rats, with 128-channel LFP at 1250 Hz, spike-sorted units
labelled excitatory or inhibitory, and tracked position. The analysis uses the four
sessions recorded on the 1.6 m linear track (subjects Achilles, Cicero, Gatsby, Buddy);
the other two assets in the dandiset use a circular maze whose linearization is not
directly comparable and are excluded. Files are streamed from S3 with LINDI and a local
cache, so nothing is downloaded in full. All time-series handling and tuning-curve
computation uses Pynapple.

Two quirks of this NWB conversion are handled explicitly in `theta_lib.get_position`.
The behaviour `SpatialSeries` stores the sampling *period* (0.0256 s, i.e. 39.06 Hz) in
its `rate` field rather than the rate; reading it as a period reproduces the `MazeEpoch`
duration exactly. And the linearized position is defined only while the animal is
actually crossing the track, so the contiguous non-NaN blocks are used directly as the
individual traversals.

## What was analyzed

For each session, the LFP channel with the highest theta (6-12 Hz) to delta (1-4 Hz)
power ratio during running is selected, bandpassed, and Hilbert-transformed to give a
continuous theta phase (0 degrees is the peak of the filtered signal). Directional place
fields are computed over 50 spatial bins per running direction, and place cells are
selected by Skaggs spatial information, peak rate, and spike count. Phase entrainment is
quantified per unit by the mean resultant length and a Rayleigh test on the theta phases
of its spikes during running, and independently by the theta modulation of its spike
autocorrelogram. Phase precession is quantified by the circular-linear regression of
Kempter et al. (2012) applied to spike phase against normalized within-field position
along the direction of travel, with significance assessed against 200 permutations that
shuffle phases against positions.

## Key findings

Across 4 sessions and 358 sorted units, 81% of pyramidal cells (n = 155) and 97% of
interneurons (n = 67) fire at a significantly non-uniform theta phase during running
(Rayleigh p < 0.05), with interneurons locked more tightly (median MRL 0.22 versus 0.15).
Entrainment is equally clear without any reference to the LFP: the mean spike
autocorrelogram of both cell classes carries pronounced side peaks at the theta period.
Those side peaks fall at slightly shorter lags than the LFP theta period, so place cells
oscillate at a median 9.55 Hz against LFP theta frequencies of 7.75 to 9.25 Hz. That
small positive frequency offset is precisely the dual-oscillator signature that generates
a steady phase advance.

Of the 98 place cells identified, 65 (66%) show a circular-linear relationship between
theta phase and within-field position that survives the shuffle test, and 62 of those 65
have a negative slope: phase advances as the animal moves through the field. The median
advance is 206 degrees per field traversal (median |rho| = 0.34), within the published
range, and it is consistent across all four sessions (64% to 69% of place cells per
session, median slopes of -160 to -221 degrees). Critically, the effect is visible pass by
pass rather than only in pooled data, so it is a within-traversal phenomenon rather than
slow drift across the session.

## Files

| file | contents |
| --- | --- |
| `theta_precession_hc11.py` | consolidated jupytext script, runs end to end |
| `theta_precession_hc11.ipynb` | the same as an executed notebook |
| `theta_lib.py` | streaming/loading helpers, filtering, circular statistics |
| `analysis.py` | per-session pipeline: place fields, phase locking, precession |
| `plots.py` | all figure generation |
| `unit_table_all_sessions.csv` | per-unit results for all 358 units |
| `results.pkl` | pickled results (table, per-cell precession data, rhythmicity) |
| `figures/*.png` | all figures |

## Figures

| figure | contents |
| --- | --- |
| `00_channel_selection.png` | theta/delta ratio and theta amplitude across 128 channels, LFP spectrum during running |
| `01_raw_streams.png` | raw and theta-filtered LFP, linearized position with detected traversals, 2 s zoom with Hilbert phase |
| `02_place_fields.png` | population place-field maps per running direction, example directional tuning curves |
| `03_theta_phase_locking.png` | spikes on the theta cycle, spike-phase histograms, preferred phase vs locking strength |
| `04_theta_rhythmicity_<session>.png` | mean spike autocorrelograms, theta modulation index, intrinsic vs LFP theta frequency |
| `05_precession_examples.png` | six example place cells: field plus phase against within-field position with the fitted regression |
| `06_precession_population.png` | pooled phase-position density, mean phase per position bin, slope distribution, shuffle comparison, per-session summary |
| `07_single_run_precession.png` | precession within individual traversals of one field |
| `08_entrainment_population.png` | pooled locking strength, fraction locked, preferred phase distribution, rhythmicity, intrinsic frequency |

## Reproducing

```
pip install pynapple pynwb lindi h5py tqdm matplotlib scipy pandas jupytext
python theta_precession_hc11.py          # or run the notebook
```

The full run takes roughly four minutes on a warm LINDI cache and streams a few hundred
megabytes on a cold one.
