# Theta phase precession in hippocampal CA1 place cells

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*
(also distributed as `hc-11` on CRCNS). Four rats (Achilles, Cicero, Gatsby, Buddy) ran back
and forth on a linear track for water reward while 128-site silicon probes recorded from
dorsal CA1 of both hemispheres. Each NWB file holds spike-sorted units labelled putative
excitatory or inhibitory, a 1250 Hz LFP on all 128 channels, and video-tracked position
linearized onto the track. Files were read by streaming from the DANDI S3 bucket through
LINDI with a local cache; nothing was downloaded in full (the example session alone is
8.7 GB, of which the analysis touches a few tens of megabytes).

Five of the eight sessions used a linear track (four at 1.6 m, one at 2 m) and all five were
analysed. The three circular-maze sessions were left out: that linearization wraps around and
does not decompose into unidirectional traversals under the same rule, and adapting it was
not needed to make the point.

## What was analysed

Laps were reconstructed from the linearized position, which is NaN whenever the animal is not
traversing the track, so each contiguous non-NaN block is one traversal; blocks were kept if
they lasted at least 0.5 s, covered at least 60% of the track, and were monotonic. Theta
phase came from the LFP channel with the highest theta-to-delta power ratio during running,
band-passed at 6-12 Hz and passed through a Hilbert transform. Place fields were computed
separately for the two directions of travel with Pynapple's tuning-curve routines, and a
cell-direction pair was kept if the smoothed peak rate reached 1 Hz, Skaggs spatial
information reached 0.5 bits/spike, the field (contiguous region above 20% of the peak) was
10-100 cm wide and peaked away from the track ends, and at least 50 spikes fell inside it.

For every field, spike theta phase was regressed against normalised within-field position
(0 at field entry, 1 at exit, in the animal's direction of travel) using the circular-linear
methods of Kempter, Leibold, Buzsáki, Diba and Schmidt (2012): the slope maximises the mean
resultant length, and significance comes from 500 permutations that break the phase-position
pairing while leaving both marginal distributions intact. The slope search is symmetric about
zero, so a negative result is not built into the method.

Two properties of these particular NWB files needed handling and are worth flagging for
anyone reusing them. The position `SpatialSeries` stores the sampling *period* (0.0256 s) in
the attribute named `rate`, so naive timestamp reconstruction stretches the behavioural
timeline by a factor of about 1500 and destroys alignment with the spikes. And one unit in
`Buddy-06272013` has a single out-of-order spike time out of 70,899, which Pynapple sorts on
construction after warning.

## Key finding

Theta phase precession is clear and consistent in this dataset. Of the 96 place fields that
met criteria across five sessions and four rats, 85.4% had a negative phase-position slope
(sign test p = 3.6e-13), 72.9% reached significance against the within-field shuffle, and
65.6% were both significant and negative. The permutation test is two-sided in the magnitude
of the correlation, so significance alone does not imply a descending slope, which is why
both numbers are reported. The median slope was -225 degrees per field traversal
(IQR -262 to -141), close to the two-thirds of a theta cycle described in the original
reports, and the median circular-linear correlation was -0.275 (Wilcoxon p = 2.8e-12). Every
rat contributed a negative median correlation, and every session had a majority of fields
precessing.

The effect is visible at every level of aggregation. On a single pass through a field, spikes
appear at successively earlier points on each theta cycle (figure 4, left). Pooling all passes
through one field gives a descending band of several hundred degrees (figure 4, right). Pooling
25,000 spikes from all 70 significant fields across all sessions gives a single continuous
band whose circular mean phase falls from about 280 degrees at field entry to about 80 degrees
at exit (figure 7), and the observed distribution of correlations is plainly displaced from
the shuffle null.

Absolute phase offsets depend on where the reference channel sat relative to the pyramidal
layer and should not be compared across sessions; the slope, which is the quantity of interest
here, does not depend on that choice. Four of the 96 fits ran into the edge of the ±1.5 cycle
search range; two of those still had a clearly negative correlation and two did not, and they
were left in the tallies rather than removed by hand.

## Files

| file | contents |
| --- | --- |
| `theta_phase_precession.py` | consolidated jupytext script, runs end to end |
| `theta_phase_precession.ipynb` | the same, converted and executed with outputs |
| `fig01_behavior_and_raster.png` | position, velocity, detected laps and a spike raster |
| `fig02_lfp_theta_validation.png` | channel selection, power spectrum, raw vs filtered LFP and Hilbert phase |
| `fig03_place_fields.png` | direction-specific place field maps and example tuning curves |
| `fig04_precession_example.png` | precession in the raw signals: one pass, then all passes through one field |
| `fig05_precession_gallery.png` | eight individual fields with significant precession |
| `fig06_precession_population.png` | population statistics for the example session |
| `fig07_multisession_summary.png` | all five sessions pooled, with the shuffle null |
| `precession_all_sessions.csv` | per-field results for all sessions |
| `precession_single_session.csv` | per-field results for the example session |
| `place_fields_summary.csv` | place field properties for the example session |
| `precession_lib.py`, `01`-`05_*.py`, `precession_03.py` | the modular development scripts the notebook was built from |

## Running it

```
pip install pynapple lindi pynwb h5py numpy scipy pandas matplotlib tqdm jupytext
python theta_phase_precession.py          # writes all figures
jupytext --to notebook theta_phase_precession.py
```

About one minute with a warm LINDI cache, a few minutes cold.

## References

- O'Keefe J, Recce ML (1993). Phase relationship between hippocampal place units and the EEG
  theta rhythm. *Hippocampus* 3:317-330.
- Grosmark AD, Buzsáki G (2016). Diversity in neural firing dynamics supports both rigid and
  learned hippocampal sequences. *Science* 351:1440-1443.
- Kempter R, Leibold C, Buzsáki G, Diba K, Schmidt R (2012). Quantifying circular-linear
  associations: hippocampal phase precession. *J Neurosci Methods* 207:113-124.
