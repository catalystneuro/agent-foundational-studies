# Orientation Selectivity in the Mouse Visual System

This analysis demonstrates orientation selectivity using Neuropixels recordings from
[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute *Visual
Coding, Neuropixels* dataset (Brain Observatory 1.1 stimulus set). Six sessions from six
mice were analysed, giving 1501 quality-passing units recorded simultaneously in the
dorsal lateral geniculate nucleus (LGd, n = 153) and in six visual cortical areas (VISp,
VISl, VISal, VISrl, VISpm, VISam; n = 1348 combined). The stimuli used are the drifting
gratings (8 drift directions at 45° steps × 5 temporal frequencies × 15 repeats, 2 s each,
with interleaved blank sweeps) and the static gratings (6 orientations × 5 spatial
frequencies × 4 phases, 0.25 s each). NWB files are 2–3 GB apiece and were never
downloaded: `remfile` with a local disk cache served only the byte ranges holding the
units table, the selected units' spike times, and the stimulus interval tables. All
analysis is done in `pynapple`, with the encoding models fit in `NeMoS`.

The result is a clear demonstration that visual cortex, but not its thalamic input, encodes
the *axis* of a grating rather than its direction of motion. Individual VISp units respond
strongly at two drift directions 180° apart and are near-silent at the orthogonal axis; a
two-lobed von Mises function accounts for essentially all of the variance across
directions in the best-tuned cells. Across the pooled population the median global OSI is
0.162 in cortex against 0.063 in LGd (Mann-Whitney p = 4.4 × 10⁻³¹), and 25–41% of
cortical units per area clear a joint criterion of a shuffle test (p < 0.05, with a null
built separately for every unit because the index is positively biased by noise) and an
effect size of gOSI ≥ 0.25, compared with 5% of LGd units. Sorting the whole population by
a preferred orientation estimated on one half of the trials and displaying the other half
produces the two parallel diagonal bands that orientation selectivity predicts in every
cortical area and in none of the thalamic data.

Three independent checks support the interpretation. Preferred orientation estimated from
two disjoint halves of the trials agrees with a circular correlation of 0.95 among
selective units, so the tuning is not noise. Preferred orientation measured with 2 s
drifting gratings agrees with the value measured using 0.25 s static gratings (142 units
tuned to both families, circular r = 0.86, median absolute offset 5.4°), so it is a
property of the receptive field rather than of one stimulus. A leave-one-trial-out
population decoder applied to VISp identifies which of the 8 drift directions was shown on
68.4% of trials (chance 12.5%), but its mistakes are not spread evenly: 24% of trials are
decoded as the direction exactly 180° away while each of the other six errors accounts for
under 2%, so the correct *axis* is recovered on 92.1% of trials against a chance level of
25%. Finally, a Poisson GLM whose predictor is a cyclic B-spline basis periodic over 180°,
and which therefore cannot represent any difference between a direction and its opposite,
predicts held-out cortical spike counts about as well as a 360°-periodic model with twice
the parameters (median cross-validated pseudo-R² ratio 0.99 in VISp), while in LGd the
same ratio falls to 0.85.

## Files

| File | Contents |
| --- | --- |
| `orientation_selectivity.py` | Consolidated jupytext (percent format) analysis script, runs end to end |
| `orientation_selectivity.ipynb` | The same notebook with all outputs executed |
| `dandi_io.py` | Streaming access to DANDI assets via `remfile` with a disk cache |
| `extract.py` | Pulls units, spike times and stimulus tables from a session; caches to `.npz` |
| `analysis.py` | Tuning curves, selectivity indices, permutation null, von Mises fits, decoder |
| `glm_tuning.py` | NeMoS Poisson GLMs comparing 360°- and 180°-periodic encoding models |
| `pipeline.py` | Per-session driver producing a tidy per-unit table |
| `figures.py` | All plotting |
| `orientation_selectivity_units.csv` | Per-unit results for all 1501 units |
| `fig01` to `fig10` `.png` | Figures, described below |

`session_cache/` holds the extracted per-session `.npz` files (roughly 1 GB total) and
`/tmp/remfile_cache_000021` holds the raw HTTP range cache. Both are regenerated
automatically if deleted; the first run takes about 90 s per session to stream, and
subsequent runs complete in about 70 s in total.

## Figures

1. **fig01_raw_activity**: every unit's spikes during the first 16 grating trials, sorted
   by depth and coloured by structure, with one VISp unit shown separately.
2. **fig02_direction_rasters**: the same unit's trials and PSTHs sorted by drift
   direction, showing two response lobes 180° apart.
3. **fig03_example_tuning**: Cartesian and polar tuning curves with double von Mises fits
   for the four most selective VISp units.
4. **fig04_population_tuning**: cross-validated population heatmaps per structure, and
   the peak-aligned average showing the second lobe at ±180°.
5. **fig05_selectivity_distributions**: gOSI distributions against the shuffled null, per
   structure, and the fraction of units meeting the selectivity criterion.
6. **fig06_reliability**: split-half reproducibility of preferred orientation and the
   distribution of preferred orientations across the population.
7. **fig07_static_vs_drifting**: agreement between the drifting- and static-grating
   measurements.
8. **fig08_decoding**: VISp population decoding of drift direction, the 180° structure of
   its errors, and the dependence on population size.
9. **fig09_multisession**: per-session medians, pooled cumulative distributions, and
   tuned fractions among visually responsive units.
10. **fig10_glm_encoding**: the two NeMoS encoding models on an example unit, their
    cross-validated pseudo-R² per unit, and the ratio by structure.

## Caveats

The permutation test detects any reproducible orientation bias, however small, so
significance alone is a poor definition of "orientation selective": 69% of pooled LGd
units clear p < 0.05 despite a median gOSI of 0.063. All headline fractions
therefore require both the shuffle test and gOSI ≥ 0.25. Drift directions are sampled only
every 45°, so the von Mises concentration parameter is bounded to keep the fitted lobe no
narrower than 22.5°; fitted widths at that bound should be read as at or below the
resolution of the stimulus set. Tuning curves pool over the five temporal frequencies
rather than being computed at each unit's preferred frequency, which avoids selecting the
condition with the same data used to measure the tuning at the cost of some dilution.
Recording structures differ between sessions because probe placement differs, so LGd
appears in only two of the six sessions and the thalamic comparison rests on 153 units
from two mice rather than six; the per-session panel in fig09 shows which structures each
session contributes. Higher visual areas are also sampled unevenly, which is why the
ordering among cortical areas in fig05 should not be read as a claim about an areal
hierarchy.

## Requirements

`pynapple`, `nemos`, `pynwb`, `h5py`, `remfile`, `numpy`, `scipy`, `pandas`,
`scikit-learn`, `matplotlib`, `tqdm`, and `jupytext` for the notebook conversion.
