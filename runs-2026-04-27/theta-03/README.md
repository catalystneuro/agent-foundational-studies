# Theta Phase Entrainment and Phase Precession in Hippocampal Place Cells

This analysis demonstrates two canonical features of CA1 pyramidal cells in the
freely-behaving rat:

1. **Theta-rhythm entrainment** — spikes are not uniformly distributed across
   the theta cycle but cluster around a preferred phase.
2. **Theta phase precession** — within a place field, spikes occur at
   progressively earlier phases of theta as the animal traverses the field.

## Dataset

- **DANDI ID:** [DANDI:000044](https://dandiarchive.org/dandiset/000044)
- **Reference:** Grosmark, A.D., & Buzsáki, G. (2016). *Diversity in neural
  firing dynamics supports both rigid and learned hippocampal sequences.*
  Science 351, 1440–1443.
- **Session used:** `sub-Achilles_ses-Achilles-10252013` (rat *Achilles*,
  bilateral CA1 silicon-probe recording during running on a 1.6-m linear
  track).
- The NWB file is streamed directly from S3 with `remfile` + a local disk
  cache; no full download is required.

The session contains a `MazeEpoch` (≈2068 s) sandwiched between pre- and
post-task sleep epochs. We restrict all analyses to the `MazeEpoch`.

## Pipeline

The full pipeline lives in **`theta_analysis.py`** (jupytext-formatted
markdown + code; also exported as `theta_analysis.ipynb`). Key steps:

1. **Load NWB file from DANDI via remfile streaming.** Extract spikes (137
   sorted units; we keep the 120 putative pyramidal cells), 1-D linearized
   position (~39 Hz), and the 128-channel CA1 LFP (1250 Hz).
2. **Behavior:** smooth position with a 250 ms Gaussian, derive running speed,
   and build IntervalSets of *running* (≥5 cm/s, ≥0.5 s) for each direction.
3. **LFP channel selection:** probe one channel per shank, compute Welch PSDs,
   and select the channel with the highest 6–10 Hz/(2–4 Hz + 12–20 Hz) ratio.
   In this session that is channel **69** (theta/flank ≈ 7.2).
4. **Theta phase:** band-pass filter the chosen channel to 6–10 Hz (Butterworth
   order 4, zero-phase) and apply the Hilbert transform to get the
   instantaneous phase and amplitude.
5. **Place fields:** smoothed (σ = 2 bins) firing-rate maps over a trimmed
   linearized track (0.10–1.50 m, 40 bins), separately for left- and
   right-bound runs; classify a cell as a **place cell** if peak rate ≥ 2 Hz,
   spatial information ≥ 0.7 bits/spike, and ≥ 3 contiguous bins above
   peak/3.
6. **Entrainment:** for each place cell, sample theta phase at every running
   spike and run a Rayleigh test for non-uniformity.
7. **Precession:** delineate each cell's field at peak/3, then fit a
   circular-linear regression (Kempter, Leibold, Buzsáki & Schmitzer-Torbert
   2012) of theta phase vs. normalized in-field position. Report slope (rad
   per field), circular-linear correlation ρ, and its asymptotic p-value.

## Key results (single session)

| metric | value |
|---|---|
| pyramidal cells | 120 |
| place cells (after criteria) | 34 (15 R-bound, 19 L-bound) |
| significantly theta-entrained (Rayleigh p<0.05) | 28 / 34 (82%) |
| population mean theta phase | ≈149° |
| precession analyzed | 33 cells |
| cells with negative phase slope | 26 / 33 (79%) |
| cells with significant negative slope | 19 / 33 (58%) |
| population-mean phase slope | **−1.48 rad / field** |
| population-mean ρ (phase, position) | **+0.26** |

Interpretation. Almost every place cell is locked to theta (Rayleigh p<0.05),
with a population-preferred phase near the descending portion of the theta
cycle. Plotted spike phase against normalized in-field position, the great
majority of fields show the textbook negative slope: a single field-traversal
takes the spike phase forward by roughly half a theta cycle — the hallmark of
phase precession. Both phenomena emerge directly from the raw NWB data with no
post-hoc tuning.

## Outputs

- `theta_analysis.py` — jupytext-formatted, runs end-to-end.
- `theta_analysis.ipynb` — same content as a Jupyter notebook.
- `figures/01_behavior_traces.png` — example position and running speed.
- `figures/02_lfp_theta.png` — raw LFP, 6–10 Hz band-pass, Hilbert phase.
- `figures/03_lfp_psd.png` — power spectrum of the selected channel.
- `figures/04_place_field_population.png` — population place-field heatmaps,
  sorted by peak position, separated by running direction.
- `figures/05_theta_entrainment.png` — pooled spike-phase histogram (left)
  and per-cell preferred phase × resultant length (right).
- `figures/06_phase_precession_examples.png` — six representative cells:
  theta phase vs. normalized in-field position with circular-linear fit.
- `figures/07_precession_population.png` — population distributions of
  precession slopes and circular-linear correlations.

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy dandi jupytext
python theta_analysis.py        # runs the full pipeline (~2 min after cache warm)
jupytext --to notebook theta_analysis.py
```
