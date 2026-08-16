# Theta Phase Precession in Hippocampal CA1 Place Cells

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark, Diba & Buzsaki,
*"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences."*
Bilateral silicon-probe recordings from dorsal hippocampus (CA1) while rats ran back and
forth on a 1.6 m linear track for water reward. Each NWB file contains spike times with
cell-type and region labels, an LFP electrical series (128 channels at 1250 Hz), a
linearized position track, and epoch intervals (PRE / Maze / POST). Data were streamed
directly from S3 with `remfile` chunk caching; no file was downloaded in full.

Three sessions from two animals were analyzed: `sub-Achilles/ses-10252013`,
`sub-Achilles/ses-11012013`, and `sub-Cicero/ses-09172014`.

## What was analyzed

Theta phase precession is the phenomenon whereby a hippocampal place cell fires at
progressively earlier phases of the ~8 Hz theta rhythm as the animal moves through the
cell's firing field. The pipeline:

1. Restricts to the running (Maze) epoch and keeps excitatory CA1 units.
2. Extracts theta phase from the CA1 LFP channel with the strongest theta/delta power ratio
   (6–12 Hz band-pass + Hilbert transform).
3. Builds direction-specific 1D place fields (rightward vs. leftward laps) with Pynapple and
   selects place cells (peak rate > 5 Hz, spatial information > 0.5 bits/spike).
4. For each place cell, takes in-field spikes during running in the preferred direction and
   reads the animal's normalized position-in-field and the theta phase at every spike.
5. Fits a circular-linear regression (Kempter et al., 2012) per cell; a **negative slope**
   (phase advancing as position increases) is the signature of precession.
6. Repeats across sessions/animals to test robustness.

## Key finding

CA1 place cells show robust theta phase precession. In the Achilles 10-25-2013 session,
about **92% of place cells had a negative phase-position slope** (median ≈ −0.5 theta cycles
across the field), and among cells with a statistically significant circular-linear
correlation (p < 0.05) roughly 98% were negative. Individual cells display the classic
diagonal band of spikes advancing from late theta phase at field entry to early phase at
field exit (`fig_03_example_cells.png`), and the pooled spike-density map reproduces the same
population signature (`fig_04_population.png`, panel c). The effect replicated across all three
sessions and both animals, with 78–95% of place cells precessing in every session, far above
the 50% chance level (`fig_05_multisession.png`). This confirms, on real DANDI Archive data,
that position within a place field is encoded in the timing of spikes relative to the theta
rhythm.

## Files

| File | Purpose |
|------|---------|
| `01_load_data.py` | Stream one session; cache spikes, position, best-theta LFP channel |
| `02_preprocess.py` | Speed/running epochs, lap direction, theta phase → `fig_01_raw_streams.png` |
| `03_place_fields.py` | Directional place fields and place-cell selection → `fig_02_place_fields.png` |
| `04_phase_precession.py` | Circular-linear regression, single session → `fig_03`, `fig_04` |
| `pipeline.py` | Reusable load + analysis functions |
| `05_multi_session.py` | Replication across 3 sessions → `fig_05_multisession.png`, `multisession_summary.csv` |
| `phase_precession_analysis.py` / `.ipynb` | Consolidated end-to-end narrative (jupytext) |

The consolidated notebook additionally writes `fig_06`–`fig_10_*_notebook.png`.

### Reproducing

```bash
python 01_load_data.py
python 02_preprocess.py
python 03_place_fields.py
python 04_phase_precession.py
python 05_multi_session.py
# or run the consolidated notebook:
jupytext --to notebook phase_precession_analysis.py
jupyter nbconvert --to notebook --execute --inplace phase_precession_analysis.ipynb
```

Requires `pynapple`, `pynwb`, `lindi`, `remfile`, `h5py`, `scipy`, `matplotlib`, `pandas`,
`tqdm`, `jupytext`.

## Methods notes

- **Theta phase** is computed by band-passing the LFP at 6–12 Hz and taking the angle of the
  analytic (Hilbert) signal, so 0°/360° corresponds to the reference trough of the filtered
  theta wave.
- **Place fields** are the contiguous region around the tuning-curve peak where the rate
  exceeds 25% of the peak. Position within the field is normalized to [0, 1] and oriented
  along the direction of travel so that 0 = field entry.
- **Circular-linear regression** finds the slope maximizing the mean resultant length of
  `phase − 2π·slope·position`, and the signed circular-linear correlation ρ with an
  asymptotic p-value. Slopes are reported in theta cycles across the field.
