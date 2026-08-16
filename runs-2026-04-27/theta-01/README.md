# Theta phase entrainment & phase precession in CA1 place cells

## Dataset

**DANDI Archive Dandiset 000044** — *Diversity in neural firing dynamics supports both
rigid and learned hippocampal sequences* (Grosmark & Buzsáki, *Science* 2016).
Single session: `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` — Long-Evans
rat with bilateral silicon-probe recordings in dorsal CA1, running back and forth on a
1.6 m linear track.

The NWB file was streamed directly from S3 with `remfile`; only the maze epoch
(2067 s) and a single CA1 channel of LFP were ever transferred to disk.

## What was analyzed

1. The 6–12 Hz **theta** oscillation in the CA1 LFP (Hilbert-extracted instantaneous
   phase from the channel with the strongest theta power).
2. The position-tuned firing of CA1 pyramidal cells (linearized 1-D place fields,
   computed separately for rightward and leftward traversals on the 1.6 m track,
   restricted to track interior 0.10–1.50 m and run speed > 0.15 m/s).
3. **Theta phase entrainment** — circular distribution of theta phase at every
   pyramidal-cell spike during running.
4. **Theta phase precession** — circular–linear regression of theta phase against
   normalized position-within-field for each place cell, in its preferred direction.

## Key findings (single session, Achilles 10/25/2013)

| Quantity | Value |
| --- | --- |
| Pyramidal cells (excitatory, FR > 0.2 Hz on maze) | 111 |
| Place cells (peak rate > 2 Hz, Skaggs info > 0.7 bits/spike, interior peak) | 35 |
| Place cells used in precession analysis | 33 |
| Mean phase-locking strength `R` (population) | **0.19** |
| Population mean preferred phase | ≈ −2.07 rad (≈ trough of LFP) |
| Median phase-precession slope | **−2.64 rad / field width** (≈ −0.42 cycles/field) |
| Cells with significant negative slope (permutation p < 0.05) | **25 / 33 (76 %)** |
| Pooled (all cells) circular–linear ρ | 0.27 (p < 0.005) |

CA1 pyramidal cells fire at a preferred phase of the ongoing theta cycle (entrainment),
and within each cell's place field spikes occur at progressively earlier phases as the
animal advances through the field — the canonical signature of phase precession
(O'Keefe & Recce, 1993; Skaggs et al., 1996).

## Files

| File | Description |
| --- | --- |
| `theta_phase_precession_analysis.py` | jupytext-percent script, runs end-to-end |
| `theta_phase_precession_analysis.ipynb` | Same content as a Jupyter notebook |
| `fig01_lfp_and_theta.png` | Raw vs theta-filtered LFP & extracted phase (5 s snippet) |
| `fig02_place_fields.png` | Place-field heatmap, both run directions |
| `fig03_theta_entrainment.png` | Pooled phase histogram, preferred-phase rose plot, R distribution |
| `fig04_precession_examples.png` | Top 9 phase-precessing place cells |
| `fig05_precession_population.png` | Slope distribution, ρ distribution, pooled scatter |
| `phase_precession_per_cell.csv` | Per-cell precession statistics |

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy jupytext
python theta_phase_precession_analysis.py
```

The first run downloads ~30 MB of LFP/position bytes via `remfile` and caches them in
`/tmp/remfile_cache`; subsequent runs are essentially instant.
