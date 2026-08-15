# Theta Phase Precession in Hippocampal Place Cells

**Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044), *"Diversity in
neural firing dynamics supports both rigid and learned hippocampal sequences"*
(Grosmark, Long & Buzsaki). This analysis uses session `sub-Buddy`, a bilateral CA1
silicon-probe recording made while a rat runs back and forth on a 1.6 m linear maze.
The NWB file provides linearized position (~39 Hz), 128-channel LFP at 1250 Hz, and
spike-sorted CA1 units. Data are streamed directly from the DANDI S3 store with
`remfile` disk caching; nothing is downloaded in full.

**What was analyzed.** Theta phase precession is the phenomenon, first reported by
O'Keefe & Recce (1993), in which a place cell fires at progressively earlier phases of
the 6-12 Hz hippocampal theta rhythm as the animal advances through the cell's firing
field. The pipeline (1) selects the LFP channel with the strongest theta power and
extracts instantaneous theta phase by Hilbert transform of the band-pass-filtered
signal; (2) keeps only samples where the rat is actually running (speed > 0.15 m/s) and
splits them into rightward and leftward traversals; (3) builds direction-specific 1-D
place fields for the 48 putative pyramidal cells; and (4) for each place field, assigns
every in-field spike a normalized within-field position and a theta phase (interpolated
on the unit circle to avoid wrap artifacts) and fits the circular-linear regression of
Kempter et al. (2012), yielding a slope, phase offset, and signed circular-linear
correlation.

**Key finding.** The classic precession signature is clearly reproduced. Of 19 place
cell x direction fits with enough in-field spikes, 13 show a significant circular-linear
correlation (p < 0.05) and 11 of those have the expected negative slope. Individual
example cells precess through roughly one full theta cycle across the field, with
circular-linear correlations as strong as rho = -0.64. The population median slope is
-0.50 theta cycles per field pass, the slope distribution is shifted well below zero,
and pooling the significant precessing cells (aligned by phase offset) yields a compact
band whose circular mean falls by about 150 degrees from field entry to exit. Together
these confirm that CA1 place cells in this freely-moving rat convert position within the
field into an advancing theta-phase code.

## Files

- `phase_precession.py` — final consolidated jupytext script (markdown + code cells), runs end-to-end.
- `phase_precession.ipynb` — executed Jupyter notebook version.
- `pp_lib.py` — shared loading/analysis helpers (streaming, theta phase, running epochs, circular-linear regression).
- `fig0_validation.png` — raw vs theta-filtered LFP, extracted phase, and back-and-forth position (data QC).
- `fig1_place_fields.png` — direction-specific place fields of the example cells (shaded = field used).
- `fig2_precession_examples.png` — spike theta-phase vs. within-field position for six example cells with fits.
- `fig3_population_summary.png` — slope and correlation distributions, and the pooled precessing band.

## Reproducing

```bash
python phase_precession.py            # runs the full pipeline, writes the four figures
jupytext --to notebook --execute phase_precession.py   # regenerate the notebook
```

Requires `pynapple`, `lindi`, `remfile`, `pynwb`, `h5py`, `scipy`, `matplotlib`, `tqdm`.
