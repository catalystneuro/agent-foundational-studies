# Theta Phase Entrainment of Hippocampal CA1 Neurons

## Dataset

**DANDI 000044** — *Hippocampal spatial coding during navigation*
(Grosmark, Long & Buzsáki; the `hc-11` dataset). Session **`sub-Buddy`**
(`sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb`): a rat running back and
forth on a linear maze with dense silicon-probe recordings in dorsal CA1. The
NWB file provides wideband LFP (1250 Hz), spike-sorted single units with
cell-type labels (excitatory / inhibitory) and anatomical locations, and the
animal's linearized position. Data were streamed directly from the DANDI S3
store with `remfile` byte-range reads and a local disk cache; nothing was fully
downloaded.

## What was analyzed

Theta phase entrainment, the tendency of hippocampal neurons to fire at a
preferred phase of the 6-10 Hz theta rhythm. The pipeline:

1. Selected the CA1 channel with the strongest theta rhythm (highest
   theta / broadband power ratio over the maze epoch; channel 83).
2. Band-passed that LFP to 6-10 Hz (zero-phase Butterworth) and took the
   Hilbert transform to obtain instantaneous theta phase and amplitude.
3. Restricted the analysis to running epochs (speed > 5 cm/s, ≥ 0.5 s;
   2279 s of the 2328 s maze session), since theta is a movement-related
   rhythm.
4. For each CA1 unit, collected the theta phase at every spike during running
   and quantified phase locking with the mean resultant length (MRL) and the
   Rayleigh test for circular non-uniformity.

All time-series handling uses **pynapple**; figures are written to `figures/`.

## Key finding

Theta phase entrainment is strong and near-universal in this CA1 population:
**61 of 65 CA1 units (94%) are significantly phase locked to theta**
(Rayleigh p < 0.05) during running, pooling roughly 0.9 million spikes. Single
units show clear preferred phases (example MRLs of 0.3-0.47), and the pooled
spike-phase histogram is visibly modulated across the theta cycle. Preferred
phases cluster on one side of the cycle, with excitatory principal cells firing
near the descending flank / trough of the LFP theta oscillation. This
reproduces the classic result that hippocampal principal cells and interneurons
are entrained by the theta rhythm.

## Files

- `theta_phase_entrainment.py` — consolidated jupytext script (runs end to end;
  streams and caches the data on first run, then reuses the cache).
- `theta_phase_entrainment.ipynb` — executed notebook version.
- `prep_data.py`, `analysis.py`, `visualize.py` — the modular
  load / analyze / plot scripts used during development.
- `figures/`
  - `fig1_channel_selection.png` — theta / broadband power ratio across shanks.
  - `fig2_lfp_theta_raster.png` — raw LFP, theta band, envelope, and a spike
    raster of strongly locked units over 3 s of running.
  - `fig3_example_polar.png` — polar spike-phase histograms for the six most
    strongly locked units.
  - `fig4_population_summary.png` — MRL distribution (exc vs inh), Rayleigh
    significance, and preferred-phase distribution.
  - `fig5_pooled_spike_phase.png` — pooled spike-phase histogram (two cycles).
- `cache/` — streamed data slices and intermediate results
  (`phase_locking_results.csv` holds the per-unit statistics).

## Reproducing

```bash
pip install pynapple pynwb lindi remfile h5py dandi scipy matplotlib tqdm jupytext
python theta_phase_entrainment.py          # first run streams ~tens of MB from DANDI
```

The first run streams the required LFP and spike data and writes
`cache/buddy_prep.npz`; subsequent runs load from the cache and regenerate all
figures in seconds.
