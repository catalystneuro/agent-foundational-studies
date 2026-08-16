# Theta Phase Entrainment of Hippocampal CA1 Neurons

**Dataset**: DANDI dandiset [000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences" (Buzsaki lab). Session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: 128-channel LFP at 1250 Hz and 137 sorted CA1 units (120 excitatory, 17 inhibitory) recorded while a rat ran ~42 laps on a 1.6 m linear maze. The 8.7 GB file is streamed with remfile plus a disk cache; only the needed chunks are fetched.

**Analysis**: The LFP channel with the highest theta (5-11 Hz) to delta (1-4 Hz) power ratio during the maze epoch (channel 105, theta peak at 9.25 Hz) serves as the phase reference. The maze-epoch LFP is bandpass-filtered at 5-11 Hz with a zero-phase FIR filter, and instantaneous theta phase is taken from the Hilbert transform. Spikes are restricted to running bouts (smoothed speed > 10 cm/s; 163 bouts, 1044 s total). For each of the 136 units with at least 100 spikes while running, spike phases yield a mean resultant length (MRL), a preferred phase, and a Rayleigh p-value, plus a time-shift control that destroys spike-phase correspondence while preserving all marginal statistics.

**Key finding**: Theta phase entrainment is widespread and strong. 103/136 units (75.7%) are significantly phase-locked (Rayleigh p < 0.01): 87/119 excitatory pyramidal cells (median MRL 0.108) and 16/17 inhibitory interneurons (median MRL 0.249), with interneurons significantly more strongly locked (Mann-Whitney p = 1.5e-4). Observed locking far exceeds the time-shift control (median MRL 0.116 vs 0.025, p ~ 1e-33). Locked pyramidal cells prefer the descending phase approaching the trough of the reference theta (pooled preferred phase -142 deg, where 0 is the LFP peak), consistent with the classic description of CA1 entrainment during running (O'Keefe & Recce 1993).

## Files

- `theta_phase_entrainment.py` — final consolidated jupytext script, runs end-to-end
- `theta_phase_entrainment.ipynb` — executed notebook (converted with jupytext + nbconvert)
- `01_load_and_inspect.py`, `02_theta_phase.py`, `03_phase_locking.py` — modular development scripts
- `figures/fig1_psd_reference_channel.png` — LFP spectrum and reference-channel selection
- `figures/fig2_epoch_overview.png` — speed and theta amplitude across the maze epoch
- `figures/fig3_theta_phase_validation.png` — raw/filtered LFP, phase, and speed snippet
- `figures/fig4_example_phase_histograms.png` — polar spike-phase histograms, example units
- `figures/fig5_population_summary.png` — MRL distributions, locked fractions, preferred phases
- `figures/fig6_pooled_phase.png` — pooled phase histogram of locked pyramidal cells
- `figures/fig7_spikes_on_theta.png` — spikes of one unit riding the theta trace
- `results/phase_locking_per_unit.csv` — per-unit MRL, preferred phase, Rayleigh p, control

## Running

```
python theta_phase_entrainment.py
```

Requires pynapple, pynwb, remfile, h5py, scipy, pandas, matplotlib, tqdm, requests. Network access to the DANDI Archive is required; the S3 URL is resolved at runtime.
