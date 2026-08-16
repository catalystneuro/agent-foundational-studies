# Theta phase entrainment of hippocampal CA1 neurons

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044) (Petersen &
Buzsáki), hippocampal CA1 recordings from freely moving rats. Each session
provides a 128-channel silicon-probe LFP (1250 Hz), spike-sorted single units
labeled as excitatory (pyramidal) or inhibitory (interneuron), and tracked
position on a linear maze. The analysis streams four sessions directly from
the DANDI S3 store with `remfile` disk caching, so no full downloads are
required. The four rats used are Buddy, Achilles, Cicero, and Gatsby.

## What was analyzed

The notebook demonstrates **theta phase entrainment**: the tendency of CA1
neurons to fire at a preferred phase of the 6-10 Hz theta rhythm during
locomotion. The pipeline (all in Pynapple) selects the LFP channel in the CA1
pyramidal layer with the strongest theta (theta/delta power ratio), band-pass
filters it to 6-10 Hz, and extracts the instantaneous theta phase with the
Hilbert transform. Running epochs are derived from position speed (theta is a
movement-related rhythm), and for every unit the theta phase at each spike
during running is collected. Phase locking is quantified by the **mean
resultant length (MRL)** and tested for significance with the **Rayleigh
test**. The single-session prototype is then rerun across all four rats and
the per-unit results are pooled.

## Key finding

Theta phase entrainment is pervasive and highly significant across the CA1
population. Pooling 345 units from four rats, **87% fire at a significantly
non-uniform theta phase** during running (Rayleigh p < 0.05), and in the best
session 61/64 units are significant. Inhibitory interneurons entrain more
strongly than excitatory pyramidal cells (pooled median MRL 0.207 vs 0.124;
Mann-Whitney p = 4.2e-12), and essentially all interneurons are significant in
every animal, consistent with the classic view that interneuron networks pace
the theta rhythm while pyramidal cells participate more sparsely. The
spike-triggered LFP average provides an independent, phase-free confirmation:
averaging the raw LFP around a strongly locked cell's spikes recovers a clean
~8 Hz oscillation centered on the spike.

## Files

- `theta_phase_entrainment.py` — consolidated jupytext script (runs end-to-end,
  ~30 s with a warm cache).
- `theta_phase_entrainment.ipynb` — the same notebook converted with jupytext.
- `phase_locking_results.csv` — per-unit results for the prototype session.
- `phase_locking_all_sessions.csv` — pooled per-unit results across four rats.
- Figures:
  - `fig_channel_selection.png` — theta prominence across channels; raw/theta/phase on the chosen channel.
  - `fig2_lfp_raster.png` — raw LFP, theta band, spike raster, and speed during a running bout.
  - `fig3_example_polar.png` — spike-theta-phase polar histograms for example cells.
  - `fig4_population.png` — population MRL, preferred-phase rose, and entrained fraction (single session).
  - `fig5_sta_summary.png` — spike-triggered LFP average and preferred-phase-vs-MRL scatter.
  - `fig6_cross_session.png` — phase-locking strength and entrained fraction across four rats.

## Reproduce

```bash
python theta_phase_entrainment.py          # regenerates all CSVs and figures
jupytext --to notebook theta_phase_entrainment.py   # rebuild the .ipynb
```

Requires `pynapple`, `pynwb`, `h5py`, `remfile`, `scipy`, `matplotlib`,
`tqdm`, and `jupytext`.
