# Hippocampal replay during sharp-wave ripples

This analysis demonstrates **hippocampal memory replay** by decoding the spatial
trajectory that CA1 place-cell ensembles represent during sharp-wave ripple (SWR)
events, using real electrophysiology from the DANDI Archive.

## Dataset

**DANDI:000044** — *"Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences"* (Grosmark, Long & Buzsáki, 2016). Bilateral CA1 silicon-probe
recordings in rats running a 1.6 m linear track, flanked by pre-run and post-run sleep.
This analysis uses one session, `sub-Achilles/ses-Achilles-10252013`. The NWB file
(~9 GB) is streamed with `remfile` and a local cache; only the units, the linearized
position, and a single ripple-band LFP channel are read (the LFP is chunked one channel
at a time, so ripple detection pulls only ~90 MB).

## What was analyzed

The pipeline (in `hippocampal_replay.py`, converted to `hippocampal_replay.ipynb`) runs
end-to-end:

1. **Place fields** — direction-specific firing-rate tuning curves vs. linearized
   position during running (>10 cm/s), restricted to the track interior [0.1, 1.5] m so
   that reward-zone / turnaround firing at the boundaries does not swamp the interior
   fields. 55 cells with well-localized single fields form the decoding ensemble
   (`fig1`).
2. **Decoder validation** — a memoryless Bayesian decoder (Poisson likelihood, uniform
   prior) reconstructs the animal's true position during running with ~20 cm median error
   on the 140 cm interior; the decoded-vs-true confusion matrix lies on the diagonal
   (`fig3`).
3. **Ripple detection** — the CA1 LFP is band-passed at 150–250 Hz, Hilbert-enveloped,
   z-scored against sleep, and thresholded (5 SD peak, 2 SD edges, 20–250 ms). This
   yields 4976 POST-sleep ripples at a physiological ~0.34 Hz, with textbook morphology
   and a ~50 ms modal duration (`fig2`).
4. **Replay decoding** — within each POST ripple, spikes are binned at ~15 ms and decoded
   to a posterior over position per bin. Sequential structure is scored by the weighted
   correlation between decoded position and time, and tested against a place-field
   circular-shift shuffle (300 shuffles per event) (`fig4`, `fig5`).

## Key finding

Roughly **one in five post-sleep sharp-wave ripples (700 of 3467, 20.2%) contains a
statistically significant, sequential spatial trajectory** — a fourfold enrichment over
the 5% chance level expected from the shuffle. Significant events have high sequence
scores (|weighted correlation| ≈ 0.6–0.8), sweep a median of ~1.1 m of the track within
~100 ms, and occur in both **forward** (same order as the animal's running) and
**reverse** directions in roughly equal numbers. Individual examples show a decoded
posterior sweeping smoothly across the track alongside the corresponding place cells
firing in place-field order (`fig4`).

This is the signature of hippocampal replay: during sharp-wave ripples, place-cell
ensembles reactivate the compressed spatial trajectories the animal experienced on the
track, the proposed neural substrate of memory consolidation.

## Files

- `hippocampal_replay.py` — consolidated jupytext pipeline (runs end-to-end from
  streaming; no manual steps).
- `hippocampal_replay.ipynb` — executed notebook version.
- `fig0_raw_data.png` — behavior on the track and raw CA1 LFP.
- `fig1_place_fields.png` — direction-specific place-field maps.
- `fig2_ripples.png` — example ripple and detection summary.
- `fig3_decode_validation.png` — decoder confusion matrix and example.
- `fig4_replay_examples.png` — example replay sweeps with spike rasters.
- `fig5_replay_summary.png` — replay prevalence, score distribution, trajectory spans.
- `analysis_*.py`, `cache_data.py` — the modular development scripts used to build and
  validate each stage.

## Reproducing

```
pip install dandi pynwb remfile h5py pynapple scipy matplotlib tqdm jupytext
python hippocampal_replay.py          # streams from DANDI, writes all figures
jupytext --to notebook hippocampal_replay.py   # or run the .ipynb directly
```
