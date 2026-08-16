# Sharp-wave ripples and replay in rat hippocampal CA1

**Dataset**: DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044)
("Diversity in neural firing dynamics supports sparsity and specificity in memory",
Buzsaki lab), session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`.
The session contains 137 CA1 units (120 excitatory, 17 inhibitory), 128-channel
LFP at 1250 Hz, sleep-state labels (Awake / Non-REM / REM), and position on a
1.6 m linear maze, organized into PRE-sleep, Maze (~34 min), and POST-sleep
epochs. The 8.7 GB NWB file is streamed with remfile plus a local disk cache;
nothing is downloaded in full.

**What was analyzed**. Sharp-wave ripples were detected on the best CA1 LFP
channel (chosen data-driven by ripple-band envelope peakiness x ripple/delta
power on a POST Non-REM block; channel 117) using the standard recipe:
100-250 Hz bandpass, Hilbert envelope smoothed with a 4 ms Gaussian, peak
threshold mean+4SD with edges at mean+1SD, merging gaps <30 ms, keeping 30-500
ms events fully inside Non-REM. This yielded 8369 Non-REM SWRs (5426 PRE, 2943
POST; ~31-33 per minute of Non-REM; median duration 60 ms) with the canonical
~150 Hz oscillation riding on a sharp wave. Place fields were computed from
maze run bouts (per-direction ratemaps, 50 bins over the 1.6 m track), and
88/120 excitatory cells passed the place-cell criteria (Skaggs spatial
information significant against 200 circular time-shift shuffles on the
concatenated run-bout axis, p<0.05, peak rate >=1 Hz). Each SWR was then
decoded with a Bayesian decoder over the direction-pooled place-cell template
(20 ms bins), and replay strength was quantified as the posterior-mass-weighted
correlation between event time and decoded position, with significance from
500 cell-ID shuffles per event.

**Key finding**. POST-sleep SWRs replay the maze trajectory at well above
chance: 11.0% of the 654 decodable POST events show significant sequential
replay (46 forward, 26 reverse), versus 4.5% of 1628 PRE-sleep events, which
sit at the 5% chance level expected from the shuffle. The full replay-strength
distributions differ strongly (Mann-Whitney p = 1.0e-9). Peri-SWR firing of
both excitatory and inhibitory populations rises ~20-25 z at the ripple center,
and SWRs with higher place-cell participation show stronger replay. This
reproduces the classic result of Foster & Wilson (2006) and Diba & Buzsaki
(2007): after spatial experience, sleep sharp-wave ripples reactivate
compressed place-cell sequences that sweep along the experienced trajectory.

## Files

- `swr_replay_achilles.py`: consolidated jupytext script (percent format);
  runs the full pipeline end-to-end (~10 min) and writes all figures.
- `swr_replay_achilles.ipynb`: the same analysis as a Jupyter notebook
  (converted with jupytext).
- `figures/fig1_swr_detection.png`: SWR detection: raw LFP, ripple band and
  envelope with thresholds, example event, duration/rate/interval statistics,
  mean ripple waveform.
- `figures/fig2_place_fields.png`: place-field template (88 cells sorted by
  peak) and example per-direction ratemaps.
- `figures/fig3_replay.png`: example forward and reverse replay events
  (posterior heatmaps with spike overlays), replay-strength distributions
  PRE vs POST, fraction significant, and replay direction over POST.
- `figures/fig4_periswr.png`: peri-SWR firing modulation by cell class, a
  place-cell raster around one SWR, and participation vs replay strength.
- `scripts/`: the modular development scripts (01-10) that the consolidated
  script was built from, plus cached intermediate results (`.npz`).

## Reproducing

```
python swr_replay_achilles.py          # end-to-end, writes figures/*.png
jupytext --to notebook swr_replay_achilles.py   # regenerate the .ipynb
```

Requirements: pynapple 0.11, pynwb 4.0, h5py, remfile, scipy, matplotlib,
tqdm, requests, jupytext.
