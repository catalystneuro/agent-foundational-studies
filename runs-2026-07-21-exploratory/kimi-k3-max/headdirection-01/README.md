# Head Direction Cells from the DANDI Archive

This analysis demonstrates the head direction (HD) cell phenomenon using dataset
[000056](https://dandiarchive.org/dandiset/000056) from the DANDI Archive:
Peyrache, Lacroix, Petersen and Buzsáki, "Internally organized mechanisms of the
head direction sense" (Nature Neuroscience, 2015). The dataset contains
recordings from the anterodorsal thalamic nucleus and the postsubiculum of
freely moving mice, the two structures at the core of the mammalian head
direction circuit. All data is streamed with LINDI; nothing is downloaded in
full, and all analysis uses Pynapple.

## What was analyzed

The main session (Mouse32-140820, about 2.7 hours) contains 65 sorted units, of
which 42 have spikes, plus the positions of two head-mounted LEDs sampled at
39 Hz. The head-direction angle is reconstructed as the angle of the
blue-to-red LED vector. Because the NWB file carries no epoch annotations, the
open-field foraging epoch is identified automatically: long periods where the
smoothed head speed exceeds 3 px/s and the trajectory spans the full arena
(this rejects movement inside the small rest box). For each unit we compute the
firing rate as a function of head direction (60 bins of 6 degrees) during
exploration, quantify tuning strength with the mean vector length of the tuning
curve, and call a unit an HD cell when its mean vector length exceeds the 99th
percentile of 500 circular time-shift shuffles of its spike train. We then
verify that preferred directions are stable between the first and second half
of exploration, and decode head direction from the HD population with Bayesian
decoding (100 ms bins, trained on the first half, tested on the second). The
core pipeline is finally re-run on four more sessions from four other mice.

## Key findings

In the main session, 21 of 42 units are significantly direction tuned. Their
tuning curves are unimodal with peak rates up to about 100 Hz, and their
preferred directions tile the full circle, the classic signature of a head
direction ensemble. Preferred directions are stable within the environment:
the median split-half tuning-curve correlation is 0.91 for HD cells against
0.12 for untuned units, and the median absolute preferred-direction change is
12 degrees. A Bayesian decoder reads head direction from the 21 HD cells with
a median circular error of 10.9 degrees (84% of 100 ms bins within 30 degrees),
against 88.6 degrees at chance. The result replicates across 5 sessions from 5
mice: 75 of 148 units (51%) are HD cells, with pooled split-half correlations
of 0.83 for HD cells and 0.05 for untuned units.

## Outputs

- `head_direction_cells.py`: consolidated jupytext script, runs end-to-end
- `head_direction_cells.ipynb`: the same analysis as a Jupyter notebook
- `fig1_session_overview.png`: speed and head-direction traces, detected epoch,
  arena coverage, HD occupancy
- `fig2_hd_spike_overlay.png`: spikes of the 8 strongest HD cells on the
  head-direction trace over 90 s of exploration
- `fig3_tuning_curves.png`: polar tuning curves of the 12 strongest HD cells
- `fig4_population_stats.png`: tuning-strength distribution with shuffle
  threshold, preferred-direction coverage, peak rates
- `fig5_stability.png`: split-half stability of preferred direction and tuning
- `fig6_decoding.png`: Bayesian decoding of head direction, error distribution,
  decoded vs actual
- `fig7_multisession.png`: pooled statistics across 5 sessions from 5 mice

## Running

```
python head_direction_cells.py        # end-to-end, about 10 minutes
jupytext --to ipynb head_direction_cells.py
```

Requires `pynapple`, `pynwb`, `lindi`, `h5py`, `numpy`, `matplotlib`, `tqdm`.
