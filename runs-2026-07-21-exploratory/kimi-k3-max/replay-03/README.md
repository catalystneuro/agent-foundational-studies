# Hippocampal replay during sharp-wave ripples

**Dataset:** DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044)
(Grosmark & Buzsáki 2016, *"Diversity in neural firing dynamics supports both rigid and
learned hippocampal sequences"*), session
`sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1
tetrode recording (137 sorted units: 120 excitatory, 17 inhibitory), 128-channel LFP at
1250 Hz, and 39 Hz position tracking on a 1.6 m linear track, across PRE sleep, ~35 min
of maze running, and POST sleep. The 8.7 GB NWB file is streamed with remfile + a local
disk cache (no full download).

**Analysis.** A spatial template was built from the maze epoch: run bouts were extracted
from the linearized position (86 bouts, 245 s of running), and 87 of 120 excitatory units
qualified as place cells (Skaggs spatial information above a 500-draw circular time-shift
shuffle, mean rate >0.1 Hz, peak >1 Hz). Sharp-wave ripples were detected in Non-REM sleep
from the 100–250 Hz LFP envelope on the data-driven best channel (ch 117; peak > mean+4 SD,
edges > mean+1 SD, 30–500 ms duration): 5470 PRE and 2788 POST events (~31/min of Non-REM,
median 48 ms). Each SWR was Bayesian-decoded in 20 ms bins against the place-field template,
and trajectory structure was scored with the posterior-mass-weighted correlation between
time and position, tested against 500 cell-ID shuffles per event (events required ≥5 spiked
bins and ≥5 active place cells).

**Key finding.** SWRs during POST sleep encode spatial trajectories of the track far above
chance: 41/211 decodable POST events (19.4%) are significant vs 29/637 (4.6%) in PRE sleep,
where the shuffle-calibrated false-positive rate is 5% (Fisher exact p = 3.2e-10; |score|
distributions differ at Mann-Whitney p = 1.9e-10). Significant events sweep a large fraction
of the 1.6 m track within ~100 ms (~20× real-time speed) in both the forward (23) and
reverse (18) directions, with no significant directional bias in this session (binomial
p = 0.53). This is the classic signature of hippocampal replay: experience on the maze is
reactivated as compressed spatial trajectories during subsequent sleep ripples, at rates
absent before the experience.

## Files

- `hippocampal_replay_achilles.py`: consolidated jupytext script, runs end-to-end
  (~4–5 min with a warm `/tmp/remfile_cache`; longer on first run while LFP streams)
- `hippocampal_replay_achilles.ipynb`: same analysis as a Jupyter notebook
- `fig1_raw_data_overview.png`: linearized trajectory, run bouts, example place-cell raster
- `fig2_place_fields.png`: 87 place fields tiling the track, SI distributions, shuffle nulls
- `fig3_ripple_detection.png`: channel selection, SWR duration distribution, example SWR,
  peri-SWR place-cell raster
- `fig4_replay.png`: example decoded forward/reverse trajectories and PRE/POST statistics
- `01_load_data.py` … `07_fig_replay.py`: modular development scripts (same pipeline in steps)

## Reproduce

```bash
python hippocampal_replay_achilles.py        # or run the notebook
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `scipy`, `matplotlib`, `tqdm`
(tested with pynapple 0.11.2, pynwb 4.0.0).
