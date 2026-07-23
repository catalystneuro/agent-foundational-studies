# Orientation and Direction Selectivity in Mouse Visual Cortex

This analysis uses [DANDI:000039](https://dandiarchive.org/dandiset/000039), "Allen
Institute – Contrast tuning in mouse visual cortex with calcium imaging" (Millman &
de Vries, 2023). The dataset contains two-photon calcium imaging (GCaMP6f) from mouse
visual cortex, recorded while animals viewed full-field drifting gratings at 8
directions (0-315 degrees in 45 degree steps) and 6 contrasts (5-80%), across several
excitatory and inhibitory Cre lines and cortical layers. Sessions are shared as NWB
files and were streamed directly from the DANDI S3 bucket with `remfile` (disk-cached
locally), so no full files were downloaded.

The analysis pools regions of interest (ROIs, treated here as putative single
neurons) from 10 sessions across different subjects and Cre lines. For each ROI, we
computed a trial-averaged tuning curve across the 8 grating directions at the
contrast that drove the strongest responses in that session, then summarized tuning
with two standard vector-based metrics: the orientation selectivity index (OSI,
sensitive to the grating axis, using doubled angles) and the direction selectivity
index (DSI, sensitive to direction of motion). A Kruskal-Wallis test across
directions was used to flag cells with a statistically significant dependence of
response on grating direction. Pynapple (`TsdFrame`, `IntervalSet`) was used for all
time-series alignment between the dF/F traces and stimulus epochs.

Of 114 pooled ROIs, 58% were significantly direction-tuned (Kruskal-Wallis p<0.05),
and OSI values in that significantly tuned group were shifted toward higher values
than in the non-tuned group. Preferred directions were distributed across the full
range of grating directions rather than clustered at one value, and a heatmap of
peak-normalized tuning curves sorted by preferred direction shows a clear diagonal
band, meaning each cell's response is concentrated near its own preferred direction.
Together these are the expected population-level signature of orientation and
direction selectivity in mouse visual cortex.

## Contents

- `orientation_selectivity_analysis.py` — jupytext (percent format) script that runs
  the full pipeline end-to-end: streaming data access, single-session prototyping,
  the 10-session population analysis, and all figure generation.
- `orientation_selectivity_analysis.ipynb` — the same analysis as an executed Jupyter
  notebook.
- `fig1_raw_traces.png` — raw dF/F traces from one session with stimulus epochs
  shaded by grating direction.
- `fig2_example_tuning_curves.png` — linear and polar tuning curves for the 6 most
  orientation-selective, significantly tuned cells.
- `fig3_population_osi_dsi.png` — population histograms of OSI and DSI, split by
  tuning significance.
- `fig4_preferred_direction_distribution.png` — rose histogram of preferred
  directions across significantly tuned cells.
- `fig5_population_tuning_heatmap.png` — peak-normalized tuning curves for all
  significantly tuned cells, sorted by preferred direction.
