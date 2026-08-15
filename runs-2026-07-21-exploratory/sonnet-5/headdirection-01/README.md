# Head Direction Cells in the Antero-Dorsal Thalamus / Postsubiculum

This analysis demonstrates head-direction (HD) tuning using data from
[Dandiset 000056](https://dandiarchive.org/dandiset/000056), "Internally organized
mechanisms of the head direction sense" (Peyrache lab). We used session
`sub-Mouse20/sub-Mouse20_ses-Mouse20-130514_behavior+ecephys.nwb`, which contains
17 spike-sorted units from multi-shank silicon-probe recordings together with two
head-mounted LED trackers and brain-state annotations, streamed directly from the
DANDI S3 bucket with `remfile` (no local download of the ~1.9 GB file). Head
direction was reconstructed from the vector between the red and blue LEDs, and
tuning curves were computed for each unit during the longest continuous open-field
"Awake" bout (32.5 minutes).

To identify genuine head-direction cells, we used a circular-shift shuffle test:
for each unit, 500 random circular shifts of its own spike train were used to build
a null distribution of mean vector length (MVL), and a unit was classified as a
head-direction cell if its observed MVL exceeded this null distribution (`p <
0.05`) and was also large in absolute terms (`MVL > 0.15`), which excludes
high-firing-rate units that reach nominal significance from sample size alone
despite negligible directional modulation. This identified 4 of the 17 units (units
5, 7, 8, and 16) as head-direction cells, each with a sharp, unimodal polar tuning
curve and a distinct preferred direction. The remaining 13 units, including several
very high-firing-rate units, showed flat or only weakly modulated tuning curves and
did not pass the significance criterion.

As a further test of the phenomenon, we fit tuning curves on the first 70% of the
open-field epoch and used Bayesian decoding (`pynapple.decode_bayes`) on the
held-out final 30% to reconstruct the animal's head direction from the spiking of
just these 4 cells. The median absolute circular decoding error was about 33
degrees, well below the 90-degree expectation for a uniform random guess, showing
that this small ensemble carries substantial information about instantaneous head
direction even without any tracking data as input.

## Files

- `head_direction_cells.py` - jupytext (percent format) analysis script, runs
  end-to-end from streaming data access through figure generation.
- `head_direction_cells.ipynb` - executed Jupyter notebook version of the same
  analysis.
- `fig1_raw_data_validation.png` - raw LED tracking, trajectory, and derived
  head-direction trace for a 60 s window (data quality check).
- `fig2_polar_tuning_curves.png` - polar head-direction tuning curves for all 17
  units, with the 4 identified head-direction cells highlighted in red.
- `fig3_mvl_summary.png` - bar chart of directional tuning strength (mean vector
  length) across the population, showing the separation between head-direction
  cells and the rest.
- `fig4_population_raster.png` - spike raster of the 4 head-direction cells
  (sorted by preferred angle) aligned to the simultaneous head-direction trace.
- `fig5_decoding.png` - held-out Bayesian decoding of head direction from the
  4-cell ensemble, and the distribution of decoding error.
