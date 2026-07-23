# Auditory Frequency Tuning

This analysis demonstrates auditory frequency tuning using [DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex Neuropixels recordings and pupil diameter traces from mice during passive exposure to pure tones." The dataset contains 15 sessions from 5 head-fixed mice in which a Neuropixels probe recorded spiking activity in auditory cortex while brief (25 ms) pure tones at five frequencies (2, 4, 8, 16, and 32 kHz, one octave apart) were played at a fixed 60 dB amplitude, along with running speed and pupil diameter. Data were streamed directly from the DANDI S3 bucket with `remfile` and analyzed with `pynapple`, without downloading full files.

For every unit, we computed the mean spike rate in a 100 ms window after tone onset for each frequency and compared it to a 100 ms pre-tone baseline, testing for frequency selectivity with a one-way ANOVA across the five frequency conditions. We prototyped this pipeline on one session (`sub-LA3_ses-3`, 35 units), where 25 of 35 units (71%) were significantly tuned (p < 0.01), each with a clear best frequency and a several-fold increase in evoked rate over baseline. We then scaled the same pipeline to all 15 sessions, which was cheap to do because only spike times and the trial table need to be streamed, not the raw ephys traces. Across the full dataset (1564 units), 85.6% of units showed significant frequency tuning. Best frequencies were distributed across the full 2-32 kHz range tested, and averaging each tuned unit's normalized tuning curve after aligning it to its own best frequency (expressed as octave distance from BF) produced a roughly symmetric fall-off over +/-2 octaves, the classic signature of band-pass frequency tuning in auditory cortex.

## Files

- `auditory_frequency_tuning.py` - consolidated jupytext analysis script (source of truth, runs end-to-end)
- `auditory_frequency_tuning.ipynb` - the same analysis as an executed Jupyter notebook
- `fig1_raw_data_overview.png` - raw spike raster, running speed, and pupil diameter with tone presentations overlaid
- `fig2_example_psth.png` - peri-stimulus rasters and PSTHs by frequency for three example units
- `fig3_tuning_curves_examples.png` - evoked-rate tuning curves for the same example units
- `fig4_population_heatmap.png` - normalized tuning curves for all significantly tuned units in one session, sorted by best frequency
- `fig5_population_summary.png` - population summary across all 15 sessions: fraction tuned per session, pooled best-frequency distribution, and the BF-aligned population tuning curve
