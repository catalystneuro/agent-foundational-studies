# Hippocampal place cells from DANDI:000044

This run demonstrates the canonical signatures of **hippocampal place cells**
from a single CA1 silicon-probe recording streamed directly from the DANDI
Archive.

## Dataset

- **Dandiset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) —
  *"Diversity in neural firing dynamics supports both rigid and learned
  hippocampal sequences"* (Grosmark, Long, & Buzsáki).
- **Session used:**
  `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
  (rat *Achilles*, 2013-10-25). 137 sorted units (120 pyramidal, 17 inhibitory)
  recorded bilaterally from CA1 with silicon probes while the rat ran on a
  1.6 m linear track for ~33 min between pre- and post-task sleep epochs.
- **Streaming:** the NWB file (~9 GB) is read with `remfile` + a local disk
  cache, so only the byte ranges needed for spike times, position, and
  metadata are downloaded.

## Analysis

`place_cells_analysis.py` (jupytext `.py:percent`, also exported to
`place_cells_analysis.ipynb`):

1. Load the NWB file from S3 and inspect it with Pynapple.
2. Rebuild the linearised-position timestamps (the file's `rate` field
   actually stores `dt`, ~0.0256 s).
3. Define **running periods** — speed > 10 cm/s (computed from 2-D position)
   and at least 5 cm from either reward well — to avoid biasing tuning toward
   the track ends where the rat lingers.
4. Compute 1-D firing-rate-vs-position **tuning curves** for every excitatory
   CA1 unit with `nap.compute_1d_tuning_curves` (50 bins, 3.2 cm each).
5. Quantify spatial coding with **Skaggs spatial information** (bits/spike)
   and call a unit a place cell if SI > 0.5 bits/spike, peak rate > 1 Hz,
   mean rate < 10 Hz.
6. Decode the rat's position from the place-cell ensemble on held-out runs
   using `nap.decode_1d` (250 ms bins, 80/20 train/test split on running
   epochs).

## Key findings (this session)

| metric | value |
|---|---|
| pyramidal cells analysed | 104 |
| place cells (SI > 0.5 b/spk, peak > 1 Hz) | **43** |
| median spatial information | 0.51 bits/spike |
| max spatial information | 2.49 bits/spike |
| median Bayesian decoding error | **8.8 cm** (chance ≈ 40 cm) |

The population heat-map (`fig03_population_heatmap.png`) shows the place
fields tiling the entire 1.6 m track in a clean diagonal — the hallmark
of a hippocampal cognitive map (O'Keefe & Dostrovsky 1971;
Wilson & McNaughton 1993). Held-out Bayesian decoding
(`fig05_decoding.png`) recovers the animal's position to within a few
centimetres, confirming that the population genuinely *encodes* space.

## Files

| file | description |
|---|---|
| `place_cells_analysis.py` | end-to-end jupytext script (run as `python place_cells_analysis.py`) |
| `place_cells_analysis.ipynb` | same content as a Jupyter notebook |
| `fig01_raw_position_and_rasters.png` | first 200 s of position + 6 example unit rasters |
| `fig02_top_place_cells.png` | tuning curves of the 9 most spatially-informative units |
| `fig03_population_heatmap.png` | normalised place fields of all 43 place cells, sorted by peak |
| `fig04_spatial_information.png` | SI distribution & SI-vs-peak-rate scatter |
| `fig05_decoding.png` | Bayesian-decoded position on the longest held-out runs + error histogram |
