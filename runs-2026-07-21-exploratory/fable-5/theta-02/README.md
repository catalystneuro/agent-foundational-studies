# Theta phase entrainment and phase precession of hippocampal place cells

This analysis demonstrates two related properties of hippocampal spike timing in real
recordings from the DANDI Archive: that CA1 spiking during running is entrained to the
6-10 Hz theta rhythm of the local field potential, and that within a place field the theta
phase of firing advances systematically as the animal moves through the field.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*
(the `hc-11` dataset). Rats run for water reward while dorsal CA1 is recorded with
128-channel silicon probes. Four sessions from three rats were analysed: two on a 1.6 m
linear track (Achilles 10/25/2013, Gatsby 08/02/2013) and two on a ~2.9 m circular track
(Achilles 11/01/2013, Cicero 09/10/2014). Each file provides spike-sorted units labelled
excitatory or inhibitory, a linearized position signal defined only while the animal is
traversing the track, and a 1250 Hz LFP on all 128 channels. Everything is streamed from S3
with `remfile` and a local disk cache: the session files are 8-9 GB each, but the LFP is
chunked one channel at a time, so a single channel over the whole maze epoch costs a few MB.

## What was analysed

Track traversals were taken as the contiguous stretches over which the linearized position is
defined, restricted to running above 10 cm/s. For each session the LFP channel with the
largest theta/delta power ratio was selected automatically (one candidate per shank), and
theta phase was obtained from the Hilbert transform of the 6-10 Hz bandpass-filtered signal
(0° = peak of the filtered LFP on that channel). Place cells were identified per running
direction from smoothed rate maps (peak ≥ 1 Hz, Skaggs information ≥ 0.5 bits/spike,
odd/even-lap map correlation ≥ 0.4). Phase precession was quantified for each place field by
circular-linear regression (Kempter et al., 2012) of spike theta phase on normalized position
within the field, with a within-field phase shuffle as the null and a NeMoS Poisson GLM as an
independent, model-based check. All computations use Pynapple containers and tuning-curve
machinery.

## Key finding

Both phenomena are present and replicate across sessions and animals. During running, 62-75%
of CA1 pyramidal cells and essentially every interneuron fire non-uniformly across the theta
cycle (Rayleigh p < 0.01); interneurons lock more strongly than pyramidal cells and prefer a
theta phase roughly 130° away from them, and the spike-triggered LFP average of pooled
pyramidal spikes is itself a theta oscillation with a trough at the spike. Within place
fields, theta phase falls monotonically with distance travelled into the field: of 155
analysable place fields pooled across the four sessions, 75% (68-80% per session) show a
significant negative circular-linear phase-position slope, with a median advance of 247°
(about 0.7 theta cycles, or 5.1°/cm) across a field of median width 48 cm and a median
circular-linear correlation of −0.27. The effect disappears entirely when spike phases are
shuffled within a field, it is visible on individual traversals rather than only after pooling
laps, and a Poisson GLM whose preferred phase is allowed to rotate with position predicts
held-out spikes better than an otherwise identical GLM with a fixed preferred phase in 37 of
37 fields tested (Wilcoxon p = 1.5e-11). Because the average rate map is by construction blind
to spike timing, this is a direct demonstration that CA1 place cells carry positional
information in the theta phase of their spikes over and above their firing rate.

## Files

| File | Contents |
| --- | --- |
| `theta_precession_dandi000044.py` | Consolidated, self-contained jupytext script (percent format) that runs the whole analysis end to end and writes every figure |
| `theta_precession_dandi000044.ipynb` | The same notebook with executed outputs |
| `theta_lib.py` | Shared library used by the modular prototype scripts |
| `01_load_and_inspect.py` | Streaming, stream-by-stream validation of one session |
| `02_place_fields_and_theta.py` | Place fields and theta phase locking (figures 3-4) |
| `03_phase_precession.py` | Precession, shuffle control, single-pass check (figures 5-7) |
| `04_multi_session.py` | Four-session replication (figure 8), writes the two CSVs |
| `05_glm_position_phase.py` | NeMoS GLM model comparison (figure 9) |
| `session_summary.csv` | One row per session: laps, cells, place fields, % precessing, % theta-locked |
| `place_field_precession.csv` | One row per place field: field bounds, width, spike count, slope, ρ, p |

### Figures

| Figure | Contents |
| --- | --- |
| `fig01_raw_streams.png` | Position, speed, raw and theta-filtered LFP, and the pyramidal raster during one traversal |
| `fig02_behavior_and_spectrum.png` | Detected traversals, LFP power spectrum during running, speed distribution |
| `fig03_place_fields.png` | Directional rate maps, example place fields, place-cell selection criteria |
| `fig04_theta_entrainment.png` | Spike-phase histograms by cell type, preferred phase vs. locking strength, spike-triggered LFP average |
| `fig05_precession_examples.png` | Six single place fields: rate map and phase vs. position with the circular-linear fit |
| `fig06_precession_population.png` | Pooled phase-position density, population phase advance, slope and ρ distributions vs. shuffle |
| `fig07_single_passes.png` | Precession on individual traversals of one field |
| `fig08_multisession_summary.png` | Four-session replication: pooled density, per-session slopes and precessing fractions, locking by cell type |
| `fig09_glm_position_phase.png` | NeMoS GLM comparison and the fitted preferred phase across the field |

## Caveats

The absolute preferred phase reported here is relative to the peak of the 6-10 Hz filtered
signal on the selected channel and is not corrected for recording depth or the sign convention
of the file, so only relative phase statements (pyramidal cells versus interneurons, or phase
change across a field) should be read from the numbers. Fields whose peak falls within 15 cm
of a track end are excluded because the animal never traverses them fully, and fields narrower
than 15 cm, wider than 120 cm, or with fewer than 40 in-field spikes are not analysed; the
remaining 155 fields are a subset of the 257 directional place fields detected. The two
circular-track sessions are traversed in one direction only, so their fields contribute a
single direction each.

Two quirks of the source files are worth recording. The `rate` attribute of the position
SpatialSeries is actually the sampling period (0.0256 s, i.e. 39.06 Hz) rather than a rate; the
loader detects this and asserts that the resulting duration matches the maze epoch. And one
unit in the Gatsby file (unit 1) has spike times that are not monotonically increasing in the
file; Pynapple sorts them on construction, which is why a "timestamps are not sorted" warning
appears for that session.
