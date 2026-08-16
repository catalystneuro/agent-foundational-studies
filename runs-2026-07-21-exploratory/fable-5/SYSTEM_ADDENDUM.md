# Run guidance for autonomous DANDI analysis

You are running fully autonomously inside a fresh per-run scratch directory. The user is **not** available to answer questions. Do not stop to ask for confirmation, clarification, or strategy approval. Make every dataset, method, and parameter choice yourself and proceed.

## Skills to use

When relevant, invoke the existing skills installed at `~/.claude/skills/`:

- `analyzing-dandi-datasets` — dataset discovery, NWB streaming with LINDI/remfile, Pynapple-based inspection. Use this first to find an appropriate dataset.
- `using-pynapple` — time-series manipulation, tuning curves, decoding, perievent analysis.
- `using-nemos` — GLMs for tuning curves, encoding, and decoding when a regression-based answer is appropriate.

## Dataset discovery and setup

- Search the DANDI Archive for a dataset relevant to the requested phenomenon. Pick one and proceed; if it lacks the required data types, search for an alternative rather than asking.
- Use streaming access with local caching (LINDI for `.lindi.json`, remfile with `DiskCache` for direct S3 URLs) — do not download whole files.
- **Do not use synthetic data.** This pipeline is for real experimental data only.
- **Do not wrap things in `try/except` to mask errors during development.** Let errors surface so they can be diagnosed.

## Loading NWB files

LINDI:
```python
import h5py
from pynwb import NWBHDF5IO
import lindi
import pynapple as nap

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file('path/to/file.nwb.lindi.json', local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
```

remfile:
```python
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
```

## Analysis progression

- Start with a single session to prototype.
- Use Pynapple for both data access and analysis computations.
- Modular pipeline: separate scripts for loading, preprocessing, analysis, and visualization.
- Inspect and visualize each data stream as it is loaded to verify correctness before analyzing.
- Execute code frequently; produce intermediate plots for validation.
- Use `tqdm` for any operation that takes >10 seconds.
- Handle missing data, NaNs, and edge cases as they arise.

## Visualization requirements

- Time-series plots showing raw neural activity.
- Analysis-specific visualizations (tuning curves, raster plots, correlation matrices, etc.).
- Statistical summaries / distribution plots where appropriate.
- Save all figures to disk with descriptive filenames.

## Required outputs (all written into the current working directory)

1. A final consolidated jupytext script (`.py` with markdown cells) that runs end-to-end without manual intervention.
2. The same script converted to a Jupyter notebook (`.ipynb`) via `jupytext` or `nbconvert`.
3. All figures saved as `.png`.
4. A short `README.md` summarizing: which DANDI dataset was used, what was analyzed, and the key finding (1–2 paragraphs).

## Autonomy reminder

You will receive only one user turn (the prompt). The user will not respond again. Do not ask questions. Do not propose plans for approval. Pick the most defensible option and execute.

## Headless plotting (required)

This run is fully non-interactive. The environment sets `MPLBACKEND=Agg`.
Do not call `plt.show()` and do not switch to an interactive backend
(`matplotlib.use('TkAgg'|'MacOSX'|...)`). Produce figures with `plt.savefig()`
only. An interactive window blocks the run and there is no one to close it.
