# Decoding an upcoming decision bias from pre-stimulus activity

*(results section filled in at the end of the run from `results_summary.txt`)*

## Dataset

[DANDI:000409](https://dandiarchive.org/dandiset/000409), the *IBL Brain Wide Map*:
Neuropixels recordings from mice performing the International Brain Laboratory visual
decision task. The processed NWB files (`*_desc-processed_behavior+ecephys.nwb`) were
streamed from S3 with `remfile` and a local disk cache and opened through Pynapple; no file
was downloaded in full.

## Files

| file | what it does |
| --- | --- |
| `ibl_common.py` | streaming access, trials table, Pynapple spike/behaviour extraction, pseudo-session generator |
| `decoding.py` | contiguous-fold cross-validated decoders, surrogate nulls |
| `session_analysis.py` | the whole per-session analysis |
| `01_scan_sessions.py` | screens candidate sessions against pre-specified criteria |
| `02_run_all.py` | runs every selected session, writes `all_results.pkl` / `session_summary.csv` |
| `03_figures.py` | writes all figures and `results_summary.txt` |
| `04_make_notebook.py` | jupytext -> executed `.ipynb` |
| `decision_bias_prestimulus.py` | consolidated jupytext narrative (percent format) |
| `decision_bias_prestimulus.ipynb` | the same, executed |

## Reproducing

```bash
python 01_scan_sessions.py     # ~6 min, screens 30 sessions
python 02_run_all.py           # ~1-2 h on a cold cache, writes all_results.pkl
python 03_figures.py           # seconds
python 04_make_notebook.py     # executes the notebook
```
