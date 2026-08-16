"""
Assemble the standalone jupytext deliverable from the validated module sources.

The notebook must run on a bare checkout, so the module code is inlined into
code cells (with the `module.` prefixes stripped, since everything ends up in a
single namespace) rather than imported.
"""

import re
import textwrap

SRC = {name: open(f"{name}.py").read() for name in
       ["orilib", "analysis", "decoding", "glm", "figures", "pooled"]}


def strip(name, drop_imports=(), subs=()):
    s = SRC[name]
    lines = []
    for ln in s.splitlines():
        if any(ln.strip() == d for d in drop_imports):
            continue
        lines.append(ln)
    s = "\n".join(lines)
    for a, b in subs:
        s = re.sub(a, b, s)
    # drop the module docstring (it is reproduced as a markdown cell instead)
    s = re.sub(r'\A"""(.*?)"""\n', "", s, flags=re.S).lstrip("\n")
    return s.rstrip() + "\n"


COMMON_DROP = ("import orilib as O", "import analysis as A", "import decoding as D",
               "import glm as G", "import figures as F", "import pooled as P",
               "from analysis import DIRECTIONS, STATIC_ORIS",
               "from analysis import DIRECTIONS",
               'matplotlib.use("Agg")')

cells = []


def md(text):
    cells.append(("markdown", textwrap.dedent(text).strip()))


def code(text):
    cells.append(("code", text.rstrip()))


# ---------------------------------------------------------------------------- intro
md(r"""
# Orientation selectivity in the mouse visual system

Neurons in visual cortex respond selectively to the *orientation* of an edge or
grating in their receptive field. A cell that fires strongly to a vertical bar
fires weakly to a horizontal one, and, because an orientation is defined modulo
180 degrees, it responds about equally well to a grating drifting left-to-right
and to the same grating drifting right-to-left. Orientation selectivity is the
canonical example of a computed feature: it is largely absent in the retinal
input reaching the dorsal lateral geniculate nucleus and emerges in cortex.

This notebook demonstrates the phenomenon in real extracellular recordings from
the DANDI Archive and asks four questions, each with an explicit control:

1. **Do single units show orientation tuning?** Direction tuning curves and a
   within-block label-shuffle permutation test.
2. **Is it specific to visual structures?** The same analysis applied to
   simultaneously recorded hippocampal units, which serve as a negative control.
3. **Does the preference generalise across stimuli?** Preferred orientation
   estimated from drifting gratings is compared with the value estimated from
   *static* gratings, an independent stimulus class recorded in the same session.
4. **Does it survive a behavioural confound?** A Poisson GLM (NeMoS) asks whether
   grating direction predicts held-out spike counts over and above running speed,
   which strongly modulates firing rate in mouse visual cortex.

## Dataset

**DANDI:000021**, *Allen Institute Visual Coding, Neuropixels (Brain Observatory
1.1 stimulus set)*. Head-fixed mice on a running wheel viewed a battery of visual
stimuli while up to six Neuropixels probes recorded simultaneously from visual
cortex, visual thalamus, hippocampus and midbrain. Two stimulus classes are used
here:

| stimulus | conditions | repeats | duration |
|---|---|---|---|
| drifting gratings | 8 directions x 5 temporal frequencies | 15 | 2 s |
| static gratings | 6 orientations x 5 spatial frequencies x 4 phases | ~50 | 0.25 s |

Files are read by streaming from the DANDI S3 bucket with `remfile` plus a local
disk cache; nothing is downloaded in full. Eight sessions are analysed.
""")

md("""
## Setup

`jax` is configured for 64-bit before NeMoS is imported, and matplotlib runs on
the non-interactive Agg backend so the notebook executes headless.
""")
code("""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
jax.config.update("jax_enable_x64", True)

import glob
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import nemos as nmo
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

N_SESSIONS = int(os.environ.get("N_SESSIONS", "8"))
OUT = "session_results"
os.makedirs(OUT, exist_ok=True)
print("pynapple", nap.__version__, "| nemos", nmo.__version__)
""")

# ------------------------------------------------------------------- data access
md("""
## 1. Data access

Session-level NWB assets are listed through the DANDI REST API. Loading is
deliberately column-selective: `units.to_dataframe()` would pull `waveform_mean`
and `spike_amplitudes`, several hundred megabytes per session that this analysis
never touches.

One subtlety matters for correctness. `nap.TsGroup` sorts its keys, so the
per-unit metadata frame has to be reordered to match, or every unit annotation
(brain area, quality metrics) is silently shuffled relative to its spike train.
The assertion in `load_units` guards against that.
""")
code(strip("orilib", COMMON_DROP))

md("""
### Unit quality control

Units are kept if they are labelled `good` by the Allen sorting pipeline and pass
the standard metric thresholds (ISI violations < 0.5, amplitude cutoff < 0.1,
presence ratio > 0.9, SNR > 1). This removes roughly two thirds of the raw
clusters.
""")

# ------------------------------------------------------------------- analysis
md(r"""
## 2. Tuning curves and selectivity metrics

For each trial the firing rate is the spike count inside the presentation window
divided by its duration. Rates are then averaged within each stimulus condition.
Drifting-grating trials are pooled across the five temporal frequencies, giving
75 repeats of each of the 8 drift directions; static-grating trials are pooled
across spatial frequency and phase, giving roughly 200 repeats of each of the
6 orientations.

Selectivity is quantified with resultant-vector (circular-variance) indices. With
mean rate $r_k$ at stimulus angle $\theta_k$,

$$\mathrm{OSI} = \frac{\left|\sum_k r_k e^{2i\theta_k}\right|}{\sum_k r_k},
\qquad
\mathrm{DSI} = \frac{\left|\sum_k r_k e^{i\theta_k}\right|}{\sum_k r_k},
\qquad
\theta_{\text{pref}} = \tfrac12 \arg\sum_k r_k e^{2i\theta_k}.$$

Doubling the stimulus angle maps orientation (defined modulo 180 degrees) onto
the full circle, so the same expression serves drift directions and static
orientations. A ratio index $(R_{\text{pref}} - R_{\text{orth}}) /
(R_{\text{pref}} + R_{\text{orth}})$ is reported alongside it.

Significance comes from a permutation test in which stimulus labels are shuffled
**within each stimulus block**. The drifting-grating trials are delivered in
three blocks spread over a two-and-a-half-hour session, and firing rates drift
slowly over that timescale; shuffling within block means slow drift cannot
masquerade as tuning. Split-half reliability (the correlation between tuning
curves computed from independent random halves of the trials) is reported as a
model-free measure of how repeatable each curve is.
""")
code(strip("analysis", COMMON_DROP, subs=[(r"\bO\.", "")]))

# ------------------------------------------------------------------- single session
md("""
## 3. Prototype on a single session

Before scaling up, run the whole pipeline on one session and look at the raw
spikes.
""")
code("""
assets = get_assets()
print(f"{len(assets)} session-level assets in dandiset {DANDISET}")
first = assets[0]
print(first["path"], round(first["size"] / 1e9, 2), "GB")

nwbfile = open_nwb(first["url"])
print("session", nwbfile.session_id, "|", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.genotype,
      nwbfile.subject.age, nwbfile.subject.sex)
for k, v in nwbfile.intervals.items():
    print(f"  {k:42s} {len(v):6d} presentations")
""")
code("""
tsg_all, meta_all = load_units(nwbfile)
qc = passes_qc(meta_all)
print(f"{len(meta_all)} sorted units, {qc.sum()} pass quality control")
print(meta_all[qc].area.value_counts().head(12))
""")
code("""
dg_preview = _clean_trials(
    load_stim_table(nwbfile, "drifting_gratings_presentations"), DIRECTIONS)
print(dg_preview.groupby("orientation").size())
print("presentation duration:",
      np.round(np.median(dg_preview.stop_time - dg_preview.start_time), 3), "s")
print("blocks:", sorted(dg_preview.stimulus_block.unique()))
""")

md("""
### Figures

The plotting code for the whole notebook is defined in one place below.
""")
code(strip("figures", COMMON_DROP, subs=[(r"\bO\.", "")]))

code("""
proto = analyze_session(first["url"])
running = load_running_speed(proto["nwbfile"])
print(fig_raw_data(proto, running))
""")
md("""
![raw data](fig01_raw_data.png)

The example unit fires in bursts confined to a subset of grating presentations
while hippocampal units below fire continuously and without regard to the
stimulus. The animal runs almost throughout this window, so the modulation is not
simply locomotion.
""")
code("""
print(fig_example_units(proto))
""")
md("""
![example units](fig02_example_units.png)

Each of these units responds to *two* drift directions 180 degrees apart, which is
the signature of orientation rather than direction selectivity: the polar curves
are bilobed and the DSI values are small. Responses are sustained for the full 2 s
presentation.
""")

# ------------------------------------------------------------------- multi session
md("""
## 4. Scale to eight sessions

The same pipeline is run over the eight smallest session files of the dandiset
(file size is unrelated to the biology; it mostly tracks recording duration and
probe count). Results are cached to disk so the notebook can be re-executed
cheaply.
""")
code("""
def run_all_sessions(n=N_SESSIONS):
    for a in tqdm(get_assets()[:n], desc="sessions"):
        sid = a["path"].split("ses-")[1].replace(".nwb", "")
        fp = os.path.join(OUT, f"{sid}.pkl")
        if not os.path.exists(fp):
            res = analyze_session(a["url"])
            keep = {k: res[k] for k in ("session_id", "units", "tc_dg", "tc_sg",
                                        "dg_rates", "dg_labels", "dg_tf",
                                        "sg_rates", "sg_labels")}
            keep["dg_blocks"] = res["dg_table"]["stimulus_block"].values
            pickle.dump(keep, open(fp, "wb"))
        # per-trial running speed, used by the GLM below
        rfp = os.path.join(OUT, f"{sid}_running.npz")
        if not os.path.exists(rfp):
            nwbf = open_nwb(a["url"])
            run = load_running_speed(nwbf)
            dg = _clean_trials(
                load_stim_table(nwbf, "drifting_gratings_presentations"), DIRECTIONS)
            np.savez(rfp, speed=trial_mean_speed(
                run, dg["start_time"].values, dg["stop_time"].values))


def trial_mean_speed(running, starts, stops):
    t, v = running.t, running.d
    csum = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, starts, "left")
    hi = np.searchsorted(t, stops, "right")
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)


run_all_sessions()
""")
code(strip("pooled", COMMON_DROP, subs=[(r"def pool\(", "def pool_sessions(")]))
code("""
sessions = load_sessions(OUT)[:N_SESSIONS]
pooled = pool_sessions(sessions)
units = pooled["units"]
print(f"{len(sessions)} sessions, {len(units)} quality-passing units")
print(units.area.value_counts().head(15))
""")

md("""
## 5. Is the tuning orientation tuning, and is it specific to visual areas?
""")
code("""
print(fig_population_tuning(pooled))
""")
md("""
![population tuning](fig03_population_tuning.png)

Sorting every significantly tuned cortical unit by its preferred direction
produces the expected diagonal, and a clear second band appears 180 degrees away:
these cells respond to both directions of motion along their preferred axis. The
mean curve aligned to each unit's preferred direction has a second peak at 180
degrees reaching about 70% of the primary peak. In hippocampus the diagonal is
present by construction (rows are sorted by their own maximum) but there is no
second band and the aligned curve is flat away from the trivially selected peak,
which is what a curve made of noise looks like.
""")
code("""
print(fig_osi_by_area(pooled))
""")
md("""
![OSI by area](fig04_osi_by_area.png)

OSI in every visual cortical area sits far above its own shuffled null, and about
80% of cortical units are individually significant. Visual thalamus (LGd, LP) is
intermediate, and the hippocampal subfields are close to the null. The residual
above-chance fraction in CA1 is discussed at the end.
""")

md("""
## 6. Does the preference generalise to a different stimulus?

The strongest evidence that a tuning curve reflects a real receptive-field
property rather than a fluctuation is that it replicates on data the estimate was
not taken from. Static gratings were presented in the same sessions: brief (0.25 s)
stationary gratings at 6 orientations, with no motion at all. Preferred
orientation is estimated independently from each stimulus class and the two
estimates compared.
""")
code("""
print(fig_cross_stimulus(pooled))
""")
md("""
![cross-stimulus](fig05_cross_stimulus.png)

Preferred orientations line up along the identity (and its wrap-around), with a
median absolute discrepancy far below the ~45 degrees expected if the two
estimates were unrelated.
""")

# ------------------------------------------------------------------- decoding
md("""
## 7. Population decoding

Single-unit selectivity should translate into decodable population information.
A multinomial logistic decoder is trained on single-trial spike-count vectors
from a random subset of simultaneously recorded units in one structure, and
tested on held-out trials (5-fold cross-validation). Accuracy is reported against
a label-shuffled null run through the identical pipeline.
""")
code(strip("decoding", COMMON_DROP, subs=[(r"\bO\.", "")]))
code("""
dec = run_decoding(sessions)
dec.to_csv("decoding_results.csv", index=False)
print(fig_decoding(dec))
""")
md("""
![decoding](fig06_decoding.png)

Forty simultaneously recorded V1 units are enough to identify which of eight drift
directions was shown on a held-out trial with better than 80% accuracy. The same
analysis in CA1 stays near chance.
""")

# ------------------------------------------------------------------- GLM
md(r"""
## 8. Encoding model: does orientation survive the running-speed confound?

Locomotion increases firing rates throughout mouse visual cortex, so a nuisance
explanation has to be ruled out: if running happened to co-vary with the stimulus
sequence, a rate difference between orientations could be behavioural rather than
visual. Four nested Poisson GLMs are fit to single-trial spike counts with NeMoS
and compared by 5-fold cross-validated held-out log-likelihood:

| model | predictors |
|---|---|
| M0 | intercept only |
| M1 | running speed (B-spline basis) |
| M2 | drift direction (cyclic B-spline basis over 0-360 deg) |
| M3 | drift direction + running speed |

The quantity of interest is `M3 - M1`: the held-out likelihood gained by adding
grating direction to a model that already knows how fast the animal was running.
All units of a session share the same design matrix, so they are fit together with
`nmo.glm.PopulationGLM`.
""")
code(strip("glm", COMMON_DROP,
           subs=[(r"(?m)^import jax\n", ""), (r"(?m)^jax\.config\.update\(.*\n", "")]))
code("""
glm_rows, glm_examples = [], []
for res in tqdm(sessions, desc="GLM"):
    df, dir_basis = fit_session(res, res["speed"])
    glm_rows.append(df)
    if not glm_examples:
        Y = np.rint(res["dg_rates"] * 2.0)
        X_dir, _, _ = build_design(res["dg_labels"], res["speed"])
        grid, rate = predicted_tuning(Y, X_dir, dir_basis)
        cand = df[df.area.isin(VISUAL_CORTEX)].sort_values(
            "d_dir_given_speed", ascending=False)
        for j in cand.index[:4]:
            col = int(np.where(res["units"].index.values == df.loc[j, "unit_id"])[0][0])
            sem = np.array([res["dg_rates"][res["dg_labels"] == d, col].std(ddof=1)
                            / np.sqrt((res["dg_labels"] == d).sum())
                            for d in DIRECTIONS])
            glm_examples.append(dict(
                unit_id=df.loc[j, "unit_id"], area=df.loc[j, "area"],
                emp=res["tc_dg"][:, col], sem=sem, grid=grid, fit=rate[:, col],
                d=df.loc[j, "d_dir_given_speed"]))
glm_df = pd.concat(glm_rows, ignore_index=True)
glm_df.to_csv("glm_results.csv", index=False)
print(fig_glm(glm_df, glm_examples))
""")
md("""
![GLM](fig07_glm.png)

Running speed does carry information about firing rate almost everywhere,
including hippocampus. Adding drift direction on top of it still improves
held-out likelihood for the large majority of visual cortical units and for
almost none of the hippocampal ones, so the orientation signal is not a
locomotion artefact.
""")

# ------------------------------------------------------------------- summary
md("""
## 9. Properties of the tuned population
""")
code("""
print(fig_summary(pooled))
units.to_csv("unit_metrics.csv", index=False)
""")
md("""
![summary](fig08_summary.png)
""")

md("""
## 10. Summary statistics
""")
code("""
vis = units[units.area.isin(VISUAL_CORTEX)]
ctl = units[units.area.isin(CONTROL)]
mw = stats.mannwhitneyu(vis.dg_osi.dropna(), ctl.dg_osi.dropna(), alternative="greater")
tuned = vis[vis.dg_p_perm < 0.05]
both = vis[(vis.dg_p_perm < 0.05) & (vis.sg_p_perm < 0.05)]
delta = circ_dist_ori(both.dg_pref_ori.values, both.sg_pref_ori.values)
rng = np.random.default_rng(0)
delta_shuf = circ_dist_ori(both.dg_pref_ori.values,
                           rng.permutation(both.sg_pref_ori.values))
d_card = np.minimum(circ_dist_ori(tuned.dg_pref_ori.values, 0.0),
                    circ_dist_ori(tuned.dg_pref_ori.values, 90.0))

counts = dec[dec.task == "orientation"].groupby("n_units").area.nunique()
nsel = int(counts[counts >= counts.max()].index.max())
dec_best = dec[(dec.task == "orientation") & (dec.n_units == nsel)]

summary = dict(
    n_sessions=len(sessions),
    n_units=int(len(units)),
    n_visual_cortex=int(len(vis)),
    frac_tuned_visual_cortex=float((vis.dg_p_perm < 0.05).mean()),
    frac_tuned_control=float((ctl.dg_p_perm < 0.05).mean()),
    median_osi_visual=float(vis.dg_osi.median()),
    median_osi_null_visual=float(vis.dg_osi_null_median.median()),
    median_osi_control=float(ctl.dg_osi.median()),
    mannwhitney_p=float(mw.pvalue),
    n_both_stimuli=int(len(both)),
    median_delta_pref_ori=float(np.median(delta)),
    median_delta_pref_ori_shuffled=float(np.median(delta_shuf)),
    frac_delta_under_30deg=float((delta < 30).mean()),
    frac_cardinal_preference=float((d_card < 22.5).mean()),
    frac_cardinal_chance=0.5,
    frac_cardinal_binomial_p=float(stats.binomtest(
        int((d_card < 22.5).sum()), len(d_card), 0.5, alternative="greater").pvalue),
    cardinal_bias_cos4=float(np.mean(np.cos(4 * np.deg2rad(tuned.dg_pref_ori.values)))),
    decoding_n_units=nsel,
    decoding_orientation=({a: float(g.acc.mean()) for a, g in
                           dec_best[dec_best.kind == "observed"].groupby("area")}),
    decoding_orientation_shuffled=({a: float(g.acc.mean()) for a, g in
                                    dec_best[dec_best.kind == "shuffled"].groupby("area")}),
    glm_frac_positive_dir_given_speed={
        a: float((glm_df.loc[glm_df.area == a, "d_dir_given_speed"] > 0).mean())
        for a in VISUAL_CORTEX + CONTROL if (glm_df.area == a).sum() >= 20},
)
json.dump(summary, open("summary_stats.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
""")

md("""
## 11. Conclusions

Orientation selectivity is present, strong and specific in this dataset. Units in
every visual cortical area carry orientation-tuned responses with a median OSI
several times the shuffled null, roughly four fifths of them individually
significant against a within-block permutation test. The tuning is bilobed: the
mean curve aligned to each unit's preferred direction shows a second peak at the
opposite direction, which is the defining property of orientation rather than
direction selectivity.

Three controls make the result hard to explain away. Simultaneously recorded
hippocampal units, analysed identically, sit at the shuffled null. Preferred
orientation estimated from drifting gratings predicts preferred orientation
estimated from static gratings, an independent stimulus class with no motion,
with a median discrepancy far below chance. And a Poisson GLM shows that grating
direction improves held-out likelihood after running speed is already in the
model, so the effect is not a by-product of locomotion.

Two observations deserve qualification. First, visual thalamus (LGd and LP) is
not at the null: a substantial minority of thalamic units reach significance,
with OSI values between cortex and hippocampus. This is consistent with the
current literature, in which mouse dLGN contains a genuine orientation-biased
subpopulation, and it is a reminder that "orientation selectivity is created in
cortex" is an approximation. Second, CA1 shows a small but above-chance fraction
of significant units and modestly above-chance orientation decoding. The GLM
suggests why: hippocampal firing is strongly modulated by running speed, and any
residual coupling between behavioural state and the stimulus sequence produces
weak apparent tuning. The effect is an order of magnitude smaller than the
cortical one and disappears in the direction-given-speed comparison.
""")

# ---------------------------------------------------------------------- write out
with open("orientation_selectivity.py", "w") as fh:
    fh.write("# ---\n# jupyter:\n#   jupytext:\n#     text_representation:\n"
             "#       extension: .py\n#       format_name: percent\n"
             "#       format_version: '1.3'\n#       jupytext_version: 1.16.7\n"
             "#   kernelspec:\n#     display_name: Python 3\n#     language: python\n"
             "#     name: python3\n# ---\n\n")
    for kind, body in cells:
        if kind == "markdown":
            fh.write("# %% [markdown]\n")
            for ln in body.splitlines():
                fh.write(("# " + ln).rstrip() + "\n")
            fh.write("\n")
        else:
            fh.write("# %%\n" + body.strip() + "\n\n")
print("wrote orientation_selectivity.py", sum(1 for _ in open("orientation_selectivity.py")), "lines")
