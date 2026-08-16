"""Repeat the core analysis across six sessions from six different mice.

Tuning curves are computed separately per arena. That matters: the head-direction
reference frame is anchored to the environment, so between the square and the
triangular arena the whole ensemble can rotate by a session-specific angle.
Pooling the two arenas smears the tuning curves and, in the worst session here,
hides almost every head-direction cell.
"""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from tqdm.auto import tqdm

import hd_lib

SESSIONS = [
    ("A3701", "e8a3ee2b-940a-4abe-a020-3821fc7e4f65"),
    ("A3702", "6fe925dd-ea41-4a84-9cee-ca6bb039edcb"),
    ("A3703", "a640fd89-41fa-4ba1-81e7-3bab5e84da63"),
    ("A3705", "c72bd94f-b744-483e-a782-3a3c475a5276"),
    ("A3706", "f7d0f925-04d3-47d7-8be7-0d14bb52672f"),
    ("A5505", "f1553d12-5d2e-4908-9a90-3989bb7ee0c1"),
]
NB_BINS = 120
N_SHUFFLES = 100
MVL_THRESHOLD = 0.3
BIN = 0.2


def alternating_blocks(ep, block=60.0):
    """Split an IntervalSet into two interleaved sets of `block`-second pieces."""
    starts, ends = [], []
    for st, en in zip(ep.start, ep.end):
        edges = np.arange(st, en, block)
        starts.extend(edges)
        ends.extend(np.minimum(edges + block, en))
    starts, ends = np.array(starts), np.array(ends)
    return nap.IntervalSet(starts[::2], ends[::2]), nap.IntervalSet(starts[1::2], ends[1::2])


def decode_error(units, hd, train_ep, test_ep):
    """Median absolute error (deg) of a Bayesian decoder, and the fraction < 30 deg."""
    tc_train = hd_lib.compute_tuning_xr(units, hd, train_ep, NB_BINS // 2)
    dec, _ = nap.decode_bayes(tc_train, units, test_ep, bin_size=BIN)
    true_hd = hd.restrict(test_ep).bin_average(BIN, test_ep)
    common = np.intersect1d(dec.t, true_hd.t)
    d = dec.values[np.isin(dec.t, common)]
    tr = true_hd.values[np.isin(true_hd.t, common)]
    ok = ~np.isnan(tr)
    err = np.degrees(hd_lib.circ_diff(d[ok], tr[ok]))
    return err


REPLOT = bool(os.environ.get("REPLOT")) and os.path.exists("multi_session_results.pkl")

all_stats, summary, rotations = [], [], {}
for subj, asset in (tqdm(SESSIONS, desc="sessions", ncols=70) if not REPLOT else []):
    s = hd_lib.session_bundle(asset)
    units, hd = s["units"], s["hd"]
    labels = s["epoch_labels"]
    wake_names = [l for l in labels if str(l).startswith("wake")]
    # Reference arena = first wake epoch; second arena = last wake epoch.
    ref = nap.IntervalSet(s["wake"].start[0], s["wake"].end[0])
    alt = nap.IntervalSet(s["wake"].start[-1], s["wake"].end[-1])

    # Tuning and significance in the reference arena only.
    tc_ref, stats = hd_lib.tuning_stats(units, hd, ref, NB_BINS)
    mvl_null, info_null, order = hd_lib.shuffle_null(
        units, hd, ref, n_shuffles=N_SHUFFLES, nb_bins=NB_BINS, seed=2)
    stats["mvl_p99"] = np.percentile(mvl_null, 99, axis=0)
    stats["info_p99"] = np.percentile(info_null, 99, axis=0)
    stats["is_hd"] = ((stats["mvl"] > stats["mvl_p99"]) &
                      (stats["hd_info"] > stats["info_p99"]) &
                      (stats["mvl"] >= MVL_THRESHOLD))
    stats["is_hd_dataset"] = np.asarray(units.is_head_direction).astype(bool)
    stats["subject"], stats["session"] = subj, s["name"]

    # Same measures in the second arena, and the rotation between the two.
    _, stats_alt = hd_lib.tuning_stats(units, hd, alt, NB_BINS)
    stats["mvl_alt"] = stats_alt["mvl"]
    stats["pref_alt"] = stats_alt["pref_dir"]
    stats["rotation"] = hd_lib.circ_diff(stats_alt["pref_dir"], stats["pref_dir"])
    tuned_both = stats["is_hd"] & (stats["mvl_alt"] >= MVL_THRESHOLD)
    rot = np.degrees(stats.loc[tuned_both, "rotation"].values)
    rotations[subj] = rot
    rot_med = float(np.median(rot)) if len(rot) else np.nan
    rot_iqr = float(np.subtract(*np.percentile(rot, [75, 25]))) if len(rot) else np.nan

    # Cross-validated tuning within the reference arena (interleaved 60 s blocks).
    ep_a, ep_b = alternating_blocks(ref)
    tc_a = hd_lib.compute_tuning(units, hd, ep_a, NB_BINS)
    tc_b = hd_lib.compute_tuning(units, hd, ep_b, NB_BINS)
    stats["split_half_r"] = [np.corrcoef(tc_a[u].values, tc_b[u].values)[0, 1]
                             for u in tc_a.columns]

    hd_units = units[list(stats.index[stats["is_hd"]])]
    err_within = decode_error(hd_units, hd, ep_a, ep_b)      # held out, same arena
    err_across = decode_error(hd_units, hd, ref, alt)        # different arena
    # A decoder built from reference-arena tuning curves reports directions in
    # the reference frame, so if the ensemble rotated by +rot the decoded value
    # is shifted by -rot. Adding the single session-wide rotation back should
    # therefore recover the animal's true heading.
    err_across_corr = np.degrees(hd_lib.circ_diff(np.radians(err_across),
                                                  np.radians(-rot_med)))

    all_stats.append(stats)
    summary.append({
        "subject": subj, "session": s["name"],
        "regions": ",".join(sorted(set(np.asarray(units.location)))),
        "epochs": ",".join(map(str, wake_names)),
        "ref_min": ref.tot_length() / 60, "alt_min": alt.tot_length() / 60,
        "n_units": len(units), "n_hd": int(stats["is_hd"].sum()),
        "frac_hd": float(stats["is_hd"].mean()),
        "n_hd_dataset": int(stats["is_hd_dataset"].sum()),
        "agreement": float((stats["is_hd"] == stats["is_hd_dataset"]).mean()),
        "median_mvl_hd": float(stats.loc[stats["is_hd"], "mvl"].median()),
        "median_split_half_r": float(stats.loc[stats["is_hd"], "split_half_r"].median()),
        "decode_err_within_deg": float(np.median(np.abs(err_within))),
        "decode_frac30_within": float(np.mean(np.abs(err_within) < 30)),
        "decode_err_across_deg": float(np.median(np.abs(err_across))),
        "decode_signed_err_across_deg": float(np.median(err_across)),
        "rotation_med_deg": rot_med, "rotation_iqr_deg": rot_iqr,
        "decode_err_across_derotated_deg": float(np.median(np.abs(err_across_corr))),
    })
    s["io"].close()

if REPLOT:
    with open("multi_session_results.pkl", "rb") as f:
        cached = pickle.load(f)
    pooled, summary, rotations = cached["pooled"], cached["summary"], cached["rotations"]
else:
    pooled = pd.concat(all_stats, ignore_index=True)
    summary = pd.DataFrame(summary)
pd.set_option("display.width", 250)
print(summary.drop(columns=["session", "regions", "epochs"]).to_string(index=False))
print("\npooled: %d units, %d HD cells (%.0f%%) across %d mice"
      % (len(pooled), pooled["is_hd"].sum(), 100 * pooled["is_hd"].mean(),
         summary["subject"].nunique()))
if not REPLOT:
    pooled.to_csv("multi_session_units.csv", index=False)
    summary.to_csv("multi_session_summary.csv", index=False)

# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
subs = summary["subject"].tolist()

ax = axes[0, 0]
ax.bar(subs, summary["frac_hd"] * 100, color="#4C72B0", label="this analysis")
ax.plot(subs, summary["n_hd_dataset"] / summary["n_units"] * 100, "o", color="k",
        label="dataset's own flag")
ax.set_ylabel("HD cells (% of units)")
ax.set_title("Proportion of HD cells per session")
ax.legend(fontsize=8)
for i, n in enumerate(summary["n_units"]):
    ax.text(i, 2, f"n={n}", ha="center", fontsize=8, color="w")

ax = axes[0, 1]
for subj, grp in pooled.groupby("subject"):
    x = np.sort(grp["mvl"].values)
    ax.plot(x, np.linspace(0, 1, len(x)), lw=1.2, label=subj)
ax.axvline(MVL_THRESHOLD, color="k", ls="--", lw=1)
ax.set_xlabel("mean vector length (reference arena)")
ax.set_ylabel("cumulative fraction of units")
ax.set_title("Directional tuning is consistent across mice")
ax.legend(fontsize=7, ncol=2)

ax = axes[0, 2]
hdp = pooled[pooled["is_hd"]]
ax.hist(hdp["hd_info"], bins=30, color="#4C72B0", alpha=0.85, label="HD cells")
ax.hist(pooled.loc[~pooled["is_hd"], "hd_info"], bins=30, color="0.6", alpha=0.85,
        label="other units")
ax.set_xlabel("directional information (bits/spike)")
ax.set_ylabel("units")
ax.set_title("Pooled across %d sessions (%d units)" % (len(summary), len(pooled)))
ax.legend(fontsize=8)

axes[1, 0].remove()
ax = fig.add_subplot(2, 3, 4, projection="polar")
edges = np.linspace(0, 2 * np.pi, 25)
cnt, _ = np.histogram(hdp["pref_dir"], edges)
ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge", color="#4C72B0",
       edgecolor="w")
ax.set_title("Pooled preferred directions\n(%d HD cells)" % len(hdp), pad=28)

ax = axes[1, 1]
w = 0.27
x = np.arange(len(subs))
ax.bar(x - w, summary["decode_err_within_deg"], w, color="#55A868",
       label="held out, same arena")
ax.bar(x, summary["decode_err_across_deg"], w, color="#C44E52",
       label="tested in the other arena")
ax.bar(x + w, summary["decode_err_across_derotated_deg"], w, color="#DD8452",
       label="other arena, session rotation removed")
ax.set_xticks(x, subs)
ax.set_ylabel("median absolute decoding error (deg)")
ax.axhline(90, color="k", ls=":", lw=1)
ax.text(len(subs) - 0.9, 93, "chance", fontsize=8)
ax.set_title("Decoding head direction from spikes")
ax.legend(fontsize=8)

ax = axes[1, 2]
for i, subj in enumerate(subs):
    r = np.asarray(rotations[subj])
    med0 = np.median(r)
    r = med0 + np.degrees(hd_lib.circ_diff(np.radians(r), np.radians(med0)))
    ax.scatter(np.full(len(r), i) + np.random.default_rng(i).normal(0, 0.06, len(r)),
               r, s=8, alpha=0.6, color="#4C72B0")
    ax.plot([i - 0.3, i + 0.3], [np.median(r)] * 2, color="k", lw=2)
ax.set_xticks(range(len(subs)), subs)
ax.axhline(0, color="k", ls=":", lw=1)
ax.set_ylabel("preferred-direction rotation between arenas (deg)")
ax.set_title("The ensemble rotates as a rigid ring\n"
             "(each dot is one HD cell; bar = session median)")
ax.set_ylim(-200, 220)

fig.suptitle("Head-direction coding across %d sessions from %d mice "
             "(DANDI:000939, postsubiculum)"
             % (len(summary), summary["subject"].nunique()), y=1.0)
fig.tight_layout()
fig.savefig("fig07_multi_session.png", dpi=150, bbox_inches="tight")
print("saved fig07_multi_session.png")

with open("multi_session_results.pkl", "wb") as f:
    pickle.dump({"pooled": pooled, "summary": summary, "rotations": rotations}, f)
