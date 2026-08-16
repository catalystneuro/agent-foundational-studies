"""Run the place-cell pipeline over all eight sessions of DANDI:000044."""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pf_lib as pf

import os

urls = pf.session_urls()
if os.path.exists("multisession.pkl"):
    results = pickle.load(open("multisession.pkl", "rb"))
    print(f"loaded cached results for {len(results)} sessions")
else:
    results = {}
    for name, url in tqdm(list(urls.items()), desc="sessions"):
        results[name] = pf.analyze_session(name, url, n_shuffle=pf.N_SHUFFLE, seed=0)
    with open("multisession.pkl", "wb") as fh:
        pickle.dump(results, fh)

# The reliability criterion puts a floor on the within-direction split-half
# correlation, so the directionality comparison uses a mask without it.
for r in results.values():
    for d in r["directions_analysed"]:
        m = r["per_direction"][d]["metrics"]
        if "is_place_cell_norel" not in m:
            m["is_place_cell_norel"] = (
                (m.si > m.si_thresh) & (m.peak_rate >= pf.MIN_PEAK_RATE)
                & (m.n_spikes >= pf.MIN_SPIKES)
            )

# ---------------------------------------------------------------- aggregate
rows = []
pooled = []
for name, r in results.items():
    n_pc = int(r["is_place_cell_either"].sum())
    rows.append(
        {
            "session": name,
            "subject": r["subject"],
            "maze": r["maze_type"],
            "track_m": r["track_length"],
            "n_units": r["n_units"],
            "n_pyr": r["n_pyramidal"],
            "n_pc": n_pc,
            "frac_pc": n_pc / max(r["n_pyramidal"], 1),
            "n_trav": sum(r["n_traversals"][d] for d in pf.DIRS),
            "run_s": r["run_duration"]["run"],
            "dirs": ",".join(r["directions_analysed"]),
        }
    )
    for d in r["directions_analysed"]:
        m = r["per_direction"][d]["metrics"].copy()
        m["session"] = name
        m["subject"] = r["subject"]
        m["maze"] = r["maze_type"]
        m["direction"] = d
        m["track_m"] = r["track_length"]
        pooled.append(m)

summary = pd.DataFrame(rows)
pooled = pd.concat(pooled, ignore_index=True)
pooled["dir_corr"] = np.nan
for name, r in results.items():
    for d in r["directions_analysed"]:
        sel = (pooled.session == name) & (pooled.direction == d)
        pooled.loc[sel, "dir_corr"] = r["dir_corr"]

summary.to_csv("session_summary.csv", index=False)
pooled.to_csv("pooled_units.csv", index=False)
print(summary.to_string(index=False))
pc = pooled[pooled.is_place_cell]
print(f"\npooled: {len(pooled)} unit-direction pairs, {len(pc)} with a significant place field")
print(f"median SI          {pc.si.median():.2f} bits/spike")
print(f"median field width {100 * pc.width.median():.0f} cm")
print(f"median peak rate   {pc.peak_rate.median():.1f} Hz")
print(f"median sparsity    {pc.sparsity.median():.2f}")
print(f"median reliability {pc.reliability.median():.2f}")
print(f"place-cell fraction per session: {summary.frac_pc.min():.2f}-{summary.frac_pc.max():.2f} "
      f"(mean {summary.frac_pc.mean():.2f})")

# ------------------------------------------------------------------ figure
SUBJ_COL = {s: c for s, c in zip(sorted(summary.subject.unique()),
                                 ["#4c72b0", "#dd8452", "#55a868", "#c44e52"])}
fig, axs = plt.subplots(2, 4, figsize=(19, 9))
axs = axs.ravel()

ax = axs[0]
o = np.argsort(summary.frac_pc.values)
y = np.arange(len(summary))
ax.barh(y, 100 * summary.frac_pc.values[o],
        color=[SUBJ_COL[s] for s in summary.subject.values[o]])
ax.set_yticks(y)
ax.set_yticklabels([f"{s}\n({m}, {n} cells)" for s, m, n in
                    zip(summary.session.values[o], summary.maze.values[o],
                        summary.n_pyr.values[o])], fontsize=7)
ax.set_xlabel("pyramidal cells with a place field (%)")
ax.axvline(100 * summary.frac_pc.mean(), color="k", ls="--", lw=1)
ax.set_title("(a) Place cells are found in every session", fontsize=11)

ax = axs[1]
for s, g in pc.groupby("session"):
    ax.hist(g.si, np.linspace(0, 4, 33), histtype="step", lw=1.2, density=True,
            color=SUBJ_COL[g.subject.iloc[0]], alpha=0.8)
ax.hist(pc.si, np.linspace(0, 4, 33), histtype="step", lw=2.5, color="k", density=True,
        label=f"pooled (n={len(pc)})")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("probability density")
ax.set_title(f"(b) Spatial information, median {pc.si.median():.2f} bits/spike", fontsize=11)
ax.legend(fontsize=8)

ax = axs[2]
data = [100 * g.width.dropna().values for _, g in pc.groupby("session")]
labels = [s.split("_")[0][:3] + s.split("_")[1][:4] for s in pc.session.unique()]
bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
for patch, s in zip(bp["boxes"], [g.subject.iloc[0] for _, g in pc.groupby("session")]):
    patch.set_facecolor(SUBJ_COL[s])
ax.tick_params(axis="x", rotation=60, labelsize=7)
ax.set_ylabel("place-field width (cm)")
ax.set_title(f"(c) Field width, pooled median {100 * pc.width.median():.0f} cm", fontsize=11)

ax = axs[3]
ax.scatter(pc.peak_rate, 100 * pc.width, s=10,
           c=[SUBJ_COL[s] for s in pc.subject], alpha=0.6, edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel("in-field peak rate (Hz)")
ax.set_ylabel("field width (cm)")
ax.set_title("(d) Peak rate vs field width", fontsize=11)
handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=s) for s, c in SUBJ_COL.items()]
ax.legend(handles=handles, fontsize=8, title="rat", title_fontsize=8)

ax = axs[4]
ax.hist(pc.sparsity.dropna(), np.linspace(0, 1, 33), color="#4c72b0", edgecolor="w")
ax.axvline(pc.sparsity.median(), color="k", ls="--")
ax.set_xlabel(r"sparsity $\langle\lambda\rangle^2/\langle\lambda^2\rangle$")
ax.set_ylabel("count")
ax.set_title(f"(e) Sparsity, median {pc.sparsity.median():.2f}", fontsize=11)

ax = axs[5]
ax.hist(pc.reliability.dropna(), np.linspace(-0.2, 1, 33), color="#55a868", edgecolor="w")
ax.axvline(pc.reliability.median(), color="k", ls="--")
ax.set_xlabel("odd vs even traversal map correlation\n(>=0.4 by selection)")
ax.set_ylabel("count")
ax.set_title(f"(f) Reliability, median r = {pc.reliability.median():.2f}", fontsize=11)

ax = axs[6]
for s, g in pc.groupby("session"):
    ax.hist(g.peak_pos / g.track_m.iloc[0], np.linspace(0, 1, 17), histtype="step",
            lw=1.2, density=True, color=SUBJ_COL[g.subject.iloc[0]], alpha=0.8)
ax.hist(pc.peak_pos / pc.track_m, np.linspace(0, 1, 17), histtype="step", lw=2.5,
        color="k", density=True)
ax.set_xlabel("field peak, fraction of track length")
ax.set_ylabel("probability density")
ax.set_title("(g) Fields cover the track in every session", fontsize=11)

ax = axs[7]
bi_sessions = [s for s in pooled.session.unique()
               if len(results[s]["directions_analysed"]) == 2]
sub = pooled[pooled.session.isin(bi_sessions) & pooled.is_place_cell_norel]
d1 = sub.dir_corr.dropna()
d2 = sub.reliability.dropna()
e = np.linspace(-1, 1, 33)
ax.hist(d1, e, color="0.75", density=True,
        label=f"opposite directions ({d1.median():.2f})")
ax.hist(d2, e, histtype="step", color="k", lw=1.8, density=True,
        label=f"same direction, odd vs even ({d2.median():.2f})")
from scipy import stats as _st
u_dir = _st.mannwhitneyu(d2, d1, alternative="greater")
ax.set_xlabel("rate-map correlation")
ax.set_ylabel("probability density")
ax.set_title(f"(h) Direction specificity, {len(bi_sessions)} bidirectional sessions\n"
             f"(n = {len(d1)} fields, p = {u_dir.pvalue:.1e})", fontsize=11)
ax.legend(fontsize=7.5, loc="upper left")
print(f"\ndirectionality (no reliability selection, n={len(d1)}): "
      f"cross-direction r = {d1.median():.2f}, within-direction r = {d2.median():.2f}, "
      f"Mann-Whitney p = {u_dir.pvalue:.2e}")

fig.suptitle(
    "DANDI:000044, all 8 sessions / 4 rats: place fields replicate across animals and mazes",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig06_multisession.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig06 done")

# per-session population maps
n = len(results)
fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
for ax, (name, r) in zip(axs.ravel(), results.items()):
    # show whichever direction the animal ran most
    d = max(r["directions_analysed"], key=lambda k: r["n_traversals"][k])
    pd_ = r["per_direction"][d]
    m = pd_["metrics"]
    sel = np.where(m.is_place_cell.values)[0]
    o = sel[np.argsort(m.peak_pos.values[sel])]
    M = pd_["tc_smooth"][:, o].T
    mx = M.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    im = ax.imshow(M / mx, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                   extent=[0, r["track_length"], 0, len(o)])
    ax.set_title(f"{name}\n{r['maze_type']}, {d}ward, {len(o)} place cells", fontsize=9)
    ax.set_xlabel("position (m)", fontsize=8)
    ax.set_ylabel("place cell", fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle("Sorted place-field maps, one direction per session", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig07_multisession_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig07 done")
