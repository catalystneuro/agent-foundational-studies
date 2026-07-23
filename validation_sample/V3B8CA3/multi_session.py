"""Figure 9: the same pipeline applied to every linear-track session of DANDI:000044."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

import pf_core as pf
from dandi_io import SESSIONS

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160})

rng = np.random.default_rng(7)
rows, summaries = [], []

# Three of the eight sessions were recorded on a closed circular maze, where a
# lap wraps around and the two-direction traversal logic used here does not
# apply. They are skipped and reported rather than silently dropped.
linear_sessions, skipped = [], []
for name in SESSIONS:
    _, _, _, _, m, handles = pf.load_behavior_and_spikes(name)
    handles[1].close()
    (linear_sessions if m["maze_type"] == "linear" else skipped).append((name, m["maze"]))
print("linear-track sessions:", [n for n, _ in linear_sessions])
print("skipped (circular maze):", [n for n, _ in skipped])

for name, _ in tqdm(linear_sessions, desc="sessions"):
    res = pf.analyze_session(name, n_shuffles=500, progress=False)
    meta = res["meta"]
    tab = pd.concat([res["per_dir"][d]["table"] for d in pf.DIRECTIONS], ignore_index=True)
    rows.append(tab)

    err = pf.decoding_error(res)
    err_shuf = pf.decoding_error(res, shuffle_tc=True, rng=rng)
    any_pc = tab.groupby("unit").is_place_cell.any()
    summaries.append(dict(
        session=name, subject=meta["subject"], maze=meta["maze"],
        track_length_cm=meta["track_length_cm"], n_units=meta["n_units_total"],
        n_pyramidal=meta["n_pyramidal"], n_runs=meta["n_runs"],
        run_time_s=meta["run_time_s"], median_speed_cms=meta["median_speed_cms"],
        n_place_cells=int(any_pc.sum()), frac_place_cells=float(any_pc.mean()),
        median_info=float(tab.loc[tab.is_place_cell, "info_bits_per_spike"].median()),
        median_width_cm=float(tab.loc[tab.is_place_cell, "field_width_cm"].median()),
        median_peak_hz=float(tab.loc[tab.is_place_cell, "peak_rate"].median()),
        median_decode_err_cm=float(np.median(err)),
        median_decode_err_shuffled_cm=float(np.median(err_shuf)),
    ))

all_cells = pd.concat(rows, ignore_index=True)
summary = pd.DataFrame(summaries)
all_cells.to_csv("place_cell_table_all_sessions.csv", index=False)
summary.to_csv("session_summary.csv", index=False)
print(summary.to_string(index=False))

labels = [s.replace("_", "\n") for s in summary.session]
x = np.arange(len(summary))
colors = plt.get_cmap("tab10")(np.arange(len(summary)) % 10)

fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
fig.subplots_adjust(wspace=0.32, top=0.76, bottom=0.28, left=0.05, right=0.99)

ax = axes[0]
ax.bar(x, 100 * summary.frac_place_cells, color=colors)
for xi, (f, n) in enumerate(zip(summary.frac_place_cells, summary.n_place_cells)):
    ax.text(xi, 100 * f + 1.5, str(n), ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("place cells (% of pyramidal cells)")
ax.set_ylim(0, 100)
ax.set_title("A  Place-cell yield", loc="left", fontweight="bold")

ax = axes[1]
data = [all_cells.loc[(all_cells.session == s) & all_cells.is_place_cell,
                      "info_bits_per_spike"].values for s in summary.session]
bp = ax.boxplot(data, positions=x, widths=0.65, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for med in bp["medians"]:
    med.set_color("k")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("spatial information (bits/spike)")
ax.set_title("B  Spatial information", loc="left", fontweight="bold")

ax = axes[2]
data = [all_cells.loc[(all_cells.session == s) & all_cells.is_place_cell,
                      "field_width_cm"].values for s in summary.session]
bp = ax.boxplot(data, positions=x, widths=0.65, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for med in bp["medians"]:
    med.set_color("k")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("field width at 50 % of peak (cm)")
ax.set_title("C  Field width", loc="left", fontweight="bold")

ax = axes[3]
ax.bar(x - 0.19, summary.median_decode_err_cm, width=0.38, color="#1f77b4", label="observed")
ax.bar(x + 0.19, summary.median_decode_err_shuffled_cm, width=0.38, color="0.7",
       label="shuffled tuning curves")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("median decoding error (cm)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("D  Cross-validated decoding", loc="left", fontweight="bold")

fig.suptitle(f"All {len(summary)} linear-track sessions of DANDI:000044 "
             f"({summary.subject.nunique()} rats): the place code is present in every session\n"
             f"pooled: {int(summary.n_place_cells.sum())} place cells of "
             f"{int(summary.n_pyramidal.sum())} pyramidal cells "
             f"({100 * summary.n_place_cells.sum() / summary.n_pyramidal.sum():.0f} %), "
             f"median decoding error {summary.median_decode_err_cm.median():.1f} cm",
             fontweight="bold", fontsize=10)
fig.savefig("fig09_all_sessions.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig09_all_sessions.png")
