"""Run the entrainment / precession pipeline over four sessions and pool the results."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import theta_lib as tl

all_fields, all_lock, summary = [], [], []
for path in tqdm(tl.SESSIONS, desc="sessions"):
    res = tl.analyze_session(path)
    all_fields.extend(res["fields"])
    for uid, L in res["locking"].items():
        all_lock.append(dict(session=res["session"], unit=uid, **L))
    sl = np.array([f["slope"] for f in res["fields"]])
    pv = np.array([f["p"] for f in res["fields"]])
    pyr_lock = [L for L in res["locking"].values() if L["cell_type"] == "excitatory" and L["n"] >= 50]
    summary.append(
        dict(
            session=res["session"],
            subject=path.split("/")[0].replace("sub-", ""),
            n_laps=len(res["laps"]),
            run_time_s=round(res["run_ep"].tot_length(), 1),
            n_pyr=len(res["pyr"]),
            n_place_fields=int(sum(res["maps"][d]["is_place"].sum() for d in res["maps"])),
            n_analysed=len(res["fields"]),
            pct_precessing=round(100 * float(np.mean((pv < 0.05) & (sl < 0))), 1),
            median_slope_deg=round(float(np.degrees(np.median(sl))), 1),
            pct_theta_locked=round(100 * float(np.mean([L["p"] < 0.01 for L in pyr_lock])), 1),
            lfp_channel=res["best_ch"],
        )
    )
    res["h5"].close()

df = pd.DataFrame(summary)
print(df.to_string(index=False))
df.to_csv("session_summary.csv", index=False)

lock_df = pd.DataFrame(all_lock)
field_df = pd.DataFrame(
    [{k: v for k, v in f.items() if k not in ("x", "phase")} for f in all_fields]
)
field_df.to_csv("place_field_precession.csv", index=False)

slopes = np.array([f["slope"] for f in all_fields])
rhos = np.array([f["rho"] for f in all_fields])
pvals = np.array([f["p"] for f in all_fields])
sig = (pvals < 0.05) & (slopes < 0)
print(
    "\nPooled: %d fields from %d sessions, %.0f%% significantly precessing, "
    "median slope %.0f deg/field, median rho %.2f"
    % (len(all_fields), len(tl.SESSIONS), 100 * sig.mean(), np.degrees(np.median(slopes)), np.median(rhos))
)

# ---------------------------------------------------------------------------------
# Figure 8: cross-session summary
# ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)
colors = plt.cm.tab10(np.arange(len(tl.SESSIONS)))
sessions = df["session"].tolist()

# (a) pooled phase-position density
ax = fig.add_subplot(gs[0, 0])
X = np.concatenate([f["x"] for f, s in zip(all_fields, sig) if s])
P = np.degrees(np.concatenate([f["phase"] for f, s in zip(all_fields, sig) if s]))
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360], bins=[20, 40], range=[[0, 1], [0, 720]])
H = H / H.sum(axis=1, keepdims=True)
im = ax.pcolormesh(xe, ye, H.T, cmap="magma")
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Theta phase (deg)")
ax.set_yticks([0, 180, 360, 540, 720])
ax.set_title("All sessions pooled\n(%d fields, %d spikes)" % (sig.sum(), len(X)))
plt.colorbar(im, ax=ax, label="P(phase | position)")

# (b) slope distributions per session
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(-720, 360, 31)
for i, s in enumerate(sessions):
    m = np.array([f["session"] == s for f in all_fields])
    ax.hist(np.degrees(slopes[m]), bins=bins, histtype="step", lw=1.8, color=colors[i], label=s)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Slope (deg per field traversal)")
ax.set_ylabel("Number of fields")
ax.set_title("Precession slope, per session")
ax.legend(fontsize=7)

# (c) fraction precessing per session
ax = fig.add_subplot(gs[0, 2])
ax.bar(range(len(df)), df["pct_precessing"], color=colors[: len(df)])
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["session"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% of fields with significant\nnegative phase-position slope")
ax.axhline(5, ls="--", color="k", lw=1)
ax.set_title("Consistency across sessions\n(dashed line = chance, 5%)")
for i, v in enumerate(df["pct_precessing"]):
    ax.text(i, v + 1.5, "%.0f" % v, ha="center", fontsize=9)

# (d) preferred theta phase of pyramidal cells, per session
ax = fig.add_subplot(gs[1, 0], projection="polar")
for i, s in enumerate(sessions):
    m = (lock_df.session == s) & (lock_df.cell_type == "excitatory") & (lock_df.p < 0.01)
    ax.scatter(lock_df.mu[m], lock_df.mrl[m], s=16, color=colors[i], alpha=0.7, label=s)
ax.set_title("Preferred theta phase of\nsignificantly locked pyramidal cells", pad=25)
ax.set_rlabel_position(120)
ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.15))

# (e) locking strength by cell type, pooled
ax = fig.add_subplot(gs[1, 1])
enough = lock_df.n >= 50
for ct, color in (("excitatory", "tab:blue"), ("inhibitory", "tab:purple")):
    v = lock_df.mrl[enough & (lock_df.cell_type == ct)]
    ax.hist(v, bins=25, alpha=0.65, color=color, label="%s (n=%d)" % (ct, len(v)))
ax.set_xlabel("Mean resultant length")
ax.set_ylabel("Number of cells")
ax.set_title("Theta phase locking, all sessions")
ax.legend(fontsize=9)

# (f) precession strength vs locking strength of the same cell
ax = fig.add_subplot(gs[1, 2])
mrl = np.array([f["mrl"] for f in all_fields])
ax.scatter(mrl[~sig], rhos[~sig], s=18, color="0.7", label="n.s.")
ax.scatter(mrl[sig], rhos[sig], s=18, color="tab:red", label="precessing")
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Theta phase locking (MRL)")
ax.set_ylabel("Phase-position correlation $\\rho$")
ax.set_title("Entrainment vs. precession\n(one point per place field)")
ax.legend(fontsize=9)

fig.suptitle(
    "Theta entrainment and phase precession across %d sessions of DANDI:000044 (%d CA1 place fields)"
    % (len(tl.SESSIONS), len(all_fields)),
    fontsize=13,
)
fig.savefig("fig08_multisession_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done")
