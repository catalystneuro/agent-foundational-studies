"""Template-free check: explained variance of run-time correlations in post-task sleep."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

from dandi_io import open_session
from swr_lib import linearize_position, running_speed

SESSION = "Achilles_10252013"
CORR_BIN = 0.100

nwbfile, nwb, h5 = open_session(SESSION)
units = nwb["units"]
pyr = units[units.cell_type == "excitatory"]
pf = np.load("place_fields_Achilles_10252013.npz")
cells = pf["place_units"]
group = pyr[list(cells)]

ep_df = nwbfile.epochs.to_dataframe()
ep = {lab.replace("Epoch", ""): nap.IntervalSet(
    start=ep_df.query("label==@lab").start_time.values,
    end=ep_df.query("label==@lab").stop_time.values) for lab in ep_df.label}

st = nwbfile.processing["behavior"]["states"].to_dataframe()
nrem = nap.IntervalSet(start=st.query("label=='Non-REM'").start_time.values,
                       end=st.query("label=='Non-REM'").stop_time.values)

pos, TRACK = linearize_position(nwb)
speed = running_speed(pos)
run_ep = speed.threshold(0.05).time_support.merge_close_intervals(0.5)

windows = {"PRE sleep": nrem.intersect(ep["PRE"]),
           "RUN": run_ep,
           "POST sleep": nrem.intersect(ep["POST"])}


def corr_matrix(epoch):
    C = group.count(CORR_BIN, ep=epoch).values.astype(float)
    C = (C - C.mean(0)) / np.where(C.std(0) > 0, C.std(0), 1)
    R = (C.T @ C) / C.shape[0]
    return R


R = {k: corr_matrix(v) for k, v in windows.items()}
iu = np.triu_indices(len(cells), 1)
v = {k: m[iu] for k, m in R.items()}
for k, e in windows.items():
    print(f"{k}: {e.tot_length()/60:.0f} min, "
          f"{int(e.tot_length()/CORR_BIN)} bins")


def pcorr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    rxy, rxz, ryz = (np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1],
                     np.corrcoef(y, z)[0, 1])
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


ev = pcorr(v["RUN"], v["POST sleep"], v["PRE sleep"]) ** 2
rev = pcorr(v["RUN"], v["PRE sleep"], v["POST sleep"]) ** 2
print(f"explained variance EV = {ev*100:.1f}%, reverse EV = {rev*100:.1f}%")
print("pairwise r:  RUN-POST %.3f  RUN-PRE %.3f  PRE-POST %.3f" % (
    np.corrcoef(v["RUN"], v["POST sleep"])[0, 1],
    np.corrcoef(v["RUN"], v["PRE sleep"])[0, 1],
    np.corrcoef(v["PRE sleep"], v["POST sleep"])[0, 1]))

fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)
order = np.argsort([np.argmax(pf["tc_right"][:, list(pf["units"]).index(u)])
                    for u in cells])
for j, k in enumerate(windows):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(R[k][np.ix_(order, order)], cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    ax.set_title(k, fontsize=10)
    ax.set_xlabel("cell (by field position)")
    if j == 0:
        ax.set_ylabel("cell (by field position)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="pairwise correlation" if j == 2 else None)

ax = fig.add_subplot(gs[1, :2])
ax.scatter(v["RUN"], v["PRE sleep"], s=3, alpha=0.3, color="#8ecae6", label="PRE")
ax.scatter(v["RUN"], v["POST sleep"], s=3, alpha=0.3, color="#023047", label="POST")
ax.set_xlabel("pairwise correlation during RUN")
ax.set_ylabel("pairwise correlation in sleep")
ax.legend(fontsize=8, markerscale=3)
ax.set_title("reinstatement of run-time\ncorrelation structure", fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.bar(["EV", "reverse\nEV"], [ev * 100, rev * 100], color=["#023047", "#adb5bd"])
ax.set_ylabel("explained variance (%)")
ax.set_title("EV / REV", fontsize=10)

fig.suptitle(f"{SESSION}: sleep reactivation of run-time correlations "
             f"({len(cells)} place cells, {int(CORR_BIN*1000)} ms bins)", y=0.96)
fig.savefig("fig06_reactivation.png", dpi=150, bbox_inches="tight")
print("wrote fig06_reactivation.png")
