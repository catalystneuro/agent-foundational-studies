"""QC the linearization and lap detection for every linear-track session."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import place_cell_lib as pcl

n = len(pcl.SESSIONS)
fig, axs = plt.subplots(n, 3, figsize=(17, 3.1 * n))
for i, (aid, subj, label, maze) in enumerate(pcl.SESSIONS):
    sess = pcl.load_session(aid)
    laps = pcl.lap_epochs(sess)
    mov = pcl.moving_epochs(sess, laps)
    xy = sess["xy"]
    info = sess["lin_info"]

    ax = axs[i, 0]
    ax.plot(xy.values[:, 0], xy.values[:, 1], ".", ms=0.6, alpha=0.15, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_title(f"{label} ({maze})\naccepted 2D samples", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axs[i, 1]
    p = sess["position"]
    t0 = float(sess["maze_ep"].start[0])
    ax.plot(p.t - t0, p.values, ".", ms=0.8, color="0.35")
    for s, e in zip(laps["rightward"].start, laps["rightward"].end):
        ax.axvspan(s - t0, e - t0, color="tab:red", alpha=0.25, lw=0)
    for s, e in zip(laps["leftward"].start, laps["leftward"].end):
        ax.axvspan(s - t0, e - t0, color="tab:blue", alpha=0.25, lw=0)
    ax.set_xlim(0, 400)
    ax.set_xlabel("time from maze onset (s)"); ax.set_ylabel("position (m)")
    ax.set_title(f"extent {sess['extent']:.2f} m, "
                 f"{laps['n_laps']['rightward']}R/{laps['n_laps']['leftward']}L laps", fontsize=9)

    ax = axs[i, 2]
    ax.hist(p.values, bins=60, color="0.6", label="all tracked")
    ax.hist(p.restrict(mov["both"]).values, bins=60, color="tab:green", label="running, in lap")
    ax.set_xlabel("position (m)"); ax.set_ylabel("samples")
    ax.legend(fontsize=8)
    ax.set_title(f"running time {mov['both'].tot_length():.0f} s", fontsize=9)

    print(f"{label:20s} extent={sess['extent']:.2f} laps={laps['n_laps']} "
          f"run={mov['both'].tot_length():.0f}s tracked={sess['pos_ep'].tot_length():.0f}s "
          f"maze={float(sess['maze_ep'].end[0]-sess['maze_ep'].start[0]):.0f}s")
    sess["io"].close()

plt.tight_layout()
plt.savefig("fig_qc_geometry.png", dpi=110)
print("saved fig_qc_geometry.png")
