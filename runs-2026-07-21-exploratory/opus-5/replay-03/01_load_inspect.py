"""Stage 1: load one session, inspect every data stream, pick the ripple channel."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from tqdm import tqdm

import replaylib as R

SESSION = os.environ.get("SESSION", "Achilles-10252013")
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
os.makedirs("cache", exist_ok=True)

h5, nwbfile, nwb = R.open_session(SESSION)
print(nwb)
print("session:", nwbfile.session_id, "| subject:", nwbfile.subject.subject_id)

# ---------------------------------------------------------------- epochs
ep = nwbfile.epochs.to_dataframe()
print("\nEpochs:\n", ep)
epochs = {
    row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
    for row in ep.itertuples()
}
states_df = nwbfile.processing["behavior"]["states"].to_dataframe()
print("\nBrain states:", states_df.label.value_counts().to_dict())
nrem_rows = states_df[states_df.label == "Non-REM"]
states_iv = nap.IntervalSet(start=nrem_rows.start_time.values,
                            end=nrem_rows.stop_time.values)

# ---------------------------------------------------------------- units
units = nwb["units"]
cell_type = np.array(nwbfile.units["cell_type"][:])
location = np.array(nwbfile.units["location"][:])
print(f"\n{len(units)} units | excitatory {np.sum(cell_type == 'excitatory')}"
      f" | inhibitory {np.sum(cell_type == 'inhibitory')}")
print("locations:", {u: int((location == u).sum()) for u in np.unique(location)})

# ---------------------------------------------------------------- behaviour
position, posinfo = R.load_position(h5)
print("\nposition:", posinfo)
speed = R.compute_speed(position)
maze = epochs["MazeEpoch"]
run_dirs = R.direction_intervals(position, speed)
print("run traversals: right %d (%.0f s), left %d (%.0f s)"
      % (len(run_dirs["right"]), run_dirs["right"].tot_length(),
         len(run_dirs["left"]), run_dirs["left"].tot_length()))

# ---------------------------------------------------------------- LFP survey
rate, t0, n_t, n_ch = R.lfp_meta(h5)
print(f"\nLFP: {n_ch} ch @ {rate} Hz, {n_t / rate / 3600:.2f} h")
post = epochs["POSTEpoch"]

# Score every channel by ripple-band power during a slice of POST sleep.
probe_start = float(post.start[0]) + 300.0
probe_stop = probe_start + 120.0
scores = []
for ch in tqdm(range(n_ch), desc="scoring channels"):
    seg = R.read_lfp(h5, [ch], probe_start, probe_stop)
    x = seg.values[:, 0]
    if np.allclose(x, 0):
        scores.append(np.nan)
        continue
    _, env = R.ripple_envelope(x, rate)
    # Fraction of time the robustly scaled envelope exceeds 5 robust SD: the CA1
    # pyramidal layer has the highest density of ripple-band transients.  Scoring
    # by peak amplitude instead favours quiet channels carrying a few artifacts.
    scores.append(R.ripple_density(env))
scores = np.array(scores)
best_ch = int(np.nanargmax(scores))
print(f"best ripple channel: {best_ch} (density {scores[best_ch]:.4f})")

np.savez("cache/session_meta.npz", session=SESSION, best_ch=best_ch,
         scores=scores, lfp_rate=rate)

# ---------------------------------------------------------------- figure 1
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.55, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
epoch_colors = {"PREEpoch": "tab:blue", "MazeEpoch": "tab:green", "POSTEpoch": "tab:red"}
for lab, iv in epochs.items():
    ax.axvspan(iv.start[0] / 3600, iv.end[0] / 3600, alpha=0.3,
               color=epoch_colors[lab], label=f"{lab.replace('Epoch', '')} "
                                              f"({iv.tot_length() / 3600:.1f} h)")
for s, e in zip(states_iv.start, states_iv.end):
    ax.axvspan(s / 3600, e / 3600, ymin=0.0, ymax=0.25, color="k", alpha=0.55, lw=0)
ax.set_xlim(0, n_t / rate / 3600)
ax.set_yticks([])
ax.set_xlabel("time (h)")
ax.set_title(f"{SESSION}: session structure ({n_t / rate / 3600:.1f} h total); "
             f"black bars = scored non-REM")
ax.legend(loc="upper center", ncol=3, fontsize=9, framealpha=0.95)

ax = fig.add_subplot(gs[1, :])
ax.plot(position.t, position.d, "k.", ms=0.6)
ax.set_xlim(maze.start[0], maze.end[0])
ax.set_xlabel("time (s)")
ax.set_ylabel("linear position (m)")
ax.set_title("Linear-track behaviour (full maze epoch)")

ax = fig.add_subplot(gs[2, 0])
w = (position.t > maze.start[0] + 250) & (position.t < maze.start[0] + 400)
ax.plot(position.t[w], position.d[w], "k-", lw=1)
for name, col in (("right", "tab:red"), ("left", "tab:blue")):
    for s, e in zip(run_dirs[name].start, run_dirs[name].end):
        if s > maze.start[0] + 250 and e < maze.start[0] + 400:
            m = (position.t >= s) & (position.t <= e)
            ax.plot(position.t[m], position.d[m], color=col, lw=2)
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.set_title("Traversals: rightward (red) / leftward (blue)", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
ax.hist(speed.d[np.isfinite(speed.d)], bins=60, color="0.4")
ax.axvline(0.05, color="r", ls="--", label="run threshold 5 cm/s")
ax.set_xlabel("speed (m/s)")
ax.set_ylabel("samples")
ax.set_title("Speed distribution", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[3, 0])
ax.plot(scores, "k.-", ms=3, lw=0.5)
ax.plot(best_ch, scores[best_ch], "r*", ms=14)
ax.set_xlabel("LFP channel")
ax.set_ylabel("fraction of time above 5 robust SD")
ax.set_title(f"Ripple channel selection (best = ch {best_ch})", fontsize=10)

ax = fig.add_subplot(gs[3, 1])
rates = np.array([len(units[i]) / (n_t / rate) for i in units.index])
ax.hist(rates[cell_type == "excitatory"], bins=np.logspace(-2, 1.6, 30),
        alpha=0.7, label="excitatory")
ax.hist(rates[cell_type == "inhibitory"], bins=np.logspace(-2, 1.6, 30),
        alpha=0.7, label="inhibitory")
ax.set_xscale("log")
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("units")
ax.set_title("Unit firing rates", fontsize=10)
ax.legend(fontsize=8)

fig.savefig(f"{FIG}/01_session_overview.png", dpi=130, bbox_inches="tight")
print("wrote", f"{FIG}/01_session_overview.png")
