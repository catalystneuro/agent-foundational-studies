"""Load one session of DANDI:000044 and validate every data stream before analysis."""

import numpy as np
import scipy.signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import theta_lib as tl

SESSION = tl.SESSIONS[0]

h5 = tl.open_session(SESSION)
print("session:", h5["general/session_id"][()])

d = tl.load_behavior_and_spikes(h5)
spikes, position, maze_ep = d["spikes"], d["position"], d["maze_ep"]

print("maze epoch: %.1f - %.1f s (%.1f s)" % (maze_ep.start[0], maze_ep.end[0], maze_ep.end[0] - maze_ep.start[0]))
print("units:", len(spikes), np.unique(spikes.cell_type, return_counts=True))

laps, pos = tl.get_laps(position)
speed = tl.lap_speed(pos, laps)
right_ep = laps[laps.direction == "right"]
left_ep = laps[laps.direction == "left"]
print(
    "laps: %d total (%d right, %d left); %.1f s of running; median lap %.2f s"
    % (
        len(laps),
        len(right_ep),
        len(left_ep),
        laps.tot_length(),
        np.median(laps.end - laps.start),
    )
)
print(
    "position on laps: %.1f - %.1f cm, %d samples, median speed %.1f cm/s"
    % (pos.restrict(laps).min(), pos.restrict(laps).max(), len(pos.restrict(laps)), np.median(speed.values))
)

# ---------------------------------------------------------------------------------
# Choose the LFP channel with the strongest theta during running
# ---------------------------------------------------------------------------------
groups = np.array([s.decode() for s in h5["general/extracellular_ephys/electrodes/group_name"][:]])
bad = h5["general/extracellular_ephys/electrodes/bad_electrode"][:]
candidates = []
for g in np.unique(groups):
    idx = np.where((groups == g) & (~bad))[0]
    if len(idx):
        candidates.append(int(idx[len(idx) // 2]))

best_ch, scores = tl.pick_theta_channel(h5, maze_ep, candidates)
print("theta/delta ratio per candidate:", {k: round(v, 2) for k, v in scores.items()})
print("selected LFP channel:", best_ch)

lfp = tl.read_lfp_channel(h5, best_ch, maze_ep.start[0], maze_ep.end[0])
theta_filt, theta_phase, theta_amp = tl.theta_phase_from_lfp(lfp)
print("LFP samples: %d at %.0f Hz" % (len(lfp), lfp.rate))

# ---------------------------------------------------------------------------------
# Figure 1: raw streams around a single rightward run
# ---------------------------------------------------------------------------------
t0, t1 = right_ep.start[5] - 0.5, right_ep.end[5] + 0.5
win = (t0, t1)

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, height_ratios=[1, 1, 1.2, 2])

pos_w = pos.get(*win)
axes[0].plot(pos_w.t - t0, pos_w.values, "k.-", ms=3, lw=1)
axes[0].set_ylabel("Position\n(cm)")
axes[0].set_title("Raw data streams during one rightward traversal (Achilles 10252013)")

sp_w = speed.get(*win)
axes[1].plot(sp_w.t - t0, sp_w.values, color="tab:green")
axes[1].set_ylabel("Speed\n(cm/s)")

lfp_w = lfp.get(*win)
tf_w = theta_filt.get(*win)
axes[2].plot(lfp_w.t - t0, lfp_w.values * 1e3, color="0.65", lw=0.7, label="raw LFP")
axes[2].plot(tf_w.t - t0, tf_w.values * 1e3, color="tab:red", lw=1.5, label="6-10 Hz")
axes[2].set_ylabel("LFP\n(mV)")
axes[2].legend(loc="upper right", fontsize=8, ncol=2)

pyr = spikes.getby_category("cell_type")["excitatory"]
for row, uid in enumerate(pyr.index):
    st = pyr[uid].get(*win)
    if len(st):
        axes[3].plot(st.t - t0, np.full(len(st), row), "|", color="k", ms=4, mew=0.8)
axes[3].set_ylabel("Pyramidal unit")
axes[3].set_xlabel("Time from traversal onset (s)")
axes[3].set_xlim(0, t1 - t0)

fig.tight_layout()
fig.savefig("fig01_raw_streams.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------------
# Figure 2: behaviour overview, LFP spectrum, speed distribution
# ---------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

seg = (maze_ep.start[0] + 150, maze_ep.start[0] + 270)
p_seg = position.get(*seg)
axes[0].plot(p_seg.t - seg[0], p_seg.values, ".", color="0.4", ms=2)
for s, e, dr in zip(laps.start, laps.end, laps.direction):
    if seg[0] < s < seg[1]:
        axes[0].axvspan(
            s - seg[0],
            min(e, seg[1]) - seg[0],
            color="tab:blue" if dr == "right" else "tab:orange",
            alpha=0.25,
        )
axes[0].set_xlim(0, seg[1] - seg[0])
axes[0].set_xlabel("Time from segment start (s)")
axes[0].set_ylabel("Linearized position (cm)")
axes[0].set_title("Track traversals\n(blue = rightward, orange = leftward)")

f, pxx = scipy.signal.welch(lfp.restrict(laps).values, fs=tl.LFP_FS, nperseg=int(2 * tl.LFP_FS))
axes[1].semilogy(f, pxx, "k")
axes[1].axvspan(6, 10, color="tab:red", alpha=0.2)
axes[1].set_xlim(0, 40)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD (V$^2$/Hz)")
axes[1].set_title("LFP spectrum during running\n(channel %d, peak %.1f Hz)" % (best_ch, f[(f > 4) & (f < 14)][np.argmax(pxx[(f > 4) & (f < 14)])]))

axes[2].hist(speed.values, bins=50, color="tab:green")
axes[2].set_xlabel("Speed (cm/s)")
axes[2].set_ylabel("Count")
axes[2].set_title("Running speed within traversals")

fig.tight_layout()
fig.savefig("fig02_behavior_and_spectrum.png", dpi=150)
plt.close(fig)

np.save("_cache_best_channel.npy", np.array([best_ch]))
print("done")
