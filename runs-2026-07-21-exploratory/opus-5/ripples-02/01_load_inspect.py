"""Load one session of DANDI:000044, inspect the data streams, pick a ripple channel."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert

from dandi_io import open_session

SESSION = "Achilles_10252013"
FS = 1250.0

nwbfile, nwb, h5 = open_session(SESSION)
lfp_es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
elec = nwbfile.electrodes.to_dataframe()
units_df = nwbfile.units.to_dataframe()

epochs = nwb["epochs"]
states = nwb["states"]
print("epochs:\n", nwbfile.epochs.to_dataframe())
print("units:", len(units_df), units_df.cell_type.value_counts().to_dict())
print("electrodes:", len(elec), "bad:", int(elec.bad_electrode.sum()))

# ---------------------------------------------------------------- channel pick
# One 136 s slab of POST-task non-REM sleep, all 128 channels (chunks are (n, 1)
# so this is a cheap read).  The best ripple channel is the one with the largest
# ripple-band envelope standard deviation.
st_df = nwbfile.processing["behavior"]["states"].to_dataframe()
post_start = nwbfile.epochs.to_dataframe().query("label == 'POSTEpoch'").start_time.iloc[0]
nrem = st_df[(st_df.label == "Non-REM") & (st_df.start_time > post_start)]
nrem = nrem.assign(dur=nrem.stop_time - nrem.start_time).sort_values("dur", ascending=False)
t0 = float(nrem.start_time.iloc[0])
print(f"probe slab: {t0:.0f}-{t0 + 120:.0f} s (longest POST non-REM bout, {nrem.dur.iloc[0]:.0f} s)")

i0 = int(t0 * FS)
slab = lfp_es.data[i0 : i0 + int(120 * FS), :] * lfp_es.conversion  # volts

b, a = butter(4, [130, 250], btype="bandpass", fs=FS)
ripple_sd = np.zeros(slab.shape[1])
for ch in range(slab.shape[1]):
    ripple_sd[ch] = np.std(np.abs(hilbert(filtfilt(b, a, slab[:, ch]))))

ripple_sd[elec.bad_electrode.values.astype(bool)] = 0.0
best_ch = int(np.argmax(ripple_sd))
print(f"best ripple channel = {best_ch} "
      f"(shank {elec.group_name.iloc[best_ch]}, {elec.location.iloc[best_ch]})")
np.save("ripple_channel.npy", np.array([best_ch]))

# ---------------------------------------------------------------------- figure
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25, height_ratios=[1, 1, 1, 1.1])

ax = fig.add_subplot(gs[0, :])
ep_df = nwbfile.epochs.to_dataframe()
colors = {"PREEpoch": "#8ecae6", "MazeEpoch": "#fb8500", "POSTEpoch": "#219ebc"}
for _, r in ep_df.iterrows():
    ax.axvspan(r.start_time / 60, r.stop_time / 60, color=colors[r.label], alpha=0.6)
    ax.text((r.start_time + r.stop_time) / 120, 1.35, r.label.replace("Epoch", ""),
            ha="center", fontsize=9)
scol = {"Awake": "#adb5bd", "Non-REM": "#023047", "REM": "#e63946"}
for lab, c in scol.items():
    sub = st_df[st_df.label == lab]
    ax.barh(0.3, sub.stop_time / 60 - sub.start_time / 60, left=sub.start_time / 60,
            height=0.35, color=c, label=lab)
ax.set_ylim(0, 1.6)
ax.set_yticks([])
ax.set_xlabel("time (min)")
ax.set_title(f"{SESSION}: session structure and scored sleep states", pad=14)
ax.legend(ncol=3, fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
ax.plot(ripple_sd * 1e6, ".", ms=4, color="#023047")
ax.axvline(best_ch, color="#e63946", lw=1)
ax.set_xlabel("channel")
ax.set_ylabel("ripple-band env. SD (µV)")
ax.set_title(f"channel selection (best = {best_ch})", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
ax.plot(lin.t, lin.d[:, 0], lw=0.6, color="#fb8500")
ax.set_xlabel("time (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title("linear-track position (maze epoch)", fontsize=10)

# raw LFP snippet with a visible ripple
env = np.abs(hilbert(filtfilt(b, a, slab[:, best_ch])))
pk = int(np.argmax(env))
w = int(0.5 * FS)
sl = slice(max(pk - w, 0), pk + w)
tt = np.arange(sl.start, sl.stop) / FS + t0

ax = fig.add_subplot(gs[2, :])
ax.plot(tt, slab[sl, best_ch] * 1e6, lw=0.7, color="k")
ax.set_ylabel("LFP (µV)")
ax.set_title(f"raw LFP, channel {best_ch}, around the largest ripple in the probe slab",
             fontsize=10)
ax.set_xlim(tt[0], tt[-1])

ax = fig.add_subplot(gs[3, :])
ax.plot(tt, filtfilt(b, a, slab[:, best_ch])[sl] * 1e6, lw=0.7, color="#e63946")
ax.set_xlabel("time (s)")
ax.set_ylabel("130–250 Hz (µV)")
ax.set_xlim(tt[0], tt[-1])
ax.set_title("same trace, ripple band", fontsize=10)

fig.savefig("fig01_session_overview.png", dpi=150, bbox_inches="tight")
print("wrote fig01_session_overview.png")
