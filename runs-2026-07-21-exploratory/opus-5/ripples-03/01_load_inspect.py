"""Stage 1: load the session, inspect every data stream, and pick a ripple channel."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert

import swr_utils as su

SESSION = "Achilles-10252013"

h5 = su.open_session(SESSION)
epochs = su.load_epochs(h5)
states = su.load_states(h5)
units = su.load_units(h5)
pos = su.load_position(h5)
ds, rate, t0 = su.lfp_meta(h5)

print("session:", SESSION)
print("epochs:", {k: (float(v.start[0]), float(v.end[-1])) for k, v in epochs.items()})
print("states:", {k: (len(v), float(v.tot_length())) for k, v in states.items()})
print("units:", units)
print("position: n=%d  range=%.1f-%.1f cm  dur=%.0f s"
      % (len(pos), pos.min(), pos.max(), pos.time_support.tot_length()))
print("LFP:", ds.shape, "at", rate, "Hz  ->", ds.shape[0] / rate / 3600, "h")


# ---------------------------------------------------------------- channel choice
def bandpass(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


# 200 s of non-REM sleep in the POST epoch: ripples are densest here.
post = epochs["POST"]
nrem_post = states["Non-REM"].intersect(post)
seg = max(range(len(nrem_post)), key=lambda i: nrem_post.end[i] - nrem_post.start[i])
probe_start = float(nrem_post.start[seg]) + 5
probe_stop = probe_start + 200
print("probing channels on non-REM %.1f-%.1f s" % (probe_start, probe_stop))

probe = su.load_lfp_window(h5, probe_start, probe_stop)
rip = bandpass(probe.values, 130, 250, rate)
env = np.abs(hilbert(rip, axis=0))
ripple_sd = env.std(axis=0)
best_ch = int(np.argmax(ripple_sd))
print("best ripple channel:", best_ch, " SD=%.1f" % ripple_sd[best_ch])

shank = h5["general/extracellular_ephys/electrodes/shank_electrode_number"][:]
group = su._decode(h5["general/extracellular_ephys/electrodes/group_name"][:])
print("  on", group[best_ch], "site", shank[best_ch])

# A second, independent channel on a different shank for cross-validation.
other = [c for c in range(len(ripple_sd)) if group[c] != group[best_ch]]
best_ch2 = int(other[np.argmax(ripple_sd[other])])
print("independent channel:", best_ch2, "on", group[best_ch2])
np.save("channels.npy", np.array([best_ch, best_ch2]))

fig, ax = plt.subplots(figsize=(10, 3.2))
colors = ["C0" if g == group[best_ch] else "0.6" for g in group]
ax.bar(np.arange(len(ripple_sd)), ripple_sd, color=colors)
ax.plot(best_ch, ripple_sd[best_ch] * 1.06, "v", color="C3", ms=9)
ax.plot(best_ch2, ripple_sd[best_ch2] * 1.06, "v", color="C1", ms=9)
ax.set(xlabel="LFP channel", ylabel="130-250 Hz envelope SD (a.u.)",
       title=f"{SESSION}: ripple-band power across 128 sites (200 s of non-REM)")
ax.text(best_ch, ripple_sd[best_ch] * 1.12, f"ch {best_ch}", color="C3", ha="center")
ax.text(best_ch2, ripple_sd[best_ch2] * 1.12, f"ch {best_ch2}", color="C1", ha="center")
ax.margins(y=0.18)
fig.tight_layout()
fig.savefig("fig01_channel_selection.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------ raw-data overview
lfp_probe = probe[:, best_ch]
t_ex = float(lfp_probe.index[0])
win = nap.IntervalSet(t_ex + 20, t_ex + 24)
raw = lfp_probe.restrict(win)
filt = nap.Tsd(t=lfp_probe.index, d=bandpass(lfp_probe.values, 130, 250, rate)).restrict(win)

fig, axes = plt.subplots(4, 1, figsize=(11, 9))
axes[0].plot(raw.index - t_ex, raw.values, lw=0.6, color="k")
axes[0].set(ylabel="LFP (a.u.)", title=f"Raw CA1 LFP, ch {best_ch} (non-REM sleep)")
axes[1].plot(filt.index - t_ex, filt.values, lw=0.6, color="C3")
axes[1].set(ylabel="130-250 Hz", xlabel="time in window (s)")

mz = epochs["MAZE"]
p = pos.restrict(mz)
axes[2].plot(p.index, p.values, lw=0.8, color="C0")
axes[2].set(ylabel="position (cm)", xlabel="time (s)",
            title="Linearized position on the 1.6 m track (MAZE epoch)")

exc = units.getby_category("cell_type")["excitatory"]
sub = exc.restrict(nap.IntervalSet(mz.start[0], mz.start[0] + 120))
for i, k in enumerate(list(sub.keys())[:60]):
    tt = sub[k].index
    axes[3].plot(tt, np.full_like(tt, i), "|", ms=2.5, color="k", mew=0.5)
axes[3].set(ylabel="unit #", xlabel="time (s)", title="Spike raster, 60 pyramidal cells (first 2 min on track)")
fig.tight_layout()
fig.savefig("fig02_raw_streams.png", dpi=150)
plt.close(fig)
print("wrote fig01, fig02")
