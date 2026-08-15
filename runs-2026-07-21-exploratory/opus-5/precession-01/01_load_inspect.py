"""Step 1: load one hc-11 session, inspect every data stream, save validation plots."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pynapple as nap

import precession_lib as pl

SESSION = "Achilles-10252013"

f, nwbfile = pl.open_session(SESSION)
print("session:", nwbfile.session_id, "|", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.species)

nwb = nap.NWBFile(nwbfile)
print(nwb)

# ---- units -----------------------------------------------------------------
units = nwb["units"]
print("\nunits:", units)
meta = units.metadata_info if hasattr(units, "metadata_info") else None
print("unit columns:", list(units.metadata_columns))
print(units.get_info("cell_type").value_counts())
print(units.get_info("location").value_counts())

# ---- behaviour -------------------------------------------------------------
pos, fs_pos, track_len, maze_name = pl.load_position(f)
print(f"\nposition: {len(pos)} samples at {fs_pos:.2f} Hz, "
      f"{pos.time_support.start[0]:.1f}-{pos.time_support.end[0]:.1f} s, "
      f"range {np.nanmin(pos.values):.2f}-{np.nanmax(pos.values):.2f} m, "
      f"{np.isnan(pos.values).sum()} NaNs")

maze = pl.maze_epoch(f)
print("maze epoch:", maze)

vel, speed = pl.compute_speed(pos, fs_pos)
runs_r, runs_l = pl.find_runs(pos, vel, fs_pos, track_len)
print(f"runs: {len(runs_r)} rightward ({runs_r.tot_length():.0f} s), "
      f"{len(runs_l)} leftward ({runs_l.tot_length():.0f} s)")

# ---- LFP -------------------------------------------------------------------
chan, ratios = pl.pick_theta_channel(f, runs_r)
print(f"\nbest theta/delta channel: {chan} (ratio {ratios[chan]:.2f}); "
      f"median across 128 channels {np.median(ratios):.2f}")

lfp = pl.read_lfp_channel(f, chan, maze.start[0], maze.end[0])
print("lfp:", lfp)
filt, phase, amp = pl.theta_phase(lfp)

np.save("cache_theta_ratios.npy", ratios)

# ===========================================================================
# Validation figure 1: behaviour
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
t0 = maze.start[0]
axes[0].plot(pos.times() - t0, pos.values, "k", lw=0.7)
for ep, c in ((runs_r, "tab:red"), (runs_l, "tab:blue")):
    for s, e in zip(ep.start, ep.end):
        axes[0].axvspan(s - t0, e - t0, color=c, alpha=0.25, lw=0)
axes[0].set_ylabel("linearized\nposition (m)")
axes[0].set_title(f"{SESSION}: behaviour on the 1.6 m linear track "
                  f"(red = rightward runs, blue = leftward runs)")

axes[1].plot(vel.times() - t0, vel.values, "k", lw=0.7)
axes[1].axhline(pl.SPEED_THRESH, color="tab:red", ls="--", lw=1)
axes[1].axhline(-pl.SPEED_THRESH, color="tab:blue", ls="--", lw=1)
axes[1].set_ylabel("velocity (m/s)")

# spike raster for a subset of units
exc = units.getby_category("cell_type")["excitatory"]
rast_ids = list(exc.keys())[:40]
for i, uid in enumerate(rast_ids):
    st = exc[uid].restrict(maze).times() - t0
    axes[2].plot(st, np.full(st.size, i), "|", color="k", ms=2, mew=0.4)
axes[2].set_ylabel("unit #")
axes[2].set_xlabel("time from maze onset (s)")
axes[2].set_xlim(100, 300)
fig.tight_layout()
fig.savefig("fig01_behavior_and_raster.png", dpi=150)
plt.close(fig)

# ===========================================================================
# Validation figure 2: LFP and theta extraction
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# channel selection
axes[0].plot(ratios, "k.-", ms=3, lw=0.6)
axes[0].plot(chan, ratios[chan], "r*", ms=14)
axes[0].set_xlabel("LFP channel")
axes[0].set_ylabel("theta / delta\npower ratio")
axes[0].set_title("Reference channel selection (probe window during running)")

# PSD of the chosen channel during running
seg = lfp.restrict(runs_r)
ratio, (fr, psd) = pl.theta_power_ratio(lfp.restrict(maze).values, pl.LFP_RATE)
axes[1].semilogy(fr, psd, "k")
axes[1].axvspan(*pl.THETA_BAND, color="tab:orange", alpha=0.3)
axes[1].set_xlim(0, 40)
axes[1].set_xlabel("frequency (Hz)")
axes[1].set_ylabel("PSD (V$^2$/Hz)")
axes[1].set_title(f"Power spectrum, channel {chan}, maze epoch (theta band shaded)")

# raw vs filtered trace with phase
w = nap.IntervalSet(start=runs_r.start[2], end=runs_r.start[2] + 2.0)
axes[2].plot(lfp.restrict(w).times(), lfp.restrict(w).values * 1e3, color="0.6", lw=0.8,
             label="raw LFP")
axes[2].plot(filt.restrict(w).times(), filt.restrict(w).values * 1e3, "k", lw=1.5,
             label="6-12 Hz")
ax2b = axes[2].twinx()
ax2b.plot(phase.restrict(w).times(), np.degrees(phase.restrict(w).values), color="tab:orange",
          lw=0.8, label="theta phase")
ax2b.set_ylabel("theta phase (deg)", color="tab:orange")
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("LFP (mV)")
axes[2].legend(loc="upper left", fontsize=8)
axes[2].set_title("Raw LFP, theta-filtered signal and Hilbert phase (2 s of running)")
fig.tight_layout()
fig.savefig("fig02_lfp_theta_validation.png", dpi=150)
plt.close(fig)

print("\nwrote fig01_behavior_and_raster.png, fig02_lfp_theta_validation.png")
