"""Load one session from DANDI:000059 and validate every data stream before analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import theta_lib as tl

SESSION = "Peter-MS21-180628-155921-concat"

fp, fr = tl.open_session_files(SESSION)
print("alignment:", tl.check_alignment(fp, fr))

units = tl.load_units(fp)
speed, pos, temp = tl.load_behavior(fp)
trials, cooling, condition = tl.load_trials(fp)
col, row = tl.theta_reference_column(fr)
print(f"units: {len(units)} curated | theta reference: electrode row {row} -> LFP column {col}")
print(f"behaviour: {speed.t[0]:.1f}-{speed.t[-1]:.1f} s, median speed {np.nanmedian(speed.d):.1f} cm/s")
print(f"trials: {len(trials)}, cooling states {dict(zip(*np.unique(cooling, return_counts=True)))}")

# Bound the LFP read by the behavioural epoch (the maze part of the session).
t0, t1 = float(speed.t[0]), float(speed.t[-1])
lfp, fs = tl.load_lfp(fr, t0, t1, column=col)
print(f"LFP: {len(lfp)} samples @ {fs} Hz, {lfp.t[0]:.1f}-{lfp.t[-1]:.1f} s, "
      f"range {lfp.d.min():.2f} to {lfp.d.max():.2f} mV")

filt, phase = tl.theta_phase(lfp, fs)
run = tl.running_epochs(speed)
print(f"running epochs: {len(run)}, total {run.tot_length():.0f} s "
      f"({100 * run.tot_length() / (t1 - t0):.0f}% of the behavioural period)")

# ---------------------------------------------------------------------------
# Figure 1: raw data validation
# ---------------------------------------------------------------------------
# Pick a running window with strong theta and plenty of spikes, so both the
# oscillation and the raster are legible. Selecting on spike count alone lands on
# sharp-wave bursts, which are exactly the periods when theta is absent.
env0 = nap.compute_hilbert_envelope(filt)
WIN = 2.5
cands = np.arange(float(run.start[0]), float(run.end[-1]) - WIN, 1.0)
cands = cands[[len(run.intersect(nap.IntervalSet(c, c + WIN))) == 1 and
               run.intersect(nap.IntervalSet(c, c + WIN)).tot_length() > WIN - 0.01 for c in cands]]
amp = np.array([env0.restrict(nap.IntervalSet(c, c + WIN)).d.mean() for c in cands])
nsp = np.array([len(units.restrict(nap.IntervalSet(c, c + WIN)).to_tsd()) for c in cands])
ok = amp > np.percentile(amp, 80)
w0 = float(cands[ok][np.argmax(nsp[ok])])
win = nap.IntervalSet(w0, w0 + WIN)
order = np.argsort(-np.array([len(units[u].restrict(win)) for u in units.index]))
fig, axes = plt.subplots(5, 1, figsize=(11, 11), height_ratios=[1.1, 1.1, 1.6, 1, 1])

ax = axes[0]
ax.plot(lfp.restrict(win).t, lfp.restrict(win).d, lw=0.8, color="0.35", label="LFP (wideband)")
ax.plot(filt.restrict(win).t, filt.restrict(win).d, lw=1.6, color="crimson", label=f"{tl.THETA_BAND[0]}-{tl.THETA_BAND[1]} Hz")
ax.set_ylabel("mV")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"{SESSION}: theta-reference LFP during running", fontsize=10)

ax = axes[1]
ph = phase.restrict(win)
ax.plot(ph.t, ph.d, lw=1.2, color="navy")
ax.set_ylabel("theta phase\n(rad)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])

ax = axes[2]
sub = units.restrict(win)
for i, uid in enumerate(np.array(units.index)[order][:40]):
    ax.vlines(sub[int(uid)].t, i, i + 0.85, color="k", lw=0.7)
ax.set_ylabel("unit\n(rate-ranked)")
ax.set_xlabel("time (s)")
ax.set_title("spike raster (40 most active curated units in this window)", fontsize=9)

for a in axes[:3]:
    a.set_xlim(w0, w0 + WIN)

ax = axes[3]
ax.plot(speed.t, speed.d, lw=0.5, color="0.3")
ax.axhline(tl.SPEED_THRESHOLD, color="crimson", ls="--", lw=1, label=f"{tl.SPEED_THRESHOLD} cm/s")
ax.set_ylabel("speed\n(cm/s)")
ax.set_ylim(0, np.nanpercentile(speed.d, 99.5))
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(t0, t1)

ax = axes[4]
if temp is not None:
    ax.plot(temp.t, temp.d, lw=0.8, color="steelblue")
    ax.set_ylabel("septal temp\n(°C)")
for state, color in [("Cooling on", "steelblue"), ("Pre-Cooling", "0.6"), ("Post-Cooling", "0.8")]:
    m = cooling == state
    if m.any():
        for s, e in zip(trials.start[m], trials.end[m]):
            ax.axvspan(s, e, color=color, alpha=0.25, lw=0)
ax.set_xlabel("time (s)")
ax.set_xlim(t0, t1)

fig.tight_layout()
fig.savefig("fig01_raw_data_validation.png", dpi=150)
print("wrote fig01_raw_data_validation.png")

# ---------------------------------------------------------------------------
# Figure 2: LFP power spectrum, running vs immobility
# ---------------------------------------------------------------------------
# The animal is almost never immobile on this maze (1st percentile of smoothed speed
# is 3 cm/s), so the no-theta contrast comes from the post-maze rest period instead.
rest_lfp, _ = tl.load_lfp(fr, t1 + 60, t1 + 960, column=col)
rest_ep = nap.IntervalSet(rest_lfp.t[0], rest_lfp.t[-1])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for sig, ep, label, color in [(lfp, run, "running (maze)", "crimson"),
                              (rest_lfp, rest_ep, "post-maze rest", "0.4")]:
    psd = nap.compute_mean_power_spectral_density(sig, interval_size=4.0, fs=fs, ep=ep, full_range=False)
    f = np.asarray(psd.index)
    p = np.abs(np.asarray(psd.values).ravel())
    m = (f > 1) & (f < 30)
    ax.semilogy(f[m], p[m], color=color, label=label)
ax.axvspan(*tl.THETA_BAND, color="gold", alpha=0.25, lw=0, zorder=0)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power (a.u.)")
ax.set_title("LFP power spectrum", fontsize=10)
ax.legend()

# Theta power grows with running speed, the standard check that the band is real theta.
ax = axes[1]
env = nap.compute_hilbert_envelope(filt)
sp_sm = tl.smooth_speed(speed)
env_b = env.bin_average(0.25)
sp_b = sp_sm.bin_average(0.25)
common = np.intersect1d(np.round(env_b.t, 3), np.round(sp_b.t, 3))
e = env_b.d[np.isin(np.round(env_b.t, 3), common)]
s = sp_b.d[np.isin(np.round(sp_b.t, 3), common)]
ok = np.isfinite(e) & np.isfinite(s)
e, s = e[ok], s[ok]
bins = np.array([0, 5, 10, 20, 30, 45, 60, 90, 130])
idx = np.digitize(s, bins) - 1
centers = 0.5 * (bins[:-1] + bins[1:])
mean_env = np.array([e[idx == i].mean() if (idx == i).sum() > 5 else np.nan for i in range(len(centers))])
sem_env = np.array([e[idx == i].std() / max(np.sqrt((idx == i).sum()), 1) if (idx == i).sum() > 5 else np.nan
                    for i in range(len(centers))])
ax.errorbar(centers, mean_env, yerr=sem_env, marker="o", color="crimson")
ax.set_xlabel("running speed (cm/s)")
ax.set_ylabel("theta envelope (mV)")
ax.set_title("theta amplitude increases with speed", fontsize=10)
fig.tight_layout()
fig.savefig("fig02_power_spectrum.png", dpi=150)
print("wrote fig02_power_spectrum.png")
