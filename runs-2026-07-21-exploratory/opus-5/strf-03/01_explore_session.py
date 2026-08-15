"""Load one session of DANDI:000986 and validate every data stream visually."""

import matplotlib.pyplot as plt
import numpy as np

import strf_lib as sl

PROTO = "sub-LA11/sub-LA11_ses-1_behavior.nwb"

assets = sl.list_assets()
ses = sl.load_session(assets[PROTO])
print(ses["meta"])
print(ses["blocks"])
units = ses["units"]
trials = ses["trials"]
blocks = ses["blocks"]
print("tone blocks:", len(blocks), "total tone time (s):", float(blocks.tot_length()))
print("trials per frequency:\n", trials.stim_frequency.value_counts().sort_index())
print("pupil NaNs:", np.isnan(ses["pupil"].d).mean(), "run NaNs:", np.isnan(ses["running"].d).mean())

rates = np.asarray(units.restrict(blocks).rates)
print("firing rate (Hz): median %.2f, range %.3f-%.1f"
      % (np.median(rates), rates.min(), rates.max()))

# --------------------------------------------------------------------------- #
# Figure 1: raw data overview
# --------------------------------------------------------------------------- #
t0 = float(blocks.start[0]) + 60.0
win = sl.nap.IntervalSet(start=t0, end=t0 + 8.0)
fig, axes = plt.subplots(
    4, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw=dict(height_ratios=[0.8, 2.6, 0.9, 0.9], hspace=0.28))

ax = axes[0]
sub = trials[(trials.start_time >= win.start[0]) & (trials.start_time <= win.end[0])]
cmap = plt.get_cmap("viridis")
for _, r in sub.iterrows():
    j = int(np.flatnonzero(sl.FREQS == r.stim_frequency)[0])
    ax.add_patch(plt.Rectangle((r.start_time, j - 0.4), sl.TONE_DUR, 0.8,
                               color=cmap(j / 4), lw=0))
ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_ylim(-0.7, 4.7)
ax.set_ylabel("tone\n(kHz)")
ax.set_title(f"{ses['meta']['subject']} session {ses['meta']['session']}: "
             f"raw data, 8 s of a tone block", pad=10)

ax = axes[1]
order = np.argsort(-rates)
keys = np.asarray(units.index)[order][:60]
for row, k in enumerate(keys):
    st = units[k].restrict(win).t
    ax.plot(st, np.full(st.size, row), "|", color="k", ms=3, mew=0.6)
ax.set_ylabel("unit (sorted by rate)")
ax.set_ylim(-1, len(keys))

ax = axes[2]
p = ses["pupil"].restrict(win)
ax.plot(p.t, p.d, color="tab:purple", lw=1)
ax.set_ylabel("pupil\n(a.u.)")

ax = axes[3]
r = ses["running"].restrict(win)
ax.plot(r.t, r.d, color="tab:green", lw=1)
ax.set_ylabel("speed\n(cm/s)")
ax.set_xlabel("time (s)")
for a in axes:
    for x in sub.start_time.values:
        a.axvline(x, color="0.85", lw=0.5, zorder=0)
fig.savefig(f"{sl.FIGDIR}/fig01_raw_data.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2: session structure (tone blocks vs spontaneous) + behaviour
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw=dict(hspace=0.3))
bs = 10.0
pop = units.count(bs).sum(axis=1) / (bs * len(units))
axes[0].plot(pop.t, pop.d, color="k", lw=0.8)
axes[0].set_ylabel("population\nrate (Hz/unit)")
axes[0].set_title("Session structure: 5 min tone blocks alternating with silence", pad=8)
pu = ses["pupil"].bin_average(bs)
axes[1].plot(pu.t, pu.d, color="tab:purple", lw=0.8)
axes[1].set_ylabel("pupil (a.u.)")
ru = ses["running"].bin_average(bs)
axes[2].plot(ru.t, ru.d, color="tab:green", lw=0.8)
axes[2].set_ylabel("speed (cm/s)")
axes[2].set_xlabel("time in session (s)")
for a in axes:
    for s, e in zip(blocks.start, blocks.end):
        a.axvspan(s, e, color="tab:blue", alpha=0.12, lw=0)
fig.savefig(f"{sl.FIGDIR}/fig02_session_structure.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 3: binary spectrogram used as the STRF stimulus
# --------------------------------------------------------------------------- #
stim = sl.stimulus_tsdframe(trials, blocks)
print("stimulus:", stim.shape, "duty cycle per channel:", np.asarray(stim).mean(axis=0))
fig, ax = plt.subplots(figsize=(11, 2.6))
s = stim.restrict(win)
ax.imshow(np.asarray(s).T, aspect="auto", origin="lower", cmap="Greys",
          extent=[s.t[0], s.t[-1], -0.5, 4.5], interpolation="nearest")
ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_ylabel("frequency (kHz)")
ax.set_xlabel("time (s)")
ax.set_title("Stimulus matrix S(f, t): 25 ms tones, 5 ms bins", pad=8)
fig.savefig(f"{sl.FIGDIR}/fig03_stimulus_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done")
