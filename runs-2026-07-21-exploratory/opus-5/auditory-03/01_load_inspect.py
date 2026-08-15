"""Step 1: load one session of DANDI:000986 and validate every data stream.

Writes figures/01_raw_data_overview.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

from common import (BASELINE_WIN, EVOKED_WIN, EXAMPLE_SESSION, FIGDIR, FREQS,
                    FREQ_LABELS, TONE_DUR, load_session, window_counts)

os.makedirs(FIGDIR, exist_ok=True)

name, asset = EXAMPLE_SESSION
sess = load_session(name, asset)
trials, spikes = sess["trials"], sess["spikes"]
onsets = trials.start_time.values
freq_of_trial = trials.stim_frequency.values

print(f"session {name}  subject {sess['subject']}")
print(f"{len(spikes)} units, {len(trials)} tone trials")
print(f"frequencies (Hz): {np.unique(freq_of_trial).astype(int)}")
print(f"level {np.unique(trials.stim_amplitude)} dB SPL, "
      f"tone duration {np.unique(trials.stim_duration)[0] * 1e3:.0f} ms")
print(f"trials per frequency: {trials.groupby('stim_frequency').size().values}")
print(f"median inter-tone interval {np.median(np.diff(onsets)):.3f} s")
print(f"{len(sess['spontaneous'])} silent blocks of "
      f"{(sess['spontaneous'].stop_time - sess['spontaneous'].start_time).mean():.0f} s")

rates = np.array([len(spikes[u]) / spikes.time_support.tot_length() for u in spikes.index])
print(f"unit firing rates: median {np.median(rates):.2f} Hz, "
      f"range {rates.min():.2f}-{rates.max():.2f} Hz")

# Population PSTH around tone onset (all trials pooled).
bin_size = 0.002
psth_win = (-0.1, 0.25)
edges = np.arange(psth_win[0], psth_win[1] + bin_size / 2, bin_size)
all_spikes = np.sort(np.concatenate([spikes[u].t for u in spikes.index]))
idx = np.searchsorted(all_spikes, onsets[:, None] + edges[None, :])
pop_psth = np.diff(idx, axis=1).sum(0) / (len(onsets) * bin_size * len(spikes))

# Rate in the evoked and baseline windows, per unit.
ev = window_counts(spikes, onsets, EVOKED_WIN) / (EVOKED_WIN[1] - EVOKED_WIN[0])
bl = window_counts(spikes, onsets, BASELINE_WIN) / (BASELINE_WIN[1] - BASELINE_WIN[0])
print(f"grand mean baseline {bl.mean():.2f} Hz, evoked {ev.mean():.2f} Hz")

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25, height_ratios=[1, 1, 1.3, 1])

# (a) stimulus sequence
ax = fig.add_subplot(gs[0, :])
ax.plot(onsets, freq_of_trial / 1e3, ".", ms=1.5, color="tab:purple")
for _, b in sess["spontaneous"].iterrows():
    ax.axvspan(b.start_time, b.stop_time, color="0.85", zorder=0)
ax.set_yscale("log")
ax.set_yticks([2, 4, 8, 16, 32])
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("time in session (s)")
ax.set_ylabel("tone freq (kHz)")
ax.set_title("(a) Stimulus: 25 ms pure tones at 60 dB SPL, pseudorandom order; "
             "grey = silent blocks")

# (b) behaviour
ax = fig.add_subplot(gs[1, :])
pup = sess["pupil"]
ax.plot(pup.t[::200], pup.d[::200], lw=0.6, color="tab:green", label="pupil (norm.)")
ax.set_ylabel("pupil diameter\n(max-normalised)", color="tab:green")
ax.set_xlabel("time in session (s)")
ax2 = ax.twinx()
run = sess["running"]
ax2.plot(run.t[::200], run.d[::200], lw=0.4, color="tab:brown", alpha=0.6)
ax2.set_ylabel("running speed (cm/s)", color="tab:brown")
ax.set_title("(b) Behavioural state: pupil diameter and locomotion")

# (c) raw raster, 6 s excerpt
ax = fig.add_subplot(gs[2, 0])
t0 = onsets[400]
t1 = t0 + 6.0
for row, u in enumerate(spikes.index):
    st = spikes[u].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full(st.size, row), "|", color="k", ms=1.6, mew=0.35)
sel = (onsets >= t0) & (onsets <= t1)
for on in onsets[sel]:
    ax.axvspan(on - t0, on - t0 + TONE_DUR, color="tab:red", alpha=0.35, lw=0)
ax.set_xlim(0, 6)
ax.set_ylim(-2, len(spikes))
ax.set_xlabel("time from excerpt start (s)")
ax.set_ylabel("unit #")
ax.set_title("(c) Raw spiking; red bands = 25 ms tones", fontsize=11)

# (d) population PSTH
ax = fig.add_subplot(gs[2, 1])
centers = (edges[:-1] + bin_size / 2) * 1e3
ax.plot(centers, pop_psth, color="0.2", lw=1)
ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.25, label="tone")
ax.axvspan(EVOKED_WIN[0] * 1e3, EVOKED_WIN[1] * 1e3, color="tab:orange", alpha=0.2,
           label="evoked window")
ax.axvspan(BASELINE_WIN[0] * 1e3, BASELINE_WIN[1] * 1e3, color="tab:blue", alpha=0.15,
           label="baseline window")
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("population rate (Hz/unit)")
ax.set_title(f"(d) Population PSTH, all {len(onsets)} tones", fontsize=11)
ax.legend(fontsize=8)

# (e) per-unit evoked vs baseline
ax = fig.add_subplot(gs[3, 0])
ax.loglog(bl.mean(0), ev.mean(0), ".", ms=5, color="tab:orange")
lim = [1e-2, max(ev.mean(0).max(), bl.mean(0).max()) * 1.5]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("baseline rate (Hz)")
ax.set_ylabel("evoked rate (Hz)")
ax.set_title("(e) Tone drive, one point per unit", fontsize=11)

# (f) firing-rate distribution
ax = fig.add_subplot(gs[3, 1])
ax.hist(np.log10(rates), bins=30, color="0.4")
ax.set_xlabel("log$_{10}$ mean firing rate (Hz)")
ax.set_ylabel("units")
ax.set_title("(f) Session firing-rate distribution", fontsize=11)

fig.suptitle(f"DANDI:000986 — {name} (mouse {sess['subject']}, auditory cortex "
             f"Neuropixels, {len(spikes)} units)", fontsize=13, y=0.945)
fig.savefig(f"{FIGDIR}/01_raw_data_overview.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGDIR}/01_raw_data_overview.png")
