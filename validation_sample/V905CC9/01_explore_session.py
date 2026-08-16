"""Inspect one session of dandiset 000986 and plot the raw data streams.

The point of this script is validation, not analysis: before computing any
tuning curve we check that spike times, the tone-presentation table, and the
behavioural traces are all on a common clock and look sensible.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import aft_io

PROTOTYPE = "sub-LA11/sub-LA11_ses-1_behavior.nwb"

manifest = aft_io.list_assets()
nwbfile, nwb, h5 = aft_io.open_session(manifest[PROTOTYPE])

print(nwb)
print()
print("session:", nwbfile.session_id, "| subject:", nwbfile.subject.subject_id,
      nwbfile.subject.sex, nwbfile.subject.age)
print("description:", nwbfile.session_description)

units = aft_io.load_units(nwbfile)
trials = aft_io.load_trials(nwbfile)
freqs = np.unique(trials.stim_frequency.values)

print(f"\n{len(units)} units, {len(trials)} tone trials")
print("frequencies (Hz):", freqs)
print("level (dB SPL):", np.unique(trials.stim_amplitude.values))
print("tone duration (s):", np.unique(trials.stim_duration.values))
print("trials per frequency:", trials.stim_frequency.value_counts().sort_index().to_dict())
isi = np.diff(trials.start_time.values)
print(f"inter-tone interval: median {np.median(isi):.3f} s, "
      f"min {isi.min():.3f} s, max {isi.max():.3f} s")

rates = np.asarray(units.rates)
print(f"firing rates: median {np.median(rates):.2f} Hz, "
      f"range {rates.min():.3f}-{rates.max():.1f} Hz")

# The tone block is bracketed by spontaneous blocks; restrict everything to the
# stimulus epoch so that the recording edges do not contaminate the analysis.
tone_epoch = nap.IntervalSet(start=trials.start_time.min() - 1.0,
                             end=trials.stop_time.max() + 1.0)
print("tone epoch:", tone_epoch)

pupil = nwb["pupil_diameter"]
running = nwb["running_speed"]
print("pupil:", type(pupil).__name__, pupil.shape, "| running:", running.shape)
print("NaNs in pupil:", int(np.isnan(pupil.values).sum()),
      "| NaNs in running:", int(np.isnan(running.values).sum()))

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2.4, 1, 0.8, 0.8]})
t0 = trials.start_time.values[400]
window = nap.IntervalSet(start=t0, end=t0 + 20.0)

busiest = [int(units.index[i]) for i in np.argsort(rates)[-40:]]
for row, uid in enumerate(busiest):
    ts = units[uid].restrict(window).times()
    axes[0].vlines(ts, row + 0.5, row + 1.5, lw=0.5, color="k")
axes[0].set_ylim(0.5, len(busiest) + 0.5)
axes[0].set_ylabel("unit (40 most active)")
axes[0].set_title(f"{aft_io.session_label(PROTOTYPE)}: raw spiking and behaviour "
                  f"during a 20 s stretch of the tone block", pad=10)

in_win = trials[(trials.start_time >= t0) & (trials.start_time <= t0 + 20)]
cmap = plt.get_cmap("viridis")
fcol = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}
for _, tr in in_win.iterrows():
    for ax in axes[:2]:
        ax.axvspan(tr.start_time, tr.start_time + tr.stim_duration,
                   color=fcol[tr.stim_frequency], alpha=0.55, lw=0)

pop = units.count(0.010, window).sum(axis=1) / 0.010
axes[1].plot(pop.times(), pop.values, lw=0.8, color="0.25")
axes[1].set_ylabel("population\nrate (Hz)")

pw = pupil.restrict(window)
axes[2].plot(pw.times(), pw.values, lw=1, color="tab:purple")
axes[2].set_ylabel("pupil\n(a.u.)")

axes[3].plot(running.restrict(window).times(), running.restrict(window).values,
             lw=1, color="tab:green")
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")

handles = [plt.Line2D([], [], color=fcol[f], lw=6, alpha=0.55,
                      label=f"{f / 1000:g} kHz") for f in freqs]
axes[0].legend(handles=handles, ncol=5, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, title="tone frequency")
fig.tight_layout()
fig.savefig("fig01_raw_data_overview.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Sanity check that tones drive the population at all, before splitting by
# frequency: population PSTH aligned on every tone onset.
onsets = nap.Ts(t=trials.start_time.values)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

pop_rate = units.count(0.005, tone_epoch).sum(axis=1) / 0.005
peth = nap.compute_perievent(pop_rate, onsets, window=(-0.10, 0.25))
mean_psth = np.nanmean(np.asarray(peth), axis=1)
axes[0].plot(peth.times() * 1000, mean_psth, color="k")
axes[0].axvspan(0, 25, color="tab:orange", alpha=0.25, label="25 ms tone")
axes[0].set_xlabel("time from tone onset (ms)")
axes[0].set_ylabel("population rate (Hz)")
axes[0].set_title("population PSTH, all tones pooled")
axes[0].legend(frameon=False)

axes[1].hist(np.log10(rates), bins=30, color="0.4")
axes[1].set_xlabel("log10 firing rate (Hz)")
axes[1].set_ylabel("units")
axes[1].set_title(f"firing-rate distribution (n={len(units)})")
fig.tight_layout()
fig.savefig("fig02_population_check.png", dpi=150)
plt.close(fig)

print("\nwrote fig01_raw_data_overview.png, fig02_population_check.png")
