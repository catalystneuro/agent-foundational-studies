"""Load one session of DANDI 000986, print its structure, and validate the streams."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

import audlib as A

assets = A.list_assets()
print(f"{len(assets)} sessions in dandiset {A.DANDISET}")
path, url = assets[0]
print("prototyping on:", path)

nwbfile, nap_nwb = A.load_session(url)
print(nap_nwb)
print("session_description:", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.species,
      nwbfile.subject.sex, nwbfile.subject.age, "|", nwbfile.subject.description)
print("publication:", nwbfile.related_publications)

trials = A.trial_table(nwbfile)
spikes = nap_nwb["units"]
freqs = np.unique(trials["frequency"])
print("frequencies (kHz):", freqs / 1e3)
print("level (dB SPL):", np.unique(trials["amplitude"]),
      "duration (s):", np.unique(trials["duration"]))
print("n trials:", len(trials["onset"]), "reps/frequency:",
      [int((trials["frequency"] == f).sum()) for f in freqs])
print("median inter-tone interval (s):", np.median(np.diff(trials["onset"])))
print("n units:", len(spikes))
print("firing rate: median %.2f Hz, range %.2f-%.2f Hz"
      % (np.median(spikes.rate), spikes.rate.min(), spikes.rate.max()))
print("silent (spontaneous) blocks:")
print(nwbfile.intervals["spontaneous_blocks"].to_dataframe())

pupil = nap_nwb["pupil_diameter"]
run = nap_nwb["running_speed"]
print("pupil:", pupil.shape, "NaNs:", np.isnan(pupil.d).sum())
print("running:", run.shape, "NaNs:", np.isnan(run.d).sum())

# ---------------------------------------------------------------- figure 1 --
# Raw streams over a 20 s stretch of the tone block: spike raster, population
# rate, tone times coloured by frequency, pupil and running speed.
onsets = trials["onset"]
t0 = onsets[500]
t1 = t0 + 8.0
ep = nap.IntervalSet(t0, t1)
sel = (onsets >= t0) & (onsets < t1)
cmap = plt.get_cmap("viridis")
fcol = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})

order = np.argsort(spikes.rate.values)
for row, i in enumerate(order):
    k = list(spikes.keys())[i]
    t = spikes[k].restrict(ep).t
    axes[0].plot(t, np.full_like(t, row), "|", color="k", ms=2.5, mew=0.5)
axes[0].set_ylabel("unit (sorted by rate)")


pop = spikes.count(0.005, ep).sum(axis=1) / 0.005 / len(spikes)
axes[1].plot(pop.t, pop.d, lw=0.7, color="tab:blue")
axes[1].set_ylabel("population rate\n(spikes/s/unit)")

axes[2].plot(pupil.restrict(ep).t, pupil.restrict(ep).d, lw=1, color="tab:purple")
axes[2].set_ylabel("pupil diameter\n(a.u.)")

axes[3].plot(run.restrict(ep).t, run.restrict(ep).d, lw=1, color="tab:green")
axes[3].set_ylabel("running speed\n(a.u.)")
axes[3].set_xlabel("time (s)")

for ax in axes:
    for o, f in zip(onsets[sel], trials["frequency"][sel]):
        ax.axvline(o, color=fcol[f], lw=1.2, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], color=fcol[f], lw=2, label=f"{f/1e3:g} kHz")
           for f in freqs]
axes[0].legend(handles=handles, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), fontsize=9, title="tone frequency",
               title_fontsize=9)
fig.suptitle(f"{path} — raw data streams, 8 s of the tone block", y=0.995)
axes[0].set_xlim(t0, t1)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig01_raw_data_streams.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
# Population PSTH aligned to tone onset, used to choose the response window.
bins = np.arange(-0.15, 0.351, 0.002)
centers = bins[:-1] + np.diff(bins) / 2
pop_psth = np.zeros(len(bins) - 1)
for k in spikes.keys():
    pop_psth += A.psth(spikes[k].t, onsets, bins)
pop_psth /= len(spikes)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(centers, pop_psth, color="k", lw=1.2)
ax.axvspan(0, 0.025, color="tab:orange", alpha=0.3, lw=0, label="25 ms tone")
ax.axvspan(*A.EVOKED_WIN, color="tab:red", alpha=0.15, lw=0, label="evoked window")
ax.axvspan(*A.BASELINE_WIN, color="tab:blue", alpha=0.15, lw=0, label="baseline window")
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("mean firing rate (spikes/s/unit)")
ax.set_title(f"Population PSTH, all {len(onsets)} tones ({len(spikes)} units)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig02_population_psth.png", dpi=150)
plt.close(fig)

base = pop_psth[centers < 0].mean()
peak_t = centers[np.argmax(pop_psth)]
print(f"population baseline {base:.2f} sp/s; PSTH peak {pop_psth.max():.2f} sp/s "
      f"at {peak_t*1e3:.0f} ms")
print("saved fig01_raw_data_streams.png, fig02_population_psth.png")
