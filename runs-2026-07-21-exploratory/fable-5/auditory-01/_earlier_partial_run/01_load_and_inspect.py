"""Load one DANDI:000986 session and validate every data stream visually."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import dandi_io

ASSET = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11/sub-LA11_ses-1_behavior.nwb

sess = dandi_io.load_tone_session(ASSET)
print("subject:", sess["subject"])
print("units:", len(sess["spikes"]))
print("trials:", len(sess["trials"]))
print("frequencies (Hz):", sess["frequencies"])
print("trials per frequency:", {f: int((sess["frequency"] == f).sum()) for f in sess["frequencies"]})
print("rates (Hz): median %.2f  range %.2f-%.2f" % (
    np.median(sess["spikes"].rate), sess["spikes"].rate.min(), sess["spikes"].rate.max()))

trials = sess["trials"]
tone_block = nap.IntervalSet(start=trials["start_time"].min(), end=trials["stop_time"].max())

# ---------------------------------------------------------------- raw streams
pupil = nap.Tsd(t=np.asarray(sess["pupil"].timestamps), d=np.asarray(sess["pupil"].data))
running = nap.Tsd(t=np.asarray(sess["running"].timestamps), d=np.asarray(sess["running"].data))
print("pupil samples:", len(pupil), " NaNs:", int(np.isnan(pupil.values).sum()))
print("running samples:", len(running), " NaNs:", int(np.isnan(running.values).sum()))

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1, 1, 0.6]})

win = nap.IntervalSet(start=1000.0, end=1030.0)
sp = sess["spikes"].restrict(win)
for i, u in enumerate(sp.keys()):
    t = sp[u].times()
    axes[0].plot(t, np.full_like(t, i), "|", color="k", ms=2.2, mew=0.5)
axes[0].set_ylabel("unit #")
axes[0].set_title("DANDI:000986  sub-%s  raw spiking during passive pure-tone presentation"
                  % sess["subject"], pad=10)

onsets = trials["start_time"].values
freqs = sess["frequency"]
sel = (onsets >= 1000.0) & (onsets <= 1030.0)
colors = plt.cm.viridis(np.linspace(0, 0.92, len(sess["frequencies"])))
fmap = {f: colors[i] for i, f in enumerate(sess["frequencies"])}
for t0, fr in zip(onsets[sel], freqs[sel]):
    axes[0].axvline(t0, color=fmap[fr], alpha=0.55, lw=1.2)

rate = sess["spikes"].count(0.02, win).sum(axis=1) / 0.02 / len(sess["spikes"])
axes[1].plot(rate.times(), rate.values, color="firebrick", lw=0.9)
axes[1].set_ylabel("population\nrate (Hz/unit)")

axes[2].plot(pupil.restrict(win).times(), pupil.restrict(win).values, color="tab:purple", lw=1.2)
axes[2].set_ylabel("pupil\ndiameter")

axes[3].plot(running.restrict(win).times(), running.restrict(win).values, color="tab:green", lw=1.0)
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")

handles = [plt.Line2D([], [], color=fmap[f], lw=2, label=f"{f/1000:g} kHz") for f in sess["frequencies"]]
axes[0].legend(handles=handles, ncol=5, fontsize=8, loc="upper right",
               frameon=True, framealpha=0.9, title="tone onset", title_fontsize=8)
fig.tight_layout()
fig.savefig("fig01_raw_streams.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------- trial-design summary
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
counts = [int((freqs == f).sum()) for f in sess["frequencies"]]
axes[0].bar(np.arange(len(counts)), counts, color=colors)
axes[0].set_xticks(np.arange(len(counts)))
axes[0].set_xticklabels([f"{f/1000:g}" for f in sess["frequencies"]])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("# trials")
axes[0].set_title("trials per frequency")

isi = np.diff(onsets)
axes[1].hist(isi, bins=60, color="0.4")
axes[1].set_xlabel("inter-tone interval (s)")
axes[1].set_ylabel("count")
axes[1].set_title("stimulus timing")

axes[2].hist(np.asarray(sess["spikes"].rate), bins=40, color="0.4")
axes[2].set_xlabel("mean firing rate (Hz)")
axes[2].set_ylabel("# units")
axes[2].set_title("unit firing rates (n=%d)" % len(sess["spikes"]))
fig.tight_layout()
fig.savefig("fig02_trial_design.png", dpi=150)
plt.close(fig)

# ------------------------------------------------ grand PSTH aligned to onset
import tuning

bins = np.arange(-0.1, 0.3005, 0.005)
grand = np.mean([tuning.psth(np.asarray(sess["spikes"][u].times()), onsets, bins)
                 for u in sess["spikes"].keys()], axis=0)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(bins[:-1] + 0.0025, grand, color="k", lw=1.5)
ax.axvspan(0, 0.025, color="tab:orange", alpha=0.25, label="tone (25 ms)")
ax.axvline(0, color="tab:orange", lw=1)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz/unit)")
ax.set_title("Grand PSTH, all units and all tones")
ax.legend()
fig.tight_layout()
fig.savefig("fig03_grand_psth.png", dpi=150)
plt.close(fig)

peak = (bins[:-1] + 0.0025)[np.argmax(grand)]
base = grand[bins[:-1] < 0].mean()
print(f"baseline {base:.2f} Hz/unit, peak {grand.max():.2f} Hz/unit at {peak*1000:.0f} ms")
