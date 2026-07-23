"""Load one session, validate every data stream, and plot raw activity."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import aud_common as ac

assets = ac.list_assets()
print(assets[["path", "subject", "session", "asset_id"]].to_string())

row = assets[assets.session_name == "LA11_ses1"].iloc[0]
nwb, nwbfile = ac.load_session(row.asset_id)
print(nwb)

units = nwb["units"]
print("n units:", len(units))
print("rate range: %.2f - %.2f Hz" % (units.rate.min(), units.rate.max()))

tr = ac.trial_table(nwbfile)
print(tr.head())
print("frequencies (kHz):", np.unique(tr.freq_khz.values))
print("amplitudes (dB):", np.unique(tr.stim_amplitude.values))
print("durations (s):", np.unique(tr.stim_duration.values))
print("trials per frequency:\n", tr.freq_khz.value_counts().sort_index())

onsets = tr.start_time.values
tone_ep = nap.IntervalSet(start=onsets.min() - 1, end=onsets.max() + 1)
spont = nwb["spontaneous_blocks"]
print("spontaneous blocks:\n", spont)
print("tone block: %.1f - %.1f s" % (tone_ep.start[0], tone_ep.end[0]))

pupil = nwb["pupil_diameter"]
run = nwb["running_speed"]
print("pupil:", pupil.shape, "nan frac %.3f" % np.mean(np.isnan(pupil.d)))
print("running:", run.shape, "nan frac %.3f" % np.mean(np.isnan(run.d)))

# --- Figure 1: raw data overview -------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})

t0, t1 = onsets[200], onsets[200] + 20.0
win = nap.IntervalSet(start=t0, end=t1)
for i, key in enumerate(list(units.keys())[:80]):
    st = units[key].restrict(win).t
    axes[0].plot(st, np.full_like(st, i), "|", color="k", ms=3, mew=0.6)
axes[0].set_ylabel("unit #")
axes[0].set_title("DANDI:000986  %s  raw spike raster (80 units) with tone onsets"
                  % row.session_name)

sub = tr[(tr.start_time >= t0) & (tr.start_time <= t1)]
cmap = plt.get_cmap("viridis")
ufreq = np.unique(tr.freq_khz.values)
for _, s in sub.iterrows():
    c = cmap(np.log2(s.freq_khz / ufreq[0]) / np.log2(ufreq[-1] / ufreq[0]))
    axes[0].axvline(s.start_time, color=c, alpha=0.5, lw=1.2)

pop = units.count(0.010, ep=win).sum(axis=1) / 0.010 / len(units)
axes[1].plot(pop.t, pop.d, color="firebrick", lw=0.8)
axes[1].set_ylabel("pop. rate\n(Hz/unit)")

p = pupil.restrict(win)
axes[2].plot(p.t, p.d, color="purple", lw=1)
axes[2].set_ylabel("pupil\n(norm.)")

r = run.restrict(win)
axes[3].plot(r.t, r.d, color="teal", lw=1)
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig01_raw_data_overview.png", dpi=150)
print("saved fig01")

# --- Figure 2: session-scale behaviour + firing --------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
whole = nap.IntervalSet(start=0, end=float(nwbfile.trials.stop_time[-1]) + 400)
pr = units.count(5.0, ep=whole).sum(axis=1) / 5.0 / len(units)
axes[0].plot(pr.t, pr.d, color="firebrick", lw=0.7)
axes[0].set_ylabel("pop. rate\n(Hz/unit)")
axes[0].set_title("%s  session timeline (5 s bins)" % row.session_name)
axes[1].plot(pupil.t[::50], pupil.d[::50], color="purple", lw=0.6)
axes[1].set_ylabel("pupil (norm.)")
axes[2].plot(run.t[::50], run.d[::50], color="teal", lw=0.6)
axes[2].set_ylabel("running (cm/s)")
axes[2].set_xlabel("time (s)")
for a in axes:
    for s, e in zip(spont.start, spont.end):
        a.axvspan(s, e, color="0.85", zorder=0)
    a.spines[["top", "right"]].set_visible(False)
axes[0].text(0.01, 0.9, "grey = spontaneous (no tones)", transform=axes[0].transAxes)
fig.tight_layout()
fig.savefig("fig02_session_timeline.png", dpi=150)
print("saved fig02")
