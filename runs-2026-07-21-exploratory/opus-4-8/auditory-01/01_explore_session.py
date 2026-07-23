"""Prototype: load one session, verify each data stream, plot raw activity."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import common
import analysis

SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
s = common.load_session(SESSION)
print(f"{s['subject']} ses-{s['session_id']}: {len(s['units'])} units, "
      f"{len(s['frequency'])} tones")
freqs = np.unique(s["frequency"])
print("frequencies (Hz):", freqs)

# ---------------------------------------------------------------- raw activity
t0 = s["trials"]["start_time"].iloc[0] + 100.0
dur = 8.0
win = nap.IntervalSet(t0, t0 + dur)
colors = dict(zip(freqs, plt.get_cmap("viridis")(np.linspace(0, 0.92, len(freqs)))))

fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                       gridspec_kw=dict(height_ratios=[3.2, 1.1, 1, 1], hspace=0.2))
raster = s["units"].restrict(win)
order = np.argsort(s["units"].get_info("rate").values)
for i, k in enumerate(np.array(list(s["units"].keys()))[order]):
    ts = raster[k].t
    ax[0].plot(ts, np.full_like(ts, i), "|", color="k", ms=2.2, mew=0.6)
tone_t = s["trials"].query("@t0 <= start_time <= @t0+@dur")
for _, tr in tone_t.iterrows():
    ax[0].axvspan(tr.start_time, tr.start_time + s["duration"],
                  color=colors[tr.stim_frequency], alpha=0.45, lw=0, zorder=0)
ax[0].set_ylabel("unit (sorted by rate)")
ax[0].set_title(f"{s['subject']} session {s['session_id']}: spike raster of "
                f"{len(s['units'])} auditory-cortex units; shaded bars = 25 ms pure tones")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f], alpha=0.6) for f in freqs]
ax[0].legend(handles, [f"{int(f/1000)} kHz" for f in freqs], ncol=5,
             loc="lower left", fontsize=8, framealpha=0.92, borderpad=0.4)

pop = s["units"].count(0.005, win).sum(1) / (0.005 * len(s["units"]))
ax[1].plot(pop.t, pop.d, color="tab:blue", lw=0.9)
for _, tr in tone_t.iterrows():
    ax[1].axvspan(tr.start_time, tr.start_time + s["duration"],
                  color=colors[tr.stim_frequency], alpha=0.35, lw=0, zorder=0)
ax[1].set_ylabel("population rate\n(spikes/s/unit)")

ax[2].plot(s["pupil"].restrict(win).t, s["pupil"].restrict(win).d, color="tab:purple")
ax[2].set_ylabel("pupil\n(a.u.)")
ax[3].plot(s["running"].restrict(win).t, s["running"].restrict(win).d, color="tab:green")
ax[3].set_ylabel("running\n(cm/s)")
ax[3].set_xlabel("time (s)")
fig.savefig("fig01_raw_activity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- perievent PSTH sanity
tuning, sem, driven, st, _ = analysis.session_tuning(
    s["units"], s["trials"]["start_time"].values, s["frequency"])
sig = analysis.fdr(st["p_freq"].values)
print(f"frequency-selective units (Kruskal-Wallis, FDR 0.05): {sig.sum()}/{len(sig)}")
st.to_csv("session_stats_LA11_1.csv")
tuning.to_csv("session_tuning_LA11_1.csv")

# cross-check the searchsorted counts against pynapple's epoch-response helper
eps = {f"{int(f)}Hz": nap.IntervalSet(
        start=s["trials"].start_time.values[s["frequency"] == f] + analysis.RESPONSE[0],
        end=s["trials"].start_time.values[s["frequency"] == f] + analysis.RESPONSE[1])
       for f in freqs}
nap_resp = nap.compute_response_per_epoch(s["units"], eps, return_pandas=True)
print("max |pynapple - searchsorted| driven rate:",
      np.abs(nap_resp.values.T - driven.values).max())
