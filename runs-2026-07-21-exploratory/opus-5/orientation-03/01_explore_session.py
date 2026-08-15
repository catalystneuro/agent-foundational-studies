"""Stage 1: load one session, inspect every data stream, and cache trial rates."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap

import orientation_lib as ol

os.makedirs("figures", exist_ok=True)
os.makedirs("cache", exist_ok=True)

sessions = ol.list_session_assets()
print(f"{len(sessions)} session-level NWB files in DANDI:{ol.DANDISET}")
print(sessions[["subject_id", "session_id", "size"]].head(20).to_string())

SES = "715093703"
row = sessions[sessions.session_id == SES].iloc[0]
print("\nprototype session:", row["path"])

f = ol.open_session(row["url"])
print("session_description:", f["session_description"][()].decode())
print("genotype:", f["general"]["subject"]["genotype"][()].decode())
print("age:", f["general"]["subject"]["age"][()].decode(),
      "sex:", f["general"]["subject"]["sex"][()].decode())

# ---------------------------------------------------------------- units ----
units = ol.unit_table(f)
print("\nunits: %d total, %d pass QC" % (len(units), units.pass_qc.sum()))
print(units[units.pass_qc].region.value_counts().to_string())

keep_regions = ol.VISUAL_CORTEX + ol.THALAMUS + ol.CONTROL
sel = units[units.pass_qc & units.region.isin(keep_regions)].reset_index(drop=True)
print("\nunits kept for analysis:", len(sel))

# ------------------------------------------------------------ stimuli ----
dg, dg_blank = ol.drifting_gratings(f)
sg, sg_blank = ol.static_gratings(f)
print("\ndrifting gratings: %d trials, %d blank sweeps" % (len(dg), len(dg_blank)))
print(dg.groupby(["direction", "temporal_frequency"]).size().unstack().to_string())
print("trial duration: %.2f s" % dg.duration.median())
print("\nstatic gratings: %d trials, %d blank" % (len(sg), len(sg_blank)))
print(sg.ori.value_counts().sort_index().to_string())
print("trial duration: %.3f s" % sg.duration.median())

# ------------------------------------------------------------- spikes ----
t_end = max(dg.stop_time.max(), sg.stop_time.max()) + 60
ep = nap.IntervalSet(start=0.0, end=t_end)
spikes, sel = ol.load_spikes(f, sel, time_support=ep)
spikes.set_info(region=np.asarray(sel.region.values), snr=np.asarray(sel.snr.values))
print("\n", spikes)

# --------------------------------------------------------- running speed ----
run = f["processing"]["running"]["running_speed"]
speed = nap.Tsd(t=run["timestamps"][:], d=run["data"][:], time_support=ep)
print("running speed: n=%d, median %.1f cm/s" % (len(speed), np.median(speed.d)))

# ============================ FIGURE 1: raw data ============================
vis = np.where(np.isin(spikes.region, ol.VISUAL_CORTEX))[0]
all_keys = list(spikes.keys())
vis_keys = [all_keys[i] for i in vis]
# show a readable subset: 30 units spanning the firing-rate range
rates_all = np.array([spikes[k].rate for k in vis_keys])
# sample 30 units spanning the firing-rate range rather than the 30 fastest,
# so the raster stays legible
srt = np.argsort(rates_all)[::-1]
vis_keys = [vis_keys[i] for i in srt[np.linspace(0, len(srt) - 1, 30).astype(int)]]
t0 = dg.start_time.iloc[0] - 3
t1 = dg.start_time.iloc[0] + 27
win = nap.IntervalSet(start=t0, end=t1)

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 0.8, 1], hspace=0.18))

ax = axes[0]
for i, k in enumerate(vis_keys):
    t = spikes[k].restrict(win).t
    ax.plot(t, np.full_like(t, i), "|", ms=4, color="k", alpha=0.85, mew=0.7)
ax.set_ylabel("visual cortex unit")
ax.set_title(f"DANDI:000021 session {SES} - raw spiking during drifting gratings")
ax.set_ylim(-1, len(vis_keys))

ax = axes[1]
sub = dg[(dg.start_time < t1) & (dg.stop_time > t0)]
for _, r in sub.iterrows():
    ax.axvspan(r.start_time, r.stop_time, color="C0", alpha=0.25)
    ax.text(r.start_time + 1, 0.5, "%d" % r.direction, ha="center", va="center", fontsize=8)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("grating\ndirection (deg)")

ax = axes[2]
s = speed.restrict(win)
ax.plot(s.t, s.d, lw=0.8, color="C3")
ax.set_ylabel("running\n(cm/s)")
ax.set_xlabel("time (s)")
plt.savefig("figures/fig01_raw_data.png", dpi=150, bbox_inches="tight")
plt.close()

# ===================== trial rates and caching =====================
dg_rates, dg_counts = ol.trial_rates(spikes, dg)
sg_rates, sg_counts = ol.trial_rates(spikes, sg)
dgb_rates, _ = ol.trial_rates(spikes, dg_blank)
print("\nDG rate matrix", dg_rates.shape, "SG rate matrix", sg_rates.shape)
print("mean evoked rate (all units): %.2f Hz; blank-sweep rate %.2f Hz"
      % (dg_rates.mean(), dgb_rates.mean()))

np.savez_compressed(
    f"cache/session_{SES}.npz",
    dg_rates=dg_rates, dg_counts=dg_counts,
    sg_rates=sg_rates, sg_counts=sg_counts,
    dgb_rates=dgb_rates,
    dg_direction=dg.direction.values, dg_tf=dg.temporal_frequency.values,
    dg_start=dg.start_time.values, dg_stop=dg.stop_time.values,
    sg_ori=sg.ori.values, sg_sf=sg.spatial_frequency.values,
    sg_start=sg.start_time.values,
    unit_id=sel.unit_id.values, region=sel.region.values.astype(str),
    snr=sel.snr.values, wf_duration=sel.waveform_duration.values,
)
sel.to_csv(f"cache/units_{SES}.csv", index=False)
print("saved cache/session_%s.npz" % SES)
