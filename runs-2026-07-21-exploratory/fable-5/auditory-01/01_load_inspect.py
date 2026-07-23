"""Load one session from DANDI 000986 and visually verify every data stream."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dandi_auditory as da

assets = da.list_assets()
print(assets.to_string())

# Prototype session: the one with the most units.
asset_id = assets.query("subject == 'LA11' and session == '1'")["asset_id"].iloc[0]
print("loading", asset_id)

s = da.load_session(asset_id, with_behavior=True)
print(s["units"])
print("subject", s["subject"], "session", s["session_id"])
print("n trials", len(s["trials"]), "freqs", s["freqs"], "tone dur", s["tone_duration"])
print("spontaneous blocks:\n", s["spont"])
print("pupil", s["pupil"].shape, "speed", s["speed"].shape)

rates = s["units"].get_info("rate")
print("firing rate: median %.2f Hz, range %.3f-%.1f" % (np.median(rates), rates.min(), rates.max()))

# ---------------------------------------------------------------- raw overview
fig, axes = plt.subplots(
    4, 1, figsize=(12, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1, 0.6], "hspace": 0.15},
)

t0 = float(s["trials"].start[0])
win = (t0 - 2, t0 + 12)

# spike raster for a subset of units
sub = list(s["units"].keys())[:60]
for i, u in enumerate(sub):
    st = s["units"][u].get(win[0], win[1]).t
    axes[0].plot(st, np.full_like(st, i), "|", color="k", ms=2.5, mew=0.5)
axes[0].set_ylabel("unit #")
axes[0].set_title(
    f"DANDI 000986 sub-{s['subject']} ses-{s['session_id']}: raw spiking, pupil and locomotion "
    f"around tone onset (25 ms pure tones, 60 dB SPL)"
)

# tone onsets colour-coded by frequency
for f, c in zip(s["freqs"], da.FREQ_COLORS):
    on = s["trials"].start[(s["frequency"] == f)]
    on = on[(on > win[0]) & (on < win[1])]
    axes[1].vlines(on, 0, 1, color=c, lw=2, label=f"{f/1000:g} kHz")
axes[1].set_ylim(0, 1)
axes[1].set_yticks([])
axes[1].set_ylabel("tone")
axes[1].legend(ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)

pup = s["pupil"].get(win[0], win[1])
axes[2].plot(pup.t, pup.d, color="#AA4499", lw=1)
axes[2].set_ylabel("pupil\n(frac. max)")

spd = s["speed"].get(win[0], win[1])
axes[3].plot(spd.t, spd.d, color="#44AA99", lw=1)
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")
axes[3].set_xlim(*win)

fig.savefig("fig01_raw_data_streams.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- session-level sanity checks
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(np.log10(rates + 1e-3), bins=40, color="#4477AA")
axes[0].set_xlabel("log10 firing rate (Hz)")
axes[0].set_ylabel("# units")
axes[0].set_title(f"{len(s['units'])} units")

isi = np.diff(s["trials"].start)
axes[1].hist(isi[isi < 2], bins=50, color="#4477AA")
axes[1].set_xlabel("inter-tone interval (s)")
axes[1].set_ylabel("# trials")
axes[1].set_title("stimulus timing (%d gaps >2 s excluded)" % np.sum(isi >= 2), fontsize=10)

counts = [np.sum(s["frequency"] == f) for f in s["freqs"]]
axes[2].bar([f"{f/1000:g}" for f in s["freqs"]], counts, color=da.FREQ_COLORS)
axes[2].set_xlabel("tone frequency (kHz)")
axes[2].set_ylabel("# trials")
axes[2].set_title("trials per frequency")
fig.tight_layout()
fig.savefig("fig02_session_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("wrote fig01_raw_data_streams.png, fig02_session_quality.png")
