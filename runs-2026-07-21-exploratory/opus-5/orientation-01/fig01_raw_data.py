"""Figure 1: raw data sanity checks for the prototype session."""

import matplotlib.pyplot as plt
import numpy as np

import dandi_io as dio
import plotting as pl
import tuning as tn

SESSION = "sub-699733573/sub-699733573_ses-715093703.nwb"

s = dio.extract_session(SESSION)
units = s["units"]
dg = s["drifting_gratings"]
dg_valid = dg.dropna(subset=["orientation"])

# 60 s window at the start of the first drifting-gratings block
t0 = dg_valid["start_time"].iloc[0] - 5
t1 = t0 + 60

order = []
for grp in pl.REGION_ORDER:
    order += list(units.index[units["region_group"] == grp])
order = np.array(order)

fig = plt.figure(figsize=(11, 8.5))
gs = fig.add_gridspec(3, 3, height_ratios=[2.1, 0.7, 1.5], hspace=0.55, wspace=0.32,
                      left=0.08, right=0.97, top=0.93, bottom=0.08)

# --- A: population raster -------------------------------------------------
ax = fig.add_subplot(gs[0, :])
for row, ui in enumerate(order):
    uid = int(units["unit_id"].iloc[ui])
    st = s["spike_times"][uid]
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st, np.full(len(st), row), "|", ms=1.2, mew=0.4,
            color=pl.REGION_COLORS[units["region_group"].iloc[ui]])
for _, r in dg_valid[(dg_valid.start_time > t0 - 3) & (dg_valid.start_time < t1)].iterrows():
    ax.axvspan(r.start_time, r.stop_time, color="k", alpha=0.06, lw=0)
    ax.text((r.start_time + r.stop_time) / 2, len(order) * 1.02, f"{int(r.orientation)}",
            ha="center", va="bottom", fontsize=6)
ax.set_xlim(t0, t1)
ax.set_ylim(-5, len(order) * 1.09)
ax.set_xlabel("time (s)")
ax.set_ylabel("unit (sorted by region)")
ax.set_title(f"A  Spike raster, session {s['session_id']} — shaded epochs are 2 s drifting gratings "
             "(numbers = drift direction, deg)", loc="left")
handles = [plt.Line2D([], [], color=pl.REGION_COLORS[g], lw=3, label=g) for g in pl.REGION_ORDER]
ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8,
          bbox_to_anchor=(1.0, 1.14))

# --- B: running speed -----------------------------------------------------
ax = fig.add_subplot(gs[1, :])
m = (s["running_time"] >= t0) & (s["running_time"] <= t1)
ax.plot(s["running_time"][m], s["running_speed"][m], color="#444", lw=0.8)
ax.set_xlim(t0, t1)
ax.set_xlabel("time (s)")
ax.set_ylabel("running\n(cm/s)")
ax.set_title("B  Running speed over the same window", loc="left")

# --- C: population PSTHs by region ---------------------------------------
tsg, meta = tn.make_tsgroup(s)
ax = fig.add_subplot(gs[2, 0])
for grp in pl.REGION_ORDER:
    sel = units["region_group"] == grp
    curves = []
    for uid in units.loc[sel, "unit_id"].values[:120]:
        t, r = pl.psth(s["spike_times"][int(uid)], dg_valid["start_time"].values)
        curves.append(r)
    curves = np.array(curves)
    mu = curves.mean(0)
    se = curves.std(0) / np.sqrt(len(curves))
    ax.plot(t, mu, color=pl.REGION_COLORS[grp], lw=1.3, label=f"{grp} (n={len(curves)})")
    ax.fill_between(t, mu - se, mu + se, color=pl.REGION_COLORS[grp], alpha=0.2, lw=0)
ax.axvspan(0, 2, color="k", alpha=0.06, lw=0)
ax.set_xlabel("time from grating onset (s)")
ax.set_ylabel("firing rate (spikes/s)")
ax.set_title("C  Population PSTH,\nall drifting gratings", loc="left")
ax.legend(fontsize=7, loc="upper right")

# --- D: QC metric distributions ------------------------------------------
ax = fig.add_subplot(gs[2, 1])
ax.hist(np.log10(units["firing_rate"].values.astype(float) + 1e-3), bins=30, color="#777")
ax.set_xlabel("log10 session firing rate (spikes/s)")
ax.set_ylabel("units")
ax.set_title(f"D  QC-passing units (n={len(units)})", loc="left")

# --- E: units per area ----------------------------------------------------
ax = fig.add_subplot(gs[2, 2])
vc = units["location"].value_counts()
colors = [pl.REGION_COLORS[dio.region_group(k)] for k in vc.index]
ax.barh(np.arange(len(vc))[::-1], vc.values, color=colors)
ax.set_yticks(np.arange(len(vc))[::-1])
ax.set_yticklabels(vc.index, fontsize=8)
ax.set_xlabel("units")
ax.set_title("E  Recorded areas", loc="left")

fig.savefig("fig01_raw_data.png", bbox_inches="tight")
print("wrote fig01_raw_data.png")
