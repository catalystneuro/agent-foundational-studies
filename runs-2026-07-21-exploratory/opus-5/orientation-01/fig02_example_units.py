"""Figure 2: single-unit orientation tuning, drifting and static gratings."""

import matplotlib.pyplot as plt
import numpy as np

import dandi_io as dio
import plotting as pl
import tuning as tn
from run_analysis import DG_WINDOW, SG_WINDOW

SESSION = "sub-699733573/sub-699733573_ses-715093703.nwb"

s = dio.extract_session(SESSION)
tsg, meta = tn.make_tsgroup(s)

dg = s["drifting_gratings"]
rates_dg = tn.trial_rates(tsg, dg, window=DG_WINDOW)
ang_dg, pref_tf, curves_dg, _, tab_dg, rr_dg = tn.tuning_by_preferred_condition(
    rates_dg, dg, "orientation", "temporal_frequency"
)
sel = tn.selectivity_table(
    ang_dg, curves_dg, n_shuffles=500,
    trial_data=(tab_dg["orientation"].values, rr_dg,
                tab_dg["temporal_frequency"].values, pref_tf))
sel["location"] = meta["location"].values
sel["region_group"] = meta["region_group"].values
sel["unit_id"] = meta["unit_id"].values
sel["pref_tf"] = pref_tf

sg = s["static_gratings"]
rates_sg = tn.trial_rates(tsg, sg, window=SG_WINDOW)
ang_sg, pref_sf, curves_sg, _, tab_sg, rr_sg = tn.tuning_by_preferred_condition(
    rates_sg, sg, "orientation", "spatial_frequency"
)
sel_sg = tn.selectivity_table(
    ang_sg, curves_sg, n_shuffles=500, orientation_only=True,
    trial_data=(tab_sg["orientation"].values, rr_sg,
                tab_sg["spatial_frequency"].values, pref_sf))

# three well-tuned cortical units with distinct preferred orientations ...
cort = sel[(sel.region_group == "visual cortex") & (sel.mean_rate > 2)].sort_values("gOSI", ascending=False)
picks = []
for target in [10, 60, 130]:
    cand = cort.iloc[:40]
    cand = cand.iloc[(np.abs(((cand.pref_ori - target + 90) % 180) - 90)).argsort()]
    for i in cand.index:
        if i not in picks:
            picks.append(i)
            break
# ... and a representative, well-isolated hippocampal unit that is not tuned
hipp = sel[(sel.region_group == "hippocampus") & (sel.mean_rate > 3) & (sel.p_gOSI > 0.05)]
picks.append(hipp.sort_values("mean_rate", ascending=False).index[0])

fig = plt.figure(figsize=(13, 12.5))
gs = fig.add_gridspec(4, 4, width_ratios=[1.5, 1, 1, 1], hspace=0.62, wspace=0.42,
                      left=0.06, right=0.97, top=0.90, bottom=0.05)

for row, idx in enumerate(picks):
    info = sel.loc[idx]
    uid = int(info.unit_id)
    st = s["spike_times"][uid]
    color = pl.REGION_COLORS[info.region_group]
    sub = tab_dg.loc[tab_dg["temporal_frequency"].values == info.pref_tf].sort_values("orientation")

    # --- raster sorted by drift direction ---
    ax = fig.add_subplot(gs[row, 0])
    y = 0
    for a in ang_dg:
        ev = sub.loc[sub.orientation == a, "start_time"].values
        x, yy = pl.raster_rows(st, ev, window=(-0.5, 2.5))
        ax.plot(x, yy + y, "|", ms=2.6, mew=0.6, color=color)
        y += len(ev)
        ax.axhline(y - 0.5, color="0.85", lw=0.5)
    ax.set_yticks(np.arange(len(ang_dg)) * len(ev) + len(ev) / 2)
    ax.set_yticklabels([f"{int(a)}°" for a in ang_dg], fontsize=7)
    ax.axvspan(0, 2, color="k", alpha=0.06, lw=0)
    ax.set_ylim(-1, y)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("drift direction")
    ax.set_title(f"unit {uid} — {info.location}\ndrifting gratings, TF {info.pref_tf:g} Hz",
                 loc="left", fontsize=9)

    # --- polar direction tuning (drifting) ---
    ax = fig.add_subplot(gs[row, 1], projection="polar")
    pl.polar_tuning(ax, ang_dg, curves_dg[:, idx], color=color)
    ax.set_title(f"direction tuning\ngOSI {info.gOSI:.2f} (p={info.p_gOSI:.3f}), "
                 f"gDSI {info.gDSI:.2f}", fontsize=8.5, pad=18)

    # --- static gratings: orientation tuning + von Mises fit ---
    info2 = sel_sg.loc[idx]
    ax = fig.add_subplot(gs[row, 2])
    ax.plot(ang_sg, curves_sg[:, idx], "o", color=color, ms=4)
    fit = tn.fit_von_mises(ang_sg, curves_sg[:, idx])
    xx = np.linspace(0, 180, 200)
    ax.plot(xx, tn.von_mises_ori(xx, fit["r0"], fit["amp"], fit["kappa"], fit["mu"]), "k-", lw=1.1)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel("rate (spikes/s)")
    ax.set_title(f"static gratings, SF {pref_sf[idx]:g} cpd\ngOSI {info2.gOSI:.2f} "
                 f"(p={info2.p_gOSI:.3f}), HWHM {fit['hwhm']:.0f}°", fontsize=8.5)

    # --- orientation-domain polar view (static) ---
    ax = fig.add_subplot(gs[row, 3], projection="polar")
    pl.polar_tuning(ax, ang_sg, curves_sg[:, idx], color=color, period=180)
    ax.set_thetagrids(np.arange(0, 360, 60), [f"{int(a)}°" for a in np.arange(0, 180, 30)])
    ax.set_title(f"static-grating orientation\n(axes doubled; pref {info2.pref_ori:.0f}°)",
                 fontsize=8.5, pad=18)

fig.suptitle("Single-unit orientation tuning in DANDI:000021, session 715093703\n"
             "rows 1–3: visual cortex   |   row 4: representative hippocampal unit (control)",
             y=0.965, fontsize=11)
fig.savefig("fig02_example_units.png", bbox_inches="tight")
print("wrote fig02_example_units.png")
