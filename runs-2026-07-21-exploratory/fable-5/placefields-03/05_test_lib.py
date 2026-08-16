"""Smoke-test the loading / preprocessing library on the Achilles session."""

import numpy as np
import pynapple as nap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import place_cell_lib as pcl

sess = pcl.load_session(pcl.SESSIONS[0][0])
print("session:", sess["session_id"], "subject:", sess["subject"])
print("fs =", round(sess["fs"], 3), " track extent =", round(sess["extent"], 3), "m")
print("pos_ep: n =", len(sess["pos_ep"]), " total =", round(sess["pos_ep"].tot_length(), 1), "s")
sp = sess["speed"].values
print("speed m/s: median", round(float(np.median(sp)), 3),
      "p95", round(float(np.percentile(sp, 95)), 3),
      "p99.9", round(float(np.percentile(sp, 99.9)), 3), "max", round(float(sp.max()), 3))

laps = pcl.lap_epochs(sess)
mov = pcl.moving_epochs(sess, laps)
for k in ["rightward", "leftward", "both"]:
    d = laps[k].end - laps[k].start
    print(f"  laps {k}: n={len(laps[k])}, total={laps[k].tot_length():.1f}s, "
          f"median dur={np.median(d):.2f}s, p90={np.percentile(d, 90):.2f}s "
          f"| moving total={mov[k].tot_length():.1f}s")

spikes = sess["spikes"]
exc = spikes[np.where(spikes.get_info("cell_type") == "excitatory")[0]]

tc_r = pcl.place_fields(exc, sess["position"], mov["rightward"], sess["extent"], fs=sess["fs"])
tc_l = pcl.place_fields(exc, sess["position"], mov["leftward"], sess["extent"], fs=sess["fs"])
occ_r = np.asarray(tc_r.attrs["occupancy"]) / sess["fs"]
print("\noccupancy per bin (s), rightward:", np.round(occ_r, 1))
print("sum:", round(occ_r.sum(), 1), " vs moving rightward:", round(mov['rightward'].tot_length(), 1))

rates = np.asarray(tc_r.data)
si_r = nap.compute_mutual_information(tc_r)
print("\nspatial information (rightward):")
print(si_r.describe().round(3))
print("peak rate: median", round(float(np.median(rates.max(axis=1))), 2),
      "max", round(float(rates.max()), 2))

# --- validation figure ---
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
p = sess["position"]
axs[0].plot(p.t, p.values, ".", ms=1.5, color="0.3")
for s, e in zip(laps["rightward"].start, laps["rightward"].end):
    axs[0].axvspan(s, e, color="tab:red", alpha=0.25, lw=0)
for s, e in zip(laps["leftward"].start, laps["leftward"].end):
    axs[0].axvspan(s, e, color="tab:blue", alpha=0.25, lw=0)
axs[0].set_ylabel("position (m)")
axs[0].set_title("Linearized position; detected rightward (red) / leftward (blue) laps")

axs[1].plot(sess["speed"].t, sess["speed"].values, lw=0.7)
axs[1].axhline(pcl.SPEED_THRESHOLD, color="r", ls="--")
axs[1].set_ylabel("speed (m/s)")
axs[1].set_ylim(0, 1.6)

order = np.argsort(rates.argmax(axis=1))
uids = np.asarray(tc_r.coords["unit"].values)[order]
win = nap.IntervalSet(18150, 18400)
for row, uu in enumerate(uids):
    st = exc[uu].restrict(win).t
    axs[2].plot(st, np.full_like(st, row), "|", ms=3, color="k", alpha=0.7)
axs[2].set_ylabel("unit (sorted by field position)")
axs[2].set_xlabel("time (s)")
axs[2].set_xlim(18150, 18400)
plt.tight_layout()
plt.savefig("fig_lib_smoketest.png", dpi=130)
print("saved fig_lib_smoketest.png")
