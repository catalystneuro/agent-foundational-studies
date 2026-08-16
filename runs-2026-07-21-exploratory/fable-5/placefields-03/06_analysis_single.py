"""Prototype the full place-cell analysis on one session."""

import time
import numpy as np
import pynapple as nap
import place_cell_lib as pcl

t0 = time.time()
sess = pcl.load_session(pcl.SESSIONS[0][0])
laps = pcl.lap_epochs(sess)
mov = pcl.moving_epochs(sess, laps)
print("laps:", laps["n_laps"], "moving time:",
      {k: round(mov[k].tot_length(), 1) for k in ("rightward", "leftward")})

spikes = sess["spikes"]
exc = spikes[np.where(spikes.get_info("cell_type") == "excitatory")[0]]
inh = spikes[np.where(spikes.get_info("cell_type") == "inhibitory")[0]]
print(f"{len(exc)} excitatory, {len(inh)} inhibitory units")

res = {}
for d in ("rightward", "leftward"):
    res[d] = pcl.spatial_info_with_shuffle(
        exc, sess["position"], mov[d], sess["extent"], sess["fs"], n_shuffles=500
    )
    r, _, _ = pcl.split_half_stability(exc, sess["position"], mov[d],
                                       sess["extent"], sess["fs"])
    res[d]["stability"] = r
    print(f"\n--- {d} ---")
    print("  SI bits/spike: median %.3f, max %.3f" % (np.median(res[d]["si"]), res[d]["si"].max()))
    print("  null SI      : median %.3f, 95th %.3f"
          % (np.median(res[d]["null"]), np.percentile(res[d]["null"], 95)))
    print("  p<0.05       : %d / %d units" % ((res[d]["pvals"] < 0.05).sum(), len(exc)))
    print("  peak rate    : median %.2f Hz, max %.2f Hz"
          % (np.median(res[d]["maps"].max(axis=1)), res[d]["maps"].max()))
    print("  stability r  : median %.3f" % np.nanmedian(r))

# Place-cell criterion: significant spatial information, a real peak, and enough spikes
def is_place_cell(r):
    return (r["pvals"] < 0.05) & (r["maps"].max(axis=1) >= 1.0) & (r["n_spikes"] >= 50)

pc_r, pc_l = is_place_cell(res["rightward"]), is_place_cell(res["leftward"])
pc_any = pc_r | pc_l
print(f"\nplace cells: rightward {pc_r.sum()}, leftward {pc_l.sum()}, "
      f"either {pc_any.sum()} / {len(exc)} excitatory ({100*pc_any.mean():.0f}%)")
print(f"both directions: {(pc_r & pc_l).sum()}")

# Interneuron comparison
res_inh = pcl.spatial_info_with_shuffle(inh, sess["position"], mov["rightward"],
                                        sess["extent"], sess["fs"], n_shuffles=200)
print("interneurons: SI median %.3f, p<0.05 in %d/%d"
      % (np.median(res_inh["si"]), (res_inh["pvals"] < 0.05).sum(), len(inh)))

print("\nelapsed %.1f s" % (time.time() - t0))
np.savez("prototype_results.npz",
         si_r=res["rightward"]["si"], si_l=res["leftward"]["si"],
         p_r=res["rightward"]["pvals"], p_l=res["leftward"]["pvals"])
