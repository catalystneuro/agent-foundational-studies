"""Stage 2: orientation-selectivity metrics for the prototype session."""

import os
import numpy as np
import pandas as pd

import orientation_lib as ol

SES = "715093703"
d = np.load(f"cache/session_{SES}.npz", allow_pickle=True)

res, tun = ol.analyze_session(
    d["dg_rates"], d["dgb_rates"], d["dg_direction"], d["dg_tf"],
    d["sg_rates"], d["sg_ori"], d["unit_id"], d["region"].astype(str),
)
res["session_id"] = SES
res.to_csv(f"cache/res_{SES}.csv", index=False)
np.savez_compressed(f"cache/tc_{SES}.npz", **tun)

print("\nunits %d | %d visually responsive | %d orientation-selective"
      % (len(res), res.responsive.sum(), res.sig_ori.sum()))
print("\n%-6s %5s %6s %8s %10s %10s" % ("region", "n", "resp", "med gOSI", "frac sig", "med HWHM"))
for reg in ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1", "DG"]:
    s = res[res.region == reg]
    if len(s) < 5:
        continue
    sr = s[s.responsive]
    hw = []
    for j in np.where((s.sig_ori).values)[0]:
        col = np.where(res.unit_id.values == s.unit_id.values[j])[0][0]
        hw.append(ol.fit_von_mises(tun["dirs"], tun["tc"][:, col])[1])
    print("%-6s %5d %6d %8.3f %10.2f %10s"
          % (reg, len(s), len(sr), sr.gosi.median(), s.sig_ori.mean(),
             "%.0f" % np.median(hw) if hw else "-"))

ctx = res[res.region.isin(ol.VISUAL_CORTEX) & res.sig_ori]
print("\nvisual-cortex orientation-selective units: %d" % len(ctx))
print("  split-half |d pref ori|            median %.1f deg (chance 45)"
      % ctx.split_half_diff.median())
print("  drifting vs static |d pref ori|    median %.1f deg (chance 45)"
      % ctx.ori_diff_dg_sg.median())
