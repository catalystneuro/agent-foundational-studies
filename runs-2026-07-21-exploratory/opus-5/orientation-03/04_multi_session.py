"""Stage 4: run the pipeline over multiple sessions and pool the units."""

import os
import sys
import numpy as np
import pandas as pd

import orientation_lib as ol

N_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.makedirs("cache", exist_ok=True)

sessions = ol.list_session_assets().head(N_SESSIONS)
print("processing %d sessions" % len(sessions))

all_res, all_tc, all_tcb = [], [], []
for _, row in sessions.iterrows():
    ses = row["session_id"]
    out = f"cache/res_{ses}.csv"
    if os.path.exists(out):
        print("cached:", ses)
        r = pd.read_csv(out)
        z = np.load(f"cache/tc_{ses}.npz")
        all_res.append(r)
        all_tc.append(z["tc"])
        all_tcb.append(z["tc_b_aligned"])
        continue

    print("\n=== session %s (%s) ===" % (ses, row["path"]))
    D = ol.load_and_prepare(row["url"])
    print("  %d QC units, %d DG trials, %d SG trials"
          % (len(D["units"]), len(D["dg"]), len(D["sg"])))
    res, tun = ol.analyze_session(
        D["dg_rates"], D["dgb_rates"], D["dg"].direction.values,
        D["dg"].temporal_frequency.values, D["sg_rates"], D["sg"].ori.values,
        D["units"].unit_id.values, D["units"].region.values,
    )
    res["session_id"] = ses
    res["subject_id"] = row["subject_id"]
    res.to_csv(out, index=False)
    np.savez_compressed(f"cache/tc_{ses}.npz", **tun)
    all_res.append(res)
    all_tc.append(tun["tc"])
    all_tcb.append(tun["tc_b_aligned"])
    ctx = res[res.region.isin(ol.VISUAL_CORTEX)]
    print("  cortex: %d units, %d responsive, %d orientation-selective (%.0f%%)"
          % (len(ctx), ctx.responsive.sum(), ctx.sig_ori.sum(),
             100 * ctx.sig_ori.mean()))
    del D

pooled = pd.concat(all_res, ignore_index=True)
pooled.to_csv("cache/pooled_results.csv", index=False)
np.savez_compressed("cache/pooled_tuning.npz",
                    tc=np.hstack(all_tc), tc_b_aligned=np.hstack(all_tcb))
print("\npooled: %d units from %d sessions / %d mice"
      % (len(pooled), pooled.session_id.nunique(), pooled.subject_id.nunique()))
summ = (pooled.groupby("region")
        .agg(n=("gosi", "size"), n_resp=("responsive", "sum"),
             frac_sig=("sig_ori", "mean"),
             gosi_med=("gosi", "median"))
        .sort_values("frac_sig", ascending=False))
print(summ.to_string())
