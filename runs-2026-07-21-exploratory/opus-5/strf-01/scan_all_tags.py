import json, re, collections
import numpy as np
from tqdm import tqdm
import anf_lib as al

assets = al.list_assets()
rows = []
for a in tqdm(assets, desc="tags"):
    try:
        h = al.open_h5(a["asset_id"])
        tags = [al._s(t) for t in h["units/tag"][:]]
        pre = collections.Counter(t.split("_rep")[0] for t in tags)
        noise = {k: v for k, v in pre.items() if k.upper().startswith("NOISE")}
        rec = dict(asset_id=a["asset_id"], path=a["path"], size=a["size"],
                   n_rows=len(tags), noise=noise,
                   protocols=sorted(set(t.split("_")[0] for t in tags)))
        if noise:
            at = h["analysis/analysis_table"]
            exp = [al._s(e) for e in at["experiment"][:]]
            rec["bf_hz"] = float(dict(zip(exp, at["results_bf"][:])).get("BF", np.nan))
            sr = dict(zip(exp, at["results_sr"][:]))
            thr = dict(zip(exp, at["results_threshold"][:]))
            rec["spont"] = float(np.nanmax(list(sr.values()))) if sr else float("nan")
            fin = [v for v in thr.values() if np.isfinite(v)]
            rec["threshold"] = float(np.nanmin(fin)) if fin else float("nan")
            sub = h["general/subject"]
            rec["subject"] = al._s(sub["subject_id"][()])
            rec["age_days"] = int("".join(c for c in al._s(sub["age"][()]) if c.isdigit()))
            rec["sex"] = al._s(sub["sex"][()])
            sp = h["stimulus/presentation"]
            rec["stim_keys"] = [k for k in sp if k.upper().startswith("NOISE")]
            if rec["stim_keys"]:
                w = sp[rec["stim_keys"][0]]["data"]
                rec["wav_len"] = int(w.shape[0])
        rows.append(rec)
        h.close()
    except (KeyError, OSError) as e:
        print("skip", a["path"], type(e).__name__, e)

json.dump(rows, open("all_tags.json", "w"))
have = [r for r in rows if r["noise"]]
print(f"\n{len(rows)} files scanned, {len(have)} contain a NOISE protocol")
print("noise tag vocabulary:", collections.Counter(k for r in have for k in r["noise"]).most_common(12))
print("reps per noise tag:", collections.Counter(v for r in have for v in r["noise"].values()).most_common(8))
print("subjects with noise:", len(set(r["subject"] for r in have)))
bf = np.array([r["bf_hz"] for r in have])
print("BF range:", np.nanmin(bf), np.nanmax(bf), "n finite", int(np.isfinite(bf).sum()))
