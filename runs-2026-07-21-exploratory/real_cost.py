#!/usr/bin/env python
"""Extract real OpenRouter cost and served-provider provenance per run.

For every genuine run (>15-turn success) in the open-weights lanes, collect the
OpenRouter generation ids from its transcript (the assistant message ids are
gen-...), query each id's real billed cost and serving provider, and write a
per-run CSV. This is the authoritative cost, unlike Claude Code's imputed figure.

Env: OPENROUTER_API_KEY. Usage: real_cost.py <lane-dir> [<lane-dir> ...]
"""
import os, sys, re, json, glob, time, collections, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["OPENROUTER_API_KEY"]
GEN = re.compile(r"gen-\d+-[A-Za-z0-9]+")

def gen_ids(path):
    ids = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        for m in GEN.findall(line):
            ids.append(m)
    return sorted(set(ids))

def query(gid, tries=4):
    url = f"https://openrouter.ai/api/v1/generation?id={gid}"
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read()).get("data") or {}
                return {"cost": d.get("total_cost") or 0.0, "provider": d.get("provider_name") or "?"}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and a < tries - 1:
                time.sleep(1.5 * (a + 1)); continue
            return None
        except Exception:
            if a < tries - 1:
                time.sleep(1.0); continue
            return None
    return None

def genuine(t):
    r = None
    for line in open(t, encoding="utf-8", errors="ignore"):
        if '"type":"result"' in line:
            try:
                e = json.loads(line)
                if e.get("type") == "result":
                    r = e
            except Exception:
                pass
    return r and r.get("subtype") == "success" and not r.get("is_error") and (r.get("num_turns") or 0) > 15

lanes = sys.argv[1:]
out = open("real_cost.csv", "w")
out.write("lane,run,n_requests,n_priced,real_cost_usd,top_provider,provider_mix\n")
for lane in lanes:
    for t in sorted(glob.glob(f"{lane}/*-0*/transcript.jsonl")):
        if not genuine(t):
            continue
        run = os.path.basename(os.path.dirname(t))
        ids = gen_ids(t)
        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(query, ids):
                results.append(res)
        priced = [r for r in results if r]
        cost = sum(r["cost"] for r in priced)
        prov = collections.Counter(r["provider"] for r in priced)
        top = prov.most_common(1)[0][0] if prov else "?"
        mix = " ".join(f"{p}:{n}" for p, n in prov.most_common())
        out.write(f"{lane},{run},{len(ids)},{len(priced)},{cost:.4f},{top},{mix}\n")
        out.flush()
        print(f"{lane}/{run}: ${cost:.3f} over {len(priced)}/{len(ids)} requests, providers: {mix}", flush=True)
out.close()
print("wrote real_cost.csv")
