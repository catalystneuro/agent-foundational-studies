"""Analyze a single session (by index into sessions.json) and save res_<idx>.pkl.

Run under an external `timeout` so a stalled network stream is bounded and the
other sessions still complete. Resumable: skips if the output already exists.
"""
import sys, json, pickle, os
from run_analysis import analyze_session

idx = int(sys.argv[1])
out = f"res_{idx}.pkl"
if os.path.exists(out):
    print(f"[{idx}] already done, skipping", flush=True)
    sys.exit(0)

sess = json.load(open("sessions.json"))[idx]
print(f"[{idx}] streaming {sess['path'].split('/')[-1][:45]}", flush=True)
r = analyze_session(sess["url"], sess["path"])
pickle.dump(r, open(out, "wb"))
print(f"[{idx}] DONE block AUC={r['block']['auc']:.3f} nullmu={r['block']['null'].mean():.3f} "
      f"p={r['block']['p']:.4f} units={r['n_units']} trials={r['n_trials']} "
      + (f"choiceAUC={r['choice']['auc']:.3f}" if r.get('choice') else "nochoice"), flush=True)
