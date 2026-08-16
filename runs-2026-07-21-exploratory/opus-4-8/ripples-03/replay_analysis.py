"""Full replay analysis over POST-sleep ripples: decode, score, shuffle-test."""
import numpy as np
import warnings
import pynapple as nap
from tqdm import tqdm
warnings.filterwarnings("ignore")
from load_data import load
from place_fields import build_pynapple, running_intervals
from replay import build_template, score_ripple, shuffle_scores, bin_spikes, bayes_decode, weighted_corr

N_SHUF = 250
DT = 0.02
SEED = 1


def main():
    d = load()
    tsg, pos, epochs = build_pynapple(d)
    posm = pos.restrict(epochs["MazeEpoch"])
    run_ep, vel = running_intervals(posm)
    rate, cpos, uids = build_template(tsg, posm, run_ep)
    spike_list = {i: np.asarray(d["spikes"][i], dtype=float) for i in range(len(d["spikes"]))}

    rp = np.load("ripples_post.npz")
    starts, stops, peaks = rp["start"], rp["stop"], rp["peak"]
    rng = np.random.default_rng(SEED)

    rec = []   # per scored ripple
    events = []  # rich records for example plotting (significant, high |r|)
    for s, e, p in tqdm(list(zip(starts, stops, peaks)), desc="replay scoring"):
        t0, t1 = min(s, p - 0.05), max(e, p + 0.05)
        r = score_ripple(spike_list, uids, rate, cpos, t0, t1, dt=DT)
        if r is None:
            continue
        shuf = shuffle_scores(spike_list, uids, rate, cpos, t0, t1, DT, N_SHUF, rng)
        pval = (np.sum(np.abs(shuf) >= np.abs(r["r"])) + 1) / (N_SHUF + 1)
        rec.append((p, r["r"], r["n_active"], pval, t1 - t0))
        if pval < 0.05 and abs(r["r"]) > 0.6 and len(events) < 400:
            events.append(dict(peak=p, t0=t0, t1=t1, r=r["r"], pval=pval,
                               post=r["post"], tc=r["tc"], n_active=r["n_active"]))

    rec = np.array(rec)
    peak, rr, nact, pv, dur = rec.T
    n_sig = int((pv < 0.05).sum())
    print(f"scored {len(rec)} ripples; significant replay (p<0.05): {n_sig} "
          f"({100*n_sig/len(rec):.1f}%)")
    print(f"mean |r| observed: {np.mean(np.abs(rr)):.3f}")

    # pooled shuffle |r| for the same ripples (one shuffle draw each) as null
    null_r = []
    rng2 = np.random.default_rng(SEED + 7)
    for s, e, p in zip(starts, stops, peaks):
        t0, t1 = min(s, p - 0.05), max(e, p + 0.05)
        rr0 = score_ripple(spike_list, uids, rate, cpos, t0, t1, dt=DT)
        if rr0 is None:
            continue
        sh = shuffle_scores(spike_list, uids, rate, cpos, t0, t1, DT, 1, rng2)
        null_r.append(sh[0])
    null_r = np.array(null_r)

    np.savez("replay_results.npz",
             peak=peak, r=rr, n_active=nact, pval=pv, dur=dur,
             null_r=null_r, rate=rate, cpos=cpos)
    # save a handful of example events (sort by |r|), keep object arrays
    events = sorted(events, key=lambda x: -abs(x["r"]))[:12]
    np.savez("replay_events.npz",
             events=np.array(events, dtype=object), cpos=cpos)
    print("saved replay_results.npz and", len(events), "example events")


if __name__ == "__main__":
    main()
