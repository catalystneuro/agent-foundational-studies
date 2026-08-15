"""Run the place-field pipeline across every linear-track session of DANDI:000044."""

import pickle
import time

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import pf_lib as P
from decode_lib import decode_cv

N_SHUFFLES = 500


def analyse(path, url):
    s = P.load_session(url)
    units = P.track_units(s)
    res = P.classify_place_cells(s, units, n_shuffles=N_SHUFFLES)
    dec, _ = decode_cv(s, units)
    err = np.abs(dec[:, 0] - dec[:, 1])
    rng = np.random.default_rng(0)
    chance = np.abs(dec[:, 0] - rng.permutation(dec[:, 1]))

    stats = pd.concat(res["stats"], names=["direction", "unit"]).reset_index()
    stats["session"] = s["session"]
    stats["subject"] = s["subject"]
    summary = dict(
        path=path, session=s["session"], subject=s["subject"],
        maze=s["maze_name"], track_len=s["track_len"],
        n_units_total=len(s["units"]), n_units_used=len(units),
        n_laps=len(s["laps"]),
        run_time=sum(v.tot_length() for v in res["run"].values()),
        n_place=int(stats["is_place"].sum()),
        n_pairs=len(stats),
        frac_place=float(stats["is_place"].mean()),
        frac_place_strict=float(stats["is_place_strict"].mean()),
        frac_place_any=float(stats.groupby("unit")["is_place"].any().mean()),
        median_err=float(np.median(err)),
        chance_err=float(np.median(chance)),
    )
    tc = {k: v for k, v in res["tc"].items()}
    s["io"].close()
    return summary, stats, tc, dec


if __name__ == "__main__":
    assets = P.get_assets()
    linear = {p: u for p, u in assets.items() if "Circular" not in P.maze_type(u)}
    print("linear-track sessions:", len(linear))

    summaries, all_stats, all_tc, all_dec = [], [], {}, {}
    for path, url in tqdm(sorted(linear.items()), desc="sessions"):
        t = time.time()
        summ, stats, tc, dec = analyse(path, url)
        summ["seconds"] = time.time() - t
        print(summ["session"], "%d/%d place fields, decode %.3f m (chance %.3f), %.0fs"
              % (summ["n_place"], summ["n_pairs"], summ["median_err"],
                 summ["chance_err"], summ["seconds"]))
        summaries.append(summ)
        all_stats.append(stats)
        all_tc[summ["session"]] = tc
        all_dec[summ["session"]] = dec

    summary_df = pd.DataFrame(summaries)
    stats_df = pd.concat(all_stats, ignore_index=True)
    summary_df.to_csv("session_summary.csv", index=False)
    stats_df.to_csv("all_unit_stats.csv", index=False)
    with open("multisession.pkl", "wb") as fh:
        pickle.dump(dict(summary=summary_df, stats=stats_df, tc=all_tc, dec=all_dec), fh)
    print(summary_df[["session", "n_units_used", "n_laps", "frac_place",
                      "frac_place_any", "median_err", "chance_err"]])
