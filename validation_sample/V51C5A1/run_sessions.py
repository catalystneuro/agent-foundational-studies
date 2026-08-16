"""Run the tuning analysis over every session in DANDI:000986 and cache the result.

Only reduced arrays are kept per session (PSTHs, tuning curves, statistics), plus
the full trial-by-unit count matrix for the one example session used in the
single-session figures.  Output: results.pkl, a few MB.
"""

import pickle
import sys

import numpy as np
from tqdm import tqdm

import dandi_io
import tuning as T

EXAMPLE_SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
POP_SIZES = [1, 2, 5, 10, 20, 40, 80, 160, 240]


def analyse(asset):
    nwb, nwbfile = dandi_io.open_session(asset)
    spikes, onsets, freqs, rates = T.load_session_arrays(nwb, nwbfile)
    res = T.session_response_matrices(spikes, onsets, freqs)
    st = T.unit_statistics(res)

    dec = T.poisson_template_decoder(res["evoked"], freqs)
    rng = np.random.default_rng(0)
    dec_shuf = T.poisson_template_decoder(res["evoked"], rng.permutation(freqs))
    rel, rel_shuf = T.split_half_reliability(res)

    out = dict(st)
    out.update(
        path=asset["path"],
        subject=nwbfile.subject.subject_id,
        session_id=nwbfile.session_id,
        n_trials=len(onsets),
        session_rate=rates,
        psth=res["psth"].astype(np.float32),
        decode_acc=dec["accuracy"],
        decode_conf=dec["confusion"],
        decode_acc_shuffled=dec_shuf["accuracy"],
        decode_vs_size=T.decoding_vs_population_size(res["evoked"], freqs, POP_SIZES),
        reliability=rel,
        reliability_shuffled=rel_shuf,
        reject_tuned=T.fdr_bh(st["p_tuning"]),
        reject_responsive=T.fdr_bh(st["p_responsive"]),
    )
    if asset["path"] == EXAMPLE_SESSION:
        out["evoked"] = res["evoked"]
        out["baseline"] = res["baseline"]
        out["trial_freqs"] = freqs
        out["onsets"] = onsets
    return out


def analyse_with_retry(asset, n_retry=3):
    """HDF5 reads are lazy, so an S3 stall can surface anywhere inside analyse()."""
    for attempt in range(n_retry):
        try:
            return analyse(asset)
        except OSError as err:
            if attempt == n_retry - 1:
                raise
            tqdm.write(f"  retrying {asset['path']} after {type(err).__name__}: {err}")


def main():
    assets = dandi_io.list_assets()
    results = []
    for a in tqdm(assets, desc="sessions"):
        r = analyse_with_retry(a)
        results.append(r)
        tqdm.write(
            f"  {r['path']:42s} units={len(r['unit_ids']):4d} "
            f"trials={r['n_trials']:5d} tuned={r['reject_tuned'].mean():.0%} "
            f"decode={r['decode_acc']:.3f} (shuf {r['decode_acc_shuffled']:.3f})"
        )
        sys.stdout.flush()

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    n_units = sum(len(r["unit_ids"]) for r in results)
    print(f"\n{len(results)} sessions, {n_units} units, "
          f"{sum(r['n_trials'] for r in results)} tone trials -> results.pkl")


if __name__ == "__main__":
    main()
