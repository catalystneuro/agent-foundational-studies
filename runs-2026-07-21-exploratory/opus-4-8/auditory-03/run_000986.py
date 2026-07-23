"""Run the frequency-tuning analysis across all 15 sessions of DANDI 000986.

Writes results_000986.npz with per-unit tuning curves, best frequencies, PSTHs
and statistics pooled over sessions.
"""

import numpy as np
import pynapple as nap
from tqdm import tqdm

from dandi_io import list_assets, load_tone_session
from tuning_core import bandwidth_octaves, population_psth, session_tuning

nap.nap_config.suppress_conversion_warnings = True

EVOKED = (0.010, 0.060)  # response window, from the population PSTH latency
BASE = (-0.100, -0.010)
PSTH_WIN = (-0.05, 0.15)
PSTH_BIN = 0.002


def main():
    assets = list_assets("000986")
    print(f"{len(assets)} sessions in dandiset 000986")

    rows = []
    for path, url in tqdm(assets, desc="sessions"):
        S = load_tone_session(url)
        spikes, onsets, freq = S["spikes"], S["onsets"], S["freq"]
        res = session_tuning(spikes, onsets, freq, EVOKED, BASE)
        ufreq = res["ufreq"]

        # frequency-resolved PSTHs, (n_units, n_freq, n_bins)
        psths = []
        for f in ufreq:
            t, P = population_psth(spikes, onsets[freq == f], PSTH_WIN, PSTH_BIN)
            psths.append(P)
        psths = np.stack(psths, axis=1)

        subject = S["nwbfile"].subject.subject_id
        rows.append(dict(
            session=path, subject=subject, ufreq=ufreq, psth_t=t, psth=psths,
            spont_rate=np.asarray(spikes.rate), **{k: res[k] for k in
                ("evoked_rate", "raw_rate", "base_rate", "sem", "p_anova",
                 "p_driven", "bf", "n_trials", "unit_ids")}))
        S["io"].close()
        n_sig = (res["p_anova"] < 0.01).sum()
        tqdm.write(f"{path}: {len(spikes)} units, {len(onsets)} tones, "
                   f"{n_sig} frequency-tuned (ANOVA p<0.01)")

    ufreq = rows[0]["ufreq"]
    assert all(np.array_equal(r["ufreq"], ufreq) for r in rows)

    cat = lambda k: np.concatenate([r[k] for r in rows], axis=0)
    out = dict(
        ufreq=ufreq,
        psth_t=rows[0]["psth_t"],
        evoked_rate=cat("evoked_rate"),
        raw_rate=cat("raw_rate"),
        base_rate=cat("base_rate"),
        sem=cat("sem"),
        p_anova=cat("p_anova"),
        p_driven=cat("p_driven"),
        bf=cat("bf"),
        spont_rate=cat("spont_rate"),
        psth=cat("psth"),
        unit_ids=cat("unit_ids"),
        session=np.concatenate([[r["session"]] * len(r["bf"]) for r in rows]),
        subject=np.concatenate([[r["subject"]] * len(r["bf"]) for r in rows]),
        n_trials=np.stack([r["n_trials"] for r in rows]),
        session_list=np.array([r["session"] for r in rows]),
        evoked_window=np.array(EVOKED),
        base_window=np.array(BASE),
    )
    out["bandwidth"] = bandwidth_octaves(out["evoked_rate"], ufreq)
    np.savez_compressed("results_000986.npz", **out)
    print(f"\npooled: {len(out['bf'])} units from {len(rows)} sessions, "
          f"{len(set(out['subject']))} mice")
    print("saved results_000986.npz")


if __name__ == "__main__":
    main()
