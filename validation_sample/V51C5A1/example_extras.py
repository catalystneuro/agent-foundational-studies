"""Extra material from the example session: raw traces for figure 1 and an
arousal control.

The arousal control asks whether the frequency tuning measured above is a
property of the neurons or an artefact of fluctuating brain state: it splits
trials by pupil diameter at tone onset and re-measures each tuning curve within
each half.
"""

import pickle

import numpy as np
import pynapple as nap

import dandi_io
import tuning as T

EXAMPLE_SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
RAW_SPAN = 1200.0  # s of spike times kept for the raster figures


def main():
    asset = next(a for a in dandi_io.list_assets() if a["path"] == EXAMPLE_SESSION)
    nwb, nwbfile = dandi_io.open_session(asset)
    spikes, onsets, freqs, _ = T.load_session_arrays(nwb, nwbfile)

    # ---- raw traces for figure 1
    raw_ep = nap.IntervalSet(start=onsets[0] - 5, end=onsets[0] + RAW_SPAN)
    sub = spikes.restrict(raw_ep)
    spike_times = {int(uid): np.asarray(sub[uid].t) for uid in sub.index}
    pop_rate = spikes.restrict(raw_ep).count(0.02).sum(axis=1) / 0.02 / len(spikes)

    pupil = nwb["pupil_diameter"]
    print(f"pupil: {len(pupil)} samples at {pupil.rate:.0f} Hz, "
          f"{np.isnan(np.asarray(pupil)).mean():.1%} NaN")
    pupil_ds = pupil.bin_average(1.0)  # 1 Hz, for plotting the whole session

    raw = {
        "spike_times": spike_times,
        "pop_rate": pop_rate,
        "pupil": pupil_ds,
        "spont": nwb["spontaneous_blocks"],
        "raw_span": RAW_SPAN,
        "t_start": onsets[0] - 5,
    }
    with open("example_raw.pkl", "wb") as f:
        pickle.dump(raw, f)

    # ---- arousal control
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)
    ex = next(r for r in results if r["path"] == EXAMPLE_SESSION)
    assert np.array_equal(ex["onsets"], onsets)

    pupil_onset = np.asarray(nap.Ts(t=onsets).value_from(pupil))
    good = np.isfinite(pupil_onset)
    med = np.nanmedian(pupil_onset)
    low, high = good & (pupil_onset <= med), good & (pupil_onset > med)
    print(f"{low.sum()} low-pupil and {high.sum()} high-pupil trials "
          f"({(~good).sum()} dropped for missing pupil)")

    ev_dur = T.EVOKED_WIN[1] - T.EVOKED_WIN[0]
    bl_dur = T.BASELINE_WIN[1] - T.BASELINE_WIN[0]
    dr = ex["evoked"] / ev_dur - ex["baseline"] / bl_dur

    def tuning_curves(mask):
        return np.stack([dr[mask & (freqs == f)].mean(axis=0) for f in T.FREQS], axis=1)

    arousal = {
        "pupil_at_onset": pupil_onset[good],
        "median_pupil": med,
        "tc_low": tuning_curves(low),
        "tc_high": tuning_curves(high),
        "tuned": ex["reject_tuned"],
        "bf_idx": ex["bf_idx"],
        "mean_rate_low": dr[low].mean(),
        "mean_rate_high": dr[high].mean(),
    }
    with open("arousal.pkl", "wb") as f:
        pickle.dump(arousal, f)

    t = arousal["tuned"]
    r = np.corrcoef(arousal["tc_low"][t].ravel(), arousal["tc_high"][t].ravel())[0, 1]
    agree = np.mean(np.argmax(arousal["tc_low"][t], axis=1)
                    == np.argmax(arousal["tc_high"][t], axis=1))
    print(f"tuning curves across arousal halves: r = {r:.2f}, same BF in {agree:.0%}")
    print(f"mean evoked rate: low pupil {arousal['mean_rate_low']:.2f} Hz, "
          f"high pupil {arousal['mean_rate_high']:.2f} Hz")


if __name__ == "__main__":
    main()
