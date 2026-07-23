"""Streaming access helpers for DANDI dandiset 001419 (auditory cortex linear probes).

All access is by streaming from the DANDI S3 bucket with a local remfile disk cache,
so no whole-file downloads are required.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "001419"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_001419")

# Tone-presentation sessions in dandiset 001419 ("pure tones" protocol).
# path -> asset id, resolved once from the DANDI API and frozen here for reproducibility.
TONE_SESSIONS = {
    "AK190207A_tones1": "b651a329-009d-4b85-b08c-ce99de014c18",
    "AK190207A_tones2": "178b9acd-870d-4902-9201-f49385c6fdab",
    "AK190207C_tones1": "163f4aca-eeaf-4b72-9226-f45c45d1504d",
    "AK190208_tones1": "6a39d1a4-6226-4a6f-939a-74acc5e1d46e",
    "AK190502A_tones1": "37fb14d7-b392-47b8-add2-4081111aeb6b",
    "AK190502A_tones2": "f3ad28b8-73dd-4901-8322-2a851ed5d9e8",
    "AK190502B_tones1": "4f150b2f-ef65-4e0e-9a15-3ce660d84dfb",
    "AK190506_tones1": "63152e12-669d-47e6-9d8c-68e7a8a15f24",
    "AK190508_tones1": "90b7810c-a577-4dde-8f19-ec58b9380c35",
    "AK210825B_tones1": "f637f8b9-70b1-437d-8332-1a375d7c3cc5",
    "AK210826_tones1": "22e2e691-a6dd-45de-b3ec-81fdc72a9264",
    "AK211103B_tones1": "42e0e33d-2194-4264-83e8-fb60139073d6",
    "AK211117_tones1": "75471b6f-8ec9-4d21-9139-9a759737e0ed",
    "AK211222_tones1": "3cfaa0f0-c37d-4635-a6f7-7d28606011d7",
    "AK220630_tones1": "a939b732-e66d-4b58-bc17-865930bf638a",
    "AK221130_tones1": "4e7cfd53-095c-4174-8efb-b968fb706d0f",
    "AK230127_tones1": "591e3076-b66b-41be-9c81-da695dcd49bf",
    "AK230201_tones1": "0334dd22-cdfa-4ba1-9343-1816b651646e",
    "HK190220_tones1": "3890acc6-b446-42ce-a36d-158654560703",
    "HK190525_tones1": "3f1c7fc1-fb5a-4a5f-98b5-92b3dcd01093",
    "HK190624_tones1": "c78f97cd-5eb7-44c1-94f4-771b9159aaa9",
    "HK190625_tones1": "c9f664c1-1db9-4b6e-8c0a-73a70725fde7",
    "HK210212_tones1": "f9b3f985-ae86-4d9f-b848-0b7b4517f256",
    "HK220729_tones1": "86a275f7-a588-4c48-b12c-e69177c285e8",
    "HK220729_tones2": "f34da583-c601-440e-b171-68278ba1af32",
    "HK220816_tones1": "c8cfc3f4-187b-4893-98dc-eec9cec597fa",
    "HK230224_tones1": "1d1facc3-8235-4d74-aeeb-272ef0e5a960",
    "HK230225_tones1": "b664ce7e-a036-498e-881f-5ec8fee7a117",
    "HK230405_tones1": "3f0aa99a-46e5-4e60-99ae-a01e11351889",
    "HK230724_tones1": "a692f122-abb0-42b1-96b8-8c2868d7b5b4",
    "HK230902_tones1": "6cf0454b-52ae-4337-bd91-b030428f738b",
    "HK230904_tones1": "0317b852-b71e-419f-92d5-59aea0d9476f",
    "HK230916_tones1": "d92d9f57-9030-49ba-90a2-09ee56e3fcd8",
    "HK231121_tones1": "8ccf8ab4-2b64-4fd9-ad7c-8c36becb7d11",
    "HK231122_tones1": "dd1262dd-cd41-45ad-ac09-a07e1b08fc84",
    "HK231207_tones1": "b18aa06e-eb70-4f2e-a79b-61ff016c90fe",
    "KO220727B_tones1": "08419b3a-89ff-4309-8dd1-d9fb5315dd29",
    "KO220804B_tones1": "abfcc741-8209-4eb6-9baa-f925a36b929f",
    "KO220804B_tones2": "f7974cd2-340b-4f65-afb1-e62ad9fa3e16",
    "KO230215_tones1": "8e1a4ecb-dcc3-4f7a-b518-7ae52880850f",
    "KO230406_tones1": "23d554c3-4046-4270-8244-cca9fe6420bd",
    "KO230421B_tones1": "96c8a1b0-b9a2-417d-8771-3de6677e5418",
    "KO230827B_tones1": "9e04d734-9e65-4afe-8478-7c1ece1fe423",
    "KO230828_tones1": "ee7b3cc2-d228-4395-a24f-502e5b1f7ab9",
    "KO230828_tones2": "53518e84-5164-43c9-b3d8-fc8fbb2ecf9b",
    "KO230910_tones1": "94e81dca-4f52-4d96-9fe3-2306dca54512",
    "KO230910_tones2": "59798ea9-eb41-4eed-8963-51854af103ae",
}


def asset_s3_url(asset_id, dandiset=DANDISET):
    """Resolve a DANDI asset id to its S3 URL without downloading the blob."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{asset_id}/download/",
        allow_redirects=False,
    )
    r.raise_for_status()
    return r.headers["Location"]


def open_nwb(asset_id, dandiset=DANDISET):
    """Stream an NWB file from DANDI and return the pynwb NWBFile."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rem = remfile.File(asset_s3_url(asset_id, dandiset), disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    return io.read()


def unit_peak_electrode(nwbfile):
    """Index (into the electrodes table) of the peak-amplitude electrode of each unit.

    Sessions differ: some store one electrode per unit, others store a ragged list
    whose first entry is the peak channel.
    """
    region = np.asarray(nwbfile.units["electrodes"].data[:])
    if len(region) == len(nwbfile.units):
        return region
    offsets = np.asarray(nwbfile.units.get("electrodes_index").data[:])
    starts = np.concatenate([[0], offsets[:-1]])
    return region[starts]


def load_session(session_key):
    """Load one tone session as pynapple objects plus tidy trial/unit tables.

    Returns a dict with:
      units          TsGroup of spike times, metadata columns 'quality' and 'depth_um'
      trials         DataFrame of tone trials (start_time, frequency, intensity, led)
      tone_onsets    pynapple Ts of tone onset times for LED-free trials
      nwb            the underlying NWBFile (for LFP/CSD and subject metadata)
    """
    asset_id = TONE_SESSIONS[session_key]
    nwbfile = open_nwb(asset_id)

    trials = nwbfile.trials.to_dataframe()
    # Some sessions interleave optogenetic (LED) trials; others have no LED column at all.
    if "led_on_time" in trials.columns:
        trials["led"] = ~np.isnan(trials["led_on_time"].values)
    else:
        trials["led"] = False

    # Spike times -> pynapple TsGroup, keyed by NWB unit id.
    unit_ids = np.asarray(nwbfile.units.id[:])
    spikes = {int(uid): np.asarray(nwbfile.units["spike_times"][i]) for i, uid in enumerate(unit_ids)}
    quality = np.asarray(nwbfile.units["quality"].data[:]).astype(str)

    elec_idx = unit_peak_electrode(nwbfile)
    elec_y = np.asarray(nwbfile.electrodes["y"].data[:])
    elec_loc = np.asarray(nwbfile.electrodes["location"].data[:]).astype(str)
    depth_um = elec_y[elec_idx]
    layer = np.array([l.strip() for l in elec_loc[elec_idx]])

    units = nap.TsGroup(
        {int(uid): nap.Ts(t=spikes[int(uid)]) for uid in unit_ids},
        metadata={"quality": quality, "depth_um": depth_um, "layer": layer},
    )

    ok = ~trials["led"].values
    tone_onsets = nap.Ts(t=trials["start_time"].values[ok])

    return {
        "session": session_key,
        "subject": nwbfile.subject.subject_id,
        "units": units,
        "trials": trials,
        "tone_onsets": tone_onsets,
        "nwb": nwbfile,
    }


def session_summary(session_key):
    """Cheap structural summary of one session (used for the multi-session survey)."""
    nwbfile = open_nwb(TONE_SESSIONS[session_key])
    tr = nwbfile.trials
    freq = np.asarray(tr["frequency"].data[:])
    inten = np.asarray(tr["intensity"].data[:])
    if "led_on_time" in tr.colnames:
        led = ~np.isnan(np.asarray(tr["led_on_time"].data[:]))
    else:
        led = np.zeros(len(freq), dtype=bool)
    quality = np.asarray(nwbfile.units["quality"].data[:]).astype(str)
    return dict(
        session=session_key,
        subject=nwbfile.subject.subject_id,
        n_trials=len(freq),
        n_freq=len(np.unique(freq)),
        f_min=float(np.min(freq)),
        f_max=float(np.max(freq)),
        intensities=tuple(np.unique(inten).tolist()),
        n_led=int(led.sum()),
        n_no_led=int((~led).sum()),
        trials_per_freq_no_led=int((~led).sum() / len(np.unique(freq))),
        n_units=len(quality),
        n_good=int((quality == "good").sum()),
        location=str(np.unique(np.asarray(nwbfile.electrodes["location"].data[:]).astype(str))[0]),
    )


if __name__ == "__main__":
    from tqdm import tqdm

    rows = [session_summary(k) for k in tqdm(sorted(TONE_SESSIONS), desc="survey")]
    df = pd.DataFrame(rows)
    df.to_csv("session_survey.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
