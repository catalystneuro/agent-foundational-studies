"""Loading utilities for DANDI:000056 (Peyrache et al. 2015, Nat Neurosci)."""
import h5py, remfile, numpy as np, pandas as pd
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000056"
VERSION = "0.250624.0430"

# session_id -> asset_id
SESSIONS = {
    "Mouse12-120806": "33f6aa51-f82f-4467-95b0-3f1403668654",
    "Mouse12-120807": "69655198-97af-4e65-9198-1c94d0830d24",
    "Mouse12-120808": "88587645-9aa1-4247-84b2-69bd45a23ce9",
    "Mouse12-120809": "c816972c-f501-4726-9333-a42aa2321d90",
    "Mouse12-120810": "00bf96d4-2c4d-44e9-93ea-a21e7c6e6959",
    "Mouse17-130125": "e62179fc-cd8d-4901-98aa-ccb35086f9f7",
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse17-130129": "533ebd09-8af4-4238-8c51-8a679c427c96",
    "Mouse17-130130": "e562f907-b93d-49fa-a1b1-3a3a89645e4f",
    "Mouse17-130131": "d6b3126c-2c99-45b4-a1e6-8f0e5acda578",
    "Mouse17-130201": "7f408e4a-3218-4a17-983b-d7b5cdb9dc82",
    "Mouse17-130202": "c86c7e25-fa65-4356-a699-4b1ab8ab0d64",
    "Mouse17-130203": "74079bfe-223b-45ea-b71b-40d7b2ff6151",
    "Mouse17-130204": "f92f2709-4469-4c1d-a883-75eb66ee898a",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse20-130515": "f1a2857b-22b7-4e9e-a6af-211c76081391",
    "Mouse20-130516": "714ae657-90a5-44ae-82c5-c1f7e762c2b2",
    "Mouse20-130517": "d3f6aa86-20fb-4330-97dc-8e4900e16a0e",
    "Mouse20-130520": "c0104ace-0807-473a-ad59-a8cc594ac4e9",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse24-131216": "14dafd30-f9fb-4aac-bc8e-e0d46d1b35e6",
    "Mouse24-131217": "c6cfc0f2-dda6-4136-be2c-be4a4de5a50e",
    "Mouse24-131218": "07ef959c-6778-4401-9434-641b88fc75a2",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse25-140124": "43c07029-2ac8-4900-9e51-8c44c3264080",
    "Mouse25-140128": "91242a8e-2402-40b3-926b-8d07aa406748",
    "Mouse25-140129": "8750d43b-c61d-461c-a30c-e3f071a835e6",
    "Mouse25-140130": "9c49d0c6-6e94-405b-b4e9-cb2ac744a36a",
    "Mouse25-140131": "dba64364-e09b-4ad5-b54c-baaa689e975a",
    "Mouse25-140203": "ac07ef7d-3c95-407d-ba01-f872b756b46c",
    "Mouse25-140204": "0889d7ae-7db5-4304-90ca-064cb7671791",
    "Mouse25-140205": "48dae2dc-35f7-4a9e-bcdb-883a7d5e2d32",
    "Mouse25-140206": "2a2e6913-5101-438d-bada-9f7dd46d3507",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
    "Mouse28-140311": "2c0950ed-43c6-4331-adbd-37a6c764758a",
    "Mouse28-140312": "ff614bbb-8194-4372-9c7e-bcc0966bd453",
    "Mouse28-140313": "56d2e2d7-ba41-40a1-b017-b81871f3f0c3",
    "Mouse28-140317": "3c4d6960-bdea-41da-a98f-842be83abf41",
    "Mouse28-140318": "174c0da6-6216-4f12-862b-0a082ce847df",
    "Mouse32-140820": "43be9315-c73f-4e83-adea-20e73ce75466"
}

def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
            f"{VERSION}/assets/{asset_id}/download/")

def open_session(asset_id, cache_dir="/tmp/remfile_cache"):
    rf = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(cache_dir))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()

def load_session(asset_id, **kw):
    """Return dict with spikes (TsGroup), hd (Tsd, radians), epochs (dict of IntervalSet)."""
    nwbfile = open_session(asset_id, **kw)
    nwb = nap.NWBFile(nwbfile)
    spikes = nwb["units"]

    blue = nwb["BlueLED"]      # TsdFrame (x, y)
    red = nwb["RedLED"]
    b = blue.values.astype(float)
    r = red.values.astype(float)
    t = blue.index.values
    valid = (b[:, 0] > 0) & (b[:, 1] > 0) & (r[:, 0] > 0) & (r[:, 1] > 0)
    d = b[valid] - r[valid]
    ang = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    hd = nap.Tsd(t=t[valid], d=ang)

    states = nwbfile.processing["behavior"]["states"].to_dataframe()
    epochs = {}
    for lab in states["label"].unique():
        s = states[states["label"] == lab]
        epochs[lab] = nap.IntervalSet(start=s["start_time"].values, end=s["stop_time"].values)

    # Wake epoch with valid tracking (the open-field foraging session)
    dt = np.median(np.diff(t))
    gaps = np.diff(hd.index.values)
    brk = np.where(gaps > 5 * dt)[0]
    starts = np.concatenate([[hd.index.values[0]], hd.index.values[brk + 1]])
    ends = np.concatenate([hd.index.values[brk], [hd.index.values[-1]]])
    tracked = nap.IntervalSet(start=starts, end=ends).drop_short_intervals(1.0)
    epochs["wake_tracked"] = tracked.intersect(epochs["Awake"])

    return dict(session_id=nwbfile.session_id, subject=nwbfile.subject.subject_id,
                spikes=spikes, hd=hd, epochs=epochs, nwbfile=nwbfile)
