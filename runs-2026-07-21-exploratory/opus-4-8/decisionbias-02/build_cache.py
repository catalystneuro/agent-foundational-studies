"""Extract per-session trial variables and time-resolved pre/peri-stimulus
population spike-count matrices, caching them to .npz for fast downstream use.

For each session we compute, for a grid of window centers relative to stimulus
onset, an (n_trials x n_units) firing-rate matrix. Downstream decoding then just
loads these arrays.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os, time
from tqdm import tqdm
import ibl_lib as L

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Sliding-window grid (seconds relative to stimOn). Width 0.2 s.
WIN_CENTERS = np.round(np.arange(-1.0, 0.51, 0.1), 2)
WIN_WIDTH = 0.2


def build(asset_id, label):
    out = os.path.join(CACHE_DIR, f"sess_{label}.npz")
    if os.path.exists(out):
        print("exists:", out)
        return
    t = time.time()
    f, nwbfile = L.load_session(asset_id)
    df = L.get_trials_df(nwbfile)
    df.columns = [c.replace(".npy", "") for c in df.columns]

    choice = df["choice"].values.astype(float)
    cL = np.nan_to_num(df["contrastLeft"].values)
    cR = np.nan_to_num(df["contrastRight"].values)
    pL = df["probabilityLeft"].values.astype(float)
    stimOn = df["stimOn_times"].values.astype(float)
    firstMove = df["firstMovement_times"].values.astype(float)
    feedback = df["feedbackType"].values.astype(float)

    gu = L.select_good_units(nwbfile)
    tsg = L.get_spike_tsgroup(f, gu)
    print(f"  {label}: {len(gu)} good units, loading spikes...")

    # region labels for the selected units
    units = nwbfile.units
    if "brainLocationAcronyms_ccf_2017" in units.colnames:
        reg_all = np.asarray(units["brainLocationAcronyms_ccf_2017"][:]).astype(str)
        regions = reg_all[gu]
    else:
        regions = np.array(["n/a"] * len(gu))

    # time-resolved rate matrices: shape (n_win, n_trials, n_units)
    valid = ~np.isnan(stimOn)
    Xs = np.stack([
        L.prestim_count_matrix(tsg, stimOn[valid],
                               c - WIN_WIDTH / 2, c + WIN_WIDTH / 2)
        for c in WIN_CENTERS
    ], axis=0)

    np.savez_compressed(
        out,
        win_centers=WIN_CENTERS, win_width=WIN_WIDTH,
        X=Xs, valid=valid,
        choice=choice[valid], contrastLeft=cL[valid], contrastRight=cR[valid],
        probabilityLeft=pL[valid], stimOn=stimOn[valid],
        firstMovement=firstMove[valid], feedback=feedback[valid],
        regions=regions,
    )
    print(f"  saved {out} in {time.time()-t:.0f}s  X={Xs.shape}")


if __name__ == "__main__":
    for asset_id, label in tqdm(L.SESSIONS.items(), desc="sessions"):
        build(asset_id, label)
    print("done")
