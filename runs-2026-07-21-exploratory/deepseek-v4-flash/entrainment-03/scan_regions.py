# %% [markdown]
# # Scan all four IBL sessions: unit regions + LFP layout
# %%
import sys
sys.path.insert(0, ".")
import numpy as np
import h5py
import session_config as sc

for aid, meta in sc.SESSIONS.items():
    print("=" * 100)
    print(aid, meta["label"])
    f = sc.open_h5py(aid)

    U = f["units"]
    cols = list(U.keys())
    print("unit columns:", cols)

    acr = "brainLocationAcronyms_ccf_2017.npy"
    if acr in U:
        vals = U[acr][:]
        print("n units:", len(vals))
        uniq, cnt = np.unique(vals, return_counts=True)
        for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1]):
            print(f"  {u!s:12s} {c}")
    else:
        print("n units:", U["spike_times_index"].shape[0], "(no region column)")
        print("  -> no region labels for this session")

    # total spikes
    st = U["spike_times"][:]
    sti = U["spike_times_index"][:]
    print("total spikes:", len(st), "spike time range:", st.min(), st.max())

    # ---- acquisition
    print("acquisition datasets:")
    for k in f["acquisition"].keys():
        d = f["acquisition"][k]
        if hasattr(d, "shape") and not hasattr(d, "keys"):
            print("  ", k, d.shape, d.dtype)

    # ---- LFP: find the .lf dataset and its rate/timestamps
    lf_key = [k for k in f["acquisition"].keys() if "lf.cbin" in k]
    if lf_key:
        k = lf_key[0]
        dset = f["acquisition"][k]
        print("LFP dataset:", k, "shape", dset.shape, dset.dtype)
        # conversion factor
        if "conversion" in dset.attrs:
            print("  conversion:", dset.attrs["conversion"])
        for ak in dset.attrs:
            if "rate" in ak.lower() or "time" in ak.lower() or "start" in ak.lower():
                print("  attr", ak, "=", dset.attrs[ak])

    # ---- processing /efficacy
    print("processing/ecephys datasets:")
    for k in f["processing"]["ecephys"].keys():
        d = f["processing"]["ecephys"][k]
        if hasattr(d, "shape") and not hasattr(d, "keys"):
            print("  ", k, d.shape, d.dtype)

    # ---- behavioral position
    if "processing" in f and "behavior" in f["processing"]:
        for k in f["processing"]["behavior"].keys():
            print("  behavior:", k)
    f.close()