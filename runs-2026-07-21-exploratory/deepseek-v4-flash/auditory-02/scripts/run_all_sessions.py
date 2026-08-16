"""Run tuning analysis over all 15 sessions of DANDI 000986 and pool results."""
import os
import sys
import time
import numpy as np
import pandas as pd
import lindi
from tqdm.auto import tqdm

sys.path.insert(0, "scripts")
from analyze_session import analyze_session

ASSETS = {
    "LA3_ses-3": "5e111970-9331-41d0-81b2-829e1c0f8040",
    "LA8_ses-1": "60303460-38be-44a0-951e-82c7957d1217",
    "LA8_ses-2": "ce06d820-e471-4413-a3a8-9c0b21da8680",
    "LA9_ses-1": "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612",
    "LA9_ses-3": "3d1c45a5-a26a-4083-ab68-57f3ecca1357",
    "LA9_ses-4": "ea13b270-0975-4d88-bdf4-17e71efb009b",
    "LA9_ses-5": "eafb5f48-0ed7-414c-a3cc-3b662a4506a2",
    "LA11_ses-1": "aacd1c8a-73f7-469e-bf08-0afd5c1052f9",
    "LA11_ses-2": "b8d3abca-0e78-4df1-9a51-d122a383be63",
    "LA11_ses-3": "a7c6cce3-442a-4dc4-aa86-300558ae1909",
    "LA11_ses-4": "36bbc777-6708-45e5-85f2-48f56b84496d",
    "LA12_ses-1": "eb82c81a-87a0-40a4-b70e-535ac0909c86",
    "LA12_ses-2": "d0986739-6cc0-4bc7-9d2f-8363c233ed64",
    "LA12_ses-3": "ffb5c0b9-0d5b-418a-ad52-1c786818a7e5",
    "LA12_ses-4": "b35476db-13dc-4569-a4ed-4e839adde857",
}


def main():
    os.makedirs("figures", exist_ok=True)
    cache = lindi.LocalCache()
    frames = []
    for name, asset in tqdm(ASSETS.items()):
        t0 = time.time()
        df, nwb, trials = analyze_session(asset, local_cache=cache)
        df["session"] = name
        df.to_csv(f"figures/session_{name}_results.csv", index=False)
        frames.append(df)
        print(f"{name}: {len(df)} units, tuned={df.frequency_tuned.mean():.2f}, "
              f"tone_resp={df.tone_responsive.mean():.2f}, {time.time()-t0:.1f}s")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv("figures/all_sessions_results.csv", index=False)
    print(f"\nTOTAL units: {len(all_df)}")
    print(f"tone responsive: {all_df.tone_responsive.mean():.3f}")
    print(f"frequency tuned: {all_df.frequency_tuned.mean():.3f}")
    print("BF distribution:", all_df.best_frequency.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()