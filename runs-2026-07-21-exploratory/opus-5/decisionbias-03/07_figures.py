"""Build every figure from results.pkl."""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import figures_lib as F

if __name__ == "__main__":
    blob = pd.read_pickle("results.pkl")
    R, G, WS, CS, params = (blob["results"], blob["grid"], blob["WS"], blob["CS"],
                            blob["params"])
    F.fig_behavior(R)
    F.fig_raw(R, which=int(np.argmax([r["acc"] for r in R])))
    F.fig_single_units(R)
    F.fig_decoding(R)
    F.fig_time_resolved(R)
    F.fig_controls(R, G, WS, CS, params)
    F.fig_choice(R)
    print(F.fig_regions(R).round(3).to_string())
