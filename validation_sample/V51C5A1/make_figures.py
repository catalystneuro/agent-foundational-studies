"""Generate every figure from the cached analysis results."""

import pickle

import figures as F

results = F.load()
ex = F.example(results)
with open("example_raw.pkl", "rb") as f:
    raw = pickle.load(f)
with open("arousal.pkl", "rb") as f:
    arousal = pickle.load(f)

for name, fn, args in [
    ("fig01_raw_data", F.fig_raw_data, (ex, raw)),
    ("fig02_design_and_alignment", F.fig_design_and_alignment, (results, ex)),
    ("fig03_example_units", F.fig_example_units, (ex, raw)),
    ("fig04_population_tuning", F.fig_population_tuning, (results,)),
    ("fig05_statistics", F.fig_statistics, (results,)),
    ("fig06_decoding", F.fig_decoding, (results, ex)),
    ("fig07_glm", F.fig_glm, (ex,)),
    ("fig08_arousal_control", F.fig_arousal, (arousal,)),
]:
    fn(*args)
    print("wrote", name + ".png")
