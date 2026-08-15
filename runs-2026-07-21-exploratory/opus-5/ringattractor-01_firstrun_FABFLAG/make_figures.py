"""Generate every figure: single-session deep dive plus the multi-session summary."""
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import analysis as A  # noqa: E402
import figures as F  # noqa: E402

EXAMPLE = "Mouse28-140313"


def main():
    res, S = A.analyse_session(EXAMPLE, n_shuffle_mvl=200, n_emb_shuffle=5)
    F.fig_overview(S)
    F.fig_tuning(res["sel"])
    F.fig_bump(S, res["sel"])
    F.fig_pairwise(res)
    F.fig_manifold(res)
    F.fig_coherence(res)
    F.fig_dimensionality(S, res["sel"], res)
    with open("results_example_session.pkl", "wb") as f:
        pickle.dump({k: v for k, v in res.items() if k != "sel"}, f)
    try:
        allres = pickle.load(open("results_multisession.pkl", "rb"))
        F.fig_multisession(allres)
    except FileNotFoundError:
        print("results_multisession.pkl not found; run run_all_sessions.py first")
    print("figures written")


if __name__ == "__main__":
    main()
