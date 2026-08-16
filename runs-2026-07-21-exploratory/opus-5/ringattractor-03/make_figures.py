"""Driver: run the pipeline over sessions, cache results, and build all figures."""
import warnings, os, sys, json
warnings.filterwarnings("ignore")
import numpy as np

from loader import load_session, SESSIONS
import run_session as rs
import plots

PRIMARY = "Mouse28-140312"
ALL_SESSIONS = ["Mouse28-140312", "Mouse28-140313", "Mouse12-120807",
                "Mouse25-140130", "Mouse17-130130", "Mouse24-131217"]


def get(sid):
    fn = f"cache_{sid}.npy"
    if os.path.exists(fn):
        return np.load(fn, allow_pickle=True).item()
    r = rs.analyze(sid)
    np.save(fn, r, allow_pickle=True)
    return r


def summarize(r):
    row = dict(sid=r["sid"], subject=r["subject"], n_units=r["n_units"],
               n_hd=len(r["hd_units"]))
    for st in ["wake", "REM", "nREM"]:
        d = r[st]
        row[f"cos_r2_{st}"] = d["cos_r2"]
        row[f"null_cos_r2_{st}"] = d["null_cos_r2"].mean()
        row[f"hollowness_{st}"] = d["hollowness"]
        row[f"null_hollowness_{st}"] = d["null_hollowness"].mean()
        row[f"ev1d_{st}"] = float(d["ev1d"])
        row[f"null_ev1d_{st}"] = d["null_ev1d"].mean()
        row[f"split_coh_{st}"] = d["split_coh"]
        row[f"null_split_coh_{st}"] = d["null_coh"].mean()
        row[f"map_conc_{st}"] = d["map_conc"]
        row[f"null_map_conc_{st}"] = d["null_map_conc"]
        row[f"drift_median_{st}"] = float(np.degrees(np.median(d["drift"])))
        row[f"pr_{st}"] = d["participation_ratio"]
        row[f"pc2_pc1_{st}"] = float(d["pca_spectrum"][1] / d["pca_spectrum"][0])
        row[f"dur_{st}"] = r["ep_len"][st]
    row["decode_err_median_deg"] = float(np.degrees(np.median(r["wake"]["decode_err"])))
    row["decode_circ_corr"] = r["wake"]["decode_circ_corr"]
    if "ctrl" in r["nREM"]:
        row["ctrl_hollowness_nREM"] = r["nREM"]["ctrl"]["hollowness"]
        row["ctrl_cos_r2_nREM"] = r["nREM"]["ctrl"]["cos_r2"]
        cs = r["nREM"]["ctrl"].get("pca_spectrum")
        if cs is not None:
            row["ctrl_pc2_pc1_nREM"] = float(cs[1] / cs[0])
        row["ctrl_n"] = r["nREM"]["ctrl"]["n"]
    return row


if __name__ == "__main__":
    sessions = sys.argv[1:] or ALL_SESSIONS
    res = {}
    for sid in sessions:
        print("===", sid, flush=True)
        res[sid] = get(sid)

    r = res[PRIMARY]
    s = load_session(SESSIONS[PRIMARY])
    plots.fig_overview(r, s, "fig01_data_overview.png")
    plots.fig_tuning(r, "fig02_hd_tuning_curves.png")
    plots.fig_correlations(r, "fig03_pairwise_correlations.png")
    plots.fig_manifold(r, "fig04_ring_manifold.png")
    plots.fig_dimensionality(r, "fig05_dimensionality.png")
    plots.fig_coherence(r, "fig06_internal_coherence.png")
    plots.fig_decode_validation(r, "fig07_decoding_validation.png")
    beh = plots.fig_sleep_vs_behavior(r, s, "fig09_internal_vs_measured.png",
                                      nearest_hd=rs.nearest_hd)

    summary = [summarize(res[sid]) for sid in sessions]
    summary[sessions.index(PRIMARY)].update(beh)
    df = plots.fig_multisession(summary, "fig08_multisession_summary.png")
    df.to_csv("session_summary.csv", index=False)
    print(df[["sid", "subject", "n_hd", "cos_r2_nREM", "hollowness_nREM",
              "ev1d_nREM", "split_coh_nREM"]].to_string(index=False))
