"""Why dandiset 001419 was not used.

Dandiset 001419 (linear probe recordings from mouse A1/A2 during pure tones)
samples 10 to 18 frequencies at half-octave spacing and several sound levels,
which would resolve tuning curve shape much better than the five octave-spaced
tones of 000986. Its sorted spike times, however, show no tone-locked response
when they are aligned to the onsets in the file's own trials table: the
population PSTH is flat at every sound level, in every session we checked, from
three different experimenters. Rather than guess at an undocumented offset, the
dataset was excluded. This script regenerates the evidence.
"""

import numpy as np

from dandi_io import list_assets, open_nwb
from plotting import plt
from tuning_core import population_psth

SESSIONS = ["sub-AK190207A/sub-AK190207A_ses-tones1_ecephys+image.nwb",
            "sub-HK231121/sub-HK231121_ses-tones1_ecephys+image.nwb",
            "sub-KO230828/sub-KO230828_ses-tones1_ecephys+image.nwb"]
REF = "sub-LA11/sub-LA11_ses-1_behavior.nwb"


def main():
    import pynapple as nap

    nap.nap_config.suppress_conversion_warnings = True
    assets = dict(list_assets("001419"))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)

    # reference: the dataset that is used, same analysis, same axes
    from dandi_io import load_tone_session
    S = load_tone_session(dict(list_assets("000986"))[REF])
    t, P = population_psth(S["spikes"], S["onsets"], (-0.15, 0.25), 0.010)
    g = P.mean(0)
    axes[0].plot(t * 1000, g / g[t < 0].mean(), "k")
    axes[0].axvline(0, color="r", lw=0.8)
    axes[0].set(title="000986 (used)\nsub-LA11_ses-1", xlabel="time from tone onset (ms)",
                ylabel="population rate / pre-tone rate")
    S["io"].close()

    for ax, sess in zip(axes[1:], SESSIONS):
        nwbfile, nwb, io = open_nwb(assets[sess])
        df = nwbfile.trials.to_dataframe()
        spikes = nwb["units"]
        for lvl in sorted(df["intensity"].unique()):
            ons = df["start_time"].to_numpy()[df["intensity"].to_numpy() == lvl]
            t, P = population_psth(spikes, ons, (-0.15, 0.25), 0.010)
            g = P.mean(0)
            ax.plot(t * 1000, g / g[t < 0].mean(), lw=1, label=f"{lvl:.0f} dB (n={len(ons)})")
        ax.axvline(0, color="r", lw=0.8)
        ax.set(title=f"001419 (excluded)\n{sess.split('/')[-1][:22]}",
               xlabel="time from tone onset (ms)")
        ax.legend(fontsize=6.5)
        io.close()

    for ax in axes:
        ax.set_ylim(0, 2.6)
    fig.suptitle("Tone-locked population response, aligned to each file's own trial table")
    fig.savefig("figS1_001419_alignment_check.png")
    plt.close(fig)
    print("saved figS1_001419_alignment_check.png")


if __name__ == "__main__":
    main()
