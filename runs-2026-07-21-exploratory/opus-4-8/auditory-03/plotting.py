"""Shared plotting helpers and figure style."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "legend.frameon": False,
})


def freq_colors(ufreq, cmap="viridis"):
    """One color per tone frequency, ordered low to high."""
    norm = colors.Normalize(np.log2(ufreq[0]), np.log2(ufreq[-1]))
    return [cm.get_cmap(cmap)(norm(np.log2(f))) for f in ufreq]


def khz(f):
    return f"{f/1000:g}"


def octaves_from_bf(ufreq, bf):
    """(n_units, n_freq) matrix of log2 frequency distance from each unit's BF."""
    return np.log2(ufreq)[None, :] - np.log2(bf)[:, None]


def normalize_curves(evoked_rate):
    """Min-max scale each tuning curve to [0, 1].

    Only units with a positive peak (an excitatory tone response) are kept; the
    rest are returned as NaN together with a boolean mask. Min-max scaling keeps
    every curve on a common bounded axis, which peak-only scaling does not: a
    unit whose peak evoked rate is small but whose off-BF response is strongly
    suppressed would otherwise dominate any population average.
    """
    lo = evoked_rate.min(axis=1, keepdims=True)
    hi = evoked_rate.max(axis=1, keepdims=True)
    ok = (hi[:, 0] > 0) & (hi[:, 0] > lo[:, 0])
    out = np.full_like(evoked_rate, np.nan)
    out[ok] = (evoked_rate[ok] - lo[ok]) / (hi[ok] - lo[ok])
    return out, ok
