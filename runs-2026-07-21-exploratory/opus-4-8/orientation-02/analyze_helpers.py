"""Selection criteria shared by the analysis and figure scripts."""


def responsive(df):
    """Visually driven units.

    Two conditions, both needed. The t-test establishes that the unit is driven by
    the grating at all; the 1 Hz floor on the evoked response removes units whose
    tuning curve is a couple of spikes, for which gOSI is essentially undefined
    (one positive direction among zeros scores 1.0 no matter how few spikes it took).
    """
    return df[(df.p_resp < 0.01) & (df.peak_evoked > 1.0)]
