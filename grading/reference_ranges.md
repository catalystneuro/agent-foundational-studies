# Preregistered reference ranges (axis 4)

Provisional, pending domain-expert review. A headline value outside its range caps
axis 4 (result correctness) at 1. Ranges are deliberately wide because legitimate
methodological choices produce real variation; they are meant to exclude results
that are clearly wrong, not to enforce one correct analysis.

| Problem | Headline statistic | Plausible range |
|---|---|---|
| Place cells | Fraction of CA1 pyramidal cells spatially selective | 0.30 to 0.75 |
| Place cells | Spatial information of place cells | 0.4 to 2.5 bits/spike |
| Orientation | Fraction of visually responsive V1 units orientation-selective | 0.25 to 0.75 |
| Orientation | Median orientation selectivity index (OSI) in V1 | 0.25 to 0.70 |
| Auditory | Fraction of A1 units with significant frequency tuning | 0.40 to 0.90 |
| Auditory | Tonotopic gradient present along the probe | Qualitative, must be shown |
| Reach | Fraction of M1/PMd units significantly directionally tuned | 0.50 to 0.95 |
| Reach planning | Fraction of PMd units directionally tuned during the delay period | 0.25 to 0.75 |
| Theta | Fraction of place fields with significant negative phase-position slope | 0.35 to 0.85 |
| Theta | Typical phase-position slope | -0.5 to -5 rad per field |
| Theta | Fraction of pyramidal cells significantly theta-entrained | 0.45 to 0.95 |

## Legitimate datasets per problem (axis 1 sanity, non-exhaustive)

These are dandisets known to contain the phenomenon; a run using one of them is on
solid ground for axis 1. A run using something else is not automatically wrong, but
verify the dataset actually contains the phenomenon before scoring axis 1 highly.

- Place cells / theta: 000044 (Grosmark & Buzsaki CA1), 000447, 000940, 000059.
- Orientation: 000021 (Allen Visual Coding Neuropixels), 000168.
- Auditory frequency tuning: 000986, 001419.
- Reach (execution and planning): 000128 (MC_Maze), 000140, 000688.
