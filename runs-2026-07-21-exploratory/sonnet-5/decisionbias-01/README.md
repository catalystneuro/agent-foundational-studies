# Decoding an Upcoming Decision Bias from Pre-Stimulus Neural Activity

## Dataset

[DANDI dandiset 000149](https://dandiarchive.org/dandiset/000149) contains Neuropixels
recordings from mice performing the International Brain Laboratory (IBL) ephys
choice-world task: a visual grating appears on the left or right of a screen at one
of several contrasts (including 0%, i.e. no stimulus at all), and the mouse reports
the side by turning a wheel. The probability that the stimulus will appear on the
left, `probabilityLeft`, is fixed within blocks of trials at 0.2, 0.5, or 0.8, and a
quiescence period of several hundred milliseconds is enforced immediately before
stimulus onset. We used all four sessions in the dandiset that contain spike-sorted
units and a trials table, streaming the `units` and `trials` NWB tables directly via
the LINDI index rather than downloading the full (400-650 GB) files.

## Analysis

For each session we built a trial-by-unit matrix of spike counts from well-isolated
("good") units in the 400 ms immediately preceding stimulus onset, which falls within
the task's enforced pre-stimulus quiescence period. We first confirmed behaviorally
that the block prior biases choice on 0%-contrast trials, where the stimulus itself
carries no side information. We then used cross-validated logistic regression
(with PCA dimensionality reduction and a label-permutation significance test) to
decode, from this pre-stimulus population activity alone: (1) the block identity
(high-left-probability vs. high-right-probability), and (2) the animal's actual
upcoming choice, restricted to the 0%-contrast trials where any successful decoding
must reflect an internally generated bias rather than sensory evidence.

## Key finding

The block prior produced a strong behavioral bias in all four sessions (e.g. P(choice)
on ambiguous trials swinging from ~0.2-0.6 to ~0.8-0.95 between blocks). Consistent
with this, the block identity was decodable above chance from pre-stimulus population
activity in 3 of 4 sessions (permutation p = 0.050, 0.010, 0.005; one session was not
significant, p = 0.26), and the animal's upcoming choice on genuinely ambiguous
0%-contrast trials was decodable above chance in 2 of 4 sessions (p = 0.025, 0.035).
Because this activity was measured before the stimulus appeared and within the
movement-free quiescence window, it cannot reflect visual input or overt movement,
indicating that a substantial component of the decision bias imposed by the task's
block structure is already present in the recorded neural population before the
decision-relevant sensory evidence arrives. The effect was not uniform across
sessions, consistent with genuine biological and recording variability rather than
an artificially clean result.

## Files

- `decision_bias_decoding.py` - jupytext source script (run end-to-end without manual intervention)
- `decision_bias_decoding.ipynb` - executed notebook
- `fig1_raw_raster.png` - raw spike rasters around stimulus onset across trials
- `fig2_behavioral_bias.png` - choice bias by block on 0%-contrast trials
- `fig3_block_decoding.png` - decoding accuracy for block identity vs. permutation null
- `fig4_choice_decoding.png` - decoding accuracy for upcoming choice (0%-contrast trials) vs. permutation null
- `fig5_example_unit_bias.png` - example single unit with block-dependent pre-stimulus firing
