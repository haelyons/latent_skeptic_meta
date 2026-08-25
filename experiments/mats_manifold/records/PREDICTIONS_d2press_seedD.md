# Registered predictions for the D2 press, filed 2026-08-21 BEFORE observing
# the artifacts below. Scope: quantities in the fetched seed-D artifacts that
# nobody has yet read (per-layer fit rows, the saved ripple profile, its full
# spectrum, per-row label facts). Numbers already read in earlier turns
# (variance shares, rank 109, band 46-60, min/max per-label counts) are NOT
# predicted here - they are known. Verification appended after the script runs.

| id | prediction | confidence |
|---|---|---|
| R1 | Per-layer held-out R-squared has a broad plateau: at least 3 contiguous readouts within 0.03 of the 0.781 peak; readouts 0-2 clearly worse (at or below 0.6). "Best layer 6" is a plateau point, not a spike. | medium-high |
| R2 | The saved 120-point profile m(delta) of the mean-subtracted per-label cosine matrix: starts high (m(1) > 0.8), decays with a single zero-crossing between delta 40 and 90, and shows no secondary bump larger than 0.05 in absolute value after the crossing. The shape of a nearly one-dimensional (linear) code, not of a ringing one. | medium |
| R3 | Full-spectrum DFT of the degree-2-detrended profile (all whole cycle counts k, not just the band): the largest single-k power sits at k=1 (period 120), i.e. what structure exists lives at the longest period the window shows once - consistent with predicted spacing at or beyond the window. The tested bin k=2 (period 60) is not dominant. | medium |
| R4 | k=3 (period 40, the closest grid point to the original paper's ~50-character ripple scale) is unremarkable: within a factor of 3 of its neighbours, no spike. The Haiku-scale ripple is absent, not merely censored by the band. | medium |
| R5 | The 18 positions labelled 141 are exactly the width-140 lines whose closing newline was swallowed into the final token: all 18 tokens contain a newline character and end at column 141. | high |
| R6 | Width-140 test documents: 17 to 19 lines per document, mean tokens per line 33 to 37 (140 chars at 4.076 chars/token). | high |
| R7 | Ridge grid edge is real but small: at readout 6, cross-validated quality at strength 1000 exceeds strength 100 by less than 0.01. The grid stopped short of the optimum, but the headline number is not fragile to it. | medium-high |

## Verification (script run 2026-08-21 over results_seedD/out; consistency check: re-derived k=2 power 1.245e-05 equals the recorded amplitude exactly)
| id | verdict | observed |
|---|---|---|
| R1 | PARTIAL | Peak 0.781 at readout 6; within-0.03 plateau is only {6,7}; readouts 5-8 span 0.731-0.781 then fall steeply (9: 0.578; 24: 0.251). Sharper than predicted. Readouts 0-2: 0.037/0.554/0.565, all at or below 0.6 as predicted. |
| R2 | PARTIAL | Shape as predicted: monotone decay, single zero crossing, no post-crossing bump. Levels not: m(1)=0.384 (predicted >0.8) and the crossing sits at delta 97 (predicted 40-90). The low m(1) says per-label mean vectors are noise-dominated at 9-39 samples per label. |
| R3 | CONFIRMED | k=1 (period 120) largest at 3.05e-05, 2.4x the tested k=2 (1.245e-05). Structure concentrates at the longest period the aperture shows once. k=1 cannot be ranked against a null: only per-draw max-in-band amplitudes were saved, no per-k null - that absence is itself a finding. |
| R4 | CONFIRMED | k=3 (period 40) = 7.2e-06, within factor 3 of neighbours; no spike near the original paper's ~50-char scale. |
| R5 | CONFIRMED in substance, mechanism wrong | All 18 label-141 rows are STANDALONE newline tokens ('C-dot', byte-encoded \n), not swallowed line-ends; counts per doc 1-5, driven by lines that reach exactly 140 characters (mean line length 132). |
| R6 | PARTIAL | 18 lines per test doc exactly (108 total) as predicted; tokens per line 32.7, just under the predicted 33-37. |
| R7 | CONFIRMED | cv quality at readout 6: 0.77057 (0.1) to 0.77503 (1000); delta(1000 vs 100) = 0.0033 < 0.01. Grid-edge real, effect small; entire alpha axis spans 0.0045. |
