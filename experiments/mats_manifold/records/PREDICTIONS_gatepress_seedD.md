# Registered predictions on the gate-press questions, seed D
# Filed 2026-08-21, BEFORE re-reading the committed source documents and before
# fetching the original paper. Verification appended below after checking.
# The press (human, at the budget gate): what does the literature say about the
# ripple; how would it define "real"; does it constrain the null-rank cutoff;
# can constants be calibrated before pre-registration; does that cover the
# ridge grid and layer indexing.

| id | prediction | confidence |
|---|---|---|
| R1 | The original paper defines ringing only qualitatively: a signal-processing analogy (Gibbs phenomenon), shown visually in similarity/tuning structure. No statistical test, no null model, no significance criterion anywhere in it. | high |
| R2 | No source defines "real ripple" operationally. Closest substitutes: the section 2.5 optimality argument, and a toy simulation where ripple arises dynamically. The reproduction never measured ripple at all. Therefore the permutation null in seed D is new, and no literature number can set its alpha. | high |
| R3 | The rank-10-of-200 cutoff (alpha 0.05) has no basis in any source; it is pure policy. The literature is silent on required rank in the significance sense. | high |
| R4 | The literature DOES constrain the ripple's expected period, via the mechanism: truncating a similarity structure to K Fourier modes makes sidelobes with period about (count range)/K, and K is about half the ambient dimension. Paper: ambient dim about 6 (top 6 PCs, 95 percent) gives K about 3, so period about 140/3, about 47 characters on our range. Pythia (PC1-3 = 97.4 percent) suggests K nearer 1-2, period 70-140. Verification will therefore show seed D's GUESS search band (4-35 chars) sits BELOW the predicted period, and the moving-average detrend (window 21) actively suppresses periods above about 21 - both misauthored, both derivable from the errata I had already read. | medium-high |
| R5 | The paper's ringing lives in fitted-feature geometry (feature tuning curves / reconstruction along the manifold), not in raw per-label activation means - i.e. seed D's definition-5 caveat is the faithful reading and definition 3 is the stand-in. | medium |
| R6 | No source states the layer-indexing convention behind "best layer 6" or "first two layers". Unresolvable from source material; no measurement settles a naming convention. The layer-location clause must become recorded-not-gating, or be evaluated under both conventions. | medium-high |
| R7 | Observe-then-preregister is already the program's own shape (the proposal's go/no-go pilot gates the sweep). The heuristics permit it exactly when the threshold-setter never sees the data the threshold will judge (H1) and the refuting outcome is named before the judging run (H2). So: a train-documents-only calibration stage may fix band, window, and grid; test documents are judged once. The ridge grid already follows this pattern (train-only cross-validation). Layer indexing does NOT fit it - it is a convention, not a measurement. | high |

## Verification (appended after fetching arXiv 2601.04480 HTML and the t-tech space; errata doc re-read)
| id | verdict | evidence |
|---|---|---|
| R1 | CONFIRMED | Paper: ringing "in the sense of signal processing... Gibbs Phenomenon"; shown as "faint off-diagonal bands on each side" of the diagonal in cosine-similarity heatmaps. Qualitative only. No test, no null, no threshold. |
| R2 | CONFIRMED | No operational "real" anywhere. Section 2.5 is normative; toy simulation shows ripple "arise[s] even when the solution is found dynamically" (errata line 21-23). Reproduction: "Never tested: ringing" (errata line 37). |
| R3 | CONFIRMED | No source mentions any significance criterion. Rank-10-of-200 is pure policy. |
| R4 | PARTIAL | Paper gives NO numeric wavelength or mode count ("NOT FOUND"). Mechanism confirmed (similarity structure realizable in 150 dims, forced into ~6, truncated Fourier coefficients, Gibbs overshoot - errata line 23). DERIVED, not quoted: truncation at K modes puts the first sidelobe near range/(2K+1) and spaces later ones near range/K; ambient ~6 (K~3) on range 150 puts bands at roughly delta 21 and spacing roughly 43-50; Pythia PC1-3=97.4% (K~1-2) pushes periods to 70+. Conclusion stands: GUESS band 4-35 clips the predicted range's top, and the window-21 moving-average subtract suppresses exactly the periods the theory predicts. Both misauthored; both derivable from the errata before authoring. |
| R5 | MISS, favorable | Paper shows ringing in cosine-similarity matrices of MEAN ACTIVATION VECTORS, linear probes, AND feature decoders, plus logistic-probe response curves. Seed definition 3 (mean-activation matrix) is paper-faithful, not a stand-in; the deviation record's framing was overcautious. Definition 5 stays as cross-check (probe-vector ambient is noisier: 6 PCs = 82% vs 95% for activations). |
| R6 | CONFIRMED | Paper says "probes trained after layer 1"; indexing base not stated cleanly (fetch self-contradicts). t-tech space: app shell only, all four instrument questions NOT FOUND - their "layer 6" convention is unpublished. No measurement settles it. |
| R7 | CONFIRMED | H1: "whoever sets a threshold must not have seen the data it will judge." H2: refuting outcome named in a committed artifact predating the run. Proposal's own pilot-gates-sweep shape. Gate text permits an amended design. Train-only calibration stage is compatible; test docs judged once. |

Unresolved low-stakes conflict: errata quotes the paper's widths as "k = 20, 40, ..., 140"; the arXiv fetch reports "k=15,20,...,150". Our widths 20-140 are in-family under either. Noted, not chased.
