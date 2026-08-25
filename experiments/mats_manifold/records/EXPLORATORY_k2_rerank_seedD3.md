# Exploratory re-analysis: ranks with the one-cycle bin excluded. 2026-08-21.
# POST-HOC, not pre-registered; filed to decide whether a closing rung is
# worth running, and superseded by seed D3.1's committed version if run.

Derivation of the statistic change (first principles, plus the D3 record):
after a degree-2 polynomial detrend, the one-cycle (k=1) Fourier bin measures
residual trend curvature, not repetition - a single smooth drop across the
window lands there. Seed D3's only clearing was exactly that shape (cycles=1,
profile falling 0.36 to -0.16, correlation with position -0.914, per the post
reader's re-derivation from the committed record). Recomputed from the SAVED
per-draw spectra (same shuffles, same observed data, statistic = max power
over cycles >= 2, period >= 4):

| run | window | k>=2 max power | at period | rank of 200 |
|---|---|---|---|---|
| seedD3 (1.4b) | 140 | 3.723e-05 | 60.0 | 196 |
| seedD3 (1.4b) | 200 | 1.264e-05 | 90.0 | 198 |
| seedD3 (1.4b) | 280 | 1.788e-05 | 65.0 | 87 |
| seedD2.1 (410m) | 140 | 2.613e-05 | 60.0 | 201 |
| seedD2.1 (410m) | 200 | 1.409e-06 | 13.8 | 201 |
| seedD2.1 (410m) | 280 | 4.075e-05 | 130.0 | 33 |

Nothing clears (threshold: rank 10). Both models, all windows, the observed
repeating structure sits at or below the null's middle. The instrument-design
lesson, booked for the chain: seed D1 censored too much (a narrow band), seeds
D2.1/D3 censored too little (a trend-contaminated bin); the defensible
statistic excludes exactly the single-cycle bin and nothing else.

## Closing interpretation, 2026-08-22, after seed D3.1 (exploratory synthesis, not a gated claim)

The one-cycle component the k>=2 rule bars is real, reproducible, and
identified: on fresh text it again ranks 1 of 201 at window 280 with a
near-identical shape (falls 0.365 to -0.163, position-correlation -0.917,
against seed D3's 0.36 to -0.16 at -0.914), and its position-correlations
across windows run -0.92 to -0.9985. A monotone, label-locked similarity
decay is the count curve itself - adjacent counts more similar than distant
ones - incompletely flattened by the degree-2 detrend. The exclusion rule
does not hide an unexplained signal; it separates the twice-established curve
from the question asked ON TOP of it (does anything oscillate), whose answer
is no, everywhere tested.

Two bookkeeping findings from the final post read, non-decision-bearing:
1. The token-control records' "observed maximum" (58/73/97) exceeds the raw
   held-out rows' maxima (44/66/85): the record's maximum is measured over a
   different population (training documents, where the ceiling rule lives)
   without saying so. The cut-size figures inherit that ambiguity. No verdict
   changes (nothing cleared under any reading); records should name their
   population.
2. The design prose overstated the width-280 bin check as run-invalidating;
   the runner (correctly) lets it feed readings only. The runner is the
   binding artifact.

## Formation-rung post findings, settled and booked, 2026-08-23

1. LABEL ERRATUM, settled by re-derivation from raw rows (step143000, width
   140): the four bin values ARE genuine Pearson correlations (raw recompute
   0.6460/0.8196/0.2106/0.2266 vs recorded 0.646/0.820/0.211/0.227). One
   prose note in the formation-rung records misnames the QUALITY field as
   "Pearson correlation"; that field is, and always was, held-out R-squared
   (raw Pearson at width 140, step 0, is 0.28 while the R-squared quality is
   -0.0076 - both correct numbers for their own constructs). Numbers and
   decisions stand; one note's label was wrong.
2. The boundary-stability label depends on the committed startpoint: from the
   shared "formed" marker (step 512, per the gate answer), range 0.774 ->
   UNSTABLE; from the boundary's own crossing (step 64000, recorded-only),
   range 0.014 -> would read HELD. Both are in the records; the headline
   follows the committed rule and the alternative is visible beside it.
3. Formation timing is BOUNDED, not timed: both milestones fall at the first
   trained grid point (step 512), so the entire rise lives inside the
   unmeasured 0-512 gap. "Early" is correct but the two-tier milestone system
   contributed no discrimination in this run.
4. Standing instrument caveats for any next rung: width-20 quality is
   strongly negative at every checkpoint including final (-0.888) while the
   pooled headline stays 0.876 - position-pooling hides a wrap length where
   the linear read never works, unstated in any design; per-position rows are
   saved only for width 140, so six of seven widths' contributions to gating
   numbers cannot be recomputed from raw output (an auditability gap); the
   dimension collapse is of the dominant structure only (90-percent count
   falls 77 to 1-2 while the 99-percent count stays 80-130).
