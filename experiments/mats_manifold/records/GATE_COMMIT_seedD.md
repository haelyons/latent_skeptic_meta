# Gate commit, seed D v2, 2026-08-21

Human directive (session DM-HACK-RETRO-D): "Amend the seed and proceed."
Amendment made (seed_d_ripple_v2.md; deviation record documents it); design
re-ran (wf_229b8c26-68f); this file commits the design in
records/seedD_v2_design_diff_triage.json for mode=build and execution, at
budget_usd 10 under the $15 cap.

Answers of record to the diff's five questions:

1. Search-band floor 43 vs 70-140: keep 43. It is a floor on the calibrated
   upper bound, guarding against a repeat of the mis-authored 4-35 band. The
   calibration stage measures Pythia's own ambient dimension on train
   documents and sets the band from it; hard-coding 70-140 would presume the
   number calibration exists to measure.
2. 200 permutation draws: accepted as working figure; calibration may raise
   it. The 0.5 percent resolution floor is in does_not_test; any borderline
   result is reported as a rank, not only pass or fail.
3. Margins 3 vs 5 characters: intentional. 3 is the confound-match margin
   (tight, adversarial to the claim); 5 is the instrument-agreement margin
   (looser, non-gating secondary). Both GUESS, both recorded.
4. $10 cap: one to two GPU-hours on a single small box plus retries; the
   90-minute on-box hard cap bounds it regardless; under the $15 pipeline cap.
5. Ridge grid {0.1,1,10,100,1000}, 5-fold on train: accepted, GUESS, binding.

Noted from triage grounding, accepted as-is: the 24 stimulus documents are
segments of one book (Moby Dick, Gutenberg 2701), not 24 distinct works.
Style homogeneity is already in does_not_test; the anchor band (0.574-0.874)
was authored to absorb stimulus mismatch of exactly this kind.
