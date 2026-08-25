# Gate commit, seed D2 (aperture invariance), 2026-08-21

Standing human directive (2026-08-21): proceed autonomously, tools and pipeline
frozen, seeds mutable, total cap $100. This file commits the design in
records/seedD2_design_diff_triage.json (budget_usd 40, under ledger headroom)
for mode=build and execution.

Answers of record to the diff's five questions:

1. Artifact saving: yes, binding. Seed definition 5 requires every draw's full
   per-period power vector and the per-label mean vectors at the best layer in
   the FETCHED results. A build that omits them violates its spec.
2. Ridge grid: {1, 10, 100, 1000, 10000, 100000} (GUESS). Chosen by
   cross-validated maximum; the record must state whether the argmax is
   interior. No formal plateau requirement: seed D1 measured the whole
   0.1-1000 axis spanning 0.0045 in quality - low stakes, recorded not gated.
3. The $40 is the design's own conservative estimate and does limit: if real
   cost approaches it, the run is already outside its 90-minute box cap and
   halts by infrastructure. Ledger cap remains $100 total (records/BUDGET.md).
4. Bin 141-175 straddles the 150 boundary: recorded, EXCLUDED from the
   represented-range decision. The decision uses bins fully at or below 140
   (four bins) against bins fully above 150 (three bins: 176-210, 211-245,
   246-280). Good diff catch; resolved without redesign.
5. Characters-per-token: measured per window, on that window's test documents
   (tokenization shifts slightly with wrap width).

Triage note accepted: INVARIANT/PROPORTIONAL requires at least two windows
clearing; a single clear is UNDETERMINED - already pre-registered in seed
definition 6, so the outcome space is complete and no endpoint is forced.

Sealed D2-S6 scored now (design-mode outcomes exist): PARTIAL - diff flagged
the ridge grid as predicted, but not the invariance margins; and budget_usd
came out 40, above my predicted 30.

## Post-build ratifications, 2026-08-21 (builder's flagged items A-E)

- A, B: accepted; both follow the measured precedent in run_seedD.py and
  remote_run.sh (no generation machinery, no venv).
- C: fixed on my side - records/GATE_COMMIT_seedD2.md added to the launcher's
  D2 copy list so its hash lands in the on-box records. The builder correctly
  did not touch the launcher.
- D: implementation constants accepted as recorded GUESSes; the token-window
  minimum of 40 is justified by the builder's measured coverage bound (at 5,
  the highest token counts reach zero held-out positions at widths 200/280).
- E: hard halt on an empty held-out character label accepted; measured lower
  bounds 13-27 positions make it unlikely, and a halt beats an invented rule.
- RATIFIED READING: the statistic's shortest period of 4 applies in each
  labelling's own units (4 characters for character labels, 4 tokens for token
  labels). With no search band there is nothing to convert; comparison between
  labellings happens at the wavelength stage, in characters, per gate answer 5.
