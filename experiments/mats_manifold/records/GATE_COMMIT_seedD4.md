# Gate commit, seed D4 (formation rung, 12-checkpoint sweep), 2026-08-22

Standing human directive: autonomous, tools frozen, seeds mutable, $100 cap.
Commits records/seedD4_design_diff_triage.json (budget_usd 2.5) for build and
execution. Note of record: the first design pass lost one triage skeptic to a
structured-output retry cap; the run was resumed from cache and the skeptic
completed - the committed record carries both findings.

Answers of record to the diff's four questions:

1. Both formation markers stay, because they answer different questions: the
   0.5 crossing locates when the counter becomes usable (formation step); the
   90-percent-of-final crossing starts the hold test (formed). One is about
   appearance, the other about stability.
2. The $2.50 estimate comes from twelve one-pass measurements of a small
   model on one box; the ledger (records/BUDGET.md, ~$3.6 spent of $100)
   governs, and the 90-minute box cap bounds the run physically.
3. Each checkpoint uses its own best layer, selected by its own pooled
   quality - the layer trajectory is itself a recorded observable (the seed
   predicts migration during formation). Fixing layer 6 everywhere would
   presume the answer.
4. Yes: the four-bin score is the within-bin Pearson correlation, exactly as
   the seed states it.

Triage answers, both on the record:

- Decidability (EXPLAINS): the two endpoints routing refutation to
  "instrument fault" are integrity gates - the anchor reproduction and the
  step0 negative control - the same class as every prior rung's void
  machinery. The scientific fork has live refuting outcomes on both sides:
  HELD refutes the drift hypothesis, DRIFTED refutes the fossil hypothesis,
  and both are reachable. NOT_FORMED_BY_FINAL is treated as instrument fault
  because two independent committed anchors (0.780903, 0.781031) already
  establish the counter exists at the final checkpoint; a sweep contradicting
  both impeaches the sweep, not the twice-measured fact.
- Coordinates/scale (NEEDS_RUN): the queued control is ADOPTED AS BINDING for
  the build - per checkpoint, the selected best layer and selected ridge
  strength are logged beside pooled quality and boundary ratio, and any
  checkpoint whose best layer sits at the sweep edge (0 or 24) or whose ridge
  strength sits at the grid edge (1 or 100000) is flagged degenerate in its
  own record. The step0 control is expected to trip these flags (a random
  model has no meaningful best layer); that expectation is recorded here so
  the flag on step0 reads as confirmation, not alarm.

Post-build ratifications, 2026-08-22:
- The formation-timing call follows the seed's 0.5-crossing rule where the
  design object's seventh endpoint text diverged; both crossings are computed
  and recorded. The seed is senior where seed and design conflict.
- Band comparisons round differences to nine decimals so a value sitting
  exactly on a stated edge is judged in decimals, not in bits. Accepted.
- Local re-verification: 211-check selftest exits 0 (also with numpy hidden),
  onbox syntax clean. Launcher copy list extended for this rung (payload-only,
  documented in the launcher header's adaptation surface).
