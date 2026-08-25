# Gate commit, seed D4.1 (formation sweep, pythia-1.4b), 2026-08-22

Standing human directive: autonomous, tools frozen, seeds mutable, $100 cap.
Commits records/seedD4_1_design_diff_triage.json (budget_usd 5) for build and
execution.

Triage settled before spend:
- The decidability NEEDS_RUN (are the readings reachable, or is the favored
  one forced by the coarse grid?) is answered by a committed model-free
  control, records/CONTROL_d4_1_label_reachability.txt: synthetic early
  sigmoid, late sigmoid, form-then-drift, noisy plateau, and never-forms
  trajectories, evaluated only at the twelve grid steps with the committed
  bands, produce all five labels. None is forced; all are reachable.
- The coordinates NEEDS_RUN is the run itself, with the binding grid-edge
  degeneracy flags carried from records/GATE_COMMIT_seedD4.md.

Answers of record to the diff's five questions:
1. Widths are the seven discrete values in the committed stimulus file (20,
   40, 60, 80, 100, 120, 140) - the file defines them; nothing is invented.
2. Early-versus-late is judged at the 0.5 crossing (the formation step); the
   90-percent crossing only starts the hold test. Both are recorded.
3. All randomness streams from base seed 0, the chain's committed convention,
   recorded in every record.
4. The hold test for BOTH channels starts at the shared "formed" checkpoint
   (seed definition 4). The boundary's own first crossing is a separate,
   recorded observable (seed definition 5), not the start of its hold window.
5. The $5 estimate is accepted; the ledger and the 90-minute box cap govern.

Post-build ratifications, 2026-08-22:
- Selftest re-verified locally: 340 checks, exit 0; the committed-control
  replay is load-bearing (perturbed constants produce failures).
- The builder's honest pre-registration note is accepted onto this record:
  once the gate cell passes, the two NOT_FORMED readings cannot fire (the
  gate forces the final checkpoint's values); the four live labels remain
  reachable per the committed control.
- Launcher inputs set by the orchestrator, per the launcher's own documented
  surface: payload extended with this rung's four files; REMOTE_TIMEOUT=8100
  for this launch only (measured estimate ~100 minutes exceeds the 5400s
  default; the on-box self-destruct backstop scales with it automatically).
