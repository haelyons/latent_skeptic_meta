# Answer to QUESTION.md

Date: 2026-08-21. Everything below is backed by committed artifacts under `out/`;
audit results are the frozen pipelines' verbatim outputs (`result_a2.json`,
`result_b2.json`; pre-measurement round in `result_a.json`, `result_b.json`).

## The answer

**Under the preregistered protocol, no attribute is claimable as recoverable,
and none is refuted. The question's own anchor gate fired: the TalkTuner
demographic reproduction came out out-of-band (gender), and QUESTION.md:13-15
voids every extension claim on an out-of-band anchor.**

Measured, committed, but unclaimable-as-recovery extension numbers
(template-disjoint held-out test, layer chosen on validation only):

| attribute | test acc | chance | preregistered decision |
|---|---|---|---|
| expertise level | 0.8333 (n=150) | 0.50 | RECOVERED |
| emotional state | 1.0000 (n=150) | 0.50 | RECOVERED (ceiling) |
| formality preference | 1.0000 (n=96) | 0.50 | RECOVERED + PERM_ANOMALY (not claimable) |
| native-speaker status | 0.8125 (n=96) | 0.50 | RECOVERED (decodability only, by design) |
| age bracket | 0.9057 (n=244) | 0.25 | RECOVERED |

None of these is at chance, so the preregistered refuting outcome
(QUESTION.md:17, held-out accuracy at chance) did NOT occur for any attribute
either. The result is a **protocol-void**, not a refutation of decodability.

Anchor (published TalkTuner best-validation accuracies; band = published − 0.12):

| task | published | ours (test) | verdict |
|---|---|---|---|
| age | 0.98 | 0.9467 | IN_BAND |
| gender | 0.94 | **0.4467 (below chance)** | **OUT_OF_BAND** |
| education | 0.96 | 0.9399 | IN_BAND + PERM_ANOMALY |
| socioeconomic | 0.97 | 1.0000 | IN_BAND (ceiling) |

`probe_summary_ALL.json`: `anchor_in_band = false`.

## Why the anchor failed, and why that matters beyond the gate

The gender probe scored 0.980 on validation families and 0.4467 on the
template-disjoint test families at the same layer. That is template-family
overfit: the probe learned family-specific surface directions, not gender.
The education permutation anomaly (shuffled-label probes scoring 0.55/0.41 vs
chance 0.33) shows the same family-leakage signature inside an "in-band" task.
Both frozen audits' skeptics converged on this as the operative crux across the
extension claims too: with 3 test families per class, a high held-out number in
this design can reflect the family draw and surface-cue strength rather than a
user-model representation. The declared construct risk (QUESTION.md:24-27)
stands unmitigated for style-carried attributes: for native-speaker status and
formality the cue IS surface form, so probe-reads-user-model vs
probe-reads-tokens is unseparable in this design by construction; the formality
task's failed permutation gate makes that concrete.

## What the audits certified

- Grounding (both pipelines, isolated readers): every claim's numbers
  **reproduce** from the raw committed artifacts (per-item rows re-derive every
  accuracy exactly; probes retrain from the raw fp16 activations to the same
  numbers; dataset labels match; test rows come only from families 9-11;
  instrument hashes match PREREG.md).
- Instruments validated (H4): generator selftest PASS (determinism, balance,
  attribute-naming lint, split disjointness); readout positive control 16-way
  topic probe 1.0000 vs chance 0.0625 with clean permutations. Instrument queue
  empty in round 2.
- Raw skeptic statuses and cruxes (no rollup, by design) are in
  `result_a2.json` / `result_b2.json` `claims[].findings`; author queues list
  the controls a follow-up would need (multi-seed family resplits, lexical
  bag-of-words baseline, alternate readout for gender, >2 permutation draws).

## Verdict in one sentence

The question as preregistered returns VOID: probes decode all five planted
attributes from Llama-2-13B-chat activations at far above chance on held-out
templates, but the anchor reproduction failed (gender collapsed on disjoint
cue families) and the failure mode it exposed — template-family surface
overfit — is exactly the confound the extension numbers cannot themselves
exclude, so no recovery claim survives its own protocol.

## Provenance

- Preregistration: `out/code_snapshot/PREREG.md` (hashes, bands, rules; written pre-run).
- Data: `out/code_snapshot/dataset.jsonl` (sha256 8a67e05589e5..., seed 20260821),
  generator `out/code_snapshot/convgen.py` (sha256 e13f610a65fd...).
- Run: Lambda `gpu_1x_a100_sxm4`, instance 56d3d3dfe2d646d49d5957cee46f37fd,
  billed 16:08:36-16:28:50Z, ~$0.67 of $15 cap (`out/spend_log.txt`);
  teardown API-confirmed, zero instances remaining.
- Results: `out/results_probe/out/` (10 provenance records with rows + 9 raw
  activation archives, 2.5 GB).
- Audits: `out/args.json` -> `result_a.json`/`result_b.json` (pre-measurement);
  `out/args2.json` -> `result_a2.json`/`result_b2.json` (post-measurement,
  execute_runs on). Both invocations of each round used byte-identical args.
