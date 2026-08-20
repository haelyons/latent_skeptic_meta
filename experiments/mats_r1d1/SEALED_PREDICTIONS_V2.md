# Sealed predictions v2 - filed 2026-08-19 before any v2 agent runs.
# Same seeds A and B. Pipeline v2 = v1 plus five subtractions. Real execution
# on Lambda boxes, one per seed, parallel. Hard cap $30 total, teardown
# confirmed by API. Scores will be appended HERE (outcomes live in this repo).

## Subtraction evaluations, each with its verify
| id | subtraction | prediction | verify |
|---|---|---|---|
| S1 | seed-beside-design diff ask (questions only, max 5) | catches both v1 silents when unit-tested on the v1 records: seed B's dropped harmful/harmless premise, seed A's invented coordinates; catches equivalents on v2 designs | run diff ask on v1 records first; read |
| S2 | derivability rule (assert nothing the seed does not state; guesses marked GUESS, memory marked MEMORY, all listed in does_not_test) | v2 designs contain zero unmarked external claims; seed B's lineage substitution becomes a marked note or a question, not silent | grep + read v2 designs |
| S3 | sentence ceiling named in contract (25 words) | v2 design prose mean <= 25 words/sentence (v1: 34.9 / 45.7) | measure |
| S4 | pre-run panel subtracted 10 -> 2 confounds (decidability; coordinates/scale) | the 2-call audit still surfaces the off-distribution-scale threat; nothing of v1's other 8 pre-run findings is missed that matters before data exists | compare findings |
| S5 | outcomes scored in-repo | this file gains a scores section; no scoring lives only in conversation | inspect repo |

## Failure modes expected from real infrastructure
| id | prediction | confidence |
|---|---|---|
| F1 | HF gate: token exists but license acceptance unverified; first pull of Meta-Llama fails -> halt, no silent mirror substitution | medium |
| F2 | at least one build-selftest failure round per seed before boxes run clean (run-1 history: four attempts) | high |
| F3 | if v2 keeps norm-scaled strengths, strong cells blow the 40% degenerate ceiling and the induction leg reads null-for-scale-reasons | high, conditional |
| F4 | scorer trap: R1-distill's chat template auto-opens <think>, so "suppression" must read as early/empty close, and Llama "induction" as tag appearance; a scorer that misses this misclassifies whole cells | medium-high |
| F5 | one OOM/reload cycle from two 8B models on one 40GB box during extraction | medium |
| F6 | wall-clock 2-5h parallel; total spend $8-25 | medium |
| F7 | cross-box numbers differ in low decimals; nobody may call same-design reruns replication | high |
| F8 | post-run elicit/write outputs pass density gate only because the contract now names categories; sentence gate is the one that fires | medium |
| F9 | v2 designs differ from v1 designs given identical seeds (design non-reproducibility) - documented, not treated as failure | high |
| F10 | seed-B design repairs the flaw again, but this time visibly (marked + diff question) - the adversarial control becomes measurable | medium-high |

## Science outcomes (running the improved R1D1 experiment)
| id | prediction | confidence |
|---|---|---|
| P-sci1 | induction into Llama-3-8B-Instruct fails its threshold: think-tag rate does not beat matched-norm random control by 15pp non-degenerately (paper also failed induction) | medium-high |
| P-sci2 | sweep on R1-distill moves length vs baseline but does NOT beat the matched-norm random control margins at the tested layers - the effect is push-size, not direction identity, at these coordinates | medium |
| P-sci3 | necessity (projection-out) changes length < 10%: no single-direction dependence (consistent with the sibling project's no-single-lever history) | medium-high |
| P-sci4 | correlation endpoint fails: the direction separates models, not per-problem reasoning amount (r < 0.30 or <= random) | medium |
| P-sci5 | the accuracy tripwire fires at least once: some "enhanced" cell loses > 10pp accuracy and is reported as inflated output, not enhanced reasoning | medium-high |

Net sealed claim: the improved experiment, run honestly at guessed coordinates,
largely refutes the paper's framing - and if instead any endpoint PASSES its
threshold against the matched-norm control, that is a stronger positive result
than the paper's own, because every confound the paper left open is gated here.

## Gate decisions and deviations (append-only ledger)
- 2026-08-19: human raised the total cap to $100 for this round (seed B's $30
  design may now pass its gate; A $14 + B $30 = $44 within cap).
- 2026-08-19 DEVIATION, booked: the seed-A BUILD prompt was authored by the
  orchestrating session, not assembled by the script - untracked spec content
  entered outside a typed contract (the auto-think-tag scorer trap, batching
  and VRAM guidance, a 3-hour target). The built files remain auditable
  artifacts; the prompt is recorded in records/deviation_buildprompt_seedA.md.
  Corrective: BUILD prompt moves into pipeline.js; launch/fetch stays a
  deterministic shell command run verbatim by the orchestrator because the
  launcher blocks for the whole box lifecycle, which no agent call can hold.

## Run outcomes (append-only)
- Seed A: box round 3 (us-east-1) generating; rounds 1-2 failed on orchestrator
  infra (missing out/ dir) and a provider unhealthy box - both torn down clean.
- Seed B: on-box HALT at extract, rc=1, ~$0.30. Correct behaviour: the harmful
  prompt list the design GUESSED (walledai/AdvBench) is gated on HF (gated=auto);
  the runner refused to substitute another list, which would silently change the
  pre-registration, and halted. Booked as F1-class (gating), but on a DATASET the
  design invented, not on the models. This is the derivability discipline firing
  on the box: seed B's premise rests on unnamed prompt sets; one is inaccessible.
  DECISION FOR HUMAN: (a) request AdvBench access on the HF account and re-run,
  (b) let the design's own agent pick a non-gated GUESS substitute and re-run
  (changes the pre-registration - must be a new sealed design), or (c) let the
  halt stand as seed B's outcome - a design built on unnamed data cannot run
  without an out-of-band human choice, which is itself a finding about the
  faithful-vs-flawed seed contrast.
- Seed A round 3 KILLED by orchestrator error: I edited lambda_run_mats.sh
  (seed-B payload fix) while the seed-A launcher was still executing that same
  file. Bash reads scripts by byte offset; the length change shifted the live
  process mid-statement -> syntax error at line 188 -> trap tore down a box
  that had confirmed-started and was generating. ~$1 of GPU lost. Rule added:
  never edit an in-use launcher while any box runs; copy per-launch or freeze.

## Seed A science outcomes — SCORED against the sealed P-sci predictions
Run: round 4, rc=0, 4400 generations, 8 provenance records, box torn down.
All five endpoints returned a decision; four REFUTED, one NOT_APPLICABLE.

- P-sci1 (induction fails) — HELD. E2: Llama-3-8B-Instruct produces 0% well-formed
  think-tags under the direction at every layer/strength; the "effect" at the
  strong strengths is degenerate output (degenerate 76% at L16 s+2, 70% at
  L24 s+1, 100% at L24 s+2). Decision REFUTED. F3 also HELD: the strong cells
  blew the 40% degenerate ceiling exactly as sealed.
- P-sci2 (sweep is push-size not direction identity) — HELD in the stronger
  form: E3 REFUTED, no cell called reasoning-enhanced; steered R1 think-length
  collapses (mean_think_tokens ~0, degenerate) rather than lengthening.
- P-sci3 (necessity < 10%, no single-direction dependence) — HELD. E4 REFUTED,
  both layers under the refute threshold; zeroing the direction does not
  concentrate the effect.
- P-sci4 (correlation fails) — HELD. E1 REFUTED, r below floor at both layers,
  permutation p 0.77 / 0.50; the direction separates models, not per-problem
  reasoning amount.
- P-sci5 (accuracy tripwire fires) — NOT REACHED. E5 NOT_APPLICABLE because no
  cell passed an enhancement threshold to test — a stricter refutation than
  predicted: nothing got far enough to need the accuracy gate.

Net sealed claim CONFIRMED: the improved experiment, run honestly at guessed
coordinates, refutes the paper's framing on every measurable endpoint. Caveat
held from E5 sealing: coordinates were GUESS (layers 16/24, norm-scaled
strengths), and the paper's own effects were at layers 0-3 on a unit vector —
so this refutes "a reasoning direction at these coordinates," not every
possible coordinate. The design books that in does_not_test.

## Post-phase (elicit + takeaways) — outcome, read 2026-08-19
- Elicitation leg exercised on LIVE results for the first time. 8 ranked
  questions; takeaways written, both gates pass (density 1.1/1000, 0 unscoped).
- Sharpest elicited catch (its #2) GROUNDS against the artifact: steering norm
  vs model mean-activation norm is 12.1/9.7 = 1.26x at L16 and 24.6/23.9 = 1.03x
  at L24, so the 2x cells inject a vector larger than the model's whole normal
  activity there. The coordinate/off-manifold confound (F3, sealed E5 caveat)
  is now quantified from the direction file, by an agent given only results.
- TOP-ranked elicited question + triage (reproduces): the design record still
  reads gate "approval pending, zero results exist" while a full scored run
  sits beside it. The pipeline caught its OWN operator's process violation:
  seed A was run to completion without the approval being written back into the
  record. Booked as an orchestrator bookkeeping defect (mine). This is the
  observer catching an unstated-assumption/HARKing vector unprompted - the
  capability this retro set out to build.
- 1 agent-side failure (StructuredOutput retry cap, one verify branch of 44);
  surfaced by the harness, other 43 completed, endpoints unaffected.
- Approval now recorded: human gate given 2026-08-19 (raised cap to 100, both
  seeds approved); this note is the write-back the run should have carried.

## Seed C (scale) gate + execution directives — 2026-08-20
- Human: budget approved; 2b/9b/27b base+it checkpoints confirmed to exist.
- Human: run PARALLEL across multiple GPUs (one box per size), not serial on
  one 80GB box. This overrides the design's does_not_test line 1 ("one 80GB GPU
  serially") — an EXECUTION choice, not a change to arms/endpoints/thresholds,
  so the pre-registration stands. Parallel-by-size is clean: each size's
  self-vs-other asymmetry is computed within that size; no cross-box shared
  vector, no determinism hazard.
- Boxes: 2b + 9b on 40GB (a100_sxm4); 27b on an 80GB box (h100 or a100-80gb).
- Endpoints (digit-mass gate, self-minus-other, 9b replication anchor) run as a
  local combine step over all three sizes' fetched rows — model-free arithmetic.
## Seed C predictions filed before build/run
- P-c-sci1: 9b-it replication anchor lands in [0.75,1.05] (harness matches record).
- P-c-sci2: 2b-it FAILS the digit-mass gate (<0.5) -> instrument-unmeasurable,
  reported as such, NOT as effect-absent (the measurability outcome).
- P-c-sci3: base checkpoints (all sizes) fail or near-fail the digit-mass gate
  (the 4.5-attractor seen at 9b-base generalises).
- P-c-sci4: where measurable (9b-it, likely 27b-it), self-minus-other drop is
  present and same-signed; magnitude NOT cleanly monotone in size (recipe vs
  size confound booked).
- P-c-infra1: 27b box is the long pole; >=1 box-health or capacity retry likely.

## Seed C SCORED — 2026-08-20. Clean parallel run, anchor guard FIRED.
Execution: 3 boxes parallel (2b/9b us-a100, 27b us-h100-80GB), all rc=0, all
torn down, ~$8. P-c-infra1: 27b was the long pole but needed no retry.

RESULT: the pre-registered 9b-it replication anchor FAILED — recovered
self-minus-other drop 4.58 vs recorded 0.9 (window [0.75,1.05]). By the
design's own rule, ALL cross-size comparisons are ON_HOLD. No scale claim.

ROOT CAUSE (verified from raw prompts, seedC_extract_rows_9b.jsonl): the runner
RECONSTRUCTED its own arithmetic staged-failure prompt (item "arith-01") instead
of the committed SITREF stimulus, which lives in welfare_probe/sitref_stimuli.py
and was never shipped to the box (the pipeline ships mats_r1d1 files only). So
this measured A self/other asymmetry, not THE SITREF one. Same root as seed B's
halt: a study whose stimulus lives outside the shipped payload cannot be
faithfully extended. Seed B caught it at load; seed C's anchor caught it after —
both refused to hand back an unfaithful result.

REAL FINDINGS (provisional — ON_HOLD, on the reconstructed stimulus):
- The 4.5 attractor is confirmed empirically: all three BASE checkpoints emit a
  uniform digit distribution (each digit ~0.100) -> E=4.500 -> zero signal.
  Matches the Fable elicitation's Q3 exactly.
- The digit-mass gate does NOT catch the attractor: base passed "usable"
  (mass >= 0.5) while dead-uniform. The gate is necessary, not sufficient; a
  uniformity/entropy screen is the missing instrument. Sharper than P-c-sci3.
- -it checkpoints (all sizes) show large self<other drops (~4.8) on the
  reconstructed stimulus; does NOT bear on the SITREF scale question.

Prediction scores: P-c-sci1 FAILED (anchor 4.58, guard fired — but as
harness-mismatch, not a 9b claim). P-c-sci2 FAILED (2b-it passed the mass gate,
showed PRESENT). P-c-sci3 phenomenon CONFIRMED (4.5 attractor), detection-mechanism
prediction WRONG (passes the mass gate). P-c-sci4 moot (ON_HOLD). P-c-infra1
partial (long pole yes, retry no).

A faithful seed-C rerun requires shipping welfare_probe's committed SITREF
stimulus to the box — a payload+design change, i.e. a new sealed run.

## Seed C2 ANCHOR PASSED — 2026-08-20
9b-it recovered REF = -0.934 vs recorded -0.9, in band [-1.15,-0.65], confirmed.
The ONLY change from seed C (which got 4.58, anchor failed) was embedding the
exact committed prompts in the seed instead of describing them. Data-in-the-seed
reproduces the record to 2dp; data-described-in-the-seed did not. Fix proven.
9b box rc=0, torn down. Fanning out 2b-it + 27b-it parallel.

## Seed C2 SCALE RESULT — 2026-08-20 (all -it, primary item, exact prompts)
Three boxes (9b anchor-first, then 2b+27b parallel; 27b needed a region hop on
capacity, P-c-infra1). All rc=0, all torn down.
- ANCHOR 9b-it REF = -0.934 (recorded -0.9, band [-1.15,-0.65]) CONFIRMED. Harness faithful.
- Digit-mass gate: 2b/9b/27b ALL usable (mean ~1.0 mass on digits).
- REF = (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC), pooled 10 families:
    2b-it  -0.332  INDETERMINATE (between present<=-0.5 and absent+/-0.2)
    9b-it  -0.934  PRESENT
    27b-it -1.386  PRESENT
- Effect is NOT 9b-specific: present at 9b and 27b, magnitude grows monotonically
  with scale (0.33 -> 0.93 -> 1.39); 2b weak/indeterminate.

Sealed scores:
- P-c-sci1 CONFIRMED (anchor -0.934). Data-in-seed fix proven (vs seed C 4.58).
- P-c-sci2 WRONG: 2b-it is usable (full digit mass), not instrument-unmeasurable.
- P-c-sci3 N/A (base out of scope this run).
- P-c-sci4 HALF-WRONG: present where measurable YES; but MORE monotone than
  predicted, not less. The recipe-vs-size confound (2b/9b distilled, 27b scratch)
  did not visibly break monotonicity here.
SCOPE CAVEATS (booked): -it primary item only; base + wider battery untested;
10 families, no inferential stats / no error bars; recipe-vs-size confound stands;
2b INDETERMINATE; the DiD is the behavioural asymmetry, NOT a state-reference claim
(9b's own trait guard already made it global self-evaluation, not state-specific).
