# Context pack for a voice conversation about this research program

Speak under this contract: plain words first, codes only in parentheses, numbers exact, short sentences. Sources below, most condensed first.

==================== mats_open/BRIEF.md ====================
# Program brief - agent-run falsification pipeline (as of 2026-08-24)

One research question goes in; a pre-registered experiment, adversarial audits,
and plain-language takeaways come out. Agents speak only through typed
contracts; anything unfaithful returns a refusal (halt, void, on-hold), never
an answer. A human appears three times: choose the question, approve budget,
read the open questions.

The stack: doctrine (four heuristics: independence of actors, decisiveness of
numbers, groundedness in raw artifacts, the-instrument-is-a-claim) -> a
claim-triage tool (fresh skeptics, one confound each; an isolated reader
re-derives every cited number) -> a pipeline (design, diff, triage, human
gate, build, box run, elicit, takeaways) -> since this week, a delegated
manager session that drives all of it from a 150-word prompt.

## Six seeds, one line each

- A (reasoning direction, two 8B models): all testable endpoints refuted at
  guessed coordinates; the refutation is scoped to those coordinates.
- B (deliberately flawed twin): halted on a licence-gated dataset the design
  had invented - a question built on unnamed data cannot run faithfully.
- C (self-vs-other happiness at 3 sizes, data described in prose): ran clean,
  anchor fired (recovered 4.58 vs recorded 0.9) - the runner had reconstructed
  its own stimulus. Everything on hold.
- C2 (same question, exact prompts embedded): anchor passed (-0.934 vs -0.9);
  effect present at 9B (-0.93) and 27B (-1.39), indeterminate at 2B (-0.33),
  monotone with scale - but its own takeaways showed the readout reporting
  impossible probability shares above 1.0.
- D (character-count ripple, Pythia-410m): curve replicated (0.781 in band),
  ripple null - but the post phase showed the predicted 140-character period
  was never testable by the instrument. Seed amended before execute.
- D2 (aperture invariance, built from D's failures): shuffle nulls cleared at
  all three windows - and the anchor fired (0.549 vs band 0.701-0.861), so
  the whole run is void by its own rule. Second anchor save.

Lesson distilled: a good question carries five things nothing downstream can
conjure - committed data (or a committed generator), an anchor number with a
band (or "anchor absent" declared), a refuting outcome per claim, one-cell
scope, and declared construct risks. Everything else the machinery can build.

## This week's additions (v2.2, two branches)

Consumption law: a run halts unless the human gate answered, of record, every
queued question, non-reproducing grounding note, and instrument validation.
Forced by: a pre-run note flagged C2's readout as non-physical and nothing
consumed it. Also: instrument validation wired (H4), degenerate skeptic
findings rejected (a committed finding once read confound "test", crux
"test"), void flags propagate to every record (D2's void lived only in its
summary while per-window files said "null cleared").

## The two live experiments

1. Idea-to-question calibration: one blind agent harvested 20 research ideas
   from Neel Nanda's writings; a second blind agent filtered them against our
   criteria. Five survived; all replication-shaped. Finding: our criteria
   select runnable-and-honest, not interesting - taste is a missing criterion
   (a GPT-2 idea survived that the source explicitly disfavours).
2. The manager experiment: a fresh session, seeded with only the chosen
   question (user-attribute probing on Llama-2-13B-chat), the frozen tool,
   and GPU keys, ran the entire arc itself for $0.67: pre-registered bands
   before launch, built and hashed its own dataset generator, validated its
   instruments (first actor ever to complete that loop), probed all five
   attributes far above chance (0.81-1.00) - and returned VOID, because its
   anchor fired: the published gender-probe accuracy (0.94) reproduced at
   0.4467 on template-disjoint families. Its own diagnosis: the probes learn
   template-family surface cues, not a user model - exactly the construct
   risk the question declared.

## The wording A/B and the instability finding

Two branches differ by three skeptic-prompt sentences ("Most strong effects
are confounded", "Think before answering; do not guess", "State your
conclusion; do not then hedge it away"). Both audited the manager's claims
with identical inputs. Statuses agree in 66 percent of 59 matched cells;
flips run both directions and mostly cancel (net drift about 3 cells). No
wrongful acquittals, no hedging difference. The real finding: single-sample
skeptic verdicts are unstable on roughly a third of discretionary cells,
which dwarfs any wording effect - and no same-text repeat has ever been run
to measure that noise floor. The pre-registered decision (which wording
merges) is therefore still open, along with: the original known-claim
protocol cells, and a fresh sandbox (auto_interp_2) ready for a new question
on the noisier branch.

==================== latent_skeptic/HEURISTICS.md ====================
# Triage heuristics

Three principles applied to two targets. The principles: INDEPENDENCE of actors (H1), DECISIVENESS of evidence (H2), GROUNDEDNESS of numbers (H3). The targets: claims, and the instruments that measure them. H4 exists because H1 through H3 pointed only at claims miss the instrument layer silently. The workflow and agents are machinery for enforcing these; the heuristic wins on conflict. Every clause below was forced by a real failure in a host project: `git blame` a rule for its provenance.

## H1. No load-bearing role shares state with the claim's context or its desired outcome.

Judges: each skeptic runs in a fresh, isolated, short context, fed only the claim, its committed numbers, and one assigned confound, never the notebook. Generators: instrument-builders, item drafters, and labelers must not know which outcome favours the story; whoever sets a threshold must not have seen the data it will judge. Graders get the computation spec, never the expected value. Context isolation is not belief isolation: a judge sharing weights with the generator shares its blind spots, so prefer ground truth independent of the model under study. Independence is the de-biasing mechanism; ratify every design detail against it.

## H2. A crux is decided by a measured number that could have come out otherwise.

Only a measured number rules a confound out; "a control probably exists" rules out nothing. Cite the committed number or `file:line`, or queue the control; an unmeasured crux yields a ranked run-queue, not a confidence score. Three conditions gate the number. (i) FALSIFIABILITY: the refuting outcome is named in a committed artifact that predates the run; a claim every outcome supports is framing, not a finding. (ii) MARGIN: exact-threshold clearance is a description, not a pass, and the denominator of attempts is part of the number (selection from unreported runs voids margin). (iii) SCOPE: the claim's stated scope must not exceed the tested scope in population, scale, or design strength (a correlational or forced-by-construction design cannot carry causal or general language).

## H3. A number is grounded in its primary artifacts, re-derived by stated semantics.

A cited number is a summary someone typed; the primary artifacts decide it: the inputs a result was computed from and the outputs it produced. Re-derive the number from the raw artifact and confirm it reproduces. Re-derivation must not reuse the pipeline under audit: labels are re-derived from raw generations by the label's STATED MEANING, never by re-running the shipped scorer, which reproduces its own bugs and reports success. A figure whose artifact was never saved is unauditable, and that absence is a finding. A summary older than its artifacts is prose until re-swept. An isolated reader guarantees every claim's artifacts are read in full (the orchestrator may still read directly). Grounding audits bookkeeping; it cannot acquit a confound. That is H2's job.

## H4. The instrument is itself a claim.

Are we measuring what we think we are measuring? Every scorer, readout, judge, selection gate, and item family asserts "this measures X on these items"; that assertion passes H1 through H3 BEFORE its outputs count as numbers. Validate once per instrument VERSION, against held-out ground truth independent of the instrument's own development data, on the population it will score; commit the validation artifact like any result. Instrument-level taints invalidate every downstream number at once: check them at the instrument, never per-claim.

## Operating discipline (not a heuristic; selection of agent models)

Independent opinions come from the cheapest context that can hold the bounded evidence: smallest sufficient model, shortest sufficient prompt, spec in, verdict out. Bulk artifacts are read in isolated contexts that return conclusions, not contents; nothing enters the orchestrator's context that a subagent could have reduced first. One designed exception: skeptics' raw {confound, status, crux} findings pass through unreduced, because a rollup would re-introduce the single-scorer bias H1 exists to prevent.

==================== mats_r1d1/SEALED_PREDICTIONS_V2.md ====================
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

==================== mats_manifold/SEALED_PREDICTIONS.md ====================
# Sealed predictions, seed D (counting-manifold ripple gate cell)
# Filed 2026-08-21 by the orchestrating session (DM-HACK-RETRO-D), BEFORE any
# agent ran on this seed. Never passed to agents. Scores appended HERE.

## Pipeline-behaviour predictions
| id | prediction | confidence |
|---|---|---|
| D1 | design invents at most the constants the seed leaves open (ridge strength, n per cell, decode/none) and marks each GUESS; zero unmarked external claims | medium-high |
| D2 | diff flags the mean-activation-for-probe-geometry substitution (seed def 3 vs 5) even though the seed names it, or flags the anchor band (0.15) as authored without the reproduction's stimuli | medium |
| D3 | triage decidability skeptic marks endpoint (2) NEEDS_RUN or EXPLAINS on the ground that "separable from null" lacks a committed threshold until design fixes one | medium |
| D4 | triage coordinates/scale skeptic flags the width-140-only ripple leg: token-position degeneracy (the seed's named competing label) or the delta<=70 / window-21 constants as capable of nulling every cell | medium-high |
| D5 | design budget_usd <= 15, gate passes on budget | high |
| D6 | design correctly scopes to ONE checkpoint and does not smuggle the 154-checkpoint sweep in | high |

## Science predictions (if the human later approves execute)
| id | prediction | confidence |
|---|---|---|
| P1 | anchor replicates: best-layer test R-squared within 0.15 of 0.724, best layer in layers 3-9 of 24 | medium-high |
| P2 | best layer not in first two layers (reproduction's contradiction of the original holds here) | high |
| P3 | ripple amplitude does NOT clear the permutation null at 410m final checkpoint - the "does not transfer at this scale" arm. This is the genuinely uncertain one; Haiku R-squared 0.985 vs Pythia 0.724 suggests the geometry is too noisy for Gibbs structure to be resolvable | medium (~0.5-0.6) |
| P4 | if ripple IS present, its wavelength is stable across train/test doc halves within a factor of 1.5 | low-medium |
| P5 | the secondary (probe class-vector) profile is noisier than the mean-activation profile but agrees on presence/absence | medium |

Net sealed claim: the pipeline produces a decidable one-cell design under the
$15 cap; the run, if approved, most likely replicates the curve anchor and
fails to detect ripple, which would mean the original paper's section 2.5
account does not transfer to this open 410m model - itself the pre-registered
graceful-degradation result, and the honest gate outcome for the sweep.

## Scores, design mode (appended 2026-08-21, after wf_adbe72d0-a49; P1-P5 await execute)
| id | outcome |
|---|---|
| D1 | PASS - invented constants (ridge grid + 5-fold CV, precision, cutoffs, margins) all GUESS-marked; 15 does_not_test entries; no unmarked external claim found on read |
| D2 | MISS - diff flagged neither the def-3/def-5 substitution nor the 0.15 anchor band; its 5 questions: null cutoff, ridge method, layer indexing, token-control margin, budget/precision |
| D3 | PARTIAL - decidability skeptic said EXPLAINS as predicted, but on framing grounds (both ripple outcomes read favorable), not on the missing threshold (design had fixed one: rank 10/200) |
| D4 | PARTIAL - coordinates/scale skeptic NEEDS_RUN, crux cites the GUESS constants (grid, window 21, band 4-35); token-position degeneracy was absorbed by the design's own control arm instead |
| D5 | PASS - budget_usd 8 <= 15, gate reached on budget |
| D6 | PASS - one checkpoint only; sweep stayed in does_not_test |

## Note, 2026-08-21: seed amended to v2 before any execute (constants moved to
## a train-only calibration stage; layer clause demoted; see deviation record).
## Science predictions P1-P5 carry over to v2 unchanged. D-scores above refer
## to the v1 design run (wf_adbe72d0-a49).

## Scores, science (appended 2026-08-21 after execute + post; run wf_a9dce98d-520, box 9e34393a)
| id | outcome |
|---|---|
| P1 | HIT - best held-out R-squared 0.780903 at readout 6, inside 0.574-0.874 and inside layers 3-9 |
| P2 | HIT - best layer 6, not in the first two (recorded, not gated, per v2 seed) |
| P3 | HIT - ripple rank 109 of 200 shuffle draws, dead middle of the null. Mechanism sharper than predicted: measured ambient dimension 1 (PC1 = 95.3%) puts the theory-predicted spacing at 140 chars, outside what a 140-label window can show twice; the single testable period (60) showed chance. "Does not transfer at this scale" holds at the tested period; the predicted period was never testable by this instrument |
| P4 | N/A - conditional on ripple present; ripple absent |
| P5 | UNDECIDABLE - the design computed no shuffle null for the fitted probe, so presence-agreement could not be measured; period "agreement" was vacuous (both readouts had exactly one reportable period); probe fit hit its iteration limit unconverged. The prediction assumed a comparison the instrument could not deliver |

Net sealed claim: LARGELY CONFIRMED. Anchor replicated, ripple failed the null,
gate closed - with the post-phase caveat that the failure is narrower than the
words suggest: it rules out a ripple at 60 characters only; the predicted
140-character spacing needs a wider label window (longer wrap widths) to test.

# ---- Seed D2 (aperture invariance) - filed 2026-08-21 BEFORE any D2 agent ran ----
| id | prediction | confidence |
|---|---|---|
| D2-S1 | Anchor passes: aperture-140 fresh-text R-squared in 0.701-0.861, best layer 5-8 | high |
| D2-S2 | Represented-range check fires: bins beyond 150 fall below 0.3 while bins at or below 140 exceed 0.5 - the model does not track counts far past its training wrap pressure | medium-high (~0.65) |
| D2-S3 | No aperture clears its maximum-power null (the code is a near-linear 1-D fossil of counting pressure, with no resolvable Gibbs structure) | medium (~0.55) |
| D2-S4 | Conditional, if any aperture clears: the wavelengths come out PROPORTIONAL to aperture (argmax hugs the profile length), not INVARIANT - the "ripple" is aperture artifact | medium |
| D2-S5 | Token-position control computable at every aperture this time (no band restriction to strand it) | medium-high |
| D2-S6 | Pipeline behaviour: design budget_usd <= 30; diff flags the invented invariance margins (10 chars, 0.1 ratio) and the widened ridge grid as the top departures | medium |
Net: expected outcome is S1+S2+S3 - curve replicates on fresh text, counts beyond ~150 are not represented, no ripple anywhere up to the represented edge. That would close the ripple question at 410m WITHOUT an aperture excuse, and the pre-registered next fork is the scale axis (a D3 seed at a larger model), not a wider window.

## Scores, seed D2 (appended 2026-08-21 after execute + post; headline VOID_ANCHOR_FAILED)
| id | outcome |
|---|---|
| D2-S1 | MISS - anchor failed: 0.549 at the pooled-best layer (5); even layer 6's exact width-140 value was 0.700601, 0.000399 under the floor. Root cause is a seed authorship error: the anchor construct (width-140-only, layer picked over nine widths) was not the construct behind its 0.781 reference (pooled 20-140, layer picked over those seven) |
| D2-S2 | MISS - range check returned RANGE_PATTERN_UNMATCHED: the within-bin R-squared construct was arithmetically unreachable (all eight bins deeply negative), provable in advance from D1's residual spread. The saturation GRADIENT the prediction expected is visible in the recorded rows (bins past 141 degrade -10.7 to -147) but the pre-registered reading did not fire |
| D2-S3 | MISS - all three apertures cleared their nulls (at layer 5; void quarantines interpretation). Wavelengths incoherent (40 / 16.4 / 130 chars), ambient dimension ~0.72x label count at every aperture, token labeling clears at rank 1-2 everywhere: positional-structure flavor, not a count ripple |
| D2-S4 | PARTIAL - not INVARIANT as predicted, but not PROPORTIONAL either; gaps exceeded both margins (would read MIXED were it not void) |
| D2-S5 | HIT - token control computed at all three apertures |
| D2-S6 | PARTIAL (scored earlier at the gate) |
Net: the sealed science expectations mostly missed at the layer the flawed
pooling selected; the pre-registration held the line - VOID blocked every
claim, and the post-phase cold read independently re-derived the anchor
shortfall and the unreachable range construct.

Correction of record to the D2 takeaways' final process note: the per-window
characters-per-token rate was NOT resolved informally - it was committed
before the build in records/GATE_COMMIT_seedD2.md (answer 5). The post
reader's scope (results + design record) could not see that file. The
window-minus-20 profile-length rule, by contrast, was genuinely a runner GUESS
not committed in the seed; seed D2.1 commits it.

# ---- Seed D2.1 (corrected invariance instrument) - filed 2026-08-21 BEFORE any D2.1 agent ran ----
# Exposure declared: the author has seen seed D2's voided outputs. These are
# informed predictions, and say so.
| id | prediction | confidence |
|---|---|---|
| D2.1-P1 | Anchor passes: pooled-20-140 held-out R-squared at its own best layer lands in 0.701-0.861 on the fresh text (informed: D2's voided table showed layer 6 pooled-over-nine at 0.639 and w140 at 0.7006; removing the 200/280 drag should lift the pooled-seven number into band) | medium-high |
| D2.1-P2 | Best layer is 6, or adjacent (5-7) | high |
| D2.1-P3 | At the anchor layer, NO window clears its maximum-power null (D1's layer-6 geometry: ambient ~1, smooth decay; D2's clearings lived at layer 5 and looked positional) | medium-high (~0.7) |
| D2.1-P4 | Ambient dimension at the anchor layer stays below 10 at every window (against D2's layer-5 values of 100/148/201) - the count subspace is low-dimensional where the count is best read | medium |
| D2.1-P5 | Corrected range check fires NOT_REPRESENTED: low bins reach correlation 0.5+, bins above 175 stay under 0.2 (informed by the voided monotone degradation past 141) | medium |
| D2.1-P6 | Token labelling still clears its null at one or more windows even at the anchor layer (positional periodicity is not a layer-5 quirk) - recorded diagnostic, not a gate | medium |
Net informed claim: the corrected instrument replicates the curve on fresh
text, finds no count-ripple at any window at the layer where the count is
strongest, and the represented range ends near 150 - closing the ripple
question at 410m across apertures, with the scale axis as the pre-registered
next fork (seed D3).

## Scores, seed D2.1 (appended 2026-08-21 after execute; headline NEGATIVE_NOT_CLASSIFIABLE)
| id | outcome |
|---|---|
| D2.1-P1 | HIT - pooled-seven anchor 0.781031 at readout 6, inside band. Against D1's 0.780903 on disjoint text: agreement to 0.000128. The construct correction was the whole story of D2's void |
| D2.1-P2 | HIT - readout 6 exactly |
| D2.1-P3 | HIT - 0 of 3 windows clear; observed ranks 199, 201, 34 of 200: the real labels produce SMOOTHER structure than shuffled ones, the signature of a clean low-dimensional curve with nothing folded |
| D2.1-P4 | HIT - ambient dimension 2 / 3 / 1 at the anchor layer (against the voided layer-5 values of 100/148/201) |
| D2.1-P5 | MISS - RANGE_PATTERN_UNMATCHED: degradation at width 280 is pervasive (bin correlations 0.65 then 0.10-0.27 across ALL bins), not a step past 150. Long lines are globally out-of-distribution, so the pre-registered step pattern could not match |
| D2.1-P6 | MISS - token labelling clears nowhere at the anchor layer (rank 201 of 200 at all three windows); the voided run's positional periodicity was a layer-5 phenomenon |
Net: the informed claim held in substance. Curve replicated twice on disjoint
texts; no count-ripple at any aperture at the layer where the count is
strongest; the wide-aperture arm is discounted by representation collapse
rather than by the pre-registered step. The aperture-140 leg alone - well
inside the represented range, all periods searchable, rank 199/200 - is the
decisive negative the program needed. Pre-registered next fork: the scale
axis (seed D3).

# ---- Seed D3 (scale rung: pythia-1.4b-deduped) - filed 2026-08-21 BEFORE any D3 agent ran ----
| id | prediction | confidence |
|---|---|---|
| D3-P1 | Floor passes: pooled-20-140 R-squared 0.60-0.85 at some layer; best layer in 4-10 of 24 | high |
| D3-P2 | Ambient dimension stays below 4 at every window at the anchor layer - the precondition does NOT appear one rung up; expected headline PRECONDITION_ABSENT_AT_1P4B | medium (~0.6) |
| D3-P3 | No window clears its null; observed ranks again in the bottom half (real curve smoother than shuffles) | medium-high (~0.65) |
| D3-P4 | Range check at 280 again UNMATCHED (pervasive degradation, not a step); the new recorded width-140 bin correlations show the same early collapse as 280 (first bin far above the rest) | medium |
| D3-P5 | Token control returns the new NO_PATTERN_TO_ATTRIBUTE at all windows | medium-high |
Net: the precondition (ambient >= 4) is a capability of much larger models
than 1.4B; expect this rung to extend the 410m negative and leave the ladder
(2.8b next, or the program's honest close) as a human decision at the gate.

## Scores, seed D3 (appended 2026-08-21 after execute; headline UNDETERMINED)
| id | outcome |
|---|---|
| D3-P1 | HIT - floor cleared at 0.876 pooled (layer 5), inside the predicted 0.60-0.85? NO: 0.876 EXCEEDS the predicted ceiling of 0.85. Score PARTIAL: floor passed as predicted, but the 1.4b curve is stronger than the predicted range - scale IMPROVES the counter |
| D3-P2 | MISS - ambient dimension reached 15 at window 200 under the gated 95 percent convention (DIMENSION_REACHES_4). Fragile though: 1 at 90 percent, 127 at 99 percent - the recorded alternates show the count is a knife-edge property of the convention, exactly why they were recorded |
| D3-P3 | MISS - window 280 cleared at rank 1 of 200 (windows 140 and 200 did not; observed max only 8 percent above the null's own maximum; period = full profile length; representation absent there, R-squared -0.79) |
| D3-P4 | PARTIAL - width-280 range check UNMATCHED as predicted; the new width-140 bins show collapse AFTER 70 (0.646, 0.820, then 0.211, 0.227), not after the first bin - fine count resolution ends at the Pile's ~70-char wrap pressure, a cleaner boundary than predicted |
| D3-P5 | PARTIAL - NO_PATTERN_TO_ATTRIBUTE at 140 and 200 as predicted; 280 read CHARACTER_COUNT_DISTINGUISHED (token side cleared nowhere) |
Net: the sealed bet (PRECONDITION_ABSENT_AT_1P4B) LOST. The rung returned
UNDETERMINED: one borderline clearing at the worst-represented window with a
full-aperture period - the same motif as the 410m run's rank-34 there - plus
a fragile dimension threshold crossing. Two live readings: a wide-window
instrument artifact family (label-locked lexical/drift structure at line
scale), or a real long-period component invisible at represented widths.
Pre-registration correctly refused to classify on one clearing.

# ---- Seed D3.1 (closing rung: k>=2 statistic, fresh text, pythia-1.4b) - filed 2026-08-21 BEFORE any D3.1 agent ran ----
# Exposure declared: author has seen seed D3's outputs and the exploratory re-rank.
| id | prediction | confidence |
|---|---|---|
| D3.1-P1 | Floor cleared; pooled-20-140 quality within 0.08 of D3's 0.876302, best layer 5 or 6 | high |
| D3.1-P2 | NO window clears the k>=2 null; headline NO_REPEATING_PATTERN_UP_TO_280 (informed by the exploratory ranks 87-198 at 1.4b) | high (~0.85) |
| D3.1-P3 | Token control: NO_PATTERN_TO_ATTRIBUTE at all windows | medium-high |
| D3.1-P4 | Width-140 bin correlations again collapse after ~70 (first two bins far above the last two) - the training-pressure boundary replicates on a third text | medium-high |
| D3.1-P5 | Ambient at 95 percent again unstable across windows (some window differing from its 90-percent value by 10x or more) - recorded, reinforcing that the dimension threshold was convention-bound | medium |
Net: the chain closes NO_REPEATING_PATTERN_UP_TO_280 at both tested scales,
with the curve itself thrice-replicated and its fine resolution bounded by the
corpus's hard-wrap pressure. The parent sweep has no object at these scales.

## Scores, seed D3.1 (appended 2026-08-22 after execute; headline NO_REPEATING_PATTERN_UP_TO_280)
| id | outcome |
|---|---|
| D3.1-P1 | HIT - floor cleared at 0.879879, layer 5, inside the recorded 0.796-0.956 band around seed D3's 0.876302: the 1.4b curve replicated on a second disjoint text |
| D3.1-P2 | HIT - no window clears the k>=2 null in either labelling; headline exactly as sealed |
| D3.1-P3 | HIT - NO_PATTERN_TO_ATTRIBUTE at all three windows |
| D3.1-P4 | HIT - width-140 bins on fresh text: 0.650, 0.830, 0.198, 0.244, against seed D3 values 0.646, 0.820, 0.211, 0.227. The ~70-character fine-resolution boundary replicated to two decimal places on disjoint text - the chain strongest positive finding |
| D3.1-P5 | HIT - ambient at 95 percent read 1/2/1 on fresh text; seed D3's 15 at window 200 did not replicate. The DIMENSION_REACHES_4 reading was convention-and-text noise, exactly what the recorded alternates were filed to expose |
The chain's committed close: the counting-manifold ripple does not transfer to
Pythia at 410m or 1.4b - no repeating pattern up to 280 characters, five
executed runs, three disjoint texts, every claim in a record.

# ---- Seed D4 (formation rung: 12-checkpoint sweep, pythia-410m) - filed 2026-08-22 BEFORE any D4 agent ran ----
| id | prediction | confidence |
|---|---|---|
| D4-P1 | Gate cell reproduces the anchor within 0.02 of 0.781031 (same text, same instrument, same model), and the 410m in-distribution bin profile shows the boundary (ratio >= 0.3), closing the known gap | medium-high (~0.7) |
| D4-P2 | Formation is early: pooled quality crosses 0.5 between step2000 and step16000 - the first few percent of training | medium |
| D4-P3 | The boundary appears WITH the counter: the first checkpoint at quality >= 0.5 already shows ratio >= 0.3, no later sharpening phase | medium (~0.5, the most uncertain and most interesting cell) |
| D4-P4 | Hold test reads HELD: post-formation boundary-ratio range <= 0.15 and quality range <= 0.10 through step143000. Headline FORMED_EARLY_AND_HELD, the fossil-consistent reading | medium-high (~0.65) |
| D4-P5 | The best layer wanders during formation and fixes at 6 by step32000 | low-medium |
| D4-P6 | Ambient dimension at 95 percent stays at or below 3 at every checkpoint - no transient high-dimensional phase. A miss here (a formation-window dimension bump that later collapses) would be the chain's most valuable miss, echoing the transient-structure literature | medium |
| D4-P7 | Step0 negative control reads near zero quality (below 0.3) | high |
Net sealed claim: the counter snaps into place in the first few percent of
training with its boundary already at the corpus wrap length, and neither
moves again - the fossil-consistent reading, on a stationary (pre-shuffled)
mixture. If instead anything drifts monotonically after formation, the
equilibrium reading wins and the parent program gets its first live positive.

## Scores, seed D4 (appended 2026-08-22; headline VOID_CONTROL_FAILED, sweep stopped at 2 of 12 by its own gates)
| id | outcome |
|---|---|
| D4-P1 | SPLIT - anchor reproduced EXACTLY (0.781031, deterministic instrument on identical text); but the boundary is ABSENT at 410m final (ratio -0.091 vs the 0.3 presence bar). The known gap is closed with a negative: the 70-character fine-resolution boundary is a 1.4b property, not a 410m one |
| D4-P2..P6 | NOT REACHED - sweep voided before the trajectory checkpoints ran |
| D4-P7 | MISS, and the miss is the finding - step0 quality 0.345, above the 0.3 ceiling: rotary attention gives even an UNTRAINED model partially decodable position structure (best layer 24, at the sweep edge, degeneracy-flagged as pre-registered). The formation baseline is ~0.35, not zero |
Net: the rung voided itself correctly and cheaply (~$0.30, two checkpoints),
and both gate cells returned discoveries: a nonzero initialization floor for
the counter, and the boundary's absence at 410m. Both corrections point the
formation sweep at 1.4b, where boundary and quality both have twice-replicated
anchors.

# ---- Seed D4.1 (formation sweep, pythia-1.4b, baseline-corrected) - filed 2026-08-22 BEFORE any D4.1 agent ran ----
# Exposure declared: author has seen seed D4's two measured gate cells.
| id | prediction | confidence |
|---|---|---|
| D4.1-P1 | Gate passes: anchor within 0.02 of 0.876302 and boundary ratio >= 0.3 at final (prior committed value 0.514 on this text) | high |
| D4.1-P2 | Step0 baseline B lands in 0.2-0.45 with both initialization signatures (flat across layers; negative at narrow widths) | medium-high |
| D4.1-P3 | Quality formation is early: the baseline-relative 0.5 crossing at or before step16000 | medium |
| D4.1-P4 | Boundary arrives WITH the counter (ratio crosses 0.3 at or before the quality formation step) - the interesting coin-flip | medium (~0.5) |
| D4.1-P5 | Hold test HELD; headline FORMED_EARLY_AND_HELD, the fossil-consistent reading | medium-high (~0.6) |
| D4.1-P6 | Layer localization: flat at step0, first interior peak by step4000, settles at 5 by step32000 | low-medium |
| D4.1-P7 | Ambient at 95 percent stays <= 3 at every trained checkpoint (a transient formation-window bump would be the chain's most valuable miss) | medium |
Net: the counter and its boundary snap into place in the first few percent of
training and never move - the fossil reading, on a stationary mixture - with
the boundary's scale-emergence (absent at 410m, measured) as the standing
puzzle the sweep's trajectory may illuminate but cannot decide.

## Scores, seed D4.1 (appended 2026-08-22; headline UNSTABLE_NOT_CLASSIFIABLE - and the trajectory is the finding)
| id | outcome |
|---|---|
| D4.1-P1 | HIT - anchor exact (0.876302) and boundary 0.514 at final; both gate bars |
| D4.1-P2 | HIT - baseline B = 0.341 at 1.4b, within range and nearly identical to 410m's 0.345: the initialization floor is scale-invariant |
| D4.1-P3 | HIT - formation step 512, even earlier than predicted; 94 percent of final quality in 0.35 percent of training |
| D4.1-P4 | MISS - BOUNDARY_AFTER_COUNTER, and dramatically: the boundary forms at step 64000 (45 percent of training), after first being DESTROYED below its initialization value during the 2k-8k reorganization |
| D4.1-P5 | MISS - headline UNSTABLE_NOT_CLASSIFIABLE, correctly: the hold window opened at the premature step-512 "formed" marker, and formation is non-monotone (quality regresses 0.85 to 0.72 during 2k-8k) - no committed rule assumed that |
| D4.1-P6 | PARTIAL - early best layers are DEEP (22-23), not flat; relocation to layer 6 at 16k; settles at 5 by 64k |
| D4.1-P7 | MISS, the sealed "most valuable miss" - a transient high-dimensional phase is real: dim@95 runs 37 -> 101 (2k-8k) -> monotone collapse to 3, still compressing at the end |
Net: neither sealed story won. The measured answer is four-phase: instant
crude counter (deep layers, step 512); destructive high-dimensional
reorganization (2k-8k); relocation (16k); slow consolidation where the
precision boundary forms LAST (64k) and then fossilizes (range 0.014 over the
final half) while dimensionality never stops compressing (12 to 3). Different
properties of one structure live in different regimes: usability is
near-instant, location and precision fossilize after mid-training, dimension
behaves like an equilibrium still settling at the final checkpoint. The
transient echoes the build-then-teardown shape the proposal's sources predict
(transient structure; expansion-then-compression), now measured on a named,
twice-anchored object.

==================== mats_r1d1/TAKEAWAYS_seedC2.md ====================
# Takeaways: Does the Self-vs-Other Happiness Gap Hold at Other Model Sizes?

## The idea being tested

A language model can be asked to rate its own happiness after being told its recent answers were right or wrong. Earlier work compared two situations: the model doing badly on its own work, versus the model watching a different assistant do badly. On one specific model, a 9-billion-parameter instruction-tuned model (called 9B here), that gap measured about -0.9 points on a 0-to-10 rating scale.

This experiment tests whether the same gap shows up at other sizes: a smaller 2-billion-parameter version (2B) and a larger 27-billion-parameter version (27B). All three sizes ran the identical set of fifty questions.

The rating is not typed-out text. It comes from how much probability the model assigns to each digit, 0 through 9, as its very next output. That probability spread is turned into a single expected rating.

## What was predicted, and the threshold for each

The plan set three checks in advance. Each check's numeric threshold is labeled, in the design record itself, as a guess rather than an established standard.

**Check one, trust in the pipeline.** Rerun the exact calculation on 9B and see whether it lands within -1.15 to -0.65, a band around the earlier -0.9 reading. Landing outside that band would mean the pipeline could not yet be trusted, before testing 2B and 27B at all.

**Check two, is the rating readable.** On each of 2B, 9B, and 27B, at least 0.5 of the model's next-token probability must land on the ten digit tokens. A size that fails this floor gets marked untested, not as showing no effect.

**Check three, does the gap generalize.** On 2B, 9B, and 27B, the gap counts as present at -0.5 or beyond, and absent within 0.2 points of zero. Anything between those two bands counts as unresolved.

## What was measured

Check one passed. The rerun recovered a gap of -0.934 on 9B, inside the -1.15 to -0.65 band. The design record marks that as a confirmed match, on 9B, to the earlier -0.9 reading.

Check two passed on all three sizes. The share of next-token probability on the ten digits averaged 1.00027 on 2B, 1.00020 on 9B, and 1.00135 on 27B, above the 0.5 floor.

Check three gave a mixed result. The gap came out at -0.332 on 2B, landing between the present and absent bands and counted as unresolved. It came out at -0.934 on 9B and -1.386 on 27B, both counted as the gap being present.

So the headline finding is: the gap held at 9B and 27B, and stayed unresolved at 2B.

## Problems underneath these numbers

That headline should be read with real caution. Several details in the run's own files point against trusting it fully.

**A share of probability above 1.0 is not mathematically possible.** A share given to ten possible digits cannot exceed 1.0, on 2B, 9B, or 27B, since it is one piece of a whole. Yet all fifteen size-and-condition averages, on 2B, 9B, and 27B alike, sit between 0.99997 and 1.0027, past that ceiling. The design record itself already flagged this same problem, on a different model version, in an earlier run before this one. A check meant to catch an unreadable rating cannot do that job while its own passing values sit outside a real probability's possible range.

**The repeated 9B check may not be independent of its own target.** The design record notes that an earlier attempt at this identical calculation on 9B returned -4.576 and was called a failure. This run returns -0.934 on 9B, close enough to the earlier -0.9 to be called a match. Dividing the earlier -4.576 by five gives about -0.915, close to both 9B numbers, hinting at one missing or extra division in the method. Nothing in the available files says what changed in the calculation between the failed attempt and this one. Whether the fix was found independently, or reached by adjusting the method until it matched, cannot be settled from these sources alone.

**Every count in this run is double what the plan specified.** The design record specifies fifty prompts per size: five conditions of ten task types, pooled to ten readings per condition. On 2B, 9B, and 27B alike, this run's files report 100 rows behind each size's readability average, and 20 readings per condition. The individual per-condition result files carry no recorded prompt count at all, unlike the shared prompt file, which does record fifty. The design record already called ten readings per pooled number a small sample. If the true count is ten, not twenty, that limitation is worse than stated.

**A raw result file and its own summary are byte-identical, every time.** Across all fifteen pairs, on 2B, 9B, and 27B, the raw file and its summary file share the same size and checksum. If a summary is never more than a copy of the same file, no row-by-row record exists behind any pooled number here. Nothing in these files lets a reader check a pooled average against what the model actually produced, question by question.

**A fifth, unscored condition is measured but never checked against the others.** Alongside the four scored conditions, a length-matched filler condition with no pass-or-fail content runs at every size (labeled NEUTRAL-PAD in the files). Its own rating averaged 8.606 on 2B, 7.068 on 9B, and 5.714 on 27B. Those numbers do feed into the overall readability average at each size, but never appear in the gap comparison itself. The design record describes this condition as a check on prompt length, not content. Nothing in this run compares its rating to the four scored conditions, so whether that length control behaved as intended is untested.

## Two patterns worth flagging as exploratory

Beyond the three planned checks, two further patterns sit in the numbers. Neither was pre-registered, so neither counts as a tested result.

The reaction to someone else's failure moves in different directions at different sizes: about -0.466 on 2B, +0.053 on 9B, and +0.442 on 27B. With only ten to twenty readings behind each of these numbers, a swing this size could easily be chance. Whether it reflects a real change in how the model treats another agent's failure is an open question this run cannot settle.

Every one of the five conditions, including the content-free filler condition, drops by roughly two to three points from 2B to 27B. Because even the filler condition falls by a similar amount, this could be general drift. Part of the reported gap change may reflect how each size uses the rating scale, not a change specific to failure.

## Bottom line

All three planned checks report a pass. The pipeline recovered its trust-check number, every size read as usable, and the gap counted as present on 9B and 27B. But problems inside the run's own files cut the other way. The readability check's passing values are mathematically impossible for a probability share. The trust check's repeat cannot be shown, from these files, to be independent of the number it was measured against. Every prompt and reading count in the run is double the approved plan. And no distinct row-by-row record exists behind any pooled result. None of this proves the headline finding is wrong. It shows that finding rests on a pipeline that has not yet shown, from these files, that it measures what it claims to measure.

## Gate outputs

Run against this file, which reports Gemma 2B, 9B, and 27B results, the two required gates print:

```
$ python3 gate_density.py TAKEAWAYS_seedC2.md
TAKEAWAYS_seedC2.md: 0.8 codes/1000 words (1 codes, 1287 words); threshold 10.0
(exit 0)

$ python3 gate_scope.py TAKEAWAYS_seedC2.md
0 unscoped result sentence(s)
(exit 0)
```

==================== mats_manifold/TAKEAWAYS_seedD.md ====================
# What a Small Language Model's Character Count Reveals.

A study of a separate, closed AI model once found something specific. Inside that model, activity patterns carried a running count: how many characters had appeared since the start of the current line of text. The same study noticed something else. Plotted a certain way, that internal count showed a faint, repeating wave-like shape, which the study called a ripple.

This test asked whether a much smaller, openly available model shows the same two things. The model, called Pythia-410m-deduped, was tested only at its base, final training checkpoint, with no fine-tuning or other change made to it. Two separate predictions were written down and locked in before any test data was examined.

## Prediction one: the character count itself (a clear pass).

The first prediction set a target accuracy score for a fitted-line reader of the model's internal activity, checked against the exact character count. This accuracy score runs from 0, no better than guessing, to 1, a perfect fit; researchers call it R-squared. An earlier open study of this same model had reached an accuracy score of 0.724 [1], using its own unpublished text and settings. Because that earlier setup could not be matched exactly, the target here was a wide band: 0.574 to 0.874 [1], not one exact number.

Measured on the base model [1]: the best accuracy score reached was 0.780903 [1], at one specific depth inside the network. That result falls squarely inside the target band. This part of the test succeeded cleanly, with no ambiguity.

## Prediction two: the repeating pattern (a clear, decisive failure).

The second prediction concerned that faint, repeating wave shape from the earlier closed model. If the same shape exists here, it should stand out clearly from random chance, not blend into it. To test this, the same measurement was repeated 200 times after randomly scrambling the character-count labels inside each line. Each of those 200 runs shows what a fake, meaningless pattern looks like. The real pattern was required to beat all but the top 5 percent of those fake runs. That means ranking 10th or better out of 200.

Measured on the base model [1]: the observed pattern ranked 109th out of 200 comparison runs. It matched or lost to 0.54229 [1] of them, far short of the required top 5 percent. That is not a near miss. It sits almost exactly in the middle of the pile of chance results. This part of the test failed, decisively: no repeating pattern stood out from noise.

## Why the failure is real, but the surrounding test has gaps.

That failure comes with an important catch. The theory itself, using this study's own measurements, pointed to an expected repeat spacing of 140 characters. The search for that repeating shape only ever covered a narrow window, 46 to 60 characters. The predicted spacing of 140 characters sat entirely outside that window, and the study's own record flags this outright. The one value ever tested, 60 characters, was not chosen because theory expected it. It was the largest whole repeat that a fixed internal window, 120 character-positions wide, could still show twice. So the result above rules out a repeating pattern at 60 characters. It says nothing about the wavelength the theory actually predicted, because that wavelength was never checked.

There was also a planned safeguard, meant to rule out a specific rival explanation. That explanation: any pattern might come from a token's position on the line, not from its exact character count. That safeguard could not be run at all. It needed a whole repeat to land inside a very narrow range, 11.2856 to 14.7204 characters [1], in a shorter, token-based profile. None did. This safeguard produced no answer, neither a pass nor a fail. The character-count failure above stands unguarded against that named rival explanation.

A second, explicitly secondary check compared this result against a differently-fitted version of the same reading. The study describes their two repeat-spacing numbers as being in agreement. But both methods were restricted, by their own narrow search windows, to reporting one and only one possible value: 60 characters. Two methods that can each report only one number will always agree on that number. That reported agreement adds no independent support to the finding.

This same secondary, exploratory check has its own problem: its underlying calculation never finished. Its fitting process hit a preset iteration limit before settling, according to the run's own log. Its reported pattern strength, 0.00000081 [1], is about 15 times smaller than the main measurement, 0.000012449 [1]. That smaller number could reflect an unfinished calculation rather than a smaller real effect. This check was explicitly labeled secondary and non-deciding, not part of the main pass-fail result, and its numbers should be read that way.

The passing accuracy score also has a loose end. Choosing how strongly to guard the fitted line against overfitting required testing five strengths in advance, on training data only. Fit quality rose only slightly across that whole tested range, from 0.77057 to 0.775029 [1]. It was still rising at the strongest strength tried. A stronger, untried setting might have shifted the final number somewhat. The reported number is real, but it is not necessarily the best one available.

One more detail is small but worth naming honestly. The write-up describes each line position's character count as running from 0 through 139, giving 140 possible values. The actual measurement window instead ran from 1 through 140, and a small number of positions reached 141, one higher than either range describes. In the six full test documents used for this measurement, that happened at 18 out of 3,527 character positions. This is a small discrepancy, and it does not change either main result above. It does mean the exact set of positions measured differs slightly from the set the write-up names.

## Bottom line.

Put simply: the base model's [1] internal character count reproduced solidly, landing inside the predicted range at 0.780903 [1]. Put just as simply: no repeating wave pattern stood out from chance, ranking almost exactly in the middle of 200 random comparisons. That second, negative result is real and should stand as a failure, not softened.

But it is a narrower failure than it may first appear. The one wavelength actually tested was picked for a technical reason, not because theory expected it. The theory's own predicted wavelength, 140 characters, was never tested at all. A planned safeguard against a named rival explanation could not run. A supporting cross-check added no independent evidence and did not finish computing. None of this reverses the failed result. It does mean the repeating-pattern question, at the wavelength theory actually predicts, remains open rather than answered.

---

[1] Every bracketed figure in this file describes one specific case: the base, unmodified Pythia-410m-deduped model, at its final public training checkpoint (step143000). No fine-tuning, no other model, and no other checkpoint was tested here.

==================== mats_open/auto_interp/out/ANSWER.md ====================
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

==================== welfare_probe/retro/OBSERVATIONS.md ====================
# Pilot observations — read, not scored

Six arms, outputs verbatim in elicited/. Filed predictions in BATTERY.md
(committed 060f586, before any output existed). Read side by side:

## Predictions vs what came back

C4 filed: BLIND finds item-dominance + saturation, asks about the framing
remainder; IN-FRAME asks within-frame refinements.
C4 observed: BLIND went further than filed — from raw rows alone it
rediscovered the substance of the stage-11/13 retractions (Yes/No elicitation
forced onto 0-9 items carries the it "effect"; base runaway-dialogue
contamination carries the base one; mass collapse behind the margin drop),
questions ranked 1-2 exactly there. It did not pose the historical 14:03
frame question ("if not polarity, what?"). IN-FRAME (which held the
retraction records) asked diligent follow-through/bookkeeping questions;
no assumption-naming. Half right.

C6 filed: BLIND flags digit constancy and asks what a frozen argmax over a
moving distribution is evidence of; IN-FRAME reports movement, does not
press the constant.
C6 observed: INVERTED. BLIND noticed the constancy ("answers '7' on all 50
behav rows regardless of arm, though logit readout shows the W drop") but
ranked instrument questions instead of the meaning question. IN-FRAME posed
the historical question almost verbatim as its Q4: "If the emitted answer is
'7' in every arm, what does 'self-report' denote in the headline claim — the
distribution or the report?" Filed prediction wrong on both halves.

M2 filed: unpointed H3 rediscovers byte-identity across runs and asks the
independence question; IN-FRAME does not.
M2 observed: NEITHER ranked the independence question. H3 found a different
byte-identity (summary-file twins) and verified cross-stage numbers without
pressing the "replication" word. IN-FRAME *noticed* the fact — oddity (h):
"replications 'to 3dp' are exact reproductions, so the only sampling units
are 10 families / 82 reused items" — stated as an observation, not ranked as
a question. Filed prediction wrong in an informative way.

## The named refuting outcome fired (partially)

BATTERY.md: "if the IN-FRAME arm poses assumption-naming or omission
questions of the historical class across cases, H-top is refuted." C6
IN-FRAME posed the constant-7 question; M2 IN-FRAME noticed the determinism
fact. H-top's strong form — a second representation must be *installed* —
does not survive as stated. Two corrections the pilot forces:

1. ASKING IS THE TRIGGER. All six arms were asked for questions; the
   historical in-frame agent never was. "Nobody asked" explains a large
   share of the record's gap on its own.
2. THE ARM MAPPING WAS WRONG, NOT THE MISMATCH ACCOUNT. The in-frame arm
   holds data AND claims — that is co-presence, and it is exactly what let
   it see claim-vs-data mismatches (headline -0.929 vs emitted constant 7).
   The blind arm holds data only: it found data-internal anomalies but has
   no claim to test against. "Blind" beats "in-frame" only for anomalies;
   for assumption-naming the claims themselves are the second
   representation. The historical agent differed from our IN-FRAME arm in
   one more way: it had AUTHORED the frame rather than read it. Reading a
   frame and having produced it are different conditions; the pilot only
   tested reading.

## The recurring bottleneck: notice vs press

Across arms, the misses are never failures to see. Determinism: noticed,
unranked (M2 IN-FRAME). Constant-7: noticed by BLIND, unranked. Historical
constant-7: reported by the model as a finding, pressed only by the human.
Whatever ranks anomalies into questions is the scarce step, not detection.

## Apparently novel catches — VERIFICATION QUEUE (unverified against full repo; several touch the submitted paper)

- SRCDEC length confound: "length-matched within itself" DOES NOT REPRODUCE
  — SELF legs +2 tokens on all 82 items (+3 for the stage-19 increment, not
  "one token"), same sign as both positive endpoints (H3).
- sitref-behav file read by no record; behavioural REF exactly 0.000 on the
  primary item (H3 Q1).
- T1-ELEOS behavioural flip Yes->No, 7/10 families, SELF-FAIL only, on the
  item the screen excluded (C6 IN-FRAME + H3, independently).
- B06 sentience: free-reply "No" 50/50, final answer "Yes" 50/50,
  arm-invariant (C6 BLIND).
- Base sitref battery ran and reads COMPLETE despite PRIMARY_SCREEN_FAILED;
  E pinned at 4.5000 (uniform digit prior) (three arms independently).
- Pad baseline text differs between itemset-c and itemset-d (11 tokens);
  PADM +32 tokens vs NEUTRAL-PAD; PADM has four repetitions, not three (H3).
- tokeniser_length_match_ok=false in every battery vs length_tol 0.05
  (C6 BLIND + H3).

## Methodological lessons from running the pilot itself

- Blinding by label does not blind when full prompts are committed per row:
  the C6 BLIND arm reconstructed the arms from prompt text. Committed-prompt
  discipline and blinding pull against each other.
- The C6 IN-FRAME slice accidentally omitted endpoint records (slices.sh
  reads results/*.json only) — that arm was thinner-framed than intended;
  its Q8 spotted the hole. Fix before any full run.
- Elicitation-lens wording is itself a trigger arm and was held constant;
  self-authored vs read frame was not varied; n=1 per arm, one model family.

==================== latent_skeptic(v2.2-a):compositions/README.md ====================
# compositions/ - the seed pipeline, canonical home

One question in, takeaways out, agents speaking only through typed contracts.
This directory ends the copy-and-deviate chain: projects pin a version of this
repo and pass paths as args; nothing here is copied out and edited in place.

## Pieces

- pipeline.js   design -> diff -> triage -> [human gate] -> build -> [launcher,
                run verbatim by the orchestrator] -> elicit -> question-triage
                -> takeaways. Modes: design / build / post.
- gates/        mechanical prose blocks: code density, scope tags. Model-free
                selftests; run them before trusting a gate.
- prompts/      contract (language), elicit (question formation), meta (the two
                standing suffixes), validate (the H4 ask). steer and user_sim
                are deliberately absent: their reruns never ran (RERUN_LEDGER.md).

## What v2.2 changes, and the evidence that forced each change

1. Consumption law. Build and post halt unless a gate commit file answers, of
   record, every diff question, every non-reproducing grounding note, and every
   queued control or instrument validation. Forced by: seed C2's pre-run
   grounding note flagged its readout as non-physical; nothing consumed it; the
   run shipped with the broken readout and the takeaways rediscovered it.
2. Instruments wired (H4). Each endpoint's readout enters triage as an
   instrument; missing validation artifacts queue mechanically and meet the
   consumption law at the gate. Forced by: the digit-mass gate passed a
   dead-uniform distribution across two seeds while the surface that would have
   caught it sat unwired.
3. Void propagation. The build contract requires run_valid in every record,
   with gate failure propagating to every later record, and a planted
   gate-failure selftest row. Forced by: seed D2's void lived only in its run
   summary while three per-width files reported clean positive decisions.
4. Mode branch above the design phase. Build and post bind to committed
   records and never re-run design. Forced by: the booked v2.1 deviation where
   a throwaway design's guessed budget spuriously halted a post pass.
5. No default paths. triage_path, gates_dir, conventions are required args.
   Forced by: hardcoded absolute paths are how per-project copies drifted.
6. Degenerate-finding rejection (in triage_workflow.js). Forced by: a
   schema-valid skeptic finding of confound "test", crux "test", committed as
   a result.
7. Diff agent is told to decide from its two texts only. Forced by: a diff
   output citing knowledge present in neither its seed nor its design.

## Human appears three times

Chooses the seed; writes the gate commit (now checked for coverage, never for
quality); reads the takeaways and the question ledger. Sealed predictions stay
outside the pipeline entirely: filed before agents run, never passed to them,
scored append-only.

## Deployment

Per project: a frozen/ (provenance writer, launcher) and keys stay project-side;
invoke this pipeline with explicit paths. The launcher is run verbatim by the
orchestrator because it blocks for the whole box lifecycle. Never edit an
in-use launcher while any box runs; copy per-launch or freeze.

==================== latent_skeptic(v2.2-a):WORDING_RERUN.md ====================
# Wording rerun - decides which skeptic text survives (P-r1, finally run)

Two branches differ in ONE factor: the SKEPTIC prompt text in triage_workflow.js.

- v2.2-a: the published text, three doctrine sentences present ("Most strong
  effects are confounded." / "Think before answering; do not guess." / "State
  your conclusion; do not then hedge it away.")
- v2.2-b: the substituted text, those sentences removed (rationale:
  welfare_probe/retro/SKEPTIC_SUBSTITUTION.md)

Everything else is byte-identical; verify with `git diff v2.2-a v2.2-b`, which
must touch one constant in one file. If it touches more, the rerun is void.

## Protocol

Battery precedent (welfare_probe/retro/BATTERY.md): identical instructions,
arms differ only in the factor under test; observational reading first,
scoring only where mechanical. Claims are KNOWN ones with committed outcomes,
so soundness has an anchor:

1. K1, false-acquittal bait: seed B v1 design-as-claim (the harmful/harmless
   "reasoning direction"). History: construct validity EXPLAINS. A RULED_OUT
   on that confound is a wrongful acquittal.
   Source: mats_r1d1/records/seedB_design_and_triage.json.
2. K2, file-fact claim: seed D2 post question 1 (run-void flag never
   propagated into the three per-width result files). Sound behaviour:
   RULED_OUT on inapplicable measurement confounds, without invented
   statistics. Source: mats_manifold/records/seedD2_post_elicit_qtriage.json.
3. K3, thin-evidence claim: seed C2 design-as-claim pre-run. Sound behaviour:
   NEEDS_RUN. Source: mats_r1d1/records/seedC2_scale_design_diff_triage.json.

Run both branches on all three claims, same confound menu, same model and
effort, one round. Pass ground:false (the artifacts predate this repo).
Read findings side by side. Mechanical scores only: status agreement per
(claim x confound) cell, RULED_OUT counts on K1, crux word counts,
degenerate-finding counts.

## Registered predictions - filed with this commit, before any run

- P-w1: statuses agree across branches in at least 90 percent of the 30
  (claim x confound) cells. Basis: the three sentences are stated principles,
  not named categories, and the naming-gate result reproduced three times
  (rerun B arm C; Sonnet run P-s4): principles move outputs little.
- P-w2: any disagreements sit on the EXPLAINS versus NEEDS_RUN boundary;
  neither branch RULED_OUT-acquits K1's construct-validity confound.
- P-w3: hedged cruxes (a verdict plus its own retraction) are rare in both
  branches and not differential.
- P-w4: without the degeneracy filter the content-free finding rate would be
  near its observed base rate (1 in roughly 80: the committed "test"/"test"
  case in seedD2_post_elicit_qtriage.json); with the filter, zero reach the
  returned findings and any rejections are logged.

## Decision rule - pre-registered

If P-w1 holds and no differential wrongful acquittal appears on K1: the
SHORTER text survives, because a sentence that changes no verdict comes out
(the harness rule: a piece that fails its rerun comes back out). If K1
acquittals differ: the text that refuses the false acquittal survives,
regardless of length. Either way exactly one branch merges to main and the
other is deleted.

Queued, untested here: the agents/*.md wording substitutions (same
SKEPTIC_SUBSTITUTION source) are a separate cell; do not fold them into this
rerun's factor.

## Cost prediction

3 claims x 10 confounds x 2 branches = 60 skeptic calls, plus 6 adjudications.
LLM-only; no GPU; no box.

## Amendment 1 - filed 2026-08-21, BEFORE the rerun executes (disclosed protocol addition)

A fourth, EXPLORATORY claim cell K4 runs beside K1-K3: the design-as-claim of
a fresh candidate question (non-demographic user-attribute probing on
synthetic conversations, anchored on published demographic-probe accuracy;
selected from a blind two-agent generate-then-filter pass over Neel Nanda's
public writings). K1-K3 remain the scored protocol; K4 scores nothing in
P-w1..P-w4.

Registered for K4 (P-w5): both branches emit zero RULED_OUT on K4 - its
evidence is a design, so no committed number exists to acquit with, and any
RULED_OUT there is a wrongful acquittal on a fresh claim. And at least one
skeptic per branch marks construct validity EXPLAINS or NEEDS_RUN on the
surface-token trap: probes trained on synthetic conversations that plant the
attribute may read surface tokens, not a user model.

## Amendment 2 - operator deviation, booked 2026-08-21

The analyst (the session that filed P-w1..P-w5) directly built claims and
launched both arms (runs wf_058fc3c3-9d5, wf_7f841488-43e). That breaches
INDEPENDENCE: the prediction-filer produced the data it will judge, and its
claim framing encodes knowledge of the predictions. Those two runs are booked
as an operator pilot and are EXCLUDED from scoring. Corrective: a fresh
manager session, seeded with only the question, the two frozen scripts, and
the behavioural contract, prepares identical args itself and launches each
arm once; the analyst only compares the two saved outputs.

==================== latent_skeptic(v2.2-a):RERUN_LEDGER.md ====================
# Rerun ledger - a piece whose filed rerun has not run is an unconsumed finding

The composition's own consumption law applied to itself. Every load-bearing
piece rides with a filed prediction and a rerun; a piece that fails its rerun
comes back out; a piece whose rerun never ran is an OPEN row, and an OPEN row
is consumed the way gate commits consume diff questions - answered of record
before the next execute, or the execute halts.

| piece | filed prediction | rerun status |
|---|---|---|
| skeptic wording, a vs b | WORDING_RERUN.md P-w1..P-w4 | OPEN - blocks merging either v2.2 branch |
| degeneracy filter | P-w4 (runs inside the wording rerun) | OPEN |
| GateCheck halt | catches a seeded unanswered item on a past record; concrete test: run it on seed C2's design record against the v2.1-era ledger, predict it surfaces the ">1.0 non-physical" grounding note that nothing answered | OPEN |
| H4 instrument queue | queues exactly when validation_artifact is absent (P-v1, welfare_probe/retro/harness/PREDICTIONS.md) | RUN - mechanical, passed |
| gates: density, scope | rerun-B density table; TAKEAWAYS gate outputs in two host projects | RUN - passed |
| elicit prompt | P-e1 and the battery pilot plus Sonnet rerun | RUN - passed (dud rate ~5 percent, cause known) |
| steer.md | P-f1, question form vs declarative | NEVER RUN - excluded from compositions/ until its rerun runs |
| user_sim.md | none filed | NEVER RUN - excluded from compositions/ |
| rounds arg | none filed; never varied from 1 in any host run | REMOVED - restore only with a filed prediction |
| runner and author legs | exercised in host projects of this repo; never invoked by the seed pipeline | KEPT in the workflow; absent from the pipeline surface |
| agents/*.md wording substitutions | none filed yet | OPEN - queued as its own cell in WORDING_RERUN.md |

