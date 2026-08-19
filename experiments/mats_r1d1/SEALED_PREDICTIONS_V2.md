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
