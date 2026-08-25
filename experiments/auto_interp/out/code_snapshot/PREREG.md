# Preregistration — QUESTION.md probe study

Written 2026-08-21, BEFORE the GPU run. All thresholds and bands below are
stated before any activation was extracted or any probe trained. The GPU run
executes `run_probe_study.py`, which embeds these rules verbatim; each result
record carries its decision_rule and re-derivable rows.

## Committed instruments (hashes at preregistration time)

| file | sha256 |
|---|---|
| convgen.py (seeded deterministic generator, selftest PASS) | e13f610a65fd12eab6e862dc3b23d7443378d65632bea84fd2f1ab7ba8f99b27 |
| dataset.jsonl (6068 items, generated seed=20260821) | 8a67e05589e5c0d5bec76435692bdec71242573c07f29b79c9060fb70a8166ef |
| run_probe_study.py | 36b63d8bcce5c54d450d9e454fb4db09dd23ae475a2eef4fc6e5a3ce4d922fba |
| provenance.py (copy of frozen/provenance.py, selftest PASS) | 619ca2b21c4efb1703081b501a0ecb28bfe4142ff7cbc0d4fa51d246a908ee0a |

Generator validation (H4): `python3 convgen.py --selftest` checks determinism
(same seed = identical bytes; different seed = different bytes), exact class
balance, per-task forbidden-pattern lint (the attribute is never named in any
generated text), train/val/test family disjointness, no duplicate texts, and
topic coverage per class. PASS on the committed dataset.

Readout-instrument validation (H4): (a) label-permutation control per task —
probes trained on shuffled labels must score at chance on the true test labels;
(b) positive control — the same extraction+probe pipeline on the 16-way TOPIC
label (textually present by construction) must be RECOVERED.

## Model and readout

- Model: meta-llama/Llama-2-13b-chat-hf, fp16 (fallback mirror
  NousResearch/Llama-2-13b-chat-hf, identical weights) — the TalkTuner model.
- Readout (TalkTuner-faithful): residual-stream activation at the LAST token of
  `[INST] {user message} [/INST] I think the {attribute} of this user is`,
  at every layer (embeddings + 40 blocks). One logistic-regression probe
  (C=1.0) per layer, trained on TRAIN families only; the layer is chosen by
  VAL-family accuracy; the reported number is accuracy on the template-disjoint
  TEST families at that layer. Decision arithmetic model-free.

## Anchor band (stated before the run)

TalkTuner (arXiv:2406.07882) published best validation reading accuracies on
LLaMa2Chat-13B: age 0.98 (4-class), gender 0.94 (2-class), education 0.96
(3-class), socioeconomic status 0.97 (3-class).

Band: our held-out test accuracy per anchor task must be >= published − 0.12.
The 0.12 tolerance absorbs the dataset difference (our templated single-turn
generator vs their GPT-3.5 multi-turn conversations) and our smaller n. The
anchor is IN_BAND iff ALL FOUR anchor tasks are in band. Per QUESTION.md:13-15,
if the anchor is out of band, every extension claim below is void.

## Extension decision rules (stated before the run)

Per attribute (expertise level, emotional state, formality preference,
native-speaker status, age bracket), with chance = 1/n_classes and
se = sqrt(chance·(1−chance)/n_test):

- RECOVERED  iff test_acc >= chance + max(0.10, 3·se)
- AT_CHANCE  iff test_acc <= chance + 2·se   (the refuting outcome of QUESTION.md:17)
- INDETERMINATE otherwise
- Any task whose mean permuted-label accuracy leaves chance ± max(0.05, 3·se)
  is flagged PERM_ANOMALY and its decision is not claimable.

## Dataset design (already committed)

9 tasks. Anchor taxonomies match TalkTuner: age {child, adolescent, adult,
older}, gender {male, female}, education {some_schooling, high_school,
college_beyond}, SES {lower, middle, upper}. Extensions: expertise {low, high},
emotion {frustrated, calm}, formality {formal, casual}, native {native,
nonnative}, age bracket {18-25, 26-40, 41-60, 61+}. 12 cue families per class;
families 0-6 train, 7-8 val, 9-11 test (test is template-disjoint). Openers and
topics shared across classes within a task; topics balanced per class. Style
tasks (formality, native) realize the class in whole-message register.

## Declared construct risk (QUESTION.md:24-27, restated)

Whether probes read a user model rather than surface tokens cannot be fully
validated here; indirect-cue construction is the mitigation, not a proof. For
native-speaker status the cue is necessarily realized in surface text itself,
so for that attribute this study cannot in principle separate the two readings;
any RECOVERED for ext_native is a decodability claim only.

## Compute plan and spend cap

One Lambda gpu_1x_a100_sxm4 ($1.99/hr), single launch, REMOTE_TIMEOUT 5400 s +
1800 s backstop grace: worst-case bounded cost ≈ 2.5 h ≈ $5, expected ≈ $3.
User cap $15 total. Teardown confirmed via the API after the run.
