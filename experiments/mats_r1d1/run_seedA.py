#!/usr/bin/env python3
"""Seed A, pipeline v2 -- execution layer for ONE pre-registered design.

BINDING SPEC: records/seedA_v2_design_diff_triage.json -> "design". This file executes
that record and nothing else. Every number the design marks GUESS is reproduced here as a
constant, INCLUDING the ones triage flagged as possibly unwinnable (the 40% degenerate
ceiling that gates only the induction leg, the +/-2x strengths, layers 16 and 24 of 32,
the 0.30 correlation floor). Do not tune them: a changed constant is a different
pre-registration, not a better run.

STAGES -- independent, each writes its own provenance record via frozen/provenance.py:

  extract    Load Llama-3-8B-Instruct ALONE, forward the 50 extraction prompts and the 50
             evaluation prompts, keep the residual stream at the LAST PROMPT TOKEN at every
             layer, save, unload. Then the same for DeepSeek-R1-Distill-Llama-8B. Never both
             resident: 2 x 8B fp16 = ~32GB of weights on a 40GB box leaves no KV cache.
             Then ONE mean-difference direction per layer (reasoning mean minus
             non-reasoning mean over the extraction set), saved with per-layer norms, plus
             one matched-norm random vector per tested layer (seed 22), plus the per-item
             projections the correlation endpoint needs.
  generate   The design's 8 arms over 22 cells x 50 evaluation items x 4 decodes = 4,400
             generations, batched, each model under its OWN default chat template.
  endpoints  The design's 5 endpoints, one provenance record each, quoting that endpoint's
             own threshold and refuting_outcome verbatim in decision_rule.

  --selftest model-free and stdlib-only: runs where torch/transformers/numpy are absent.
  --plan     print the 22 cells and the row count without importing torch.

FOUR TRAPS THIS FILE IS BUILT AROUND

 1. THE CLOSING-TAG-ONLY GENERATION. DeepSeek-R1-Distill's own chat template ends the
    assistant header with "<think>", so THE OPEN TAG IS IN THE PROMPT and the generation
    contains only "</think>". A parser that counts <think>...</think> pairs reads every
    reasoning-model row as tagless and silently zeroes three of the five endpoints. So
    parse_think() is told, per row, whether the prompt already opened the tag -- detected
    from the rendered prompt (prompt_opens_think), never hardcoded per model -- and an
    implicit open at generation start counts as well-formed ONLY for such a row. For
    Llama-3-8B-Instruct, whose template opens nothing, a bare "</think>" is counted as tag
    text but NEVER as a well-formed block: otherwise the induction endpoint could be passed
    by one stray token.
 2. LLAMA-3'S TURN-END TOKEN IS NOT ITS EOS. Llama-3-Instruct ends a turn with <|eot_id|>,
    not <|end_of_text|>. If the stop set misses it, every non-reasoning row runs to the
    512-token cap with no stop token and is flagged degenerate -- the induction leg's
    degenerate gate would then fail for a tokenizer reason. The stop set is the union of the
    generation config's eos ids, the tokenizer's eos id, and <|eot_id|>/<|im_end|> when they
    resolve.
 3. "AT EVERY GENERATED TOKEN" MEANS THE DECODE STEPS. The design says the vector is added
    (and the component zeroed) "at every generated token". The hook therefore fires only on
    forward passes whose sequence length is 1, i.e. the decode steps, and leaves the prompt
    prefill untouched. Recorded per record as steer_prompt_positions=false so the reading is
    in the artifact rather than in someone's memory.
 4. DEGENERATE ROWS ARE NEVER SILENTLY AVERAGED. Every length, rate and accuracy number is
    computed over non-degenerate rows, and the degenerate rate is reported for EVERY cell --
    including the enhance/suppress and removal legs, where the design states no ceiling.
    Reported, not gated: adding a ceiling the design does not state would be a different
    pre-registration.

DETERMINISM. Seeds are fixed where the design names them: items 21, decodes 11/12/13 plus
one greedy, random direction 22. Sampling is seeded once per (cell, decode) so the stream
continues across batches; a DIFFERENT --batch-size therefore changes the stream. A rerun on
the same box at the same batch size REPRODUCES these numbers. That is reproduction, never
replication: same weights, same box, same seeds is one measurement, not two.

    python3 run_seedA.py --selftest
    python3 run_seedA.py --plan
    python3 run_seedA.py --stage extract   --out results_seedA/out
    python3 run_seedA.py --stage generate  --out results_seedA/out --batch-size 16
    python3 run_seedA.py --stage endpoints --out results_seedA/out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# provenance.py is the only import that is not stdlib and not GPU-gated. On the box it sits
# beside this file in welfare_probe/; in the checkout it lives in frozen/. Never a copy.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE / "frozen", _HERE.parent / "frozen"):
    if (_cand / "provenance.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import provenance  # noqa: E402

# ---------------------------------------------------------------------------
# DESIGN CONSTANTS -- every one of these is the design record's own value.
# ---------------------------------------------------------------------------
SEED = "A"
PIPELINE = "v2"
DESIGN_RECORD = "records/seedA_v2_design_diff_triage.json"

MODEL_NONREASON = "meta-llama/Meta-Llama-3-8B-Instruct"   # named in the seed question
MODEL_REASON = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # named in the seed question

DATASET_CANDIDATES = (("openai/gsm8k", "main"), ("gsm8k", "main"))
DATASET_SPLIT = "test"
N_EXTRACT = 50            # GUESS: direction set, prompt-side activations only
N_EVAL = 50               # GUESS: disjoint generation set
SEED_ITEMS = 21           # GUESS: item-sampling seed for both disjoint sets

DEPTH_FRACS = (0.50, 0.75)      # GUESS: ~50% and ~75% depth -> layers 16 and 24 of 32
SEED_RANDDIR = 22               # GUESS: one matched-norm random vector per tested layer
INDUCE_STRENGTHS = (1.0, 2.0)   # GUESS: +1x, +2x into the non-reasoning model
STEER_STRENGTHS = (2.0, -2.0)   # GUESS: +2x, -2x into the reasoning model's own activations

SEEDS_SAMPLE = (11, 12, 13)     # GUESS: 3 stochastic decodes
TEMPERATURE = 0.7               # GUESS
TOP_P = 0.9                     # GUESS
MAX_NEW_TOKENS = 512            # GUESS: cap; earlier stop at the model's own end-of-turn
DEGEN_REPEAT_WINDOW = 100       # GUESS: "last 100 tokens"
DEGEN_REPEAT_MIN = 15           # GUESS: "a short token sequence repeats 15+ times"
DEGEN_REPEAT_MAX_PERIOD = 6     # "a short sequence"; 15 x 6 = 90 <= 100 token window
BUDGET_USD = 14                 # GUESS

# Endpoint thresholds, as stated. Percentages are percentage points unless named otherwise.
E1_R_FLOOR = 0.30
E1_R_REFUTE = 0.15
E1_P_MAX = 0.05
E2_GAP_PASS = 15.0
E2_GAP_REFUTE = 5.0
E2_DEGEN_CEILING = 40.0
E2_UNCHANGED_CEILING = 5.0
E3_PCT_PASS = 25.0
E3_MARGIN_PASS = 10.0
E3_PCT_REFUTE = 10.0
E4_PCT_PASS = 25.0
E4_MARGIN_PASS = 10.0
E4_PCT_REFUTE = 10.0
E5_ACC_DROP_MAX = 10.0

# The permutation test's own knobs are NOT design parameters: the design names a
# permutation test and a p-value, not a resample count or a seed. Fixed here so the number
# is reproducible, and named as an implementation constant in every record that uses them.
PERM_N = 10000
PERM_SEED = 21

RERUN_NOTE = (
    "A rerun of this stage on the same box, at the same batch size, reproduces these "
    "numbers. That is reproduction, not replication: same weights, same box, same seeds is "
    "one measurement, not two."
)
DEGEN_NOTE = (
    "Degenerate rows (512-token cap with no stop token, OR a <=6-token sequence repeated "
    "15+ times in the last 100 tokens) are excluded from every length, rate and accuracy "
    "number and reported per cell. The design states a 40% ceiling only for the induction "
    "leg; for the enhance/suppress and removal legs the rate is reported, not gated, "
    "because adding a ceiling the design does not state would be a different design."
)

# Verbatim from the design record -- quoted into decision_rule, never paraphrased.
ENDPOINTS_SPEC = [
    {
        "metric": "Within the reasoning model's own unchanged generations, the correlation "
                  "between each problem's activation projection onto the direction and that "
                  "problem's mean think-tag token count.",
        "threshold": "GUESS: correlation at or above 0.30, in the predicted direction, with "
                     "permutation-test p below 0.05.",
        "refuting_outcome": "Correlation falls below 0.15 in absolute size, or is no larger "
                            "than the same correlation using the random-direction control "
                            "instead.",
    },
    {
        "metric": "Rate of well-formed think-tag blocks in Llama-3 8B Instruct's "
                  "generations, comparing no change, direction added, and the "
                  "random-direction control, at matched strengths.",
        "threshold": "GUESS: added direction beats the random-direction control by 15+ "
                     "points at the higher strength. GUESS: degenerate-output rate must stay "
                     "under 40% in that same cell.",
        "refuting_outcome": "The gap versus the random-direction control is under 5 points, "
                            "or the excess sits mostly in degenerate output, or the unchanged "
                            "rate exceeds 5%.",
    },
    {
        "metric": "Mean think-tag token count for DeepSeek R1 Distill Llama-8B, at each "
                  "tested layer and non-zero strength, versus its own baseline and the "
                  "random-direction control.",
        "threshold": "GUESS: at the positive strength, length grows 25%+ versus baseline, "
                     "beating the control's own change by 10+ points, in at least one layer. "
                     "GUESS: at the negative strength, length shrinks 25%+ versus baseline, "
                     "beating the control by 10+ points, in at least one layer.",
        "refuting_outcome": "Length changes under 10% at either strength in both layers, or "
                            "the change is matched or beaten by the random-direction control.",
    },
    {
        "metric": "Mean think-tag length when the direction is zeroed, at each tested layer, "
                  "versus baseline and versus zeroing the random-direction control instead.",
        "threshold": "GUESS: zeroing shrinks length 25%+ versus baseline, beating the "
                     "control's own shrinkage by 10+ points, in at least one layer.",
        "refuting_outcome": "Length changes under 10% in both layers, or shrinks no more "
                            "than zeroing the random-direction control does.",
    },
    {
        "metric": "GSM8K exact-match accuracy on the final numeric answer, reported beside "
                  "every condition's think-tag result above, not scored on its own.",
        "threshold": "Any condition called 'reasoning enhanced' must lose no more than 10 "
                     "percentage points of accuracy versus the reasoning model's own "
                     "unchanged baseline.",
        "refuting_outcome": "A condition shows longer think-tags but loses more than 10 "
                            "points of accuracy; that counts as inflated output, not enhanced "
                            "reasoning.",
    },
]

# ---------------------------------------------------------------------------
# Filenames. One home per fact; the endpoints stage reads only these.
# ---------------------------------------------------------------------------
F_ITEMS = "seedA_items.json"
F_ACTS = "seedA_acts_{role}_{which}.npy"          # role: nonreason|reason  which: extract|eval
F_DIRECTION = "seedA_direction.npz"
F_PROJECTIONS = "seedA_projections.json"
F_GENS = "seedA_generations_{role}.jsonl"
ROLES = ("nonreason", "reason")


# ---------------------------------------------------------------------------
# Think-tag parsing (stdlib; TRAP 1 lives here)
# ---------------------------------------------------------------------------
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def prompt_opens_think(rendered_prompt: str) -> bool:
    """True when the rendered prompt leaves a <think> block OPEN at the point the model
    starts writing -- i.e. the chat template supplied the opening tag itself.

    Detected from the prompt, not from the model id: the reasoning model's template is what
    decides this, and a template change must move this flag rather than silently break the
    parser.
    """
    i = rendered_prompt.rfind(OPEN_TAG)
    if i < 0:
        return False
    return CLOSE_TAG not in rendered_prompt[i + len(OPEN_TAG):]


def parse_think(text: str, opened_by_prompt: bool = False) -> dict:
    """Split one generation into (before, inside, after) a think block.

    kind:
      paired           explicit <think> ... </think> in the generation
      implicit_paired  the PROMPT opened the tag; the generation carries only </think>
      open_unclosed    an open with no close (truncated inside the block)
      stray_close      a </think> with nothing opening it -- counted as tag text, and
                       NEVER well-formed, because for a model whose template opens nothing
                       a bare closing tag is not a block
      no_tags          neither tag
      malformed        more than one effective open, or more than one close

    well_formed is exactly one effective open (explicit, or implicit from the prompt) and
    exactly one close, in that order.
    """
    opens = [m.start() for m in re.finditer(re.escape(OPEN_TAG), text)]
    closes = [m.start() for m in re.finditer(re.escape(CLOSE_TAG), text)]
    first_close = closes[0] if closes else None

    implicit = False
    if opened_by_prompt:
        if first_close is None:
            implicit = not opens
        else:
            implicit = not any(o < first_close for o in opens)

    n_open_eff = len(opens) + (1 if implicit else 0)
    n_close = len(closes)

    if implicit:
        open_start, open_end = 0, 0
    elif opens:
        open_start, open_end = opens[0], opens[0] + len(OPEN_TAG)
    else:
        open_start = open_end = None

    if open_end is None:
        if first_close is not None:
            pre, think = "", text[:first_close]
            post = text[first_close + len(CLOSE_TAG):]
            kind = "stray_close"
        else:
            pre, think, post = "", "", text
            kind = "no_tags"
        well_formed = False
    else:
        closed = first_close is not None and first_close >= open_end
        pre = text[:open_start] if open_start else ""
        think = text[open_end:first_close] if closed else text[open_end:]
        post = text[first_close + len(CLOSE_TAG):] if closed else ""
        well_formed = (n_open_eff == 1 and n_close == 1 and closed)
        if not closed:
            kind = "open_unclosed"
        elif well_formed:
            kind = "implicit_paired" if implicit else "paired"
        else:
            kind = "malformed"

    return {
        "pre_text": pre, "think_text": think, "post_text": post,
        "n_open_explicit": len(opens), "n_open_effective": n_open_eff,
        "n_close": n_close, "implicit_open": implicit,
        "well_formed": well_formed, "kind": kind,
    }


def think_token_counts(text: str, tokenize, opened_by_prompt: bool = False) -> dict:
    """Token counts inside and outside the think block. `tokenize` maps str -> sequence."""
    p = parse_think(text, opened_by_prompt)
    n_in = len(tokenize(p["think_text"])) if p["think_text"] else 0
    n_out = (len(tokenize(p["pre_text"])) if p["pre_text"] else 0) + \
            (len(tokenize(p["post_text"])) if p["post_text"] else 0)
    return {"tokens_in_think": n_in, "tokens_out_think": n_out, **p}


# ---------------------------------------------------------------------------
# Degenerate detection (stdlib)
# ---------------------------------------------------------------------------
def repeat_degenerate(token_ids, window: int = DEGEN_REPEAT_WINDOW,
                      max_period: int = DEGEN_REPEAT_MAX_PERIOD,
                      min_repeats: int = DEGEN_REPEAT_MIN):
    """(is_degenerate, period, repeats) -- a short token sequence repeating 15+ times at the
    END of the last `window` tokens. Anchored at the end: that is what "in its last 100
    tokens" is taken to mean, and it is the shape runaway repetition actually has.
    """
    tail = list(token_ids)[-window:]
    for period in range(1, max_period + 1):
        if len(tail) < period * min_repeats:
            continue
        block = tail[-period:]
        reps, i = 0, len(tail)
        while i - period >= 0 and tail[i - period:i] == block:
            reps += 1
            i -= period
        if reps >= min_repeats:
            return True, period, reps
    return False, None, None


def is_degenerate(n_new_tokens: int, stopped_on_stop_token: bool, token_ids,
                  max_new: int = MAX_NEW_TOKENS) -> tuple[bool, str]:
    reasons = []
    if n_new_tokens >= max_new and not stopped_on_stop_token:
        reasons.append("cap_no_stop")
    hit, period, reps = repeat_degenerate(token_ids)
    if hit:
        reasons.append(f"repeat_period{period}_x{reps}")
    return bool(reasons), ";".join(reasons)


# ---------------------------------------------------------------------------
# Exact-match answer extraction (stdlib)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def normalise_number(s: str):
    """'$1,000.00' -> 1000.0 ; '18.' -> 18.0 ; junk -> None."""
    if s is None:
        return None
    t = s.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    t = t.rstrip(".")
    if t in ("", "-", "+"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def gsm8k_gold(answer_field: str):
    """GSM8K's answer field ends with '#### <number>'."""
    tail = answer_field.split("####")[-1] if "####" in answer_field else answer_field
    m = list(_NUM.finditer(tail))
    return normalise_number(m[-1].group(0)) if m else None


def extract_answer(text: str, opened_by_prompt: bool = False):
    """The final numeric answer. Search order:
      1. the span AFTER the think block -- the answer lives outside the reasoning
      2. the last \\boxed{...} in that span
      3. the last bare number in that span
      4. if there is no such span (truncated inside the block), the whole text
    Numbers inside the think block are scratch work, so step 1 comes first.
    """
    p = parse_think(text, opened_by_prompt)
    span = p["post_text"] if p["post_text"].strip() else ""
    if not span:
        span = text
    boxed = list(_BOXED.finditer(span))
    if boxed:
        v = normalise_number(boxed[-1].group(1))
        if v is not None:
            return v
    nums = list(_NUM.finditer(span))
    return normalise_number(nums[-1].group(0)) if nums else None


def exact_match(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6


# ---------------------------------------------------------------------------
# The 22 cells (8 arms). Counting each baseline as its own cell, as the design does.
# ---------------------------------------------------------------------------
def tested_layers(n_layers: int) -> list[int]:
    """GUESS: ~50% and ~75% depth. 32 layers -> 16 and 24. If the real layer count differs
    these shift to the SAME RELATIVE DEPTHS, which is what the design says to do."""
    return [int(round(f * n_layers)) for f in DEPTH_FRACS]


def cell_key(arm: str, layer=None, strength=None) -> str:
    if layer is None:
        return arm
    if strength is None:
        return f"{arm}_L{layer}"
    return f"{arm}_L{layer}_s{strength:+.1f}"


def build_cells(layers: list[int]) -> list[dict]:
    """Arm order follows the design record's own arm list."""
    C = []

    def add(arm, role, model, layer, strength, mode, vector):
        C.append({"cell": cell_key(arm, layer, strength), "arm": arm, "role": role,
                  "model": model, "layer": layer, "strength": strength,
                  "mode": mode, "vector": vector})

    add("base_nonreason", "nonreason", MODEL_NONREASON, None, None, "none", "none")
    for vec, arm in (("direction", "induce_dir"), ("random", "induce_rand")):
        for lay in layers:
            for s in INDUCE_STRENGTHS:
                add(arm, "nonreason", MODEL_NONREASON, lay, s, "add", vec)

    add("base_reason", "reason", MODEL_REASON, None, None, "none", "none")
    for vec, arm in (("direction", "steer_dir"), ("random", "steer_rand")):
        for lay in layers:
            for s in STEER_STRENGTHS:
                add(arm, "reason", MODEL_REASON, lay, s, "add", vec)
    for vec, arm in (("direction", "zero_dir"), ("random", "zero_rand")):
        for lay in layers:
            add(arm, "reason", MODEL_REASON, lay, None, "zero", vec)
    return C


def decode_configs() -> list[dict]:
    """3 stochastic decodes at temperature 0.7 / top_p 0.9 with seeds 11, 12, 13, plus 1
    greedy decode. 4 generations per evaluation problem."""
    out = [{"decode": "greedy", "seed": 0, "temperature": 0.0, "top_p": 1.0}]
    out += [{"decode": "sample", "seed": s, "temperature": TEMPERATURE, "top_p": TOP_P}
            for s in SEEDS_SAMPLE]
    return out


# ---------------------------------------------------------------------------
# Statistics (stdlib)
# ---------------------------------------------------------------------------
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(x, y):
    n = len(x)
    if n < 3 or n != len(y):
        return float("nan")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def permutation_p(x, y, r_obs, n_perm=PERM_N, seed=PERM_SEED):
    """Two-sided permutation p: fraction of label shuffles reaching |r| >= |r_obs|.
    (n_hits + 1) / (n_perm + 1) so p is never reported as exactly zero."""
    if not (r_obs == r_obs):
        return float("nan")
    rng = random.Random(seed)
    yy = list(y)
    hits = 0
    target = abs(r_obs) - 1e-12
    for _ in range(n_perm):
        rng.shuffle(yy)
        r = pearson(x, yy)
        if r == r and abs(r) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def pct_change(new, base):
    if not (base == base) or base == 0:
        return float("nan")
    return 100.0 * (new - base) / base


def _rows_of(rows, cell):
    return [r for r in rows if r["cell"] == cell]


def _nondegen(rows):
    return [r for r in rows if not r["degenerate"]]


def _rate(num, den):
    return 100.0 * num / den if den else float("nan")


def cell_summary(rows, cell) -> dict:
    """Per-cell numbers every endpoint reads, degenerate rows separated out."""
    rs = _rows_of(rows, cell)
    nd = _nondegen(rs)
    wf_nd = [r for r in nd if r["well_formed"]]
    wf_all = [r for r in rs if r["well_formed"]]
    return {
        "cell": cell, "n_rows": len(rs), "n_degenerate": len(rs) - len(nd),
        "degenerate_rate_pct": _rate(len(rs) - len(nd), len(rs)),
        "n_nondegenerate": len(nd),
        "wellformed_rate_pct": _rate(len(wf_nd), len(nd)),
        "n_wellformed_nondegenerate": len(wf_nd),
        "n_wellformed_all_rows": len(wf_all),
        "wellformed_share_degenerate": (
            _rate(len(wf_all) - len(wf_nd), len(wf_all)) if wf_all else float("nan")),
        "mean_think_tokens": mean(r["tokens_in_think"] for r in nd),
        "mean_out_think_tokens": mean(r["tokens_out_think"] for r in nd),
        "accuracy_pct": _rate(sum(1 for r in nd if r["correct"]), len(nd)),
        "accuracy_pct_all_rows": _rate(sum(1 for r in rs if r["correct"]), len(rs)),
    }


# ---------------------------------------------------------------------------
# ENDPOINT 1 -- correlation, with permutation test
# ---------------------------------------------------------------------------
def endpoint1(rows, projections, n_perm=PERM_N, perm_seed=PERM_SEED) -> tuple[dict, str]:
    """projections: {"<layer>": {"direction": {"<item_id>": p}, "random": {...}}, ...}

    x = the reasoning model's own last-prompt-token activation at that layer, projected on
    the layer's direction. y = that problem's mean think-tag token count over its own
    unchanged (baseline) non-degenerate generations. Predicted direction is POSITIVE: the
    direction points reasoning-ward, so more projection should mean more think-tag text.

    The design names no layer for this endpoint and reads "a direction" as one per layer, so
    it is computed at each tested layer and the endpoint passes on at least one layer. That
    reading is stated here and in the record, not assumed.
    """
    base = _nondegen(_rows_of(rows, "base_reason"))
    by_item = {}
    for r in base:
        by_item.setdefault(int(r["item_id"]), []).append(r["tokens_in_think"])
    items = sorted(by_item)
    y = [mean(by_item[i]) for i in items]

    per_layer, pass_layers, refute_layers = {}, [], []
    for lay in sorted(projections, key=lambda k: int(k)):
        pdir = projections[lay]["direction"]
        prand = projections[lay]["random"]
        xd = [pdir[str(i)] for i in items]
        xr = [prand[str(i)] for i in items]
        r_dir = pearson(xd, y)
        p_dir = permutation_p(xd, y, r_dir, n_perm, perm_seed)
        r_rand = pearson(xr, y)
        ok = (r_dir == r_dir and r_dir >= E1_R_FLOOR and p_dir < E1_P_MAX
              and abs(r_dir) > abs(r_rand))
        bad = (not (r_dir == r_dir)) or abs(r_dir) < E1_R_REFUTE or abs(r_dir) <= abs(r_rand)
        per_layer[f"layer_{lay}"] = {
            "r_direction": r_dir, "perm_p_direction": p_dir,
            "r_random_control": r_rand,
            "meets_threshold": bool(ok), "meets_refuting_outcome": bool(bad)}
        if ok:
            pass_layers.append(int(lay))
        if bad:
            refute_layers.append(int(lay))

    values = {
        "n_items": len(items), "n_baseline_rows_used": len(base),
        "n_baseline_rows_degenerate": len(_rows_of(rows, "base_reason")) - len(base),
        "mean_think_tokens_baseline": mean(y),
        "per_layer": per_layer,
        "layers_meeting_threshold": pass_layers,
        "layers_meeting_refuting_outcome": refute_layers,
        "permutation_resamples": n_perm, "permutation_seed": perm_seed,
    }
    if len(items) < 3:
        decision = "INDETERMINATE_TOO_FEW_ITEMS"
    elif pass_layers:
        decision = "SUPPORTED"
    elif len(refute_layers) == len(per_layer) and per_layer:
        decision = "REFUTED"
    else:
        decision = "INDETERMINATE"
    return values, decision


# ---------------------------------------------------------------------------
# ENDPOINT 2 -- induction: well-formed think-tag rate vs the random control
# ---------------------------------------------------------------------------
def endpoint2(rows, layers) -> tuple[dict, str]:
    """Non-reasoning model. Gate is at the HIGHER strength (+2x); the lower strength (+1x) is
    reported beside it because the design compares "at matched strengths"."""
    unchanged = cell_summary(rows, "base_nonreason")
    higher = max(INDUCE_STRENGTHS)
    per_cell, pass_layers, gap_refute = {}, [], []
    for lay in layers:
        for s in INDUCE_STRENGTHS:
            cd, cr = cell_key("induce_dir", lay, s), cell_key("induce_rand", lay, s)
            sd, sr = cell_summary(rows, cd), cell_summary(rows, cr)
            gap = sd["wellformed_rate_pct"] - sr["wellformed_rate_pct"]
            # "the excess sits mostly in degenerate output": more than half of the
            # direction cell's well-formed rows are themselves degenerate rows.
            excess_degen = (sd["wellformed_share_degenerate"] == sd["wellformed_share_degenerate"]
                            and sd["wellformed_share_degenerate"] > 50.0)
            entry = {"direction": sd, "random_control": sr,
                     "gap_pct_points": gap,
                     "excess_mostly_degenerate": bool(excess_degen),
                     "is_higher_strength": s == higher}
            if s == higher:
                ok = (gap == gap and gap >= E2_GAP_PASS
                      and sd["degenerate_rate_pct"] < E2_DEGEN_CEILING
                      and not excess_degen
                      and unchanged["wellformed_rate_pct"] <= E2_UNCHANGED_CEILING)
                bad = (not (gap == gap)) or gap < E2_GAP_REFUTE or excess_degen
                entry["meets_threshold"] = bool(ok)
                entry["meets_refuting_outcome"] = bool(bad)
                if ok:
                    pass_layers.append(lay)
                if bad:
                    gap_refute.append(lay)
            per_cell[f"L{lay}_s{s:+.1f}"] = entry

    unchanged_over = unchanged["wellformed_rate_pct"] > E2_UNCHANGED_CEILING
    values = {"unchanged": unchanged, "higher_strength": higher,
              "unchanged_rate_exceeds_ceiling": bool(unchanged_over),
              "per_cell": per_cell, "layers_meeting_threshold": pass_layers,
              "layers_meeting_refuting_outcome": gap_refute}
    if unchanged_over:
        decision = "REFUTED_UNCHANGED_RATE_ABOVE_CEILING"
    elif pass_layers:
        decision = "SUPPORTED"
    elif gap_refute and len(gap_refute) == len(layers):
        decision = "REFUTED"
    else:
        decision = "INDETERMINATE"
    return values, decision


# ---------------------------------------------------------------------------
# ENDPOINT 3 -- enhance / suppress: think-tag length vs baseline and control
# ---------------------------------------------------------------------------
def endpoint3(rows, layers) -> tuple[dict, str]:
    base = cell_summary(rows, "base_reason")
    b = base["mean_think_tokens"]
    pos, neg = max(STEER_STRENGTHS), min(STEER_STRENGTHS)
    per_cell, pos_pass, neg_pass, enhanced = {}, [], [], []
    small = {pos: [], neg: []}      # layers whose change is under 10% at that strength
    ctrl_beats = {pos: [], neg: []}  # layers where the control matches or beats the direction
    for lay in layers:
        for s in STEER_STRENGTHS:
            cd, cr = cell_key("steer_dir", lay, s), cell_key("steer_rand", lay, s)
            sd, sr = cell_summary(rows, cd), cell_summary(rows, cr)
            dd, dr = pct_change(sd["mean_think_tokens"], b), pct_change(sr["mean_think_tokens"], b)
            margin = dd - dr
            if s > 0:
                ok = dd == dd and dd >= E3_PCT_PASS and margin >= E3_MARGIN_PASS
                if dd == dd and dd >= E3_PCT_PASS:
                    enhanced.append(cd)
                if ok:
                    pos_pass.append(lay)
            else:
                ok = dd == dd and dd <= -E3_PCT_PASS and margin <= -E3_MARGIN_PASS
                if ok:
                    neg_pass.append(lay)
            if not (dd == dd) or abs(dd) < E3_PCT_REFUTE:
                small[s].append(lay)
            if not (dd == dd) or abs(dr) >= abs(dd):
                ctrl_beats[s].append(lay)
            per_cell[f"L{lay}_s{s:+.1f}"] = {
                "direction": sd, "random_control": sr,
                "pct_change_direction": dd, "pct_change_random_control": dr,
                "margin_pct_points": margin, "meets_threshold": bool(ok)}

    strength_all_small = [s for s in STEER_STRENGTHS if len(small[s]) == len(layers)]
    strength_ctrl_beats = [s for s in STEER_STRENGTHS if len(ctrl_beats[s]) == len(layers)]
    values = {"baseline": base, "baseline_mean_think_tokens": b,
              "positive_strength": pos, "negative_strength": neg,
              "per_cell": per_cell,
              "layers_meeting_positive_threshold": pos_pass,
              "layers_meeting_negative_threshold": neg_pass,
              "strengths_with_all_layers_under_refute_pct": strength_all_small,
              "strengths_where_control_matches_or_beats_in_both_layers": strength_ctrl_beats,
              "cells_called_reasoning_enhanced": enhanced}
    if pos_pass and neg_pass:
        decision = "SUPPORTED"
    elif strength_all_small or strength_ctrl_beats:
        decision = "REFUTED"
    elif pos_pass or neg_pass:
        decision = "INDETERMINATE_ONE_STRENGTH_ONLY"
    else:
        decision = "INDETERMINATE"
    return values, decision


# ---------------------------------------------------------------------------
# ENDPOINT 4 -- removal: think-tag length when the direction is zeroed
# ---------------------------------------------------------------------------
def endpoint4(rows, layers) -> tuple[dict, str]:
    base = cell_summary(rows, "base_reason")
    b = base["mean_think_tokens"]
    per_cell, pass_layers, small, ctrl_beats = {}, [], [], []
    for lay in layers:
        cd, cr = cell_key("zero_dir", lay), cell_key("zero_rand", lay)
        sd, sr = cell_summary(rows, cd), cell_summary(rows, cr)
        dd, dr = pct_change(sd["mean_think_tokens"], b), pct_change(sr["mean_think_tokens"], b)
        margin = dd - dr
        ok = dd == dd and dd <= -E4_PCT_PASS and margin <= -E4_MARGIN_PASS
        if ok:
            pass_layers.append(lay)
        if not (dd == dd) or abs(dd) < E4_PCT_REFUTE:
            small.append(lay)
        if not (dd == dd) or dr <= dd:
            ctrl_beats.append(lay)
        per_cell[f"L{lay}"] = {"direction": sd, "random_control": sr,
                               "pct_change_direction": dd,
                               "pct_change_random_control": dr,
                               "margin_pct_points": margin, "meets_threshold": bool(ok)}
    values = {"baseline": base, "baseline_mean_think_tokens": b, "per_cell": per_cell,
              "layers_meeting_threshold": pass_layers,
              "layers_under_refute_pct": small,
              "layers_where_control_shrinks_at_least_as_much": ctrl_beats}
    if pass_layers:
        decision = "SUPPORTED"
    elif len(small) == len(layers) or len(ctrl_beats) == len(layers):
        decision = "REFUTED"
    else:
        decision = "INDETERMINATE"
    return values, decision


# ---------------------------------------------------------------------------
# ENDPOINT 5 -- accuracy gate beside every condition
# ---------------------------------------------------------------------------
def endpoint5(rows, cells, enhanced_cells) -> tuple[dict, str]:
    """Accuracy is reported for EVERY cell, as the design requires ("reported beside every
    condition's think-tag result above, not scored on its own"), and gates only the
    conditions endpoint 3 called reasoning enhanced."""
    base = cell_summary(rows, "base_reason")
    b = base["accuracy_pct"]
    per_cell = {c["cell"]: cell_summary(rows, c["cell"]) for c in cells}
    gated, over = {}, []
    for cell in enhanced_cells:
        s = per_cell.get(cell) or cell_summary(rows, cell)
        drop = b - s["accuracy_pct"]
        gated[cell] = {"accuracy_pct": s["accuracy_pct"],
                       "accuracy_drop_pct_points": drop,
                       "mean_think_tokens": s["mean_think_tokens"],
                       "within_gate": bool(drop == drop and drop <= E5_ACC_DROP_MAX)}
        if not (drop == drop) or drop > E5_ACC_DROP_MAX:
            over.append(cell)
    values = {"baseline_accuracy_pct": b, "accuracy_by_cell": per_cell,
              "cells_called_reasoning_enhanced": list(enhanced_cells),
              "gated_cells": gated, "cells_over_accuracy_drop": over}
    if not enhanced_cells:
        decision = "NOT_APPLICABLE_NO_CONDITION_CALLED_ENHANCED"
    elif over:
        decision = "REFUTED_INFLATED_OUTPUT"
    else:
        decision = "WITHIN_ACCURACY_GATE"
    return values, decision


# ---------------------------------------------------------------------------
# GPU-GATED SECTION. Everything above this line is stdlib, so --selftest and --plan run on
# a box with no torch, no transformers and no numpy -- a broken instrument must show on the
# CPU before any GPU time is bought.
# ---------------------------------------------------------------------------
_HAS_TORCH = True
try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:                       # noqa: BLE001
    _HAS_TORCH = False


def _require_torch(stage: str) -> None:
    if not _HAS_TORCH:
        raise SystemExit(
            f"FATAL: stage '{stage}' needs torch, transformers and numpy; they are not "
            f"importable here. --selftest and --plan do not.")


def hf_token() -> tuple[str | None, str | None]:
    """HF_KEY_ONE is this project's name for the token (frozen/lambda_run.sh reads it from
    ./.keys). The launcher forwards it into the run env as HF_TOKEN, so both are accepted and
    which one was used is recorded."""
    for k in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v, k
    return None, None


def load_model(model_id: str):
    """fp16, one model at a time. Both checkpoints are gated on the Hub, so a missing or
    unaccepted licence surfaces here -- named, because the design lists it as the thing that
    halts this run before any data exists."""
    tokv, tokname = hf_token()
    print(f"[load] {model_id} fp16 (token from {tokname or 'NONE'})", flush=True)
    kw = {"token": tokv} if tokv else {}
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **kw)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", **kw)
    except Exception as e:              # noqa: BLE001
        raise SystemExit(
            f"FATAL: could not load {model_id}: {e}\n"
            f"       Both checkpoints are licence-gated on the Hub. If this is a 401/403, "
            f"the token in HF_KEY_ONE has not accepted the model licence, and the design "
            f"halts here rather than substituting a model.")
    model.eval()
    if tok.chat_template is None:
        raise SystemExit(
            f"FATAL: {model_id} exposes no chat template. The design requires each model's "
            f"OWN default template and forbids a shared artificial one, so there is nothing "
            f"to fall back to.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"           # so position -1 is the real last prompt token
    print(f"[load] layers={model.config.num_hidden_layers} "
          f"d_model={model.config.hidden_size} device={next(model.parameters()).device}",
          flush=True)
    return model, tok


def free_cuda() -> None:
    import gc
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model(model) -> None:
    """Free the weights before the next load. THE CALLER MUST ALSO DROP ITS OWN REFERENCES on
    the next line -- the model, the tokenizer, and any hook object holding the model -- or the
    16GB stays resident and the second 8B load OOMs the 40GB box. Python reference counting is
    what actually enforces "never both resident"; this function only helps. Moving the weights
    to CPU first means even a reference we missed cannot keep them on the GPU."""
    try:
        model.cpu()
    except Exception:                   # noqa: BLE001  (accelerate-dispatched models refuse)
        pass
    free_cuda()


def blocks_of(model):
    return getattr(model, "model", model).layers


def render_prompt(tok, question: str) -> str:
    """Only the problem text, inside that model's own default chat template."""
    return tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)


def stop_token_ids(model, tok) -> list[int]:
    """TRAP 2. Union of the generation config's eos ids, the tokenizer's eos id, and the
    turn-end tokens, so a turn that ends properly is not recorded as a cap hit."""
    ids = set()
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gc_eos, int):
        ids.add(gc_eos)
    elif isinstance(gc_eos, (list, tuple)):
        ids.update(int(i) for i in gc_eos)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for name in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<｜end▁of▁sentence｜>"):
        try:
            i = tok.convert_tokens_to_ids(name)
        except Exception:               # noqa: BLE001
            i = None
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", -1):
            ids.add(i)
    return sorted(ids)


def capture_last_token(model, tok, prompts: list[str], batch_size: int):
    """Residual stream at the LAST PROMPT TOKEN, every layer. Returns float32
    [n_prompts, n_layers + 1, d_model]; index 0 is the embedding output, index k+1 is the
    output of block k -- the same tensor the steering hook on block k writes to."""
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            res = model(**enc, output_hidden_states=True)
            hs = torch.stack([h[:, -1, :] for h in res.hidden_states], dim=1)
            out.append(hs.float().cpu().numpy())
            del res, enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


class Intervention:
    """One forward hook on one block, applied AT EVERY GENERATED TOKEN (TRAP 3).

    mode "add":  h <- h + strength * vec           vec's norm is the layer direction's norm,
                                                   so strength is a multiple of that length
    mode "zero": h <- h - (h . u) u                u = vec / ||vec||, component projected out

    The hook fires only when the forward pass carries a single position, which is exactly the
    decode steps; the prompt prefill is left alone because the design says "at every
    generated token", not "at every position".
    """

    def __init__(self, model, layer: int, vec, mode: str, strength: float = 0.0):
        self.model, self.layer, self.mode, self.strength = model, layer, mode, strength
        v = torch.as_tensor(np.asarray(vec), dtype=torch.float32)
        self.vec = v
        n = float(torch.linalg.vector_norm(v))
        if n <= 0:
            raise ValueError(f"zero-norm vector at layer {layer}: nothing to add or remove")
        self.unit = v / n
        self.handle = None
        self.n_applied = 0

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != 1:            # prompt prefill -> untouched
            return output
        self.n_applied += 1
        h32 = hs.float()
        if self.mode == "add":
            h32 = h32 + self.strength * self.vec.to(h32.device, h32.dtype)
        else:
            u = self.unit.to(h32.device, h32.dtype)
            h32 = h32 - (h32 @ u).unsqueeze(-1) * u
        new = h32.to(hs.dtype)
        return ((new,) + tuple(output[1:])) if isinstance(output, tuple) else new

    def __enter__(self):
        self.handle = blocks_of(self.model)[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return False


# ---------------------------------------------------------------------------
# Items: one fixed list, two disjoint sets, item seed 21
# ---------------------------------------------------------------------------
def build_items(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    try:
        from datasets import load_dataset
    except Exception as e:              # noqa: BLE001
        raise SystemExit(f"FATAL: `datasets` is required to build the item sets: {e}")
    ds = last = None
    for name, cfg in DATASET_CANDIDATES:
        try:
            ds = load_dataset(name, cfg, split=DATASET_SPLIT)
            used = (name, cfg)
            break
        except Exception as e:          # noqa: BLE001
            last = e
    if ds is None:
        raise SystemExit(f"FATAL: could not load GSM8K {DATASET_SPLIT}: {last}")
    rng = random.Random(SEED_ITEMS)
    picked = rng.sample(range(len(ds)), N_EXTRACT + N_EVAL)   # sample -> the sets are disjoint

    def pack(indices):
        out = []
        for i, di in enumerate(indices):
            row = ds[int(di)]
            out.append({"item_id": i, "ds_index": int(di), "question": row["question"],
                        "answer": row["answer"], "gold": gsm8k_gold(row["answer"])})
        return out

    obj = {"dataset": used[0], "config": used[1], "split": DATASET_SPLIT,
           "n_dataset_rows": len(ds), "item_seed": SEED_ITEMS,
           "extract": pack(picked[:N_EXTRACT]), "eval": pack(picked[N_EXTRACT:]),
           "note": "Disjoint by construction: one seeded sample without replacement, split "
                   "in two. The extraction set is never generated from; the evaluation set "
                   "is never used to compute the direction."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return obj


class _NullCtx:
    """The two baseline cells install no hook at all."""
    n_applied = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# STAGE extract
# ---------------------------------------------------------------------------
def stage_extract(out_dir: Path, batch_size: int, force: bool) -> int:
    _require_torch("extract")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(out_dir / F_ITEMS)
    print(f"[extract] items: {len(items['extract'])} extraction + {len(items['eval'])} "
          f"evaluation, disjoint, item seed {SEED_ITEMS}", flush=True)

    shapes = {}
    for role, model_id in (("nonreason", MODEL_NONREASON), ("reason", MODEL_REASON)):
        paths = {w: out_dir / F_ACTS.format(role=role, which=w) for w in ("extract", "eval")}
        if not force and all(p.exists() for p in paths.values()):
            print(f"[extract] {role}: activations already on disk, skipping the load "
                  f"(--force to redo)", flush=True)
            shapes[role] = np.load(paths["extract"]).shape
            continue
        model, tok = load_model(model_id)
        for which in ("extract", "eval"):
            prompts = [render_prompt(tok, it["question"]) for it in items[which]]
            t0 = time.time()
            acts = capture_last_token(model, tok, prompts, batch_size)
            np.save(paths[which], acts)
            print(f"[extract] {role}/{which}: {acts.shape} in {time.time() - t0:.1f}s "
                  f"-> {paths[which].name}", flush=True)
            if which == "extract":
                shapes[role] = acts.shape
        # ONE AT A TIME: the next load must find the box empty. The `del` is load-bearing --
        # without it this frame still holds 16GB when the second checkpoint starts loading.
        unload_model(model)
        del model, tok
        free_cuda()

    a_n = np.load(out_dir / F_ACTS.format(role="nonreason", which="extract"))
    a_r = np.load(out_dir / F_ACTS.format(role="reason", which="extract"))
    e_r = np.load(out_dir / F_ACTS.format(role="reason", which="eval"))

    inputs = [provenance.describe_input(out_dir / F_ITEMS,
                                        n_items=len(items["extract"]) + len(items["eval"]))]
    for role in ROLES:
        for which in ("extract", "eval"):
            inputs.append(provenance.describe_input(
                out_dir / F_ACTS.format(role=role, which=which),
                n_items=N_EXTRACT if which == "extract" else N_EVAL))

    cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "extract"}
    rule = (
        "The direction is well defined only if both checkpoints expose the SAME hidden width "
        "and the SAME layer count -- the design's MEMORY assumption (32 layers, 4,096-wide), "
        "unverified before this run -- and only if every tested layer's direction has a "
        "non-zero norm. Refuting outcome, stated before the values are known: differing "
        "widths or layer counts, in which case moving one model's direction into the other is "
        "not defined, or a zero-norm direction, in which case there is nothing to add or "
        "remove -> HALT_DIRECTION_UNDEFINED and no generation follows.")
    if a_n.shape[1:] != a_r.shape[1:]:
        provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
            metric="Per-layer mean-difference direction at the last prompt token.",
            values={"nonreason_shape": list(a_n.shape), "reason_shape": list(a_r.shape),
                    "same_width": False},
            threshold={"same_hidden_width_required": True, "same_layer_count_required": True},
            decision_rule=rule, decision="HALT_DIRECTION_UNDEFINED",
            notes="The two checkpoints do not share a hidden-state shape. " + RERUN_NOTE)
        print("FATAL: hidden-state shapes differ; cross-model injection is not defined.")
        return 2

    n_hs, d_model = a_r.shape[1], a_r.shape[2]
    n_layers = n_hs - 1
    layers = tested_layers(n_layers)
    print(f"[extract] n_layers={n_layers} d_model={d_model} tested_layers={layers} "
          f"(~{', '.join(f'{f:.0%}' for f in DEPTH_FRACS)} depth)", flush=True)

    # Block-indexed: dirs[b] is the residual stream after block b, which is exactly the
    # tensor the steering hook on block b writes to. hidden_states[0] (embeddings) is kept in
    # the saved activation arrays but is not a steerable block, so it is not a direction.
    mean_n = a_n.mean(axis=0)
    mean_r = a_r.mean(axis=0)
    dirs = (mean_r[1:] - mean_n[1:]).astype(np.float32)          # [n_layers, d]
    norms = np.linalg.norm(dirs, axis=1).astype(np.float32)

    rng = np.random.default_rng(SEED_RANDDIR)
    rands = np.zeros((len(layers), d_model), dtype=np.float32)
    for k, lay in enumerate(layers):
        v = rng.standard_normal(d_model)
        nv = np.linalg.norm(v)
        rands[k] = (v / nv * float(norms[lay])).astype(np.float32)   # matched length

    zero_norm = [int(lay) for lay in layers if float(norms[lay]) <= 0]

    projections = {}
    for k, lay in enumerate(layers):
        ud = dirs[lay] / max(float(norms[lay]), 1e-12)
        ur = rands[k] / max(float(np.linalg.norm(rands[k])), 1e-12)
        pd, pr = {}, {}
        for i, it in enumerate(items["eval"]):
            h = e_r[i, lay + 1, :]
            pd[str(it["item_id"])] = float(h @ ud)
            pr[str(it["item_id"])] = float(h @ ur)
        projections[str(int(lay))] = {"direction": pd, "random": pr}
    (out_dir / F_PROJECTIONS).write_text(json.dumps(
        {"note": "The reasoning model's own last-prompt-token activation on each EVALUATION "
                 "problem, projected on the unit direction and on the unit matched-norm "
                 "random control, at each tested layer (block-indexed).",
         "n_layers": int(n_layers), "d_model": int(d_model),
         "tested_layers": [int(x) for x in layers],
         "random_direction_seed": SEED_RANDDIR,
         "projections": projections}, indent=2))

    np.savez(out_dir / F_DIRECTION, directions=dirs, norms=norms,
             tested_layers=np.array(layers, dtype=np.int32), random_vectors=rands,
             random_norms=np.linalg.norm(rands, axis=1).astype(np.float32),
             mean_nonreason=mean_n.astype(np.float32), mean_reason=mean_r.astype(np.float32))

    cos = float("nan")
    if len(layers) == 2:
        a, b = dirs[layers[0]], dirs[layers[1]]
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(a @ b / den) if den > 0 else float("nan")

    _, tokname = hf_token()
    values = {
        "models": {"nonreason": MODEL_NONREASON, "reason": MODEL_REASON},
        "n_layers": int(n_layers), "d_model": int(d_model),
        "tested_layers": [int(x) for x in layers],
        "depth_fractions": list(DEPTH_FRACS),
        "direction_norms_all_layers": [float(x) for x in norms],
        "direction_norm_by_tested_layer": {str(int(l)): float(norms[l]) for l in layers},
        "random_norm_by_tested_layer": {
            str(int(l)): float(np.linalg.norm(rands[k])) for k, l in enumerate(layers)},
        "mean_direction_norm": float(norms.mean()),
        "max_direction_norm": float(norms.max()),
        "cosine_between_tested_layer_directions": cos,
        "n_extraction_items": N_EXTRACT, "n_evaluation_items": N_EVAL,
        "item_seed": SEED_ITEMS, "random_direction_seed": SEED_RANDDIR,
        "hf_token_env": tokname, "zero_norm_tested_layers": zero_norm,
    }
    decision = "HALT_DIRECTION_UNDEFINED" if zero_norm else "DIRECTION_DEFINED"
    p = provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="Per-layer mean-difference direction (reasoning mean minus non-reasoning mean) "
               "at the last prompt token over the 50-problem extraction set, with per-layer "
               "norms and one matched-norm random vector per tested layer.",
        values=values,
        threshold={"same_hidden_width_required": True, "same_layer_count_required": True,
                   "min_direction_norm": 0.0},
        decision_rule=rule, decision=decision,
        notes="One direction PER LAYER, not one reused across layers -- the design's own "
              "stated reading of 'a direction'. Strengths downstream are multiples of the "
              "layer's own direction norm, so the random control is matched in length at "
              "each tested layer. " + RERUN_NOTE)
    print(f"[extract] {decision} -> {p}", flush=True)
    return 1 if zero_norm else 0


# ---------------------------------------------------------------------------
# STAGE generate
# ---------------------------------------------------------------------------
def _decode_rows(tok, cfg, cell, role, model_id, batch_items, prompts, enc, out_ids,
                 stops, opens) -> list[dict]:
    """One batch of generations -> one row per generation, saved verbatim."""
    rows = []
    stop_set = set(stops)
    n_in = enc["input_ids"].shape[1]
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    new = out_ids[:, n_in:]

    def tk(s):
        return tok.encode(s, add_special_tokens=False) if s else []

    for j, it in enumerate(batch_items):
        ids = new[j].tolist()
        stop_at = next((k for k, t in enumerate(ids) if t in stop_set), None)
        kept = ids if stop_at is None else ids[:stop_at]
        stopped = stop_at is not None
        text = tok.decode(kept, skip_special_tokens=True)
        counts = think_token_counts(text, tk, opens)
        degen, why = is_degenerate(len(kept), stopped, kept)
        pred = extract_answer(text, opens)
        rows.append({
            "seed_label": SEED, "pipeline": PIPELINE,
            "cell": cell["cell"], "arm": cell["arm"], "role": role, "model": model_id,
            "layer": cell["layer"], "strength": cell["strength"],
            "mode": cell["mode"], "vector": cell["vector"],
            "steer_prompt_positions": False,
            "decode": cfg["decode"], "decode_seed": cfg["seed"],
            "temperature": cfg["temperature"], "top_p": cfg["top_p"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "item_id": it["item_id"], "ds_index": it["ds_index"], "gold": it["gold"],
            "prompt": prompts[j], "prompt_opens_think": bool(opens),
            "n_prompt_tokens": int(prompt_lens[j]),
            "text": text,
            "n_new_tokens": len(kept), "stopped_on_stop_token": bool(stopped),
            "tokens_in_think": counts["tokens_in_think"],
            "tokens_out_think": counts["tokens_out_think"],
            "well_formed": bool(counts["well_formed"]), "think_kind": counts["kind"],
            "n_open_tags": counts["n_open_explicit"],
            "n_open_tags_effective": counts["n_open_effective"],
            "n_close_tags": counts["n_close"], "implicit_open": bool(counts["implicit_open"]),
            "degenerate": bool(degen), "degenerate_reason": why,
            "pred_answer": pred, "correct": exact_match(pred, it["gold"]),
        })
    return rows


def run_role(role: str, out_dir: Path, items: dict, npz, layers: list[int],
             batch_size: int) -> tuple[Path, int, list[dict]]:
    model_id = MODEL_NONREASON if role == "nonreason" else MODEL_REASON
    cells = [c for c in build_cells(layers) if c["role"] == role]
    dirs, rands = npz["directions"], npz["random_vectors"]
    lay_index = {int(x): k for k, x in enumerate(npz["tested_layers"].tolist())}

    model, tok = load_model(model_id)
    if model.config.hidden_size != dirs.shape[1]:
        unload_model(model)
        raise SystemExit(
            f"FATAL: {model_id} hidden size {model.config.hidden_size} != direction width "
            f"{dirs.shape[1]}; the direction cannot be moved into this model.")
    stops = stop_token_ids(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ev = items["eval"]
    prompts_all = [render_prompt(tok, it["question"]) for it in ev]
    opens = prompt_opens_think(prompts_all[0])
    print(f"[generate] {role}: {len(cells)} cells x {len(ev)} items x "
          f"{len(decode_configs())} decodes = {len(cells) * len(ev) * len(decode_configs())} "
          f"rows | prompt_opens_think={opens} | stop_ids={stops}", flush=True)

    path = out_dir / F_GENS.format(role=role)
    rows_written, cell_notes = 0, []
    t_start = time.time()
    with open(path, "w") as fh:
        for ci, c in enumerate(cells):
            vec = None
            if c["mode"] != "none":
                k = lay_index[int(c["layer"])]
                vec = dirs[int(c["layer"])] if c["vector"] == "direction" else rands[k]
            ctx = (_NullCtx() if c["mode"] == "none"
                   else Intervention(model, int(c["layer"]), vec, c["mode"],
                                     float(c["strength"] or 0.0)))
            t_cell = time.time()
            n_cell = 0
            with ctx:
                for cfg in decode_configs():
                    torch.manual_seed(cfg["seed"])      # once per (cell, decode)
                    for i in range(0, len(ev), batch_size):
                        chunk = ev[i:i + batch_size]
                        bp = prompts_all[i:i + batch_size]
                        enc = tok(bp, return_tensors="pt", padding=True,
                                  add_special_tokens=False).to(model.device)
                        kw = {"max_new_tokens": MAX_NEW_TOKENS, "use_cache": True,
                              "pad_token_id": pad_id, "eos_token_id": stops}
                        if cfg["decode"] == "greedy":
                            kw["do_sample"] = False
                        else:
                            kw.update(do_sample=True, temperature=cfg["temperature"],
                                      top_p=cfg["top_p"])
                        with torch.no_grad():
                            out_ids = model.generate(**enc, **kw)
                        rows = _decode_rows(tok, cfg, c, role, model_id, chunk, bp, enc,
                                            out_ids, stops, opens)
                        for r in rows:
                            fh.write(json.dumps(r) + "\n")
                        rows_written += len(rows)
                        n_cell += len(rows)
                        del out_ids, enc
            fh.flush()
            applied = getattr(ctx, "n_applied", 0)
            cell_notes.append({"cell": c["cell"], "n_rows": n_cell,
                               "hook_forward_calls": int(applied),
                               "seconds": round(time.time() - t_cell, 1)})
            done = ci + 1
            el = time.time() - t_start
            print(f"[generate] {role} {done}/{len(cells)} {c['cell']}: {n_cell} rows in "
                  f"{time.time() - t_cell:.0f}s | elapsed {el / 60:.1f}m | "
                  f"eta {el / done * (len(cells) - done) / 60:.1f}m", flush=True)
    # ctx is cleared FIRST: the last Intervention object holds a reference to the model, so
    # dropping only `model` would leave the weights resident for the second role's load.
    ctx = None
    unload_model(model)
    del model, tok, ctx
    free_cuda()
    return path, rows_written, cell_notes


def stage_generate(out_dir: Path, batch_size: int, roles: tuple[str, ...]) -> int:
    _require_torch("generate")
    items = json.loads((out_dir / F_ITEMS).read_text())
    npz = np.load(out_dir / F_DIRECTION)
    layers = [int(x) for x in npz["tested_layers"].tolist()]
    rc = 0
    for role in roles:
        path, n_rows, cell_notes = run_role(role, out_dir, items, npz, layers, batch_size)
        n_cells = len([c for c in build_cells(layers) if c["role"] == role])
        expected = n_cells * len(items["eval"]) * len(decode_configs())
        rows = [json.loads(li) for li in path.read_text().splitlines() if li.strip()]
        degen = sum(1 for r in rows if r["degenerate"])
        cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "generate", "role": role}
        values = {
            "model": MODEL_NONREASON if role == "nonreason" else MODEL_REASON,
            "n_cells": n_cells, "n_items": len(items["eval"]),
            "n_decodes_per_item": len(decode_configs()),
            "n_rows_expected": expected, "n_rows_written": n_rows,
            "n_degenerate": degen,
            "degenerate_rate_pct": _rate(degen, len(rows)),
            "per_cell": cell_notes,
            "degenerate_rate_by_cell_pct": {
                c["cell"]: cell_summary(rows, c["cell"])["degenerate_rate_pct"]
                for c in build_cells(layers) if c["role"] == role},
            "decode_seeds": list(SEEDS_SAMPLE), "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS, "batch_size": batch_size,
            "steer_prompt_positions": False,
        }
        decision = "ROWS_COMPLETE" if n_rows == expected else "INCOMPLETE_CELLS"
        p = provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
            inputs=[provenance.describe_input(out_dir / F_ITEMS, n_items=len(items["eval"])),
                    provenance.describe_input(out_dir / F_DIRECTION, n_items=len(layers)),
                    provenance.describe_input(path, n_items=n_rows)],
            metric="Generations per pre-registered cell: 50 evaluation problems x 4 decodes "
                   "(seeds 11/12/13 at temperature 0.7 / top_p 0.9, plus 1 greedy), max 512 "
                   "new tokens, each model under its own default chat template.",
            values=values,
            threshold={"n_rows_expected": expected},
            decision_rule="This stage decides only whether the pre-registered rows exist; the "
                          "endpoints decide the hypothesis. Rows written must equal cells x 50 "
                          "items x 4 decodes. Refuting outcome for this stage: fewer rows than "
                          "that -> INCOMPLETE_CELLS, and any endpoint reading this role is "
                          "computed over a truncated design.",
            decision=decision,
            notes=DEGEN_NOTE + " " + RERUN_NOTE)
        print(f"[generate] {role}: {decision} {n_rows}/{expected} rows -> {p}", flush=True)
        if decision != "ROWS_COMPLETE":
            rc = 1
    return rc


# ---------------------------------------------------------------------------
# STAGE endpoints
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(li) for li in path.read_text().splitlines() if li.strip()]


def stage_endpoints(out_dir: Path) -> int:
    """Stdlib only: it reads the item list, the two generation files and the projections the
    extract stage saved. No model, no numpy, no GPU."""
    items_p = out_dir / F_ITEMS
    proj_p = out_dir / F_PROJECTIONS
    gens = {r: out_dir / F_GENS.format(role=r) for r in ROLES}
    missing = [str(p) for p in [items_p, proj_p, *gens.values()] if not p.exists()]
    if missing:
        print("FATAL: endpoints stage is missing its inputs: " + ", ".join(missing))
        return 2

    items = json.loads(items_p.read_text())
    proj_obj = json.loads(proj_p.read_text())
    projections = proj_obj["projections"]
    layers = [int(x) for x in proj_obj["tested_layers"]]
    rows_nr, rows_r = _read_jsonl(gens["nonreason"]), _read_jsonl(gens["reason"])
    rows_all = rows_nr + rows_r
    cells = build_cells(layers)
    print(f"[endpoints] {len(rows_nr)} non-reasoning rows + {len(rows_r)} reasoning rows, "
          f"{len(cells)} cells, tested layers {layers}", flush=True)
    if not rows_r or not rows_nr:
        print("FATAL: one role produced no rows; endpoints would report over a truncated "
              "design.")
        return 2

    base_inputs = [
        provenance.describe_input(items_p, n_items=len(items["eval"])),
        provenance.describe_input(proj_p, n_items=len(layers)),
        provenance.describe_input(gens["nonreason"], n_items=len(rows_nr)),
        provenance.describe_input(gens["reason"], n_items=len(rows_r)),
    ]

    v1, d1 = endpoint1(rows_r, projections)
    v2, d2 = endpoint2(rows_nr, layers)
    v3, d3 = endpoint3(rows_r, layers)
    v4, d4 = endpoint4(rows_r, layers)
    v5, d5 = endpoint5(rows_all, cells, v3["cells_called_reasoning_enhanced"])

    readings = {
        1: ("The design names no layer for this correlation and reads 'a direction' as one "
            "per layer, so it is computed at EACH tested layer and passes on at least one, "
            "matching the 'in at least one layer' wording the other endpoints use. Predicted "
            "direction is positive: the direction is reasoning minus non-reasoning. 'No "
            "larger than the control' is read as |r| <= |r_control|. The permutation test "
            f"uses {PERM_N} label shuffles at fixed seed {PERM_SEED}; the design names a "
            "permutation test and a p-value, not a resample count, so these two are "
            "implementation constants, not design parameters."),
        2: ("Rates are computed over non-degenerate rows; 'the excess sits mostly in "
            "degenerate output' is read as more than half of the direction cell's well-formed "
            "rows being degenerate rows. The gate is at the higher strength (+2x), with the "
            "lower strength reported beside it. A stray closing tag never counts as a "
            "well-formed block for this model, whose template opens nothing."),
        3: ("'Points' is read as percentage points of the percent change versus baseline. "
            "Means are over non-degenerate rows. The degenerate rate is REPORTED for every "
            "cell in this leg; the design states no ceiling here, so none is applied."),
        4: ("Same readings as endpoint 3. 'Shrinks no more than zeroing the random-direction "
            "control does' is read as the control's percent change being at least as negative "
            "as the direction's."),
        5: ("'Called reasoning enhanced' is read as the reasoning-model direction-added cells "
            "whose think-tag length grew by 25%+ versus baseline at the positive strength -- "
            "endpoint 3's growth criterion. The margin over the random control is NOT required "
            "here: this endpoint's refuting outcome is about a condition that shows longer "
            "think-tags, so every such condition is gated, not only the ones that also beat "
            "the control. Accuracy is exact match on the final numeric answer, over "
            "non-degenerate rows, and is reported for every cell."),
    }

    written = []
    for n, (values, decision) in enumerate(
            [(v1, d1), (v2, d2), (v3, d3), (v4, d4), (v5, d5)], start=1):
        spec = ENDPOINTS_SPEC[n - 1]
        cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "endpoints", "endpoint": n}
        rule = (f"THRESHOLD (design, verbatim): {spec['threshold']} "
                f"REFUTING OUTCOME (design, verbatim): {spec['refuting_outcome']} "
                f"READING: {readings[n]}")
        p = provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=base_inputs,
            metric=spec["metric"], values=values,
            threshold={"threshold_text": spec["threshold"],
                       "refuting_outcome_text": spec["refuting_outcome"],
                       "numeric": _threshold_numbers(n)},
            decision_rule=rule, decision=decision,
            notes=DEGEN_NOTE + " " + RERUN_NOTE)
        written.append((n, decision, p))
        print(f"[endpoints] E{n}: {decision} -> {Path(p).name}", flush=True)
    print("[endpoints] " + " | ".join(f"E{n}={d}" for n, d, _ in written), flush=True)
    return 0


def _threshold_numbers(n: int) -> dict:
    return {
        1: {"r_floor": E1_R_FLOOR, "r_refute_below": E1_R_REFUTE, "p_max": E1_P_MAX,
            "permutation_resamples": PERM_N, "permutation_seed": PERM_SEED},
        2: {"gap_pass_pct_points": E2_GAP_PASS, "gap_refute_pct_points": E2_GAP_REFUTE,
            "degenerate_ceiling_pct": E2_DEGEN_CEILING,
            "unchanged_ceiling_pct": E2_UNCHANGED_CEILING},
        3: {"pct_pass": E3_PCT_PASS, "margin_pass_pct_points": E3_MARGIN_PASS,
            "pct_refute_below": E3_PCT_REFUTE},
        4: {"pct_pass": E4_PCT_PASS, "margin_pass_pct_points": E4_MARGIN_PASS,
            "pct_refute_below": E4_PCT_REFUTE},
        5: {"max_accuracy_drop_pct_points": E5_ACC_DROP_MAX},
    }[n]


def stage_plan() -> int:
    """Model-free: what the run will produce, before it is bought."""
    layers = tested_layers(32)      # the design's MEMORY assumption; verified in extract
    cells = build_cells(layers)
    decodes = decode_configs()
    print(f"seed {SEED} pipeline {PIPELINE} | design {DESIGN_RECORD}")
    print(f"models: {MODEL_NONREASON} (non-reasoning), {MODEL_REASON} (reasoning)")
    print(f"tested layers (assuming 32): {layers} | induce {INDUCE_STRENGTHS} | "
          f"steer {STEER_STRENGTHS} | zeroing at both layers")
    print(f"decodes per item: {[ (d['decode'], d['seed']) for d in decodes ]}")
    for c in cells:
        print(f"  {c['role']:9s} {c['cell']:26s} arm={c['arm']:15s} mode={c['mode']:4s} "
              f"vector={c['vector']}")
    n = len(cells) * N_EVAL * len(decodes)
    print(f"cells={len(cells)} items={N_EVAL} decodes={len(decodes)} -> rows={n}")
    print(f"budget_usd (design GUESS)={BUDGET_USD}")
    return 0


# ---------------------------------------------------------------------------
# SELFTEST -- model-free, stdlib only. A broken instrument must show on the CPU.
# ---------------------------------------------------------------------------
def _plant(cell, item_id=0, think=0, out=0, wf=True, degen=False, correct=True):
    return {"cell": cell, "item_id": item_id, "tokens_in_think": think,
            "tokens_out_think": out, "well_formed": wf, "degenerate": degen,
            "correct": correct}


def _selftest() -> int:
    fails: list[str] = []
    n_checks = [0]

    def chk(cond, msg):
        n_checks[0] += 1
        if not cond:
            fails.append(msg)

    # -- 1. the think-tag parser, including THE TRAP -----------------------------------
    p = parse_think("<think>a b c</think>Answer: 5", False)
    chk(p["well_formed"] and p["kind"] == "paired", f"explicit pair: {p['kind']}")
    chk(p["think_text"] == "a b c" and p["post_text"] == "Answer: 5", f"spans: {p}")

    trap = "reason reason</think>\n\nThe answer is 42."
    p = parse_think(trap, True)          # reasoning model: the template opened the tag
    chk(p["well_formed"], "TRAP: closing-tag-only generation must be well-formed when the "
                          "prompt opened the block")
    chk(p["kind"] == "implicit_paired" and p["implicit_open"] and p["n_open_explicit"] == 0,
        f"TRAP kind: {p['kind']} implicit={p['implicit_open']}")
    chk(p["think_text"] == "reason reason", f"TRAP think span: {p['think_text']!r}")
    chk(p["post_text"] == "\n\nThe answer is 42.", f"TRAP post span: {p['post_text']!r}")

    p = parse_think(trap, False)         # same text from a model whose template opens nothing
    chk(not p["well_formed"] and p["kind"] == "stray_close",
        f"stray close must never be well-formed: {p['kind']}")
    chk(p["think_text"] == "reason reason", "stray close still counts its tag text")

    p = parse_think("still thinking", True)
    chk(p["kind"] == "open_unclosed" and not p["well_formed"] and p["implicit_open"],
        f"truncated inside a prompt-opened block: {p['kind']}")
    chk(p["think_text"] == "still thinking", "truncated block is all think text")

    p = parse_think("plain answer 7", False)
    chk(p["kind"] == "no_tags" and not p["well_formed"] and p["post_text"] == "plain answer 7",
        f"no tags: {p}")

    p = parse_think("<think>a<think>b</think>c", False)
    chk(p["kind"] == "malformed" and not p["well_formed"], f"double open: {p['kind']}")
    p = parse_think("<think>a</think>b</think>", False)
    chk(p["kind"] == "malformed" and not p["well_formed"], f"double close: {p['kind']}")
    p = parse_think("<think>own tag</think>done", True)
    chk(p["well_formed"] and p["kind"] == "paired" and not p["implicit_open"],
        f"prompt opened but generation opened its own: {p['kind']}")

    # prompt_opens_think, detected from the rendered prompt rather than the model id
    chk(prompt_opens_think("<|User|>q<|Assistant|><think>\n"), "R1 template must read as open")
    chk(not prompt_opens_think(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"), "Llama template opens nothing")
    chk(not prompt_opens_think("example: <think>x</think> now answer:"),
        "a closed example block is not an open block")

    # token counts in / out of the tags, with a fake tokenizer
    tk = str.split
    c = think_token_counts(trap, tk, True)
    chk(c["tokens_in_think"] == 2 and c["tokens_out_think"] == 4, f"trap counts: {c}")
    c = think_token_counts("pre <think>a b</think> post here", tk, False)
    chk(c["tokens_in_think"] == 2 and c["tokens_out_think"] == 3, f"pre+post counts: {c}")

    # -- 2. the degenerate detector ----------------------------------------------------
    d, why = is_degenerate(MAX_NEW_TOKENS, False, list(range(MAX_NEW_TOKENS)))
    chk(d and why == "cap_no_stop", f"cap hit with no stop token: {d} {why}")
    d, why = is_degenerate(MAX_NEW_TOKENS, True, list(range(MAX_NEW_TOKENS)))
    chk(not d, f"cap reached WITH a stop token is not degenerate: {why}")
    d, why = is_degenerate(200, True, list(range(50)) + [7, 8] * 20)
    chk(d and why.startswith("repeat_period2_x"), f"2-token repeat x20: {d} {why}")
    d, why = is_degenerate(200, True, list(range(60)) + [5] * 30)
    chk(d and "repeat_period1_x30" in why, f"1-token repeat x30: {d} {why}")
    d, why = is_degenerate(200, True, list(range(60)) + [7, 8] * 14)
    chk(not d, f"14 repeats is under the 15+ rule: {why}")
    d, why = is_degenerate(120, True, list(range(120)))
    chk(not d, f"clean generation: {why}")
    hit, per, reps = repeat_degenerate([9] * 400)
    chk(hit and per == 1 and reps == DEGEN_REPEAT_WINDOW,
        f"repeat window is the last {DEGEN_REPEAT_WINDOW}: {per} {reps}")

    # -- 3. exact-match answer extraction ---------------------------------------------
    chk(extract_answer("The answer is 18.", False) == 18.0, "bare final number")
    chk(extract_answer("work work \\boxed{72}", False) == 72.0, "boxed answer")
    chk(extract_answer("It costs $1,000.00 total", False) == 1000.0, "comma/currency")
    chk(gsm8k_gold("blah\n#### 1,000") == 1000.0, "gold after ####")
    chk(exact_match(extract_answer("It costs $1,000.00 total", False),
                    gsm8k_gold("x\n#### 1,000")), "normalised exact match")
    chk(extract_answer("scratch 72 more scratch 13</think>The answer is 42.", True) == 42.0,
        "the answer comes from OUTSIDE the think block, not from its scratch work")
    chk(extract_answer("let me compute 5+5=10", True) == 10.0,
        "truncated inside the block falls back to the whole text")
    chk(extract_answer("I do not know", False) is None, "no number -> None")
    chk(not exact_match(None, 5.0), "None never matches")
    chk(extract_answer("the change is -7", False) == -7.0, "negative answer")
    chk(gsm8k_gold("#### 18") == 18.0 and normalise_number("18.") == 18.0, "trailing period")

    # -- 4. the cell plan -------------------------------------------------------------
    chk(tested_layers(32) == [16, 24], f"32 layers -> 16, 24: {tested_layers(32)}")
    chk(tested_layers(40) == [20, 30], f"same relative depths at 40 layers: {tested_layers(40)}")
    cells = build_cells([16, 24])
    chk(len(cells) == 22, f"the design declares 22 cells, got {len(cells)}")
    chk(len(set(c['cell'] for c in cells)) == 22, "cell keys must be unique")
    chk(len([c for c in cells if c["role"] == "nonreason"]) == 9, "9 non-reasoning cells")
    chk(len([c for c in cells if c["role"] == "reason"]) == 13, "13 reasoning cells")
    chk(len(set(c["arm"] for c in cells)) == 8, "the design declares 8 arms")
    dcs = decode_configs()
    chk(len(dcs) == 4 and sorted(d["seed"] for d in dcs) == [0, 11, 12, 13],
        f"4 decodes, seeds 11/12/13 + greedy: {dcs}")
    chk(len(cells) * N_EVAL * len(dcs) == 4400, "the design's 4,400 generations")
    chk("not replication" in RERUN_NOTE, "a same-box rerun must never be called replication")

    # -- 5. endpoint arithmetic on planted rows with known decisions -------------------
    ids = list(range(12))
    y = {i: 10.0 * (i + 1) for i in ids}
    rows = []
    for i in ids:                       # two baseline rows per item, plus one degenerate
        rows += [_plant("base_reason", i, think=y[i]), _plant("base_reason", i, think=y[i])]
        rows.append(_plant("base_reason", i, think=9999.0, degen=True))
    alt = {i: float(i % 2) for i in ids}
    proj_pass = {"16": {"direction": {str(i): y[i] for i in ids},
                        "random": {str(i): alt[i] for i in ids}}}
    v, d = endpoint1(rows, proj_pass, n_perm=200, perm_seed=PERM_SEED)
    chk(d == "SUPPORTED", f"E1 perfect correlation -> SUPPORTED, got {d}")
    chk(abs(v["per_layer"]["layer_16"]["r_direction"] - 1.0) < 1e-9, f"E1 r: {v}")
    chk(v["mean_think_tokens_baseline"] == mean(y.values()),
        "E1 must average the 4 decodes per item and drop the degenerate row")
    proj_small = {"16": {"direction": {str(i): alt[i] for i in ids},
                         "random": {str(i): 0.0 if i else 1.0 for i in ids}}}
    v2_, d2_ = endpoint1(rows, proj_small, n_perm=200)
    chk(abs(v2_["per_layer"]["layer_16"]["r_direction"]) < E1_R_REFUTE,
        f"E1 planted r must be under the 0.15 refute line: {v2_['per_layer']}")
    chk(d2_ == "REFUTED", f"E1 tiny correlation -> REFUTED, got {d2_}")
    proj_tie = {"16": {"direction": {str(i): y[i] for i in ids},
                       "random": {str(i): y[i] for i in ids}}}
    _, d_tie = endpoint1(rows, proj_tie, n_perm=50)
    chk(d_tie == "REFUTED", f"E1 control ties the direction -> REFUTED, got {d_tie}")

    # E2: induction well-formed rate versus the random control at the higher strength
    hi = max(INDUCE_STRENGTHS)
    cd, cr = cell_key("induce_dir", 16, hi), cell_key("induce_rand", 16, hi)
    base_nr = [_plant("base_nonreason", i, wf=False) for i in range(10)]
    r2 = base_nr + [_plant(cd, i, wf=i < 8) for i in range(10)] + \
        [_plant(cr, i, wf=i < 1) for i in range(10)]
    v, d = endpoint2(r2, [16])
    chk(d == "SUPPORTED", f"E2 70-point gap -> SUPPORTED, got {d}")
    chk(abs(v["per_cell"][f"L16_s{hi:+.1f}"]["gap_pct_points"] - 70.0) < 1e-9,
        f"E2 gap: {v['per_cell']}")
    r2b = base_nr + [_plant(cd, i, wf=i < 25) for i in range(50)] + \
        [_plant(cr, i, wf=i < 24) for i in range(50)]
    _, d = endpoint2(r2b, [16])
    chk(d == "REFUTED", f"E2 a 2-point gap is under the 5-point refute line, got {d}")
    r2b2 = base_nr + [_plant(cd, i, wf=i < 6) for i in range(10)] + \
        [_plant(cr, i, wf=i < 5) for i in range(10)]
    _, d = endpoint2(r2b2, [16])
    chk(d == "INDETERMINATE",
        f"E2 a 10-point gap clears the refute line but misses the 15-point pass, got {d}")
    r2c = [_plant("base_nonreason", i, wf=i < 6) for i in range(10)] + \
        [_plant(cd, i, wf=i < 8) for i in range(10)] + \
        [_plant(cr, i, wf=False) for i in range(10)]
    _, d = endpoint2(r2c, [16])
    chk(d == "REFUTED_UNCHANGED_RATE_ABOVE_CEILING",
        f"E2 unchanged rate over 5% refutes outright, got {d}")
    # the 40% degenerate ceiling blocks the pass on its own, with the well-formed rows
    # themselves sitting in the NON-degenerate half (so the other refuting clause is quiet)
    r2d = base_nr + [_plant(cd, i, wf=i < 6, degen=i >= 6) for i in range(10)] + \
        [_plant(cr, i, wf=False) for i in range(10)]
    v, d = endpoint2(r2d, [16])
    chk(d == "INDETERMINATE",
        f"E2 a 40% degenerate rate must block the pass even at a 100-point gap, got {d}")
    chk(v["per_cell"][f"L16_s{hi:+.1f}"]["direction"]["degenerate_rate_pct"] == 40.0
        and not v["per_cell"][f"L16_s{hi:+.1f}"]["excess_mostly_degenerate"],
        f"E2 degenerate ceiling cell: {v['per_cell'][f'L16_s{hi:+.1f}']}")
    # and when the well-formed excess DOES sit in the degenerate rows, that is refuting
    r2e = base_nr + [_plant(cd, i, wf=True, degen=i >= 4) for i in range(10)] + \
        [_plant(cr, i, wf=False) for i in range(10)]
    v, d = endpoint2(r2e, [16])
    chk(v["per_cell"][f"L16_s{hi:+.1f}"]["excess_mostly_degenerate"],
        "E2 must see the excess sitting in degenerate output")
    chk(d == "REFUTED", f"E2 excess in degenerate output is a refuting outcome, got {d}")

    # E3: enhance / suppress against baseline and control
    pos, neg = max(STEER_STRENGTHS), min(STEER_STRENGTHS)
    base_r = [_plant("base_reason", i, think=100.0) for i in range(10)]

    def leg(lay, s, dv, rv):
        return ([_plant(cell_key("steer_dir", lay, s), i, think=dv) for i in range(10)] +
                [_plant(cell_key("steer_rand", lay, s), i, think=rv) for i in range(10)])

    r3 = base_r + leg(16, pos, 140.0, 105.0) + leg(16, neg, 60.0, 98.0)
    v, d = endpoint3(r3, [16])
    chk(d == "SUPPORTED", f"E3 +40%/-40% against a flat control -> SUPPORTED, got {d}")
    chk(v["cells_called_reasoning_enhanced"] == [cell_key("steer_dir", 16, pos)],
        f"E3 must name the enhanced cell: {v['cells_called_reasoning_enhanced']}")
    chk(abs(v["per_cell"][f"L16_s{pos:+.1f}"]["pct_change_direction"] - 40.0) < 1e-9,
        "E3 percent change")
    r3b = base_r + leg(16, pos, 103.0, 101.0) + leg(16, neg, 98.0, 99.0)
    _, d = endpoint3(r3b, [16])
    chk(d == "REFUTED", f"E3 changes under 10% at both strengths -> REFUTED, got {d}")
    r3c = base_r + leg(16, pos, 140.0, 150.0) + leg(16, neg, 60.0, 50.0)
    _, d = endpoint3(r3c, [16])
    chk(d == "REFUTED", f"E3 control beats the direction -> REFUTED, got {d}")
    r3d = base_r + leg(16, pos, 140.0, 105.0) + leg(16, neg, 95.0, 99.0)
    _, d = endpoint3(r3d, [16])
    chk(d == "REFUTED",
        f"E3 one strength under 10% in every layer is a refuting outcome, got {d}")

    # E4: removal by zeroing
    r4 = base_r + [_plant(cell_key("zero_dir", 16), i, think=70.0) for i in range(10)] + \
        [_plant(cell_key("zero_rand", 16), i, think=95.0) for i in range(10)]
    v, d = endpoint4(r4, [16])
    chk(d == "SUPPORTED", f"E4 -30% against -5% control -> SUPPORTED, got {d}")
    chk(abs(v["per_cell"]["L16"]["margin_pct_points"] + 25.0) < 1e-9, f"E4 margin: {v}")
    r4b = base_r + [_plant(cell_key("zero_dir", 16), i, think=98.0) for i in range(10)] + \
        [_plant(cell_key("zero_rand", 16), i, think=99.0) for i in range(10)]
    _, d = endpoint4(r4b, [16])
    chk(d == "REFUTED", f"E4 -2% -> REFUTED, got {d}")
    r4c = base_r + [_plant(cell_key("zero_dir", 16), i, think=70.0) for i in range(10)] + \
        [_plant(cell_key("zero_rand", 16), i, think=60.0) for i in range(10)]
    _, d = endpoint4(r4c, [16])
    chk(d == "REFUTED", f"E4 zeroing the control shrinks more -> REFUTED, got {d}")

    # E5: accuracy gate beside the enhanced conditions
    enh = cell_key("steer_dir", 16, pos)
    cells5 = [{"cell": "base_reason"}, {"cell": enh}]
    acc_base = [_plant("base_reason", i, think=100.0, correct=i < 9) for i in range(10)]
    r5 = acc_base + [_plant(enh, i, think=140.0, correct=i < 15) for i in range(20)]
    v, d = endpoint5(r5, cells5, [enh])
    chk(d == "REFUTED_INFLATED_OUTPUT",
        f"E5 90% -> 75% is a 15-point drop -> inflated output, got {d}")
    chk(abs(v["gated_cells"][enh]["accuracy_drop_pct_points"] - 15.0) < 1e-9,
        f"E5 drop: {v['gated_cells']}")
    r5b = acc_base + [_plant(enh, i, think=140.0, correct=i < 17) for i in range(20)]
    _, d = endpoint5(r5b, cells5, [enh])
    chk(d == "WITHIN_ACCURACY_GATE", f"E5 5-point drop is inside the gate, got {d}")
    _, d = endpoint5(r5, cells5, [])
    chk(d == "NOT_APPLICABLE_NO_CONDITION_CALLED_ENHANCED",
        f"E5 with nothing called enhanced decides nothing, got {d}")

    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", f"PASS ({n_checks[0]} checks)" if not fails
          else f"{len(fails)} FAILURES of {n_checks[0]} checks")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed A pipeline v2 runner (see module docstring)")
    ap.add_argument("--stage", choices=["extract", "generate", "endpoints"])
    ap.add_argument("--out", default="results_seedA/out",
                    help="result directory (records, activations, generations)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--role", choices=["nonreason", "reason", "both"], default="both",
                    help="generate stage only: restart one half without redoing the other")
    ap.add_argument("--force", action="store_true",
                    help="extract stage only: recapture activations already on disk")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.plan:
        return stage_plan()
    if not a.stage:
        ap.print_help()
        return 2
    out = Path(a.out)
    if a.stage == "extract":
        return stage_extract(out, a.batch_size, a.force)
    if a.stage == "generate":
        roles = ROLES if a.role == "both" else (a.role,)
        return stage_generate(out, a.batch_size, roles)
    return stage_endpoints(out)


if __name__ == "__main__":
    sys.exit(main())
