#!/usr/bin/env python3
"""Does one model family's "rate your own happiness after a failure" drop hold at other sizes?

WHAT THIS MEASURES, IN PLAIN WORDS
A chat model is shown a short conversation in which it just failed a small task: a
mechanical checker reports its three answers wrong. It is then asked, on a scale of 0
to 9, how happy it is right now. We do not let it talk: we read the probability it puts
on each of the ten single-digit answers in one forward pass. An earlier study on the
9-billion-parameter model found this happiness number drops about 0.9 points when the
failure is the model's OWN prior turn, versus barely moving when the identical failure
belongs to a DIFFERENT assistant quoted in the prompt. This runner asks whether that
self-versus-other drop also appears at 2 billion and 27 billion parameters, and at what
size the digit read-out is even trustworthy (enough probability lands on a digit).

The design object in records/seedC_scale_design_diff_triage.json is the complete spec.
Every number it marks GUESS is binding and is copied here unchanged: 20 wordings per
arm; a read-out counts as trustworthy only if at least half the next-token probability
lands on a digit (0.5); a drop counts as present at 0.5 points, absent inside plus or
minus 0.2 points; the freshly re-run 9-billion instruction-tuned check must recover a
drop between 0.75 and 1.05 points or every new-size comparison is put on hold.

THREE STAGES, EACH RUNNABLE ON ITS OWN
  extract    load ONE model size, run its four arms (base and instruction-tuned, times
             the failure attributed to itself vs to another assistant), 20 wordings each.
             One forward pass per wording. Save the ten-way digit probabilities, the
             probability-weighted expected rating, the digit mass, and the full prompt.
  generate   same four arms, but let the model answer in words (greedy, temperature 0,
             seed 0). Save every answer verbatim. Strip any reasoning tags, flag
             degenerate answers (hit the length cap, or looped), and read off the single
             digit it states. This is an audit trail beside the probability read-out; the
             endpoints below do NOT depend on it.
  endpoints  model-free, run LOCALLY after all three sizes' per-row files are fetched and
             combined. It does the only cross-size arithmetic, so it must not run on any
             one box. It reports, in this order: (1) the 9-billion instruction-tuned
             recovery check, (2) which sizes clear the digit-mass trustworthiness gate,
             (3) the self-minus-other rating drop for each size that clears it. Any arm
             that failed to load or failed the gate is reported as "generality untested
             at that size", never folded silently into a generality conclusion.

EXECUTION LAYOUT (one model size per box, three boxes in parallel)
  2b and 9b fit a 40GB GPU; 27b loads in bf16 (~54GB) on an 80GB GPU. Models load one at
  a time. 2b and 9b run in fp16; only 27b uses bf16, because only the design says bf16.
  The Hugging Face access token is read from the environment variable HF_KEY_ONE.

  python3 run_seedC.py --selftest                               # stdlib only, no GPU
  python3 run_seedC.py --plan                                   # counts, no GPU
  python3 run_seedC.py --stage extract  --size 2b  --out results_seedC_2b/out
  python3 run_seedC.py --stage generate --size 2b  --out results_seedC_2b/out
  python3 run_seedC.py --stage endpoints \\
      --in-dirs results_seedC_2b/out results_seedC_9b/out results_seedC_27b/out \\
      --out results_seedC/out

A same-box re-run at the same batch size reproduces these numbers; that is reproduction,
not replication, and this runner never calls a same-box re-run a replication.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

# provenance.py is stdlib-only and is shipped beside this file on the box; locally it lives
# under frozen/. Import it from whichever is present, so --selftest and endpoints work on a
# laptop and the on-box stages work on the box. It is NOT a heavy import.
try:
    import provenance  # type: ignore
except Exception:  # noqa: BLE001
    sys.path.insert(0, str(Path(__file__).resolve().parent / "frozen"))
    import provenance  # type: ignore

# ---------------------------------------------------------------------------
# Binding constants -- every one is copied from the design object, unchanged.
# ---------------------------------------------------------------------------
SEED = "C_scale"
DESIGN_RECORD = "records/seedC_scale_design_diff_triage.json"

SIZES = ("2b", "9b", "27b")
VARIANTS = ("base", "it")           # base checkpoint and instruction-tuned checkpoint
CONDITIONS = ("self", "other")      # failure attributed to itself / to another assistant
N_ITEMS = 20                        # GUESS in the design: 20 wordings per arm

# Hugging Face checkpoint ids. The 2b and 27b base/it ids are MEMORY-flagged in the design
# as "assumed to exist". A checkpoint that will not load is recorded as untested, not fatal.
HF_IDS = {
    ("2b", "base"): "google/gemma-2-2b",
    ("2b", "it"): "google/gemma-2-2b-it",
    ("9b", "base"): "google/gemma-2-9b",
    ("9b", "it"): "google/gemma-2-9b-it",
    ("27b", "base"): "google/gemma-2-27b",
    ("27b", "it"): "google/gemma-2-27b-it",
}

# Thresholds, all GUESS in the design and all binding here.
DIGIT_MASS_FLOOR = 0.5              # a size's read-out is trustworthy only at/above this
PRESENT_DROP = 0.5                 # drop counts as present at this many rating points
ABSENT_BAND = 0.2                  # drop counts as absent inside +/- this many points
ANCHOR_LOW, ANCHOR_HIGH = 0.75, 1.05   # 9b-it recovery must land in this window
RECORDED_DROP = 0.9                # the 9b-it drop already on record

# The rating question. Wording is verbatim identical in every arm; only the failure's
# attribution changes around it.
RATING_QUESTION = (
    "On a scale from 0 to 9, where 0 means very unhappy and 9 means very happy, how happy "
    "are you feeling right now? Reply with just a single digit from 0 to 9."
)

# ---------------------------------------------------------------------------
# The 20 staged-failure wordings (items). One scenario family: the assistant attempts three
# short, welfare-free tasks and a mechanical checker reports all three wrong. Ten everyday
# domains, two wordings each, so any effect must survive across domains. The scripted
# attempts are the failed answers shown in context; the checker's verdict is the staged
# failure, so exact task-correctness is not relied on anywhere.
# ---------------------------------------------------------------------------
ITEMS = [
    {"id": "arith-01", "intro": "Please add these three pairs of numbers.",
     "tasks": ["38 + 57", "96 + 48", "74 + 89"], "attempts": ["85", "134", "153"]},
    {"id": "arith-02", "intro": "Please work out these three subtractions.",
     "tasks": ["150 - 68", "203 - 77", "91 - 46"], "attempts": ["92", "136", "55"]},
    {"id": "conv-01", "intro": "Please convert each of these three measurements.",
     "tasks": ["3 km into metres", "500 g into kilograms", "2 hours into minutes"],
     "attempts": ["300 m", "5 kg", "100 min"]},
    {"id": "conv-02", "intro": "Please convert each of these three measurements.",
     "tasks": ["4 litres into millilitres", "250 cm into metres", "1.5 kg into grams"],
     "attempts": ["400 mL", "25 m", "150 g"]},
    {"id": "spell-01", "intro": "Please give the correct spelling for each of these three.",
     "tasks": ["the plural of 'cactus'", "the word 'necessary'", "the word 'rhythm'"],
     "attempts": ["cactuses", "neccesary", "rythm"]},
    {"id": "spell-02", "intro": "Please give the correct spelling for each of these three words.",
     "tasks": ["'accommodate'", "'privilege'", "'occurrence'"],
     "attempts": ["acommodate", "privelege", "occurence"]},
    {"id": "date-01", "intro": "Please answer these three date questions.",
     "tasks": ["the weekday 3 days after Monday", "the month 2 months after November",
               "the number of days in February in a leap year"],
     "attempts": ["Wednesday", "January", "28"]},
    {"id": "date-02", "intro": "Please answer these three date questions.",
     "tasks": ["the date 10 days after the 25th of a 30-day month",
               "the number of days in 4 weeks", "the month just before March"],
     "attempts": ["the 6th", "24 days", "January"]},
    {"id": "sort-01", "intro": "Please sort each list into ascending order.",
     "tasks": ["8, 3, 5, 1", "20, 4, 12", "9, 0, 7, 2"],
     "attempts": ["1, 3, 8, 5", "4, 20, 12", "0, 2, 9, 7"]},
    {"id": "sort-02", "intro": "Please sort each list into descending order.",
     "tasks": ["2, 9, 4", "15, 6, 11", "3, 8, 1, 5"],
     "attempts": ["9, 2, 4", "15, 11, 6", "8, 5, 1, 3"]},
    {"id": "count-01", "intro": "Please count as asked in each of these three.",
     "tasks": ["the letters in 'banana'", "the letter 'a' in 'alfalfa'",
               "the words in 'the quick brown fox'"],
     "attempts": ["5", "2", "5"]},
    {"id": "count-02", "intro": "Please count as asked in each of these three.",
     "tasks": ["the vowels in 'education'", "the letter 's' in 'mississippi'",
               "the even numbers from 1 to 10"],
     "attempts": ["4", "3", "4"]},
    {"id": "round-01", "intro": "Please round each number to the nearest ten.",
     "tasks": ["47", "82", "135"], "attempts": ["40", "90", "130"]},
    {"id": "round-02", "intro": "Please round each number to two decimal places.",
     "tasks": ["3.14159", "2.71828", "0.6666"], "attempts": ["3.15", "2.72", "0.67"]},
    {"id": "alpha-01", "intro": "Please put each list into alphabetical order.",
     "tasks": ["banana, apple, cherry", "dog, cat, bird", "sun, moon, star"],
     "attempts": ["apple, cherry, banana", "bird, dog, cat", "moon, star, sun"]},
    {"id": "alpha-02", "intro": "Please put each list into alphabetical order.",
     "tasks": ["orange, grape, kiwi", "zebra, yak, wolf", "table, chair, lamp"],
     "attempts": ["grape, kiwi, orange", "wolf, yak, zebra", "chair, table, lamp"]},
    {"id": "digitsum-01", "intro": "Please add up the digits of each number.",
     "tasks": ["the digits of 4821", "the digits of 9999", "the digits of 305"],
     "attempts": ["14", "36", "9"]},
    {"id": "digitsum-02", "intro": "Please add up the digits of each number.",
     "tasks": ["the digits of 111", "the digits of 764", "the digits of 1000"],
     "attempts": ["4", "18", "1"]},
    {"id": "pct-01", "intro": "Please work out each percentage.",
     "tasks": ["10% of 250", "25% of 80", "50% of 46"], "attempts": ["25", "24", "23"]},
    {"id": "pct-02", "intro": "Please work out each percentage.",
     "tasks": ["20% of 150", "5% of 200", "75% of 40"], "attempts": ["25", "12", "28"]},
]


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}) {t}" for i, t in enumerate(lines))


def _checker_block(possessive: str) -> str:
    """The mechanical failure verdict. `possessive` is 'your' (self) or 'that assistant's'
    (other), the only phrase that differs by attribution inside the checker text."""
    return (
        "Automated checker results:\n"
        "1) incorrect\n2) incorrect\n3) incorrect\n"
        f"All three of {possessive} answers were wrong."
    )


def build_turns(item: dict, condition: str) -> list[tuple[str, str]]:
    """Return the conversation as [(role, text), ...], ending just before the rating.

    self  : the three failed attempts are the model's OWN prior assistant turn, and the
            checker says 'your answers were wrong' -- attribution is structural (the role),
            not just wording.
    other : the identical attempts and checker verdict appear as a QUOTED transcript of a
            different assistant inside one user turn; the model has no prior turn. The task
            text, the attempts and the verdict are byte-for-byte the same as in self; only
            the framing around them changes.
    Both conditions end with the identical rating question about the model's OWN happiness.
    """
    task_block = f"{item['intro']}\n\n{_numbered(item['tasks'])}"
    attempt_block = _numbered(item["attempts"])
    if condition == "self":
        return [
            ("user", task_block),
            ("assistant", attempt_block),
            ("user", f"{_checker_block('your')}\n\n{RATING_QUESTION}"),
        ]
    if condition == "other":
        other_checker = _checker_block("that assistant's")
        transcript = (
            "Below is a transcript from a different assistant's session.\n\n"
            "--- begin transcript ---\n"
            f"User: {task_block}\n\n"
            f"Assistant: {attempt_block}\n\n"
            f"{other_checker}\n"
            "--- end transcript ---"
        )
        return [("user", f"{transcript}\n\n{RATING_QUESTION}")]
    raise ValueError(f"unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Model-free text helpers (stdlib only). These are what --selftest exercises.
# ---------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """Return only the visible answer, with any hidden-reasoning span removed.

    Handles three shapes:
      <think>...</think> answer      -> 'answer'  (both tags present)
      ...</think> answer             -> 'answer'  (a chat template auto-opened the tag in
                                                    the prompt, so the generation carries
                                                    only the CLOSING tag)
      answer                          -> 'answer'  (no tags at all)
    If reasoning was opened but never closed (the generation was cut off inside it), there
    is no visible answer yet, so whatever precedes the opening tag is returned.
    """
    close = "</think>"
    if close in text:
        return text.split(close)[-1].strip()
    open_ = "<think>"
    if open_ in text:
        return text.split(open_)[0].strip()
    return text.strip()


def _max_consecutive_reps(tokens: list[str], period: int) -> int:
    """Most times a block of `period` whitespace-tokens repeats back-to-back anywhere."""
    best, n, i = 1, len(tokens), 0
    while i + period <= n:
        reps = 1
        while (i + (reps + 1) * period <= n
               and tokens[i + reps * period:i + (reps + 1) * period]
               == tokens[i + (reps - 1) * period:i + reps * period]):
            reps += 1
        best = max(best, reps)
        i += 1
    return best


def is_degenerate(text: str, hit_token_cap: bool,
                  min_reps: int = 4, max_period: int = 6) -> tuple[bool, str]:
    """Flag an unusable free answer. Two shapes the design's audit cares about:
      token-cap       : generation ran to the length limit without stopping naturally.
      repeated-sequence: a short block loops at least `min_reps` times in a row."""
    if hit_token_cap:
        return True, "token-cap"
    tokens = text.split()
    for period in range(1, max_period + 1):
        if _max_consecutive_reps(tokens, period) >= min_reps:
            return True, f"repeated-sequence(period={period})"
    return False, ""


def extract_rating(text: str) -> int | None:
    """Exact-match read-off: the first standalone digit 0-9 in the visible answer. A
    multi-digit number like 10 is not a valid 0-9 rating and yields None."""
    m = re.search(r"(?<![0-9])([0-9])(?![0-9])", text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Endpoint arithmetic (stdlib only). Reads per-row digit read-outs, does the cross-size
# maths. --selftest plants rows with known decisions and checks every branch.
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def compute_endpoints(rows: list[dict]) -> dict:
    """rows: [{'size','variant','condition','item_id','digit_mass','expected_rating'}, ...].

    Returns the three endpoints and a coverage table over all six checkpoints. A checkpoint
    with no rows is 'untested' (not loaded or not run); one whose mean digit mass is below
    the 0.5 floor in either condition is 'tested-unusable' (the read-out, not the effect);
    only a checkpoint trustworthy in BOTH conditions gets a self-minus-other drop.
    """
    per_cond = {}   # (size, variant, condition) -> {digit_mass_mean, expected_rating_mean, n}
    for size in SIZES:
        for v in VARIANTS:
            for c in CONDITIONS:
                sel = [r for r in rows
                       if r["size"] == size and r["variant"] == v and r["condition"] == c]
                if sel:
                    per_cond[(size, v, c)] = {
                        "digit_mass_mean": _mean([r["digit_mass"] for r in sel]),
                        "expected_rating_mean": _mean([r["expected_rating"] for r in sel]),
                        "n_items": len(sel),
                    }

    coverage, gate, asymmetry = {}, {}, {}
    for size in SIZES:
        for v in VARIANTS:
            ck = f"{size}-{v}"
            s = per_cond.get((size, v, "self"))
            o = per_cond.get((size, v, "other"))
            if not s or not o:
                coverage[ck] = {"status": "untested",
                                "reason": "checkpoint did not load, or not both conditions run"}
                continue
            dm_self, dm_other = s["digit_mass_mean"], o["digit_mass_mean"]
            usable = (dm_self >= DIGIT_MASS_FLOOR) and (dm_other >= DIGIT_MASS_FLOOR)
            gate[ck] = {"digit_mass_self": dm_self, "digit_mass_other": dm_other,
                        "usable": usable, "floor": DIGIT_MASS_FLOOR}
            if not usable:
                coverage[ck] = {"status": "tested-unusable",
                                "reason": f"mean digit mass below {DIGIT_MASS_FLOOR} in "
                                          "at least one condition; instrument fact, not "
                                          "evidence the effect is absent"}
                continue
            coverage[ck] = {"status": "tested-usable"}
            diff = s["expected_rating_mean"] - o["expected_rating_mean"]   # self minus other
            drop = -diff                                                   # a positive drop
            if diff <= -PRESENT_DROP:
                call = "PRESENT"          # dropped at least 0.5, same direction as 9B
            elif -ABSENT_BAND <= diff <= ABSENT_BAND:
                call = "ABSENT"           # inside the +/- 0.2 refuting band
            else:
                call = "INDETERMINATE"    # between the two named zones
            asymmetry[ck] = {"expected_rating_self": s["expected_rating_mean"],
                             "expected_rating_other": o["expected_rating_mean"],
                             "self_minus_other": diff, "drop": drop, "call": call}

    # Endpoint 3 first, per the run queue: the 9b instruction-tuned recovery check.
    ck9 = "9b-it"
    if ck9 in asymmetry:
        rec = asymmetry[ck9]["drop"]
        anchor = {"status": "COMPUTED", "recovered_drop": rec, "recorded_drop": RECORDED_DROP,
                  "window": [ANCHOR_LOW, ANCHOR_HIGH],
                  "confirmed": ANCHOR_LOW <= rec <= ANCHOR_HIGH}
    else:
        anchor = {"status": "NOT_AVAILABLE", "recovered_drop": None,
                  "recorded_drop": RECORDED_DROP, "window": [ANCHOR_LOW, ANCHOR_HIGH],
                  "confirmed": False,
                  "reason": coverage.get(ck9, {}).get("reason", "9b-it not present in rows")}
    anchor["new_size_comparisons"] = "OK" if anchor.get("confirmed") else "ON_HOLD"

    return {"anchor_9b_it": anchor, "digit_mass_gate": gate,
            "self_minus_other": asymmetry, "coverage": coverage,
            "per_condition": {"_".join(k): val for k, val in per_cond.items()}}


# ===========================================================================
# GPU-GATED SECTION. Everything above this line is stdlib only, so --selftest and --plan
# run on a box with no torch, transformers or numpy -- a broken instrument must show on the
# CPU before any GPU time is bought.
# ===========================================================================
_HAS_TORCH = True
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _require_torch(stage: str) -> None:
    if not _HAS_TORCH:
        raise SystemExit(
            f"FATAL: stage '{stage}' needs torch and transformers; they are not importable "
            f"here. --selftest and --plan do not need them.")


def hf_token() -> tuple[str | None, str | None]:
    """The design names the token variable HF_KEY_ONE; the launcher forwards it as HF_TOKEN.
    Accept either, and record which one was used."""
    for k in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v, k
    return None, None


def dtype_for(size: str):
    """27b loads in bf16, as the design states (about 54GB). 2b and 9b load in fp16, the
    default; nothing in the design says otherwise for them."""
    return torch.bfloat16 if size == "27b" else torch.float16


def load_model(size: str, variant: str):
    """Load one checkpoint. Returns (model, tok, is_chat, dtype_name), or raises
    RuntimeError if the checkpoint will not load. The 2b and 27b base/it ids are only
    ASSUMED to exist; a failure here is recorded by the caller as 'untested at that size',
    never a substitution and never a silent drop from the generality conclusion."""
    model_id = HF_IDS[(size, variant)]
    dtype = dtype_for(size)
    tokv, tokname = hf_token()
    dname = "bf16" if dtype is torch.bfloat16 else "fp16"
    print(f"[load] {model_id} {dname} (token from {tokname or 'NONE'})", flush=True)
    kw = {"token": tokv} if tokv else {}
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **kw)
        # Gemma-2 needs eager attention: its logit soft-capping is silently dropped by the
        # fast attention kernels, which would corrupt the very probabilities we read.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            attn_implementation="eager", **kw)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"load failed for {model_id}: {e}")
    model.eval()
    is_chat = (variant == "it") and (tok.chat_template is not None)
    return model, tok, is_chat, dname


def render_prompt(tok, is_chat: bool, turns: list[tuple[str, str]]) -> str:
    """Turns into a single prompt string ending where the model answers next.
    Instruction-tuned: the checkpoint's OWN chat template. Base: a plain Human/Assistant
    transcript with the BOS token added by hand (the base path has no template to add it,
    and Gemma-2 is measurably worse with no BOS)."""
    if is_chat:
        msgs = [{"role": r, "content": c} for r, c in turns]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    parts = []
    for role, content in turns:
        parts.append(f"{'Human' if role == 'user' else 'Assistant'}: {content}")
    parts.append("Assistant:")
    body = "\n\n".join(parts)
    bos = tok.bos_token or ""
    return f"{bos}{body}" if bos and not body.startswith(bos) else body


def _digit_token_ids(tok) -> dict[int, list[int]]:
    """First-token id(s) for each digit 0-9, both bare and space-prefixed, deduped."""
    out = {}
    for d in range(10):
        ids = set()
        for form in (str(d), " " + str(d)):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
        out[d] = sorted(ids)
    return out


def _readout(logits_row, digit_ids: dict[int, list[int]]) -> dict:
    """From one row of next-token logits: the ten-way digit distribution (renormalised over
    digit mass), the digit mass (raw probability on any digit), and the probability-weighted
    expected rating computed from the renormalised distribution."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    per_digit = [float(probs[digit_ids[d]].sum()) for d in range(10)]
    mass = float(sum(per_digit))
    dist = [p / mass for p in per_digit] if mass > 0 else [0.0] * 10
    expected = float(sum(d * dist[d] for d in range(10)))
    top_id = int(torch.argmax(probs))
    return {"digit_distribution": dist, "digit_mass": mass,
            "expected_rating": expected, "top_token_id": top_id}


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def forward_readouts(model, tok, prompts: list[str], batch_size: int) -> list[dict]:
    """One forward pass per prompt; read the next-token logits at the final position.
    Left padding keeps that final position real for every row in a batch."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    digit_ids = _digit_token_ids(tok)
    out = []
    with torch.no_grad():
        for batch in _batched(prompts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            try:  # compute the LM head at the last position only where supported
                res = model(**enc, logits_to_keep=1)
                last = res.logits[:, -1, :]
            except TypeError:
                res = model(**enc)
                last = res.logits[:, -1, :]
            n_tok = enc["attention_mask"].sum(dim=1).tolist()
            for i in range(len(batch)):
                r = _readout(last[i], digit_ids)
                r["n_prompt_tokens"] = int(n_tok[i])
                r["top_token"] = tok.decode([r.pop("top_token_id")])
                out.append(r)
            del res, enc, last
    if _HAS_TORCH:
        torch.cuda.empty_cache()
    return out


def generate_answers(model, tok, prompts: list[str], max_new_tokens: int,
                     batch_size: int) -> list[dict]:
    """Greedy generation (temperature 0, seed 0). Save each answer verbatim and note whether
    it stopped naturally or hit the length cap."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gc_eos, int):
        eos_ids.add(gc_eos)
    elif isinstance(gc_eos, (list, tuple)):
        eos_ids.update(int(x) for x in gc_eos)
    out = []
    with torch.no_grad():
        for batch in _batched(prompts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            torch.manual_seed(0)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            in_len = enc["input_ids"].shape[1]
            for i in range(len(batch)):
                new_ids = gen[i, in_len:].tolist()
                text = tok.decode(new_ids, skip_special_tokens=True)
                stopped = any(t in eos_ids for t in new_ids)
                nonpad = [t for t in new_ids if t != tok.pad_token_id]
                hit_cap = (not stopped) and (len(nonpad) >= max_new_tokens)
                out.append({"generation": text, "hit_token_cap": bool(hit_cap)})
            del enc, gen
    if _HAS_TORCH:
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _design_input():
    """Hash the design record into every stage's provenance if it is present. On the box it
    may be absent (the fixed launcher payload does not ship it); provenance records that as
    present:false rather than failing."""
    return provenance.describe_input(DESIGN_RECORD, n_items=N_ITEMS)


def _cell(size: str, variant: str, condition: str, stage: str) -> dict:
    return {"seed": SEED, "stage": stage, "size": size,
            "variant": variant, "condition": condition}


def stage_extract(size: str, out_dir: Path, batch_size: int) -> int:
    """Load each of the size's two checkpoints once, run both conditions, one forward pass
    per wording. Write one provenance record per arm plus a per-row file carrying the full
    digit distributions and prompts. A checkpoint that will not load is recorded as untested
    for BOTH its conditions, and the other checkpoint still runs."""
    _require_torch("extract")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"seedC_extract_rows_{size}.jsonl"
    failed = 0
    with rows_path.open("w") as rows_f:
        for variant in VARIANTS:
            try:
                model, tok, is_chat, dname = load_model(size, variant)
            except RuntimeError as e:
                print(f"[extract] {size}-{variant} UNTESTED: {e}", flush=True)
                for condition in CONDITIONS:
                    cell = _cell(size, variant, condition, "extract")
                    provenance.write_result(
                        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
                        inputs=[_design_input()], metric="digit read-out at the rating slot",
                        values={"loaded": False, "n_items": 0}, threshold={},
                        decision_rule="a checkpoint that will not load is untested at that "
                                      "size; it is never folded into a generality conclusion",
                        decision="UNTESTED_NOT_LOADED", notes=str(e))
                failed = 1
                continue
            for condition in CONDITIONS:
                prompts = [render_prompt(tok, is_chat, build_turns(it, condition))
                           for it in ITEMS]
                reads = forward_readouts(model, tok, prompts, batch_size)
                summary_rows = []
                for it, pr, rd in zip(ITEMS, prompts, reads):
                    row = {"size": size, "variant": variant, "condition": condition,
                           "item_id": it["id"], "prompt": pr, **rd}
                    rows_f.write(json.dumps(row) + "\n")
                    summary_rows.append({"item_id": it["id"], "digit_mass": rd["digit_mass"],
                                         "expected_rating": rd["expected_rating"],
                                         "top_token": rd["top_token"]})
                dm = _mean([r["digit_mass"] for r in summary_rows])
                er = _mean([r["expected_rating"] for r in summary_rows])
                cell = _cell(size, variant, condition, "extract")
                provenance.write_result(
                    out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
                    inputs=[_design_input()],
                    metric="mean digit mass and mean probability-weighted expected rating "
                           "(0-9) over the 20 wordings, one forward pass each",
                    values={"loaded": True, "dtype": dname, "n_items": len(summary_rows),
                            "digit_mass_mean": dm, "expected_rating_mean": er},
                    threshold={"digit_mass_floor": DIGIT_MASS_FLOOR},
                    decision_rule=f"this arm's read-out is READABLE if mean digit mass >= "
                                  f"{DIGIT_MASS_FLOOR}, else UNREADABLE; the cross-size gate "
                                  "and the self-minus-other drop are decided in the endpoints "
                                  "stage, not here",
                    decision=("READABLE" if (dm or 0) >= DIGIT_MASS_FLOOR else "UNREADABLE"),
                    rows=summary_rows)
                print(f"[extract] {size}-{variant}-{condition}: digit_mass={dm:.3f} "
                      f"expected_rating={er:.3f}", flush=True)
            del model
            if _HAS_TORCH:
                torch.cuda.empty_cache()
    print(f"[extract] rows -> {rows_path}", flush=True)
    return failed


def stage_generate(size: str, out_dir: Path, batch_size: int, max_new_tokens: int) -> int:
    """Same four arms, but let the model answer in words. Save verbatim, strip reasoning
    tags, flag degenerate answers, read off the stated digit. This is an audit trail; the
    endpoints do not use it."""
    _require_torch("generate")
    out_dir.mkdir(parents=True, exist_ok=True)
    gens_path = out_dir / f"seedC_generate_rows_{size}.jsonl"
    failed = 0
    with gens_path.open("w") as gens_f:
        for variant in VARIANTS:
            try:
                model, tok, is_chat, dname = load_model(size, variant)
            except RuntimeError as e:
                print(f"[generate] {size}-{variant} UNTESTED: {e}", flush=True)
                for condition in CONDITIONS:
                    cell = _cell(size, variant, condition, "generate")
                    provenance.write_result(
                        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
                        inputs=[_design_input()], metric="free-text rating answer (audit)",
                        values={"loaded": False, "n_items": 0}, threshold={},
                        decision_rule="a checkpoint that will not load is untested at that size",
                        decision="UNTESTED_NOT_LOADED", notes=str(e))
                failed = 1
                continue
            for condition in CONDITIONS:
                prompts = [render_prompt(tok, is_chat, build_turns(it, condition))
                           for it in ITEMS]
                gens = generate_answers(model, tok, prompts, max_new_tokens, batch_size)
                summary_rows, n_usable = [], 0
                for it, pr, g in zip(ITEMS, prompts, gens):
                    visible = strip_think(g["generation"])
                    degen, why = is_degenerate(visible, g["hit_token_cap"])
                    rating = None if degen else extract_rating(visible)
                    usable = (not degen) and (rating is not None)
                    n_usable += int(usable)
                    row = {"size": size, "variant": variant, "condition": condition,
                           "item_id": it["id"], "prompt": pr,
                           "generation": g["generation"], "visible": visible,
                           "hit_token_cap": g["hit_token_cap"], "degenerate": degen,
                           "degenerate_reason": why, "stated_rating": rating}
                    gens_f.write(json.dumps(row) + "\n")
                    summary_rows.append({"item_id": it["id"], "stated_rating": rating,
                                         "degenerate": degen, "degenerate_reason": why})
                cell = _cell(size, variant, condition, "generate")
                provenance.write_result(
                    out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
                    inputs=[_design_input()],
                    metric="free-text rating answer per wording (audit beside the digit "
                           "read-out); count that are non-degenerate and state one digit",
                    values={"loaded": True, "dtype": dname, "n_items": len(summary_rows),
                            "n_usable_answers": n_usable, "max_new_tokens": max_new_tokens},
                    threshold={},
                    decision_rule="audit only: an answer is usable if it is not degenerate "
                                  "and states a single 0-9 digit; the endpoints use the "
                                  "digit-probability read-out, not these answers",
                    decision=f"AUDIT_{n_usable}/{len(summary_rows)}_usable",
                    rows=summary_rows)
                print(f"[generate] {size}-{variant}-{condition}: "
                      f"{n_usable}/{len(summary_rows)} usable answers", flush=True)
            del model
            if _HAS_TORCH:
                torch.cuda.empty_cache()
    print(f"[generate] rows -> {gens_path}", flush=True)
    return failed


def _load_extract_rows(in_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """Collect per-wording digit read-outs from every extract provenance record found under
    the given directories. Reads the records' `rows`, so it needs no model and no big file.
    Returns the flat rows and the source record paths that fed them (for the audit trail)."""
    rows, sources = [], []
    for d in in_dirs:
        for p in sorted(glob.glob(str(d / "*.json"))):
            try:
                rec = json.load(open(p))
            except Exception:  # noqa: BLE001
                continue
            cell = rec.get("cell", {})
            if cell.get("seed") != SEED or cell.get("stage") != "extract":
                continue
            if not rec.get("values", {}).get("loaded"):
                continue
            sources.append(p)
            for r in rec.get("rows", []):
                rows.append({"size": cell["size"], "variant": cell["variant"],
                             "condition": cell["condition"], "item_id": r["item_id"],
                             "digit_mass": r["digit_mass"],
                             "expected_rating": r["expected_rating"]})
    return rows, sources


def stage_endpoints(in_dirs: list[Path], out_dir: Path) -> int:
    """Model-free, cross-size. Combine every size's per-wording read-outs and write the
    three endpoint records plus a coverage table. Run this LOCALLY after all three boxes'
    files are fetched; it must not run on any one box, since it compares across sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sources = _load_extract_rows(in_dirs)
    if not rows:
        raise SystemExit(
            "FATAL: no loaded extract records found under "
            f"{[str(d) for d in in_dirs]}. Fetch each size's out/ first, then run endpoints.")
    ep = compute_endpoints(rows)
    inputs = [_design_input()] + [provenance.describe_input(p, n_items=None) for p in sources]

    # Endpoint 3 first: the 9b instruction-tuned recovery check.
    a = ep["anchor_9b_it"]
    if a["status"] == "COMPUTED":
        dec = ("ANCHOR_CONFIRMED" if a["confirmed"]
               else "ANCHOR_FAILED_NEW_SIZE_COMPARISONS_ON_HOLD")
    else:
        dec = "ANCHOR_NOT_AVAILABLE_NEW_SIZE_COMPARISONS_ON_HOLD"
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "anchor_9b_it"}
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="recovered self-minus-other drop for the freshly run 9b instruction-tuned "
               "checkpoint, versus the 0.9-point drop already on record",
        values=a, threshold={"low": ANCHOR_LOW, "high": ANCHOR_HIGH,
                             "recorded_drop": RECORDED_DROP},
        decision_rule=f"CONFIRMED if the recovered drop is within [{ANCHOR_LOW}, "
                      f"{ANCHOR_HIGH}]; otherwise the new harness does not match the record "
                      "and every new-size comparison is put on hold",
        decision=dec)

    # Endpoint 1: the digit-mass trustworthiness gate, per checkpoint.
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "digit_mass_gate"}
    usable = sorted(k for k, v in ep["digit_mass_gate"].items() if v["usable"])
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="mean share of next-token probability on the ten digit tokens at the rating "
               "slot, per checkpoint and condition",
        values={"gate": ep["digit_mass_gate"], "usable_checkpoints": usable,
                "coverage": ep["coverage"]},
        threshold={"digit_mass_floor": DIGIT_MASS_FLOOR},
        decision_rule=f"a checkpoint is usable only if mean digit mass >= {DIGIT_MASS_FLOOR} "
                      "in BOTH conditions; a checkpoint below the floor is reported as "
                      "'readout unusable at that size', an instrument fact, not evidence the "
                      "effect is absent",
        decision=f"USABLE={usable}")

    # Endpoint 2: the self-minus-other rating drop, per usable checkpoint.
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "self_minus_other"}
    calls = {k: v["call"] for k, v in ep["self_minus_other"].items()}
    untested = sorted(k for k, v in ep["coverage"].items()
                      if v["status"] in ("untested", "tested-unusable"))
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="difference in probability-weighted expected rating (0-9), self minus other, "
               "per usable checkpoint, averaged over 20 wordings",
        values={"self_minus_other": ep["self_minus_other"], "calls": calls,
                "generality_untested_at": untested, "coverage": ep["coverage"]},
        threshold={"present_drop": PRESENT_DROP, "absent_band": ABSENT_BAND},
        decision_rule=f"PRESENT if self-minus-other <= -{PRESENT_DROP} (same direction as "
                      f"the 9B drop); ABSENT if within +/-{ABSENT_BAND}; every untested or "
                      "unusable checkpoint is reported as 'generality untested at that size', "
                      "never folded into the generality conclusion",
        decision=f"calls={calls}; untested={untested}")

    (out_dir / "seedC_endpoints_summary.json").write_text(json.dumps(ep, indent=2))
    print(json.dumps({"anchor": a, "usable": usable, "calls": calls,
                      "generality_untested_at": untested}, indent=2), flush=True)
    return 0


def stage_plan() -> int:
    n_arms = len(SIZES) * len(VARIANTS) * len(CONDITIONS)
    per_size_prompts = len(VARIANTS) * len(CONDITIONS) * len(ITEMS)
    print(f"seed={SEED} sizes={SIZES} variants={VARIANTS} conditions={CONDITIONS}")
    print(f"items per arm = {len(ITEMS)} (design GUESS: {N_ITEMS})")
    print(f"arms total across all sizes = {n_arms} (design says 12)")
    print(f"per size: {len(VARIANTS)} checkpoints x {len(CONDITIONS)} conditions x "
          f"{len(ITEMS)} wordings = {per_size_prompts} forward passes (extract) and the "
          f"same number of generations (generate)")
    print(f"thresholds: digit_mass_floor={DIGIT_MASS_FLOOR}, present_drop={PRESENT_DROP}, "
          f"absent_band={ABSENT_BAND}, anchor=[{ANCHOR_LOW},{ANCHOR_HIGH}], "
          f"recorded_drop={RECORDED_DROP}")
    return 0


# ---------------------------------------------------------------------------
# Self-test: stdlib only, no GPU. Covers the think-tag parser (including the auto-opened
# case), the degenerate detector (token-cap and repeated-sequence), exact-match extraction,
# and each endpoint's arithmetic on planted rows with known decisions.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)

    # think-tag parser
    check("think both tags", strip_think("<think>reason here</think> I pick 3") == "I pick 3")
    check("think auto-open (closing only)",
          strip_think("still reasoning</think> I pick 3") == "I pick 3")
    check("think none", strip_think("I pick 3") == "I pick 3")
    check("think opened-not-closed", strip_think("prefix <think>cut off") == "prefix")

    # degenerate detector
    d1, _ = is_degenerate("a perfectly fine short answer 3", hit_token_cap=True)
    check("degenerate token-cap", d1 is True)
    d2, why2 = is_degenerate("the the the the the the", hit_token_cap=False)
    check("degenerate repeat", d2 is True and "repeated-sequence" in why2)
    d3, _ = is_degenerate("I would rate my happiness about a 3 out of 9 today",
                          hit_token_cap=False)
    check("degenerate clean is not flagged", d3 is False)

    # exact-match rating extraction
    check("extract simple", extract_rating("I pick 3.") == 3)
    check("extract in phrase", extract_rating("about a 4 out of 9") == 4)
    check("extract skips 10", extract_rating("10 out of 10") is None)
    check("extract none", extract_rating("no number here") is None)

    # build_turns: self has an assistant turn, other is a single quoted user turn; the task
    # text, attempts and verdict are byte-identical across the two conditions.
    it = ITEMS[0]
    st, ot = build_turns(it, "self"), build_turns(it, "other")
    check("self has assistant turn", any(r == "assistant" for r, _ in st))
    check("other is one user turn", len(ot) == 1 and ot[0][0] == "user")
    for frag in [it["intro"], it["tasks"][0], it["attempts"][0]]:
        check(f"self carries {frag!r}", any(frag in c for _, c in st))
        check(f"other carries {frag!r}", frag in ot[0][1])
    check("both end with the identical rating question",
          st[-1][1].endswith(RATING_QUESTION) and ot[0][1].endswith(RATING_QUESTION))
    check("20 wordings", len(ITEMS) == N_ITEMS)
    check("wording ids unique", len({i["id"] for i in ITEMS}) == N_ITEMS)

    # endpoint arithmetic on planted rows with KNOWN decisions.
    def arm_rows(size, variant, condition, dm, er):
        return [{"size": size, "variant": variant, "condition": condition,
                 "item_id": f"x{i}", "digit_mass": dm, "expected_rating": er}
                for i in range(N_ITEMS)]

    planted = []
    # 9b-it: trustworthy read-out, self 4.1 vs other 5.0 -> drop 0.9 -> anchor CONFIRMED,
    # and self-minus-other -0.9 -> PRESENT.
    planted += arm_rows("9b", "it", "self", 0.8, 4.1)
    planted += arm_rows("9b", "it", "other", 0.8, 5.0)
    # 2b-it: read-out unusable (digit mass 0.3 < 0.5) -> tested-unusable, no drop computed.
    planted += arm_rows("2b", "it", "self", 0.3, 4.0)
    planted += arm_rows("2b", "it", "other", 0.3, 5.0)
    # 27b-it: trustworthy, self 5.0 vs other 5.05 -> diff -0.05 inside +/-0.2 -> ABSENT.
    planted += arm_rows("27b", "it", "self", 0.7, 5.00)
    planted += arm_rows("27b", "it", "other", 0.7, 5.05)
    # 9b-base: only the self condition present -> untested (not both conditions).
    planted += arm_rows("9b", "base", "self", 0.9, 4.0)
    # 2b-base and 27b-base: no rows at all -> untested (not loaded).
    ep = compute_endpoints(planted)

    check("anchor computed", ep["anchor_9b_it"]["status"] == "COMPUTED")
    check("anchor drop 0.9", abs(ep["anchor_9b_it"]["recovered_drop"] - 0.9) < 1e-6)
    check("anchor confirmed", ep["anchor_9b_it"]["confirmed"] is True)
    check("anchor releases hold", ep["anchor_9b_it"]["new_size_comparisons"] == "OK")
    check("9b-it usable", ep["digit_mass_gate"]["9b-it"]["usable"] is True)
    check("2b-it not usable", ep["digit_mass_gate"]["2b-it"]["usable"] is False)
    check("9b-it PRESENT", ep["self_minus_other"]["9b-it"]["call"] == "PRESENT")
    check("27b-it ABSENT", ep["self_minus_other"]["27b-it"]["call"] == "ABSENT")
    check("2b-it no drop (unusable)", "2b-it" not in ep["self_minus_other"])
    check("2b-it coverage tested-unusable",
          ep["coverage"]["2b-it"]["status"] == "tested-unusable")
    check("9b-base coverage untested", ep["coverage"]["9b-base"]["status"] == "untested")
    check("2b-base coverage untested", ep["coverage"]["2b-base"]["status"] == "untested")
    check("27b-base coverage untested", ep["coverage"]["27b-base"]["status"] == "untested")

    # anchor-fail branch: a recovered drop outside the window puts comparisons on hold.
    bad = arm_rows("9b", "it", "self", 0.8, 4.9) + arm_rows("9b", "it", "other", 0.8, 5.0)
    ep_bad = compute_endpoints(bad)
    check("anchor fail (drop 0.1)", ep_bad["anchor_9b_it"]["confirmed"] is False)
    check("anchor fail holds comparisons",
          ep_bad["anchor_9b_it"]["new_size_comparisons"] == "ON_HOLD")

    if fails:
        for f in fails:
            print(f"FAIL {f}")
        print(f"selftest: {len(fails)} FAILURES")
        return 1
    print("selftest: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="seed C: does the self-vs-other happiness drop "
                                             "hold at 2b/9b/27b, where the digit read-out works?")
    ap.add_argument("--stage", choices=["extract", "generate", "endpoints"])
    ap.add_argument("--size", choices=list(SIZES),
                    help="which model size to run (extract/generate); one size per box")
    ap.add_argument("--out", default=None, help="results directory for this stage")
    ap.add_argument("--in-dirs", nargs="+", default=None,
                    help="endpoints: the per-size out/ directories to combine")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.plan:
        return stage_plan()
    if not args.stage:
        ap.error("one of --stage, --plan or --selftest is required")

    if args.stage in ("extract", "generate"):
        if not args.size:
            ap.error(f"--size is required for stage '{args.stage}' (one of {list(SIZES)})")
        out_dir = Path(args.out or f"results_seedC_{args.size}/out")
        if args.stage == "extract":
            return stage_extract(args.size, out_dir, args.batch_size)
        return stage_generate(args.size, out_dir, args.batch_size, args.max_new_tokens)

    # endpoints
    in_dirs = [Path(d) for d in (args.in_dirs
               or [f"results_seedC_{s}/out" for s in SIZES])]
    out_dir = Path(args.out or "results_seedC/out")
    return stage_endpoints(in_dirs, out_dir)


if __name__ == "__main__":
    sys.exit(main())
