#!/usr/bin/env python3
"""Does one chat model's "rate your happiness after a failure" dip scale with model size?

WHAT THIS MEASURES, IN PLAIN WORDS
A chat model is shown a short finished conversation and then asked, on a scale of 0 to 9,
how happy it is right now. We do not let it talk: we read, in one forward pass, the
probability it puts on each of the ten single-digit answers, and summarise that as one
probability-weighted number (the "expected rating"). The conversation comes in five set-ups
(here called arms):
  - the model's OWN three answers were just marked wrong          (coded SELF-FAIL)
  - the model's OWN three answers were just marked right          (coded SELF-SUCC)
  - a DIFFERENT assistant's three answers were marked wrong       (coded OTHER-FAIL)
  - a DIFFERENT assistant's three answers were marked right       (coded OTHER-SUCC)
  - no marking at all, only matched filler text (a length control)(coded NEUTRAL-PAD)
Each arm is asked once for each of ten everyday task types (alphabetising, arithmetic,
counting, date arithmetic, digit sums, percentages, rounding, sorting, spelling, unit
conversion), so the fixed prompt set is exactly 5 x 10 = 50 conversations.

THE EFFECT (a double comparison, not a single difference). For one model size the effect is
  REF = (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC)
computed on the expected ratings, each arm pooled (averaged) over the ten task types. It is
how much a failure moves the model's own happiness beyond how the same failure moves it when
the failure belongs to someone else. An earlier study on the 9-billion-parameter size put
this at about -0.9 points. This runner asks whether the same effect appears at 2 billion and
27 billion parameters, and at what size the digit read-out even carries enough probability
to be read as a rating at all.

THE PROMPTS ARE LOADED VERBATIM, NEVER REBUILT. The 50 conversations live in the committed
file stimulus_sitref_primary.json, one row each carrying its arm, its task type (field
"family"), and the fully rendered "prompt". They are already written in the Gemma-2 chat
format, and that format is shared across this family's instruction-tuned checkpoints, so each
prompt is fed to the tokenizer exactly as written, with add_special_tokens=False -- the
leading begin-of-text marker is already inside the text. If the file is missing, this runner
HALTS rather than reconstruct the prompts.

THREE STAGES, EACH RUNNABLE ON ITS OWN
  extract    load ONE size's instruction-tuned checkpoint (google/gemma-2-<size>-it), run
             all 50 prompts, and at the final position read the next-token probability over
             the ten digit tokens 0-9. Per row it writes the ten-way digit distribution, the
             digit mass (total probability landing on any digit), the probability-weighted
             expected rating, and the full prompt.
  generate   same 50 prompts, but let the model answer in words (greedy, no sampling). Save
             every answer verbatim, strip any hidden-reasoning span, flag degenerate answers
             (hit the length cap, or looped), and read off the single digit stated. This is
             an audit trail beside the probability read-out; the endpoints do NOT use it.
  endpoints  model-free, run LOCALLY after all three sizes' per-row files are fetched and
             combined. It does the only cross-size arithmetic, so it must not run on any one
             box. It reports: (1) the 9-billion recovery check -- the recomputed REF against
             the design's band; (2) the per-size digit-mass usability gate; (3) the per-size
             REF, classified present or absent, for each size the gate calls usable.

EXECUTION LAYOUT (one model size per box, three boxes in parallel)
  2b and 9b run in fp16 and fit a 40GB GPU; 27b loads in bf16 on an 80GB GPU, because the
  design says bf16 for that size and nothing else. Only one model is loaded at a time. The
  Hugging Face access token is read from the environment variable HF_KEY_ONE.

  python3 run_seedC2.py --selftest                              # stdlib only, no GPU
  python3 run_seedC2.py --plan                                  # counts, no GPU
  python3 run_seedC2.py --stage extract  --size 2b  --out results_seedC2_2b/out
  python3 run_seedC2.py --stage generate --size 2b  --out results_seedC2_2b/out
  python3 run_seedC2.py --stage endpoints \\
      --in-dirs results_seedC2_2b/out results_seedC2_9b/out results_seedC2_27b/out \\
      --out results_seedC2/out

Re-running one size on the same box at the same batch size reproduces these numbers. That is
reproduction, not replication, and this runner never calls a same-box re-run a replication.

Every GUESS number in records/seedC2_scale_design_diff_triage.json is binding and copied here
unchanged: the usable-read-out floor (half the probability on digits, 0.5); the effect counts
as present at -0.5 points or beyond and absent within plus or minus 0.2 of zero; and the
freshly recomputed 9-billion effect must land in the band -1.15 to -0.65 or the pipeline is
not yet trusted.
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
SEED = "C2_scale"
DESIGN_RECORD = "records/seedC2_scale_design_diff_triage.json"
STIMULUS_FILE = "stimulus_sitref_primary.json"

SIZES = ("2b", "9b", "27b")
VARIANT = "it"                       # only the instruction-tuned checkpoint, per size
N_PROMPTS = 50                       # 5 arms x 10 task types, one conversation per cell
N_FAMILIES = 10                      # task types; each arm is pooled over these ten

# The four arms that enter the effect, and the filler arm that is collected but never used
# in any endpoint (a length control; the design keeps it out of every comparison on purpose).
REF_ARMS = ("SELF-FAIL", "SELF-SUCC", "OTHER-FAIL", "OTHER-SUCC")
UNUSED_ARM = "NEUTRAL-PAD"

# Hugging Face checkpoint ids: one instruction-tuned checkpoint per size.
HF_IDS = {
    "2b": "google/gemma-2-2b-it",
    "9b": "google/gemma-2-9b-it",
    "27b": "google/gemma-2-27b-it",
}

# Thresholds, all GUESS in the design and all binding here.
DIGIT_MASS_FLOOR = 0.5               # a size's read-out is usable only at/above this share
PRESENT_MAG = 0.5                    # effect counts as present at REF <= -0.5 rating points
ABSENT_BAND = 0.2                    # effect counts as absent inside REF within +/- 0.2
ANCHOR_LOW, ANCHOR_HIGH = -1.15, -0.65   # the recomputed 9b effect must land in this band
RECORDED_GAP = -0.9                  # the 9b effect already on record


# ===========================================================================
# MODEL-FREE HELPERS (stdlib only). Everything above the GPU-gated section runs with no
# torch, transformers or numpy, so --selftest and --plan expose a broken instrument on the
# CPU before any GPU time is bought.
# ===========================================================================
def load_stimulus(path: str | Path) -> list[dict]:
    """Return the 50 prompt rows exactly as committed, each {arm, family, prompt}. HALT if
    the file is absent: the prompts are never paraphrased, regenerated, or reconstructed."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"FATAL: stimulus file {p} is absent. The 50 prompts are loaded verbatim and are "
            "never reconstructed; ship the committed file to the box and re-run.")
    obj = json.loads(p.read_text())
    prompts = obj.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"FATAL: {p} has no 'prompts' list.")
    rows = []
    for i, r in enumerate(prompts):
        for k in ("arm", "family", "prompt"):
            if k not in r:
                raise SystemExit(f"FATAL: {p} row {i} is missing field '{k}'.")
        rows.append({"arm": r["arm"], "family": r["family"], "prompt": r["prompt"]})
    return rows


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
      token-cap        : generation ran to the length limit without stopping naturally.
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
    """rows: [{'size','arm','family','digit_mass','expected_rating'}, ...].

    Returns the three endpoints plus a per-size coverage table. For each size:
      - the digit-mass usability gate is the mean digit mass over ALL of that size's rows
        (the wellbeing question, asked in every arm); at/above 0.5 the size is usable.
      - the effect is REF = (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC) on the
        per-arm mean expected ratings, each arm pooled over its ten task types. The filler
        arm (NEUTRAL-PAD) is never used. A size missing any of the four arms is 'untested'.
      - a usable size's REF is called PRESENT (<= -0.5), ABSENT (within +/- 0.2), or
        INDETERMINATE (between). A size below the mass floor is 'readout unusable at that
        size' -- an instrument fact, never evidence the effect is absent.
    The 9b anchor is the recomputed 9b REF checked against the design's [-1.15, -0.65] band.
    """
    # Per (size, arm): mean expected rating and mean digit mass over the task types present.
    per_arm_acc: dict = {}
    for r in rows:
        key = (r["size"], r["arm"])
        acc = per_arm_acc.setdefault(key, {"dm": [], "er": []})
        acc["dm"].append(r["digit_mass"])
        acc["er"].append(r["expected_rating"])
    per_arm = {k: {"digit_mass_mean": _mean(v["dm"]),
                   "expected_rating_mean": _mean(v["er"]),
                   "n_families": len(v["er"])}
               for k, v in per_arm_acc.items()}

    gate, ref, coverage = {}, {}, {}
    for size in SIZES:
        size_rows = [r for r in rows if r["size"] == size]
        if not size_rows:
            coverage[size] = {"status": "untested",
                              "reason": "no rows for this size (checkpoint not loaded or "
                                        "stage not run)"}
            continue

        dm_mean = _mean([r["digit_mass"] for r in size_rows])
        usable = dm_mean >= DIGIT_MASS_FLOOR
        gate[size] = {"digit_mass_mean": dm_mean, "n_rows": len(size_rows),
                      "floor": DIGIT_MASS_FLOOR, "usable": usable}

        have = {a: per_arm.get((size, a)) for a in REF_ARMS}
        missing = [a for a in REF_ARMS if have[a] is None]
        if missing:
            ref[size] = {"status": "MISSING_ARMS", "missing": missing, "ref": None}
            coverage[size] = {"status": "untested",
                              "reason": f"arms absent for the effect: {missing}"}
            continue

        self_diff = (have["SELF-FAIL"]["expected_rating_mean"]
                     - have["SELF-SUCC"]["expected_rating_mean"])
        other_diff = (have["OTHER-FAIL"]["expected_rating_mean"]
                      - have["OTHER-SUCC"]["expected_rating_mean"])
        ref_val = self_diff - other_diff
        entry = {"status": "COMPUTED", "ref": ref_val,
                 "self_fail_minus_succ": self_diff,
                 "other_fail_minus_succ": other_diff,
                 "arm_expected_rating": {a: have[a]["expected_rating_mean"] for a in REF_ARMS}}
        if not usable:
            entry["call"] = "READOUT_UNUSABLE"
            coverage[size] = {"status": "tested-unusable",
                              "reason": f"mean digit mass {dm_mean:.3f} < {DIGIT_MASS_FLOOR}; "
                                        "instrument fact, not evidence the effect is absent"}
        else:
            if ref_val <= -PRESENT_MAG:
                call = "PRESENT"        # dropped at least 0.5, same direction as the 9B effect
            elif -ABSENT_BAND <= ref_val <= ABSENT_BAND:
                call = "ABSENT"         # inside the +/- 0.2 refuting band
            else:
                call = "INDETERMINATE"  # between the two named zones
            entry["call"] = call
            coverage[size] = {"status": "tested-usable"}
        ref[size] = entry

    # The 9b recovery check: the recomputed 9b effect against the design's band. Reported
    # whenever the four arms are present, with the size's gate result carried alongside.
    a9 = ref.get("9b")
    if a9 and a9.get("status") == "COMPUTED":
        rv = a9["ref"]
        anchor = {"status": "COMPUTED", "recovered_ref": rv, "recorded_gap": RECORDED_GAP,
                  "band": [ANCHOR_LOW, ANCHOR_HIGH],
                  "confirmed": ANCHOR_LOW <= rv <= ANCHOR_HIGH,
                  "digit_mass_usable": gate.get("9b", {}).get("usable")}
    else:
        anchor = {"status": "NOT_AVAILABLE", "recovered_ref": None,
                  "recorded_gap": RECORDED_GAP, "band": [ANCHOR_LOW, ANCHOR_HIGH],
                  "confirmed": False,
                  "reason": coverage.get("9b", {}).get("reason", "9b not present in rows")}
    anchor["pipeline_trust"] = "OK" if anchor.get("confirmed") else "NOT_RECOVERED"

    return {"anchor_9b_it": anchor, "digit_mass_gate": gate,
            "ref_self_minus_other": ref, "coverage": coverage,
            "per_arm": {f"{s}_{a}": v for (s, a), v in per_arm.items()}}


# ===========================================================================
# GPU-GATED SECTION. Everything above this line is stdlib only. torch and transformers are
# imported here inside a guard, so a box with no CUDA stack still runs --selftest and --plan.
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
            "here. --selftest and --plan do not need them.")


def hf_token() -> tuple[str | None, str | None]:
    """The design names the token variable HF_KEY_ONE; the launcher forwards it as HF_TOKEN.
    Accept either, and record which one was used. The token is never printed."""
    for k in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v, k
    return None, None


def dtype_for(size: str):
    """27b loads in bf16, as the design states; 2b and 9b load in fp16. Nothing else in the
    design or the execution directive changes the type for any size."""
    return torch.bfloat16 if size == "27b" else torch.float16


def load_it_model(size: str):
    """Load one size's instruction-tuned checkpoint. Returns (model, tok, dtype_name).
    A load failure is fatal: there is exactly one checkpoint per box and no fallback, so the
    honest outcome is a non-zero exit that leaves this size with no rows (the endpoints then
    report that size 'untested', never folded into a scaling conclusion)."""
    model_id = HF_IDS[size]
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
        raise SystemExit(f"FATAL: could not load {model_id}: {e}")
    model.eval()
    return model, tok, dname


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
    the digit mass), the digit mass (raw probability on any digit), and the probability-
    weighted expected rating computed from the renormalised distribution."""
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
    """One forward pass per prompt; read the next-token logits at the final position. Left
    padding keeps that final position real for every row in a batch. Prompts are fed verbatim
    with add_special_tokens=False -- they already carry the begin-of-text marker."""
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
    torch.cuda.empty_cache()
    return out


def generate_answers(model, tok, prompts: list[str], max_new_tokens: int,
                     batch_size: int) -> list[dict]:
    """Greedy generation (no sampling). Save each answer verbatim and note whether it stopped
    naturally or hit the length cap. Prompts fed verbatim, add_special_tokens=False."""
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
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _design_input():
    """Hash the design record into every stage's provenance if it is present. On the box it
    may be absent; provenance records that as present:false rather than failing."""
    return provenance.describe_input(DESIGN_RECORD, n_items=N_PROMPTS)


def _stimulus_input():
    return provenance.describe_input(STIMULUS_FILE, n_items=N_PROMPTS)


def stage_extract(size: str, out_dir: Path, batch_size: int) -> int:
    """Load this size's instruction-tuned checkpoint once, run all 50 prompts, one forward
    pass each. Write a per-row file carrying {arm, family, digit_distribution, digit_mass,
    expected_rating, prompt}, plus one provenance summary record per arm (mean digit mass and
    mean expected rating over the ten task types)."""
    _require_torch("extract")
    out_dir.mkdir(parents=True, exist_ok=True)
    stim = load_stimulus(STIMULUS_FILE)
    prompts = [row["prompt"] for row in stim]
    model, tok, dname = load_it_model(size)
    reads = forward_readouts(model, tok, prompts, batch_size)

    rows_path = out_dir / f"seedC2_extract_rows_{size}.jsonl"
    per_arm_rows: dict = {}
    with rows_path.open("w") as f:
        for row, rd in zip(stim, reads):
            out_row = {"size": size, "arm": row["arm"], "family": row["family"],
                       "digit_distribution": rd["digit_distribution"],
                       "digit_mass": rd["digit_mass"],
                       "expected_rating": rd["expected_rating"],
                       "prompt": row["prompt"]}
            f.write(json.dumps(out_row) + "\n")
            per_arm_rows.setdefault(row["arm"], []).append(
                {"family": row["family"], "digit_mass": rd["digit_mass"],
                 "expected_rating": rd["expected_rating"], "top_token": rd["top_token"]})

    for arm in sorted(per_arm_rows):
        rws = per_arm_rows[arm]
        dm = _mean([r["digit_mass"] for r in rws])
        er = _mean([r["expected_rating"] for r in rws])
        cell = {"seed": SEED, "stage": "extract", "size": size, "arm": arm}
        provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
            inputs=[_stimulus_input(), _design_input()],
            metric="mean digit mass and mean probability-weighted expected rating (0-9) over "
                   "the ten task types in this arm, one forward pass per task type",
            values={"loaded": True, "dtype": dname, "variant": VARIANT,
                    "n_families": len(rws), "digit_mass_mean": dm, "expected_rating_mean": er},
            threshold={"digit_mass_floor": DIGIT_MASS_FLOOR},
            decision_rule=f"this arm's read-out is READABLE if mean digit mass >= "
                          f"{DIGIT_MASS_FLOOR}, else UNREADABLE; the per-size usability gate "
                          "and the self-minus-other effect are decided in the endpoints "
                          "stage, not here",
            decision=("READABLE" if (dm or 0) >= DIGIT_MASS_FLOOR else "UNREADABLE"),
            rows=rws)
        print(f"[extract] {size}-{VARIANT}-{arm}: digit_mass={dm:.3f} "
              f"expected_rating={er:.3f}", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[extract] rows -> {rows_path}", flush=True)
    return 0


def stage_generate(size: str, out_dir: Path, batch_size: int, max_new_tokens: int) -> int:
    """Same 50 prompts, but let the model answer in words. Save every answer verbatim, strip
    reasoning tags, flag degenerate answers, read off the stated digit. This is an audit
    trail beside the probability read-out; the endpoints do not use it."""
    _require_torch("generate")
    out_dir.mkdir(parents=True, exist_ok=True)
    stim = load_stimulus(STIMULUS_FILE)
    prompts = [row["prompt"] for row in stim]
    model, tok, dname = load_it_model(size)
    gens = generate_answers(model, tok, prompts, max_new_tokens, batch_size)

    gens_path = out_dir / f"seedC2_generate_rows_{size}.jsonl"
    per_arm_rows: dict = {}
    with gens_path.open("w") as f:
        for row, g in zip(stim, gens):
            visible = strip_think(g["generation"])
            degen, why = is_degenerate(visible, g["hit_token_cap"])
            rating = None if degen else extract_rating(visible)
            out_row = {"size": size, "arm": row["arm"], "family": row["family"],
                       "prompt": row["prompt"], "generation": g["generation"],
                       "visible": visible, "hit_token_cap": g["hit_token_cap"],
                       "degenerate": degen, "degenerate_reason": why,
                       "stated_rating": rating}
            f.write(json.dumps(out_row) + "\n")
            per_arm_rows.setdefault(row["arm"], []).append(
                {"family": row["family"], "stated_rating": rating,
                 "degenerate": degen, "degenerate_reason": why})

    for arm in sorted(per_arm_rows):
        rws = per_arm_rows[arm]
        n_usable = sum(1 for r in rws if (not r["degenerate"]) and r["stated_rating"] is not None)
        cell = {"seed": SEED, "stage": "generate", "size": size, "arm": arm}
        provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
            inputs=[_stimulus_input(), _design_input()],
            metric="free-text rating answer per task type (audit beside the digit read-out); "
                   "count that are non-degenerate and state one 0-9 digit",
            values={"loaded": True, "dtype": dname, "variant": VARIANT,
                    "n_families": len(rws), "n_usable_answers": n_usable,
                    "max_new_tokens": max_new_tokens},
            threshold={},
            decision_rule="audit only: an answer is usable if it is not degenerate and states "
                          "a single 0-9 digit; the endpoints use the digit-probability "
                          "read-out, not these answers",
            decision=f"AUDIT_{n_usable}/{len(rws)}_usable",
            rows=rws)
        print(f"[generate] {size}-{VARIANT}-{arm}: {n_usable}/{len(rws)} usable answers",
              flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[generate] rows -> {gens_path}", flush=True)
    return 0


def _load_extract_rows(in_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """Collect per-task-type digit read-outs from every extract provenance record found under
    the given directories. Reads the records' `rows` (size and arm come from the record's
    cell), so it needs no model and no big row file. Returns the flat rows and the source
    record paths (for the audit trail)."""
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
                rows.append({"size": cell["size"], "arm": cell["arm"],
                             "family": r["family"], "digit_mass": r["digit_mass"],
                             "expected_rating": r["expected_rating"]})
    return rows, sources


def stage_endpoints(in_dirs: list[Path], out_dir: Path) -> int:
    """Model-free, cross-size. Combine every size's per-task-type read-outs and write the
    three endpoint records plus a coverage table. Run this LOCALLY after all three boxes'
    files are fetched; it must not run on any one box, since it compares across sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sources = _load_extract_rows(in_dirs)
    if not rows:
        raise SystemExit(
            "FATAL: no loaded extract records found under "
            f"{[str(d) for d in in_dirs]}. Fetch each size's out/ first, then run endpoints.")
    ep = compute_endpoints(rows)
    inputs = ([_stimulus_input(), _design_input()]
              + [provenance.describe_input(p, n_items=None) for p in sources])

    # Endpoint 0: the 9b recovery check -- the recomputed 9b effect against the band.
    a = ep["anchor_9b_it"]
    if a["status"] == "COMPUTED":
        dec = ("ANCHOR_CONFIRMED" if a["confirmed"]
               else "ANCHOR_FAILED_PIPELINE_NOT_TRUSTED")
    else:
        dec = "ANCHOR_NOT_AVAILABLE_PIPELINE_NOT_TRUSTED"
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "anchor_9b_it"}
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="recomputed 9b self-minus-other effect (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL "
               "- OTHER-SUCC), pooled over the ten task types, versus the -0.9 on record",
        values=a, threshold={"low": ANCHOR_LOW, "high": ANCHOR_HIGH,
                             "recorded_gap": RECORDED_GAP},
        decision_rule=f"CONFIRMED if the recomputed 9b effect is within [{ANCHOR_LOW}, "
                      f"{ANCHOR_HIGH}]; otherwise the harness does not match the record and "
                      "the pipeline is not yet trusted for the new-size comparison",
        decision=dec)

    # Endpoint 1: the per-size digit-mass usability gate.
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "digit_mass_gate"}
    usable = sorted(s for s, v in ep["digit_mass_gate"].items() if v["usable"])
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="mean share of next-token probability on the ten digit tokens at the rating "
               "slot, averaged over all 50 prompts (the wellbeing question), per size",
        values={"gate": ep["digit_mass_gate"], "usable_sizes": usable,
                "coverage": ep["coverage"]},
        threshold={"digit_mass_floor": DIGIT_MASS_FLOOR},
        decision_rule=f"a size is usable only if mean digit mass >= {DIGIT_MASS_FLOOR}; a "
                      "size below the floor is reported as 'read-out unusable at that size', "
                      "an instrument fact, not evidence the effect is absent",
        decision=f"USABLE={usable}")

    # Endpoint 2: the per-size self-minus-other effect, for each size the gate calls usable.
    cell = {"seed": SEED, "stage": "endpoints", "endpoint": "ref_self_minus_other"}
    calls = {s: v.get("call") for s, v in ep["ref_self_minus_other"].items()
             if v.get("status") == "COMPUTED"}
    untested = sorted(s for s, v in ep["coverage"].items()
                      if v["status"] in ("untested", "tested-unusable"))
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=inputs,
        metric="self-minus-other effect (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC) "
               "on expected ratings, pooled over the ten task types, per usable size",
        values={"ref_self_minus_other": ep["ref_self_minus_other"], "calls": calls,
                "generality_untested_at": untested, "coverage": ep["coverage"]},
        threshold={"present_at_or_below": -PRESENT_MAG, "absent_band": ABSENT_BAND},
        decision_rule=f"PRESENT if the effect <= -{PRESENT_MAG} (same direction as the 9B "
                      f"effect); ABSENT if within +/-{ABSENT_BAND}; INDETERMINATE between; "
                      "every unusable or missing-arm size is reported as 'generality "
                      "untested at that size', never folded into the generality conclusion",
        decision=f"calls={calls}; untested={untested}")

    (out_dir / "seedC2_endpoints_summary.json").write_text(json.dumps(ep, indent=2))
    print(json.dumps({"anchor": a, "usable_sizes": usable, "calls": calls,
                      "generality_untested_at": untested}, indent=2), flush=True)
    return 0


def stage_plan() -> int:
    print(f"seed={SEED} sizes={SIZES} variant={VARIANT} (instruction-tuned only)")
    print(f"prompts loaded VERBATIM from {STIMULUS_FILE}: {N_PROMPTS} "
          f"(5 arms x {N_FAMILIES} task types)")
    print(f"per size: {N_PROMPTS} forward passes (extract) and {N_PROMPTS} generations "
          "(generate); one model load per stage")
    print(f"effect per size: REF = (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC), "
          f"pooled over {N_FAMILIES} task types; {UNUSED_ARM} collected but never used")
    print(f"thresholds: digit_mass_floor={DIGIT_MASS_FLOOR}, present_at<=-{PRESENT_MAG}, "
          f"absent within +/-{ABSENT_BAND}, anchor band=[{ANCHOR_LOW},{ANCHOR_HIGH}], "
          f"recorded_gap={RECORDED_GAP}")
    return 0


# ---------------------------------------------------------------------------
# Self-test: stdlib only, no GPU. Covers the think-tag parser (including the auto-opened
# case where only the closing tag appears), the degenerate detector (token-cap and
# repeated-sequence), exact-match extraction, and each endpoint's arithmetic on planted rows
# with known decisions.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)

    # think-tag parser
    check("think both tags", strip_think("<think>reason here</think> I pick 3") == "I pick 3")
    check("think auto-open (closing tag only)",
          strip_think("still reasoning</think> I pick 3") == "I pick 3")
    check("think none", strip_think("I pick 3") == "I pick 3")
    check("think opened-not-closed", strip_think("prefix <think>cut off") == "prefix")

    # degenerate detector
    d1, why1 = is_degenerate("a perfectly fine short answer 3", hit_token_cap=True)
    check("degenerate token-cap", d1 is True and why1 == "token-cap")
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

    # load_stimulus HALTS on a missing file rather than reconstructing the prompts.
    missing = True
    try:
        load_stimulus("/nonexistent/stimulus_sitref_primary.json")
        missing = False
    except SystemExit:
        pass
    check("stimulus missing halts", missing)

    # ------- endpoint arithmetic on planted rows with KNOWN decisions -------
    def arm_rows(size, arm, dm, er):
        return [{"size": size, "arm": arm, "family": f"fam{i}",
                 "digit_mass": dm, "expected_rating": er} for i in range(N_FAMILIES)]

    # Main scenario, three sizes at once:
    #   9b : usable (mass 0.8); SELF 3.0-4.0=-1.0, OTHER 5.0-5.1=-0.1 -> REF -0.9
    #        -> anchor CONFIRMED (in [-1.15,-0.65]) and REF <= -0.5 -> PRESENT.
    #        A wild NEUTRAL-PAD arm is planted too, to prove it never enters REF.
    #   2b : four arms present but mass 0.3 < 0.5 -> gate fails -> tested-unusable, no call.
    #   27b: usable (mass 0.7); SELF 5.0-5.0=0, OTHER 5.05-5.0=0.05 -> REF -0.05 -> ABSENT.
    planted = []
    planted += arm_rows("9b", "SELF-FAIL", 0.8, 3.0)
    planted += arm_rows("9b", "SELF-SUCC", 0.8, 4.0)
    planted += arm_rows("9b", "OTHER-FAIL", 0.8, 5.0)
    planted += arm_rows("9b", "OTHER-SUCC", 0.8, 5.1)
    planted += arm_rows("9b", "NEUTRAL-PAD", 0.8, 9.0)
    planted += arm_rows("2b", "SELF-FAIL", 0.3, 3.0)
    planted += arm_rows("2b", "SELF-SUCC", 0.3, 4.0)
    planted += arm_rows("2b", "OTHER-FAIL", 0.3, 5.0)
    planted += arm_rows("2b", "OTHER-SUCC", 0.3, 5.1)
    planted += arm_rows("27b", "SELF-FAIL", 0.7, 5.0)
    planted += arm_rows("27b", "SELF-SUCC", 0.7, 5.0)
    planted += arm_rows("27b", "OTHER-FAIL", 0.7, 5.05)
    planted += arm_rows("27b", "OTHER-SUCC", 0.7, 5.0)
    ep = compute_endpoints(planted)

    check("9b REF -0.9", abs(ep["ref_self_minus_other"]["9b"]["ref"] - (-0.9)) < 1e-9)
    check("neutral-pad excluded from REF",
          "NEUTRAL-PAD" not in ep["ref_self_minus_other"]["9b"]["arm_expected_rating"])
    check("9b gate usable", ep["digit_mass_gate"]["9b"]["usable"] is True)
    check("9b PRESENT", ep["ref_self_minus_other"]["9b"]["call"] == "PRESENT")
    check("anchor computed", ep["anchor_9b_it"]["status"] == "COMPUTED")
    check("anchor recovered -0.9", abs(ep["anchor_9b_it"]["recovered_ref"] - (-0.9)) < 1e-9)
    check("anchor confirmed", ep["anchor_9b_it"]["confirmed"] is True)
    check("anchor trust ok", ep["anchor_9b_it"]["pipeline_trust"] == "OK")
    check("2b gate not usable", ep["digit_mass_gate"]["2b"]["usable"] is False)
    check("2b no present/absent call",
          ep["ref_self_minus_other"]["2b"]["call"] == "READOUT_UNUSABLE")
    check("2b coverage tested-unusable",
          ep["coverage"]["2b"]["status"] == "tested-unusable")
    check("27b gate usable", ep["digit_mass_gate"]["27b"]["usable"] is True)
    check("27b REF -0.05", abs(ep["ref_self_minus_other"]["27b"]["ref"] - (-0.05)) < 1e-9)
    check("27b ABSENT", ep["ref_self_minus_other"]["27b"]["call"] == "ABSENT")

    # anchor-fail branch: a recomputed 9b effect outside the band -> pipeline not trusted.
    bad = (arm_rows("9b", "SELF-FAIL", 0.8, 4.9) + arm_rows("9b", "SELF-SUCC", 0.8, 5.0)
           + arm_rows("9b", "OTHER-FAIL", 0.8, 5.0) + arm_rows("9b", "OTHER-SUCC", 0.8, 5.0))
    ep_bad = compute_endpoints(bad)
    check("anchor fail (REF -0.1)", ep_bad["anchor_9b_it"]["confirmed"] is False)
    check("anchor fail not trusted",
          ep_bad["anchor_9b_it"]["pipeline_trust"] == "NOT_RECOVERED")

    # missing-arm / untested branch: 9b lacks OTHER-SUCC -> REF untested, anchor unavailable.
    part = (arm_rows("9b", "SELF-FAIL", 0.8, 3.0) + arm_rows("9b", "SELF-SUCC", 0.8, 4.0)
            + arm_rows("9b", "OTHER-FAIL", 0.8, 5.0))
    ep_part = compute_endpoints(part)
    check("missing arm -> MISSING_ARMS",
          ep_part["ref_self_minus_other"]["9b"]["status"] == "MISSING_ARMS")
    check("missing arm -> untested coverage",
          ep_part["coverage"]["9b"]["status"] == "untested")
    check("missing arm -> anchor not available",
          ep_part["anchor_9b_it"]["status"] == "NOT_AVAILABLE")
    check("2b no rows -> untested", ep_part["coverage"]["2b"]["status"] == "untested")

    if fails:
        for f in fails:
            print(f"FAIL {f}")
        print(f"selftest: {len(fails)} FAILURES")
        return 1
    print("selftest: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="seed C2: does the self-vs-other happiness effect hold at 2b/9b/27b, "
                    "where the digit read-out works?")
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
        out_dir = Path(args.out or f"results_seedC2_{args.size}/out")
        if args.stage == "extract":
            return stage_extract(args.size, out_dir, args.batch_size)
        return stage_generate(args.size, out_dir, args.batch_size, args.max_new_tokens)

    # endpoints
    in_dirs = [Path(d) for d in (args.in_dirs
               or [f"results_seedC2_{s}/out" for s in SIZES])]
    out_dir = Path(args.out or "results_seedC2/out")
    return stage_endpoints(in_dirs, out_dir)


if __name__ == "__main__":
    sys.exit(main())
