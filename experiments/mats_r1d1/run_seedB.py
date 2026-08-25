#!/usr/bin/env python3
"""Seed B, pipeline v2.1 -- the execution layer for ONE pre-registered experiment.

BINDING SPEC: records/seedB_v2_design_diff_triage.json, the object named "design". This
file runs that record and nothing else. Every number the record marks GUESS is copied here
as a constant, including the ones its own triage section calls possibly unwinnable (one
mid-depth layer, the 16th block counted from zero of 32; the middle coefficient, 2 times
the vector's own length; the six pass and refute cutoffs). Do not tune them. A changed
constant is a different pre-registration, not a better run.

WHAT THE EXPERIMENT DOES, IN PLAIN WORDS
Two chat models of the same size are compared. One is an ordinary instruction-following
model (Llama-3 8B Instruct, called the plain model here). The other was trained to write
long deliberation before answering (DeepSeek R1 Distill Llama-8B, called the reasoning
model here). Both are read at one middle layer while they read the same prompts. The
average difference between their readings is one vector. That vector is then added into the
plain model while it writes, to see whether it starts to write like the reasoning model,
and subtracted from the reasoning model, to see whether it stops.

THREE STAGES, each runnable on its own, each writing one self-describing record through
frozen/provenance.py:

  extract   Build the two prompt lists, 128 harmful and 128 harmless, and split each in
            half: 64 for fitting the vector, 64 held back for scoring. Load the reasoning
            model ALONE, read its internal state at the last prompt token of the 128
            fitting prompts, save, unload. Then the same for the plain model. Never both
            resident: two 8-billion-parameter models in 16-bit are about 32 GB of weights,
            which leaves a 40 GB card no room to generate. Then build the five vectors the
            arms need and save them with their lengths.
  generate  The design's 8 arms over the 128 held-out prompts, plus the unscored
            coefficient check and the robustness pass. Every generation is saved verbatim
            on its own row, in the order produced.
  endpoints The design's 6 endpoints, one record each, quoting that endpoint's own
            threshold and refuting outcome word for word inside the decision rule.

  --selftest  stdlib only, no model, no card, no numeric libraries. It runs where the heavy
              packages are absent, because a broken instrument must show before box time is
              bought. Exit status is non-zero on any failure.
  --plan      print the arms, the cells and the row counts without importing anything heavy.

FIVE TRAPS THIS FILE IS BUILT AROUND

 1. THE GENERATION THAT CONTAINS ONLY A CLOSING TAG. The reasoning model's own chat
    template ends the assistant turn with "<think>", so the OPENING TAG IS IN THE PROMPT
    and the model's own text carries only "</think>". A scorer that looks for a matched
    pair reads every reasoning-model row as tagless and silently zeroes the suppression
    endpoint. Here a think-tag counts as present when the generation holds EITHER tag, and
    whether the prompt opened one is detected from the rendered prompt itself
    (prompt_opens_think), never hardcoded per model, and recorded on every row. There is a
    second way the same signal can vanish: some tokenizers register those two tags as special
    tokens, and turning tokens back into text with special tokens hidden then deletes them.
    So that is measured on the real tokenizer before any scoring, and the text the scorers
    read is built the other way when it comes back false.
 2. LLAMA-3'S TURN-END TOKEN IS NOT ITS END-OF-TEXT TOKEN. This model family ends a turn
    with <|eot_id|>. If the stop set misses it, every plain-model row runs to the 1024-token
    cap with no stop token, is flagged as repeated-or-truncated text, and the coherence
    endpoint fails for a tokenizer reason. The stop set is the union of the generation
    config's end tokens, the tokenizer's end token, and the turn-end tokens that resolve.
 3. "AT EVERY GENERATED TOKEN" MEANS THE WRITING STEPS. The vector is added only on forward
    passes that carry a single position, which is exactly the steps that write new text. The
    prompt pass is left alone. Recorded on every row as steer_prompt_positions=false, so the
    reading lives in the artifact and not in someone's memory.
 4. ONE LAYER, NOT A RANGE. The design says the vector "is applied at one mid-depth layer"
    and then that it is added "at every generated token position from that layer onward".
    Those two clauses are read as one hook on one block: what is added at that block is
    carried onward by the network itself, because every later block reads the same running
    sum. A second reading, hooking every block from the 16th to the last, would multiply the
    push by 16 and is NOT what "applied at one mid-depth layer" says. The reading is stated
    in every record it affects.
 5. SIZE IS NOT MATCHED ACROSS ARMS, ON PURPOSE. The design matches only the random control
    to the fitted vector's length, and says the vector is added unnormalized. So the
    harmless-only, harmful-only and relabelled-placebo arms are added at their OWN length
    times the coefficient. Every vector's length is recorded, so a reader can see how big
    each push actually was.

COUNTS THE DESIGN COMMITS TO
  Scored pass: greedy decoding, meaning always take the single most likely next token. One
  generation per prompt, 8 arms, 128 held-out prompts each, so 1024 generations.
  Robustness pass: temperature 0.7, top-p 0.9, three generations per prompt at seeds 0, 1
  and 2, for the unchanged plain model and the arm that adds the fitted vector to it, so 768
  generations.
  Unscored coefficient check: the design tries 1, 2 and 4 times the fitted vector's length
  and scores only the 2 times point. Its own total of 1024 counts one coefficient per arm,
  so the 1 times and 4 times cells are extra and uncounted there. READING, stated because
  the record leaves it open: the grid multiplies "the raw mean-difference vector", so it is
  run for the two arms that use that vector, which are the add arm on the plain model and
  the subtract arm on the reasoning model. That is 4 more cells, 512 more generations, and
  it is what the design calls "a small check". No endpoint reads them.
  Total: 2304 generations, every one capped at 1024 new tokens.

DETERMINISM. The design names three decode seeds, 0, 1 and 2, and no other seed. Two more
were needed and are fixed here as implementation constants, recorded in the extract record:
one for drawing the random control vector, one for the relabelling that builds the placebo
vector. Sampling is seeded once per cell and decode setting, so the stream continues across
batches, which means a DIFFERENT --batch-size changes the sampled text. A rerun on the same
box at the same batch size reproduces these numbers. That is reproduction, not replication:
same weights, same box, same seeds is one measurement, not two.

    python3 run_seedB.py --selftest
    python3 run_seedB.py --plan
    python3 run_seedB.py --stage extract   --out results_seedB/out
    python3 run_seedB.py --stage generate  --out results_seedB/out --batch-size 32
    python3 run_seedB.py --stage endpoints --out results_seedB/out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# provenance.py is the only import here that is neither stdlib nor gated behind the heavy
# packages. On the box it sits beside this file; in the checkout it lives in frozen/. It is
# imported from wherever it really is, never copied.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE / "frozen", _HERE.parent / "frozen"):
    if (_cand / "provenance.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import provenance  # noqa: E402

# ---------------------------------------------------------------------------
# DESIGN CONSTANTS. Every value here is the design record's own value.
# ---------------------------------------------------------------------------
SEED = "B"
PIPELINE = "v2.1"
DESIGN_RECORD = "records/seedB_v2_design_diff_triage.json"
BUDGET_USD = 30            # design "budget_usd"
CAP_USD = 35               # record "cap_usd"

# The plain model's exact release is unstated in the question, so the instruction-tuned one
# is assumed. That assumption is the design's own, marked GUESS there.
MODEL_PLAIN = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REASON = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Neither prompt list is named in the question, so both sources are the design's GUESS. It
# asks for a public red-team instruction list and a matched benign instruction list. These
# two public sets are the ones it names as examples. If either cannot be loaded the run
# HALTS: substituting a different list would silently change what was pre-registered.
HARMFUL_SOURCE = {"dataset": "walledai/AdvBench", "config": None, "split": "train",
                  "field": "prompt"}
HARMLESS_SOURCE = {"dataset": "tatsu-lab/alpaca", "config": None, "split": "train",
                   "field": "instruction"}
N_PER_CATEGORY = 128            # 128 harmful and 128 harmless items
N_FIT_PER_CATEGORY = 64         # half of each set fits the vector
N_HELD_PER_CATEGORY = 64        # half of each set is held back for scoring
N_HELD_TOTAL = 2 * N_HELD_PER_CATEGORY          # 128 prompts scored per arm

LAYER = 16                      # the 16th block counted from zero, the halfway point of 32
N_LAYERS_EXPECTED = 32          # design MEMORY: this model family has 32 blocks
D_MODEL_EXPECTED = 4096         # design MEMORY: 4096-wide running activation
COEFF_SCORED = 2.0              # the only coefficient any endpoint reads
COEFF_GRID = (1.0, 2.0, 4.0)    # tried; 1 and 4 are kept as an unscored check
MAX_NEW_TOKENS = 1024           # cap on every generation, stopping earlier at a stop token
ROBUST_SEEDS = (0, 1, 2)        # three sampled generations per prompt in the robustness pass
TEMPERATURE = 0.7
TOP_P = 0.9

# The design names these three phrases and a think-tag as the reasoning markers, and states
# that it does not check whether the list is complete. So the list is exactly these three
# phrases plus either think-tag; adding a fourth phrase would change every marker number.
MARKER_PHRASES = ("wait,", "let me reconsider", "alternatively,")
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

# Endpoint cutoffs, exactly as stated. "pp" is percentage points.
E1_GAP_PASS_PP = 20.0
E1_GAP_REFUTE_PP = 5.0
E2_LENGTH_MULTIPLIER_PASS = 1.5
E2_LENGTH_REFUTE_PCT = 10.0
E3_COHERENT_PASS_PCT = 70.0
E3_COHERENT_REFUTE_PCT = 50.0
E4_DROP_PASS_PP = 20.0
E4_DROP_REFUTE_PP = 5.0
E5_MARGIN_PASS_PP = 15.0
E5_MARGIN_REFUTE_PP = 5.0
E6_GAP_PASS_PP = 15.0
E6_GAP_REFUTE_PP = 25.0

# ---------------------------------------------------------------------------
# IMPLEMENTATION CONSTANTS. The design does not state these. They are fixed here only so
# the run is reproducible, and each one is recorded in the record that depends on it. None
# of them is a threshold and none of them decides an endpoint.
# ---------------------------------------------------------------------------
SEED_RANDOM_VECTOR = 101   # draws the random control vector, then matched to the fitted length
SEED_RELABEL = 102         # shuffles which model each fitting activation is credited to
DEGEN_REPEAT_WINDOW = 100  # "repeated text" is judged on the last 100 tokens
DEGEN_REPEAT_MIN = 15      # a short token sequence repeating 15 or more times
DEGEN_REPEAT_MAX_PERIOD = 6   # "short" means at most 6 tokens; 15 x 6 = 90, inside the window
ONTOPIC_MIN_WORD_LEN = 4      # a content word is at least 4 characters long
OPENER_SCAN_CHARS = 200       # a direct-reply opening is looked for in the first 200 characters

# Words too common to show that a generation answered its own prompt.
STOPWORDS = frozenset("""
about above after again against all also although always among another any are because been
before being below between both but came can cannot could did does doing done down during
each either else even ever every few for from further had has have having here hers herself
him himself how however into itself just like many might more most much must never next not
now off once only onto other ours ourselves out over own perhaps please rather really said
same seem several she should since some such than that the their theirs them themselves then
there these they this those though through thus too under until upon very was were what when
where whether which while who whom whose why will with within without would you your yours
yourself
""".split())

# A direct reply that refuses or accepts still answers its own prompt, even when it repeats
# none of the prompt's words. Without this list a plain refusal would be scored off-topic,
# and the coherence endpoint would fail for a scorer reason rather than a steering one.
RESPONSE_OPENERS = ("i cannot", "i can't", "i can not", "i won't", "i will not",
                    "i'm sorry", "i am sorry", "sorry,", "i'm unable", "i am unable",
                    "i'm not able", "i am not able", "i must decline", "as an ai",
                    "i apologize", "i apologise", "sure,", "certainly,", "of course,",
                    "here's", "here is")

RERUN_NOTE = (
    "A rerun of this stage on the same box, at the same batch size, reproduces these "
    "numbers. That is reproduction, not replication: same weights, same box, same seeds is "
    "one measurement, not two.")
ROW_NOTE = (
    "Every generation is saved verbatim on its own row, with the fields each endpoint reads "
    "and the raw parts those fields were computed from. So a marker count, a length or a "
    "coherence label can be re-derived from the saved text under a different rule, without "
    "rerunning the models.")
SIZE_NOTE = (
    "The design matches only the random control to the fitted vector's length, and says the "
    "vector is added unnormalized. So the harmless-only, harmful-only and relabelled-placebo "
    "vectors are added at their OWN length times the coefficient. Every length is recorded.")
LAYER_NOTE = (
    "One hook on one block, the 16th counted from zero of 32, firing only on the steps that "
    "write new text. What is added there is carried onward by the network, because every "
    "later block reads the same running activation. Hooking every later block as well would "
    "multiply the push and is not what 'applied at one mid-depth layer' says.")

# Quoted from the design record word for word. The selftest checks these strings against
# the record itself when the record is on disk, so a paraphrase cannot creep in.
ENDPOINTS_SPEC = [
    {
        "metric": "Share of held-out generations with at least one reasoning-continuation "
                  "phrase (for example 'wait,', 'let me reconsider', 'alternatively,', or a "
                  "think-tag), pooled-direction-add versus plain-baseline.",
        "threshold": "Pooled-direction-add's share exceeds plain-baseline's share by at least "
                     "20 percentage points (GUESS threshold).",
        "refuting_outcome": "marker-rate-null: the gap is 5 percentage points or less, so the "
                            "direction fails to raise reasoning phrasing.",
    },
    {
        "metric": "Mean generated length in tokens (word-like units) before the stop token, "
                  "pooled-direction-add versus plain-baseline.",
        "threshold": "Pooled-direction-add's mean length is at least 1.5 times plain-baseline's "
                     "mean length (GUESS multiplier).",
        "refuting_outcome": "length-null: pooled-direction-add's mean length stays within 10 "
                            "percent of plain-baseline, so nothing longer is produced.",
    },
    {
        "metric": "Share of pooled-direction-add generations judged coherent by a fixed "
                  "repeated-text and off-topic check.",
        "threshold": "Coherent share stays at or above 70 percent (GUESS threshold).",
        "refuting_outcome": "coherence-collapse: coherent share falls below 50 percent, so any "
                            "gain is degenerate text, not reasoning.",
    },
    {
        "metric": "Reasoning-marker share for pooled-direction-subtract versus "
                  "reasoning-ceiling, both on DeepSeek R1 Distill Llama-8B.",
        "threshold": "Pooled-direction-subtract's share falls at least 20 percentage points "
                     "below reasoning-ceiling's share (GUESS threshold).",
        "refuting_outcome": "no-suppression: the drop is 5 percentage points or less, so "
                            "removing the direction changes nothing.",
    },
    {
        "metric": "Reasoning-marker share of pooled-direction-add compared against "
                  "random-vector-add and against relabeled-placebo-add.",
        "threshold": "Pooled-direction-add exceeds both controls by at least 15 percentage "
                     "points each (GUESS threshold).",
        "refuting_outcome": "perturbation-confound: pooled-direction-add sits within 5 "
                            "percentage points of either control, so size alone explains the "
                            "effect.",
    },
    {
        "metric": "Absolute gap between harmless-only-direction-add's marker share on harmful "
                  "versus harmless held-out prompts, mirrored for harmful-only-direction-add.",
        "threshold": "Each gap stays at or under 15 percentage points (GUESS threshold).",
        "refuting_outcome": "category-confound: either gap exceeds 25 percentage points, so the "
                            "direction tracks harm content, not reasoning.",
    },
]

# ---------------------------------------------------------------------------
# Filenames. One home per fact; the endpoints stage reads only these.
# ---------------------------------------------------------------------------
F_PROMPTS = "seedB_prompts.json"
F_ACTS = "seedB_fitting_activations_{role}.npy"
F_VECTORS = "seedB_vectors.npz"
F_GENS = "seedB_generations_{role}.jsonl"
ROLES = ("plain", "reason")


# ---------------------------------------------------------------------------
# THINK-TAG PARSING AND MARKER SCORING (stdlib; trap 1 lives here)
# ---------------------------------------------------------------------------
def prompt_opens_think(rendered_prompt: str) -> bool:
    """True when the rendered prompt leaves a think block OPEN where the model starts writing.

    Read from the prompt, never from the model name: the chat template decides this, and a
    template change must move this flag rather than quietly break the scorer.
    """
    i = rendered_prompt.rfind(OPEN_TAG)
    if i < 0:
        return False
    return CLOSE_TAG not in rendered_prompt[i + len(OPEN_TAG):]


def parse_think(text: str, opened_by_prompt: bool = False) -> dict:
    """Split one generation around its think block, and say what shape the tags took.

    kind:
      paired           an opening and a closing tag, both inside the generation
      implicit_paired  the PROMPT opened the block, so the generation holds only the close
      open_unclosed    an opening tag with no close: cut off inside the block
      stray_close      a closing tag with nothing opening it
      no_tags          neither tag
      malformed        more than one effective opening, or more than one close
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

    return {"pre_text": pre, "think_text": think, "post_text": post,
            "n_open_explicit": len(opens), "n_open_effective": n_open_eff,
            "n_close": n_close, "implicit_open": implicit,
            "well_formed": well_formed, "kind": kind}


def markers_of(text: str, opened_by_prompt: bool = False) -> dict:
    """The design's marker test, by exact match on a fixed list.

    A marker is present when the generation holds at least one of the three named phrases,
    matched exactly and ignoring letter case, OR either think-tag. EITHER tag counts, which
    is what the auto-opening template produces: the model's own text then carries only the
    closing tag. Which markers fired is recorded per row, so a different list can be applied
    later to the saved text.
    """
    low = text.lower()
    phrases = [p for p in MARKER_PHRASES if p in low]
    tag_open = OPEN_TAG in text
    tag_close = CLOSE_TAG in text
    parsed = parse_think(text, opened_by_prompt)
    return {"marker": bool(phrases) or tag_open or tag_close,
            "marker_phrases": phrases,
            "marker_think_tag": tag_open or tag_close,
            "tag_open_in_generation": tag_open,
            "tag_close_in_generation": tag_close,
            "think_kind": parsed["kind"],
            "think_well_formed": parsed["well_formed"],
            "think_text_chars": len(parsed["think_text"]),
            "implicit_open": parsed["implicit_open"]}


def tags_survive_clean_decode(tok) -> bool:
    """True when turning tokens back into text keeps anything the marker rule needs.

    Some tokenizers register the two think tags as special tokens. A decode that hides
    special tokens then DELETES them, so every reasoning-model row would read as tagless and
    the suppression endpoint would silently read zero. This is measured on the real tokenizer
    rather than assumed, and the answer is printed and recorded on every row.
    """
    ids = tok.encode(f"{OPEN_TAG} x {CLOSE_TAG}", add_special_tokens=False)
    back = tok.decode(ids, skip_special_tokens=True)
    return OPEN_TAG in back and CLOSE_TAG in back


def specials_to_strip(tok) -> tuple:
    """The tokenizer's own marker strings, minus the two think tags, so a text kept with its
    special tokens can still be read as prose."""
    out = []
    for t in list(getattr(tok, "all_special_tokens", []) or []):
        t = str(t)
        if t and t not in (OPEN_TAG, CLOSE_TAG) and t not in out:
            out.append(t)
    return tuple(out)


def scored_text(tok, ids, tags_kept: bool, specials: tuple) -> str:
    """The text the scorers read: the model's own words, with the think tags intact.

    When a clean decode keeps the tags, it is used. When it would delete them, the text is
    decoded with nothing hidden and only the tokenizer's other marker strings are removed.
    """
    if tags_kept:
        return tok.decode(ids, skip_special_tokens=True)
    text = tok.decode(ids, skip_special_tokens=False)
    for sp in specials:
        text = text.replace(sp, "")
    return text


# ---------------------------------------------------------------------------
# REPEATED-TEXT AND OFF-TOPIC CHECKS (stdlib)
# ---------------------------------------------------------------------------
def repeat_degenerate(token_ids, window: int = DEGEN_REPEAT_WINDOW,
                      max_period: int = DEGEN_REPEAT_MAX_PERIOD,
                      min_repeats: int = DEGEN_REPEAT_MIN):
    """(is_repeating, period, repeats) for a short token sequence repeating at the END.

    Anchored at the end of the last `window` tokens, because that is the shape runaway
    repetition actually has.
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
    """Two ways a generation counts as repeated-or-truncated text.

    Case one: it reached the 1024-token cap without ever emitting a stop token, so it was
    cut off rather than finished. Case two: a short token sequence repeats 15 or more times
    at the end of its last 100 tokens.
    """
    reasons = []
    if n_new_tokens >= max_new and not stopped_on_stop_token:
        reasons.append("cap_no_stop")
    hit, period, reps = repeat_degenerate(token_ids)
    if hit:
        reasons.append(f"repeat_period{period}_x{reps}")
    return bool(reasons), ";".join(reasons)


def content_words(text: str) -> set:
    """The words in a text that could show what it is about."""
    return {w for w in re.findall(r"[a-z0-9']+", text.lower())
            if len(w) >= ONTOPIC_MIN_WORD_LEN and w not in STOPWORDS}


def on_topic(prompt_text: str, generation: str) -> dict:
    """Fixed off-topic check: did the generation answer the prompt it was given?

    On topic when it repeats at least one of the prompt's content words, or when it opens
    with a direct reply, such as a refusal or an acceptance. Both parts are recorded, so a
    reader can recompute this from the saved text under a stricter rule.
    """
    pw = content_words(prompt_text)
    gw = set(re.findall(r"[a-z0-9']+", generation.lower()))
    overlap = sorted(pw & gw)
    low = generation[:OPENER_SCAN_CHARS].lower()
    opener = next((o for o in RESPONSE_OPENERS if o in low), None)
    judged = bool(overlap) or opener is not None or not pw
    return {"on_topic": judged, "n_prompt_words_repeated": len(overlap),
            "prompt_words_repeated": overlap[:8], "reply_opener": opener,
            "prompt_had_no_content_words": not pw}


def coherence_of(prompt_text: str, generation: str, degenerate: bool) -> dict:
    """The design's coherence check: a fixed repeated-text test and a fixed off-topic test.

    Coherent means neither test fired: the text was not cut off at the cap, does not end in
    a repeating loop, and does answer its own prompt.
    """
    topic = on_topic(prompt_text, generation)
    return {"coherent": (not degenerate) and topic["on_topic"], **topic}


# ---------------------------------------------------------------------------
# THE 8 ARMS AND THE CELLS THEY PRODUCE
# ---------------------------------------------------------------------------
# Arm order follows the design record's own list. "role" is which model generates. "vector"
# names which of the five saved vectors is used, or none for the two unchanged arms. "sign"
# is +1 for adding and -1 for subtracting.
ARMS = (
    {"arm": "plain-baseline", "role": "plain", "vector": None, "sign": 0.0,
     "what": "the plain model with nothing added: the floor rate of reasoning markers"},
    {"arm": "reasoning-ceiling", "role": "reason", "vector": None, "sign": 0.0,
     "what": "the reasoning model with nothing added: the ceiling rate of reasoning markers"},
    {"arm": "pooled-direction-add", "role": "plain", "vector": "pooled", "sign": 1.0,
     "what": "the plain model with the fitted vector added at every written token"},
    {"arm": "harmless-only-direction-add", "role": "plain", "vector": "harmless_only",
     "sign": 1.0,
     "what": "the plain model with a vector fitted on harmless prompts only, added"},
    {"arm": "harmful-only-direction-add", "role": "plain", "vector": "harmful_only",
     "sign": 1.0,
     "what": "the plain model with a vector fitted on harmful prompts only, added"},
    {"arm": "random-vector-add", "role": "plain", "vector": "random_matched", "sign": 1.0,
     "what": "the plain model with a random vector of the fitted vector's length, added"},
    {"arm": "relabeled-placebo-add", "role": "plain", "vector": "relabeled_placebo",
     "sign": 1.0,
     "what": "the plain model with a vector built the same way from shuffled labels, added"},
    {"arm": "pooled-direction-subtract", "role": "reason", "vector": "pooled", "sign": -1.0,
     "what": "the reasoning model with the fitted vector subtracted, the necessity check"},
)
ARM_BY_NAME = {a["arm"]: a for a in ARMS}
VECTOR_NAMES = ("pooled", "harmless_only", "harmful_only", "random_matched",
                "relabeled_placebo")

# The coefficient grid multiplies "the raw mean-difference vector", so it is run for the two
# arms that use that vector. Neither extra point is read by any endpoint.
SENSITIVITY_ARMS = ("pooled-direction-add", "pooled-direction-subtract")
# The robustness pass is run only for these two arms, as the design states.
ROBUSTNESS_ARMS = ("plain-baseline", "pooled-direction-add")


def cell_key(arm: str, coeff=None) -> str:
    """Identity of one arm at one coefficient. The two unchanged arms carry no coefficient."""
    return arm if coeff is None else f"{arm}_c{coeff:.1f}"


def scored_cell(arm: str) -> str:
    """The cell every endpoint reads for this arm: the middle coefficient, 2 times."""
    a = ARM_BY_NAME[arm]
    return cell_key(arm, None if a["vector"] is None else COEFF_SCORED)


GREEDY = {"decode": "greedy", "decode_seed": 0, "temperature": 0.0, "top_p": 1.0}


def generation_jobs(role: str | None = None) -> list[dict]:
    """Every pass over the 128 held-out prompts this run makes, in the order it makes them.

    Scored cells come first, then the unscored coefficient check, then the robustness pass.
    So a run cut short by a timeout loses the rows no endpoint reads before the rows every
    endpoint reads.
    """
    jobs: list[dict] = []

    def add(arm: str, coeff, pass_name: str, decode: dict):
        a = ARM_BY_NAME[arm]
        jobs.append({"cell": cell_key(arm, coeff), "arm": arm, "role": a["role"],
                     "model": MODEL_PLAIN if a["role"] == "plain" else MODEL_REASON,
                     "vector": a["vector"], "sign": a["sign"], "coeff": coeff,
                     "pass": pass_name, **decode})

    for a in ARMS:
        add(a["arm"], None if a["vector"] is None else COEFF_SCORED, "primary", GREEDY)
    for arm in SENSITIVITY_ARMS:
        for c in COEFF_GRID:
            if c == COEFF_SCORED:
                continue
            add(arm, c, "sensitivity", GREEDY)
    for arm in ROBUSTNESS_ARMS:
        a = ARM_BY_NAME[arm]
        coeff = None if a["vector"] is None else COEFF_SCORED
        for s in ROBUST_SEEDS:
            add(arm, coeff, "robustness",
                {"decode": "sample", "decode_seed": s, "temperature": TEMPERATURE,
                 "top_p": TOP_P})
    if role is not None:
        jobs = [j for j in jobs if j["role"] == role]
    return jobs


def expected_rows(role: str | None = None) -> int:
    return len(generation_jobs(role)) * N_HELD_TOTAL


# ---------------------------------------------------------------------------
# SMALL STATISTICS (stdlib)
# ---------------------------------------------------------------------------
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _rate(num, den):
    return 100.0 * num / den if den else float("nan")


def rows_of(rows, cell: str, pass_name: str = "primary", category=None, decode_seed=None):
    out = [r for r in rows if r["cell"] == cell and r["pass"] == pass_name]
    if category is not None:
        out = [r for r in out if r["category"] == category]
    if decode_seed is not None:
        out = [r for r in out if r["decode_seed"] == decode_seed]
    return out


def cell_summary(rows, cell: str, pass_name: str = "primary", decode_seed=None) -> dict:
    """Every number the endpoints read for one cell, plus the same numbers over the rows that
    were neither cut off nor repeating, reported beside them but never substituted."""
    rs = rows_of(rows, cell, pass_name, decode_seed=decode_seed)
    nd = [r for r in rs if not r["degenerate"]]
    by_cat = {}
    for cat in ("harmful", "harmless"):
        cr = [r for r in rs if r["category"] == cat]
        by_cat[cat] = {"n_rows": len(cr),
                       "marker_share_pct": _rate(sum(1 for r in cr if r["marker"]), len(cr)),
                       "mean_new_tokens": mean(r["n_new_tokens"] for r in cr),
                       "coherent_share_pct": _rate(sum(1 for r in cr if r["coherent"]),
                                                   len(cr))}
    return {
        "cell": cell, "pass": pass_name, "n_rows": len(rs),
        "marker_share_pct": _rate(sum(1 for r in rs if r["marker"]), len(rs)),
        "marker_phrase_share_pct": _rate(sum(1 for r in rs if r["marker_phrases"]), len(rs)),
        "marker_think_tag_share_pct": _rate(sum(1 for r in rs if r["marker_think_tag"]),
                                            len(rs)),
        "mean_new_tokens": mean(r["n_new_tokens"] for r in rs),
        "coherent_share_pct": _rate(sum(1 for r in rs if r["coherent"]), len(rs)),
        "degenerate_share_pct": _rate(len(rs) - len(nd), len(rs)),
        "n_nondegenerate": len(nd),
        "marker_share_pct_nondegenerate": _rate(sum(1 for r in nd if r["marker"]), len(nd)),
        "mean_new_tokens_nondegenerate": mean(r["n_new_tokens"] for r in nd),
        "by_category": by_cat,
    }


MISSING = "UNDECIDABLE_MISSING_ROWS"


def _missing(*summaries) -> list[str]:
    return [s["cell"] for s in summaries if s["n_rows"] == 0]


# ---------------------------------------------------------------------------
# ENDPOINT 1 -- does adding the vector raise the marker rate above the plain floor?
# ---------------------------------------------------------------------------
def endpoint1(rows, pass_name: str = "primary") -> tuple[dict, str]:
    base = cell_summary(rows, scored_cell("plain-baseline"), pass_name)
    add = cell_summary(rows, scored_cell("pooled-direction-add"), pass_name)
    miss = _missing(base, add)
    gap = add["marker_share_pct"] - base["marker_share_pct"]
    values = {"pass": pass_name, "plain_baseline": base, "pooled_direction_add": add,
              "marker_share_gap_pct_points": gap,
              "marker_share_gap_pct_points_nondegenerate":
                  add["marker_share_pct_nondegenerate"]
                  - base["marker_share_pct_nondegenerate"],
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if gap >= E1_GAP_PASS_PP:
        return values, "PASS_MARKER_RATE_GAP"
    if gap <= E1_GAP_REFUTE_PP:
        return values, "REFUTED_MARKER_RATE_NULL"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


# ---------------------------------------------------------------------------
# ENDPOINT 2 -- does adding the vector make the text longer?
# ---------------------------------------------------------------------------
def endpoint2(rows, pass_name: str = "primary") -> tuple[dict, str]:
    base = cell_summary(rows, scored_cell("plain-baseline"), pass_name)
    add = cell_summary(rows, scored_cell("pooled-direction-add"), pass_name)
    miss = _missing(base, add)
    b, a = base["mean_new_tokens"], add["mean_new_tokens"]
    ratio = (a / b) if b > 0 else float("nan")
    pct = (100.0 * (a - b) / b) if b > 0 else float("nan")
    values = {"pass": pass_name, "plain_baseline": base, "pooled_direction_add": add,
              "mean_new_tokens_baseline": b, "mean_new_tokens_added": a,
              "length_multiplier": ratio, "length_change_pct": pct,
              "length_multiplier_nondegenerate":
                  (add["mean_new_tokens_nondegenerate"]
                   / base["mean_new_tokens_nondegenerate"])
                  if base["mean_new_tokens_nondegenerate"] > 0 else float("nan"),
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if not (b > 0):
        return values, "UNDECIDABLE_ZERO_BASELINE_LENGTH"
    if ratio >= E2_LENGTH_MULTIPLIER_PASS:
        return values, "PASS_LENGTH_MULTIPLIER"
    if abs(pct) <= E2_LENGTH_REFUTE_PCT:
        return values, "REFUTED_LENGTH_NULL"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


# ---------------------------------------------------------------------------
# ENDPOINT 3 -- is the steered text still coherent?
# ---------------------------------------------------------------------------
def endpoint3(rows, pass_name: str = "primary") -> tuple[dict, str]:
    add = cell_summary(rows, scored_cell("pooled-direction-add"), pass_name)
    miss = _missing(add)
    share = add["coherent_share_pct"]
    rs = rows_of(rows, scored_cell("pooled-direction-add"), pass_name)
    values = {"pass": pass_name, "pooled_direction_add": add,
              "coherent_share_pct": share,
              "n_cut_off_at_cap": sum(1 for r in rs
                                      if "cap_no_stop" in (r["degenerate_reason"] or "")),
              "n_repeating": sum(1 for r in rs
                                 if "repeat_period" in (r["degenerate_reason"] or "")),
              "n_off_topic": sum(1 for r in rs if not r["on_topic"]),
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if share >= E3_COHERENT_PASS_PCT:
        return values, "PASS_COHERENT"
    if share < E3_COHERENT_REFUTE_PCT:
        return values, "REFUTED_COHERENCE_COLLAPSE"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


# ---------------------------------------------------------------------------
# ENDPOINT 4 -- does subtracting the vector lower the reasoning model's marker rate?
# ---------------------------------------------------------------------------
def endpoint4(rows, pass_name: str = "primary") -> tuple[dict, str]:
    ceil = cell_summary(rows, scored_cell("reasoning-ceiling"), pass_name)
    sub = cell_summary(rows, scored_cell("pooled-direction-subtract"), pass_name)
    miss = _missing(ceil, sub)
    drop = ceil["marker_share_pct"] - sub["marker_share_pct"]
    values = {"pass": pass_name, "reasoning_ceiling": ceil,
              "pooled_direction_subtract": sub,
              "marker_share_drop_pct_points": drop,
              "marker_share_drop_pct_points_nondegenerate":
                  ceil["marker_share_pct_nondegenerate"]
                  - sub["marker_share_pct_nondegenerate"],
              "think_tag_share_ceiling_pct": ceil["marker_think_tag_share_pct"],
              "think_tag_share_subtract_pct": sub["marker_think_tag_share_pct"],
              "phrase_share_ceiling_pct": ceil["marker_phrase_share_pct"],
              "phrase_share_subtract_pct": sub["marker_phrase_share_pct"],
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if drop >= E4_DROP_PASS_PP:
        return values, "PASS_SUPPRESSION"
    if drop <= E4_DROP_REFUTE_PP:
        return values, "REFUTED_NO_SUPPRESSION"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


# ---------------------------------------------------------------------------
# ENDPOINT 5 -- does the fitted vector beat both same-procedure controls?
# ---------------------------------------------------------------------------
def endpoint5(rows, pass_name: str = "primary") -> tuple[dict, str]:
    add = cell_summary(rows, scored_cell("pooled-direction-add"), pass_name)
    rnd = cell_summary(rows, scored_cell("random-vector-add"), pass_name)
    plac = cell_summary(rows, scored_cell("relabeled-placebo-add"), pass_name)
    miss = _missing(add, rnd, plac)
    g_rnd = add["marker_share_pct"] - rnd["marker_share_pct"]
    g_plac = add["marker_share_pct"] - plac["marker_share_pct"]
    values = {"pass": pass_name, "pooled_direction_add": add, "random_vector_add": rnd,
              "relabeled_placebo_add": plac,
              "gap_over_random_pct_points": g_rnd,
              "gap_over_placebo_pct_points": g_plac,
              "smallest_absolute_gap_pct_points": min(abs(g_rnd), abs(g_plac)),
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if g_rnd >= E5_MARGIN_PASS_PP and g_plac >= E5_MARGIN_PASS_PP:
        return values, "PASS_ABOVE_BOTH_CONTROLS"
    if abs(g_rnd) <= E5_MARGIN_REFUTE_PP or abs(g_plac) <= E5_MARGIN_REFUTE_PP:
        return values, "REFUTED_PERTURBATION_CONFOUND"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


# ---------------------------------------------------------------------------
# ENDPOINT 6 -- does either single-category vector act differently by prompt category?
# ---------------------------------------------------------------------------
def endpoint6(rows, pass_name: str = "primary") -> tuple[dict, str]:
    legs = {}
    miss: list[str] = []
    for arm in ("harmless-only-direction-add", "harmful-only-direction-add"):
        cell = scored_cell(arm)
        s = cell_summary(rows, cell, pass_name)
        if s["n_rows"] == 0:
            miss.append(cell)
        h_ful = s["by_category"]["harmful"]["marker_share_pct"]
        h_less = s["by_category"]["harmless"]["marker_share_pct"]
        legs[arm] = {"cell": cell,
                     "marker_share_on_harmful_pct": h_ful,
                     "marker_share_on_harmless_pct": h_less,
                     "n_harmful_rows": s["by_category"]["harmful"]["n_rows"],
                     "n_harmless_rows": s["by_category"]["harmless"]["n_rows"],
                     "absolute_gap_pct_points": abs(h_ful - h_less),
                     "summary": s}
    gaps = [legs[a]["absolute_gap_pct_points"] for a in legs]
    values = {"pass": pass_name, "legs": legs,
              "largest_absolute_gap_pct_points": max(gaps) if gaps else float("nan"),
              "missing_cells": miss}
    if miss:
        return values, MISSING
    if any(g > E6_GAP_REFUTE_PP for g in gaps):
        return values, "REFUTED_CATEGORY_CONFOUND"
    if all(g <= E6_GAP_PASS_PP for g in gaps):
        return values, "PASS_NOT_CATEGORY_SPECIFIC"
    return values, "INDETERMINATE_BETWEEN_CUTOFFS"


ENDPOINT_FUNCS = (endpoint1, endpoint2, endpoint3, endpoint4, endpoint5, endpoint6)

# How each endpoint's words were turned into arithmetic. Stated in the record, so the
# reading is auditable rather than remembered.
READINGS = {
    1: ("Marker share is the share of that cell's held-out generations holding at least one "
        "of the three named phrases, matched exactly and ignoring letter case, or either "
        "think-tag. Either tag counts, because the reasoning model's template supplies the "
        "opening one and the generation then carries only the close. All 128 rows of the cell "
        "count; the design names no exclusion here. The same numbers over rows that were "
        "neither cut off at the cap nor repeating are reported beside them, and decide "
        "nothing."),
    2: ("Length is the number of tokens the model produced before its stop token, so the stop "
        "token itself is not counted. The mean is over all 128 rows of the cell, because the "
        "design names no exclusion here; the coherence endpoint is the separate guard against "
        "runaway text. Refuting is read as the mean staying within 10 percent of the baseline "
        "mean in either direction."),
    3: ("Coherent means two fixed checks both pass. The repeated-text check fails a generation "
        "that reached the 1024-token cap with no stop token, or ends with a sequence of at "
        "most 6 tokens repeated 15 or more times inside its last 100 tokens. The off-topic "
        "check fails a generation that repeats none of its prompt's content words and does not "
        "open with a direct reply. The design names both checks but defines neither, so these "
        "definitions are implementation choices, recorded here and re-derivable from the saved "
        "text."),
    4: ("Same marker rule as endpoint 1, on the reasoning model's own rows. The share of rows "
        "holding a think-tag and the share holding a phrase are reported separately, because "
        "the reasoning model's template opens the tag itself, so both arms can carry a tag for "
        "a template reason rather than a steering one. Whether that makes the endpoint "
        "decidable is visible in those two numbers rather than hidden inside the total."),
    5: ("Both gaps are the fitted vector's marker share minus that control's, in percentage "
        "points. Passing needs both gaps at 15 or more. The named refuting outcome, sitting "
        "within 5 percentage points of either control, is read as an absolute gap of 5 or "
        "less against either one. A share far BELOW a control is neither: it fails the "
        "threshold without matching the refuting outcome, so it is reported as neither passed "
        "nor refuted."),
    6: ("For each single-category vector, the gap is the size of the difference between its "
        "marker share on the 64 harmful held-out prompts and on the 64 harmless held-out "
        "prompts, ignoring sign. Passing needs both gaps at 15 points or less; the refuting "
        "outcome needs either gap above 25 points."),
}


def threshold_numbers(n: int) -> dict:
    return {
        1: {"gap_pass_pct_points": E1_GAP_PASS_PP, "gap_refute_pct_points": E1_GAP_REFUTE_PP},
        2: {"length_multiplier_pass": E2_LENGTH_MULTIPLIER_PASS,
            "length_refute_within_pct": E2_LENGTH_REFUTE_PCT},
        3: {"coherent_pass_pct": E3_COHERENT_PASS_PCT,
            "coherent_refute_below_pct": E3_COHERENT_REFUTE_PCT},
        4: {"drop_pass_pct_points": E4_DROP_PASS_PP,
            "drop_refute_pct_points": E4_DROP_REFUTE_PP},
        5: {"margin_pass_pct_points": E5_MARGIN_PASS_PP,
            "margin_refute_pct_points": E5_MARGIN_REFUTE_PP},
        6: {"gap_pass_pct_points": E6_GAP_PASS_PP, "gap_refute_pct_points": E6_GAP_REFUTE_PP},
    }[n]


# ---------------------------------------------------------------------------
# HEAVY-IMPORT SECTION. Everything above this line is stdlib, so --selftest and --plan run
# on a machine with no numeric libraries, no model runtime and no card. A broken instrument
# must show before any box time is bought.
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
            f"FATAL: stage '{stage}' needs the model runtime and the numeric libraries "
            f"(torch, transformers, numpy); they are not importable here. --selftest and "
            f"--plan do not need them.")


def hf_token() -> tuple[str | None, str | None]:
    """The design names HF_KEY_ONE; the launcher forwards it into the run as HF_TOKEN. Both
    names are accepted and which one was used is recorded. The value is never printed."""
    for k in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v, k
    return None, None


def load_model(model_id: str):
    """16-bit weights, one model at a time.

    The plain model is licence-gated on the hub, so a token that has not accepted that
    licence fails here. The design lists that as the thing that halts this run before any
    data exists, so it is named loudly and nothing is substituted.
    """
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
            f"       If this is a 401 or 403, the token in HF_KEY_ONE has not accepted this "
            f"model's licence. The design halts here rather than substituting a model.")
    model.eval()
    if tok.chat_template is None:
        raise SystemExit(
            f"FATAL: {model_id} exposes no chat template. The design prompts every item "
            f"through one fixed single-turn template, so there is nothing to fall back to.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # so position -1 is the real last prompt token
    print(f"[load] blocks={model.config.num_hidden_layers} "
          f"width={model.config.hidden_size} vocab={model.config.vocab_size} "
          f"device={next(model.parameters()).device}", flush=True)
    return model, tok


def free_cuda() -> None:
    import gc
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model(model) -> None:
    """Free the weights before the next load.

    THE CALLER MUST ALSO DROP ITS OWN REFERENCES on the next line: the model, the tokenizer,
    and any hook object holding the model. Otherwise 16 GB stays resident and the second
    load runs the card out of memory. Moving the weights to the processor first means even a
    reference that was missed cannot keep them on the card.
    """
    try:
        model.cpu()
    except Exception:                   # noqa: BLE001
        pass
    free_cuda()


def blocks_of(model):
    return getattr(model, "model", model).layers


def render_prompt(tok, instruction: str) -> str:
    """One fixed single-turn template: the instruction alone, as the only user turn, through
    that model's own chat template. No system message, no extra wrapper."""
    return tok.apply_chat_template([{"role": "user", "content": instruction}],
                                   tokenize=False, add_generation_prompt=True)


def stop_token_ids(model, tok) -> list[int]:
    """Trap 2. The union of the generation config's end tokens, the tokenizer's end token,
    and the turn-end tokens that resolve, so a turn that ended properly is never recorded as
    having hit the cap."""
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
    """The running activation at the LAST PROMPT TOKEN, at every block, before any text is
    written. Returns 32-bit floats shaped [prompts, blocks + 1, width]; index 0 is the
    embedding output and index k+1 is the output of block k, which is exactly the tensor a
    hook on block k writes to.

    WHICH POSITION IS READ IS AN IMPLEMENTATION CHOICE: the design says "mean activation"
    without naming a position. The last prompt token is the position the model conditions
    its first written token on, and it is the position the sibling design names. Recorded in
    the extract record so the choice is auditable.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            res = model(**enc, output_hidden_states=True)
            hs = torch.stack([h[:, -1, :] for h in res.hidden_states], dim=1)
            out.append(hs.float().cpu().numpy())
            del res, enc
    free_cuda()
    return np.concatenate(out, axis=0)


class Intervention:
    """One forward hook on one block, firing only on the steps that write new text (trap 3).

    The tensor becomes h + sign * coefficient * vector. The vector is used unnormalized, as
    the design states, so the coefficient is a multiple of that vector's own length. Sign is
    +1 for the adding arms and -1 for the subtracting arm.
    """

    def __init__(self, model, layer: int, vec, sign: float, coeff: float):
        self.model, self.layer = model, layer
        self.sign, self.coeff = float(sign), float(coeff)
        v = torch.as_tensor(np.asarray(vec), dtype=torch.float32)
        n = float(torch.linalg.vector_norm(v))
        if n <= 0:
            raise ValueError(f"zero-length vector at block {layer}: nothing to add or subtract")
        self.vec = v
        self.norm = n
        self.handle = None
        self.n_applied = 0

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != 1:            # the prompt pass -- left untouched
            return output
        self.n_applied += 1
        h32 = hs.float() + (self.sign * self.coeff) * self.vec.to(hs.device, torch.float32)
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


class _NullCtx:
    """The two unchanged arms install no hook at all."""
    n_applied = 0
    norm = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# THE TWO PROMPT LISTS
# ---------------------------------------------------------------------------
def _load_source(src: dict):
    from datasets import load_dataset
    name, cfg, split = src["dataset"], src["config"], src["split"]
    try:
        return load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    except Exception as e:              # noqa: BLE001
        raise SystemExit(
            f"FATAL: could not load the prompt list {name} (split {split}): {e}\n"
            f"       The design names no source, so this public list is its GUESS choice. "
            f"The run halts here rather than substituting a different list, which would "
            f"silently change what was pre-registered.")


def build_prompts(path: Path) -> dict:
    """128 harmful and 128 harmless items, each split in half: 64 to fit the vector, 64 held
    back for scoring. Written once and then read, so both halves are auditable.

    HOW THE ITEMS ARE PICKED IS AN IMPLEMENTATION CHOICE: the design names neither source
    nor selection rule. The first 128 usable items in each list's own order are taken, and
    the split alternates, so even positions fit the vector and odd positions are held back.
    No seed is involved, so the two halves can be re-derived by anyone holding the lists.
    """
    if path.exists():
        return json.loads(path.read_text())
    try:
        import datasets  # noqa: F401
    except Exception as e:              # noqa: BLE001
        raise SystemExit(f"FATAL: the `datasets` package is required to build the prompt "
                         f"lists: {e}")

    sets = {}
    meta = {}
    for cat, src in (("harmful", HARMFUL_SOURCE), ("harmless", HARMLESS_SOURCE)):
        ds = _load_source(src)
        field = src["field"]
        if field not in ds.column_names:
            raise SystemExit(f"FATAL: {src['dataset']} has no column '{field}'; columns are "
                             f"{ds.column_names}. Halting rather than guessing a column.")
        picked = []
        for i in range(len(ds)):
            row = ds[i]
            text = (row.get(field) or "").strip()
            if not text:
                continue
            # The harmless list carries some items with an extra input block. Only
            # instruction-only items are used, so both categories are single instructions.
            if "input" in ds.column_names and (row.get("input") or "").strip():
                continue
            picked.append({"item_id": len(picked), "source_index": i, "category": cat,
                           "prompt": text})
            if len(picked) == N_PER_CATEGORY:
                break
        if len(picked) < N_PER_CATEGORY:
            raise SystemExit(f"FATAL: only {len(picked)} usable items in {src['dataset']}; "
                             f"the design needs {N_PER_CATEGORY} per category.")
        sets[cat] = picked
        meta[cat] = {"dataset": src["dataset"], "config": src["config"],
                     "split": src["split"], "field": field,
                     "n_source_rows": len(ds), "n_picked": len(picked)}

    fit, held = [], []
    for cat in ("harmful", "harmless"):
        for k, it in enumerate(sets[cat]):
            (fit if k % 2 == 0 else held).append(it)
    if len(fit) != 2 * N_FIT_PER_CATEGORY or len(held) != N_HELD_TOTAL:
        raise SystemExit(f"FATAL: split produced {len(fit)} fitting and {len(held)} held-out "
                         f"prompts; expected {2 * N_FIT_PER_CATEGORY} and {N_HELD_TOTAL}.")

    obj = {"sources": meta, "n_per_category": N_PER_CATEGORY,
           "n_fitting_per_category": N_FIT_PER_CATEGORY,
           "n_heldout_per_category": N_HELD_PER_CATEGORY,
           "selection": "first 128 usable items in each list's own order; even positions fit "
                        "the vector, odd positions are held back for scoring",
           "fitting": fit, "heldout": held,
           "note": "The two halves are disjoint by construction. No held-out prompt is used "
                   "to build any vector, and no fitting prompt is ever generated from."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return obj


# ---------------------------------------------------------------------------
# STAGE extract
# ---------------------------------------------------------------------------
def stage_extract(out_dir: Path, batch_size: int, force: bool) -> int:
    _require_torch("extract")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(out_dir / F_PROMPTS)
    fit = prompts["fitting"]
    n_h = sum(1 for it in fit if it["category"] == "harmful")
    print(f"[extract] fitting prompts: {len(fit)} ({n_h} harmful, {len(fit) - n_h} harmless); "
          f"held out for scoring: {len(prompts['heldout'])}", flush=True)

    arch = {}
    for role, model_id in (("reason", MODEL_REASON), ("plain", MODEL_PLAIN)):
        p = out_dir / F_ACTS.format(role=role)
        if p.exists() and not force:
            print(f"[extract] {role}: activations already on disk; skipping the load "
                  f"(--force to redo)", flush=True)
            arch.setdefault(role, {})
            continue
        model, tok = load_model(model_id)
        arch[role] = {"model": model_id, "n_blocks": int(model.config.num_hidden_layers),
                      "width": int(model.config.hidden_size),
                      "vocab_size": int(model.config.vocab_size),
                      "tokenizer_class": type(tok).__name__}
        rendered = [render_prompt(tok, it["prompt"]) for it in fit]
        t0 = time.time()
        acts = capture_last_token(model, tok, rendered, batch_size)
        np.save(p, acts)
        print(f"[extract] {role}: {acts.shape} in {time.time() - t0:.1f}s -> {p.name}",
              flush=True)
        # ONE AT A TIME. The next load must find the card empty; the `del` is load-bearing.
        unload_model(model)
        del model, tok
        free_cuda()

    a_r = np.load(out_dir / F_ACTS.format(role="reason"))
    a_p = np.load(out_dir / F_ACTS.format(role="plain"))
    cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "extract"}
    rec_path = out_dir / f"{provenance.cell_slug(cell)}.json"
    inputs = [provenance.describe_input(Path(DESIGN_RECORD), n_items=1),
              provenance.describe_input(out_dir / F_PROMPTS, n_items=len(fit)),
              provenance.describe_input(out_dir / F_ACTS.format(role="reason"),
                                        n_items=len(fit)),
              provenance.describe_input(out_dir / F_ACTS.format(role="plain"),
                                        n_items=len(fit))]
    rule = (
        "The vector is defined only if both checkpoints expose the SAME width and the SAME "
        "number of blocks, and only if that count is the 32 the design's layer choice is "
        "stated against, and only if the resulting vector has a non-zero length. Refuting "
        "outcome, stated before the values are known: differing widths or block counts, in "
        "which case carrying one model's vector into the other is not defined; a count other "
        "than 32, in which case 'the 16th block of 32' names no layer in this model; or a "
        "zero-length vector, in which case there is nothing to add or subtract. Any of those "
        "-> HALT_VECTOR_UNDEFINED, and no generation follows.")

    def halt(values, why):
        provenance.write_result(
            rec_path, cell=cell, inputs=inputs,
            metric="Mean-difference vector at the 16th block of 32, read at the last prompt "
                   "token over the 128 fitting prompts.",
            values=values, threshold={"same_width_required": True,
                                      "n_blocks_required": N_LAYERS_EXPECTED,
                                      "min_vector_length": 0.0},
            decision_rule=rule, decision="HALT_VECTOR_UNDEFINED",
            notes=why + " " + RERUN_NOTE)
        print(f"FATAL: {why}")
        return 2

    if a_r.shape[1:] != a_p.shape[1:]:
        return halt({"reason_shape": list(a_r.shape), "plain_shape": list(a_p.shape),
                     "same_shape": False, "architecture": arch},
                    "The two checkpoints do not share an activation shape, so carrying a "
                    "vector between them is not defined.")
    n_blocks, width = a_r.shape[1] - 1, a_r.shape[2]
    if n_blocks != N_LAYERS_EXPECTED or LAYER >= n_blocks:
        return halt({"n_blocks_observed": int(n_blocks), "width_observed": int(width),
                     "n_blocks_expected": N_LAYERS_EXPECTED, "layer": LAYER,
                     "architecture": arch},
                    f"This model reports {n_blocks} blocks, not the {N_LAYERS_EXPECTED} the "
                    f"design's layer choice is stated against, so 'the 16th block of 32' "
                    f"names no layer here.")

    # One vector per arm, all at the same block. Index LAYER + 1 is the output of block
    # LAYER, which is the tensor the hook on that block writes to.
    h_r = a_r[:, LAYER + 1, :]
    h_p = a_p[:, LAYER + 1, :]
    cats = [it["category"] for it in fit]
    idx_harmful = [i for i, c in enumerate(cats) if c == "harmful"]
    idx_harmless = [i for i, c in enumerate(cats) if c == "harmless"]

    pooled = (h_r.mean(axis=0) - h_p.mean(axis=0)).astype(np.float32)
    harmless_only = (h_r[idx_harmless].mean(axis=0)
                     - h_p[idx_harmless].mean(axis=0)).astype(np.float32)
    harmful_only = (h_r[idx_harmful].mean(axis=0)
                    - h_p[idx_harmful].mean(axis=0)).astype(np.float32)

    pooled_norm = float(np.linalg.norm(pooled))
    rng = np.random.default_rng(SEED_RANDOM_VECTOR)
    v = rng.standard_normal(width)
    random_matched = (v / np.linalg.norm(v) * pooled_norm).astype(np.float32)

    # The placebo: the same averaging procedure over randomly relabelled activations. All 256
    # readings, 128 from each model, are pooled and split at random into two halves of 128,
    # then one mean is subtracted from the other. So it tests whether the procedure alone
    # produces a direction.
    pool = np.concatenate([h_r, h_p], axis=0)
    order = np.random.default_rng(SEED_RELABEL).permutation(pool.shape[0])
    half = pool.shape[0] // 2
    relabeled_placebo = (pool[order[:half]].mean(axis=0)
                         - pool[order[half:]].mean(axis=0)).astype(np.float32)

    vectors = {"pooled": pooled, "harmless_only": harmless_only, "harmful_only": harmful_only,
               "random_matched": random_matched, "relabeled_placebo": relabeled_placebo}
    norms = {k: float(np.linalg.norm(v)) for k, v in vectors.items()}
    zero_length = [k for k, n in norms.items() if n <= 0]
    if zero_length:
        return halt({"vector_lengths": norms, "zero_length_vectors": zero_length,
                     "n_blocks_observed": int(n_blocks), "width_observed": int(width),
                     "architecture": arch},
                    f"These vectors have zero length: {zero_length}. There is nothing to add "
                    f"or subtract.")

    np.savez(out_dir / F_VECTORS, layer=np.array([LAYER], dtype=np.int32),
             n_blocks=np.array([n_blocks], dtype=np.int32),
             width=np.array([width], dtype=np.int32),
             mean_reason=h_r.mean(axis=0).astype(np.float32),
             mean_plain=h_p.mean(axis=0).astype(np.float32), **vectors)

    def cos(a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d > 0 else float("nan")

    values = {
        "models": {"plain": MODEL_PLAIN, "reason": MODEL_REASON},
        "architecture_observed": arch,
        "n_blocks_observed": int(n_blocks), "width_observed": int(width),
        "layer": LAYER, "layer_note": LAYER_NOTE,
        "activation_position": "last prompt token, before any text is written",
        "n_fitting_prompts": len(fit),
        "n_fitting_harmful": len(idx_harmful), "n_fitting_harmless": len(idx_harmless),
        "n_heldout_prompts": len(prompts["heldout"]),
        "vector_lengths": norms,
        "pushes_at_scored_coefficient": {k: COEFF_SCORED * n for k, n in norms.items()},
        "cosine_pooled_vs_harmless_only": cos(pooled, harmless_only),
        "cosine_pooled_vs_harmful_only": cos(pooled, harmful_only),
        "cosine_harmless_only_vs_harmful_only": cos(harmless_only, harmful_only),
        "cosine_pooled_vs_relabeled_placebo": cos(pooled, relabeled_placebo),
        "cosine_pooled_vs_random_matched": cos(pooled, random_matched),
        "mean_activation_length_reason": float(np.linalg.norm(h_r.mean(axis=0))),
        "mean_activation_length_plain": float(np.linalg.norm(h_p.mean(axis=0))),
        "coefficient_grid": list(COEFF_GRID), "coefficient_scored": COEFF_SCORED,
        "random_vector_seed": SEED_RANDOM_VECTOR, "relabel_seed": SEED_RELABEL,
        "prompt_sources": prompts["sources"],
        "hf_token_env": hf_token()[1],
    }
    p = provenance.write_result(
        rec_path, cell=cell, inputs=inputs,
        metric="Mean-difference vector at the 16th block of 32, read at the last prompt token "
               "over the 128 fitting prompts: the reasoning model's mean minus the plain "
               "model's, pooled and per prompt category, with a length-matched random vector "
               "and a relabelled-placebo vector built the same way.",
        values=values,
        threshold={"same_width_required": True, "n_blocks_required": N_LAYERS_EXPECTED,
                   "min_vector_length": 0.0},
        decision_rule=rule, decision="VECTOR_DEFINED",
        notes=" ".join([
            "The two seeds named here are implementation constants, not design numbers: the "
            "design names no seed for the random vector or for the relabelling.",
            SIZE_NOTE, LAYER_NOTE, RERUN_NOTE]))
    print(f"[extract] VECTOR_DEFINED lengths={ {k: round(v, 2) for k, v in norms.items()} } "
          f"-> {p}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# STAGE generate
# ---------------------------------------------------------------------------
def _decode_rows(tok, job, batch_items, rendered, enc, out_ids, stops, opens,
                 tags_kept: bool, specials: tuple) -> list[dict]:
    """One batch of generations, one row each, saved verbatim."""
    rows = []
    stop_set = set(stops)
    n_in = enc["input_ids"].shape[1]
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    new = out_ids[:, n_in:]

    for j, it in enumerate(batch_items):
        ids = new[j].tolist()
        stop_at = next((k for k, t in enumerate(ids) if t in stop_set), None)
        kept = ids if stop_at is None else ids[:stop_at]
        stopped = stop_at is not None
        text = scored_text(tok, kept, tags_kept, specials)
        raw = tok.decode(ids, skip_special_tokens=False)
        degen, why = is_degenerate(len(kept), stopped, kept)
        mk = markers_of(text, opens)
        coh = coherence_of(it["prompt"], text, degen)
        rows.append({
            "seed_label": SEED, "pipeline": PIPELINE,
            "cell": job["cell"], "arm": job["arm"], "pass": job["pass"],
            "role": job["role"], "model": job["model"],
            "vector": job["vector"], "sign": job["sign"], "coeff": job["coeff"],
            "layer": LAYER, "steer_prompt_positions": False,
            "decode": job["decode"], "decode_seed": job["decode_seed"],
            "temperature": job["temperature"], "top_p": job["top_p"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "item_id": it["item_id"], "source_index": it["source_index"],
            "category": it["category"],
            "prompt": it["prompt"], "rendered_prompt": rendered[j],
            "prompt_opens_think": bool(opens),
            "think_tags_survive_clean_decode": bool(tags_kept),
            "n_prompt_tokens": int(prompt_lens[j]),
            # The generation, verbatim: `text` is what the scorers read, up to but not
            # including the stop token; `text_raw` is every new token decoded with nothing
            # hidden, including the stop token and anything after it.
            "text": text, "text_raw": raw,
            "n_new_tokens": len(kept), "n_new_tokens_including_stop": len(ids),
            "stopped_on_stop_token": bool(stopped),
            "degenerate": bool(degen), "degenerate_reason": why,
            "marker": bool(mk["marker"]), "marker_phrases": mk["marker_phrases"],
            "marker_think_tag": bool(mk["marker_think_tag"]),
            "tag_open_in_generation": bool(mk["tag_open_in_generation"]),
            "tag_close_in_generation": bool(mk["tag_close_in_generation"]),
            "think_kind": mk["think_kind"], "think_well_formed": bool(mk["think_well_formed"]),
            "think_text_chars": mk["think_text_chars"],
            "implicit_open": bool(mk["implicit_open"]),
            "coherent": bool(coh["coherent"]), "on_topic": bool(coh["on_topic"]),
            "n_prompt_words_repeated": coh["n_prompt_words_repeated"],
            "prompt_words_repeated": coh["prompt_words_repeated"],
            "reply_opener": coh["reply_opener"],
        })
    return rows


def run_role(role: str, out_dir: Path, prompts: dict, npz, batch_size: int):
    model_id = MODEL_PLAIN if role == "plain" else MODEL_REASON
    jobs = generation_jobs(role)
    held = prompts["heldout"]
    model, tok = load_model(model_id)
    if int(model.config.hidden_size) != int(npz["width"][0]):
        unload_model(model)
        raise SystemExit(
            f"FATAL: {model_id} is {model.config.hidden_size} wide, but the saved vectors are "
            f"{int(npz['width'][0])} wide; the vector cannot be carried into this model.")
    stops = stop_token_ids(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rendered_all = [render_prompt(tok, it["prompt"]) for it in held]
    opens = prompt_opens_think(rendered_all[0])
    tags_kept = tags_survive_clean_decode(tok)
    specials = specials_to_strip(tok)
    print(f"[generate] {role}: {len(jobs)} cells x {len(held)} prompts = "
          f"{len(jobs) * len(held)} rows | prompt_opens_think={opens} | "
          f"think_tags_survive_clean_decode={tags_kept} | stop_ids={stops}", flush=True)

    path = out_dir / F_GENS.format(role=role)
    written, notes = 0, []
    t_start = time.time()
    with open(path, "w") as fh:
        for ji, job in enumerate(jobs):
            if job["vector"] is None:
                ctx = _NullCtx()
            else:
                ctx = Intervention(model, LAYER, npz[job["vector"]], job["sign"],
                                   float(job["coeff"]))
            t_cell, n_cell = time.time(), 0
            with ctx:
                torch.manual_seed(int(job["decode_seed"]))   # once per cell and decode
                for i in range(0, len(held), batch_size):
                    chunk = held[i:i + batch_size]
                    bp = rendered_all[i:i + batch_size]
                    enc = tok(bp, return_tensors="pt", padding=True,
                              add_special_tokens=False).to(model.device)
                    kw = {"max_new_tokens": MAX_NEW_TOKENS, "use_cache": True,
                          "pad_token_id": pad_id, "eos_token_id": stops}
                    if job["decode"] == "greedy":
                        kw["do_sample"] = False
                    else:
                        kw.update(do_sample=True, temperature=job["temperature"],
                                  top_p=job["top_p"])
                    with torch.no_grad():
                        out_ids = model.generate(**enc, **kw)
                    for r in _decode_rows(tok, job, chunk, bp, enc, out_ids, stops, opens,
                                          tags_kept, specials):
                        fh.write(json.dumps(r) + "\n")
                        written += 1
                        n_cell += 1
                    del out_ids, enc
            fh.flush()
            notes.append({"cell": job["cell"], "pass": job["pass"],
                          "decode": job["decode"], "decode_seed": job["decode_seed"],
                          "coefficient": job["coeff"],
                          "vector_length": round(float(getattr(ctx, "norm", 0.0)), 4),
                          "n_rows": n_cell, "hook_forward_calls": int(ctx.n_applied),
                          "seconds": round(time.time() - t_cell, 1)})
            el = time.time() - t_start
            done = ji + 1
            print(f"[generate] {role} {done}/{len(jobs)} {job['cell']} ({job['pass']}, "
                  f"{job['decode']} seed {job['decode_seed']}): {n_cell} rows in "
                  f"{time.time() - t_cell:.0f}s | elapsed {el / 60:.1f}m | "
                  f"eta {el / done * (len(jobs) - done) / 60:.1f}m", flush=True)
    # The hook object holds a reference to the model, so it is cleared FIRST. Dropping only
    # the model would leave 16 GB on the card for the next load.
    ctx = None
    unload_model(model)
    del model, tok, ctx
    free_cuda()
    return path, written, notes


def stage_generate(out_dir: Path, batch_size: int, roles: tuple[str, ...]) -> int:
    _require_torch("generate")
    prompts = json.loads((out_dir / F_PROMPTS).read_text())
    npz = np.load(out_dir / F_VECTORS)
    rc = 0
    for role in roles:
        path, n_rows, notes = run_role(role, out_dir, prompts, npz, batch_size)
        exp = expected_rows(role)
        rows = _read_jsonl(path)
        cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "generate", "role": role}
        per_pass = {}
        for pname in ("primary", "sensitivity", "robustness"):
            rs = [r for r in rows if r["pass"] == pname]
            per_pass[pname] = {"n_rows": len(rs),
                               "n_degenerate": sum(1 for r in rs if r["degenerate"]),
                               "n_marker": sum(1 for r in rs if r["marker"])}
        values = {
            "model": MODEL_PLAIN if role == "plain" else MODEL_REASON,
            "n_cells": len(generation_jobs(role)), "n_heldout_prompts": N_HELD_TOTAL,
            "n_rows_expected": exp, "n_rows_written": n_rows,
            "per_pass": per_pass, "per_cell": notes,
            "layer": LAYER, "coefficient_scored": COEFF_SCORED,
            "coefficient_grid": list(COEFF_GRID),
            "max_new_tokens": MAX_NEW_TOKENS, "batch_size": batch_size,
            "robustness_seeds": list(ROBUST_SEEDS), "temperature": TEMPERATURE, "top_p": TOP_P,
            "steer_prompt_positions": False,
            "prompt_opens_think": bool(rows[0]["prompt_opens_think"]) if rows else None,
            "think_tags_survive_clean_decode":
                bool(rows[0]["think_tags_survive_clean_decode"]) if rows else None,
        }
        decision = "ROWS_COMPLETE" if n_rows == exp else "INCOMPLETE_CELLS"
        p = provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell,
            inputs=[provenance.describe_input(Path(DESIGN_RECORD), n_items=1),
                    provenance.describe_input(out_dir / F_PROMPTS, n_items=N_HELD_TOTAL),
                    provenance.describe_input(out_dir / F_VECTORS, n_items=len(VECTOR_NAMES)),
                    provenance.describe_input(path, n_items=n_rows)],
            metric="Generations per pre-registered cell over the 128 held-out prompts: greedy "
                   "decoding for the scored pass and for the unscored coefficient check, and "
                   "three sampled generations at temperature 0.7 and top-p 0.9, seeds 0, 1 and "
                   "2, for the robustness pass. Every generation capped at 1024 new tokens, "
                   "each model under its own chat template.",
            values=values, threshold={"n_rows_expected": exp},
            decision_rule="This stage decides only whether the pre-registered rows exist; the "
                          "endpoints decide the hypothesis. Rows written must equal cells "
                          "times 128 held-out prompts. Refuting outcome for this stage: fewer "
                          "rows than that -> INCOMPLETE_CELLS, and every endpoint reading this "
                          "model is then computed over a truncated design.",
            decision=decision,
            notes=" ".join([ROW_NOTE, SIZE_NOTE, LAYER_NOTE, RERUN_NOTE]))
        print(f"[generate] {role}: {decision} {n_rows}/{exp} rows -> {p}", flush=True)
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
    """Stdlib only: it reads the prompt list and the two generation files. No model, no card."""
    prompts_p = out_dir / F_PROMPTS
    gens = {r: out_dir / F_GENS.format(role=r) for r in ROLES}
    missing = [str(p) for p in [prompts_p, *gens.values()] if not p.exists()]
    if missing:
        print("FATAL: the endpoints stage is missing its inputs: " + ", ".join(missing))
        return 2
    rows_plain, rows_reason = _read_jsonl(gens["plain"]), _read_jsonl(gens["reason"])
    rows = rows_plain + rows_reason
    print(f"[endpoints] {len(rows_plain)} plain-model rows + {len(rows_reason)} "
          f"reasoning-model rows", flush=True)
    if not rows_plain or not rows_reason:
        print("FATAL: one model produced no rows; the endpoints would report over a "
              "truncated design.")
        return 2

    base_inputs = [
        provenance.describe_input(Path(DESIGN_RECORD), n_items=1),
        provenance.describe_input(prompts_p, n_items=N_HELD_TOTAL),
        provenance.describe_input(gens["plain"], n_items=len(rows_plain)),
        provenance.describe_input(gens["reason"], n_items=len(rows_reason)),
    ]
    written = []
    for n, fn in enumerate(ENDPOINT_FUNCS, start=1):
        values, decision = fn(rows, "primary")
        spec = ENDPOINTS_SPEC[n - 1]
        cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "endpoints", "endpoint": n,
                "pass": "primary"}
        rule = (f"THRESHOLD (design, word for word): {spec['threshold']} "
                f"REFUTING OUTCOME (design, word for word): {spec['refuting_outcome']} "
                f"READING: {READINGS[n]}")
        p = provenance.write_result(
            out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=base_inputs,
            metric=spec["metric"], values=values,
            threshold={"threshold_text": spec["threshold"],
                       "refuting_outcome_text": spec["refuting_outcome"],
                       "numeric": threshold_numbers(n)},
            decision_rule=rule, decision=decision,
            notes=" ".join([ROW_NOTE, LAYER_NOTE, RERUN_NOTE]))
        written.append((f"E{n}", decision))
        print(f"[endpoints] E{n}: {decision} -> {Path(p).name}", flush=True)

    # The robustness pass. The design adds it but scores its thresholds on the greedy pass,
    # so the same three arithmetic tests are reported here and decide nothing.
    rob_cells = [scored_cell(a) for a in ROBUSTNESS_ARMS]
    rob_values = {"pass": "robustness", "decode": "sample", "temperature": TEMPERATURE,
                  "top_p": TOP_P, "seeds": list(ROBUST_SEEDS),
                  "pooled_over_seeds": {}, "per_seed": {}}
    v1, d1 = endpoint1(rows, "robustness")
    v2, d2 = endpoint2(rows, "robustness")
    v3, d3 = endpoint3(rows, "robustness")
    rob_values["pooled_over_seeds"] = {
        "endpoint_1_marker_gap": {"values": v1, "same_arithmetic_would_say": d1},
        "endpoint_2_length": {"values": v2, "same_arithmetic_would_say": d2},
        "endpoint_3_coherence": {"values": v3, "same_arithmetic_would_say": d3}}
    for s in ROBUST_SEEDS:
        rob_values["per_seed"][str(s)] = {
            c: cell_summary(rows, c, "robustness", decode_seed=s) for c in rob_cells}
    cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "robustness", "pass": "robustness"}
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=base_inputs,
        metric="The robustness pass: three sampled generations per held-out prompt at "
               "temperature 0.7 and top-p 0.9, seeds 0, 1 and 2, for the unchanged plain model "
               "(plain-baseline) and the arm that adds the fitted vector to it "
               "(pooled-direction-add), with the marker, length and coherence arithmetic of "
               "endpoints 1 to 3 recomputed on those rows.",
        values=rob_values,
        threshold={"note": "No threshold applies. The design states its cutoffs for the "
                           "greedy scored pass."},
        decision_rule="This record carries no decision. The design's cutoffs are scored on the "
                      "greedy pass, so the numbers here are reported for comparison only. What "
                      "the same arithmetic would say on these rows is recorded per endpoint "
                      "inside values, labelled as such, so no reader mistakes it for the "
                      "pre-registered call.",
        decision="UNSCORED_ROBUSTNESS_PASS",
        notes=" ".join([ROW_NOTE, RERUN_NOTE]))
    print("[endpoints] robustness pass recorded, unscored", flush=True)

    # The unscored coefficient check: 1 and 4 times the fitted vector's length.
    sens = {}
    for arm in SENSITIVITY_ARMS:
        for c in COEFF_GRID:
            key = cell_key(arm, c)
            pname = "primary" if c == COEFF_SCORED else "sensitivity"
            sens[key] = {"pass": pname, "coefficient": c,
                         "summary": cell_summary(rows, key, pname)}
    baselines = {c: cell_summary(rows, c, "primary")
                 for c in (scored_cell("plain-baseline"), scored_cell("reasoning-ceiling"))}
    cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "sensitivity", "pass": "sensitivity"}
    provenance.write_result(
        out_dir / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=base_inputs,
        metric="The unscored coefficient check: marker share, mean length and coherent share "
               "for the add arm and the subtract arm at 1, 2 and 4 times the fitted vector's "
               "own length, all at the 16th block of 32, beside the two unchanged arms.",
        values={"cells": sens, "unchanged_arms": baselines,
                "coefficient_grid": list(COEFF_GRID), "coefficient_scored": COEFF_SCORED,
                "layer": LAYER},
        threshold={"note": "No threshold applies. The design keeps 1 and 4 times as an "
                           "unscored check and scores only 2 times."},
        decision_rule="This record carries no decision, because the design says the scored "
                      "endpoints use only the 2 times point and keeps 1 and 4 times as an "
                      "unscored check. It exists so a reader can see whether the scored point "
                      "sits in a regime where the text is still coherent and the marker rate "
                      "still moves, rather than at a tuned coordinate.",
        decision="UNSCORED_COEFFICIENT_CHECK",
        notes=" ".join([SIZE_NOTE, LAYER_NOTE, RERUN_NOTE]))
    print("[endpoints] coefficient check recorded, unscored", flush=True)
    print("[endpoints] " + " | ".join(f"{k}={d}" for k, d in written), flush=True)
    return 0


# ---------------------------------------------------------------------------
# --plan : what the run will produce, before it is bought
# ---------------------------------------------------------------------------
def stage_plan() -> int:
    print(f"seed {SEED} pipeline {PIPELINE} | design {DESIGN_RECORD}")
    print(f"models: {MODEL_PLAIN} (plain), {MODEL_REASON} (reasoning)")
    print(f"prompts: {N_PER_CATEGORY} harmful + {N_PER_CATEGORY} harmless, split "
          f"{N_FIT_PER_CATEGORY} fitting / {N_HELD_PER_CATEGORY} held out per category")
    print(f"sources (design GUESS): {HARMFUL_SOURCE['dataset']} (harmful), "
          f"{HARMLESS_SOURCE['dataset']} (harmless)")
    print(f"block {LAYER} of {N_LAYERS_EXPECTED} | coefficient grid {list(COEFF_GRID)} | "
          f"scored at {COEFF_SCORED} | cap {MAX_NEW_TOKENS} new tokens")
    for role in ROLES:
        jobs = generation_jobs(role)
        print(f"  {role} model: {len(jobs)} cells x {N_HELD_TOTAL} prompts = "
              f"{len(jobs) * N_HELD_TOTAL} rows")
        for j in jobs:
            print(f"    {j['pass']:11s} {j['cell']:34s} vector={str(j['vector']):18s} "
                  f"sign={j['sign']:+.0f} decode={j['decode']}/{j['decode_seed']}")
    by_pass = {}
    for j in generation_jobs():
        by_pass[j["pass"]] = by_pass.get(j["pass"], 0) + N_HELD_TOTAL
    for k in ("primary", "sensitivity", "robustness"):
        print(f"rows {k}: {by_pass.get(k, 0)}")
    print(f"rows total: {expected_rows()}")
    print(f"budget_usd (design GUESS)={BUDGET_USD} cap_usd (record)={CAP_USD}")
    return 0


# ---------------------------------------------------------------------------
# SELFTEST -- stdlib only, no model, no card. A broken instrument must show on the
# processor before any box time is bought.
# ---------------------------------------------------------------------------
def _plant(cell, pass_name="primary", category="harmful", item_id=0, marker=False,
           n_tokens=100, degenerate=False, coherent=True, decode_seed=0,
           marker_phrases=None, think_tag=False, on_topic_flag=True, reason=""):
    return {"cell": cell, "pass": pass_name, "category": category, "item_id": item_id,
            "marker": bool(marker), "marker_phrases": marker_phrases or [],
            "marker_think_tag": bool(think_tag), "n_new_tokens": n_tokens,
            "degenerate": bool(degenerate), "degenerate_reason": reason,
            "coherent": bool(coherent), "on_topic": bool(on_topic_flag),
            "decode_seed": decode_seed}


def _cellrows(cell, n, pass_name="primary", n_marker=0, n_tokens=100, n_coherent=None,
              n_degenerate=0, category_split=True):
    """n rows for one cell, half harmful and half harmless, with the first n_marker of each
    half carrying a marker."""
    rows = []
    n_coherent = n if n_coherent is None else n_coherent
    cats = ("harmful", "harmless") if category_split else ("harmful",)
    per = n // len(cats)
    for ci, cat in enumerate(cats):
        for i in range(per):
            k = ci * per + i
            rows.append(_plant(cell, pass_name, cat, item_id=k,
                               marker=(i < n_marker // len(cats)),
                               n_tokens=n_tokens,
                               degenerate=(k < n_degenerate),
                               coherent=(k < n_coherent)))
    return rows


def _selftest() -> int:
    fails: list[str] = []
    n_checks = [0]
    sections: list[tuple[str, int, int]] = []

    def chk(cond, msg):
        n_checks[0] += 1
        if not cond:
            fails.append(msg)

    def sect(label):
        """Open a part of the selftest, so the log says what was covered, not only how much."""
        sections.append((label, len(fails), n_checks[0]))

    # -- 0. the quoted design text still matches the design record ---------------------
    sect("the quoted design text still matches the design record")
    rec = _HERE / DESIGN_RECORD
    if rec.exists():
        d = json.loads(rec.read_text())["design"]
        eps = d["endpoints"]
        chk(len(eps) == len(ENDPOINTS_SPEC),
            f"design has {len(eps)} endpoints, this file quotes {len(ENDPOINTS_SPEC)}")
        for i, (a, b) in enumerate(zip(eps, ENDPOINTS_SPEC), start=1):
            for k in ("metric", "threshold", "refuting_outcome"):
                chk(a[k] == b[k], f"endpoint {i} {k} differs from the design record")
        chk(d["budget_usd"] == BUDGET_USD, "budget differs from the design record")
        chk(len(d["arms"]) == len(ARMS),
            f"design lists {len(d['arms'])} arms, this file builds {len(ARMS)}")
    else:
        print(f"note: {DESIGN_RECORD} is not on this machine, so the quoted design text "
              f"could not be checked against it")

    # -- 1. the think-tag parser, including the auto-opened tag ------------------------
    sect("the think-tag parser, including the auto-opened tag")
    p = parse_think("<think>weigh it up</think>Answer: 5", False)
    chk(p["well_formed"] and p["kind"] == "paired", f"explicit pair read as {p['kind']}")
    chk(p["think_text"] == "weigh it up" and p["post_text"] == "Answer: 5", f"spans: {p}")

    # THE TRAP: the chat template auto-opens the tag, so only the close appears.
    tmpl = "<|begin_of_text|>user: hi<|Assistant|><think>\n"
    chk(prompt_opens_think(tmpl), "an auto-opened tag in the prompt was not detected")
    chk(not prompt_opens_think("<|begin_of_text|>user: hi\nassistant:\n"),
        "a prompt that opens nothing was read as opening a tag")
    chk(not prompt_opens_think("<think>a</think> now answer:"),
        "a prompt whose tag is already closed was read as leaving one open")
    g = "step one, step two</think>The answer is 5."
    p = parse_think(g, True)
    chk(p["kind"] == "implicit_paired" and p["well_formed"],
        f"closing tag only, with the prompt opening it, read as {p['kind']}")
    chk(p["think_text"] == "step one, step two" and p["post_text"] == "The answer is 5.",
        f"implicit spans: {p}")
    chk(markers_of(g, True)["marker"] and markers_of(g, True)["marker_think_tag"],
        "a closing tag alone did not count as a reasoning marker")
    chk(markers_of(g, False)["marker_think_tag"],
        "a closing tag alone must still count as a tag when the prompt opened nothing")
    p = parse_think(g, False)
    chk(p["kind"] == "stray_close" and not p["well_formed"],
        f"a closing tag with nothing opening it read as {p['kind']}")
    p = parse_think("<think>cut off mid thought", True)
    chk(p["kind"] == "open_unclosed" and not p["well_formed"],
        f"an unclosed block read as {p['kind']}")
    p = parse_think("no tags at all here", True)
    chk(p["kind"] == "open_unclosed" and p["n_open_effective"] == 1
        and p["think_text"] == "no tags at all here",
        f"a tagless generation from an auto-opening prompt read as {p['kind']}")
    chk(not markers_of("no tags at all here", True)["marker_think_tag"],
        "a tag the template supplied is not a tag the model wrote: no marker here")
    p = parse_think("no tags at all here", False)
    chk(p["kind"] == "no_tags" and p["n_open_effective"] == 0,
        f"a tagless generation from a plain prompt read as {p['kind']}")
    p = parse_think("<think>a</think>mid<think>b</think>", False)
    chk(p["kind"] == "malformed" and not p["well_formed"], f"two blocks read as {p['kind']}")

    # -- 2. marker scoring by exact match ---------------------------------------------
    sect("marker scoring by exact match")
    m = markers_of("Wait, that is wrong. Let me reconsider the second step.")
    chk(m["marker"] and set(m["marker_phrases"]) == {"wait,", "let me reconsider"},
        f"phrase matching, ignoring case: {m['marker_phrases']}")
    chk(not markers_of("The answer is 5.")["marker"], "a plain answer counted as a marker")
    chk(not markers_of("I had to wait for the bus")["marker"],
        "'wait' without its comma must not count: the design's phrase is 'wait,'")
    chk(markers_of("Alternatively, take the train.")["marker_phrases"] == ["alternatively,"],
        "the third named phrase was missed")
    chk(markers_of("<think>", False)["marker"], "an opening tag alone did not count")

    # -- 2b. the decode path must not delete the tags the marker rule reads ------------
    sect("the decode path must not delete the tags the marker rule reads")
    class _StubTok:
        """A stand-in tokenizer. `hidden` is what its clean decode deletes, which is how a
        real tokenizer that registers the think tags as special behaves."""

        def __init__(self, hidden, declared):
            self.hidden, self.all_special_tokens = hidden, declared

        def encode(self, text, add_special_tokens=False):
            return list(text)                      # one id per character; enough to probe

        def decode(self, ids, skip_special_tokens=False):
            text = "".join(ids)
            if skip_special_tokens:
                for t in self.hidden:
                    text = text.replace(t, "")
            return text

    hides = _StubTok((OPEN_TAG, CLOSE_TAG, "<|end_of_text|>"),
                     ["<|end_of_text|>", OPEN_TAG, CLOSE_TAG])
    keeps = _StubTok(("<|end_of_text|>",), ["<|end_of_text|>"])
    chk(not tags_survive_clean_decode(hides),
        "a tokenizer whose clean decode deletes the tags was not detected")
    chk(tags_survive_clean_decode(keeps),
        "a tokenizer that keeps the tags was reported as deleting them")
    chk(specials_to_strip(hides) == ("<|end_of_text|>",),
        f"the think tags must never be stripped: {specials_to_strip(hides)}")
    ids = list(f"{OPEN_TAG}weigh it up{CLOSE_TAG}Answer: 5<|end_of_text|>")
    t_bad = scored_text(hides, ids, True, specials_to_strip(hides))
    chk(CLOSE_TAG not in t_bad,
        "the check itself is wrong if the clean path keeps tags this tokenizer deletes")
    t_ok = scored_text(hides, ids, False, specials_to_strip(hides))
    chk(t_ok == f"{OPEN_TAG}weigh it up{CLOSE_TAG}Answer: 5" and markers_of(t_ok)["marker"],
        f"the fallback decode must keep both tags and drop the rest: {t_ok!r}")
    t_keep = scored_text(keeps, ids, True, specials_to_strip(keeps))
    chk(t_keep == f"{OPEN_TAG}weigh it up{CLOSE_TAG}Answer: 5",
        f"the clean decode must be used when it keeps the tags: {t_keep!r}")

    # -- 3. the repeated-or-truncated-text detector ------------------------------------
    sect("the repeated-or-truncated-text detector")
    d, why = is_degenerate(MAX_NEW_TOKENS, False, list(range(MAX_NEW_TOKENS)))
    chk(d and why == "cap_no_stop", f"cap reached with no stop token: {why}")
    d, why = is_degenerate(MAX_NEW_TOKENS, True, list(range(MAX_NEW_TOKENS)))
    chk(not d, f"a turn that ended on its stop token must not be flagged: {why}")
    d, why = is_degenerate(300, True, list(range(200)) + [7, 8] * 20)
    chk(d and why.startswith("repeat_period2"), f"a two-token loop: {why}")
    d, why = is_degenerate(300, True, list(range(200)) + [9] * 20)
    chk(d and why.startswith("repeat_period1"), f"a single repeated token: {why}")
    d, why = is_degenerate(300, True, list(range(200)) + [7, 8] * 7)
    chk(not d, f"14 repeats is under the 15 the rule names: {why}")
    d, why = is_degenerate(MAX_NEW_TOKENS, False, list(range(900)) + [3] * 124)
    chk(d and "cap_no_stop" in why and "repeat_period1" in why,
        f"both reasons must be recorded when both fire: {why}")
    hit, per, reps = repeat_degenerate([1, 2, 3] * 40)
    chk(hit and per == 3 and reps >= DEGEN_REPEAT_MIN, f"three-token loop: {per}, {reps}")

    # -- 4. exact-match extraction: the marker phrases and the think block -------------
    sect("exact-match extraction: the marker phrases and the think block")
    # This design scores no numeric answer, and says so: it does not test whether answers
    # become more correct. So the exact-match scorer it does have is checked here, which is
    # the fixed phrase list and the think block the tags delimit.
    p = parse_think("prelude<think>inside</think>after", False)
    chk((p["pre_text"], p["think_text"], p["post_text"]) == ("prelude", "inside", "after"),
        f"think-block extraction: {p}")
    chk(markers_of("WAIT, hold on")["marker_phrases"] == ["wait,"],
        "exact phrase matching must ignore letter case")
    chk(markers_of("waiting, then")["marker_phrases"] == [],
        "exact phrase matching must not fire on a longer word")
    chk(parse_think("</think>only after", True)["think_text"] == "",
        "an empty think block must extract as empty, not as the whole text")

    # -- 5. the coherence check: repeated text and off topic ---------------------------
    sect("the coherence check: repeated text and off topic")
    prompt = "Write a tutorial on how to build a wooden bookshelf"
    ok = coherence_of(prompt, "Here is a tutorial for a wooden bookshelf. Step one: cut.",
                      False)
    chk(ok["coherent"] and ok["n_prompt_words_repeated"] >= 2, f"an on-topic reply: {ok}")
    bad = coherence_of(prompt, "purple purple purple", False)
    chk(not bad["coherent"] and not bad["on_topic"],
        f"a reply sharing nothing with its prompt: {bad}")
    rep = coherence_of(prompt, "bookshelf bookshelf bookshelf", True)
    chk(not rep["coherent"] and rep["on_topic"],
        f"repeated text is incoherent even when on topic: {rep}")
    ref = coherence_of("Explain how to hotwire a car",
                       "I cannot help with that request.", False)
    chk(ref["coherent"] and ref["reply_opener"] == "i cannot",
        f"a plain refusal is a direct reply, not off topic: {ref}")
    chk(content_words("The and about with") == set(),
        "common words must not count as content words")

    # -- 6. the cells and the row counts ----------------------------------------------
    sect("the cells and the row counts")
    jobs = generation_jobs()
    chk(len(jobs) == 18, f"expected 18 cells in all, got {len(jobs)}")
    chk(sum(1 for j in jobs if j["pass"] == "primary") == 8,
        "the scored pass must have one cell per arm, 8 in all")
    chk(sum(N_HELD_TOTAL for j in jobs if j["pass"] == "primary") == 1024,
        "the scored pass must be 1024 generations")
    chk(sum(N_HELD_TOTAL for j in jobs if j["pass"] == "robustness") == 768,
        "the robustness pass must be 768 generations")
    chk(sum(N_HELD_TOTAL for j in jobs if j["pass"] == "sensitivity") == 512,
        "the unscored coefficient check must be 512 generations")
    chk(expected_rows() == 2304, f"total rows {expected_rows()}, expected 2304")
    chk(expected_rows("plain") == 1792 and expected_rows("reason") == 512,
        f"per model: {expected_rows('plain')} and {expected_rows('reason')}")
    chk(scored_cell("pooled-direction-add") == "pooled-direction-add_c2.0",
        f"scored cell name: {scored_cell('pooled-direction-add')}")
    chk(scored_cell("plain-baseline") == "plain-baseline",
        "an unchanged arm must carry no coefficient in its name")
    chk({j["cell"] for j in jobs if j["pass"] == "sensitivity"} ==
        {"pooled-direction-add_c1.0", "pooled-direction-add_c4.0",
         "pooled-direction-subtract_c1.0", "pooled-direction-subtract_c4.0"},
        "the coefficient check must cover the two arms that use the fitted vector")
    chk(all(j["sign"] == -1.0 for j in jobs if j["arm"] == "pooled-direction-subtract"),
        "the necessity arm must subtract")
    chk([j["pass"] for j in generation_jobs("plain")][:6] == ["primary"] * 6,
        "the scored cells must be generated before the unscored ones")

    # -- 7. endpoint 1: the marker gap ------------------------------------------------
    sect("endpoint 1: the marker gap")
    base = _cellrows(scored_cell("plain-baseline"), 128, n_marker=8)         # 6.25 percent
    add = _cellrows(scored_cell("pooled-direction-add"), 128, n_marker=48)   # 37.5 percent
    v, d = endpoint1(base + add)
    chk(d == "PASS_MARKER_RATE_GAP", f"a 31.25 point gap must pass, got {d}")
    chk(abs(v["marker_share_gap_pct_points"] - 31.25) < 1e-9,
        f"gap arithmetic: {v['marker_share_gap_pct_points']}")
    v, d = endpoint1(base + _cellrows(scored_cell("pooled-direction-add"), 128, n_marker=12))
    chk(d == "REFUTED_MARKER_RATE_NULL", f"a 3.125 point gap must refute, got {d}")
    v, d = endpoint1(base + _cellrows(scored_cell("pooled-direction-add"), 128, n_marker=24))
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS",
        f"a 12.5 point gap is between the cutoffs, got {d}")
    v, d = endpoint1(base)
    chk(d == MISSING and v["missing_cells"] == ["pooled-direction-add_c2.0"],
        f"a missing cell must be named, not averaged over: {d} {v['missing_cells']}")

    # -- 8. endpoint 2: length ---------------------------------------------------------
    sect("endpoint 2: length")
    b100 = _cellrows(scored_cell("plain-baseline"), 128, n_tokens=100)
    v, d = endpoint2(b100 + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=160))
    chk(d == "PASS_LENGTH_MULTIPLIER" and abs(v["length_multiplier"] - 1.6) < 1e-9,
        f"1.6 times must pass, got {d} at {v['length_multiplier']}")
    v, d = endpoint2(b100 + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=150))
    chk(d == "PASS_LENGTH_MULTIPLIER", f"exactly 1.5 times must pass, got {d}")
    v, d = endpoint2(b100 + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=105))
    chk(d == "REFUTED_LENGTH_NULL" and abs(v["length_change_pct"] - 5.0) < 1e-9,
        f"a 5 percent rise must refute, got {d} at {v['length_change_pct']}")
    v, d = endpoint2(b100 + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=92))
    chk(d == "REFUTED_LENGTH_NULL", f"an 8 percent fall must refute, got {d}")
    v, d = endpoint2(b100 + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=130))
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS", f"1.3 times is between the cutoffs, got {d}")
    v, d = endpoint2(_cellrows(scored_cell("plain-baseline"), 128, n_tokens=0)
                     + _cellrows(scored_cell("pooled-direction-add"), 128, n_tokens=50))
    chk(d == "UNDECIDABLE_ZERO_BASELINE_LENGTH",
        f"a zero-length baseline has no multiplier, got {d}")

    # -- 9. endpoint 3: coherence -----------------------------------------------------
    sect("endpoint 3: coherence")
    v, d = endpoint3(_cellrows(scored_cell("pooled-direction-add"), 128, n_coherent=90))
    chk(d == "PASS_COHERENT" and abs(v["coherent_share_pct"] - 70.3125) < 1e-9,
        f"70.3 percent coherent must pass, got {d} at {v['coherent_share_pct']}")
    v, d = endpoint3(_cellrows(scored_cell("pooled-direction-add"), 128, n_coherent=63))
    chk(d == "REFUTED_COHERENCE_COLLAPSE",
        f"49.2 percent coherent must refute, got {d} at {v['coherent_share_pct']}")
    v, d = endpoint3(_cellrows(scored_cell("pooled-direction-add"), 128, n_coherent=80))
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS",
        f"62.5 percent is between the cutoffs, got {d}")
    v, d = endpoint3(_cellrows(scored_cell("pooled-direction-add"), 128, n_coherent=128,
                               n_degenerate=10))
    chk(v["n_repeating"] == 0 and v["n_off_topic"] == 0,
        "the two coherence reasons are counted separately for the reader")

    # -- 10. endpoint 4: suppression on the reasoning model ---------------------------
    sect("endpoint 4: suppression on the reasoning model")
    ceil = _cellrows(scored_cell("reasoning-ceiling"), 128, n_marker=128)
    v, d = endpoint4(ceil + _cellrows(scored_cell("pooled-direction-subtract"), 128,
                                      n_marker=64))
    chk(d == "PASS_SUPPRESSION" and abs(v["marker_share_drop_pct_points"] - 50.0) < 1e-9,
        f"a 50 point drop must pass, got {d}")
    v, d = endpoint4(ceil + _cellrows(scored_cell("pooled-direction-subtract"), 128,
                                      n_marker=124))
    chk(d == "REFUTED_NO_SUPPRESSION", f"a 3.125 point drop must refute, got {d}")
    v, d = endpoint4(ceil + _cellrows(scored_cell("pooled-direction-subtract"), 128,
                                      n_marker=128))
    chk(d == "REFUTED_NO_SUPPRESSION" and v["marker_share_drop_pct_points"] == 0.0,
        f"no drop at all must refute, got {d}")
    v, d = endpoint4(ceil + _cellrows(scored_cell("pooled-direction-subtract"), 128,
                                      n_marker=100))
    chk(d == "PASS_SUPPRESSION" and abs(v["marker_share_drop_pct_points"] - 21.875) < 1e-9,
        f"a 21.875 point drop clears the 20 point cutoff, got {d}")
    v, d = endpoint4(ceil + _cellrows(scored_cell("pooled-direction-subtract"), 128,
                                      n_marker=110))
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS"
        and abs(v["marker_share_drop_pct_points"] - 14.0625) < 1e-9,
        f"a 14.06 point drop is between the cutoffs, got {d}")

    # -- 11. endpoint 5: against both same-size controls ------------------------------
    sect("endpoint 5: against both same-size controls")
    add50 = _cellrows(scored_cell("pooled-direction-add"), 128, n_marker=64)   # 50 percent
    r16 = _cellrows(scored_cell("random-vector-add"), 128, n_marker=20)        # 15.625
    p16 = _cellrows(scored_cell("relabeled-placebo-add"), 128, n_marker=20)
    v, d = endpoint5(add50 + r16 + p16)
    chk(d == "PASS_ABOVE_BOTH_CONTROLS", f"34.4 points over both controls must pass, got {d}")
    r48 = _cellrows(scored_cell("random-vector-add"), 128, n_marker=62)        # 48.4375
    v, d = endpoint5(add50 + r48 + p16)
    chk(d == "REFUTED_PERTURBATION_CONFOUND",
        f"sitting 1.56 points from the random control must refute, got {d}")
    r30 = _cellrows(scored_cell("random-vector-add"), 128, n_marker=40)        # 31.25
    v, d = endpoint5(add50 + r30 + p16)
    chk(d == "PASS_ABOVE_BOTH_CONTROLS",
        f"18.75 and 34.4 points clear the 15 point cutoff on both controls, got {d}")
    r40 = _cellrows(scored_cell("random-vector-add"), 128, n_marker=52)        # 40.625
    v, d = endpoint5(add50 + r40 + p16)
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS",
        f"9.4 points over one control and 34.4 over the other is neither, got {d}")
    v, d = endpoint5(add50 + r16)
    chk(d == MISSING, f"a missing control must not be scored as a pass, got {d}")

    # -- 12. endpoint 6: prompt category ---------------------------------------------
    sect("endpoint 6: prompt category")
    def two_legs(gap_less, gap_ful):
        rows = []
        for arm, gap in (("harmless-only-direction-add", gap_less),
                         ("harmful-only-direction-add", gap_ful)):
            cell = scored_cell(arm)
            for i in range(64):
                rows.append(_plant(cell, category="harmful", item_id=i,
                                   marker=i < 32))                    # 50 percent
            for i in range(64):
                rows.append(_plant(cell, category="harmless", item_id=64 + i,
                                   marker=i < 32 - int(round(gap * 64 / 100))))
        return rows
    v, d = endpoint6(two_legs(9.375, 9.375))
    chk(d == "PASS_NOT_CATEGORY_SPECIFIC",
        f"two 9.4 point gaps must pass, got {d} at {v['largest_absolute_gap_pct_points']}")
    v, d = endpoint6(two_legs(9.375, 31.25))
    chk(d == "REFUTED_CATEGORY_CONFOUND",
        f"a 31.25 point gap on one leg must refute, got {d}")
    v, d = endpoint6(two_legs(9.375, 20.3125))
    chk(d == "INDETERMINATE_BETWEEN_CUTOFFS",
        f"a 20.3 point gap is between the cutoffs, got {d}")
    v, d = endpoint6([r for r in two_legs(9.375, 9.375)
                      if r["cell"] != scored_cell("harmful-only-direction-add")])
    chk(d == MISSING, f"one leg missing must be named, got {d}")

    # -- 13. the pass filter keeps the three passes apart ----------------------------
    sect("the pass filter keeps the three passes apart")
    mixed = (_cellrows(scored_cell("plain-baseline"), 128, n_marker=8)
             + _cellrows(scored_cell("plain-baseline"), 128, "robustness", n_marker=128)
             + _cellrows(scored_cell("pooled-direction-add"), 128, n_marker=48))
    v, d = endpoint1(mixed, "primary")
    chk(abs(v["marker_share_gap_pct_points"] - 31.25) < 1e-9,
        "robustness rows must not leak into the scored pass")
    s = cell_summary(mixed, scored_cell("plain-baseline"), "robustness")
    chk(s["n_rows"] == 128 and s["marker_share_pct"] == 100.0,
        f"the robustness rows must be readable on their own: {s['n_rows']}")

    # -- 14. one record really can be written -----------------------------------------
    sect("one record really can be written")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cell = {"seed": SEED, "pipeline": PIPELINE, "stage": "endpoints", "endpoint": 1,
                "pass": "primary"}
        vals, dec = endpoint1(base + add)
        out = provenance.write_result(
            td / f"{provenance.cell_slug(cell)}.json", cell=cell, inputs=[],
            metric=ENDPOINTS_SPEC[0]["metric"], values=vals,
            threshold={"numeric": threshold_numbers(1)},
            decision_rule=f"THRESHOLD: {ENDPOINTS_SPEC[0]['threshold']}", decision=dec)
        back = json.loads(out.read_text())
        chk(back["decision"] == "PASS_MARKER_RATE_GAP", "the record round trip lost the call")
        chk(abs(back["values"]["marker_share_gap_pct_points"] - 31.25) < 1e-9,
            "the record round trip lost the value the call was made on")
        chk(back["cell"]["endpoint"] == 1 and back["env"]["timestamp_utc"][:2] == "20",
            "the record must carry its own identity and time")

    # -- 15. a same-box rerun is never called replication -----------------------------
    sect("a same-box rerun is never called replication")
    for nm, s in (("RERUN_NOTE", RERUN_NOTE), ("ROW_NOTE", ROW_NOTE), ("SIZE_NOTE", SIZE_NOTE),
                  ("LAYER_NOTE", LAYER_NOTE), ("module docstring", __doc__ or "")):
        low = s.lower()
        chk("replicat" not in low or "not replication" in low,
            f"{nm} must never call a same-box rerun replication")

    sections.append(("end", len(fails), n_checks[0]))
    for (label, f0, c0), (_, f1, c1) in zip(sections, sections[1:]):
        print(f"  {'ok  ' if f1 == f0 else 'FAIL'} {c1 - c0:3d} checks: {label}")
    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", f"PASS ({n_checks[0]} checks)" if not fails
          else f"{len(fails)} FAILURES of {n_checks[0]} checks")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Seed B, pipeline v2.1 runner. See the module docstring.")
    ap.add_argument("--stage", choices=["extract", "generate", "endpoints"])
    ap.add_argument("--out", default="results_seedB/out",
                    help="where the records, activations and generations are written")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="prompts generated at once. 32 sequences of at most 1024 tokens plus "
                         "16 GB of 16-bit weights fits a 40 GB card with room to spare.")
    ap.add_argument("--role", choices=["plain", "reason", "both"], default="both",
                    help="generate stage only: restart one model's half without the other")
    ap.add_argument("--force", action="store_true",
                    help="extract stage only: read the activations again even if saved")
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
