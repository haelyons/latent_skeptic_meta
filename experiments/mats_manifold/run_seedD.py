#!/usr/bin/env python3
"""Measures a character-count curve and a ripple statistic in one open language model.

WHAT THIS RUNS. One public checkpoint of one open model reads fixed, committed text.
The text is prose that has been hard-wrapped at seven line widths. At every token
position we know how many characters have passed since the last line break. A ridge
regression (a linear fit with a penalty on large weights) predicts that count from the
model's internal vector at that position. We report the fit quality on held-out
documents, for every layer. Then, at the best-scoring layer, we look for a faint
periodic pattern in the similarity structure of the per-count average vectors. That
pattern is the "ripple". A label-shuffling comparison says whether the pattern is
bigger than chance.

The complete specification is the design object inside
records/seedD_v2_design_diff_triage.json. This file implements it and changes nothing.
Every constant that the design marks as a guess is treated as binding here.

NO TEXT IS GENERATED ANYWHERE. The design says so in one line: "No text generation
anywhere: one deterministic, teacher-forced forward pass per stimulus variant, fixed
text." So this file has no sampling settings, no decoding, and no answer parsing. The
build brief that produced this file asks for a generation stage, a reasoning-tag
parser, and an answer extractor. Those three things cannot exist for this experiment.
The brief is a fixed template shared with a different, text-generating project
(pipeline.js line 104). The three closest real equivalents are implemented and tested
instead, and named at each definition below.

STAGES, in order, each runnable on its own:
  --stage extract    checks the text file and the model, then runs one forward pass per
                     text variant and saves what the later stages need.
  --stage calibrate  fixes every remaining constant using training documents only, and
                     writes that record before any held-out number is computed.
  --stage endpoints  computes the per-layer fit, the ripple, the shuffle comparison,
                     the token-position control, and the fitted-probe cross-check.
  --stage all        the three above, in order.
  --selftest         uses only the Python standard library, loads no model, needs no
                     graphics card, and exits non-zero on any failure.

The model is loaded once, in the extract stage, and released before the analysis
stages. Only one model exists in this experiment, so only one is ever resident.

Every result stage writes one self-describing record through provenance.py. A record
carries its own identity, its inputs with content hashes, its measurement, the rule
that judges it, and the resulting call. The rule text is written in this file before
any value is known.

A second run of this script on the same rented box is a rerun. It is not a
replication. No output here calls it one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Constants. Source of each is marked. "DESIGN" means the committed design object.
# "MEMORY" means recalled from a paper and checked against the download.
# "GUESS" means this file fixes it because the design leaves it to calibration.
# ---------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-410m-deduped"          # DESIGN
REVISION = "step143000"                              # DESIGN, final pretraining checkpoint
EXPECT_LAYERS = 24                                   # MEMORY, Pythia paper arXiv 2304.01373 table 1
EXPECT_HIDDEN = 1024                                 # MEMORY, same table
STIMULUS_SHA256 = "1fe73074a3d127e0bc546fb05a5adf9c2f0715f035b7f6fa7ac546afeaefe8ca"   # DESIGN
DESIGN_V2_SHA256 = "45e160eb9fb41ff6995457347b2e8bf9f78d16dcad3145e62dbddc2e37736127"  # committed design object, hashed on this workstation 2026-08-21
WIDTHS = [20, 40, 60, 80, 100, 120, 140]             # DESIGN
N_DOCS = 24                                          # DESIGN
N_TRAIN_DOCS = 18                                    # DESIGN
RIPPLE_WIDTH = 140                                   # DESIGN
N_LABELS = 140                                       # DESIGN, a 140-by-140 similarity matrix
SEED = 20260821                                      # DESIGN, one fixed seed
RIDGE_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]         # DESIGN, binding grid
N_FOLDS = 5                                          # DESIGN, five-way split of training documents
N_PERMUTATIONS = 200                                 # DESIGN, working figure, binding
NULL_RANK_MAX = 10                                   # DESIGN, "rank 10 or better of 200"
ANCHOR_LO = 0.574                                    # DESIGN, 0.724 minus 0.15
ANCHOR_HI = 0.874                                    # DESIGN, 0.724 plus 0.15
BAND_UPPER_FLOOR = 43                                # DESIGN, lowest acceptable search-band top
CONFOUND_MARGIN_CHARS = 3                            # DESIGN, token-position control margin
SECONDARY_MARGIN_CHARS = 5                           # DESIGN, fitted-probe agreement margin
VARIANCE_TARGET = 0.95                               # GUESS, the 95 percent convention in records/PREDICTIONS_gatepress_seedD.md line R4
MIN_ENTRIES_PER_DELTA = 20                           # GUESS, a distance is used only if at least 20 matrix entries average into it
DETREND_DEGREE = 2                                   # GUESS, see calibrate_constants for why a moving average is refused
BAND_HI_MULTIPLE = 3                                 # GUESS, search up to three times the predicted spacing
TOKEN_WINDOW_MIN_TRAIN = 5                           # GUESS, a token-count slot needs 5 training positions to enter the window
REPEAT_BLOCK_MAX = 8                                 # GUESS, longest repeating block the degenerate check looks for
REPEAT_MIN_RUNS = 6                                  # GUESS, that many back-to-back repeats counts as degenerate
CONTEXT_LIMIT_FALLBACK = 2048                        # MEMORY, Pythia context length; the downloaded config overrides it

RESULTS_DIR_DEFAULT = "results_seedD"


class Halt(Exception):
    """Raised where the design says halt. Never caught to keep a run going."""


# ---------------------------------------------------------------------------
# Part 1. The position labeller.
#
# This is the equivalent of the reasoning-tag parser the build template asks for.
# There, an opening marker can be missing because the chat template already emitted
# it, so only the closing marker appears in the text. Here the same shape occurs
# twice, and both are covered by the self-test:
#   - the first line of a document has no line break in front of it, so its opening
#     boundary is implied rather than present;
#   - a single token can swallow a line break, so the boundary sits inside a unit
#     rather than between two units.
# Getting either wrong shifts every label on a line and silently moves the whole
# measurement. So this function is small, pure, and tested.
# ---------------------------------------------------------------------------

def line_columns(text: str):
    """For each character, its column on its line and its line number.

    The column of a character is the count of characters on its line, up to and
    including itself. It is 1 for the first character of a line. A line-break
    character belongs to the line it ends, and takes the next column after the last
    visible character. The count restarts at zero after each line break, so the first
    character of the next line is column 1 again. This is definition 1 of the design.
    """
    cols = [0] * len(text)
    lines = [0] * len(text)
    col = 0
    line = 0
    for i, ch in enumerate(text):
        col += 1
        cols[i] = col
        lines[i] = line
        if ch == "\n":
            col = 0
            line += 1
    return cols, lines


def labels_for_spans(text: str, spans):
    """Labels for token positions, from character spans over the same text.

    `spans` holds one (start, end) character range per token, in reading order, as a
    tokenizer that reports offsets gives them. Returns three lists:
      character labels: the column of the token's last character;
      token labels: how many tokens have ended on that line so far, counting this one;
      line numbers: which line the token's last character sits on.
    A token with an empty range gets None in all three, and is never counted.
    """
    cols, lines = line_columns(text)
    n = len(text)
    char_labels = []
    token_labels = []
    line_nums = []
    per_line_counts = {}
    for (start, end) in spans:
        if end <= start or end > n or start < 0:
            char_labels.append(None)
            token_labels.append(None)
            line_nums.append(None)
            continue
        last = end - 1
        line = lines[last]
        per_line_counts[line] = per_line_counts.get(line, 0) + 1
        char_labels.append(cols[last])
        token_labels.append(per_line_counts[line])
        line_nums.append(line)
    return char_labels, token_labels, line_nums


# ---------------------------------------------------------------------------
# Part 2. The degenerate-input detector.
#
# This is the equivalent of the degenerate-generation detector the template asks for.
# The template's two cases map onto the two cases that can occur here:
#   - the length-cap case: a text variant whose token count reaches the model's
#     context limit, so the committed text would be silently cut short;
#   - the repeated-sequence case: a text variant containing the same short block of
#     tokens many times in a row, which would flood a label with copies of one vector.
# ---------------------------------------------------------------------------

def max_block_repeats(seq, max_block: int = REPEAT_BLOCK_MAX) -> int:
    """Largest number of back-to-back copies of any block of 1 to max_block items."""
    best = 1
    n = len(seq)
    for b in range(1, max_block + 1):
        i = 0
        while i + b <= n:
            if i >= b and seq[i - b:i] == seq[i:i + b]:
                i += 1                      # already inside a run counted earlier
                continue
            runs = 1
            j = i + b
            while j + b <= n and seq[j:j + b] == seq[i:i + b]:
                runs += 1
                j += b
            if runs > best:
                best = runs
            i += 1
    return best


def degenerate_flags(n_tokens: int, context_limit: int, token_texts) -> list:
    """Names of the degenerate conditions present. Empty list means clean."""
    flags = []
    if n_tokens >= context_limit:
        flags.append("token_cap")
    if max_block_repeats(list(token_texts)) >= REPEAT_MIN_RUNS:
        flags.append("repeated_sequence")
    return flags


# ---------------------------------------------------------------------------
# Part 3. Exact-match checks and exact slot assignment.
#
# This is the equivalent of the exact-match answer extraction the template asks for.
# Nothing here is fuzzy, normalised, or rounded. Three things must match exactly: the
# text file's content hash, the downloaded configuration's layer count and width, and
# the integer label of a position against a label slot of the similarity matrix. An
# off-by-one in the last one would move every row of the matrix without any error.
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_hash_match(actual: str, expected: str) -> bool:
    return isinstance(actual, str) and isinstance(expected, str) and actual == expected


def exact_config_match(n_layers, hidden_size) -> bool:
    return n_layers == EXPECT_LAYERS and hidden_size == EXPECT_HIDDEN


def label_slot(label, window_lo: int, n_labels: int):
    """Slot index of a label inside a window, or None when it is outside.

    Membership is exact integer equality against one of the window's values. A float,
    a bool, or a None label is never accepted.
    """
    if label is None or isinstance(label, bool) or not isinstance(label, int):
        return None
    off = label - window_lo
    if 0 <= off < n_labels:
        return off
    return None


def slot_counts(labels, window_lo: int, n_labels: int):
    """Per-slot position counts, and the slot index of every label (None when outside)."""
    counts = [0] * n_labels
    slots = []
    for lab in labels:
        s = label_slot(lab, window_lo, n_labels)
        slots.append(s)
        if s is not None:
            counts[s] += 1
    return counts, slots


def first_full_window(counts_by_label: dict, n_labels: int, min_count: int, max_start: int = 8):
    """Lowest window start whose n_labels consecutive values all reach min_count.

    The design's definition 1 gives a lowest possible label of 1, because a label
    counts the characters up to and including a token's last character. The design's
    definition 3 names the label range 0 to 139. The two differ by one. The design
    also says the calibration stage fixes every remaining constant of definitions 2
    to 4, so this window start is fixed there, from training documents only, and is
    written into the calibration record.
    """
    for lo in range(0, max_start + 1):
        if all(counts_by_label.get(lo + k, 0) >= min_count for k in range(n_labels)):
            return lo
    return None


# ---------------------------------------------------------------------------
# Part 4. The ripple statistic. Pure Python so the self-test needs no extra packages.
# ---------------------------------------------------------------------------

def offdiag_profile(matrix, max_delta: int):
    """Mean of the entries a given distance above the diagonal, for each distance.

    `matrix` is a square similarity matrix as a list of rows. Distance runs from 1 to
    max_delta. This is the profile the design calls m(delta).
    """
    n = len(matrix)
    if max_delta < 1 or max_delta >= n:
        raise Halt("max_delta %r out of range for a %d-by-%d matrix" % (max_delta, n, n))
    prof = []
    for d in range(1, max_delta + 1):
        total = 0.0
        count = n - d
        row = 0
        while row < count:
            total += matrix[row][row + d]
            row += 1
        prof.append(total / count)
    return prof


def _solve_small(a, b):
    """Gaussian elimination with partial pivoting, for a few unknowns."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise Halt("singular system in the detrend fit")
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def poly_detrend(y, degree: int = DETREND_DEGREE):
    """Subtract the best-fitting low-order polynomial in distance.

    The design leaves the detrend to calibration. A moving average is refused here:
    subtracting a moving average of window W removes every period longer than about
    W, and the periods this study predicts are longer than any window a 120-point
    profile could carry. A polynomial of degree 2 removes the slow decay of the
    profile and its curvature, and leaves every period inside the searched band.
    """
    n = len(y)
    if n < degree + 2:
        raise Halt("profile of length %d is too short to detrend at degree %d" % (n, degree))
    ts = [(2.0 * i / (n - 1)) - 1.0 for i in range(n)]
    powers = [[t ** p for p in range(degree + 1)] for t in ts]
    a = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    b = [0.0] * (degree + 1)
    for i in range(n):
        pi = powers[i]
        for r in range(degree + 1):
            b[r] += pi[r] * y[i]
            for c in range(degree + 1):
                a[r][c] += pi[r] * pi[c]
    coef = _solve_small(a, b)
    return [y[i] - sum(coef[p] * powers[i][p] for p in range(degree + 1)) for i in range(n)]


_TRIG_CACHE = {}


def _trig_table(n: int):
    tab = _TRIG_CACHE.get(n)
    if tab is None:
        tab = []
        for k in range(1, n // 2 + 1):
            row = []
            for t in range(n):
                ang = 2.0 * math.pi * k * t / n
                row.append((math.cos(ang), math.sin(ang)))
            tab.append(row)
        _TRIG_CACHE[n] = tab
    return tab


def dft_power(y):
    """Power at each whole number of cycles across the profile.

    Returns a dictionary from cycle count to power. The zero-cycle term is dropped.
    The period in characters of a cycle count k is len(y) divided by k.
    """
    n = len(y)
    tab = _trig_table(n)
    out = {}
    for idx, row in enumerate(tab):
        k = idx + 1
        re = 0.0
        im = 0.0
        for t in range(n):
            c, s = row[t]
            re += y[t] * c
            im -= y[t] * s
        out[k] = (re * re + im * im) / (n * n)
    return out


def band_bins(length: int, band_lo: float, band_hi: float):
    """Cycle counts whose period in characters lies inside the band, longest first."""
    bins = []
    for k in range(1, length // 2 + 1):
        period = length / k
        if band_lo - 1e-9 <= period <= band_hi + 1e-9:
            bins.append((k, period))
    return bins


def ripple_stat(matrix, max_delta: int, band_lo: float, band_hi: float,
                degree: int = DETREND_DEGREE, keep_profile: bool = False):
    """Ripple size and its period, from one similarity matrix.

    Returns a dictionary. "amplitude" is the largest power at any period inside the
    band. "wavelength" is that period, in the units of the matrix's labels. When two
    periods tie on power the longer one wins, so the result never depends on ordering.
    """
    prof = offdiag_profile(matrix, max_delta)
    det = poly_detrend(prof, degree)
    power = dft_power(det)
    bins = band_bins(len(det), band_lo, band_hi)
    if not bins:
        raise Halt("no whole number of cycles gives a period inside %.4f to %.4f "
                   "characters for a profile of length %d" % (band_lo, band_hi, len(det)))
    best_k, best_period = bins[0]
    best_power = power[best_k]
    for k, period in bins[1:]:
        if power[k] > best_power:
            best_k, best_period, best_power = k, period, power[k]
    out = {
        "amplitude": best_power,
        "wavelength": best_period,
        "cycles": best_k,
        "n_bins": len(bins),
        "profile_length": len(det),
    }
    if keep_profile:
        out["profile"] = prof
        out["detrended"] = det
        out["in_band"] = [{"cycles": k, "period": period, "power": power[k]} for k, period in bins]
    return out


# ---------------------------------------------------------------------------
# Part 5. Endpoint arithmetic. Every one of these is a pure function of numbers, so
# the self-test can plant values whose correct call is known in advance.
# ---------------------------------------------------------------------------

RULE_INTEGRITY = (
    "Text file hash must equal the committed hash exactly, layer count must equal 24, "
    "and hidden size must equal 1024. Anything else is SETUP_FAILURE and halts the "
    "run before any measurement."
)
RULE_BAND = (
    "The calibrated search band's top, in characters, must be at least 43. Below 43 "
    "is BAND_BELOW_FLOOR and halts before any held-out number is computed."
)
RULE_ANCHOR = (
    "Held-out fit quality at the best-scoring layer must lie between 0.574 and 0.874 "
    "inclusive. Outside that is ANCHOR_OUT_OF_BAND, a fact about this instrument, and "
    "it blocks any checkpoint sweep."
)
RULE_RIPPLE = (
    "Rank of the observed ripple size among 200 label-shuffled draws, counting the "
    "observed value itself, must be 10 or better. Worse is NULL_NOT_CLEARED."
)
RULE_CONFOUND = (
    "The character-count ripple is credited over token position when the two periods "
    "differ by more than 3 characters, or when exactly one of the two labellings "
    "clears its own shuffle comparison. Otherwise CONFOUNDED_WITH_TOKEN_POSITION."
)
RULE_SECONDARY = (
    "The average-vector readout and the fitted-probe readout agree when both clear or "
    "both fail their own shuffle comparison, and, if both clear, their periods differ "
    "by 5 characters or fewer. No shuffle comparison is computed for the fitted probe, "
    "so the presence half is recorded as not decidable and only the periods are compared."
)


def decide_integrity(hash_ok: bool, n_layers, hidden_size) -> str:
    if hash_ok and exact_config_match(n_layers, hidden_size):
        return "SETUP_OK"
    return "SETUP_FAILURE"


def decide_band(band_hi) -> str:
    return "BAND_OK" if band_hi >= BAND_UPPER_FLOOR else "BAND_BELOW_FLOOR"


def decide_anchor(best_r2) -> str:
    if best_r2 is None or best_r2 != best_r2:            # None, or not a number
        return "ANCHOR_OUT_OF_BAND"
    return "ANCHOR_IN_BAND" if ANCHOR_LO <= best_r2 <= ANCHOR_HI else "ANCHOR_OUT_OF_BAND"


def null_rank(observed, draws) -> int:
    """Position of the observed value in the list of draws plus itself, largest first."""
    return 1 + sum(1 for v in draws if v >= observed)


def decide_ripple(rank: int) -> str:
    return "NULL_CLEARED" if rank <= NULL_RANK_MAX else "NULL_NOT_CLEARED"


def decide_confound(wl_chars, wl_token_in_chars, char_cleared, token_cleared) -> str:
    if wl_token_in_chars is None or token_cleared is None:
        return "TOKEN_CONTROL_NOT_COMPUTABLE"
    if abs(wl_chars - wl_token_in_chars) > CONFOUND_MARGIN_CHARS:
        return "CHARACTER_COUNT_DISTINGUISHED"
    if bool(char_cleared) != bool(token_cleared):
        return "CHARACTER_COUNT_DISTINGUISHED"
    return "CONFOUNDED_WITH_TOKEN_POSITION"


def decide_secondary(wl_chars, wl_secondary, char_cleared, secondary_cleared=None) -> str:
    if wl_secondary is None:
        return "SECONDARY_NOT_COMPUTABLE"
    agree_period = abs(wl_chars - wl_secondary) <= SECONDARY_MARGIN_CHARS
    if secondary_cleared is None:
        return ("PERIODS_AGREE_PRESENCE_NOT_DECIDABLE" if agree_period
                else "PERIODS_DISAGREE_PRESENCE_NOT_DECIDABLE")
    if bool(char_cleared) != bool(secondary_cleared):
        return "READOUTS_DISAGREE_ON_PRESENCE"
    if not char_cleared:
        return "BOTH_FAIL_AGREEMENT_HOLDS"
    return "BOTH_CLEAR_PERIODS_AGREE" if agree_period else "BOTH_CLEAR_PERIODS_DISAGREE"


# ---------------------------------------------------------------------------
# Part 6. Ridge fitting from running totals.
#
# Storing every internal vector of every position would need over 13 gigabytes. It is
# not needed. A ridge fit and a fit-quality score depend on the data only through five
# running totals per layer: the sum of outer products of the vectors, the sum of the
# vectors, the sum of vector times label, the sum of labels, and the sum of squared
# labels. So the extract stage accumulates those, per layer, per group of documents.
# ---------------------------------------------------------------------------

def ridge_fit_from_stats(gram, vec_sum, cross, label_sum, n, alpha):
    """Weights and offset of a ridge fit, from running totals. Needs numpy."""
    import numpy as np
    xbar = vec_sum / n
    ybar = label_sum / n
    centred_gram = gram - np.outer(vec_sum, vec_sum) / n
    centred_cross = cross - vec_sum * (label_sum / n)
    dim = centred_gram.shape[0]
    weights = np.linalg.solve(centred_gram + alpha * np.eye(dim), centred_cross)
    offset = ybar - float(weights @ xbar)
    return weights, offset


def r2_from_stats(weights, offset, gram, vec_sum, cross, label_sum, label_sq_sum, n):
    """Fit quality on a group, from that group's running totals. Needs numpy."""
    sse = (float(weights @ gram @ weights)
           + 2.0 * offset * float(weights @ vec_sum)
           - 2.0 * float(weights @ cross)
           + n * offset * offset
           - 2.0 * offset * label_sum
           + label_sq_sum)
    sstot = label_sq_sum - (label_sum * label_sum) / n
    if sstot <= 0:
        raise Halt("the held-out labels have no spread, so fit quality is undefined")
    return 1.0 - sse / sstot, sse, sstot


# ---------------------------------------------------------------------------
# Part 7. Shared plumbing.
# ---------------------------------------------------------------------------

def load_provenance():
    """Import provenance.py from beside this file, or from the frozen folder."""
    try:
        import provenance                        # on the box the file sits alongside
    except ImportError:
        sys.path.insert(0, str(HERE / "frozen"))
        import provenance                        # on this workstation it sits in frozen/
    if not hasattr(provenance, "write_result"):
        raise Halt("provenance.py has no write_result; refusing to run without records")
    return provenance


def find_first(names):
    for name in names:
        p = HERE / name
        if p.exists():
            return p
    return None


def stimulus_path(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise Halt("named text file %s is absent; the design says halt, never regenerate" % p)
        return p
    p = find_first(["stimulus_linewrap.json", "records/stimulus_linewrap.json"])
    if p is None:
        raise Halt("stimulus_linewrap.json is absent; the design says halt, never regenerate")
    return p


def design_record_path(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return find_first([
        "seedD_v2_design_diff_triage.json",              # box layout, flat, committed version
        "records/seedD_v2_design_diff_triage.json",      # this workstation, committed version
        "seedD_design_diff_triage.json",                 # box layout, superseded first version
        "records/seedD_design_diff_triage.json",         # this workstation, superseded first version
    ])


def hf_token():
    """Access token and the name of the variable it came from.

    The build brief names HF_KEY_ONE. The launcher forwards the same secret as
    HF_TOKEN, and the bootstrap script also exports HUGGING_FACE_HUB_TOKEN. All three
    are tried, in that order. The model here is public, so an absent token is fine and
    is recorded as such. No token value is ever printed or written.
    """
    for name in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(name)
        if val:
            return val, name
    return None, "none"


def load_stimulus(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    docs = data.get("docs", [])
    problems = []
    if len(docs) != N_DOCS:
        problems.append("document count %d, expected %d" % (len(docs), N_DOCS))
    if list(meta.get("widths", [])) != WIDTHS:
        problems.append("line widths %r, expected %r" % (meta.get("widths"), WIDTHS))
    if meta.get("train_docs") != N_TRAIN_DOCS:
        problems.append("training document count %r, expected %d" % (meta.get("train_docs"), N_TRAIN_DOCS))
    n_train = sum(1 for d in docs if d.get("split") == "train")
    if n_train != N_TRAIN_DOCS:
        problems.append("documents marked train: %d, expected %d" % (n_train, N_TRAIN_DOCS))
    for d in docs:
        missing = [str(w) for w in WIDTHS if str(w) not in d.get("wrapped", {})]
        if missing:
            problems.append("document %r lacks widths %r" % (d.get("id"), missing))
            break
    if problems:
        raise Halt("the text file does not match the design: " + "; ".join(problems))
    return meta, docs


def fold_of_documents(train_ids):
    """Which of the five groups each training document belongs to.

    Splitting by document, not by position, keeps sentences of one document out of
    both sides of a split. The one fixed seed makes the split repeatable.
    """
    import numpy as np
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(train_ids)))
    assign = {}
    for rank, pos in enumerate(order):
        assign[train_ids[int(pos)]] = rank % N_FOLDS
    return assign


def paths_for(results_dir):
    out = Path(results_dir) / "out"
    work = Path(results_dir) / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def record_inputs(prov, stim, design, extra=()):
    items = [prov.describe_input(stim, n_items=N_DOCS * len(WIDTHS))]
    if design is not None:
        items.append(prov.describe_input(design, n_items=1))
    for path, n in extra:
        items.append(prov.describe_input(path, n_items=n))
    return items


def base_cell(stage, arm, endpoint=None):
    cell = {"model": "pythia-410m-deduped", "revision": REVISION, "stage": stage, "arm": arm}
    if endpoint is not None:
        cell["endpoint"] = endpoint
    return cell


def write_record(prov, out_dir, cell, **kwargs):
    path = Path(out_dir) / (prov.cell_slug(cell) + ".json")
    prov.write_result(path, cell=cell, **kwargs)
    print("[record] %s -> %s" % (kwargs.get("decision", ""), path))
    return path


RERUN_NOTE = ("A further run of this script on the same box is a rerun of the same "
              "instrument, not a replication.")


# ---------------------------------------------------------------------------
# Part 8. Stage one: check the inputs, run the forward passes, save the totals.
# ---------------------------------------------------------------------------

def stage_extract(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)

    actual_hash = sha256_file(stim)
    hash_ok = exact_hash_match(actual_hash, STIMULUS_SHA256)
    print("[extract] text file %s" % stim)
    print("[extract] hash %s (expected %s) match=%s" % (actual_hash, STIMULUS_SHA256, hash_ok))
    meta, docs = load_stimulus(stim)

    token, token_env = hf_token()
    print("[extract] access token from %s" % token_env)
    kw = {"revision": REVISION}
    if token:
        kw["token"] = token

    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    context_limit = int(getattr(config, "max_position_embeddings", CONTEXT_LIMIT_FALLBACK) or CONTEXT_LIMIT_FALLBACK)
    dtype_name = str(getattr(config, "torch_dtype", "") or "float32").replace("torch.", "")
    print("[extract] config layers=%r hidden=%r context=%d recommended dtype=%s"
          % (n_layers, hidden, context_limit, dtype_name))

    decision = decide_integrity(hash_ok, n_layers, hidden)
    write_record(
        prov, out_dir, base_cell("extract", "integrity_check", endpoint=1),
        inputs=record_inputs(prov, stim, design),
        metric="Committed text file hash, and the downloaded configuration's layer count and hidden size.",
        values={"stimulus_sha256": actual_hash, "stimulus_sha256_expected": STIMULUS_SHA256,
                "stimulus_hash_match": hash_ok, "n_layers": n_layers, "hidden_size": hidden,
                "context_limit": context_limit, "recommended_dtype": dtype_name,
                "model_id": MODEL_ID, "revision": REVISION,
                "design_record_found": str(design) if design else None,
                "design_record_committed_sha256": DESIGN_V2_SHA256,
                "access_token_source": token_env},
        threshold={"stimulus_sha256": STIMULUS_SHA256, "n_layers": EXPECT_LAYERS,
                   "hidden_size": EXPECT_HIDDEN},
        decision_rule=RULE_INTEGRITY,
        decision=decision,
        notes=("Layer count 24 and hidden size 1024 are recalled from the Pythia paper, "
               "arXiv 2304.01373 table 1, and checked against the download. " + RERUN_NOTE),
    )
    if decision != "SETUP_OK":
        raise Halt("setup check failed: hash_match=%s layers=%r hidden=%r" % (hash_ok, n_layers, hidden))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    if not getattr(tok, "is_fast", False):
        raise Halt("the tokenizer does not report character offsets; labels cannot be computed")

    torch.backends.cuda.matmul.allow_tf32 = False     # keeps the running totals accurate
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(dtype_name, torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[extract] loading the model in %s on %s" % (dtype_name, device))
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **kw)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    n_read = EXPECT_LAYERS + 1                        # index 0 is the embedding output
    groups = ["fold%d" % f for f in range(N_FOLDS)] + ["test"]
    gram = {g: np.zeros((n_read, EXPECT_HIDDEN, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    vec_sum = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    cross = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    label_sum = {g: 0.0 for g in groups}
    label_sq = {g: 0.0 for g in groups}
    n_pos = {g: 0 for g in groups}

    train_ids = [d["id"] for d in docs if d["split"] == "train"]
    folds = fold_of_documents(train_ids)

    w140_files = [open(work_dir / ("w140_layer%02d.f32" % l), "wb") for l in range(n_read)]
    w140_doc, w140_test, w140_char, w140_token = [], [], [], []

    rows_pos = open(out_dir / "rows_positions.jsonl", "w", encoding="utf-8")
    rows_var = open(out_dir / "rows_variants.jsonl", "w", encoding="utf-8")

    started = time.time()
    n_variants = 0
    all_flags = {}
    try:
        for doc in docs:
            did = int(doc["id"])
            split = doc["split"]
            group = "test" if split == "test" else "fold%d" % folds[did]
            for width in WIDTHS:
                text = doc["wrapped"][str(width)]
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids = enc["input_ids"]
                spans = [tuple(s) for s in enc["offset_mapping"]]
                char_labels, token_labels, line_nums = labels_for_spans(text, spans)
                pieces = tok.convert_ids_to_tokens(ids)
                flags = degenerate_flags(len(ids), context_limit, pieces)
                if flags:
                    all_flags["%d_w%d" % (did, width)] = flags
                if "token_cap" in flags:
                    raise Halt("document %d at width %d reaches the context limit of %d tokens; "
                               "the committed text would be cut short" % (did, width, context_limit))

                keep = [i for i, c in enumerate(char_labels) if c is not None]
                y = np.array([char_labels[i] for i in keep], dtype=np.float64)
                t_ids = torch.tensor([ids], dtype=torch.long, device=device)
                out = model(input_ids=t_ids, output_hidden_states=True, use_cache=False)
                states = out.hidden_states
                if len(states) != n_read:
                    raise Halt("the model returned %d readouts, expected %d" % (len(states), n_read))
                idx = torch.tensor(keep, dtype=torch.long, device=device)
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                for l in range(n_read):
                    x = states[l][0].index_select(0, idx).float()
                    gram[group][l] += (x.T @ x).double().cpu().numpy()
                    vec_sum[group][l] += x.sum(0).double().cpu().numpy()
                    cross[group][l] += (x.T @ y_t).double().cpu().numpy()
                    if width == RIPPLE_WIDTH:
                        x.cpu().numpy().astype(np.float32).tofile(w140_files[l])
                label_sum[group] += float(y.sum())
                label_sq[group] += float((y * y).sum())
                n_pos[group] += len(keep)
                if width == RIPPLE_WIDTH:
                    w140_doc.extend([did] * len(keep))
                    w140_test.extend([1 if split == "test" else 0] * len(keep))
                    w140_char.extend([char_labels[i] for i in keep])
                    w140_token.extend([token_labels[i] for i in keep])
                for i in keep:
                    rows_pos.write(json.dumps({
                        "doc": did, "width": width, "split": split, "group": group,
                        "tok_index": i, "tok_text": pieces[i],
                        "char_start": spans[i][0], "char_end": spans[i][1],
                        "line": line_nums[i], "char_label": char_labels[i],
                        "token_label": token_labels[i]}, ensure_ascii=False) + "\n")
                rows_var.write(json.dumps({
                    "doc": did, "width": width, "split": split, "group": group,
                    "n_chars": len(text), "n_tokens": len(ids), "n_labelled": len(keep),
                    "n_lines": text.count("\n") + 1,
                    "degenerate_flags": flags,
                    "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text}, ensure_ascii=False) + "\n")
                n_variants += 1
                del out, states
        elapsed = time.time() - started
    finally:
        rows_pos.close()
        rows_var.close()
        for fh in w140_files:
            fh.close()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    np.savez(work_dir / "totals.npz",
             groups=np.array(groups),
             gram=np.stack([gram[g] for g in groups]),
             vec_sum=np.stack([vec_sum[g] for g in groups]),
             cross=np.stack([cross[g] for g in groups]),
             label_sum=np.array([label_sum[g] for g in groups]),
             label_sq=np.array([label_sq[g] for g in groups]),
             n_pos=np.array([n_pos[g] for g in groups]))
    np.savez(work_dir / "w140_meta.npz",
             doc=np.array(w140_doc, dtype=np.int32),
             is_test=np.array(w140_test, dtype=np.int8),
             char_label=np.array(w140_char, dtype=np.int32),
             token_label=np.array(w140_token, dtype=np.int32),
             n_read=np.array([n_read]), hidden=np.array([EXPECT_HIDDEN]))
    with open(work_dir / "folds.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": SEED, "assignment": {str(k): v for k, v in folds.items()}}, fh, indent=2)

    write_record(
        prov, out_dir, base_cell("extract", "forward_passes"),
        inputs=record_inputs(prov, stim, design),
        metric="Counts from one forward pass per text variant, and the running totals saved for the fits.",
        values={"n_variants": n_variants, "n_positions": {g: n_pos[g] for g in groups},
                "n_positions_total": int(sum(n_pos.values())),
                "n_readouts": n_read, "device": device, "dtype": dtype_name,
                "seconds": round(elapsed, 1),
                "width140_rows": len(w140_char),
                "degenerate_variants": all_flags,
                "fold_assignment": {str(k): v for k, v in folds.items()}},
        threshold={"n_variants": N_DOCS * len(WIDTHS)},
        decision_rule=("Every one of the 24 documents at every one of the 7 widths must "
                       "produce one forward pass, with no variant flagged as reaching the "
                       "context limit. Anything else halts."),
        decision="EXTRACT_OK" if n_variants == N_DOCS * len(WIDTHS) else "EXTRACT_INCOMPLETE",
        notes=("Readout index 0 is the embedding output and index 24 is the last layer. "
               "This loader applies the model's final normalisation to index 24 only, "
               "which is a property of the library, recorded here rather than changed. "
               + RERUN_NOTE),
    )
    print("[extract] %d variants, %d labelled positions, %.1f s"
          % (n_variants, sum(n_pos.values()), elapsed))
    return 0


# ---------------------------------------------------------------------------
# Part 9. Stage two: fix every remaining constant, using training documents only.
# ---------------------------------------------------------------------------

def load_totals(work_dir):
    import numpy as np
    z = np.load(Path(work_dir) / "totals.npz", allow_pickle=False)
    groups = [str(g) for g in z["groups"]]
    return {"groups": groups, "gram": z["gram"], "vec_sum": z["vec_sum"], "cross": z["cross"],
            "label_sum": z["label_sum"], "label_sq": z["label_sq"], "n_pos": z["n_pos"]}


def sum_groups(tot, indices, layer):
    g = tot["gram"][indices, layer].sum(axis=0)
    v = tot["vec_sum"][indices, layer].sum(axis=0)
    c = tot["cross"][indices, layer].sum(axis=0)
    ls = float(tot["label_sum"][indices].sum())
    lq = float(tot["label_sq"][indices].sum())
    n = int(tot["n_pos"][indices].sum())
    return g, v, c, ls, lq, n


def read_w140_layer(work_dir, layer, n_rows, hidden=EXPECT_HIDDEN):
    import numpy as np
    path = Path(work_dir) / ("w140_layer%02d.f32" % layer)
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != n_rows * hidden:
        raise Halt("saved vectors for readout %d hold %d numbers, expected %d"
                   % (layer, arr.size, n_rows * hidden))
    return arr.reshape(n_rows, hidden)


def means_by_slot(x, slots, n_slots):
    """Average vector per slot, and how many positions entered each slot. Needs numpy."""
    import numpy as np
    keep = slots >= 0
    xk = x[keep]
    sk = slots[keep]
    ind = np.zeros((n_slots, xk.shape[0]), dtype=np.float32)
    ind[sk, np.arange(xk.shape[0])] = 1.0
    counts = ind.sum(axis=1)
    if float(counts.min()) < 1.0:
        empty = [int(i) for i in np.nonzero(counts < 1.0)[0]]
        raise Halt("label slots %r have no positions, so their average vector is undefined" % empty[:10])
    sums = ind @ xk
    return (sums / counts[:, None]).astype(np.float64), counts.astype(int)


def cosine_matrix(mat):
    """Cosine similarity between every pair of rows, as plain nested lists. Needs numpy."""
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if float(norms.min()) <= 0:
        raise Halt("a per-label average vector is exactly zero, so cosine similarity is undefined")
    unit = mat / norms
    return (unit @ unit.T).tolist()


def slots_array(labels, window_lo, n_labels):
    import numpy as np
    off = labels - window_lo
    off = np.where((off >= 0) & (off < n_labels), off, -1)
    return off.astype(np.int64)


def stage_calibrate(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    tot = load_totals(work_dir)
    groups = tot["groups"]
    fold_idx = [groups.index("fold%d" % f) for f in range(N_FOLDS)]
    n_read = tot["gram"].shape[1]

    # Ridge strength, by five-fold cross-validation over training documents only.
    table = {}
    for held in range(N_FOLDS):
        fit_idx = [fold_idx[f] for f in range(N_FOLDS) if f != held]
        eval_idx = [fold_idx[held]]
        for layer in range(n_read):
            g, v, c, ls, lq, n = sum_groups(tot, fit_idx, layer)
            ge, ve, ce, lse, lqe, ne = sum_groups(tot, eval_idx, layer)
            for alpha in RIDGE_GRID:
                w, off = ridge_fit_from_stats(g, v, c, ls, n, alpha)
                r2, _, _ = r2_from_stats(w, off, ge, ve, ce, lse, lqe, ne)
                table.setdefault((alpha, layer), []).append(float(r2))
    cv_mean = {key: sum(vals) / len(vals) for key, vals in table.items()}
    per_alpha_best = {}
    for alpha in RIDGE_GRID:
        best_layer = max(range(n_read), key=lambda l: cv_mean[(alpha, l)])
        per_alpha_best[alpha] = (best_layer, cv_mean[(alpha, best_layer)])
    alpha_star = max(RIDGE_GRID, key=lambda a: per_alpha_best[a][1])
    train_best_layer, train_best_r2 = per_alpha_best[alpha_star]
    print("[calibrate] ridge strength %g, best training readout %d, cross-validated quality %.4f"
          % (alpha_star, train_best_layer, train_best_r2))

    # Label windows, from training positions of the widest wrapping only.
    meta = np.load(work_dir / "w140_meta.npz", allow_pickle=False)
    is_test = meta["is_test"]
    char_lab = meta["char_label"]
    token_lab = meta["token_label"]
    n_rows = int(char_lab.size)
    train_mask = is_test == 0
    char_counts = {}
    for lab in char_lab[train_mask].tolist():
        char_counts[lab] = char_counts.get(lab, 0) + 1
    window_lo = first_full_window(char_counts, N_LABELS, min_count=1)
    if window_lo is None:
        raise Halt("no run of %d consecutive character-count values is populated on training text" % N_LABELS)
    token_counts = {}
    for lab in token_lab[train_mask].tolist():
        token_counts[lab] = token_counts.get(lab, 0) + 1
    token_hi = 0
    while token_counts.get(token_hi + 1, 0) >= TOKEN_WINDOW_MIN_TRAIN:
        token_hi += 1
    if token_hi < 4:
        raise Halt("only %d token-count values reach %d training positions" % (token_hi, TOKEN_WINDOW_MIN_TRAIN))
    print("[calibrate] character-count window %d to %d; token-count window 1 to %d"
          % (window_lo, window_lo + N_LABELS - 1, token_hi))

    # Ambient dimension of the count representation, on training text at the
    # training-chosen readout. The held-out readout is not known yet, and must not be.
    x = read_w140_layer(work_dir, train_best_layer, n_rows)
    slots = slots_array(char_lab, window_lo, N_LABELS)
    slots_train = np.where(train_mask, slots, -1)
    mat, counts = means_by_slot(x, slots_train, N_LABELS)
    centred = mat - mat.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv ** 2
    share = (var / var.sum()).tolist()
    cum = 0.0
    ambient = len(share)
    for i, s in enumerate(share):
        cum += s
        if cum >= VARIANCE_TARGET:
            ambient = i + 1
            break
    mode_pairs = max(1, (ambient + 1) // 2)
    label_range = N_LABELS
    max_delta = label_range - MIN_ENTRIES_PER_DELTA
    predicted_spacing = label_range / mode_pairs
    predicted_first = label_range / (2 * mode_pairs + 1)
    band_lo = max(4, int(math.floor(predicted_first)))
    band_hi = min(int(math.floor(max_delta / 2)),
                  int(math.ceil(BAND_HI_MULTIPLE * label_range / mode_pairs)))
    bins = band_bins(max_delta, band_lo, band_hi)
    print("[calibrate] ambient dimension %d, retained mode pairs %d, predicted spacing %.1f"
          % (ambient, mode_pairs, predicted_spacing))
    print("[calibrate] searched band %d to %d characters, %d whole cycle counts inside it"
          % (band_lo, band_hi, len(bins)))

    band_decision = decide_band(band_hi)
    values = {
        "ridge_strength": alpha_star,
        "ridge_grid": RIDGE_GRID,
        "cross_validated_quality_by_strength_and_readout":
            {"%g|%d" % (a, l): round(cv_mean[(a, l)], 6) for a in RIDGE_GRID for l in range(n_read)},
        "training_best_readout": train_best_layer,
        "training_best_quality": round(train_best_r2, 6),
        "fold_count": N_FOLDS,
        "seed": SEED,
        "character_window_low": window_lo,
        "character_window_high": window_lo + N_LABELS - 1,
        "character_window_min_training_positions": int(min(counts)),
        "token_window_high": token_hi,
        "ambient_dimension_at_95_percent": ambient,
        "variance_share_first_ten": [round(s, 6) for s in share[:10]],
        "retained_mode_pairs": mode_pairs,
        "predicted_sidelobe_spacing": round(predicted_spacing, 3),
        "predicted_first_sidelobe": round(predicted_first, 3),
        "max_delta": max_delta,
        "min_entries_per_delta": MIN_ENTRIES_PER_DELTA,
        "detrend_method": "polynomial",
        "detrend_degree": DETREND_DEGREE,
        "band_low_chars": band_lo,
        "band_high_chars": band_hi,
        "band_cycle_counts": [{"cycles": k, "period": round(p, 3)} for k, p in bins],
        "n_band_bins": len(bins),
        "predicted_spacing_inside_band": bool(band_lo <= predicted_spacing <= band_hi),
        "permutation_draws": N_PERMUTATIONS,
    }
    notes = [
        "Every number here comes from training documents only. No held-out number exists yet.",
        "The character-count window's low value is fixed here because definition 1 makes the "
        "lowest possible label 1, while definition 3 names the range 0 to 139.",
        "The searched band's low end is the predicted first sidelobe, from the measured ambient "
        "dimension. Its high end is the shorter of three times the predicted spacing and the "
        "longest period the profile shows twice.",
        "A moving-average detrend is refused. Subtracting a window of W removes periods longer "
        "than about W, which is exactly where this study predicts the pattern lies.",
    ]
    if predicted_spacing > band_hi:
        notes.append("The predicted spacing of %.1f characters is longer than the %d-character top "
                     "of the searched band. A 140-label range cannot show that period twice, so the "
                     "measurement below covers shorter periods only." % (predicted_spacing, band_hi))
    if len(bins) <= 2:
        notes.append("Only %d whole cycle counts fall inside the band, so the comparison rests on "
                     "%d number(s) per draw." % (len(bins), len(bins)))
    notes.append(RERUN_NOTE)

    write_record(
        prov, out_dir, base_cell("calibrate", "constants"),
        inputs=record_inputs(prov, stim, design,
                             extra=[(work_dir / "totals.npz", n_read), (work_dir / "w140_meta.npz", n_rows)]),
        metric="Every remaining constant of the fit and of the ripple readout, fixed on training documents only.",
        values=values,
        threshold={"band_high_chars_floor": BAND_UPPER_FLOOR},
        decision_rule=RULE_BAND,
        decision=band_decision,
        notes=" ".join(notes),
    )
    write_record(
        prov, out_dir, base_cell("calibrate", "search_band", endpoint=2),
        inputs=record_inputs(prov, stim, design),
        metric="Top of the calibrated search band, in characters.",
        values={"band_high_chars": band_hi, "band_low_chars": band_lo,
                "ambient_dimension_at_95_percent": ambient, "retained_mode_pairs": mode_pairs,
                "predicted_sidelobe_spacing": round(predicted_spacing, 3)},
        threshold={"floor_chars": BAND_UPPER_FLOOR},
        decision_rule=RULE_BAND,
        decision=band_decision,
        notes=("The floor of 43 characters guards against repeating an earlier band that "
               "suppressed the predicted periods. " + RERUN_NOTE),
    )
    with open(Path(out_dir) / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump({"decision": band_decision, "values": values}, fh, indent=2, sort_keys=True)
    if band_decision != "BAND_OK":
        raise Halt("the calibrated band top of %d characters is below the floor of %d; the design "
                   "says halt and recalibrate before any held-out number" % (band_hi, BAND_UPPER_FLOOR))
    return 0


# ---------------------------------------------------------------------------
# Part 10. Stage three: the held-out fit, the ripple, the shuffles, the controls.
# ---------------------------------------------------------------------------

def shuffle_within_docs(labels, docs, rng):
    """Permute labels among the positions of each document, leaving counts intact."""
    import numpy as np
    out = labels.copy()
    for d in np.unique(docs):
        sel = np.nonzero(docs == d)[0]
        out[sel] = labels[sel][rng.permutation(sel.size)]
    return out


def ripple_with_null(x, labels, docs, window_lo, n_slots, max_delta, band_lo, band_hi, n_draws, rng):
    """Observed ripple and its label-shuffled comparison, on one set of positions."""
    import numpy as np
    slots = slots_array(labels, window_lo, n_slots)
    mat, counts = means_by_slot(x, slots, n_slots)
    centred = mat - mat.mean(axis=0, keepdims=True)
    observed = ripple_stat(cosine_matrix(centred), max_delta, band_lo, band_hi, keep_profile=True)
    draws = []
    for _ in range(n_draws):
        shuffled = shuffle_within_docs(labels, docs, rng)
        s2 = slots_array(shuffled, window_lo, n_slots)
        m2, _ = means_by_slot(x, s2, n_slots)
        c2 = m2 - m2.mean(axis=0, keepdims=True)
        draws.append(ripple_stat(cosine_matrix(c2), max_delta, band_lo, band_hi)["amplitude"])
    rank = null_rank(observed["amplitude"], draws)
    return observed, draws, rank, [int(c) for c in counts]


def stage_endpoints(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        raise Halt("no calibration record at %s; the calibrate stage must run first" % calib_path)
    with open(calib_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    if calib.get("decision") != "BAND_OK":
        raise Halt("the calibration record says %r; no held-out number may be computed"
                   % calib.get("decision"))
    cv = calib["values"]
    alpha = float(cv["ridge_strength"])
    window_lo = int(cv["character_window_low"])
    token_hi = int(cv["token_window_high"])
    max_delta = int(cv["max_delta"])
    band_lo = float(cv["band_low_chars"])
    band_hi = float(cv["band_high_chars"])

    tot = load_totals(work_dir)
    groups = tot["groups"]
    fit_idx = [groups.index("fold%d" % f) for f in range(N_FOLDS)]
    test_idx = [groups.index("test")]
    n_read = tot["gram"].shape[1]

    # Endpoint 3: the per-readout held-out fit for the character count.
    per_layer = []
    for layer in range(n_read):
        g, v, c, ls, lq, n = sum_groups(tot, fit_idx, layer)
        ge, ve, ce, lse, lqe, ne = sum_groups(tot, test_idx, layer)
        w, off = ridge_fit_from_stats(g, v, c, ls, n, alpha)
        r2, sse, sstot = r2_from_stats(w, off, ge, ve, ce, lse, lqe, ne)
        per_layer.append({"readout": layer, "held_out_quality": round(float(r2), 6)})
    best = max(per_layer, key=lambda r: r["held_out_quality"])
    best_layer = int(best["readout"])
    best_r2 = float(best["held_out_quality"])
    anchor_decision = decide_anchor(best_r2)
    print("[endpoints] best readout %d, held-out fit quality %.4f -> %s"
          % (best_layer, best_r2, anchor_decision))
    write_record(
        prov, out_dir, base_cell("endpoints", "count_curve", endpoint=3),
        inputs=record_inputs(prov, stim, design, extra=[(work_dir / "totals.npz", n_read)]),
        metric="Held-out fit quality of the calibrated ridge fit for the character count, per readout.",
        values={"best_readout": best_layer, "best_held_out_quality": round(best_r2, 6),
                "ridge_strength": alpha, "training_positions": int(tot["n_pos"][fit_idx].sum()),
                "held_out_positions": int(tot["n_pos"][test_idx].sum()),
                "prior_reported_value": 0.724, "prior_reported_readout": 6},
        threshold={"low": ANCHOR_LO, "high": ANCHOR_HI},
        decision_rule=RULE_ANCHOR,
        decision=anchor_decision,
        rows=per_layer,
        notes=("The prior value 0.724 at readout 6 comes from another group's work on the same "
               "public checkpoint, with unpublished text and unpublished readout numbering. The "
               "readout where the fit peaks is recorded and never judged. " + RERUN_NOTE),
    )

    # Shared inputs for the ripple, its controls, and the fitted-probe cross-check.
    meta = np.load(work_dir / "w140_meta.npz", allow_pickle=False)
    is_test = meta["is_test"]
    docs_all = meta["doc"]
    char_all = meta["char_label"]
    token_all = meta["token_label"]
    n_rows = int(char_all.size)
    x_best = read_w140_layer(work_dir, best_layer, n_rows)
    test_mask = is_test == 1
    x_test = x_best[test_mask]
    docs_test = docs_all[test_mask]
    char_test = char_all[test_mask]
    token_test = token_all[test_mask]
    rng = np.random.default_rng(SEED)

    # Endpoint 4: the ripple in the average vectors, against its shuffled comparison.
    observed, draws, rank, counts = ripple_with_null(
        x_test, char_test, docs_test, window_lo, N_LABELS, max_delta,
        band_lo, band_hi, N_PERMUTATIONS, rng)
    ripple_decision = decide_ripple(rank)
    print("[endpoints] ripple size %.6g at period %.2f characters, rank %d of %d -> %s"
          % (observed["amplitude"], observed["wavelength"], rank, N_PERMUTATIONS, ripple_decision))
    write_record(
        prov, out_dir, base_cell("endpoints", "ripple_vs_shuffle", endpoint=4),
        inputs=record_inputs(prov, stim, design, extra=[(work_dir / "w140_meta.npz", n_rows)]),
        metric="Rank of the observed ripple size among label-shuffled draws, at the best readout.",
        values={"readout": best_layer, "amplitude": observed["amplitude"],
                "wavelength_chars": observed["wavelength"], "cycles": observed["cycles"],
                "n_band_bins": observed["n_bins"], "rank": rank, "draws": N_PERMUTATIONS,
                "empirical_share": round(rank / (N_PERMUTATIONS + 1.0), 5),
                "null_max": max(draws), "null_median": sorted(draws)[len(draws) // 2],
                "positions_per_label_min": min(counts), "positions_per_label_max": max(counts),
                "profile": [round(v, 8) for v in observed["profile"]],
                "in_band": observed["in_band"]},
        threshold={"rank_at_most": NULL_RANK_MAX, "draws": N_PERMUTATIONS},
        decision_rule=RULE_RIPPLE,
        decision=ripple_decision,
        rows=[{"draw": i, "amplitude": a} for i, a in enumerate(draws)],
        notes=("Shuffling permutes the character-count labels inside each document, so every "
               "label keeps exactly the same number of positions. With 200 draws the finest "
               "share this can resolve is about 0.5 percent. " + RERUN_NOTE),
    )

    # Endpoint 5: the same readout and text, with token position as the label.
    variants_path = Path(out_dir) / "rows_variants.jsonl"
    if not variants_path.exists():
        raise Halt("no per-variant rows at %s; the extract stage must run first" % variants_path)
    chars_total = 0
    with open(variants_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["width"] == RIPPLE_WIDTH and row["split"] == "test":
                chars_total += row["n_chars"]
    chars_per_token = chars_total / float(token_test.size) if token_test.size else None
    token_note = []
    token_values = {"readout": best_layer, "token_window_high": token_hi,
                    "chars_per_token": round(chars_per_token, 4) if chars_per_token else None}
    token_wavelength_chars = None
    token_cleared = None
    try:
        if chars_per_token is None:
            raise Halt("no held-out positions at the widest wrapping")
        tok_max_delta = token_hi - MIN_ENTRIES_PER_DELTA
        if tok_max_delta < DETREND_DEGREE + 2:
            raise Halt("a %d-value token-count range leaves only %d distances, too few to detrend"
                       % (token_hi, tok_max_delta))
        tok_band_lo = band_lo / chars_per_token
        tok_band_hi = band_hi / chars_per_token
        tok_obs, tok_draws, tok_rank, tok_counts = ripple_with_null(
            x_test, token_test, docs_test, 1, token_hi, tok_max_delta,
            tok_band_lo, tok_band_hi, N_PERMUTATIONS, rng)
        token_cleared = decide_ripple(tok_rank) == "NULL_CLEARED"
        token_wavelength_chars = tok_obs["wavelength"] * chars_per_token
        token_values.update({
            "max_delta_tokens": tok_max_delta,
            "band_low_tokens": round(tok_band_lo, 4), "band_high_tokens": round(tok_band_hi, 4),
            "amplitude": tok_obs["amplitude"], "wavelength_tokens": tok_obs["wavelength"],
            "wavelength_chars": round(token_wavelength_chars, 4),
            "n_band_bins": tok_obs["n_bins"], "rank": tok_rank, "draws": N_PERMUTATIONS,
            "positions_per_label_min": min(tok_counts), "positions_per_label_max": max(tok_counts)})
        token_note.append("The token-count period is turned into characters with the measured "
                          "%.3f characters per token, because the design's margin of 3 is in "
                          "characters." % chars_per_token)
        token_note.append("The searched band is the calibrated character band divided by that same "
                          "factor, so both labellings are searched over one range of periods.")
    except Halt as exc:
        token_values["not_computable_reason"] = str(exc)
        token_note.append("The token-position control could not be computed: %s" % exc)
    confound_decision = decide_confound(
        observed["wavelength"], token_wavelength_chars,
        ripple_decision == "NULL_CLEARED", token_cleared)
    token_values.update({"character_wavelength_chars": observed["wavelength"],
                         "character_cleared_shuffle": ripple_decision == "NULL_CLEARED",
                         "token_cleared_shuffle": token_cleared})
    token_note.append("One line at the widest wrapping holds about 140 characters but only about "
                      "%d tokens, so the two labellings cannot be searched over the same number of "
                      "periods." % token_hi)
    token_note.append(RERUN_NOTE)
    print("[endpoints] token-position control -> %s" % confound_decision)
    write_record(
        prov, out_dir, base_cell("endpoints", "token_position_control", endpoint=5),
        inputs=record_inputs(prov, stim, design, extra=[(work_dir / "w140_meta.npz", n_rows)]),
        metric="Ripple period and shuffle result under character-count labels against token-position labels.",
        values=token_values,
        threshold={"margin_chars": CONFOUND_MARGIN_CHARS},
        decision_rule=RULE_CONFOUND,
        decision=confound_decision,
        notes=" ".join(token_note),
    )

    # Endpoint 6: the same readout, with a fitted many-class probe instead of averages.
    sec_values = {"readout": best_layer, "n_classes": N_LABELS}
    sec_note = []
    sec_wavelength = None
    try:
        from sklearn.linear_model import LogisticRegression
        train_mask = is_test == 0
        x_tr = x_best[train_mask]
        slots_tr = slots_array(char_all[train_mask], window_lo, N_LABELS)
        keep = slots_tr >= 0
        y_tr = slots_tr[keep]
        present = sorted(set(int(v) for v in y_tr.tolist()))
        if len(present) != N_LABELS:
            raise Halt("only %d of %d label classes appear in the training positions" % (len(present), N_LABELS))
        clf = LogisticRegression(solver="lbfgs", max_iter=2000)
        t0 = time.time()
        clf.fit(x_tr[keep], y_tr)
        coef = clf.coef_
        if coef.shape != (N_LABELS, EXPECT_HIDDEN):
            raise Halt("the fitted probe returned weights of shape %r, expected (%d, %d)"
                       % (coef.shape, N_LABELS, EXPECT_HIDDEN))
        sec = ripple_stat(cosine_matrix(coef - coef.mean(axis=0, keepdims=True)),
                          max_delta, band_lo, band_hi, keep_profile=True)
        sec_wavelength = sec["wavelength"]
        sec_values.update({"amplitude": sec["amplitude"], "wavelength_chars": sec["wavelength"],
                           "cycles": sec["cycles"], "n_band_bins": sec["n_bins"],
                           "training_positions": int(keep.sum()),
                           "regularisation_strength_C": float(clf.C),
                           "iterations": int(np.max(clf.n_iter_)), "fit_seconds": round(time.time() - t0, 1),
                           "profile": [round(v, 8) for v in sec["profile"]]})
        sec_note.append("The probe's penalty strength is the library default of 1.0. The design "
                        "fixes no value for it, so none was invented.")
    except Exception as exc:                                  # non-gating arm: record, do not stop
        sec_values["not_computable_reason"] = "%s: %s" % (type(exc).__name__, exc)
        sec_note.append("The fitted-probe cross-check could not be computed: %s" % exc)
    secondary_decision = decide_secondary(observed["wavelength"], sec_wavelength,
                                          ripple_decision == "NULL_CLEARED", None)
    sec_values.update({"character_wavelength_chars": observed["wavelength"],
                       "shuffle_comparison_for_probe": None})
    sec_note.append("The design computes no shuffle comparison for the fitted probe, while this "
                    "endpoint's rule asks whether both readouts clear one. So the presence half is "
                    "recorded as not decidable and only the periods are compared.")
    sec_note.append(RERUN_NOTE)
    print("[endpoints] fitted-probe cross-check -> %s" % secondary_decision)
    write_record(
        prov, out_dir, base_cell("endpoints", "fitted_probe_crosscheck", endpoint=6),
        inputs=record_inputs(prov, stim, design, extra=[(work_dir / "w140_meta.npz", n_rows)]),
        metric="Agreement on ripple presence and period between the average-vector readout and a fitted many-class probe.",
        values=sec_values,
        threshold={"margin_chars": SECONDARY_MARGIN_CHARS},
        decision_rule=RULE_SECONDARY,
        decision=secondary_decision,
        notes=" ".join(sec_note),
    )

    # The setup call is read back from its own record, never restated here.
    setup_path = Path(out_dir) / (prov.cell_slug(base_cell("extract", "integrity_check", endpoint=1)) + ".json")
    setup_decision = "RECORD_ABSENT"
    if setup_path.exists():
        with open(setup_path, "r", encoding="utf-8") as fh:
            setup_decision = json.load(fh).get("decision", "RECORD_UNREADABLE")

    summary = {
        "endpoint_1_setup": setup_decision,
        "endpoint_2_search_band": calib.get("decision"),
        "endpoint_3_count_curve": anchor_decision,
        "endpoint_4_ripple_vs_shuffle": ripple_decision,
        "endpoint_5_token_position_control": confound_decision,
        "endpoint_6_fitted_probe_crosscheck": secondary_decision,
        "best_readout": best_layer,
        "best_held_out_quality": round(best_r2, 6),
        "ripple_wavelength_chars": observed["wavelength"],
        "ripple_rank_of_draws": rank,
    }
    write_record(
        prov, out_dir, base_cell("endpoints", "run_summary"),
        inputs=record_inputs(prov, stim, design),
        metric="One call per endpoint, from the records written beside this one.",
        values=summary,
        threshold={},
        decision_rule=("Each endpoint's own rule, restated in its own record. This record only "
                       "gathers the calls; it computes no new number."),
        decision=("VOID_ANCHOR_FAILED" if anchor_decision != "ANCHOR_IN_BAND" else
                  "GATE_OPEN" if ripple_decision == "NULL_CLEARED" else "GATE_CLOSED"),
        notes=("A failed count-curve endpoint makes the ripple result uninterpretable as evidence "
               "about the model, and is scored as void. " + RERUN_NOTE),
    )
    print("[endpoints] summary %s" % json.dumps(summary))
    return 0


# ---------------------------------------------------------------------------
# Part 11. Self-test. Standard library only. No model, no graphics card.
# ---------------------------------------------------------------------------

def _check(fails, name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) and isinstance(got, (int, float)) else (got == want)
    print("  %-58s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def selftest() -> int:
    fails = []
    print("== position labeller ==")
    # Plain two-line text, one character per token.
    text = "ab\ncd"
    spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    ch, tk, ln = labels_for_spans(text, spans)
    _check(fails, "columns on a two-line text", ch, [1, 2, 3, 1, 2])
    _check(fails, "token counts on a two-line text", tk, [1, 2, 3, 1, 2])
    _check(fails, "line numbers on a two-line text", ln, [0, 0, 0, 1, 1])
    # The opening boundary of the first line is implied, not present in the text.
    # This is the same shape as a template that already emitted an opening marker,
    # so only the closing one appears in the text.
    ch1, tk1, _ = labels_for_spans("abcd", [(0, 2), (2, 4)])
    _check(fails, "implied opening boundary, first line columns", ch1, [2, 4])
    _check(fails, "implied opening boundary, first line token counts", tk1, [1, 2])
    # A line break swallowed inside one token: the boundary is inside a unit.
    ch2, tk2, ln2 = labels_for_spans("ab\ncd", [(0, 2), (2, 4), (4, 5)])
    _check(fails, "swallowed line break, columns", ch2, [2, 1, 2])
    _check(fails, "swallowed line break, token counts", tk2, [1, 1, 2])
    _check(fails, "swallowed line break, line numbers", ln2, [0, 1, 1])
    # A line break as its own token takes the column after the last visible character.
    ch3, _, _ = labels_for_spans("abc\nd", [(0, 3), (3, 4), (4, 5)])
    _check(fails, "line-break token takes the next column", ch3, [3, 4, 1])
    # An empty span is never labelled and never counted.
    ch4, tk4, _ = labels_for_spans("ab", [(0, 0), (0, 2)])
    _check(fails, "empty span is unlabelled", ch4, [None, 2])
    _check(fails, "empty span is not counted", tk4, [None, 1])

    print("== degenerate-input detector ==")
    _check(fails, "length-cap case flags", degenerate_flags(2048, 2048, ["a", "b"]), ["token_cap"])
    _check(fails, "under the length cap does not flag", degenerate_flags(700, 2048, ["a", "b", "c"]), [])
    rep = ["x", "y"] * 7
    _check(fails, "repeated-sequence case flags", degenerate_flags(14, 2048, rep), ["repeated_sequence"])
    _check(fails, "seven back-to-back copies counted", max_block_repeats(rep), 7)
    _check(fails, "prose-like tokens are clean",
           degenerate_flags(9, 2048, ["It", " will", " be", " seen", " that", " this", " mere", " b", "\n"]), [])
    _check(fails, "both cases can flag together",
           degenerate_flags(2048, 2048, ["q"] * 6), ["token_cap", "repeated_sequence"])

    print("== exact-match checks and exact slot assignment ==")
    _check(fails, "hash equal", exact_hash_match("a" * 64, "a" * 64), True)
    _check(fails, "hash one character off", exact_hash_match("a" * 63 + "b", "a" * 64), False)
    _check(fails, "configuration match", exact_config_match(24, 1024), True)
    _check(fails, "configuration layer mismatch", exact_config_match(23, 1024), False)
    _check(fails, "configuration width mismatch", exact_config_match(24, 2048), False)
    _check(fails, "lowest window value maps to slot 0", label_slot(1, 1, 140), 0)
    _check(fails, "highest window value maps to the last slot", label_slot(140, 1, 140), 139)
    _check(fails, "one below the window is outside", label_slot(0, 1, 140), None)
    _check(fails, "one above the window is outside", label_slot(141, 1, 140), None)
    _check(fails, "a fractional label is never accepted", label_slot(1.0, 1, 140), None)
    _check(fails, "a true value is never accepted as 1", label_slot(True, 1, 140), None)
    counts, slots = slot_counts([1, 2, 2, 5, 0], 1, 4)
    _check(fails, "slot counts", counts, [1, 2, 0, 0])
    _check(fails, "slot indices", slots, [0, 1, 1, None, None])
    _check(fails, "window start when the lowest value is empty",
           first_full_window({1: 3, 2: 3, 3: 3}, 3, 1), 1)
    _check(fails, "window start when everything is populated",
           first_full_window({0: 1, 1: 1, 2: 1}, 3, 1), 0)
    _check(fails, "no full window", first_full_window({1: 1, 3: 1}, 3, 1), None)

    print("== ripple statistic ==")
    # A matrix whose entries depend only on distance reproduces that profile exactly.
    n = 40
    want = [1.0 / (1 + d) for d in range(n)]
    mat = [[want[abs(i - j)] for j in range(n)] for i in range(n)]
    prof = offdiag_profile(mat, 20)
    _check(fails, "profile at distance 1", prof[0], want[1], 1e-12)
    _check(fails, "profile at distance 20", prof[19], want[20], 1e-12)
    # A straight line is removed by the detrend; a pure wave is not.
    line = [3.0 + 0.5 * i for i in range(60)]
    _check(fails, "a straight line detrends to zero", max(abs(v) for v in poly_detrend(line, 2)), 0.0, 1e-9)
    wave = [math.cos(2 * math.pi * i / 12.0) for i in range(60)]
    det = poly_detrend(wave, 2)
    power = dft_power(det)
    peak = max(power, key=lambda k: power[k])
    _check(fails, "a 12-step wave peaks at 5 cycles over 60 steps", peak, 5)
    _check(fails, "that peak is a period of 12", 60 / peak, 12.0, 1e-9)
    _check(fails, "cycle counts inside a band", band_bins(120, 28, 60), [(2, 60.0), (3, 40.0), (4, 30.0)])
    _check(fails, "an empty band yields no cycle counts", band_bins(120, 70, 100), [])
    # A distance-dependent matrix carrying a known ripple recovers that period.
    n = 160
    entries = [1.0 - 0.004 * d + 0.05 * math.cos(2 * math.pi * d / 40.0) for d in range(n)]
    mat = [[entries[abs(i - j)] for j in range(n)] for i in range(n)]
    stat = ripple_stat(mat, 120, 28, 60)
    _check(fails, "planted ripple period recovered", stat["wavelength"], 40.0, 1e-9)
    _check(fails, "planted ripple searched over three cycle counts", stat["n_bins"], 3)
    try:
        ripple_stat(mat, 120, 70, 100)
        fails.append("an empty band should halt")
        print("  %-58s FAIL no halt" % "empty band halts")
    except Halt:
        print("  %-58s ok" % "empty band halts")

    print("== endpoint arithmetic on planted values ==")
    _check(fails, "endpoint 1 all three match", decide_integrity(True, 24, 1024), "SETUP_OK")
    _check(fails, "endpoint 1 hash differs", decide_integrity(False, 24, 1024), "SETUP_FAILURE")
    _check(fails, "endpoint 1 layer count differs", decide_integrity(True, 12, 1024), "SETUP_FAILURE")
    _check(fails, "endpoint 2 band top 60", decide_band(60), "BAND_OK")
    _check(fails, "endpoint 2 band top exactly 43", decide_band(43), "BAND_OK")
    _check(fails, "endpoint 2 band top 35", decide_band(35), "BAND_BELOW_FLOOR")
    _check(fails, "endpoint 3 quality 0.724", decide_anchor(0.724), "ANCHOR_IN_BAND")
    _check(fails, "endpoint 3 quality 0.574 exactly", decide_anchor(0.574), "ANCHOR_IN_BAND")
    _check(fails, "endpoint 3 quality 0.874 exactly", decide_anchor(0.874), "ANCHOR_IN_BAND")
    _check(fails, "endpoint 3 quality 0.573", decide_anchor(0.573), "ANCHOR_OUT_OF_BAND")
    _check(fails, "endpoint 3 quality 0.9", decide_anchor(0.9), "ANCHOR_OUT_OF_BAND")
    _check(fails, "endpoint 3 quality below zero", decide_anchor(-0.2), "ANCHOR_OUT_OF_BAND")
    draws = [float(i) for i in range(200)]                      # 0.0 up to 199.0
    _check(fails, "rank of a value above every draw", null_rank(500.0, draws), 1)
    _check(fails, "rank of a value above all but nine", null_rank(190.5, draws), 10)
    _check(fails, "rank of a value above all but ten", null_rank(189.5, draws), 11)
    _check(fails, "rank counts ties against the observed value", null_rank(199.0, draws), 2)
    _check(fails, "endpoint 4 rank 1", decide_ripple(1), "NULL_CLEARED")
    _check(fails, "endpoint 4 rank 10", decide_ripple(10), "NULL_CLEARED")
    _check(fails, "endpoint 4 rank 11", decide_ripple(11), "NULL_NOT_CLEARED")
    _check(fails, "endpoint 5 periods 60 and 31", decide_confound(60.0, 31.0, True, True),
           "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "endpoint 5 periods 60 and 57.5", decide_confound(60.0, 57.5, True, True),
           "CONFOUNDED_WITH_TOKEN_POSITION")
    _check(fails, "endpoint 5 periods 60 and 56.5", decide_confound(60.0, 56.5, True, True),
           "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "endpoint 5 periods 60 and 57.5 with one clearing",
           decide_confound(60.0, 57.5, True, False), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "endpoint 5 periods differ by exactly 3", decide_confound(60.0, 57.0, True, True),
           "CONFOUNDED_WITH_TOKEN_POSITION")
    _check(fails, "endpoint 5 control absent", decide_confound(60.0, None, True, None),
           "TOKEN_CONTROL_NOT_COMPUTABLE")
    _check(fails, "endpoint 6 periods within 5", decide_secondary(60.0, 56.0, True, None),
           "PERIODS_AGREE_PRESENCE_NOT_DECIDABLE")
    _check(fails, "endpoint 6 periods differ by more than 5", decide_secondary(60.0, 54.0, True, None),
           "PERIODS_DISAGREE_PRESENCE_NOT_DECIDABLE")
    _check(fails, "endpoint 6 probe absent", decide_secondary(60.0, None, True, None),
           "SECONDARY_NOT_COMPUTABLE")
    _check(fails, "endpoint 6 with a comparison, both clear and agree",
           decide_secondary(60.0, 58.0, True, True), "BOTH_CLEAR_PERIODS_AGREE")
    _check(fails, "endpoint 6 with a comparison, one clears",
           decide_secondary(60.0, 58.0, True, False), "READOUTS_DISAGREE_ON_PRESENCE")

    print("== supporting checks ==")
    try:
        prov = load_provenance()
        print("  %-58s ok" % "record writer importable")
        for name in ("write_result", "describe_input", "cell_slug"):
            if not hasattr(prov, name):
                fails.append("record writer lacks " + name)
                print("  %-58s FAIL" % ("record writer has " + name))
    except Exception as exc:
        fails.append("record writer import: %s" % exc)
        print("  %-58s FAIL %s" % ("record writer importable", exc))
    try:
        import numpy as np
    except ImportError:
        print("  %-58s skipped (numpy absent)" % "ridge fit from running totals")
    else:
        import random as _r
        _r.seed(7)
        rows = [[_r.uniform(-1, 1), _r.uniform(-1, 1)] for _ in range(50)]
        ys = [2.0 * a - 3.0 * b + 0.5 + _r.uniform(-0.01, 0.01) for a, b in rows]
        xm = np.array(rows)
        ym = np.array(ys)
        stats = (xm.T @ xm, xm.sum(0), xm.T @ ym, float(ym.sum()), float((ym * ym).sum()), len(ys))
        w, off = ridge_fit_from_stats(stats[0], stats[1], stats[2], stats[3], stats[5], 1e-6)
        r2, sse, sstot = r2_from_stats(w, off, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5])
        pred = xm @ w + off
        direct_sse = float(((pred - ym) ** 2).sum())
        direct_r2 = 1.0 - direct_sse / float(((ym - ym.mean()) ** 2).sum())
        _check(fails, "sum of squared errors from running totals", sse, direct_sse, 1e-6)
        _check(fails, "fit quality from running totals", r2, direct_r2, 1e-9)
        _check(fails, "planted weights recovered", float(abs(w[0] - 2.0) + abs(w[1] + 3.0)), 0.0, 0.02)

    print("selftest:", "PASS" if not fails else "%d FAILURES" % len(fails))
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Part 12. Entry point.
# ---------------------------------------------------------------------------

STAGES = {"extract": stage_extract, "calibrate": stage_calibrate, "endpoints": stage_endpoints}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES) + ["all", "generate"],
                    help="which stage to run")
    ap.add_argument("--selftest", action="store_true", help="standard-library checks only")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", RESULTS_DIR_DEFAULT))
    ap.add_argument("--stimulus", default=None, help="path to the committed text file")
    ap.add_argument("--design-record", default=None, help="path to the committed design record")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.stage:
        ap.error("give --stage extract, calibrate, endpoints, or all; or give --selftest")
    if args.stage == "generate":
        print("There is no generation stage. The design says: no text generation anywhere. "
              "One deterministic forward pass per text variant, on fixed text. The stage that "
              "sits in this position is calibrate, which fixes constants on training documents "
              "only, before any held-out number exists.", file=sys.stderr)
        return 2

    order = ["extract", "calibrate", "endpoints"] if args.stage == "all" else [args.stage]
    for name in order:
        print("=== stage %s ===" % name)
        started = time.time()
        try:
            rc = STAGES[name](args)
        except Halt as exc:
            print("HALT in stage %s: %s" % (name, exc), file=sys.stderr)
            return 3
        if rc:
            return rc
        print("=== stage %s done in %.1f s ===" % (name, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
