#!/usr/bin/env python3
"""Measures whether a repeating pattern in one language model's character-count code
keeps one wavelength as the probe window widens.

WHAT THIS RUNS. One public checkpoint of one open language model reads fixed,
committed prose. The prose is hard-wrapped at nine line widths. At every token
position we know how many characters have passed since the last line break; that
count is the position's label. A ridge regression (a linear fit with a penalty on
large weights) predicts the label from the model's internal vector at that position.
At the best-scoring layer we then average the internal vectors of every position that
carries the same label, compare every pair of those averages by cosine similarity,
and look for a repeating pattern in that similarity structure. The pattern is
measured at three probe windows: prose wrapped at 140, at 200, and at 280 characters.
A label-shuffling comparison says whether the pattern is larger than chance at each
window. The question is whether a pattern that beats chance keeps the same wavelength
in characters across the three windows (a property of the model) or scales with the
window (an artefact of the probe).

THE SPECIFICATION is three committed files, and this file changes none of them:
  out/seedC1_design_diff_triage.json   the design record: what is measured
  out/CONVENTIONS_manifold_c.md        where records go and how they are written
  out/GATE_COMMIT_seedC1.md            the human gate commit: pinned arithmetic,
                                       pinned stage order, pinned self-test cases,
                                       and the numbers that are recorded but never
                                       allowed into a decision
Every constant the design marks as a guess is binding here.

NO TEXT IS GENERATED ANYWHERE. The design says: "No text is generated. Every
measurement reads hidden activations from one deterministic forward pass per document
per wrap width." So this file has no sampling settings, no decoding and no answer
parsing. The build template that produced it asks for a stage named "generate"; the
gate commit's section F names the three stages instead, and the stage that sits in
that position is the calibration stage, which fixes every remaining constant on
training documents alone before any held-out number exists. Asking for the template
name prints that explanation and exits non-zero.

STAGES, in the pinned order, each runnable on its own:
  --stage extract    checks the text file and the downloaded model, then runs one
                     forward pass per text variant and saves what the later stages
                     need. 48 documents at nine wrap widths is 432 forward passes.
  --stage calibrate  reads training documents only. Fixes the ridge strength, the
                     label window and profile length per probe window, the shuffle
                     seeds, the anchor layer, the five resplit values, their spread
                     and the anchor band, and writes that record to disk.
  --stage endpoints  the only stage that reads held-out documents. Selects the main
                     layer, computes the anchor statistic and its gate, then the
                     repeating-pattern statistic, its shuffle comparisons, the
                     token-position control and the represented-range check.
  --stage all        the three above, in that order.
  --selftest         standard library only. Loads no model, needs no graphics card,
                     exits non-zero on any failure.

A VOID IS CONTAGIOUS. The anchor gate is an instrument check. If it fails, every
result this run produces is void as an instrument fault and is not evidence about the
model. Every record written after the gate then carries run_valid=false in its own
values, per-probe-window files included, not only the roll-up.

A second run of this script on the same rented box is a rerun of the same instrument.
It is not a replication, and no output here calls it one.
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
# Constants. The source of each is marked.
#   DESIGN  the committed design record.
#   GATE    the committed human gate commit.
#   MEMORY  recalled from a paper and checked against the downloaded model.
#   GUESS   fixed by the design and binding here.
# ---------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-410m-deduped"          # DESIGN
REVISION = "step143000"                              # DESIGN, final pretraining checkpoint
EXPECT_LAYERS = 24                                   # MEMORY, Pythia paper arXiv 2304.01373 table 1
EXPECT_HIDDEN = 1024                                 # MEMORY, same table
TORCH_DTYPE = "float16"                              # DESIGN, load in float16 on the graphics card

STIMULUS_NAME = "stimulus_linewrap_v2.json"          # DESIGN
STIMULUS_SHA256 = "b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd"   # DESIGN
DESIGN_NAME = "seedC1_design_diff_triage.json"
DESIGN_SHA256 = "06fae77e203c087f1f85e041966c967ce64e83f2d82712f738759271ef8a5107"      # hashed on this workstation, 2026-08-26
GATE_NAME = "GATE_COMMIT_seedC1.md"
GATE_SHA256 = "5fe5efdcbc0ef85df6b22cc59987a8db7c93ff3fd09e48fa65470e83a80dcc90"        # hashed on this workstation, 2026-08-26

WIDTHS = [20, 40, 60, 80, 100, 120, 140, 200, 280]   # DESIGN, the nine wrap widths
APERTURES = [140, 200, 280]                          # DESIGN, the three probe windows
ANCHOR_WIDTH = 140                                   # DESIGN, the anchor is width 140 alone
POOLED_TO_140 = [20, 40, 60, 80, 100, 120, 140]      # GATE A6 item 3, recorded, never gated
N_DOCS = 48                                          # DESIGN
N_TRAIN_DOCS = 36                                    # DESIGN, documents 0 to 35
N_TEST_DOCS = 12                                     # DESIGN, documents 36 to 47

RIDGE_GRID = [1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]   # DESIGN GUESS, binding
N_FOLDS = 5                                          # DESIGN GUESS, five-fold cross-validation
BASE_SEED = 0                                        # DESIGN GUESS, the one base seed
N_DRAWS = 200                                        # DESIGN GUESS, shuffle draws per window per labelling
NULL_RANK_MAX = 10                                   # DESIGN, rank 10 or better of 200 clears

ANCHOR_CENTRE = 0.781                                # DESIGN, the band's centre
ANCHOR_MAD_MULTIPLE = 3.0                            # DESIGN, half-width is 3 times the spread
N_RESPLITS = 5                                       # DESIGN, five resplits of the training documents
RESPLIT_HELDOUT_DOCS = 6                             # DESIGN GUESS, documents held out per resplit
NARROW_BAND_FLAG_AT = 0.02                           # GATE D1, recorded flag, never gated

PERIOD_MIN = 4.0                                     # DESIGN, shortest period the statistic considers
DETREND_DEGREE = 2                                   # DESIGN, the best-fitting degree-2 polynomial
INVARIANCE_MAX_GAP_CHARS = 10.0                      # DESIGN GUESS, one shared wavelength
PROPORTION_MAX_RATIO_GAP = 0.1                       # DESIGN GUESS, wavelength scales with the window
CONFOUND_MARGIN_CHARS = 3.0                          # DESIGN GUESS, token-position control margin

BIN_EDGES = [(1, 35), (36, 70), (71, 105), (106, 140),
             (141, 175), (176, 210), (211, 245), (246, 280)]    # GATE E6, the eight bins
N_BIN_DRAWS = 50                                     # DESIGN, draws per bin
BIN_PERCENTILE = 95.0                                # DESIGN, a bin must beat this percentile of its own draws
LOW_BIN_MAX_LABEL = 140                              # DESIGN, a bin is low when it ends at or below this
HIGH_BIN_MIN_LABEL = 150                             # DESIGN, a bin is high when it starts above this
RANGE_APERTURE = 280                                 # DESIGN, the range check is at window 280
VARIANCE_TARGET = 0.95                               # recorded only: directions to reach 95 percent

REPEAT_BLOCK_MAX = 8                                 # GUESS, longest repeating block the detector looks for
REPEAT_MIN_RUNS = 6                                  # GUESS, that many back-to-back repeats counts as degenerate
CONTEXT_LIMIT_FALLBACK = 2048                        # MEMORY, Pythia context length; the download overrides it

PRIOR_POOLED_TO_140_VALUE = 0.781031                 # context only, from this project's own records
PRIOR_W140_ONLY_VALUES = [0.700601, 0.549, 0.664024]  # context only, from this project's own records

RESULTS_DIR_DEFAULT = "results_seedC1"

RERUN_NOTE = ("A further run of this script on the same box is a rerun of the same "
              "instrument, not a replication.")


class Halt(Exception):
    """Raised where a measurement is undefined or a committed check fails.

    Never caught to keep a run going. The stage that raises it exits non-zero.
    """


# ---------------------------------------------------------------------------
# Part 1. The position labeller.
#
# Two boundary cases decide every label on a line, and both are covered by the
# self-test: the first line of a document has no line break in front of it, so its
# opening boundary is implied; and one token can swallow a line break, so the
# boundary sits inside a unit rather than between two units. Getting either wrong
# shifts every label on a line and silently moves the whole measurement.
# ---------------------------------------------------------------------------

def line_columns(text: str):
    """For each character: its column on its line, and its line number.

    A character's column counts the characters on its line up to and including
    itself, so the first character of a line is column 1. A line-break character
    belongs to the line it ends and takes the next column after the last visible
    character, which is why a line-final break on a full line lands at the wrap width
    plus one. Counting restarts after each break.
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
    """Labels for token positions, from one character range per token.

    `spans` holds one (start, end) character range per token, in reading order, as a
    tokenizer that reports offsets gives them. Returns three lists:
      character labels: the column of the token's last character;
      token labels: how many tokens have ended on that line so far, counting this one;
      line numbers: which line the token's last character sits on.
    A token with an empty range gets None in all three and is never counted.
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


def count_labels(labels):
    """How many positions carry each label. Absent labels are skipped."""
    counts = {}
    for lab in labels:
        if lab is None:
            continue
        counts[lab] = counts.get(lab, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Part 2. The degenerate-input and degenerate-measurement detectors.
#
# Four things can quietly turn a number into nonsense, and each has a detector here:
# a text variant whose token count reaches the model's context limit, so the
# committed text would be cut short; a text variant holding the same short block of
# tokens many times over, which would flood a label with copies of one vector; a
# per-label average vector of exactly zero, which has no direction and so no cosine
# similarity; and a set of held-out labels with no spread, which leaves fit quality
# with a zero denominator.
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
    """Names of the degenerate conditions present in one text variant.

    An empty list means clean.
    """
    flags = []
    if n_tokens >= context_limit:
        flags.append("token_cap")
    if max_block_repeats(list(token_texts)) >= REPEAT_MIN_RUNS:
        flags.append("repeated_sequence")
    return flags


def zero_norm_rows(norms) -> list:
    """Indices of average vectors with no length, whose cosine similarity is undefined."""
    return [i for i, v in enumerate(norms) if not (v > 0.0)]


def has_label_spread(labels) -> bool:
    """True when held-out labels differ from one another, so fit quality has a denominator."""
    seen = set()
    for v in labels:
        seen.add(v)
        if len(seen) > 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Part 3. Exact-match checks, label windows and committed profile lengths.
#
# Nothing here is fuzzy, normalised or rounded. The text file's hash, the downloaded
# configuration's layer count and width, and a position's whole-number label against
# a window slot must all match exactly. An off-by-one in the last one moves every row
# of the similarity structure with no error anywhere.
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_hash_match(actual, expected) -> bool:
    return isinstance(actual, str) and isinstance(expected, str) and actual == expected


def exact_config_match(n_layers, hidden_size) -> bool:
    return n_layers == EXPECT_LAYERS and hidden_size == EXPECT_HIDDEN


def label_slot(label, window_lo: int, n_labels: int):
    """Slot of a label inside a window, or None when it falls outside.

    Membership is exact whole-number equality. A fractional value, a true-or-false
    value and an absent value are never accepted.
    """
    if label is None or isinstance(label, bool) or not isinstance(label, int):
        return None
    off = label - window_lo
    if 0 <= off < n_labels:
        return off
    return None


def slot_counts(labels, window_lo: int, n_labels: int):
    """Per-slot position counts, and the slot of every label (None when outside)."""
    counts = [0] * n_labels
    slots = []
    for lab in labels:
        s = label_slot(lab, window_lo, n_labels)
        slots.append(s)
        if s is not None:
            counts[s] += 1
    return counts, slots


def committed_label_window(train_counts):
    """The label window a probe window commits, from training documents only.

    The design's calibration rule: take the largest label value that occurs at that
    wrap width in training documents, counting the width plus one that a line-final
    line break lands on, and commit that count. The window is therefore every whole
    label value from 1 up to that largest value, and the count of those values is the
    committed span. Returns (window_low, n_labels, committed_span, gaps), where gaps
    lists any value inside the window that no training position carries.
    """
    present = sorted(int(k) for k in train_counts if int(train_counts[k]) > 0)
    if not present:
        raise Halt("no training position carries any label at this wrap width")
    top = present[-1]
    gaps = [v for v in range(1, top + 1) if train_counts.get(v, 0) <= 0]
    return 1, top, top, gaps


def clamp_profile_length(committed_span: int, n_labels: int):
    """The committed profile length, and whether the clamp of the gate commit bites.

    The gate commit pins this: the off-diagonal profile is defined for distances 1 to
    the label count minus one, because a distance equal to the label count would
    average zero label pairs. So the committed profile length is the smaller of the
    committed span and the label count minus one.
    """
    limit = n_labels - 1
    length = min(committed_span, limit)
    if length < DETREND_DEGREE + 2:
        raise Halt("a profile of %d distances is too short to remove a degree-%d trend"
                   % (length, DETREND_DEGREE))
    return length, bool(committed_span > limit)


def nominal_pair_counts(n_labels: int, profile_length: int) -> list:
    """How many label pairs each distance would average over a fully populated window."""
    return [n_labels - d for d in range(1, profile_length + 1)]


def derived_seed(kind: str, aperture: int = 0, index: int = 0) -> int:
    """The shuffle seed for one stream, derived from the one base seed.

    The gate commit fixes one base seed, value 0, with a separate derived stream per
    probe window and per labelling. The formula is
    base * 10000000 + aperture * 1000 + code * 100 + index, where the code is 0 for
    character-count labels, 1 for token-position labels and 2 for a range-check bin.
    The two streams that split documents use codes of their own.
    """
    codes = {"characters": 0, "tokens": 1, "range_bin": 2}
    if kind == "fold_split":
        return BASE_SEED * 10000000 + 1
    if kind == "anchor_resplits":
        return BASE_SEED * 10000000 + 2
    if kind not in codes:
        raise Halt("unknown shuffle stream %r" % (kind,))
    return BASE_SEED * 10000000 + aperture * 1000 + codes[kind] * 100 + index


def seed_stream_table() -> dict:
    """Every derived shuffle seed this run uses, so the record can carry all of them."""
    table = {"fold_split": derived_seed("fold_split"),
             "anchor_resplits": derived_seed("anchor_resplits")}
    for a in APERTURES:
        table["%d|characters" % a] = derived_seed("characters", a)
        table["%d|tokens" % a] = derived_seed("tokens", a)
    for j in range(len(BIN_EDGES)):
        table["%d|range_bin_%d" % (RANGE_APERTURE, j)] = derived_seed("range_bin", RANGE_APERTURE, j)
    return table


# ---------------------------------------------------------------------------
# Part 4. The repeating-pattern statistic. Pure Python, so the self-test needs no
# extra packages and the tested code is the code that runs.
#
# There is no search band anywhere in this design. The statistic is the largest power
# over every whole number of cycles whose period is at least 4 characters and at most
# the profile's own committed length.
# ---------------------------------------------------------------------------

def pair_index_lists(label_values, profile_length: int):
    """For each distance, the row pairs of the similarity structure that distance averages.

    `label_values` are the labels that actually have an average vector, in ascending
    order, and the row of the similarity structure is the position in that list. A
    label the held-out documents never carry has no average vector and so enters no
    pair; the distance it would have contributed to simply averages fewer pairs, and
    the count is recorded. Returns one (rows, columns) pair of lists per distance.
    """
    pos = {int(v): i for i, v in enumerate(label_values)}
    out = []
    for d in range(1, profile_length + 1):
        rows = []
        cols = []
        for v in label_values:
            w = int(v) + d
            j = pos.get(w)
            if j is not None:
                rows.append(pos[int(v)])
                cols.append(j)
        out.append((rows, cols))
    return out


def offdiag_profile_pairs(matrix, label_values, profile_length: int):
    """The profile and its pair counts, from a similarity structure. Pure Python."""
    pairs = pair_index_lists(label_values, profile_length)
    profile = []
    counts = []
    for d, (rows, cols) in enumerate(pairs, start=1):
        counts.append(len(rows))
        if not rows:
            raise Halt("distance %d averages zero label pairs, so the profile is undefined "
                       "there; the committed profile length may not be shortened after the "
                       "fact" % d)
        total = 0.0
        for i, j in zip(rows, cols):
            total += matrix[i][j]
        profile.append(total / len(rows))
    return profile, counts


def _solve_small(a, b):
    """Gaussian elimination with partial pivoting, for a few unknowns."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise Halt("singular system while removing the trend from a profile")
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def poly_detrend(y, degree: int = DETREND_DEGREE):
    """Subtract the best-fitting polynomial in distance, of the committed degree.

    The design fixes a degree-2 polynomial and refuses a moving average, because
    subtracting a moving average of a given window removes every period longer than
    about that window, which is where this study's predicted periods lie.
    """
    n = len(y)
    if n < degree + 2:
        raise Halt("a profile of %d distances is too short to remove a degree-%d trend"
                   % (n, degree))
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


def _trig_table(n: int, cycles):
    key = (n, tuple(cycles))
    tab = _TRIG_CACHE.get(key)
    if tab is None:
        tab = []
        for k in cycles:
            row = []
            for t in range(n):
                ang = 2.0 * math.pi * k * t / n
                row.append((math.cos(ang), math.sin(ang)))
            tab.append(row)
        _TRIG_CACHE[key] = tab
    return tab


def whole_cycle_periods(profile_length: int, period_min: float = PERIOD_MIN):
    """Every whole number of cycles across the profile, with its period in characters.

    The period of k cycles is the profile length divided by k. The design searches
    every period from 4 characters up to the profile's own length, so k runs from 1
    upward while the period stays at or above 4. Longest period first.
    """
    out = []
    k = 1
    while True:
        period = profile_length / float(k)
        if period < period_min - 1e-9:
            break
        if period <= profile_length + 1e-9:
            out.append((k, period))
        k += 1
    if not out:
        raise Halt("no whole number of cycles gives a period of at least %g characters "
                   "in a profile of length %d" % (period_min, profile_length))
    return out


def dft_power(y, cycles):
    """Power at each named whole number of cycles across the profile."""
    n = len(y)
    tab = _trig_table(n, cycles)
    out = []
    for row in tab:
        re = 0.0
        im = 0.0
        for t in range(n):
            c, s = row[t]
            re += y[t] * c
            im -= y[t] * s
        out.append((re * re + im * im) / (n * n))
    return out


def max_power_statistic(profile, periods, degree: int = DETREND_DEGREE):
    """The statistic: the largest power over every searched period.

    Ties are broken toward the longer period, so the answer never depends on the
    order the periods are visited in.
    """
    detrended = poly_detrend(profile, degree)
    cycles = [k for k, _ in periods]
    powers = dft_power(detrended, cycles)
    best = 0
    for i in range(1, len(powers)):
        if powers[i] > powers[best]:
            best = i
    return {
        "max_power": powers[best],
        "wavelength": periods[best][1],
        "cycles": periods[best][0],
        "n_periods": len(periods),
        "powers": powers,
        "detrended": detrended,
    }


# ---------------------------------------------------------------------------
# Part 5. Endpoint arithmetic. Every function here is a pure function of numbers, so
# the self-test can plant values whose correct call is known in advance. The rule
# text beside each one is written before any value exists.
# ---------------------------------------------------------------------------

RULE_SETUP = (
    "The text file's hash must equal the committed hash exactly, the downloaded "
    "layer count must equal 24 and the downloaded hidden size must equal 1024. "
    "Anything else halts the run before any graphics-card time."
)
RULE_CALIBRATION = (
    "Every constant of the fit and of the pattern readout is fixed from training "
    "documents alone and written here before any held-out number exists. A measured "
    "spread of exactly zero across the five resplits halts the run rather than "
    "emitting a band of zero width."
)
RULE_MAIN_LAYER = (
    "The layer with the best held-out fit, pooled over all nine wrap widths, is the "
    "main layer. Its location is recorded and never judged."
)
RULE_ANCHOR = (
    "Held-out fit quality at the anchor layer, width 140 alone, on the 12 fresh test "
    "documents must fall inside a band centred on 0.781 whose half-width is 3 times "
    "the median absolute deviation of that same statistic across 5 resplits of the "
    "36 training documents. Outside that band the run is void as an instrument "
    "fault: every pattern, control and range-check result it produces is not "
    "evidence about the model."
)
RULE_RECORDED_ONLY = (
    "These numbers are recorded so the record can be read. None of them may enter "
    "any decision, and none of them is compared against any threshold."
)
RULE_RIPPLE = (
    "The largest power over every whole-cycle period from 4 characters up to the "
    "committed profile length must rank 10th or better among 200 label-shuffled "
    "draws, counting the observed value itself. Worse than 10th is no separable "
    "repeating pattern at that probe window."
)
RULE_COUNT = (
    "Count how many of the three probe windows clear their own shuffle comparison. "
    "None clearing is a decisive negative up to 280 characters, unless the "
    "represented-range check shows label counts above 150 were never measurable."
)
RULE_WAVELENGTH_GAP = (
    "Among the probe windows that cleared their own comparison, the largest gap in "
    "characters between any two wavelengths. At most 10 characters counts as one "
    "wavelength shared across windows. Fewer than two clearing windows leaves the "
    "gap undefined."
)
RULE_RATIO_GAP = (
    "Among the probe windows that cleared their own comparison, the largest gap "
    "between any two wavelength-to-window-width ratios. Within 0.1 counts as a "
    "wavelength that scales with the window, once one shared wavelength has already "
    "been ruled out. Exactly one clearing window is undetermined and is never mixed."
)
RULE_RANGE = (
    "A label bin counts as represented when its within-bin held-out fit quality "
    "exceeds the 95th percentile of its own 50 label-shuffled draws. The bin "
    "spanning label 150 is recorded and left out of the decision. Every bin ending "
    "at or below label 140 represented, and every bin starting above label 150 not "
    "represented, means counts beyond 150 were not measurable."
)
RULE_TOKEN = (
    "The repeating pattern is credited to counting characters when the two "
    "labellings' wavelengths differ by more than 3 characters, or when they disagree "
    "on whether they clear their own comparison. A gap of 3 characters or less with "
    "both labellings clearing credits token position instead."
)
RULE_SUMMARY = (
    "Each endpoint's own rule, restated in its own record. This record gathers the "
    "calls and computes no new number. A failed anchor gate makes every other call "
    "here void as an instrument fault."
)


def median(values):
    """The middle value, or the mean of the two middle values."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        raise Halt("no values to take a median of")
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def median_absolute_deviation(values):
    """The median of the absolute distances from the median, unscaled.

    The gate commit pins this: no 1.4826 factor.
    """
    med = median(values)
    return median([abs(float(v) - med) for v in values])


def anchor_band(values, centre: float = ANCHOR_CENTRE, multiple: float = ANCHOR_MAD_MULTIPLE):
    """The anchor band, from the spread measured across the resplits.

    A measured spread of exactly zero means the wrong source of variation was
    chosen, and halts the run rather than emitting a band of zero width.
    """
    for v in values:
        if v is None or v != v or v in (float("inf"), float("-inf")):
            raise Halt("a resplit produced %r, which is not a number; the anchor band "
                       "cannot be measured" % (v,))
    spread = median_absolute_deviation(values)
    if spread == 0.0:
        raise Halt("WRONG_VARIANCE_SOURCE: the five resplits produced a spread of exactly "
                   "zero, so the source of variation was the wrong one; the design "
                   "forbids emitting a band of zero width")
    half = multiple * spread
    return {"median": median(values), "mad": spread, "half_width": half,
            "low": centre - half, "high": centre + half, "centre": centre,
            "band_half_width_below_0.02": bool(half < NARROW_BAND_FLAG_AT)}


def decide_setup(hash_ok: bool, n_layers, hidden_size) -> str:
    if hash_ok and exact_config_match(n_layers, hidden_size):
        return "SETUP_OK"
    return "SETUP_FAILURE"


def decide_anchor(value, low: float, high: float) -> str:
    if value is None or value != value:
        return "ANCHOR_OUT_OF_BAND"
    return "ANCHOR_IN_BAND" if (low <= value <= high) else "ANCHOR_OUT_OF_BAND"


def null_rank(observed, draws) -> int:
    """Position of the observed value among the draws plus itself, largest first."""
    return 1 + sum(1 for v in draws if v >= observed)


def decide_ripple(rank: int) -> str:
    return "NULL_CLEARED" if rank <= NULL_RANK_MAX else "NULL_NOT_CLEARED"


def largest_pairwise_gap(values):
    """The largest gap between any two values, or None with fewer than two values."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    return max(vals) - min(vals)


def decide_wavelength_gap(gap) -> str:
    if gap is None:
        return "GAP_UNDEFINED_FEWER_THAN_TWO_WINDOWS"
    return "ONE_SHARED_WAVELENGTH" if gap <= INVARIANCE_MAX_GAP_CHARS else "WAVELENGTHS_DIFFER"


def decide_ratio_gap(gap) -> str:
    if gap is None:
        return "GAP_UNDEFINED_FEWER_THAN_TWO_WINDOWS"
    return "RATIOS_WITHIN_CUTOFF" if gap <= PROPORTION_MAX_RATIO_GAP else "RATIOS_DIFFER"


def decide_invariance(cleared):
    """The headline fork, over the probe windows that cleared their own comparison.

    `cleared` maps a probe window's width in characters to its wavelength in
    characters. Returns (call, wavelength gap, ratio gap).
    """
    widths = sorted(cleared)
    if len(widths) == 0:
        return "NO_WINDOW_CLEARED", None, None
    if len(widths) == 1:
        return "UNDETERMINED", None, None
    wl_gap = largest_pairwise_gap([cleared[w] for w in widths])
    ratio_gap = largest_pairwise_gap([cleared[w] / float(w) for w in widths])
    if wl_gap <= INVARIANCE_MAX_GAP_CHARS:
        return "INVARIANT", wl_gap, ratio_gap
    if ratio_gap <= PROPORTION_MAX_RATIO_GAP:
        return "PROPORTIONAL", wl_gap, ratio_gap
    return "MIXED", wl_gap, ratio_gap


def percentile_linear(values, pct: float):
    """The percentile of a list, interpolating between neighbouring sorted values."""
    vals = sorted(float(v) for v in values)
    if not vals:
        raise Halt("no draws to take a percentile of")
    if len(vals) == 1:
        return vals[0]
    pos = (pct / 100.0) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def bin_role(lo: int, hi: int) -> str:
    """Whether a label bin is low, high, or spans the boundary and is left out."""
    if hi <= LOW_BIN_MAX_LABEL:
        return "low"
    if lo > HIGH_BIN_MIN_LABEL:
        return "high"
    return "straddles"


def decide_bin(observed, draws) -> str:
    """One bin's call: represented when it exceeds the 95th percentile of its own draws."""
    if observed is None or observed != observed:
        return "BIN_NOT_COMPUTABLE"
    cut = percentile_linear(draws, BIN_PERCENTILE)
    return "REPRESENTED" if observed > cut else "NOT_REPRESENTED"


def decide_range_pattern(bins) -> str:
    """The range check's call, over bins already decided one at a time.

    `bins` is a list of dictionaries carrying a role and a call. The bin spanning the
    boundary is recorded and excluded here, exactly as the design says.
    """
    low = [b for b in bins if b["role"] == "low"]
    high = [b for b in bins if b["role"] == "high"]
    if any(b["decision"] == "REPRESENTED" for b in high):
        return "ABOVE_150_REPRESENTED"
    if low and all(b["decision"] == "REPRESENTED" for b in low) and high:
        return "ABOVE_150_NOT_REPRESENTED"
    return "PATTERN_INCOMPLETE"


def decide_token_credit(wl_chars, wl_token_in_chars, char_cleared, token_cleared) -> str:
    """Which labelling the repeating pattern at one probe window belongs to."""
    if wl_token_in_chars is None or token_cleared is None or wl_chars is None:
        return "TOKEN_CONTROL_NOT_COMPUTABLE"
    if bool(char_cleared) != bool(token_cleared):
        return "CHARACTER_COUNT_DISTINGUISHED"
    if abs(float(wl_chars) - float(wl_token_in_chars)) > CONFOUND_MARGIN_CHARS:
        return "CHARACTER_COUNT_DISTINGUISHED"
    if char_cleared and token_cleared:
        return "CREDITED_TO_TOKEN_POSITION"
    return "NO_PATTERN_TO_CREDIT"


def decide_headline(run_valid: bool, cleared, range_pattern) -> str:
    """The run's own call, from the calls beside it.

    A void anchor gate overrides everything. Otherwise the fork over the probe
    windows that cleared decides, and no window clearing defers to the
    represented-range check exactly as the design says.
    """
    if not run_valid:
        return "VOID_ANCHOR_FAILED"
    call, _, _ = decide_invariance(cleared)
    if call != "NO_WINDOW_CLEARED":
        return call
    if range_pattern == "ABOVE_150_NOT_REPRESENTED":
        return "RANGE_NOT_REPRESENTED"
    if range_pattern == "ABOVE_150_REPRESENTED":
        return "DECISIVE_NEGATIVE_NO_PATTERN_UP_TO_280"
    return "NO_WINDOW_CLEARED_RANGE_CHECK_INCONCLUSIVE"


def r2_from_pairs(y_true, y_pred):
    """Fit quality against the spread of these same held-out labels. Pure Python.

    Used for one label bin of the represented-range check, where the design says the
    bin's own label variance is the denominator.
    """
    n = len(y_true)
    if n != len(y_pred):
        raise Halt("held-out labels and predictions differ in length")
    if n == 0:
        raise Halt("no held-out positions in this bin, so fit quality is undefined")
    mean = sum(float(v) for v in y_true) / n
    sse = 0.0
    sstot = 0.0
    for t, p in zip(y_true, y_pred):
        d = float(t) - float(p)
        sse += d * d
        c = float(t) - mean
        sstot += c * c
    if sstot <= 0.0:
        raise Halt("the held-out labels in this bin have no spread, so fit quality has "
                   "no denominator")
    return 1.0 - sse / sstot


# ---------------------------------------------------------------------------
# Part 6. Ridge fitting from running sums.
#
# Holding every internal vector of every position would need many gigabytes and is
# not needed. A ridge fit and its fit quality depend on the data only through five
# running sums per layer: the sum of outer products of the vectors, the sum of the
# vectors, the sum of vector times label, the sum of labels, and the sum of squared
# labels. The extract stage accumulates those, so the later stages never hold the
# hidden states at all.
# ---------------------------------------------------------------------------

def centred_system(gram, vec_sum, cross, label_sum, n):
    """The fit's normal equations with the mean removed. Needs numpy.

    Built once per fit set and reused across ridge strengths, so the whole grid costs
    one system rather than eight.
    """
    import numpy as np
    xbar = vec_sum / n
    ybar = label_sum / n
    return gram - np.outer(vec_sum, vec_sum) / n, cross - vec_sum * ybar, xbar, ybar


def solve_ridge(centred_gram, centred_cross, xbar, ybar, alpha):
    """Weights and offset at one ridge strength. Needs numpy."""
    import numpy as np
    a = centred_gram.copy()
    dim = a.shape[0]
    a.flat[::dim + 1] += alpha            # the same as adding alpha times the identity
    weights = np.linalg.solve(a, centred_cross)
    return weights, ybar - float(weights @ xbar)


def ridge_fit_from_sums(gram, vec_sum, cross, label_sum, n, alpha):
    """Weights and offset of a ridge fit, from running sums. Needs numpy."""
    gc, cc, xbar, ybar = centred_system(gram, vec_sum, cross, label_sum, n)
    return solve_ridge(gc, cc, xbar, ybar, alpha)


def r2_from_sums(weights, offset, gram, vec_sum, cross, label_sum, label_sq_sum, n):
    """Fit quality on a group of positions, from that group's running sums. Needs numpy."""
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


def strength_is_interior(alpha, grid) -> bool:
    """True when the winning ridge strength is not at either end of the grid."""
    ordered = sorted(grid)
    return not (alpha == ordered[0] or alpha == ordered[-1])


# ---------------------------------------------------------------------------
# Part 7. Shared plumbing: inputs, paths, the void flag, and the record writer.
# ---------------------------------------------------------------------------

RUN_STATE = {"run_valid": True, "void_reason": None}
ANCHOR_CONTEXT = {"decision": "NOT_YET_COMPUTED", "statistic": None,
                  "band_low": None, "band_high": None, "band_half_width": None,
                  "band_centre": ANCHOR_CENTRE}


def load_provenance():
    """Import the record writer from beside this file, or from the frozen folder."""
    try:
        import provenance                        # on the box it sits alongside this file
    except ImportError:
        sys.path.insert(0, str(HERE / "frozen"))
        try:
            import provenance                    # on this workstation it sits in frozen/
        except ImportError as exc:
            raise Halt("the record writer provenance.py is missing: %s" % exc)
    for name in ("write_result", "describe_input", "cell_slug"):
        if not hasattr(provenance, name):
            raise Halt("provenance.py has no %s; refusing to run without records" % name)
    return provenance


def find_first(names):
    for name in names:
        p = HERE / name
        if p.exists():
            return p
    return None


def required_input(explicit, names, what):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise Halt("the named %s %s is absent; the design says halt, never "
                       "reconstruct it" % (what, p))
        return p
    p = find_first(names)
    if p is None:
        raise Halt("%s is absent (looked for %s); every record must carry it as an "
                   "input, so this halts rather than writing records without it"
                   % (what, ", ".join(names)))
    return p


def stimulus_path(explicit=None):
    return required_input(explicit, [STIMULUS_NAME, "out/" + STIMULUS_NAME], "the committed text file")


def design_path(explicit=None):
    return required_input(explicit, [DESIGN_NAME, "out/" + DESIGN_NAME], "the committed design record")


def gate_path(explicit=None):
    return required_input(explicit, [GATE_NAME, "out/" + GATE_NAME], "the committed gate commit")


def hf_token():
    """The access token and the name of the variable it came from.

    The model is public, so an absent token is fine and is recorded as such. No token
    value is ever printed or written.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_KEY_ONE"):
        val = os.environ.get(name)
        if val:
            return val, name
    return None, "none"


def load_stimulus(path):
    """Read the committed text file and check its shape against the design."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    docs = data.get("docs", [])
    problems = []
    if len(docs) != N_DOCS:
        problems.append("document count %d, expected %d" % (len(docs), N_DOCS))
    if list(meta.get("widths", [])) != WIDTHS:
        problems.append("wrap widths %r, expected %r" % (meta.get("widths"), WIDTHS))
    if meta.get("train_docs") != N_TRAIN_DOCS:
        problems.append("training document count %r, expected %d"
                        % (meta.get("train_docs"), N_TRAIN_DOCS))
    n_train = sum(1 for d in docs if d.get("split") == "train")
    n_test = sum(1 for d in docs if d.get("split") == "test")
    if n_train != N_TRAIN_DOCS or n_test != N_TEST_DOCS:
        problems.append("documents marked train/test: %d/%d, expected %d/%d"
                        % (n_train, n_test, N_TRAIN_DOCS, N_TEST_DOCS))
    for d in docs:
        missing = [str(w) for w in WIDTHS if str(w) not in d.get("wrapped", {})]
        if missing:
            problems.append("document %r lacks widths %r" % (d.get("id"), missing))
            break
    if problems:
        raise Halt("the committed text file does not match the design: " + "; ".join(problems))
    return meta, docs


def paths_for(results_dir):
    out = Path(results_dir) / "out"
    work = Path(results_dir) / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def record_inputs(prov, stim, design, gate, extra=()):
    items = [prov.describe_input(stim, n_items=N_DOCS * len(WIDTHS)),
             prov.describe_input(design, n_items=1),
             prov.describe_input(gate, n_items=1)]
    for path, n in extra:
        items.append(prov.describe_input(path, n_items=n))
    return items


def base_cell(stage, arm, endpoint=None, aperture=None, labelling=None):
    cell = {"model": "pythia-410m-deduped", "revision": REVISION, "stage": stage, "arm": arm}
    if endpoint is not None:
        cell["endpoint"] = endpoint
    if aperture is not None:
        cell["aperture"] = int(aperture)
    if labelling is not None:
        cell["labelling"] = labelling
    return cell


def void_the_run(reason: str, out_dir):
    """Mark this run void. Every record written after this carries the flag."""
    RUN_STATE["run_valid"] = False
    RUN_STATE["void_reason"] = reason
    save_run_state(out_dir)
    print("[void] %s" % reason)


def save_run_state(out_dir):
    with open(Path(out_dir) / "run_state.json", "w", encoding="utf-8") as fh:
        json.dump({"run_valid": RUN_STATE["run_valid"],
                   "void_reason": RUN_STATE["void_reason"],
                   "anchor": dict(ANCHOR_CONTEXT)}, fh, indent=2, sort_keys=True)


def reset_run_state(out_dir):
    """Start a stage with a valid run and no anchor call yet.

    A stage never inherits an earlier run's void. The stage that owns the gate
    recomputes it, and every record it writes afterwards carries the fresh call.
    """
    RUN_STATE["run_valid"] = True
    RUN_STATE["void_reason"] = None
    ANCHOR_CONTEXT.update({"decision": "NOT_YET_COMPUTED", "statistic": None,
                           "band_low": None, "band_high": None, "band_half_width": None,
                           "band_centre": ANCHOR_CENTRE})
    save_run_state(out_dir)


def write_record(prov, out_dir, cell, *, values, **kwargs):
    """Write one self-describing record, with this run's validity stamped into it.

    Every record carries run_valid. Every record also carries the anchor gate's call
    and its measured band inside its own values, which the design asks for by name,
    so no reader has to fetch a separate roll-up to learn that a number is void.
    """
    v = dict(values)
    v["run_valid"] = bool(RUN_STATE["run_valid"])
    v["run_void_reason"] = RUN_STATE["void_reason"]
    v["anchor_gate_decision"] = ANCHOR_CONTEXT["decision"]
    v["anchor_statistic"] = ANCHOR_CONTEXT["statistic"]
    v["anchor_band_low"] = ANCHOR_CONTEXT["band_low"]
    v["anchor_band_high"] = ANCHOR_CONTEXT["band_high"]
    v["anchor_band_half_width"] = ANCHOR_CONTEXT["band_half_width"]
    v["anchor_band_centre"] = ANCHOR_CONTEXT["band_centre"]
    path = Path(out_dir) / (prov.cell_slug(cell) + ".json")
    prov.write_result(path, cell=cell, values=v, **kwargs)
    print("[record] %s valid=%s -> %s"
          % (kwargs.get("decision", ""), RUN_STATE["run_valid"], path.name))
    sys.stdout.flush()
    return path


# ---------------------------------------------------------------------------
# Part 8. Stage one: check the inputs, run the forward passes, save the running sums.
#
# 48 documents at nine wrap widths is 432 forward passes. Nothing here reads a
# held-out number: it only writes down what the two later stages consume. The ridge
# fits are accumulated as running sums, so no stage ever holds all the hidden states.
# ---------------------------------------------------------------------------

GROUP_NAMES = (["train_fold%d" % f for f in range(N_FOLDS)]
               + ["train_w%d" % ANCHOR_WIDTH]
               + ["test_w%d" % w for w in WIDTHS])


def group_index(name):
    return GROUP_NAMES.index(name)


def fold_of_documents(train_ids):
    """Which of the five cross-validation groups each training document belongs to.

    Splitting by document, not by position, keeps one document's sentences out of
    both sides of a split. The seed is derived from the one base seed and recorded.
    """
    import numpy as np
    seed = derived_seed("fold_split")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(train_ids)))
    assign = {}
    for rank, pos in enumerate(order):
        assign[int(train_ids[int(pos)])] = rank % N_FOLDS
    return assign, seed


def model_kwargs(token):
    kw = {"revision": REVISION}
    if token:
        kw["token"] = token
    return kw


def load_model_in_float16(model_id, kw):
    """Load the model in float16, tolerating either spelling of the dtype argument."""
    import torch
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kw)


def stage_extract(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_path(args.design_record)
    gate = gate_path(args.gate_commit)
    reset_run_state(out_dir)

    actual_hash = sha256_file(stim)
    hash_ok = exact_hash_match(actual_hash, STIMULUS_SHA256)
    print("[extract] committed text file %s" % stim)
    print("[extract] hash %s (expected %s) match=%s" % (actual_hash, STIMULUS_SHA256, hash_ok))
    meta, docs = load_stimulus(stim)

    token, token_env = hf_token()
    kw = model_kwargs(token)
    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    context_limit = int(getattr(config, "max_position_embeddings", CONTEXT_LIMIT_FALLBACK)
                        or CONTEXT_LIMIT_FALLBACK)
    print("[extract] downloaded config: layers=%r hidden=%r context=%d"
          % (n_layers, hidden, context_limit))

    decision = decide_setup(hash_ok, n_layers, hidden)
    write_record(
        prov, out_dir, base_cell("extract", "setup_check"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Committed text file hash, and the downloaded model configuration's layer count and hidden size.",
        values={"stimulus_sha256": actual_hash, "stimulus_sha256_expected": STIMULUS_SHA256,
                "stimulus_hash_match": hash_ok,
                "design_record_sha256": sha256_file(design),
                "design_record_sha256_expected": DESIGN_SHA256,
                "design_record_hash_match": exact_hash_match(sha256_file(design), DESIGN_SHA256),
                "gate_commit_sha256": sha256_file(gate),
                "gate_commit_sha256_expected": GATE_SHA256,
                "gate_commit_hash_match": exact_hash_match(sha256_file(gate), GATE_SHA256),
                "n_layers": n_layers, "hidden_size": hidden, "context_limit": context_limit,
                "model_id": MODEL_ID, "revision": REVISION, "dtype": TORCH_DTYPE,
                "access_token_source": token_env},
        threshold={"stimulus_sha256": STIMULUS_SHA256, "n_layers": EXPECT_LAYERS,
                   "hidden_size": EXPECT_HIDDEN},
        decision_rule=RULE_SETUP,
        decision=decision,
        notes=("Layer count 24 and hidden size 1024 are recalled from the Pythia paper, "
               "arXiv 2304.01373 table 1, and the downloaded configuration is the "
               "authority. The two committed specification files are hashed here; a "
               "difference is recorded and gates nothing. " + RERUN_NOTE),
    )
    if decision != "SETUP_OK":
        raise Halt("setup check failed: text hash match=%s, layers=%r, hidden size=%r"
                   % (hash_ok, n_layers, hidden))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    if not getattr(tok, "is_fast", False):
        raise Halt("the tokenizer does not report character offsets, so labels cannot be computed")
    if not torch.cuda.is_available():
        raise Halt("no graphics card is visible to torch; the run would silently use the "
                   "processor and bill graphics-card time for processor numbers")

    torch.backends.cuda.matmul.allow_tf32 = False     # keeps the running sums accurate
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    print("[extract] loading %s revision %s in %s on %s" % (MODEL_ID, REVISION, TORCH_DTYPE, device))
    model = load_model_in_float16(MODEL_ID, kw)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    n_read = EXPECT_LAYERS + 1                        # readout 0 is the embedding output
    dim = EXPECT_HIDDEN
    n_groups = len(GROUP_NAMES)
    gram = torch.zeros((n_groups, n_read, dim, dim), dtype=torch.float64, device=device)
    vec_sum = torch.zeros((n_groups, n_read, dim), dtype=torch.float64, device=device)
    cross = torch.zeros((n_groups, n_read, dim), dtype=torch.float64, device=device)
    label_sum = [0.0] * n_groups
    label_sq = [0.0] * n_groups
    n_pos = [0] * n_groups

    train_ids = [int(d["id"]) for d in docs if d["split"] == "train"]
    test_ids = [int(d["id"]) for d in docs if d["split"] == "test"]
    folds, fold_seed = fold_of_documents(train_ids)

    # Per-document running sums at the anchor width, one file per readout, documents in
    # ascending order. The anchor procedure refits on arbitrary subsets of the training
    # documents, so it needs each document's own sums rather than a pooled total.
    doc_gram_files = [open(work_dir / ("w%d_gram_layer%02d.f64" % (ANCHOR_WIDTH, l)), "wb")
                      for l in range(n_read)]
    doc_vec = np.zeros((N_TRAIN_DOCS, n_read, dim), dtype=np.float64)
    doc_cross = np.zeros((N_TRAIN_DOCS, n_read, dim), dtype=np.float64)
    doc_label_sum = np.zeros(N_TRAIN_DOCS, dtype=np.float64)
    doc_label_sq = np.zeros(N_TRAIN_DOCS, dtype=np.float64)
    doc_n = np.zeros(N_TRAIN_DOCS, dtype=np.int64)
    doc_order = sorted(train_ids)
    doc_row = {d: i for i, d in enumerate(doc_order)}

    # Held-out per-position vectors at the three probe windows, every readout, because
    # the main layer is not known until the endpoints stage.
    act_files = {a: [open(work_dir / ("act_a%d_layer%02d.f32" % (a, l)), "wb")
                     for l in range(n_read)] for a in APERTURES}
    act_meta = {a: {"doc": [], "char": [], "token": [], "tok_index": []} for a in APERTURES}

    rows_pos = open(out_dir / "rows_positions_apertures.jsonl", "w", encoding="utf-8")
    rows_var = open(out_dir / "rows_variants.jsonl", "w", encoding="utf-8")

    label_counts = {"train": {}, "test": {}}
    for w in WIDTHS:
        for split in ("train", "test"):
            label_counts[split][str(w)] = {"characters": {}, "tokens": {}}

    started = time.time()
    n_variants = 0
    all_flags = {}
    try:
        for doc in docs:
            did = int(doc["id"])
            split = doc["split"]
            for width in WIDTHS:
                text = doc["wrapped"][str(width)]
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids = enc["input_ids"]
                spans = [tuple(s) for s in enc["offset_mapping"]]
                char_labels, token_labels, line_nums = labels_for_spans(text, spans)
                pieces = tok.convert_ids_to_tokens(ids)
                flags = degenerate_flags(len(ids), context_limit, pieces)
                if flags:
                    all_flags["doc%d_w%d" % (did, width)] = flags
                if "token_cap" in flags:
                    raise Halt("document %d at width %d reaches the context limit of %d "
                               "tokens, so the committed text would be cut short"
                               % (did, width, context_limit))

                keep = [i for i, c in enumerate(char_labels) if c is not None]
                y = np.array([char_labels[i] for i in keep], dtype=np.float64)
                for kind, labs in (("characters", char_labels), ("tokens", token_labels)):
                    bucket = label_counts[split][str(width)][kind]
                    for lab, cnt in count_labels([labs[i] for i in keep]).items():
                        bucket[str(lab)] = bucket.get(str(lab), 0) + cnt

                t_ids = torch.tensor([ids], dtype=torch.long, device=device)
                out = model(input_ids=t_ids, output_hidden_states=True, use_cache=False)
                states = out.hidden_states
                if len(states) != n_read:
                    raise Halt("the model returned %d readouts, expected %d"
                               % (len(states), n_read))
                idx = torch.tensor(keep, dtype=torch.long, device=device)
                y_t = torch.tensor(y, dtype=torch.float64, device=device)

                targets = []
                if split == "train":
                    targets.append(group_index("train_fold%d" % folds[did]))
                    if width == ANCHOR_WIDTH:
                        targets.append(group_index("train_w%d" % ANCHOR_WIDTH))
                else:
                    targets.append(group_index("test_w%d" % width))

                for l in range(n_read):
                    x = states[l][0].index_select(0, idx).double()
                    g = x.T @ x
                    v = x.sum(0)
                    c = x.T @ y_t
                    for gi in targets:
                        gram[gi, l] += g
                        vec_sum[gi, l] += v
                        cross[gi, l] += c
                    if split == "train" and width == ANCHOR_WIDTH:
                        g.cpu().numpy().astype(np.float64).tofile(doc_gram_files[l])
                        doc_vec[doc_row[did], l] = v.cpu().numpy()
                        doc_cross[doc_row[did], l] = c.cpu().numpy()
                    if split == "test" and width in APERTURES:
                        x.float().cpu().numpy().astype(np.float32).tofile(act_files[width][l])

                for gi in targets:
                    label_sum[gi] += float(y.sum())
                    label_sq[gi] += float((y * y).sum())
                    n_pos[gi] += len(keep)
                if split == "train" and width == ANCHOR_WIDTH:
                    doc_label_sum[doc_row[did]] = float(y.sum())
                    doc_label_sq[doc_row[did]] = float((y * y).sum())
                    doc_n[doc_row[did]] = len(keep)
                if split == "test" and width in APERTURES:
                    act_meta[width]["doc"].extend([did] * len(keep))
                    act_meta[width]["char"].extend([char_labels[i] for i in keep])
                    act_meta[width]["token"].extend([token_labels[i] for i in keep])
                    act_meta[width]["tok_index"].extend(keep)
                    for i in keep:
                        rows_pos.write(json.dumps({
                            "doc": did, "aperture": width, "split": split,
                            "tok_index": i, "tok_text": pieces[i],
                            "char_start": spans[i][0], "char_end": spans[i][1],
                            "line": line_nums[i], "char_label": char_labels[i],
                            "token_label": token_labels[i]}, ensure_ascii=False) + "\n")

                rows_var.write(json.dumps({
                    "doc": did, "width": width, "split": split,
                    "n_chars": len(text), "n_tokens": len(ids), "n_labelled": len(keep),
                    "n_lines": text.count("\n") + 1,
                    "chars_spanned_by_labelled_tokens":
                        int(sum(spans[i][1] - spans[i][0] for i in keep)),
                    "degenerate_flags": flags,
                    "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text}, ensure_ascii=False) + "\n")
                n_variants += 1
                del out, states
            if did % 8 == 0:
                print("[extract] document %d of %d done (%.0f s)" % (did, N_DOCS, time.time() - started))
                sys.stdout.flush()
        elapsed = time.time() - started
    finally:
        rows_pos.close()
        rows_var.close()
        for fh in doc_gram_files:
            fh.close()
        for a in APERTURES:
            for fh in act_files[a]:
                fh.close()

    del model
    torch.cuda.empty_cache()

    np.savez(work_dir / "group_sums.npz",
             groups=np.array(GROUP_NAMES),
             gram=gram.cpu().numpy(), vec_sum=vec_sum.cpu().numpy(),
             cross=cross.cpu().numpy(),
             label_sum=np.array(label_sum), label_sq=np.array(label_sq),
             n_pos=np.array(n_pos))
    del gram, vec_sum, cross
    torch.cuda.empty_cache()
    np.savez(work_dir / ("doc_sums_w%d.npz" % ANCHOR_WIDTH),
             doc_ids=np.array(doc_order, dtype=np.int64),
             vec_sum=doc_vec, cross=doc_cross,
             label_sum=doc_label_sum, label_sq=doc_label_sq, n_pos=doc_n)
    for a in APERTURES:
        np.savez(work_dir / ("act_meta_a%d.npz" % a),
                 doc=np.array(act_meta[a]["doc"], dtype=np.int64),
                 char_label=np.array(act_meta[a]["char"], dtype=np.int64),
                 token_label=np.array(act_meta[a]["token"], dtype=np.int64),
                 tok_index=np.array(act_meta[a]["tok_index"], dtype=np.int64),
                 n_read=np.array([n_read]), hidden=np.array([dim]))
    with open(work_dir / "label_counts_train.json", "w", encoding="utf-8") as fh:
        json.dump(label_counts["train"], fh, indent=2, sort_keys=True)
    with open(work_dir / "label_counts_test.json", "w", encoding="utf-8") as fh:
        json.dump(label_counts["test"], fh, indent=2, sort_keys=True)
    with open(work_dir / "folds.json", "w", encoding="utf-8") as fh:
        json.dump({"fold_split_seed": fold_seed, "base_seed": BASE_SEED,
                   "assignment": {str(k): v for k, v in sorted(folds.items())},
                   "train_doc_order": doc_order, "test_doc_ids": test_ids}, fh, indent=2)

    rows_per_aperture = {str(a): len(act_meta[a]["char"]) for a in APERTURES}
    write_record(
        prov, out_dir, base_cell("extract", "forward_passes"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Counts from one deterministic forward pass per text variant, and the running sums saved for the fits.",
        values={"n_variants": n_variants, "n_forward_passes": n_variants,
                "n_positions_by_group": {GROUP_NAMES[i]: n_pos[i] for i in range(n_groups)},
                "n_positions_total": int(sum(n_pos)),
                "held_out_rows_by_aperture": rows_per_aperture,
                "n_readouts": n_read, "device": device, "dtype": TORCH_DTYPE,
                "seconds": round(elapsed, 1),
                "degenerate_variants": all_flags,
                "fold_split_seed": fold_seed,
                "fold_assignment": {str(k): v for k, v in sorted(folds.items())},
                "train_doc_order": doc_order, "test_doc_ids": test_ids},
        threshold={"n_variants": N_DOCS * len(WIDTHS)},
        decision_rule=("Every one of the 48 documents at every one of the nine wrap widths "
                       "must produce one forward pass, and no variant may reach the model's "
                       "context limit. Anything else halts."),
        decision="EXTRACT_OK" if n_variants == N_DOCS * len(WIDTHS) else "EXTRACT_INCOMPLETE",
        notes=("Readout 0 is the embedding output and readout 24 is the last transformer "
               "layer, so a model with 24 blocks yields 25 readouts. The library applies "
               "the model's final normalisation to the last readout only, which is a "
               "property of the library and is recorded here rather than changed. No text "
               "is generated anywhere in this run. " + RERUN_NOTE),
    )
    print("[extract] %d forward passes, %d labelled positions, %.1f s"
          % (n_variants, sum(n_pos), elapsed))
    if n_variants != N_DOCS * len(WIDTHS):
        raise Halt("expected %d forward passes, ran %d" % (N_DOCS * len(WIDTHS), n_variants))
    return 0


# ---------------------------------------------------------------------------
# Part 9. Stage two: fix every remaining constant, on training documents alone.
#
# The gate commit pins what this record must hold before the endpoints stage reads
# its first held-out row: the ridge strength and whether it sits at an edge of the
# grid, the label window per probe window, the profile length per probe window with
# its pair counts, the shuffle seeds, the anchor layer, the five resplit values, the
# spread across them and the resulting anchor band. Nothing here reads a held-out
# document.
# ---------------------------------------------------------------------------

def load_group_sums(work_dir):
    import numpy as np
    path = Path(work_dir) / "group_sums.npz"
    if not path.exists():
        raise Halt("no running sums at %s; the extract stage must run first" % path)
    z = np.load(path, allow_pickle=False)
    groups = [str(g) for g in z["groups"]]
    if groups != GROUP_NAMES:
        raise Halt("the saved accumulation groups %r are not the expected %r"
                   % (groups, GROUP_NAMES))
    gram = z["gram"]
    if gram.ndim != 4 or gram.shape[0] != len(GROUP_NAMES) or gram.shape[2:] != (EXPECT_HIDDEN, EXPECT_HIDDEN):
        raise Halt("the saved running sums have shape %r, which does not match %d groups "
                   "of %d by %d" % (gram.shape, len(GROUP_NAMES), EXPECT_HIDDEN, EXPECT_HIDDEN))
    return {"groups": groups, "gram": gram,
            "vec_sum": z["vec_sum"], "cross": z["cross"],
            "label_sum": z["label_sum"], "label_sq": z["label_sq"], "n_pos": z["n_pos"]}


def sums_for_groups(tot, names, layer):
    """The running sums of one readout, added over a list of accumulation groups."""
    idx = [tot["groups"].index(n) for n in names]
    g = tot["gram"][idx, layer].sum(axis=0)
    v = tot["vec_sum"][idx, layer].sum(axis=0)
    c = tot["cross"][idx, layer].sum(axis=0)
    ls = float(tot["label_sum"][idx].sum())
    lq = float(tot["label_sq"][idx].sum())
    n = int(tot["n_pos"][idx].sum())
    return g, v, c, ls, lq, n


def read_doc_grams(work_dir, layer, n_docs=N_TRAIN_DOCS, dim=EXPECT_HIDDEN):
    """Every training document's own sum of outer products at one readout, anchor width."""
    import numpy as np
    path = Path(work_dir) / ("w%d_gram_layer%02d.f64" % (ANCHOR_WIDTH, layer))
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size != n_docs * dim * dim:
        raise Halt("saved sums for readout %d hold %d numbers, expected %d"
                   % (layer, arr.size, n_docs * dim * dim))
    return arr.reshape(n_docs, dim, dim)


def doc_subset_sums(doc_gram, doc, rows, layer):
    """The running sums of one subset of training documents at one readout."""
    import numpy as np
    idx = np.array(sorted(rows), dtype=np.int64)
    g = doc_gram[idx].sum(axis=0)
    v = doc["vec_sum"][idx, layer].sum(axis=0)
    c = doc["cross"][idx, layer].sum(axis=0)
    ls = float(doc["label_sum"][idx].sum())
    lq = float(doc["label_sq"][idx].sum())
    n = int(doc["n_pos"][idx].sum())
    return g, v, c, ls, lq, n


def fit_and_score_over_grid(fit_sums, eval_sums, grid=RIDGE_GRID):
    """Held-out fit quality of one fit set on one evaluation set, at every ridge strength."""
    g, v, c, ls, lq, n = fit_sums
    ge, ve, ce, lse, lqe, ne = eval_sums
    gc, cc, xbar, ybar = centred_system(g, v, c, ls, n)
    out = {}
    for alpha in grid:
        w, off = solve_ridge(gc, cc, xbar, ybar, alpha)
        r2, _, _ = r2_from_sums(w, off, ge, ve, ce, lse, lqe, ne)
        out[alpha] = float(r2)
    return out


def best_of_surface(surface, n_read, grid=RIDGE_GRID):
    """The best (strength, readout) pair of a cross-validated quality surface."""
    per_alpha = {}
    for alpha in grid:
        best_layer = max(range(n_read), key=lambda l: surface[(alpha, l)])
        per_alpha[alpha] = (best_layer, surface[(alpha, best_layer)])
    alpha_star = max(grid, key=lambda a: per_alpha[a][1])
    layer_star, quality = per_alpha[alpha_star]
    return alpha_star, layer_star, quality, per_alpha


def resplit_plan(doc_order, seed):
    """The five resplits: which six training documents each one holds out, and its folds.

    Each resplit runs the whole anchor procedure on its own split, so it also needs a
    five-way split of its own 30 sub-training documents. Both come from one stream
    derived from the base seed, and both are written into the calibration record.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(doc_order)))
    plans = []
    for r in range(N_RESPLITS):
        held_rows = sorted(int(perm[r * RESPLIT_HELDOUT_DOCS + k]) for k in range(RESPLIT_HELDOUT_DOCS))
        sub_rows = [i for i in range(len(doc_order)) if i not in set(held_rows)]
        order = list(rng.permutation(len(sub_rows)))
        folds = {}
        for rank, pos in enumerate(order):
            folds[int(sub_rows[int(pos)])] = rank % N_FOLDS
        plans.append({"resplit": r,
                      "held_out_rows": held_rows,
                      "held_out_doc_ids": [int(doc_order[i]) for i in held_rows],
                      "sub_train_rows": sub_rows,
                      "sub_train_doc_ids": [int(doc_order[i]) for i in sub_rows],
                      "sub_fold_assignment": {str(int(doc_order[i])): folds[i] for i in sub_rows}})
    return plans


def window_for(train_counts_for_width, kind):
    """The committed label window and profile length for one probe window and labelling."""
    counts = {int(k): int(v) for k, v in train_counts_for_width[kind].items()}
    lo, n_labels, span, gaps = committed_label_window(counts)
    length, clamped = clamp_profile_length(span, n_labels)
    periods = whole_cycle_periods(length)
    return {"label_window_low": lo, "label_window_high": lo + n_labels - 1,
            "n_labels": n_labels,
            "committed_max_label_span": span,
            "committed_profile_length": length,
            "clamped_by_one_pair_rule": clamped,
            "label_values_absent_in_training": gaps,
            "nominal_pair_counts": nominal_pair_counts(n_labels, length),
            "n_periods_searched": len(periods),
            "longest_period": periods[0][1], "shortest_period": periods[-1][1],
            "min_training_positions_over_window":
                int(min(counts.get(v, 0) for v in range(lo, lo + n_labels))),
            "training_positions_at_top_label": int(counts.get(lo + n_labels - 1, 0))}


def stage_calibrate(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    reset_run_state(out_dir)
    stim = stimulus_path(args.stimulus)
    design = design_path(args.design_record)
    gate = gate_path(args.gate_commit)

    tot = load_group_sums(work_dir)
    n_read = tot["gram"].shape[1]
    doc_npz = np.load(work_dir / ("doc_sums_w%d.npz" % ANCHOR_WIDTH), allow_pickle=False)
    doc = {k: doc_npz[k] for k in ("doc_ids", "vec_sum", "cross", "label_sum", "label_sq", "n_pos")}
    doc_order = [int(v) for v in doc["doc_ids"]]
    with open(work_dir / "folds.json", "r", encoding="utf-8") as fh:
        fold_info = json.load(fh)
    fold_of = {int(k): int(v) for k, v in fold_info["assignment"].items()}
    doc_row = {d: i for i, d in enumerate(doc_order)}
    fold_rows = {f: [doc_row[d] for d in doc_order if fold_of[d] == f] for f in range(N_FOLDS)}
    with open(work_dir / "label_counts_train.json", "r", encoding="utf-8") as fh:
        train_counts = json.load(fh)

    # 1. Ridge strength for the main selection: five-fold cross-validation over the 36
    #    training documents, pooled over all nine wrap widths.
    print("[calibrate] cross-validating the ridge strength on training documents, all nine widths")
    main_surface = {}
    for layer in range(n_read):
        fold_sums = {f: sums_for_groups(tot, ["train_fold%d" % f], layer) for f in range(N_FOLDS)}
        total = sums_for_groups(tot, ["train_fold%d" % f for f in range(N_FOLDS)], layer)
        for f in range(N_FOLDS):
            fit = tuple(a - b for a, b in zip(total, fold_sums[f]))
            scores = fit_and_score_over_grid(fit, fold_sums[f])
            for alpha, r2 in scores.items():
                main_surface.setdefault((alpha, layer), []).append(r2)
    main_mean = {k: sum(v) / len(v) for k, v in main_surface.items()}
    alpha_main, cv_layer_main, cv_quality_main, main_per_alpha = best_of_surface(main_mean, n_read)
    print("[calibrate] main ridge strength %g, cross-validated best readout %d, quality %.6f"
          % (alpha_main, cv_layer_main, cv_quality_main))
    sys.stdout.flush()

    # 2. The anchor selection and the five resplits, all at the anchor width alone and
    #    all from each subset's own sub-training documents.
    plans = resplit_plan(doc_order, derived_seed("anchor_resplits"))
    anchor_surface = {}
    resplit_surface = [{} for _ in range(N_RESPLITS)]
    all_rows = list(range(len(doc_order)))
    for layer in range(n_read):
        dg = read_doc_grams(work_dir, layer)
        total = doc_subset_sums(dg, doc, all_rows, layer)
        fold_sums = {f: doc_subset_sums(dg, doc, fold_rows[f], layer) for f in range(N_FOLDS)}
        for f in range(N_FOLDS):
            fit = tuple(a - b for a, b in zip(total, fold_sums[f]))
            for alpha, r2 in fit_and_score_over_grid(fit, fold_sums[f]).items():
                anchor_surface.setdefault((alpha, layer), []).append(r2)
        for plan in plans:
            r = plan["resplit"]
            sub_total = doc_subset_sums(dg, doc, plan["sub_train_rows"], layer)
            sub_folds = {}
            for f in range(N_FOLDS):
                rows = [doc_row[int(d)] for d, ff in plan["sub_fold_assignment"].items() if ff == f]
                sub_folds[f] = doc_subset_sums(dg, doc, rows, layer)
            for f in range(N_FOLDS):
                fit = tuple(a - b for a, b in zip(sub_total, sub_folds[f]))
                for alpha, r2 in fit_and_score_over_grid(fit, sub_folds[f]).items():
                    resplit_surface[r].setdefault((alpha, layer), []).append(r2)
        del dg
        if layer % 6 == 0:
            print("[calibrate] anchor sweep readout %d of %d" % (layer, n_read - 1))
            sys.stdout.flush()

    anchor_mean = {k: sum(v) / len(v) for k, v in anchor_surface.items()}
    alpha_anchor, anchor_layer, anchor_cv_quality, anchor_per_alpha = best_of_surface(anchor_mean, n_read)
    print("[calibrate] anchor ridge strength %g, anchor readout %d, cross-validated quality %.6f"
          % (alpha_anchor, anchor_layer, anchor_cv_quality))

    resplit_choice = []
    for r in range(N_RESPLITS):
        mean_r = {k: sum(v) / len(v) for k, v in resplit_surface[r].items()}
        a_r, l_r, q_r, _ = best_of_surface(mean_r, n_read)
        resplit_choice.append({"resplit": r, "ridge_strength": a_r, "readout": l_r,
                               "cross_validated_quality": round(q_r, 6)})

    # 3. Each resplit's own held-out fit quality, at the readout its own sub-training
    #    documents chose. Only the readouts actually chosen are read back from disk.
    needed = sorted({c["readout"] for c in resplit_choice})
    resplit_values = [None] * N_RESPLITS
    for layer in needed:
        dg = read_doc_grams(work_dir, layer)
        for c in resplit_choice:
            if c["readout"] != layer:
                continue
            plan = plans[c["resplit"]]
            fit = doc_subset_sums(dg, doc, plan["sub_train_rows"], layer)
            held = doc_subset_sums(dg, doc, plan["held_out_rows"], layer)
            w, off = ridge_fit_from_sums(fit[0], fit[1], fit[2], fit[3], fit[5], c["ridge_strength"])
            r2, _, _ = r2_from_sums(w, off, held[0], held[1], held[2], held[3], held[4], held[5])
            resplit_values[c["resplit"]] = float(r2)
            c["held_out_quality"] = round(float(r2), 6)
            c["held_out_doc_ids"] = plan["held_out_doc_ids"]
            c["held_out_positions"] = int(held[5])
        del dg
    if any(v is None for v in resplit_values):
        raise Halt("a resplit produced no held-out fit quality; the anchor band cannot be measured")
    print("[calibrate] resplit values %s" % ["%.6f" % v for v in resplit_values])

    # 4. The anchor band. A spread of exactly zero halts here, before any held-out row.
    band = anchor_band(resplit_values)
    print("[calibrate] anchor band centre %.6f, spread %.6f, half-width %.6f, band %.6f to %.6f"
          % (band["centre"], band["mad"], band["half_width"], band["low"], band["high"]))

    # 5. The label window and profile length per probe window, per labelling.
    per_aperture = {}
    for a in APERTURES:
        per_aperture[str(a)] = {
            "characters": window_for(train_counts[str(a)], "characters"),
            "tokens": window_for(train_counts[str(a)], "tokens"),
        }
        c = per_aperture[str(a)]["characters"]
        t = per_aperture[str(a)]["tokens"]
        print("[calibrate] window %d: character labels %d to %d, profile length %d, %d periods; "
              "token labels 1 to %d, profile length %d, %d periods"
              % (a, c["label_window_low"], c["label_window_high"], c["committed_profile_length"],
                 c["n_periods_searched"], t["label_window_high"],
                 t["committed_profile_length"], t["n_periods_searched"]))

    values = {
        "base_seed": BASE_SEED,
        "shuffle_seed_streams": seed_stream_table(),
        "seed_formula": ("base seed times 10000000, plus the probe window times 1000, plus a "
                         "labelling code times 100, plus a bin index; code 0 is character "
                         "counts, 1 is token positions, 2 is a range-check bin"),
        "ridge_grid": RIDGE_GRID,
        "fold_count": N_FOLDS,
        "fold_split_seed": fold_info["fold_split_seed"],
        "fold_assignment": fold_info["assignment"],
        "main_ridge_strength": alpha_main,
        "main_ridge_strength_interior_on_grid": strength_is_interior(alpha_main, RIDGE_GRID),
        "main_cross_validated_best_readout": cv_layer_main,
        "main_cross_validated_best_quality": round(cv_quality_main, 6),
        "main_cross_validated_best_by_strength":
            {"%g" % a: {"readout": main_per_alpha[a][0], "quality": round(main_per_alpha[a][1], 6)}
             for a in RIDGE_GRID},
        "main_quality_surface":
            {"%g|%d" % (a, l): round(main_mean[(a, l)], 6) for a in RIDGE_GRID for l in range(n_read)},
        "anchor_width": ANCHOR_WIDTH,
        "anchor_ridge_strength": alpha_anchor,
        "anchor_ridge_strength_interior_on_grid": strength_is_interior(alpha_anchor, RIDGE_GRID),
        "anchor_layer": anchor_layer,
        "anchor_cross_validated_quality": round(anchor_cv_quality, 6),
        "anchor_cross_validated_best_by_strength":
            {"%g" % a: {"readout": anchor_per_alpha[a][0], "quality": round(anchor_per_alpha[a][1], 6)}
             for a in RIDGE_GRID},
        "anchor_quality_surface":
            {"%g|%d" % (a, l): round(anchor_mean[(a, l)], 6) for a in RIDGE_GRID for l in range(n_read)},
        "anchor_resplit_seed": derived_seed("anchor_resplits"),
        "anchor_resplit_holdout_documents": RESPLIT_HELDOUT_DOCS,
        "anchor_resplit_values": [round(v, 6) for v in resplit_values],
        "anchor_resplit_choices": resplit_choice,
        "anchor_resplit_held_out_doc_ids": [p["held_out_doc_ids"] for p in plans],
        "anchor_resplit_sub_fold_assignment": [p["sub_fold_assignment"] for p in plans],
        "anchor_resplit_median": round(band["median"], 6),
        "anchor_resplit_median_absolute_deviation": band["mad"],
        "anchor_band_centre": band["centre"],
        "anchor_band_half_width": band["half_width"],
        "anchor_band_low": band["low"],
        "anchor_band_high": band["high"],
        "band_half_width_below_0.02": band["band_half_width_below_0.02"],
        "detrend_method": "polynomial",
        "detrend_degree": DETREND_DEGREE,
        "shortest_period_searched": PERIOD_MIN,
        "shuffle_draws": N_DRAWS,
        "range_check_bins": [{"low": lo, "high": hi, "role": bin_role(lo, hi)} for lo, hi in BIN_EDGES],
        "range_check_draws_per_bin": N_BIN_DRAWS,
        "range_check_percentile": BIN_PERCENTILE,
        "per_aperture": per_aperture,
        "training_documents": N_TRAIN_DOCS,
        "training_widths": WIDTHS,
    }
    notes = [
        "Every number here comes from training documents alone. No held-out number exists yet.",
        "The label window of a probe window runs from 1 up to the largest label value that "
        "occurs at that wrap width in training documents, which is the wrap width plus one "
        "wherever a line-final line break sits on a full line. The committed profile length "
        "is that same count, reduced to the label count minus one, because a distance equal "
        "to the label count would average zero label pairs.",
        "The anchor procedure is self-contained at width 140: each of the five resplits "
        "selects its own readout and ridge strength on its own 30 sub-training documents, "
        "fits on those same documents, and scores on its own 6 held-out documents. So the "
        "spread across the five values includes the noise of the selection itself.",
        "Five values is a coarse spread, and each resplit's held-out set is half the size of "
        "the 12-document test set the band will judge. That weakness is recorded, not fixed.",
        "The band's centre, 0.781, is fixed by the question. This project's own records carry "
        "%.6f for a fit pooled over the seven widths 20 to 140, and %s for width-140-only "
        "fits. Those numbers are context; none of them moves the centre."
        % (PRIOR_POOLED_TO_140_VALUE, ", ".join("%.6f" % v for v in PRIOR_W140_ONLY_VALUES)),
        RERUN_NOTE,
    ]
    if band["band_half_width_below_0.02"]:
        notes.insert(4, "The measured half-width is under 0.02. That flag is recorded and "
                        "changes no decision; it marks that the anchor gate on this run was "
                        "decided by a spread estimate tighter than ordinary sampling variation.")

    write_record(
        prov, out_dir, base_cell("calibrate", "constants"),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=[(work_dir / "group_sums.npz", n_read),
                                    (work_dir / ("doc_sums_w%d.npz" % ANCHOR_WIDTH), N_TRAIN_DOCS),
                                    (work_dir / "label_counts_train.json", len(WIDTHS))]),
        metric="Every remaining constant of the fit and of the pattern readout, fixed on training documents alone.",
        values=values,
        threshold={"anchor_band_low": band["low"], "anchor_band_high": band["high"]},
        decision_rule=RULE_CALIBRATION,
        decision="CALIBRATION_OK",
        rows=resplit_choice,
        notes=" ".join(notes),
    )
    with open(Path(out_dir) / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump({"decision": "CALIBRATION_OK", "values": values}, fh, indent=2, sort_keys=True)
    print("[calibrate] committed record written before any held-out number exists")
    return 0


# ---------------------------------------------------------------------------
# Part 10. Stage three: the only stage that reads held-out documents.
#
# The gate commit pins this order: select the main layer by held-out fit pooled over
# all nine wrap widths, compute the anchor statistic and its gate, then the
# repeating-pattern statistic, the shuffle comparisons, the token-position control
# and the represented-range check, in that order.
# ---------------------------------------------------------------------------

def read_activations(work_dir, aperture, layer, n_rows, dim=EXPECT_HIDDEN):
    """The held-out per-position vectors at one probe window and one readout."""
    import numpy as np
    path = Path(work_dir) / ("act_a%d_layer%02d.f32" % (aperture, layer))
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != n_rows * dim:
        raise Halt("saved vectors for window %d readout %d hold %d numbers, expected %d"
                   % (aperture, layer, arr.size, n_rows * dim))
    return arr.reshape(n_rows, dim).astype(np.float64)


def slots_array(labels, window_lo, n_labels):
    """The window slot of every position, with -1 for a label outside the window."""
    import numpy as np
    off = labels - window_lo
    return np.where((off >= 0) & (off < n_labels), off, -1).astype(np.int64)


def means_by_label(x, slots, n_labels):
    """The average vector of every window slot, and how many positions entered each."""
    import numpy as np
    keep = slots >= 0
    s = slots[keep]
    xk = x[keep]
    counts = np.bincount(s, minlength=n_labels)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    xs = xk[order]
    starts = np.zeros(n_labels, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    present = counts > 0
    means = np.zeros((n_labels, x.shape[1]), dtype=np.float64)
    if present.any():
        sums = np.add.reduceat(xs, starts[present], axis=0)
        means[present] = sums / counts[present][:, None]
    return means, counts


def cosine_matrix(mat):
    """Cosine similarity between every pair of rows. Needs numpy."""
    import numpy as np
    norms = np.linalg.norm(mat, axis=1)
    bad = zero_norm_rows(norms.tolist())
    if bad:
        raise Halt("average vectors at rows %r have no length, so cosine similarity is "
                   "undefined there" % bad[:10])
    unit = mat / norms[:, None]
    return unit @ unit.T


def document_slices(docs):
    """One array of row positions per document, so a shuffle stays inside a document."""
    import numpy as np
    out = []
    for d in np.unique(docs):
        out.append(np.nonzero(docs == d)[0])
    return out


def shuffle_slots_within_documents(slots, doc_rows, rng):
    """Permute labels among the positions of each document, leaving its counts intact."""
    out = slots.copy()
    for rows in doc_rows:
        out[rows] = slots[rows][rng.permutation(rows.size)]
    return out


def ambient_dimension(centred, target=VARIANCE_TARGET):
    """How many directions carry the target share of the variance. Recorded, never gated."""
    import numpy as np
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv ** 2
    total = float(var.sum())
    if total <= 0:
        return None, []
    share = (var / total).tolist()
    cum = 0.0
    dim = len(share)
    for i, s in enumerate(share):
        cum += s
        if cum >= target:
            dim = i + 1
            break
    return dim, share[:10]


def pattern_with_null(x, labels, docs, window_lo, n_labels, profile_length, seed, n_draws=N_DRAWS):
    """The observed statistic, its shuffle comparison, and everything worth saving.

    The label multiset of each document is untouched by a shuffle, so exactly the same
    labels carry an average vector in every draw as in the observed measurement. The
    comparison is therefore between like and like.
    """
    import numpy as np
    slots = slots_array(labels, window_lo, n_labels)
    counts = np.bincount(slots[slots >= 0], minlength=n_labels)
    present_idx = np.nonzero(counts > 0)[0]
    label_values = [int(window_lo + int(i)) for i in present_idx]
    missing = [int(window_lo + i) for i in range(n_labels) if counts[i] == 0]
    pairs = pair_index_lists(label_values, profile_length)
    pair_counts = [len(r) for r, _ in pairs]
    empty = [d + 1 for d, c in enumerate(pair_counts) if c == 0]
    if empty:
        raise Halt("distances %r average zero label pairs, because the held-out documents "
                   "carry no position at labels %r; the committed profile length may not be "
                   "shortened after the fact" % (empty[:5], missing[:10]))
    pair_np = [(np.array(r, dtype=np.int64), np.array(c, dtype=np.int64)) for r, c in pairs]
    periods = whole_cycle_periods(profile_length)
    doc_rows = document_slices(docs)

    def statistic(these_slots):
        means, _ = means_by_label(x, these_slots, n_labels)
        mat = means[present_idx]
        centred = mat - mat.mean(axis=0, keepdims=True)
        cos = cosine_matrix(centred)
        profile = [float(cos[r, c].mean()) for r, c in pair_np]
        return max_power_statistic(profile, periods), profile, centred, means

    observed, profile, centred, means = statistic(slots)
    dim95, shares = ambient_dimension(centred)
    rng = np.random.default_rng(seed)
    draws = []
    spectra = []
    for i in range(n_draws):
        shuffled = shuffle_slots_within_documents(slots, doc_rows, rng)
        stat, _, _, _ = statistic(shuffled)
        draws.append(stat["max_power"])
        spectra.append({"draw": i, "max_power": stat["max_power"],
                        "wavelength": stat["wavelength"], "cycles": stat["cycles"],
                        "power_by_cycles": stat["powers"]})
    rank = null_rank(observed["max_power"], draws)
    return {
        "observed": observed, "profile": profile, "pair_counts": pair_counts,
        "periods": periods, "draws": draws, "spectra": spectra, "rank": rank,
        "label_values": label_values, "labels_missing_from_held_out": missing,
        "positions_per_label": [int(c) for c in counts.tolist()],
        "mean_vectors": means, "present_rows": present_idx.tolist(),
        "ambient_dimension_at_95_percent": dim95, "variance_share_first_ten": shares,
        "seed": seed,
    }


def save_pattern_artifacts(out_dir, aperture, labelling, result):
    """The full spectrum of every draw, and the average vectors, both saved for audit."""
    import numpy as np
    spec = Path(out_dir) / ("null_spectra_aperture-%d_labels-%s.jsonl" % (aperture, labelling))
    with open(spec, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": True, "aperture": aperture, "labelling": labelling,
                             "seed": result["seed"], "n_draws": len(result["spectra"]),
                             "cycles": [k for k, _ in result["periods"]],
                             "periods": [p for _, p in result["periods"]],
                             "observed_max_power": result["observed"]["max_power"],
                             "observed_wavelength": result["observed"]["wavelength"],
                             "observed_power_by_cycles": result["observed"]["powers"]}) + "\n")
        for row in result["spectra"]:
            fh.write(json.dumps(row) + "\n")
    means = Path(out_dir) / ("mean_vectors_aperture-%d_labels-%s.npz" % (aperture, labelling))
    np.savez(means,
             label_values=np.array(result["label_values"], dtype=np.int64),
             positions_per_label=np.array(result["positions_per_label"], dtype=np.int64),
             mean_vectors=result["mean_vectors"][result["present_rows"]].astype(np.float32),
             grand_mean=result["mean_vectors"][result["present_rows"]].mean(axis=0).astype(np.float32),
             profile=np.array(result["profile"], dtype=np.float64),
             pair_counts=np.array(result["pair_counts"], dtype=np.int64))
    return spec, means


def characters_per_token(out_dir, aperture):
    """The measured characters per token on one probe window's own held-out positions."""
    spanned = 0
    total_chars = 0
    labelled = 0
    path = Path(out_dir) / "rows_variants.jsonl"
    if not path.exists():
        raise Halt("no per-variant rows at %s; the extract stage must run first" % path)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["width"] == aperture and row["split"] == "test":
                spanned += int(row["chars_spanned_by_labelled_tokens"])
                total_chars += int(row["n_chars"])
                labelled += int(row["n_labelled"])
    if labelled <= 0:
        raise Halt("no labelled held-out positions at window %d" % aperture)
    return spanned / float(labelled), total_chars / float(labelled), labelled


def stage_endpoints(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    reset_run_state(out_dir)
    stim = stimulus_path(args.stimulus)
    design = design_path(args.design_record)
    gate = gate_path(args.gate_commit)

    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        raise Halt("no calibration record at %s; the calibration stage must run and commit "
                   "its record before any held-out number is read" % calib_path)
    with open(calib_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    if calib.get("decision") != "CALIBRATION_OK":
        raise Halt("the calibration record says %r; no held-out number may be computed"
                   % calib.get("decision"))
    cv = calib["values"]
    alpha_main = float(cv["main_ridge_strength"])
    alpha_anchor = float(cv["anchor_ridge_strength"])
    anchor_layer = int(cv["anchor_layer"])
    band_low = float(cv["anchor_band_low"])
    band_high = float(cv["anchor_band_high"])
    ANCHOR_CONTEXT["band_low"] = band_low
    ANCHOR_CONTEXT["band_high"] = band_high
    ANCHOR_CONTEXT["band_half_width"] = float(cv["anchor_band_half_width"])
    ANCHOR_CONTEXT["band_centre"] = float(cv["anchor_band_centre"])

    tot = load_group_sums(work_dir)
    n_read = tot["gram"].shape[1]
    train_groups = ["train_fold%d" % f for f in range(N_FOLDS)]
    test_groups = ["test_w%d" % w for w in WIDTHS]
    common_inputs = [(work_dir / "group_sums.npz", n_read)]

    # 1. The main layer: best held-out fit, pooled over all nine wrap widths.
    per_layer = []
    for layer in range(n_read):
        fit = sums_for_groups(tot, train_groups, layer)
        ev = sums_for_groups(tot, test_groups, layer)
        w, off = ridge_fit_from_sums(fit[0], fit[1], fit[2], fit[3], fit[5], alpha_main)
        r2, _, _ = r2_from_sums(w, off, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5])
        per_layer.append({"readout": layer, "held_out_quality": round(float(r2), 6)})
    main_layer = int(max(per_layer, key=lambda r: r["held_out_quality"])["readout"])
    main_quality = float(max(r["held_out_quality"] for r in per_layer))
    print("[endpoints] main layer %d, held-out fit quality pooled over nine widths %.6f"
          % (main_layer, main_quality))
    write_record(
        prov, out_dir, base_cell("endpoints", "main_layer_selection"),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Held-out fit quality of the calibrated ridge fit, pooled over all nine wrap widths, at every readout.",
        values={"main_layer": main_layer, "main_layer_held_out_quality": round(main_quality, 6),
                "ridge_strength": alpha_main,
                "ridge_strength_interior_on_grid": strength_is_interior(alpha_main, RIDGE_GRID),
                "training_positions": int(sum(tot["n_pos"][tot["groups"].index(g)] for g in train_groups)),
                "held_out_positions": int(sum(tot["n_pos"][tot["groups"].index(g)] for g in test_groups)),
                "n_readouts": n_read},
        threshold={},
        decision_rule=RULE_MAIN_LAYER,
        decision="RECORDED_NOT_GATED",
        rows=per_layer,
        notes=("Readout 0 is the embedding output. Where the fit peaks is recorded and "
               "never judged, which the design says in those words. " + RERUN_NOTE),
    )

    # 2. The anchor gate. Everything written after this carries the run's validity.
    fit140 = sums_for_groups(tot, ["train_w%d" % ANCHOR_WIDTH], anchor_layer)
    test140 = sums_for_groups(tot, ["test_w%d" % ANCHOR_WIDTH], anchor_layer)
    w_a, off_a = ridge_fit_from_sums(fit140[0], fit140[1], fit140[2], fit140[3], fit140[5], alpha_anchor)
    anchor_r2, _, _ = r2_from_sums(w_a, off_a, test140[0], test140[1], test140[2],
                                   test140[3], test140[4], test140[5])
    anchor_r2 = float(anchor_r2)
    anchor_decision = decide_anchor(anchor_r2, band_low, band_high)
    ANCHOR_CONTEXT["decision"] = anchor_decision
    ANCHOR_CONTEXT["statistic"] = round(anchor_r2, 6)
    print("[endpoints] anchor statistic %.6f against band %.6f to %.6f -> %s"
          % (anchor_r2, band_low, band_high, anchor_decision))
    write_record(
        prov, out_dir, base_cell("endpoints", "anchor_gate", endpoint=1),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Held-out fit quality at the anchor layer, width 140 alone, on the 12 fresh held-out documents.",
        values={"anchor_layer": anchor_layer, "anchor_ridge_strength": alpha_anchor,
                "anchor_ridge_strength_interior_on_grid":
                    strength_is_interior(alpha_anchor, RIDGE_GRID),
                "held_out_quality": round(anchor_r2, 6),
                "resplit_values": cv["anchor_resplit_values"],
                "resplit_median": cv["anchor_resplit_median"],
                "resplit_median_absolute_deviation": cv["anchor_resplit_median_absolute_deviation"],
                "band_half_width": cv["anchor_band_half_width"],
                "band_low": band_low, "band_high": band_high,
                "band_centre": cv["anchor_band_centre"],
                "band_half_width_below_0.02": cv["band_half_width_below_0.02"],
                "held_out_positions": int(test140[5]),
                "training_positions": int(fit140[5])},
        threshold={"low": band_low, "high": band_high, "centre": cv["anchor_band_centre"]},
        decision_rule=RULE_ANCHOR,
        decision=anchor_decision,
        rows=cv["anchor_resplit_choices"],
        notes=("The band's half-width is measured from five resplits of the training "
               "documents, not guessed. The centre, 0.781, is fixed by the question and was "
               "not moved. " + RERUN_NOTE),
    )
    if anchor_decision != "ANCHOR_IN_BAND":
        void_the_run("the anchor gate failed: held-out fit quality %.6f falls outside the "
                     "measured band %.6f to %.6f, so every result this run produces is an "
                     "instrument fault and is not evidence about the model"
                     % (anchor_r2, band_low, band_high), out_dir)
    else:
        save_run_state(out_dir)

    # 3. The readings the gate commit requires to be recorded and never gated.
    fit_main = sums_for_groups(tot, train_groups, main_layer)
    w_m, off_m = ridge_fit_from_sums(fit_main[0], fit_main[1], fit_main[2], fit_main[3],
                                     fit_main[5], alpha_main)
    per_width = {}
    for wdt in WIDTHS:
        ev = sums_for_groups(tot, ["test_w%d" % wdt], main_layer)
        r2w, _, _ = r2_from_sums(w_m, off_m, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5])
        per_width[str(wdt)] = round(float(r2w), 6)
    ev_to140 = sums_for_groups(tot, ["test_w%d" % wdt for wdt in POOLED_TO_140], main_layer)
    r2_to140, _, _ = r2_from_sums(w_m, off_m, ev_to140[0], ev_to140[1], ev_to140[2],
                                  ev_to140[3], ev_to140[4], ev_to140[5])
    write_record(
        prov, out_dir, base_cell("endpoints", "recorded_fit_readings"),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Held-out fit quality of the pooled nine-width probe, read several ways, so an out-of-band anchor can be told apart from a mismatch between two different fits.",
        values={"anchor_statistic_width140_only_fit": round(anchor_r2, 6),
                "pooled_probe_scored_on_width140_only": per_width["140"],
                "pooled_probe_scored_on_widths_20_to_140": round(float(r2_to140), 6),
                "pooled_probe_by_wrap_width": per_width,
                "main_layer": main_layer, "anchor_layer": anchor_layer,
                "project_records_pooled_20_to_140": PRIOR_POOLED_TO_140_VALUE,
                "project_records_width140_only": PRIOR_W140_ONLY_VALUES},
        threshold={},
        decision_rule=RULE_RECORDED_ONLY,
        decision="RECORDED_NOT_GATED",
        notes=("The anchor statistic fits and scores at width 140 alone, which is the "
               "design's own reading. The other readings here exist so a reader can see "
               "whether an out-of-band anchor means the instrument drifted or means the "
               "band's centre belongs to a different fit. None of them decides anything. "
               + RERUN_NOTE),
    )

    # 4. The repeating-pattern statistic and its shuffle comparison, per probe window.
    act_meta = {}
    cleared = {}
    char_results = {}
    for a in APERTURES:
        meta = np.load(work_dir / ("act_meta_a%d.npz" % a), allow_pickle=False)
        act_meta[a] = {"doc": meta["doc"], "char": meta["char_label"],
                       "token": meta["token_label"], "n_rows": int(meta["char_label"].size)}
    for a in APERTURES:
        spec = cv["per_aperture"][str(a)]["characters"]
        x = read_activations(work_dir, a, main_layer, act_meta[a]["n_rows"])
        t0 = time.time()
        res = pattern_with_null(x, act_meta[a]["char"], act_meta[a]["doc"],
                                spec["label_window_low"], spec["n_labels"],
                                spec["committed_profile_length"],
                                derived_seed("characters", a))
        decision = decide_ripple(res["rank"])
        char_results[a] = res
        if decision == "NULL_CLEARED":
            cleared[a] = res["observed"]["wavelength"]
        spec_path, mean_path = save_pattern_artifacts(out_dir, a, "characters", res)
        print("[endpoints] window %d characters: largest power %.6g at period %.4f, rank %d of "
              "%d -> %s (%.0f s)"
              % (a, res["observed"]["max_power"], res["observed"]["wavelength"], res["rank"],
                 N_DRAWS, decision, time.time() - t0))
        sys.stdout.flush()
        write_record(
            prov, out_dir, base_cell("endpoints", "pattern_vs_shuffle", endpoint=2,
                                     aperture=a, labelling="characters"),
            inputs=record_inputs(prov, stim, design, gate,
                                 extra=common_inputs + [(work_dir / ("act_meta_a%d.npz" % a),
                                                         act_meta[a]["n_rows"])]),
            metric="Largest power over every whole-cycle period from 4 characters up to the committed profile length, against this probe window's own 200-draw shuffle comparison.",
            values={"aperture": a, "labelling": "characters", "main_layer": main_layer,
                    "max_power": res["observed"]["max_power"],
                    "wavelength_chars": res["observed"]["wavelength"],
                    "cycles": res["observed"]["cycles"],
                    "rank": res["rank"], "draws": N_DRAWS,
                    "share_of_draws_at_or_above": round(res["rank"] / (N_DRAWS + 1.0), 5),
                    "null_largest": max(res["draws"]), "null_median": median(res["draws"]),
                    "n_periods_searched": res["observed"]["n_periods"],
                    "committed_profile_length": spec["committed_profile_length"],
                    "label_window_low": spec["label_window_low"],
                    "label_window_high": spec["label_window_high"],
                    "labels_missing_from_held_out": res["labels_missing_from_held_out"],
                    "positions_per_label_min": min(c for c in res["positions_per_label"] if c > 0),
                    "positions_per_label_max": max(res["positions_per_label"]),
                    "held_out_positions": int(act_meta[a]["n_rows"]),
                    "ambient_dimension_at_95_percent": res["ambient_dimension_at_95_percent"],
                    "variance_share_first_ten": [round(s, 6) for s in res["variance_share_first_ten"]],
                    "pair_counts_by_distance": res["pair_counts"],
                    "profile": [round(v, 8) for v in res["profile"]],
                    "observed_power_by_cycles": res["observed"]["powers"],
                    "shuffle_seed": res["seed"],
                    "full_spectra_file": spec_path.name,
                    "mean_vectors_file": mean_path.name},
            threshold={"rank_at_most": NULL_RANK_MAX, "draws": N_DRAWS},
            decision_rule=RULE_RIPPLE,
            decision=decision,
            rows=[{"draw": s["draw"], "max_power": s["max_power"], "wavelength": s["wavelength"]}
                  for s in res["spectra"]],
            notes=("A shuffle permutes the labels inside each document, so every label keeps "
                   "exactly the same number of positions and the same labels carry an average "
                   "vector in every draw. Every draw's full per-period power vector is in %s, "
                   "and the average vectors are in %s. The far tail of the profile rests on "
                   "few label pairs; the count per distance is recorded here. The measured "
                   "number of directions is recorded and constrains nothing. %s"
                   % (spec_path.name, mean_path.name, RERUN_NOTE)),
        )
        del x

    # 5. The token-position control, per probe window.
    token_results = {}
    for a in APERTURES:
        spec = cv["per_aperture"][str(a)]["tokens"]
        rate_spanned, rate_total, n_labelled = characters_per_token(out_dir, a)
        char_res = char_results[a]
        char_cleared = decide_ripple(char_res["rank"]) == "NULL_CLEARED"
        values = {"aperture": a, "labelling": "tokens", "main_layer": main_layer,
                  "characters_per_token": round(rate_spanned, 6),
                  "characters_per_token_from_whole_text": round(rate_total, 6),
                  "held_out_labelled_positions": n_labelled,
                  "character_wavelength_chars": char_res["observed"]["wavelength"],
                  "character_cleared": char_cleared,
                  "token_label_window_high": spec["label_window_high"],
                  "committed_profile_length_tokens": spec["committed_profile_length"]}
        wl_token_chars = None
        token_cleared = None
        try:
            x = read_activations(work_dir, a, main_layer, act_meta[a]["n_rows"])
            res = pattern_with_null(x, act_meta[a]["token"], act_meta[a]["doc"],
                                    spec["label_window_low"], spec["n_labels"],
                                    spec["committed_profile_length"],
                                    derived_seed("tokens", a))
            del x
            token_results[a] = res
            token_decision = decide_ripple(res["rank"])
            token_cleared = token_decision == "NULL_CLEARED"
            wl_token_chars = res["observed"]["wavelength"] * rate_spanned
            spec_path, mean_path = save_pattern_artifacts(out_dir, a, "tokens", res)
            values.update({
                "max_power": res["observed"]["max_power"],
                "wavelength_tokens": res["observed"]["wavelength"],
                "wavelength_in_characters": round(wl_token_chars, 6),
                "cycles": res["observed"]["cycles"],
                "rank": res["rank"], "draws": N_DRAWS,
                "share_of_draws_at_or_above": round(res["rank"] / (N_DRAWS + 1.0), 5),
                "null_largest": max(res["draws"]), "null_median": median(res["draws"]),
                "token_null_decision": token_decision,
                "token_cleared": token_cleared,
                "n_periods_searched": res["observed"]["n_periods"],
                "labels_missing_from_held_out": res["labels_missing_from_held_out"],
                "positions_per_label_min": min(c for c in res["positions_per_label"] if c > 0),
                "positions_per_label_max": max(res["positions_per_label"]),
                "pair_counts_by_distance": res["pair_counts"],
                "profile": [round(v, 8) for v in res["profile"]],
                "observed_power_by_cycles": res["observed"]["powers"],
                "shuffle_seed": res["seed"],
                "full_spectra_file": spec_path.name,
                "mean_vectors_file": mean_path.name,
                "wavelength_gap_chars": round(abs(char_res["observed"]["wavelength"]
                                                  - wl_token_chars), 6)})
            rows = [{"draw": s["draw"], "max_power": s["max_power"], "wavelength": s["wavelength"]}
                    for s in res["spectra"]]
        except Halt as exc:
            values["not_computable_reason"] = str(exc)
            rows = None
            print("[endpoints] window %d token positions: not computable: %s" % (a, exc))
        credit = decide_token_credit(char_res["observed"]["wavelength"], wl_token_chars,
                                     char_cleared, token_cleared)
        print("[endpoints] window %d token-position control -> %s" % (a, credit))
        sys.stdout.flush()
        write_record(
            prov, out_dir, base_cell("endpoints", "token_position_control", endpoint=7, aperture=a),
            inputs=record_inputs(prov, stim, design, gate,
                                 extra=common_inputs + [(work_dir / ("act_meta_a%d.npz" % a),
                                                         act_meta[a]["n_rows"])]),
            metric="Gap in characters between the character-count pattern's wavelength and the token-position control's wavelength, after converting at the measured characters-per-token rate.",
            values=values,
            threshold={"margin_chars": CONFOUND_MARGIN_CHARS},
            decision_rule=RULE_TOKEN,
            decision=credit,
            rows=rows,
            notes=("A token's label here counts the sub-word tokens since the last line "
                   "break. The rate that converts token periods into characters is measured "
                   "on this probe window's own labelled held-out positions, and all three "
                   "rates are recorded. " + RERUN_NOTE),
        )

    # 6. The represented-range check, at window 280 only.
    spec280 = cv["per_aperture"][str(RANGE_APERTURE)]["characters"]
    x280 = read_activations(work_dir, RANGE_APERTURE, main_layer, act_meta[RANGE_APERTURE]["n_rows"])
    labels280 = act_meta[RANGE_APERTURE]["char"]
    pred280 = (x280 @ w_m + off_m)
    del x280
    bin_rows = []
    excluded_above = int(np.count_nonzero(labels280 > BIN_EDGES[-1][1]))
    for j, (lo, hi) in enumerate(BIN_EDGES):
        sel = np.nonzero((labels280 >= lo) & (labels280 <= hi))[0]
        row = {"bin": j, "low": lo, "high": hi, "role": bin_role(lo, hi),
               "held_out_positions": int(sel.size)}
        if sel.size == 0:
            row.update({"held_out_quality": None, "decision": "BIN_NOT_COMPUTABLE",
                        "reason": "no held-out position carries a label in this bin"})
            bin_rows.append(row)
            continue
        y = [float(v) for v in labels280[sel].tolist()]
        p = [float(v) for v in pred280[sel].tolist()]
        if not has_label_spread(y):
            row.update({"held_out_quality": None, "decision": "BIN_NOT_COMPUTABLE",
                        "reason": "the labels in this bin have no spread, so fit quality has "
                                  "no denominator"})
            bin_rows.append(row)
            continue
        observed = r2_from_pairs(y, p)
        rng = np.random.default_rng(derived_seed("range_bin", RANGE_APERTURE, j))
        draws = []
        for _ in range(N_BIN_DRAWS):
            perm = rng.permutation(len(y))
            draws.append(r2_from_pairs([y[int(k)] for k in perm], p))
        cut = percentile_linear(draws, BIN_PERCENTILE)
        row.update({"held_out_quality": round(observed, 6),
                    "null_95th_percentile": round(cut, 6),
                    "null_largest": round(max(draws), 6),
                    "null_median": round(median(draws), 6),
                    "rank_among_draws": null_rank(observed, draws),
                    "draws": N_BIN_DRAWS,
                    "null_draw_values": [round(v, 6) for v in draws],
                    "shuffle_seed": derived_seed("range_bin", RANGE_APERTURE, j),
                    "decision": decide_bin(observed, draws)})
        bin_rows.append(row)
    range_pattern = decide_range_pattern(bin_rows)
    print("[endpoints] represented-range check -> %s" % range_pattern)
    write_record(
        prov, out_dir, base_cell("endpoints", "represented_range", endpoint=6,
                                 aperture=RANGE_APERTURE),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=common_inputs + [(work_dir / ("act_meta_a%d.npz" % RANGE_APERTURE),
                                                     act_meta[RANGE_APERTURE]["n_rows"])]),
        metric="Within-bin held-out fit quality inside each of eight 35-label bins at window 280, against that same bin's own 50-draw label-shuffle comparison, at the main layer.",
        values={"aperture": RANGE_APERTURE, "main_layer": main_layer,
                "ridge_strength": alpha_main,
                "bins_represented": [r["bin"] for r in bin_rows if r["decision"] == "REPRESENTED"],
                "low_bins_all_represented":
                    all(r["decision"] == "REPRESENTED" for r in bin_rows if r["role"] == "low"),
                "high_bins_any_represented":
                    any(r["decision"] == "REPRESENTED" for r in bin_rows if r["role"] == "high"),
                "straddling_bin": [r["bin"] for r in bin_rows if r["role"] == "straddles"],
                "positions_excluded_above_label_280": excluded_above,
                "draws_per_bin": N_BIN_DRAWS,
                "percentile": BIN_PERCENTILE,
                "committed_label_window_high": spec280["label_window_high"]},
        threshold={"percentile": BIN_PERCENTILE, "draws": N_BIN_DRAWS},
        decision_rule=RULE_RANGE,
        decision=range_pattern,
        rows=bin_rows,
        notes=("A bin's own labels are the denominator of its fit quality, and each draw "
               "permutes those same positions' labels among themselves. The bin spanning "
               "label 150, from 141 to 175, is recorded and left out of the decision. "
               "Positions whose label is above 280, which only line-final line breaks reach, "
               "fall in no bin and are counted as excluded. " + RERUN_NOTE),
    )

    # 7. The count of clearing windows, the two gaps, and the fork.
    n_cleared = len(cleared)
    write_record(
        prov, out_dir, base_cell("endpoints", "windows_cleared", endpoint=3),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Count, among the three probe windows, of how many clear their own shuffle comparison.",
        values={"windows_cleared": n_cleared,
                "cleared_apertures": sorted(cleared),
                "wavelength_by_cleared_aperture": {str(k): cleared[k] for k in sorted(cleared)},
                "rank_by_aperture": {str(a): char_results[a]["rank"] for a in APERTURES},
                "represented_range_pattern": range_pattern},
        threshold={"decisive_negative_at": 0},
        decision_rule=RULE_COUNT,
        decision=("NO_WINDOW_CLEARED" if n_cleared == 0 else "AT_LEAST_ONE_WINDOW_CLEARED"),
        notes=("None clearing is a decisive negative only when the represented-range check "
               "shows label counts above 150 are represented. If instead every bin at or "
               "below 140 is represented and no bin above 150 is, the same result is "
               "recorded as a range that was never measurable. " + RERUN_NOTE),
    )
    call, wl_gap, ratio_gap = decide_invariance(cleared)
    write_record(
        prov, out_dir, base_cell("endpoints", "wavelength_gap", endpoint=4),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Largest gap in characters between any two wavelengths, among the probe windows that cleared their own comparison.",
        values={"wavelength_gap_chars": wl_gap,
                "wavelength_by_cleared_aperture": {str(k): cleared[k] for k in sorted(cleared)},
                "windows_cleared": n_cleared},
        threshold={"at_most_chars": INVARIANCE_MAX_GAP_CHARS},
        decision_rule=RULE_WAVELENGTH_GAP,
        decision=decide_wavelength_gap(wl_gap),
        notes=("Fewer than two clearing windows leaves this gap undefined, and it is "
               "recorded as absent rather than as a number. " + RERUN_NOTE),
    )
    write_record(
        prov, out_dir, base_cell("endpoints", "ratio_gap", endpoint=5),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="Largest gap between any two wavelength-to-window-width ratios, among the probe windows that cleared their own comparison.",
        values={"ratio_gap": ratio_gap,
                "ratio_by_cleared_aperture":
                    {str(k): cleared[k] / float(k) for k in sorted(cleared)},
                "wavelength_gap_chars": wl_gap,
                "windows_cleared": n_cleared,
                "fork_call": call},
        threshold={"within": PROPORTION_MAX_RATIO_GAP},
        decision_rule=RULE_RATIO_GAP,
        decision=decide_ratio_gap(ratio_gap),
        notes=("One shared wavelength is ruled out first; only then does a ratio within 0.1 "
               "count as a wavelength that scales with the window. Exactly one clearing "
               "window is undetermined and is never recorded as mixed. " + RERUN_NOTE),
    )

    # 8. The roll-up. It gathers calls and computes no new number.
    headline = decide_headline(RUN_STATE["run_valid"], cleared, range_pattern)
    summary = {
        "headline": headline,
        "run_valid": RUN_STATE["run_valid"],
        "endpoint_1_anchor_gate": anchor_decision,
        "endpoint_2_pattern_by_window":
            {str(a): decide_ripple(char_results[a]["rank"]) for a in APERTURES},
        "endpoint_3_windows_cleared": n_cleared,
        "endpoint_4_wavelength_gap_chars": wl_gap,
        "endpoint_5_ratio_gap": ratio_gap,
        "endpoint_6_represented_range": range_pattern,
        "endpoint_7_token_control_by_window":
            {str(a): decide_token_credit(
                char_results[a]["observed"]["wavelength"],
                (token_results[a]["observed"]["wavelength"]
                 * characters_per_token(out_dir, a)[0]) if a in token_results else None,
                decide_ripple(char_results[a]["rank"]) == "NULL_CLEARED",
                (decide_ripple(token_results[a]["rank"]) == "NULL_CLEARED")
                if a in token_results else None) for a in APERTURES},
        "main_layer": main_layer,
        "anchor_layer": anchor_layer,
        "anchor_statistic": round(anchor_r2, 6),
        "anchor_band_low": band_low, "anchor_band_high": band_high,
        "anchor_band_half_width": cv["anchor_band_half_width"],
        "band_half_width_below_0.02": cv["band_half_width_below_0.02"],
        "main_ridge_strength": alpha_main,
        "main_ridge_strength_interior_on_grid": strength_is_interior(alpha_main, RIDGE_GRID),
        "anchor_ridge_strength": alpha_anchor,
        "anchor_ridge_strength_interior_on_grid": strength_is_interior(alpha_anchor, RIDGE_GRID),
        "wavelength_by_cleared_aperture": {str(k): cleared[k] for k in sorted(cleared)},
        "rank_by_aperture": {str(a): char_results[a]["rank"] for a in APERTURES},
    }
    write_record(
        prov, out_dir, base_cell("endpoints", "run_summary"),
        inputs=record_inputs(prov, stim, design, gate, extra=common_inputs),
        metric="One call per endpoint, gathered from the records written beside this one.",
        values=summary,
        threshold={},
        decision_rule=RULE_SUMMARY,
        decision=headline,
        notes=("A failed anchor gate makes every other call here void as an instrument "
               "fault, not evidence about the model, and the void flag is written into "
               "every record produced after the gate, per-probe-window files included. "
               + RERUN_NOTE),
    )
    save_run_state(out_dir)
    print("[endpoints] %s" % json.dumps(summary, sort_keys=True))
    if not RUN_STATE["run_valid"]:
        print("[endpoints] THIS RUN IS VOID. The anchor gate failed, so every pattern, "
              "control and range-check number above is an instrument fault and is not "
              "evidence about the model. Every record written after the gate carries "
              "run_valid=false in its own values.")
    return 0


# ---------------------------------------------------------------------------
# Part 11. Self-test. Standard library only: no model, no graphics card, no numpy
# needed. It runs on the processor before any graphics-card time, both on this
# workstation and again on the rented box, and exits non-zero on any failure.
#
# The gate commit names, for each endpoint, the planted ground truth its case must
# use. Each block below is labelled with the endpoint it validates.
# ---------------------------------------------------------------------------

def _check(fails, name, got, want, tol=0.0):
    if isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print("  %-64s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def _check_halt(fails, name, fn, needle=None):
    try:
        fn()
    except Halt as exc:
        if needle is not None and needle not in str(exc):
            fails.append(name)
            print("  %-64s FAIL halted without %r: %s" % (name, needle, exc))
            return
        print("  %-64s ok" % name)
        return
    fails.append(name)
    print("  %-64s FAIL no halt" % name)


def selftest_parsers(fails):
    print("== the position labeller ==")
    ch, tk, ln = labels_for_spans("ab\ncd", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    _check(fails, "columns on a two-line text", ch, [1, 2, 3, 1, 2])
    _check(fails, "token counts on a two-line text", tk, [1, 2, 3, 1, 2])
    _check(fails, "line numbers on a two-line text", ln, [0, 0, 0, 1, 1])
    ch1, tk1, _ = labels_for_spans("abcd", [(0, 2), (2, 4)])
    _check(fails, "implied opening boundary, first line columns", ch1, [2, 4])
    _check(fails, "implied opening boundary, first line token counts", tk1, [1, 2])
    ch2, tk2, ln2 = labels_for_spans("ab\ncd", [(0, 2), (2, 4), (4, 5)])
    _check(fails, "a swallowed line break, columns", ch2, [2, 1, 2])
    _check(fails, "a swallowed line break, token counts", tk2, [1, 1, 2])
    _check(fails, "a swallowed line break, line numbers", ln2, [0, 1, 1])
    ch3, _, _ = labels_for_spans("abc\nd", [(0, 3), (3, 4), (4, 5)])
    _check(fails, "a line-break token takes the next column", ch3, [3, 4, 1])
    full = "x" * 140 + "\n" + "y"
    ch4, _, _ = labels_for_spans(full, [(0, 140), (140, 141), (141, 142)])
    _check(fails, "a full line's line break lands at the width plus one", ch4, [140, 141, 1])
    ch5, tk5, _ = labels_for_spans("ab", [(0, 0), (0, 2)])
    _check(fails, "an empty range is never labelled", ch5, [None, 2])
    _check(fails, "an empty range is never counted", tk5, [None, 1])
    _check(fails, "position counts per label", count_labels([1, 2, 2, None, 5]),
           {1: 1, 2: 2, 5: 1})

    print("== the committed text file reader ==")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = {"meta": {"widths": WIDTHS, "n_docs": N_DOCS, "train_docs": N_TRAIN_DOCS},
                "docs": [{"id": i, "split": "train" if i < N_TRAIN_DOCS else "test",
                          "wrapped": {str(w): "a\nb" for w in WIDTHS}} for i in range(N_DOCS)]}
        p = Path(d) / "good.json"
        p.write_text(json.dumps(good))
        meta, docs = load_stimulus(p)
        _check(fails, "a well-formed text file loads", len(docs), N_DOCS)
        bad = json.loads(json.dumps(good))
        bad["docs"] = bad["docs"][:-1]
        q = Path(d) / "bad.json"
        q.write_text(json.dumps(bad))
        _check_halt(fails, "a missing document halts", lambda: load_stimulus(q), "document count")
        bad2 = json.loads(json.dumps(good))
        bad2["meta"]["widths"] = [20, 40]
        r = Path(d) / "bad2.json"
        r.write_text(json.dumps(bad2))
        _check_halt(fails, "wrong wrap widths halt", lambda: load_stimulus(r), "wrap widths")
        bad3 = json.loads(json.dumps(good))
        del bad3["docs"][0]["wrapped"]["280"]
        t = Path(d) / "bad3.json"
        t.write_text(json.dumps(bad3))
        _check_halt(fails, "a document missing a wrap width halts", lambda: load_stimulus(t), "lacks widths")

    print("== the calibration record reader ==")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out"
        out.mkdir()
        (out / "calibration.json").write_text(json.dumps({"decision": "CALIBRATION_OK",
                                                          "values": {"main_ridge_strength": 1000.0}}))
        loaded = json.loads((out / "calibration.json").read_text())
        _check(fails, "a committed calibration record reads back", loaded["decision"], "CALIBRATION_OK")
        _check(fails, "its committed strength reads back",
               loaded["values"]["main_ridge_strength"], 1000.0)


def selftest_detectors(fails):
    print("== the degenerate-input and degenerate-measurement detectors ==")
    _check(fails, "reaching the context limit flags", degenerate_flags(2048, 2048, ["a", "b"]),
           ["token_cap"])
    _check(fails, "under the context limit does not flag",
           degenerate_flags(700, 2048, ["a", "b", "c"]), [])
    rep = ["x", "y"] * 7
    _check(fails, "a repeated block flags", degenerate_flags(14, 2048, rep), ["repeated_sequence"])
    _check(fails, "seven back-to-back copies counted", max_block_repeats(rep), 7)
    _check(fails, "prose-like tokens are clean",
           degenerate_flags(9, 2048, ["It", " will", " be", " seen", " that", " this", " mere",
                                      " b", "\n"]), [])
    _check(fails, "both degenerate conditions can flag together",
           degenerate_flags(2048, 2048, ["q"] * 6), ["token_cap", "repeated_sequence"])
    _check(fails, "an average vector of no length is found", zero_norm_rows([1.0, 0.0, 2.0]), [1])
    _check(fails, "vectors that all have length are clean", zero_norm_rows([1.0, 2.0]), [])
    _check(fails, "held-out labels that differ have spread", has_label_spread([1, 2, 2]), True)
    _check(fails, "held-out labels that are all equal have no spread",
           has_label_spread([7, 7, 7]), False)
    _check_halt(fails, "no spread halts fit quality rather than dividing by zero",
                lambda: r2_from_pairs([5.0, 5.0], [1.0, 2.0]), "no spread")


def selftest_windows(fails):
    print("== exact matches, label windows and seeds ==")
    _check(fails, "identical hashes match", exact_hash_match("a" * 64, "a" * 64), True)
    _check(fails, "hashes one character apart do not match",
           exact_hash_match("a" * 63 + "b", "a" * 64), False)
    _check(fails, "the expected configuration matches", exact_config_match(24, 1024), True)
    _check(fails, "a different layer count does not match", exact_config_match(23, 1024), False)
    _check(fails, "a different hidden size does not match", exact_config_match(24, 2048), False)
    _check(fails, "the setup call with everything matching", decide_setup(True, 24, 1024), "SETUP_OK")
    _check(fails, "the setup call with a different text hash",
           decide_setup(False, 24, 1024), "SETUP_FAILURE")
    _check(fails, "the setup call with a different layer count",
           decide_setup(True, 12, 1024), "SETUP_FAILURE")
    _check(fails, "the lowest window value maps to slot 0", label_slot(1, 1, 141), 0)
    _check(fails, "the highest window value maps to the last slot", label_slot(141, 1, 141), 140)
    _check(fails, "one below the window is outside", label_slot(0, 1, 141), None)
    _check(fails, "one above the window is outside", label_slot(142, 1, 141), None)
    _check(fails, "a fractional label is never accepted", label_slot(1.0, 1, 141), None)
    _check(fails, "a true-or-false value is never accepted as a label", label_slot(True, 1, 141), None)
    counts, slots = slot_counts([1, 2, 2, 5, 0], 1, 4)
    _check(fails, "slot counts", counts, [1, 2, 0, 0])
    _check(fails, "slot indices", slots, [0, 1, 1, None, None])

    lo, n_labels, span, gaps = committed_label_window({1: 9, 2: 4, 3: 7})
    _check(fails, "the window runs from 1 to the largest training label", (lo, n_labels, span),
           (1, 3, 3))
    _check(fails, "a fully populated window has no gaps", gaps, [])
    lo2, n2, span2, gaps2 = committed_label_window({1: 3, 3: 1})
    _check(fails, "a label no training position carries is recorded as a gap", (n2, gaps2), (3, [2]))
    _check_halt(fails, "no training label at all halts",
                lambda: committed_label_window({}), "no training position")
    length, clamped = clamp_profile_length(141, 141)
    _check(fails, "the profile length is clamped to the label count minus one", length, 140)
    _check(fails, "the clamp is recorded as having bitten", clamped, True)
    length2, clamped2 = clamp_profile_length(100, 141)
    _check(fails, "a shorter committed span is not clamped", (length2, clamped2), (100, False))
    _check(fails, "the nominal pair counts of a full window",
           nominal_pair_counts(141, 3), [140, 139, 138])

    table = seed_stream_table()
    _check(fails, "every derived shuffle seed is distinct",
           len(set(table.values())), len(table))
    _check(fails, "the character-count stream at window 140", table["140|characters"], 140000)
    _check(fails, "the token-position stream at window 140", table["140|tokens"], 140100)
    _check(fails, "the first range-check bin stream", table["280|range_bin_0"], 280200)
    _check(fails, "the base seed is 0", BASE_SEED, 0)


def selftest_pattern(fails):
    print("== the repeating-pattern readout, endpoint 2 ==")
    pairs = pair_index_lists([1, 2, 3, 4], 3)
    _check(fails, "a full window's pairs at distance 1", pairs[0], ([0, 1, 2], [1, 2, 3]))
    _check(fails, "a full window's pairs at distance 3", pairs[2], ([0], [3]))
    gapped = pair_index_lists([1, 2, 4], 3)
    _check(fails, "a missing label drops only the pairs that need it", gapped[0], ([0], [1]))
    _check(fails, "distances the missing label leaves untouched keep their pairs",
           gapped[1], ([1], [2]))

    n = 40
    want = [1.0 / (1 + d) for d in range(n)]
    mat = [[want[abs(i - j)] for j in range(n)] for i in range(n)]
    prof, counts = offdiag_profile_pairs(mat, list(range(1, n + 1)), 20)
    _check(fails, "the profile at distance 1", prof[0], want[1], 1e-12)
    _check(fails, "the profile at distance 20", prof[19], want[20], 1e-12)
    _check(fails, "the pair count at distance 1", counts[0], 39)
    _check(fails, "the pair count at distance 20", counts[19], 20)

    line = [3.0 + 0.5 * i for i in range(60)]
    _check(fails, "a straight line is removed by the trend fit",
           max(abs(v) for v in poly_detrend(line, 2)), 0.0, 1e-9)
    quad = [2.0 - 0.01 * i + 0.0004 * i * i for i in range(60)]
    _check(fails, "a curved trend is removed by the trend fit",
           max(abs(v) for v in poly_detrend(quad, 2)), 0.0, 1e-9)
    _check_halt(fails, "a profile too short to detrend halts",
                lambda: poly_detrend([1.0, 2.0, 3.0], 2), "too short")

    per = whole_cycle_periods(140)
    _check(fails, "the longest searched period is the profile's own length", per[0][1], 140.0, 1e-12)
    _check(fails, "the shortest searched period is at least 4", min(p for _, p in per) >= 4.0, True)
    _check(fails, "the number of whole-cycle periods over a 140-long profile", len(per), 35)
    _check(fails, "periods come longest first", per[1][1] < per[0][1], True)

    # Gate commit case one: a planted wave of a known whole-cycle period, on a known
    # curved trend, must be returned as the location of the largest power.
    labels = list(range(1, 142))
    def entry(d):
        return 1.0 - 0.004 * d + 0.000003 * d * d + 0.05 * math.cos(2.0 * math.pi * d / 28.0)
    planted = [[entry(abs(i - j)) for j in range(141)] for i in range(141)]
    prof2, counts2 = offdiag_profile_pairs(planted, labels, 140)
    stat = max_power_statistic(prof2, whole_cycle_periods(140))
    _check(fails, "a planted 28-character wave is recovered end to end", stat["wavelength"], 28.0, 1e-9)
    _check(fails, "the planted wave is 5 whole cycles over the profile", stat["cycles"], 5)
    _check(fails, "the far tail of the profile rests on one label pair", counts2[-1], 1)
    _check(fails, "the near end of the profile rests on many label pairs", counts2[0], 140)

    # Gate commit case two: a flat profile against planted draws that all exceed it
    # must not clear its comparison.
    flat = [0.5] * 140
    flat_stat = max_power_statistic(flat, whole_cycle_periods(140))
    _check(fails, "a flat profile carries no power", flat_stat["max_power"], 0.0, 1e-18)
    planted_draws = [1.0 + 0.001 * i for i in range(N_DRAWS)]
    rank = null_rank(flat_stat["max_power"], planted_draws)
    _check(fails, "a flat profile ranks below every draw", rank, N_DRAWS + 1)
    _check(fails, "and so does not clear its comparison", decide_ripple(rank), "NULL_NOT_CLEARED")

    # A distance whose label pairs are all missing halts rather than shortening the
    # committed profile after the fact.
    short_labels = [v for v in labels if v != 1]
    _check_halt(fails, "a distance with no label pairs halts",
                lambda: offdiag_profile_pairs(planted, short_labels, 140), "zero label pairs")


def selftest_anchor(fails):
    print("== the anchor gate readout, endpoint 1 ==")
    _check(fails, "the middle of five values", median([0.70, 0.72, 0.75, 0.80, 0.90]), 0.75, 1e-12)
    _check(fails, "the middle of an even count", median([1.0, 2.0, 3.0, 4.0]), 2.5, 1e-12)
    _check(fails, "the unscaled spread of five values",
           median_absolute_deviation([0.70, 0.72, 0.75, 0.80, 0.90]), 0.05, 1e-12)
    band = anchor_band([0.70, 0.72, 0.75, 0.80, 0.90])
    _check(fails, "the band's half-width is three times that spread", band["half_width"], 0.15, 1e-12)
    _check(fails, "the band's low end", band["low"], 0.631, 1e-12)
    _check(fails, "the band's high end", band["high"], 0.931, 1e-12)
    _check(fails, "the band is centred on 0.781", band["centre"], 0.781, 1e-12)
    _check(fails, "a wide band does not raise the narrow-band flag",
           band["band_half_width_below_0.02"], False)
    tight = anchor_band([0.781, 0.782, 0.7815, 0.7805, 0.7812])
    _check(fails, "a spread under 0.02 raises the recorded narrow-band flag",
           tight["band_half_width_below_0.02"], True)
    _check(fails, "a value inside the band", decide_anchor(0.781, band["low"], band["high"]),
           "ANCHOR_IN_BAND")
    _check(fails, "a value exactly on the low end is inside",
           decide_anchor(band["low"], band["low"], band["high"]), "ANCHOR_IN_BAND")
    _check(fails, "a value exactly on the high end is inside",
           decide_anchor(band["high"], band["low"], band["high"]), "ANCHOR_IN_BAND")
    _check(fails, "a value below the band is outside",
           decide_anchor(0.60, band["low"], band["high"]), "ANCHOR_OUT_OF_BAND")
    _check(fails, "a value above the band is outside",
           decide_anchor(0.99, band["low"], band["high"]), "ANCHOR_OUT_OF_BAND")
    _check(fails, "an absent value is outside", decide_anchor(None, band["low"], band["high"]),
           "ANCHOR_OUT_OF_BAND")
    _check(fails, "a narrow measured band puts a distant value outside it",
           decide_anchor(0.700601, tight["low"], tight["high"]), "ANCHOR_OUT_OF_BAND")
    _check(fails, "that same narrow band still admits its own centre",
           decide_anchor(0.781, tight["low"], tight["high"]), "ANCHOR_IN_BAND")
    # The planted zero-spread case the gate commit requires.
    _check_halt(fails, "five identical resplit values halt as the wrong source of variation",
                lambda: anchor_band([0.7, 0.7, 0.7, 0.7, 0.7]), "WRONG_VARIANCE_SOURCE")
    _check_halt(fails, "a resplit value that is not a number halts",
                lambda: anchor_band([0.7, 0.7, float("nan"), 0.8, 0.9]), "not a number")


def selftest_endpoint_arithmetic(fails):
    print("== the shuffle comparison, endpoint 2 ==")
    draws = [float(i) for i in range(N_DRAWS)]                 # 0.0 up to 199.0
    _check(fails, "a value above every draw ranks first", null_rank(500.0, draws), 1)
    _check(fails, "a value above all but nine draws ranks tenth", null_rank(190.5, draws), 10)
    _check(fails, "a value above all but ten draws ranks eleventh", null_rank(189.5, draws), 11)
    _check(fails, "a tie counts against the observed value", null_rank(199.0, draws), 2)
    _check(fails, "rank 1 clears", decide_ripple(1), "NULL_CLEARED")
    _check(fails, "rank 10 clears", decide_ripple(10), "NULL_CLEARED")
    _check(fails, "rank 11 does not clear", decide_ripple(11), "NULL_NOT_CLEARED")

    print("== the count of clearing windows, endpoint 3 ==")
    _check(fails, "three windows clearing with one shared wavelength",
           decide_invariance({140: 60.0, 200: 62.0, 280: 64.0})[0], "INVARIANT")
    _check(fails, "two windows clearing with one shared wavelength",
           decide_invariance({140: 60.0, 200: 62.0})[0], "INVARIANT")
    _check(fails, "exactly one window clearing is undetermined",
           decide_invariance({200: 60.0})[0], "UNDETERMINED")
    _check(fails, "no window clearing", decide_invariance({})[0], "NO_WINDOW_CLEARED")
    _check(fails, "none clearing while counts above 150 are represented is a decisive negative",
           decide_headline(True, {}, "ABOVE_150_REPRESENTED"),
           "DECISIVE_NEGATIVE_NO_PATTERN_UP_TO_280")
    _check(fails, "none clearing while that range was never measurable defers to the range check",
           decide_headline(True, {}, "ABOVE_150_NOT_REPRESENTED"), "RANGE_NOT_REPRESENTED")
    _check(fails, "none clearing with an incomplete range pattern is neither",
           decide_headline(True, {}, "PATTERN_INCOMPLETE"),
           "NO_WINDOW_CLEARED_RANGE_CHECK_INCONCLUSIVE")
    _check(fails, "one clearing window reports undetermined",
           decide_headline(True, {140: 60.0}, "ABOVE_150_REPRESENTED"), "UNDETERMINED")
    _check(fails, "two clearing windows report the fork",
           decide_headline(True, {140: 60.0, 280: 61.0}, "PATTERN_INCOMPLETE"), "INVARIANT")
    _check(fails, "three clearing windows report the fork",
           decide_headline(True, {140: 35.0, 200: 50.0, 280: 70.0}, "PATTERN_INCOMPLETE"),
           "PROPORTIONAL")
    _check(fails, "a failed anchor gate overrides every other call",
           decide_headline(False, {140: 60.0, 200: 61.0}, "ABOVE_150_REPRESENTED"),
           "VOID_ANCHOR_FAILED")

    print("== the wavelength gap, endpoint 4 ==")
    _check(fails, "gaps all at or under 10 characters", largest_pairwise_gap([60.0, 62.0, 64.0]),
           4.0, 1e-12)
    _check(fails, "a gap of exactly 10 characters counts as one shared wavelength",
           decide_wavelength_gap(10.0), "ONE_SHARED_WAVELENGTH")
    _check(fails, "a gap over 10 characters does not", decide_wavelength_gap(10.5),
           "WAVELENGTHS_DIFFER")
    _check(fails, "fewer than two clearing windows leaves the gap undefined",
           largest_pairwise_gap([60.0]), None)
    _check(fails, "and that undefined gap is reported as such",
           decide_wavelength_gap(None), "GAP_UNDEFINED_FEWER_THAN_TWO_WINDOWS")
    _check(fails, "a planted set with a gap over 10 characters is not one wavelength",
           decide_invariance({140: 35.0, 200: 50.0, 280: 70.0})[0] == "INVARIANT", False)

    print("== the ratio gap, endpoint 5 ==")
    call, wl_gap, ratio_gap = decide_invariance({140: 35.0, 200: 50.0, 280: 70.0})
    _check(fails, "wavelengths that scale with the window read as proportional", call, "PROPORTIONAL")
    _check(fails, "their ratios sit within 0.1", ratio_gap, 0.0, 1e-12)
    _check(fails, "their wavelength gap is over 10 characters", wl_gap, 35.0, 1e-12)
    _check(fails, "ratios within 0.1 pass the ratio rule", decide_ratio_gap(0.1),
           "RATIOS_WITHIN_CUTOFF")
    _check(fails, "ratios wider than 0.1 do not", decide_ratio_gap(0.2), "RATIOS_DIFFER")
    mixed, mwl, mratio = decide_invariance({140: 20.0, 200: 120.0, 280: 40.0})
    _check(fails, "wide wavelength gaps and wide ratio gaps read as mixed", mixed, "MIXED")
    _check(fails, "the mixed case's wavelength gap is over 10 characters", mwl > 10.0, True)
    _check(fails, "the mixed case's ratio gap is over 0.1", mratio > 0.1, True)
    _check(fails, "exactly one clearing window is never mixed",
           decide_invariance({280: 999.0})[0], "UNDETERMINED")

    print("== the represented-range readout, endpoint 6 ==")
    _check(fails, "a bin ending at or below 140 is a low bin", bin_role(106, 140), "low")
    _check(fails, "the bin spanning 150 straddles the boundary", bin_role(141, 175), "straddles")
    _check(fails, "a bin starting above 150 is a high bin", bin_role(176, 210), "high")
    _check(fails, "the eight committed bins", [ (lo, hi) for lo, hi in BIN_EDGES ],
           [(1, 35), (36, 70), (71, 105), (106, 140), (141, 175), (176, 210), (211, 245), (246, 280)])
    _check(fails, "exactly one bin straddles the boundary",
           sum(1 for lo, hi in BIN_EDGES if bin_role(lo, hi) == "straddles"), 1)
    fifty = [0.0 + 0.01 * i for i in range(N_BIN_DRAWS)]        # 0.00 up to 0.49
    cut = percentile_linear(fifty, BIN_PERCENTILE)
    _check(fails, "the 95th percentile of fifty draws", cut, 0.4655, 1e-9)
    _check(fails, "a bin above its own 95th percentile is represented",
           decide_bin(0.60, fifty), "REPRESENTED")
    _check(fails, "a bin below its own 95th percentile is not",
           decide_bin(0.20, fifty), "NOT_REPRESENTED")
    _check(fails, "a bin exactly on its own 95th percentile does not exceed it",
           decide_bin(cut, fifty), "NOT_REPRESENTED")
    planted_bins = [
        {"bin": 0, "low": 1, "high": 35, "role": "low", "decision": "REPRESENTED"},
        {"bin": 1, "low": 36, "high": 70, "role": "low", "decision": "REPRESENTED"},
        {"bin": 2, "low": 71, "high": 105, "role": "low", "decision": "REPRESENTED"},
        {"bin": 3, "low": 106, "high": 140, "role": "low", "decision": "REPRESENTED"},
        {"bin": 4, "low": 141, "high": 175, "role": "straddles", "decision": "REPRESENTED"},
        {"bin": 5, "low": 176, "high": 210, "role": "high", "decision": "NOT_REPRESENTED"},
        {"bin": 6, "low": 211, "high": 245, "role": "high", "decision": "NOT_REPRESENTED"},
        {"bin": 7, "low": 246, "high": 280, "role": "high", "decision": "NOT_REPRESENTED"},
    ]
    _check(fails, "the escape pattern holds even when the straddling bin is represented",
           decide_range_pattern(planted_bins), "ABOVE_150_NOT_REPRESENTED")
    with_high = [dict(b) for b in planted_bins]
    with_high[6]["decision"] = "REPRESENTED"
    _check(fails, "one represented bin above 150 makes a null result a genuine negative",
           decide_range_pattern(with_high), "ABOVE_150_REPRESENTED")
    low_missing = [dict(b) for b in planted_bins]
    low_missing[1]["decision"] = "NOT_REPRESENTED"
    _check(fails, "a low bin that is not represented leaves the pattern incomplete",
           decide_range_pattern(low_missing), "PATTERN_INCOMPLETE")
    _check(fails, "a bin's fit quality against its own labels, planted perfect fit",
           r2_from_pairs([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), 1.0, 1e-12)
    _check(fails, "a bin's fit quality, planted known value",
           r2_from_pairs([1.0, 2.0, 3.0], [1.5, 2.0, 2.5]), 0.75, 1e-12)
    _check(fails, "a bin whose predictions are its own mean scores zero",
           r2_from_pairs([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]), 0.0, 1e-12)
    _check(fails, "labels above 280 fall in no committed bin",
           any(lo <= 281 <= hi for lo, hi in BIN_EDGES), False)

    print("== the token-position control, endpoint 7 ==")
    _check(fails, "a converted gap over 3 characters credits counting characters",
           decide_token_credit(60.0, 31.0, True, True), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "a gap under 3 characters with both clearing credits token position",
           decide_token_credit(60.0, 57.5, True, True), "CREDITED_TO_TOKEN_POSITION")
    _check(fails, "a gap of exactly 3 characters with both clearing credits token position",
           decide_token_credit(60.0, 57.0, True, True), "CREDITED_TO_TOKEN_POSITION")
    _check(fails, "a gap just over 3 characters credits counting characters",
           decide_token_credit(60.0, 56.9, True, True), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "disagreement on clearing credits characters whatever the gap",
           decide_token_credit(60.0, 60.0, True, False), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "disagreement the other way also credits characters",
           decide_token_credit(60.0, 60.0, False, True), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "neither labelling clearing leaves nothing to credit",
           decide_token_credit(60.0, 59.0, False, False), "NO_PATTERN_TO_CREDIT")
    _check(fails, "an uncomputable control is recorded as such",
           decide_token_credit(60.0, None, True, None), "TOKEN_CONTROL_NOT_COMPUTABLE")
    # The conversion itself, at a planted rate of characters per token.
    rate = 4.25
    _check(fails, "a 14-token period converts to characters at the measured rate",
           14.0 * rate, 59.5, 1e-12)
    _check(fails, "and that converted period is within 3 characters of a 60-character one",
           decide_token_credit(60.0, 14.0 * rate, True, True), "CREDITED_TO_TOKEN_POSITION")


def selftest_void_propagation(fails):
    """The planted gate failure the gate commit requires: it must reach every record."""
    print("== a failed gate reaches every record written after it, endpoint 1 ==")
    import tempfile
    try:
        prov = load_provenance()
    except Halt as exc:
        fails.append("record writer import: %s" % exc)
        print("  %-64s FAIL %s" % ("record writer importable", exc))
        return
    saved_state = dict(RUN_STATE)
    saved_anchor = dict(ANCHOR_CONTEXT)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        RUN_STATE["run_valid"] = True
        RUN_STATE["void_reason"] = None
        ANCHOR_CONTEXT["decision"] = "NOT_YET_COMPUTED"
        before = write_record(prov, out, base_cell("endpoints", "planted_before_gate"),
                              inputs=[], metric="planted", values={"planted": 1},
                              decision_rule="planted rule", decision="RECORDED_NOT_GATED")
        band = anchor_band([0.70, 0.72, 0.75, 0.80, 0.90])
        planted_value = 0.20
        call = decide_anchor(planted_value, band["low"], band["high"])
        _check(fails, "the planted anchor value fails its band", call, "ANCHOR_OUT_OF_BAND")
        ANCHOR_CONTEXT["decision"] = call
        ANCHOR_CONTEXT["statistic"] = planted_value
        ANCHOR_CONTEXT["band_low"] = band["low"]
        ANCHOR_CONTEXT["band_high"] = band["high"]
        gate_rec = write_record(prov, out, base_cell("endpoints", "planted_gate", endpoint=1),
                                inputs=[], metric="planted anchor",
                                values={"held_out_quality": planted_value},
                                decision_rule=RULE_ANCHOR, decision=call)
        void_the_run("planted gate failure", out)
        after = write_record(prov, out, base_cell("endpoints", "planted_pattern", endpoint=2,
                                                  aperture=280, labelling="characters"),
                             inputs=[], metric="planted per-window measurement",
                             values={"max_power": 1.0}, decision_rule=RULE_RIPPLE,
                             decision="NULL_CLEARED")
        summary = write_record(prov, out, base_cell("endpoints", "planted_summary"),
                               inputs=[], metric="planted roll-up", values={"headline": "x"},
                               decision_rule=RULE_SUMMARY, decision="VOID_ANCHOR_FAILED")
        b = json.loads(before.read_text())
        g = json.loads(gate_rec.read_text())
        a = json.loads(after.read_text())
        s = json.loads(summary.read_text())
        _check(fails, "a record written before the gate is still valid",
               b["values"]["run_valid"], True)
        _check(fails, "the gate's own record carries its failing call",
               g["decision"], "ANCHOR_OUT_OF_BAND")
        _check(fails, "the per-probe-window record written after the gate is void",
               a["values"]["run_valid"], False)
        _check(fails, "that per-window record names the reason inside its own values",
               a["values"]["run_void_reason"], "planted gate failure")
        _check(fails, "that per-window record carries the gate's call in its own values",
               a["values"]["anchor_gate_decision"], "ANCHOR_OUT_OF_BAND")
        _check(fails, "that per-window record carries the measured band in its own values",
               (a["values"]["anchor_band_low"], a["values"]["anchor_band_high"]),
               (band["low"], band["high"]))
        _check(fails, "the roll-up is void as well", s["values"]["run_valid"], False)
        _check(fails, "the void is not only in the roll-up",
               a["values"]["run_valid"] == s["values"]["run_valid"], True)
        _check(fails, "the record's file name and its identity agree",
               Path(a["cell"] and prov.cell_slug(a["cell"]) + ".json").name, after.name)
    RUN_STATE.update(saved_state)
    ANCHOR_CONTEXT.update(saved_anchor)


def selftest_supporting(fails):
    print("== supporting checks ==")
    try:
        prov = load_provenance()
        print("  %-64s ok" % "the record writer is importable")
        for name in ("write_result", "describe_input", "cell_slug"):
            if not hasattr(prov, name):
                fails.append("record writer lacks " + name)
                print("  %-64s FAIL" % ("the record writer has " + name))
    except Halt as exc:
        fails.append("record writer import: %s" % exc)
        print("  %-64s FAIL %s" % ("the record writer is importable", exc))
    try:
        import numpy as np
    except ImportError:
        print("  %-64s skipped (numpy absent)" % "the ridge fit from running sums")
        return
    import random as _r
    _r.seed(7)
    rows = [[_r.uniform(-1, 1), _r.uniform(-1, 1)] for _ in range(50)]
    ys = [2.0 * a - 3.0 * b + 0.5 + _r.uniform(-0.01, 0.01) for a, b in rows]
    xm = np.array(rows)
    ym = np.array(ys)
    sums = (xm.T @ xm, xm.sum(0), xm.T @ ym, float(ym.sum()), float((ym * ym).sum()), len(ys))
    w, off = ridge_fit_from_sums(sums[0], sums[1], sums[2], sums[3], sums[5], 1e-6)
    r2, sse, sstot = r2_from_sums(w, off, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5])
    pred = xm @ w + off
    direct_sse = float(((pred - ym) ** 2).sum())
    direct_r2 = 1.0 - direct_sse / float(((ym - ym.mean()) ** 2).sum())
    _check(fails, "the sum of squared errors from running sums", sse, direct_sse, 1e-6)
    _check(fails, "fit quality from running sums", r2, direct_r2, 1e-9)
    _check(fails, "the planted weights are recovered",
           float(abs(w[0] - 2.0) + abs(w[1] + 3.0)), 0.0, 0.02)
    _check(fails, "the winning strength at an edge of the grid is flagged",
           strength_is_interior(100000.0, RIDGE_GRID), False)
    _check(fails, "a strength inside the grid is flagged interior",
           strength_is_interior(1000.0, RIDGE_GRID), True)
    # The numpy path that runs and the pure-Python path the self-test checks must agree.
    n = 30
    mat = np.array([[1.0 / (1.0 + abs(i - j)) for j in range(n)] for i in range(n)])
    labels = list(range(1, n + 1))
    pairs = pair_index_lists(labels, 10)
    fast = [float(mat[np.array(r), np.array(c)].mean()) for r, c in pairs]
    slow, _ = offdiag_profile_pairs(mat.tolist(), labels, 10)
    _check(fails, "the fast profile and the plain one agree",
           max(abs(a - b) for a, b in zip(fast, slow)), 0.0, 1e-12)
    # The average-vector helper, against a planted grouping.
    x = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    slots = np.array([0, 0, 2])
    means, counts = means_by_label(x, slots, 3)
    _check(fails, "the average vector of a planted group", list(means[0]), [2.0, 0.0])
    _check(fails, "an empty slot has no positions", int(counts[1]), 0)
    _check(fails, "the planted position counts", [int(c) for c in counts], [2, 0, 1])

    # The whole held-out readout on a planted held-out set: four documents, labels 1 to
    # 41 with label 7 carried by none of them, and a wave of period 10 built into the
    # per-label directions. The readout must recover that period, must rank the observed
    # value above its shuffled draws, and must report the label it never saw.
    n_labels = 41
    planted_rows, planted_labels, planted_docs = [], [], []
    for d in range(4):
        for lab in range(1, n_labels + 1):
            if lab == 7:
                continue
            for _ in range(2):
                ang = 2.0 * math.pi * lab / 10.0
                planted_rows.append([math.cos(ang), math.sin(ang), 0.01 * lab])
                planted_labels.append(lab)
                planted_docs.append(d)
    res = pattern_with_null(np.array(planted_rows, dtype=np.float64),
                            np.array(planted_labels, dtype=np.int64),
                            np.array(planted_docs, dtype=np.int64),
                            1, n_labels, 40, seed=BASE_SEED, n_draws=5)
    _check(fails, "the held-out readout recovers a planted period of 10 characters",
           res["observed"]["wavelength"], 10.0, 1e-9)
    _check(fails, "that period is 4 whole cycles over the profile", res["observed"]["cycles"], 4)
    _check(fails, "the planted pattern ranks above every shuffled draw", res["rank"], 1)
    _check(fails, "a label the held-out set never carries is reported",
           res["labels_missing_from_held_out"], [7])
    _check(fails, "the profile is as long as the committed length", len(res["profile"]), 40)
    _check(fails, "the missing label costs the nearest distance one pair", res["pair_counts"][0], 38)
    _check(fails, "the far tail still rests on one pair", res["pair_counts"][-1], 1)
    _check(fails, "the average vectors saved cover every label of the window",
           res["mean_vectors"].shape[0], n_labels)
    _check(fails, "and the rows kept are the labels the held-out set carries",
           len(res["present_rows"]), n_labels - 1)


def selftest() -> int:
    fails = []
    selftest_parsers(fails)
    selftest_detectors(fails)
    selftest_windows(fails)
    selftest_pattern(fails)
    selftest_anchor(fails)
    selftest_endpoint_arithmetic(fails)
    selftest_void_propagation(fails)
    selftest_supporting(fails)
    print("selftest:", "PASS" if not fails else "%d FAILURES" % len(fails))
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Part 12. Entry point.
# ---------------------------------------------------------------------------

STAGES = {"extract": stage_extract, "calibrate": stage_calibrate, "endpoints": stage_endpoints}
STAGE_ORDER = ["extract", "calibrate", "endpoints"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=STAGE_ORDER + ["all", "generate"],
                    help="which stage to run")
    ap.add_argument("--selftest", action="store_true",
                    help="standard-library checks only; no model and no graphics card")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", RESULTS_DIR_DEFAULT))
    ap.add_argument("--stimulus", default=None, help="path to the committed text file")
    ap.add_argument("--design-record", default=None, help="path to the committed design record")
    ap.add_argument("--gate-commit", default=None, help="path to the committed gate commit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.stage:
        ap.error("give --stage extract, calibrate, endpoints or all; or give --selftest")
    if args.stage == "generate":
        print("There is no generation stage. The design says no text is generated anywhere: "
              "every measurement reads hidden activations from one deterministic forward pass "
              "per document per wrap width. The stage that sits in this position is the "
              "calibration stage, which fixes every remaining constant on training documents "
              "alone and writes that record before any held-out number exists. Run "
              "--stage calibrate.", file=sys.stderr)
        return 2

    order = STAGE_ORDER if args.stage == "all" else [args.stage]
    for name in order:
        print("=== stage %s ===" % name)
        sys.stdout.flush()
        started = time.time()
        try:
            rc = STAGES[name](args)
        except Halt as exc:
            print("HALT in stage %s: %s" % (name, exc), file=sys.stderr)
            return 3
        if rc:
            return rc
        print("=== stage %s done in %.1f s ===" % (name, time.time() - started))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
