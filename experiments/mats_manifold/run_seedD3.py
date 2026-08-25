#!/usr/bin/env python3
"""Measures a character-count curve and a repeating pattern in a 1.4-billion-weight model.

WHAT THIS RUNS. One public checkpoint of one open language model reads fixed, committed
text. The text is prose hard-wrapped at nine line widths. At every token position we know
how many characters have passed since the last line break. A ridge regression is a linear
fit with a penalty on large weights. One such fit predicts that count from the model's
internal vector at that position. Fit quality is the share of the count's spread the fit
explains on documents the fit never saw. We report it once per readout. At the
best-scoring readout we then look for a faint repeating pattern in the similarity
structure of the per-count average vectors. A label-shuffling comparison says whether that
pattern beats chance.

THE QUESTION, one model size above the previous rung. The previous rung ran the same
instrument twice on a 410-million-weight model of the same family. It found a strong count
curve, a count code spread over only 1 to 3 directions, and no repeating pattern
separable from chance. The published pattern this program chases lived at about 6
directions in a much larger model. So the deciding question here is whether the
precondition appears one size up: does the curve exist at all, does the count code reach 4
or more directions, and does any probe window beat its own shuffles?

THREE THINGS THIS RUN DECIDES.
  1. A floor for existence. No outside measurement of this model exists, so the gate is
     not a reproduction band. Fit quality pooled over the seven wrap widths 20 through 140
     must reach 0.60 at some readout. Below that the curve is too weak to read (recorded
     as CURVE_TOO_WEAK) and every pattern result of this run is void.
  2. The number of directions holding 95 percent of the spread in the per-count average
     vectors, at the chosen readout, at each of the three probe windows. Reaching 4 at any
     one window is the pre-registered threshold. The same count at 90 and at 99 percent is
     recorded beside it and decides nothing.
  3. Whether the largest repeating power beats 200 label shuffles at any window, under
     either labelling.

THE FOUR PRE-REGISTERED READINGS.
  CURVE_TOO_WEAK                  the floor failed; the rung is void and the next step is
                                  a human decision.
  PRECONDITION_ABSENT_AT_1P4B     floor passed, no window cleared, directions stayed
                                  below 4 everywhere. The expected continuation of the
                                  410-million-weight result.
  PRECONDITION_WITHOUT_RIPPLE     floor passed, directions reached 4 somewhere, still no
                                  window cleared. A strike against the account that
                                  blames low capacity for the missing pattern.
  the invariance fork             a character-count window cleared, so the previous
                                  rung's fork decides: INVARIANT, PROPORTIONAL, MIXED or
                                  UNDETERMINED.
A clearing under the token-position control alone is recorded as TOKEN_ONLY_PATTERN. It is
a real finding about positional structure and it does NOT trigger the fork, which is
defined over character-count clearings only.

THE COMPLETE SPECIFICATION is the design object inside
records/seedD3_design_diff_triage.json, committed in records/GATE_COMMIT_seedD3.md. This
file implements it and changes nothing. Every constant that design marks as a guess is
binding here, and so are the five committed answers in the gate record.

WHAT DIFFERS FROM THE PREVIOUS RUNNER run_seedD2_1.py. Its arithmetic is proven and is
reused unchanged wherever the design matches. Six things differ, each named by the
committed design or its gate record:
  1. The model is EleutherAI/pythia-1.4b-deduped at revision step143000. Its expected
     shape is 24 transformer layers and hidden size 2048. The downloaded configuration is
     authoritative and a mismatch halts.
  2. The validity gate is a floor for existence at 0.60, not a two-sided reproduction
     band. Failing it voids every pattern result and reads CURVE_TOO_WEAK.
  3. When neither labelling clears at a window, the reading is NO_PATTERN_TO_ATTRIBUTE.
     The previous run recorded a "distinguished" verdict over two absent patterns, which
     this correction removes.
  4. The token-count window's ceiling rule is committed, not a runner constant: a token
     count enters the window only with at least 40 training positions. The ceiling is
     recorded next to the observed maximum, so the cut is visible.
  5. Within-bin correlations across characters 1 to 140 are recorded at the chosen readout
     and the 140-character window. They carry no threshold and decide nothing.
  6. The number of directions is recorded at 90 and 99 percent as well as at the gated 95
     percent. Only the 95 percent count feeds the pre-registered reading.

MEMORY. Hidden size 2048 makes every running total four times the size of the previous
rung's. The 14 groups of totals hold about 11.8 gigabytes together, which is why they stay
running totals and are never expanded into per-position vectors. Per-position vectors are
saved for the three probe windows on held-out documents only, about 6.4 gigabytes, and for
nothing else. Both numbers are printed by the first stage before it starts.

NO TEXT IS GENERATED ANYWHERE. The design says so in one line: "No text generation or
sampling occurs: hidden states come from one deterministic forward pass per document over
the 48 fixed stimulus documents". So this file has no sampling settings, no decoding, and
no answer parsing. The build brief that produced this file asks for a generation stage, a
reasoning-tag parser, and an answer extractor. Those three things cannot exist for this
experiment. That brief is a fixed template shared with a different, text-generating
project (pipeline.js line 104). The three closest real equivalents are implemented and
self-tested instead, and each is named where it is defined below. The same choice was made,
for the same reason, by run_seedD.py, run_seedD2.py and run_seedD2_1.py.

STAGES, in order, each runnable on its own:
  --stage extract    checks the text file and the download, then runs one forward pass per
                     text variant, and saves what the later stages need.
  --stage calibrate  fixes every remaining constant using training documents only, and
                     writes that record before any held-out number is computed.
  --stage generate   the same stage as calibrate. The build brief names this position
                     "generate"; nothing is generated, so the name is only an alias.
  --stage endpoints  computes the per-readout fit, the floor gate, the repeating pattern at
                     each of the three probe windows, the shuffle comparisons, the
                     token-position control, and both bin-correlation checks.
  --stage all        extract, then calibrate, then endpoints.
  --selftest         standard library only. Loads no model, needs no graphics card, and
                     exits non-zero on any failure.

Only one model exists in this experiment, so only one is ever resident. It is loaded once,
in the first stage, in half precision, and released before any analysis stage.

Every result stage writes one self-describing record through provenance.py. A record
carries its own identity, its inputs with content hashes, its measurement, the rule that
judges it, and the resulting call. Every rule text here is written before any value is
known. Every per-window pattern record and every control record also carries the floor
gate's own decision and a voided-by-gate flag, so no such record can be read without its
void status.

A second run of this script on the same rented box is a rerun of one instrument. It is not
a replication. No output here calls it one. This rung is a single run, unlike the
twice-measured rung below it.
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
#   DESIGN  the committed design object.
#   GATE    a committed answer in records/GATE_COMMIT_seedD3.md.
#   MEMORY  recalled from a paper, and checked against the download.
#   GUESS   fixed by this file because the design leaves it to implementation.
# ---------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"          # DESIGN
MODEL_SLUG = "pythia-1.4b-deduped"                   # the same name, safe in a filename
REVISION = "step143000"                              # DESIGN, final pretraining checkpoint
EXPECT_LAYERS = 24                                   # MEMORY, Pythia paper arXiv 2304.01373 table 1
EXPECT_HIDDEN = 2048                                 # MEMORY, same table
STIMULUS_SHA256 = "b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd"   # DESIGN
DESIGN_SHA256 = "60ef29b05769e0b0cdb222faadb806c1f53d52522a7568ae40c7637475db60e6"     # this design record, hashed on the workstation 2026-08-21
GATE_SHA256 = "bc6b69231f64968348dc7fd2435a2444233af36b526794e26ae53dcaffbb8e86"       # the gate record, same hashing

WIDTHS = [20, 40, 60, 80, 100, 120, 140, 200, 280]   # DESIGN, all nine wrap widths
SELECT_WIDTHS = [20, 40, 60, 80, 100, 120, 140]      # DESIGN, the seven widths that train and gate
WIDE_WIDTHS = [200, 280]                             # DESIGN, scored at every readout, never gated
APERTURES = [140, 200, 280]                          # DESIGN, the three probe windows
N_DOCS = 48                                          # DESIGN
N_TRAIN_DOCS = 36                                    # DESIGN, documents 0 to 35
N_TEST_DOCS = 12                                     # DESIGN, documents 36 to 47

RIDGE_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]   # DESIGN
N_FOLDS = 5                                          # DESIGN, five-fold cross-validation
SEED = 0                                             # DESIGN and GATE answer 4, the one base seed
N_DRAWS = 200                                        # DESIGN, shuffle draws per window per labelling
NULL_RANK_MAX = 10                                   # DESIGN, top 5 percent of 200 draws

FLOOR_R2 = 0.60                                      # DESIGN GUESS, the existence floor
DIMENSION_THRESHOLD = 4                              # DESIGN GUESS, directions that count as reaching
VARIANCE_TARGET = 0.95                               # GATE answer 1, the gated convention
VARIANCE_TARGETS_RECORDED = [0.90, 0.99]             # GATE answer 1, recorded, never gated

# Numbers from measurements already on the record, carried only as context inside records.
PRIOR_410M_POOLED = 0.781031                         # the second 410-million-weight run
PRIOR_410M_POOLED_FIRST_RUN = 0.780903               # the first, on other text
PRIOR_410M_READOUT = 6
PRIOR_410M_DIMENSIONS = {140: 2, 200: 3, 280: 1}     # directions at 95 percent, by window
PRIOR_REPRODUCTION_PYTHIA_160M = 0.744               # the floor's own justification
PRIOR_REPRODUCTION_GPT2 = 0.363                      # the floor's own justification
PRIOR_PUBLISHED_DIMENSIONS = 6                       # the published pattern's own dimension

INVARIANCE_MAX_GAP_CHARS = 10.0                      # DESIGN GUESS, fixed-period rule
PROPORTION_MAX_RATIO_GAP = 0.1                       # DESIGN GUESS, fixed-proportion rule
CONFOUND_MARGIN_CHARS = 3.0                          # DESIGN GUESS, token-position control margin

BIN_WIDTH = 35                                       # DESIGN, width of a label bin
N_BINS = 8                                           # DESIGN, eight bins covering labels 1 to 280
N_BINS_140 = 4                                       # DESIGN correction 4, four bins over 1 to 140
RANGE_MEASURABLE_FLOOR = 0.5                         # DESIGN GUESS, a high bin at or above this means measurable
RANGE_LOW_FLOOR = 0.5                                # DESIGN GUESS, every low bin must reach this
RANGE_HIGH_CEIL = 0.2                                # DESIGN GUESS, every high bin must stay under this
LOW_BIN_MAX_LABEL = 140                              # DESIGN, a bin is low if it ends at or below this
HIGH_BIN_MIN_LABEL = 175                             # DESIGN, a bin is high if it starts above this

PERIOD_MIN = 4                                       # DESIGN, shortest period the statistic considers
DETREND_DEGREE = 2                                   # DESIGN, the best-fitting degree-2 polynomial
PROFILE_MARGIN = 20                                  # DESIGN, profile length is the label count minus 20
COMMITTED_PROFILE_LENGTH = {140: 120, 200: 180, 280: 260}     # DESIGN, the same rule as three numbers
MIN_TRAIN_POSITIONS_PER_LABEL = 20                   # GUESS, a character label needs 20 training positions
MAX_WINDOW_START = 8                                 # GUESS, how far above 1 a label window may start
TOKEN_WINDOW_MIN_TRAIN = 40                          # DESIGN correction 3, the committed token ceiling rule
REPEAT_BLOCK_MAX = 8                                 # GUESS, longest repeating block the degenerate check looks for
REPEAT_MIN_RUNS = 6                                  # GUESS, that many back-to-back repeats counts as degenerate
CONTEXT_LIMIT_FALLBACK = 2048                        # MEMORY, Pythia context length; the download overrides it

LABELLING_CODES = {"characters": 0, "tokens": 1}     # GATE answer 4, the two shuffle streams

RESULTS_DIR_DEFAULT = "results_seedD3"

RERUN_NOTE = ("A further run of this script on the same box is a rerun of the same "
              "instrument, not a replication.")
VOID_NOTE = ("If the floor gate recorded in these values failed, this record is void as "
             "evidence about the model, and is an instrument fault only.")
RECORDED_ONLY_NOTE = ("This check is recorded only. It carries no threshold and decides "
                      "nothing, as the gate record's answer 5 settles.")


class Halt(Exception):
    """Raised where the design says halt. Never caught to keep a run going."""


# ---------------------------------------------------------------------------
# Part 1. The position labeller.
#
# This is the real equivalent of the reasoning-tag parser the build template asks
# for. There, an opening marker can be missing because a chat template already
# emitted it, so only the closing marker appears in the text. The same shape occurs
# here twice, and the self-test covers both:
#   - the first line of a document has no line break in front of it, so its opening
#     boundary is implied rather than present in the text;
#   - one token can swallow a line break, so the boundary sits inside a unit rather
#     than between two units.
# Getting either wrong shifts every label on a line and silently moves the whole
# measurement. So this code is small, pure, and tested.
# ---------------------------------------------------------------------------

def line_columns(text: str):
    """For each character, its column on its line, and its line number.

    A character's column counts the characters on its line up to and including itself.
    It is 1 for the first character of a line. A line-break character belongs to the
    line it ends, and takes the next column after the last visible character. Counting
    restarts at zero after each line break. This is definition 1 of the design.
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
    """Labels for token positions, from character ranges over the same text.

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
# This is the real equivalent of the degenerate-generation detector the template asks
# for. Its two cases map onto the only two that can occur here:
#   - the length-cap case: a text variant whose token count reaches the model's
#     context limit, so the committed text would be silently cut short;
#   - the repeated-sequence case: a text variant holding the same short block of
#     tokens many times over, which would flood one label with copies of one vector.
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
    """Names of the degenerate conditions present. An empty list means clean."""
    flags = []
    if n_tokens >= context_limit:
        flags.append("token_cap")
    if max_block_repeats(list(token_texts)) >= REPEAT_MIN_RUNS:
        flags.append("repeated_sequence")
    return flags


# ---------------------------------------------------------------------------
# Part 3. Exact-match checks and exact slot assignment.
#
# This is the real equivalent of the exact-match answer extraction the template asks
# for. Nothing here is fuzzy, normalised, or rounded. Three things must match exactly:
# the text file's content hash, the downloaded configuration's layer count and width,
# and the whole-number label of a position against a slot of the similarity matrix. An
# off-by-one in the last one would move every row of that matrix with no error anywhere.
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
    """Slot index of a label inside a window, or None when it falls outside.

    Membership is exact whole-number equality against one of the window's values. A
    fractional value, a true-or-false value, and an absent value are never accepted.
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


def window_from_counts(counts_by_label, top: int, min_count: int,
                       max_start: int = MAX_WINDOW_START):
    """Lowest window start whose labels up to `top` all reach min_count.

    The lowest possible label is 1, because a label counts the characters up to and
    including a token's last character. The lowest column of a line is often thin,
    since it needs a one-character opening token. So the start may sit a little above
    1, and how far above is recorded. The top of the window is the probe width itself,
    which is what makes the window comparison a comparison of probe windows.
    """
    for lo in range(1, max_start + 2):
        if all(counts_by_label.get(v, 0) >= min_count for v in range(lo, top + 1)):
            return lo
    return None


def token_window_top(counts_by_label, min_count: int = TOKEN_WINDOW_MIN_TRAIN) -> int:
    """Highest token count whose values from 1 upward all reach min_count.

    This is the committed ceiling rule of correction 3. A token count occurs at most
    once per line, so the highest counts are reached by only a handful of the longest
    lines. The minimum here sits well above the character-label minimum for that
    reason. The held-out documents are a third as many, and a token count carried by
    only a few training lines can be missing from them altogether, which would leave
    its average vector undefined. Counts above the ceiling are excluded, and the
    ceiling is recorded next to the observed maximum so the cut is visible.
    """
    top = 0
    while counts_by_label.get(top + 1, 0) >= min_count:
        top += 1
    return top


def token_ceiling_report(counts_by_label, min_count: int = TOKEN_WINDOW_MIN_TRAIN):
    """The token ceiling, the observed maximum, and the size of the cut between them."""
    ceiling = token_window_top(counts_by_label, min_count)
    observed_max = max(counts_by_label) if counts_by_label else 0
    return {
        "token_ceiling": ceiling,
        "token_label_observed_max": observed_max,
        "token_labels_excluded_above_ceiling": max(0, observed_max - ceiling),
        "token_ceiling_min_training_positions": min_count,
        "token_positions_at_ceiling": counts_by_label.get(ceiling, 0),
        "token_positions_one_above_ceiling": counts_by_label.get(ceiling + 1, 0),
    }


def label_bins(bin_width: int = BIN_WIDTH, n_bins: int = N_BINS):
    """The eight label bins of the width-280 range check, as (low, high) pairs."""
    return [(1 + bin_width * i, bin_width * (i + 1)) for i in range(n_bins)]


def label_bins_140(bin_width: int = BIN_WIDTH, n_bins: int = N_BINS_140):
    """The four label bins of the recorded-only check across characters 1 to 140."""
    return [(1 + bin_width * i, bin_width * (i + 1)) for i in range(n_bins)]


def bin_role(lo: int, hi: int) -> str:
    """Whether a bin is a low bin, a high bin, or straddles the boundary.

    The design settles the bin from 141 to 175: it is recorded and left out of the
    decision. Four bins end at or below label 140. Three bins start above label 175.
    """
    if hi <= LOW_BIN_MAX_LABEL:
        return "low"
    if lo > HIGH_BIN_MIN_LABEL:
        return "high"
    return "straddles"


def profile_length_for(n_labels: int) -> int:
    """The committed profile length: the label count minus 20."""
    return n_labels - PROFILE_MARGIN


# ---------------------------------------------------------------------------
# Part 4. The repeating-pattern statistic. Pure Python, so the self-test needs no
# extra packages. There is no search band anywhere in this design. The statistic is
# the largest power over every whole number of cycles whose period is at least 4 and
# at most the profile's own length.
# ---------------------------------------------------------------------------

def offdiag_profile(matrix, max_delta: int):
    """Mean of the entries a given distance above the diagonal, per distance.

    `matrix` is a square similarity matrix as a list of rows. The distance runs from 1
    to max_delta. This is the profile the design calls the off-diagonal profile.
    """
    n = len(matrix)
    if max_delta < 1 or max_delta >= n:
        raise Halt("distance limit %r is out of range for a %d by %d matrix"
                   % (max_delta, n, n))
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
            raise Halt("the detrend fit has no unique solution")
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def poly_detrend(y, degree: int = DETREND_DEGREE):
    """Subtract the best-fitting polynomial in distance, of the given degree.

    The design fixes this at degree 2 and rejects a moving average. Subtracting a
    moving average of some window removes every period longer than about that window,
    which is where this study predicts the pattern lies. A degree-2 polynomial removes
    the profile's slow decay and its curvature, and leaves every period the statistic
    ranges over.
    """
    n = len(y)
    if n < degree + 2:
        raise Halt("a profile of length %d is too short to detrend at degree %d"
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

    Returns a dictionary from cycle count to power. The zero-cycle term is dropped,
    because the detrend has already removed the profile's mean and slope. The period of
    a cycle count is the profile length divided by that count.
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


def periods_in_range(length: int, min_period: int = PERIOD_MIN):
    """Cycle counts whose period is at least min_period, longest period first.

    The longest is the profile length itself, at one cycle. The design names this range
    explicitly. It is applied in whatever unit the labels use, so the token control
    searches token periods and the character arm searches character periods.
    """
    out = []
    for k in range(1, length // 2 + 1):
        period = length / k
        if period >= min_period - 1e-9:
            out.append((k, period))
    return out


def max_power_stat(matrix, max_delta: int, degree: int = DETREND_DEGREE,
                   min_period: int = PERIOD_MIN, keep_profile: bool = False):
    """Largest power and its period, from one similarity matrix.

    Returns a dictionary. "max_power" is the largest power at any period from
    min_period up to the profile length. "wavelength" is that period, in the units of
    the matrix's own labels. When two periods tie on power, the longer one wins, so the
    answer never depends on ordering. The whole spectrum is returned every time, which
    is what the design requires to be saved for each shuffle draw.
    """
    prof = offdiag_profile(matrix, max_delta)
    det = poly_detrend(prof, degree)
    power = dft_power(det)
    usable = periods_in_range(len(det), min_period)
    if not usable:
        raise Halt("a profile of length %d carries no period of at least %d"
                   % (len(det), min_period))
    best_k, best_period = usable[0]
    best_power = power[best_k]
    for k, period in usable[1:]:
        if power[k] > best_power:
            best_k, best_period, best_power = k, period, power[k]
    spectrum = [{"cycles": k, "period": len(det) / k, "power": power[k],
                 "in_range": (len(det) / k) >= min_period - 1e-9}
                for k in sorted(power)]
    out = {
        "max_power": best_power,
        "wavelength": best_period,
        "cycles": best_k,
        "n_periods_in_range": len(usable),
        "profile_length": len(det),
        "spectrum": spectrum,
    }
    if keep_profile:
        out["profile"] = prof
        out["detrended"] = det
    return out


# ---------------------------------------------------------------------------
# Part 5. Endpoint arithmetic. Every function here is a pure function of numbers, so
# the self-test can plant values whose correct call is known in advance.
# ---------------------------------------------------------------------------

RULE_SETUP = (
    "The text file's hash must equal the committed hash exactly, the layer count must "
    "equal 24, and the hidden size must equal 2048. Anything else is SETUP_FAILURE and "
    "halts the run before any measurement."
)
RULE_EXTRACT = (
    "All 48 documents at all 9 wrap widths must produce one forward pass each, 432 in "
    "total, with no variant flagged as reaching the context limit. Anything else is "
    "EXTRACT_INCOMPLETE and halts."
)
RULE_CALIBRATION = (
    "The profile length at each probe window must equal that window's label count minus "
    "20, and must equal the number the design states for that window, which is 120, 180 "
    "or 260. Each window must also yield a label window whose profile can be detrended "
    "at degree 2 and can carry a period of 4 or more, under both labellings. A token "
    "count enters its window only with at least 40 training positions. Anything else is "
    "CALIBRATION_UNUSABLE and halts before any held-out number is computed."
)
RULE_FLOOR = (
    "Held-out fit quality pooled over the seven wrap widths 20 through 140, on fresh "
    "held-out documents, must reach 0.60 at some readout. Below that at every readout is "
    "CURVE_TOO_WEAK: the count curve does not exist well enough to read, every pattern "
    "result of this run is void, and the next rung is a human decision. This floor is an "
    "existence test, not a reproduction band, because no outside measurement of this "
    "model exists. The readout is chosen by that same pooled number, and its location is "
    "recorded and never judged."
)
RULE_DIMENSION = (
    "The number of directions holding 95 percent of the spread in the per-count average "
    "vectors, at the chosen readout, must reach 4 at one or more of the probe windows "
    "140, 200 and 280. Staying at 3 or fewer at every window is DIMENSION_BELOW_4, which "
    "extends the 410-million-weight result. The counts at 90 and at 99 percent are "
    "recorded beside this and carry no threshold."
)
RULE_CLEARED = (
    "The rank of the observed largest power among 200 label-shuffled draws, counting the "
    "observed value itself, must be 10 or better. That is the top 5 percent. Worse is "
    "NULL_NOT_CLEARED, meaning no separable repeating pattern at that probe window."
)
RULE_ANY_CLEARED = (
    "Rank 10 or better out of 200 at any of the three probe windows, under either "
    "labelling, is ANY_WINDOW_CLEARED. No window clearing under either labelling is "
    "NO_WINDOW_CLEARED."
)
RULE_ATTRIBUTION = (
    "At a probe window: if the token-position control cannot be computed the call is "
    "TOKEN_CONTROL_NOT_COMPUTABLE. If neither labelling clears its own shuffle draws the "
    "call is NO_PATTERN_TO_ATTRIBUTE, because no 'distinguished' verdict may be drawn "
    "over two absent patterns. If only the token labelling clears, the call is "
    "TOKEN_ONLY_PATTERN. If only the character labelling clears, the character count is "
    "credited. If both clear, the character count is credited when the two periods differ "
    "by more than 3 characters, and otherwise the pattern is "
    "CONFOUNDED_WITH_TOKEN_POSITION."
)
RULE_RANGE = (
    "At width 280, the Pearson correlation between the main fit's predicted count and the "
    "true count is measured inside each bin of 35 label values. Any bin starting above "
    "label 175 that reaches 0.5 means wide counts ARE measurable. Otherwise, if every bin "
    "ending at or below label 140 reaches 0.5 and every bin starting above 175 stays "
    "below 0.2, wide counts are NOT represented. Any other pattern is "
    "RANGE_PATTERN_UNMATCHED. The bin from 141 to 175 is recorded and excluded."
)
RULE_BINS_140 = (
    "Recorded only, with no threshold and no pass-or-fail call: the Pearson correlation "
    "between the main fit's predicted count and the true count, inside each bin of 35 "
    "label values across characters 1 to 140, at the chosen readout and the 140-character "
    "probe window. The gate record's answer 5 fixes this as purely descriptive."
)
RULE_WAVELENGTH_GAP = (
    "Over the probe windows whose character-count labelling cleared its own shuffle "
    "draws, the largest gap between any two periods, in characters, must be at most 10. "
    "Larger rules out a single fixed period. Fewer than two windows clearing leaves the "
    "gap undefined."
)
RULE_RATIO_GAP = (
    "Over the probe windows whose character-count labelling cleared its own shuffle "
    "draws, the largest gap between any two period-to-width ratios must be at most 0.1. "
    "Larger, together with a period gap above 10 characters, rules out a fixed proportion "
    "as well."
)
RULE_HEADLINE = (
    "A failed floor gate reads CURVE_TOO_WEAK and voids the rung. Otherwise, if any probe "
    "window's character-count labelling cleared its shuffle draws, the previous rung's "
    "invariance fork decides: INVARIANT when all period gaps are at most 10 characters, "
    "else PROPORTIONAL when all period-to-width ratio gaps are at most 0.1, else MIXED; "
    "exactly one clearing window is UNDETERMINED by design. If only the token-position "
    "labelling cleared, the reading is TOKEN_ONLY_PATTERN and the fork is not entered. If "
    "nothing cleared, the reading is PRECONDITION_WITHOUT_RIPPLE when the number of "
    "directions reached 4 at any window, and PRECONDITION_ABSENT_AT_1P4B when it did not."
)


def decide_setup(hash_ok: bool, n_layers, hidden_size) -> str:
    if hash_ok and exact_config_match(n_layers, hidden_size):
        return "SETUP_OK"
    return "SETUP_FAILURE"


def decide_floor(r2) -> str:
    """The existence floor. Absent or not a number counts as a failure."""
    if r2 is None or r2 != r2:
        return "CURVE_TOO_WEAK"
    return "FLOOR_CLEARED" if r2 >= FLOOR_R2 else "CURVE_TOO_WEAK"


def decide_dimension(dims_by_window) -> str:
    """Whether the count code reaches the pre-registered number of directions."""
    values = [d for d in dims_by_window.values() if d is not None]
    if not values:
        return "DIMENSION_NOT_COMPUTABLE"
    return ("DIMENSION_REACHES_4" if max(values) >= DIMENSION_THRESHOLD
            else "DIMENSION_BELOW_4")


def null_rank(observed, draws) -> int:
    """Place of the observed value among the draws plus itself, largest first."""
    return 1 + sum(1 for v in draws if v >= observed)


def decide_cleared(rank: int) -> str:
    return "NULL_CLEARED" if rank <= NULL_RANK_MAX else "NULL_NOT_CLEARED"


def decide_any_cleared(n_char_cleared: int, n_token_cleared: int) -> str:
    if n_char_cleared + n_token_cleared > 0:
        return "ANY_WINDOW_CLEARED"
    return "NO_WINDOW_CLEARED"


def decide_range(low_corrs, high_corrs) -> str:
    """The measurable-range call, from per-bin Pearson correlations.

    Low bins are the four ending at or below label 140. High bins are the three
    starting above label 175. The bin from 141 to 175 is excluded before this call.
    """
    lows = list(low_corrs)
    highs = list(high_corrs)
    if any(r is None for r in lows + highs):
        return "RANGE_CHECK_INCOMPLETE"
    if any(r >= RANGE_MEASURABLE_FLOOR for r in highs):
        return "WIDE_COUNTS_MEASURABLE"
    if all(r >= RANGE_LOW_FLOOR for r in lows) and all(r < RANGE_HIGH_CEIL for r in highs):
        return "WIDE_COUNTS_NOT_REPRESENTED"
    return "RANGE_PATTERN_UNMATCHED"


def pairwise_gaps(cleared):
    """Largest period gap and largest ratio gap over clearing windows.

    `cleared` is a list of (probe width, period in characters) pairs. Returns
    (None, None) when fewer than two windows cleared, so no pair exists.
    """
    if len(cleared) < 2:
        return None, None
    gap = 0.0
    ratio_gap = 0.0
    for i in range(len(cleared)):
        for j in range(i + 1, len(cleared)):
            a_w, a_l = cleared[i]
            b_w, b_l = cleared[j]
            gap = max(gap, abs(a_l - b_l))
            ratio_gap = max(ratio_gap, abs(a_l / float(a_w) - b_l / float(b_w)))
    return gap, ratio_gap


def decide_wavelength_gap(gap) -> str:
    if gap is None:
        return "GAP_UNDEFINED_FEWER_THAN_TWO_CLEARED"
    return ("WAVELENGTH_GAP_WITHIN_10" if gap <= INVARIANCE_MAX_GAP_CHARS
            else "WAVELENGTH_GAP_EXCEEDS_10")


def decide_ratio_gap(gap) -> str:
    if gap is None:
        return "GAP_UNDEFINED_FEWER_THAN_TWO_CLEARED"
    return ("RATIO_GAP_WITHIN_0.1" if gap <= PROPORTION_MAX_RATIO_GAP
            else "RATIO_GAP_EXCEEDS_0.1")


def decide_invariance(cleared) -> str:
    """The previous rung's fork over the windows that cleared their own draws.

    This rule is defined by the previous rung's design and is not redefined here. A
    single fixed period is checked first, because it is the stronger claim about the
    model and the design lists it first. Both rules can hold at once when the periods
    sit close together; that case is recorded alongside the call.
    """
    if not cleared:
        return "NO_WIDTH_CLEARED"
    if len(cleared) == 1:
        return "UNDETERMINED"
    gap, ratio_gap = pairwise_gaps(cleared)
    if gap <= INVARIANCE_MAX_GAP_CHARS:
        return "INVARIANT"
    if ratio_gap <= PROPORTION_MAX_RATIO_GAP:
        return "PROPORTIONAL"
    return "MIXED"


def decide_attribution(wl_chars, wl_token_in_chars, char_cleared, token_cleared) -> str:
    """Which labelling owns the pattern at one probe window.

    Correction 2 of the design adds the no-signal branch: two absent patterns can never
    produce a 'distinguished' verdict. The gate record's answer 2 adds the token-only
    branch, which is a real finding about positional structure and nothing more.
    """
    if wl_token_in_chars is None or token_cleared is None:
        return "TOKEN_CONTROL_NOT_COMPUTABLE"
    if not char_cleared and not token_cleared:
        return "NO_PATTERN_TO_ATTRIBUTE"
    if token_cleared and not char_cleared:
        return "TOKEN_ONLY_PATTERN"
    if char_cleared and not token_cleared:
        return "CHARACTER_COUNT_DISTINGUISHED"
    if abs(wl_chars - wl_token_in_chars) > CONFOUND_MARGIN_CHARS:
        return "CHARACTER_COUNT_DISTINGUISHED"
    return "CONFOUNDED_WITH_TOKEN_POSITION"


def decide_headline(floor_decision: str, cleared, token_cleared_any: bool,
                    dimension_decision: str) -> str:
    """One call for the whole run, from the calls already made.

    The four pre-registered readings are CURVE_TOO_WEAK, PRECONDITION_ABSENT_AT_1P4B,
    PRECONDITION_WITHOUT_RIPPLE, and the previous rung's fork. A token-only clearing is
    recorded as TOKEN_ONLY_PATTERN and does not enter the fork.
    """
    if floor_decision != "FLOOR_CLEARED":
        return "CURVE_TOO_WEAK"
    if cleared:
        return decide_invariance(cleared)
    if token_cleared_any:
        return "TOKEN_ONLY_PATTERN"
    if dimension_decision == "DIMENSION_REACHES_4":
        return "PRECONDITION_WITHOUT_RIPPLE"
    return "PRECONDITION_ABSENT_AT_1P4B"


# ---------------------------------------------------------------------------
# Part 6. Ridge fitting from running totals.
#
# Storing every internal vector of every position would need hundreds of gigabytes at
# this hidden size. It is not needed. A ridge fit and a fit-quality score depend on the
# data only through five running totals per readout: the sum of outer products of the
# vectors, the sum of the vectors, the sum of vector times label, the sum of labels, and
# the sum of squared labels. So the first stage accumulates those, per readout, per group
# of documents.
#
# At hidden size 2048 one group's outer-product totals hold 25 readouts times 2048 times
# 2048 double-precision numbers, about 839 megabytes. Fourteen groups hold about 11.8
# gigabytes, four times the previous rung's. That is the whole reason this design keeps
# running totals instead of vectors, and the reason per-position vectors are saved for
# the three probe windows on held-out documents only.
#
# The seven wrap widths 20 through 140 are the only training data, as the design says.
# The held-out totals are kept per wrap width, so the pooled seven-width score, the
# width-140-only score, and the width-200 and width-280 scores all come from the same
# totals at every readout.
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


def pearson_corr(pred, actual):
    """Pearson correlation inside one bin, and the count it used. Needs numpy.

    Returns (None, n) when fewer than two positions fall in the bin, or when either
    side has no spread at all. That is the design's un-measurable case, recorded as
    absent rather than as a number.
    """
    import numpy as np
    n = int(pred.size)
    if n < 2:
        return None, n
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    pd = p - p.mean()
    ad = a - a.mean()
    denom = math.sqrt(float(pd @ pd) * float(ad @ ad))
    if denom <= 0.0:
        return None, n
    return float(pd @ ad) / denom, n


# ---------------------------------------------------------------------------
# Part 7. Shared plumbing.
# ---------------------------------------------------------------------------

def load_provenance():
    """Import provenance.py from beside this file, or from the frozen folder."""
    try:
        import provenance                        # on the box the file sits alongside
    except ImportError:
        sys.path.insert(0, str(HERE / "frozen"))
        import provenance                        # on the workstation it sits in frozen/
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
            raise Halt("the named text file %s is absent; the design says halt, never "
                       "regenerate" % p)
        return p
    p = find_first(["stimulus_linewrap_v2.json", "records/stimulus_linewrap_v2.json"])
    if p is None:
        raise Halt("stimulus_linewrap_v2.json is absent; the design says halt, never "
                   "regenerate, re-wrap, or reconstruct it")
    return p


def design_record_path(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return find_first([
        "seedD3_design_diff_triage.json",              # box layout, flat
        "records/seedD3_design_diff_triage.json",      # workstation layout
    ])


def gate_record_path():
    return find_first(["GATE_COMMIT_seedD3.md", "records/GATE_COMMIT_seedD3.md"])


def hf_token():
    """Access token, and the name of the variable it came from.

    The build brief names HF_KEY_ONE. The launcher forwards the same secret as
    HF_TOKEN, and the bootstrap script also exports HUGGING_FACE_HUB_TOKEN. All three
    are tried, in that order. This model is public, so an absent token is fine and is
    recorded as such. No token value is ever printed or written.
    """
    for name in ("HF_KEY_ONE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(name)
        if val:
            return val, name
    return None, "none"


def load_stimulus(path):
    """Read the committed text file and check it against the design, or halt."""
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
    train_ids = sorted(int(d["id"]) for d in docs if d.get("split") == "train")
    test_ids = sorted(int(d["id"]) for d in docs if d.get("split") == "test")
    if train_ids != list(range(N_TRAIN_DOCS)):
        problems.append("training documents are %r, expected 0 to %d in book order"
                        % (train_ids[:5], N_TRAIN_DOCS - 1))
    if test_ids != list(range(N_TRAIN_DOCS, N_DOCS)):
        problems.append("held-out documents are %r, expected %d to %d"
                        % (test_ids[:5], N_TRAIN_DOCS, N_DOCS - 1))
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

    Splitting by document, not by position, keeps sentences of one document out of both
    sides of a split. The one fixed base seed makes the split repeatable, which is the
    gate record's answer 4.
    """
    import numpy as np
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(train_ids)))
    assign = {}
    for rank, pos in enumerate(order):
        assign[train_ids[int(pos)]] = rank % N_FOLDS
    return assign


TRAIN_GROUPS = ["fold%d" % f for f in range(N_FOLDS)]
TEST_GROUPS = ["test_w%d" % w for w in WIDTHS]
SELECT_TEST_GROUPS = ["test_w%d" % w for w in SELECT_WIDTHS]


def group_names():
    """The groups the running totals are kept for.

    Five groups split the training documents. Each holds that fold's positions at the
    seven wrap widths 20 through 140 only, which is the design's training set. One
    further group per wrap width holds the held-out documents at that width.
    """
    return list(TRAIN_GROUPS) + list(TEST_GROUPS)


def in_training_fit(split: str, width: int) -> bool:
    """Whether a text variant's positions enter the fit's running totals.

    Held-out documents are kept at every wrap width, so widths 200 and 280 can be
    scored at every readout without ever being trained on or gated. Training documents
    are kept at the seven widths 20 through 140 only, which is what the design says the
    fit and the strength search may see.
    """
    if split == "test":
        return True
    return width in SELECT_WIDTHS


def totals_bytes(n_read: int = EXPECT_LAYERS + 1, hidden: int = EXPECT_HIDDEN,
                 n_groups: int = None) -> int:
    """Size of the running totals in memory, so the log states it before the work."""
    if n_groups is None:
        n_groups = len(group_names())
    per_group = n_read * hidden * hidden * 8 + 2 * n_read * hidden * 8
    return per_group * n_groups


def shuffle_seed_entropy(aperture: int, labelling: str):
    """The seed values for one shuffle stream, all derived from base seed 0.

    The design fixes one base seed, 0. Each (probe window, labelling) pair gets its own
    stream from that base, so the character draws and the token draws at one window
    never share numbers. The values are recorded in the record they produced, which is
    the gate record's answer 4.
    """
    if labelling not in LABELLING_CODES:
        raise Halt("unknown labelling %r; the design names characters and tokens" % labelling)
    return [SEED, int(aperture), LABELLING_CODES[labelling]]


def paths_for(results_dir):
    out = Path(results_dir) / "out"
    work = Path(results_dir) / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def record_inputs(prov, stim, design, gate, extra=()):
    items = [prov.describe_input(stim, n_items=N_DOCS * len(WIDTHS))]
    if design is not None:
        items.append(prov.describe_input(design, n_items=1))
    if gate is not None:
        items.append(prov.describe_input(gate, n_items=1))
    for path, n in extra:
        items.append(prov.describe_input(path, n_items=n))
    return items


def base_cell(stage, arm, endpoint=None, aperture=None):
    cell = {"model": MODEL_SLUG, "revision": REVISION, "stage": stage, "arm": arm}
    if endpoint is not None:
        cell["endpoint"] = endpoint
    if aperture is not None:
        cell["aperture"] = aperture
    return cell


def write_record(prov, out_dir, cell, **kwargs):
    path = Path(out_dir) / (prov.cell_slug(cell) + ".json")
    prov.write_result(path, cell=cell, **kwargs)
    print("[record] %s -> %s" % (kwargs.get("decision", ""), path))
    return path


def floor_values(floor_decision, best_pooled, anchor_layer):
    """The floor gate's own decision, for embedding in another record's values.

    The design's reporting requirement is that every per-window pattern record and every
    control record carries this decision in its own values. So no such record can be
    quoted without its void status attached.
    """
    return {
        "floor_gate_decision": floor_decision,
        "floor_gate_quality_pooled_widths_20_to_140":
            (None if best_pooled is None else round(float(best_pooled), 6)),
        "floor_gate_floor": FLOOR_R2,
        "floor_gate_readout": anchor_layer,
        "this_record_voided_by_floor_gate": floor_decision != "FLOOR_CLEARED",
    }


def act_file(work_dir, aperture, layer):
    return Path(work_dir) / ("act_a%d_layer%02d.f32" % (aperture, layer))


def labels_file(work_dir, aperture):
    return Path(work_dir) / ("labels_a%d.npz" % aperture)


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
    gate = gate_record_path()

    actual_hash = sha256_file(stim)
    hash_ok = exact_hash_match(actual_hash, STIMULUS_SHA256)
    print("[extract] text file %s" % stim)
    print("[extract] hash %s (expected %s) match=%s"
          % (actual_hash, STIMULUS_SHA256, hash_ok))
    meta, docs = load_stimulus(stim)

    token, token_env = hf_token()
    print("[extract] access token from %s" % token_env)
    kw = {"revision": REVISION}
    if token:
        kw["token"] = token

    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    context_limit = int(getattr(config, "max_position_embeddings", CONTEXT_LIMIT_FALLBACK)
                        or CONTEXT_LIMIT_FALLBACK)
    print("[extract] configuration layers=%r hidden=%r context=%d"
          % (n_layers, hidden, context_limit))

    decision = decide_setup(hash_ok, n_layers, hidden)
    write_record(
        prov, out_dir, base_cell("extract", "setup_check"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Committed text file hash, and the downloaded configuration's layer count and hidden size.",
        values={"stimulus_sha256": actual_hash,
                "stimulus_sha256_expected": STIMULUS_SHA256,
                "stimulus_hash_match": hash_ok,
                "n_layers": n_layers, "hidden_size": hidden,
                "n_layers_expected": EXPECT_LAYERS,
                "hidden_size_expected": EXPECT_HIDDEN,
                "context_limit": context_limit,
                "model_id": MODEL_ID, "revision": REVISION,
                "design_record_found": str(design) if design else None,
                "design_record_expected_sha256": DESIGN_SHA256,
                "gate_record_found": str(gate) if gate else None,
                "gate_record_expected_sha256": GATE_SHA256,
                "access_token_source": token_env,
                "text_source": meta.get("source"),
                "disjoint_from": meta.get("disjoint_from")},
        threshold={"stimulus_sha256": STIMULUS_SHA256, "n_layers": EXPECT_LAYERS,
                   "hidden_size": EXPECT_HIDDEN},
        decision_rule=RULE_SETUP,
        decision=decision,
        notes=("Layer count 24 and hidden size 2048 are recalled from the Pythia paper, "
               "arXiv 2304.01373 table 1, and checked against the download, which is "
               "authoritative. A mismatch halts before any measurement. The text is loaded "
               "exactly as committed and is never regenerated or re-wrapped. It is the same "
               "text file the 410-million-weight rung used, and nothing in it is specific to "
               "that model. " + RERUN_NOTE),
    )
    if decision != "SETUP_OK":
        raise Halt("the setup check failed: hash match=%s layers=%r hidden=%r "
                   "(expected %d layers, hidden %d)"
                   % (hash_ok, n_layers, hidden, EXPECT_LAYERS, EXPECT_HIDDEN))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    if not getattr(tok, "is_fast", False):
        raise Halt("this tokenizer reports no character offsets, so labels cannot be computed")
    tokenizer_class = type(tok).__name__

    # Only one model exists in this experiment, and it is resident only here. It is
    # loaded in half precision, as the build brief requires and the design permits.
    # Reduced-precision matrix products are turned off so the running totals stay
    # accurate; the accumulation itself is in double precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    dtype_name = "float16" if device == "cuda" else "float32"
    print("[extract] loading the model in %s on %s" % (dtype_name, device))
    if device != "cuda":
        print("[extract] WARNING: no graphics card visible, so half precision is "
              "replaced by single precision on the processor")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **kw)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    n_read = EXPECT_LAYERS + 1                        # readout 0 is the embedding output
    groups = group_names()
    # State the memory bill before paying it. At hidden size 2048 these totals are four
    # times the previous rung's, which is why they stay totals and are never expanded.
    print("[extract] running totals: %d groups times %d readouts at hidden %d = %.1f GB"
          % (len(groups), n_read, EXPECT_HIDDEN, totals_bytes(n_read) / 1e9))
    print("[extract] per-position vectors are saved for the three probe windows %r on "
          "held-out documents only, at every readout" % (APERTURES,))
    gram = {g: np.zeros((n_read, EXPECT_HIDDEN, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    vec_sum = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    cross = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    label_sum = {g: 0.0 for g in groups}
    label_sq = {g: 0.0 for g in groups}
    n_pos = {g: 0 for g in groups}
    n_pos_outside_fit = 0

    train_ids = [int(d["id"]) for d in docs if d["split"] == "train"]
    folds = fold_of_documents(train_ids)

    # Per-position internal vectors are kept only for the held-out documents at the
    # three probe windows, at every readout. The chosen readout is not known until the
    # last stage, so all 25 are kept. Nothing else needs per-position vectors, and
    # keeping them for every width would be many times the disk of the whole run.
    act_files = {a: [open(act_file(work_dir, a, l), "wb") for l in range(n_read)]
                 for a in APERTURES}
    lab = {a: {"doc": [], "is_test": [], "char": [], "token": [], "line": []}
           for a in APERTURES}

    # Every text variant is saved verbatim, with its own hash. Every labelled position
    # of the three probe windows on held-out documents is saved as its own row. The same
    # table for all 432 variants is far larger and stays in the working folder, which
    # is never fetched.
    rows_var = open(Path(out_dir) / "rows_variants.jsonl", "w", encoding="utf-8")
    rows_ap = open(Path(out_dir) / "rows_positions_apertures.jsonl", "w", encoding="utf-8")
    rows_all = open(Path(work_dir) / "rows_positions_all.jsonl", "w", encoding="utf-8")

    started = time.time()
    n_variants = 0
    all_flags = {}
    try:
        for doc in docs:
            did = int(doc["id"])
            split = doc["split"]
            for width in WIDTHS:
                group = ("test_w%d" % width) if split == "test" else ("fold%d" % folds[did])
                use_for_fit = in_training_fit(split, width)
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
                t_ids = torch.tensor([ids], dtype=torch.long, device=device)
                out = model(input_ids=t_ids, output_hidden_states=True, use_cache=False)
                states = out.hidden_states
                if len(states) != n_read:
                    raise Halt("the model returned %d readouts, expected %d"
                               % (len(states), n_read))
                idx = torch.tensor(keep, dtype=torch.long, device=device)
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                want_act = (width in APERTURES) and (split == "test")
                # Training text at widths 200 and 280 is read, labelled, and saved
                # verbatim, and nothing is taken from its readouts. Walking them here
                # would only cost time, so that case skips the loop below.
                if use_for_fit or want_act:
                    for l in range(n_read):
                        x = states[l][0].index_select(0, idx).float()
                        if use_for_fit:
                            gram[group][l] += (x.T @ x).double().cpu().numpy()
                            vec_sum[group][l] += x.sum(0).double().cpu().numpy()
                            cross[group][l] += (x.T @ y_t).double().cpu().numpy()
                        if want_act:
                            x.cpu().numpy().astype(np.float32).tofile(act_files[width][l])
                if use_for_fit:
                    label_sum[group] += float(y.sum())
                    label_sq[group] += float((y * y).sum())
                    n_pos[group] += len(keep)
                else:
                    n_pos_outside_fit += len(keep)

                if width in APERTURES:
                    is_test = 1 if split == "test" else 0
                    lab[width]["doc"].extend([did] * len(keep))
                    lab[width]["is_test"].extend([is_test] * len(keep))
                    lab[width]["char"].extend([char_labels[i] for i in keep])
                    lab[width]["token"].extend([token_labels[i] for i in keep])
                    lab[width]["line"].extend([line_nums[i] for i in keep])
                for i in keep:
                    row = {"doc": did, "width": width, "split": split, "group": group,
                           "in_training_fit": use_for_fit,
                           "tok_index": i, "tok_text": pieces[i],
                           "char_start": spans[i][0], "char_end": spans[i][1],
                           "line": line_nums[i], "char_label": char_labels[i],
                           "token_label": token_labels[i]}
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    rows_all.write(line)
                    if want_act:
                        rows_ap.write(line)
                rows_var.write(json.dumps({
                    "doc": did, "width": width, "split": split, "group": group,
                    "in_training_fit": use_for_fit,
                    "n_chars": len(text), "n_tokens": len(ids), "n_labelled": len(keep),
                    "n_lines": text.count("\n") + 1,
                    "degenerate_flags": flags,
                    "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text}, ensure_ascii=False) + "\n")
                n_variants += 1
                del out, states
        elapsed = time.time() - started
    finally:
        rows_var.close()
        rows_ap.close()
        rows_all.close()
        for a in APERTURES:
            for fh in act_files[a]:
                fh.close()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    payload = {"groups": np.array(groups),
               "label_sum": np.array([label_sum[g] for g in groups]),
               "label_sq": np.array([label_sq[g] for g in groups]),
               "n_pos": np.array([n_pos[g] for g in groups]),
               "n_read": np.array([n_read]), "hidden": np.array([EXPECT_HIDDEN])}
    for g in groups:
        payload["gram__" + g] = gram[g]
        payload["vec_sum__" + g] = vec_sum[g]
        payload["cross__" + g] = cross[g]
    np.savez(Path(work_dir) / "totals.npz", **payload)
    for a in APERTURES:
        np.savez(labels_file(work_dir, a),
                 doc=np.array(lab[a]["doc"], dtype=np.int32),
                 is_test=np.array(lab[a]["is_test"], dtype=np.int8),
                 char_label=np.array(lab[a]["char"], dtype=np.int32),
                 token_label=np.array(lab[a]["token"], dtype=np.int32),
                 line=np.array(lab[a]["line"], dtype=np.int32))
    with open(Path(work_dir) / "folds.json", "w", encoding="utf-8") as fh:
        json.dump({"base_seed": SEED, "assignment": {str(k): v for k, v in folds.items()}},
                  fh, indent=2)

    rows_per_aperture = {a: int(sum(lab[a]["is_test"])) for a in APERTURES}
    write_record(
        prov, out_dir, base_cell("extract", "forward_passes"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Counts from one forward pass per text variant, and the running totals saved for the fits.",
        values={"n_variants": n_variants,
                "n_positions_by_group": {g: n_pos[g] for g in groups},
                "n_positions_in_fit_totals": int(sum(n_pos.values())),
                "n_positions_outside_fit_totals": n_pos_outside_fit,
                "training_widths_in_fit": SELECT_WIDTHS,
                "training_widths_excluded_from_fit": WIDE_WIDTHS,
                "n_readouts": n_read, "hidden_size": EXPECT_HIDDEN,
                "device": device, "dtype": dtype_name,
                "tokenizer_class": tokenizer_class,
                "running_totals_bytes": totals_bytes(n_read),
                "per_position_vectors_saved_for_apertures": APERTURES,
                "per_position_vectors_saved_for_split": "test",
                "seconds": round(elapsed, 1),
                "held_out_rows_by_aperture": rows_per_aperture,
                "degenerate_variants": all_flags,
                "fold_assignment": {str(k): v for k, v in folds.items()}},
        threshold={"n_variants": N_DOCS * len(WIDTHS)},
        decision_rule=RULE_EXTRACT,
        decision=("EXTRACT_OK" if n_variants == N_DOCS * len(WIDTHS)
                  else "EXTRACT_INCOMPLETE"),
        notes=("Readout 0 is the embedding output and readout 24 is the last layer. This "
               "library applies the model's final normalisation to readout 24 only, which "
               "is a property of the library, recorded here rather than changed. Every one "
               "of the 432 text variants gets one deterministic forward pass, one variant "
               "at a time; there is nothing to batch, because each variant is a whole "
               "document and its labels are per position. Training documents at widths 200 "
               "and 280 are read and labelled, and their positions are deliberately kept "
               "out of the fit's running totals, because the design trains on the seven "
               "widths 20 through 140 only. Per-position vectors are saved for the three "
               "probe windows on held-out documents only; at hidden size 2048 saving them "
               "for every width and readout would be many times the disk of the whole run. "
               "No text is generated at any point, and the tokenizer family is recorded "
               "above rather than assumed. " + RERUN_NOTE),
    )
    print("[extract] %d variants, %d positions in the fit totals, %d outside them, %.1f s"
          % (n_variants, sum(n_pos.values()), n_pos_outside_fit, elapsed))
    if n_variants != N_DOCS * len(WIDTHS):
        raise Halt("only %d of %d text variants were read"
                   % (n_variants, N_DOCS * len(WIDTHS)))
    return 0


# ---------------------------------------------------------------------------
# Part 9. Stage two: fix every remaining constant, using training documents only.
# The build brief calls this position the generation stage. Nothing is generated.
# ---------------------------------------------------------------------------

def load_totals(work_dir):
    import numpy as np
    path = Path(work_dir) / "totals.npz"
    if not path.exists():
        raise Halt("no running totals at %s; the extract stage must run first" % path)
    z = np.load(path, allow_pickle=False)
    groups = [str(g) for g in z["groups"]]
    tot = {"groups": groups, "n_read": int(z["n_read"][0]),
           "gram": {}, "vec_sum": {}, "cross": {},
           "label_sum": {}, "label_sq": {}, "n_pos": {}}
    for i, g in enumerate(groups):
        tot["gram"][g] = z["gram__" + g]
        tot["vec_sum"][g] = z["vec_sum__" + g]
        tot["cross"][g] = z["cross__" + g]
        tot["label_sum"][g] = float(z["label_sum"][i])
        tot["label_sq"][g] = float(z["label_sq"][i])
        tot["n_pos"][g] = int(z["n_pos"][i])
    return tot


def sum_groups(tot, names, layer):
    """Running totals of several groups added together, at one readout."""
    g = sum(tot["gram"][k][layer] for k in names)
    v = sum(tot["vec_sum"][k][layer] for k in names)
    c = sum(tot["cross"][k][layer] for k in names)
    ls = sum(tot["label_sum"][k] for k in names)
    lq = sum(tot["label_sq"][k] for k in names)
    n = sum(tot["n_pos"][k] for k in names)
    return g, v, c, ls, lq, n


def counts_from_labels(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def stage_calibrate(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    gate = gate_record_path()
    tot = load_totals(work_dir)
    n_read = tot["n_read"]

    # Ridge strength, by five-fold cross-validation over training documents only. Those
    # totals hold the seven wrap widths 20 through 140 and nothing else, so this search
    # is pooled over exactly those seven widths.
    table = {}
    for held in range(N_FOLDS):
        fit_names = [g for g in TRAIN_GROUPS if g != "fold%d" % held]
        eval_names = ["fold%d" % held]
        for layer in range(n_read):
            g, v, c, ls, lq, n = sum_groups(tot, fit_names, layer)
            ge, ve, ce, lse, lqe, ne = sum_groups(tot, eval_names, layer)
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
    interior = alpha_star not in (RIDGE_GRID[0], RIDGE_GRID[-1])
    print("[calibrate] ridge strength %g (interior on the grid: %s), best training "
          "readout %d, cross-validated quality %.4f"
          % (alpha_star, interior, train_best_layer, train_best_r2))

    # Label windows and profile lengths, one set per probe window, on training text at
    # that width only. No held-out number exists yet, and none may.
    per_aperture = {}
    problems = []
    for a in APERTURES:
        z = np.load(labels_file(work_dir, a), allow_pickle=False)
        train_mask = z["is_test"] == 0
        char_counts = counts_from_labels(z["char_label"][train_mask].tolist())
        token_counts = counts_from_labels(z["token_label"][train_mask].tolist())
        window_lo = window_from_counts(char_counts, a, MIN_TRAIN_POSITIONS_PER_LABEL)
        if window_lo is None:
            problems.append("width %d has no label window ending at %d whose values all "
                            "reach %d training positions"
                            % (a, a, MIN_TRAIN_POSITIONS_PER_LABEL))
            continue
        n_labels = a - window_lo + 1
        max_delta = profile_length_for(n_labels)
        committed = COMMITTED_PROFILE_LENGTH[a]
        if max_delta != committed:
            problems.append(
                "width %d gives a profile length of %d by the committed rule, the label "
                "count minus %d, but the design also states %d for this width; the two "
                "committed statements disagree and a human must settle it"
                % (a, max_delta, PROFILE_MARGIN, committed))
        ceiling = token_ceiling_report(token_counts)
        token_hi = ceiling["token_ceiling"]
        tok_max_delta = profile_length_for(token_hi)
        char_periods = periods_in_range(max_delta) if max_delta >= DETREND_DEGREE + 2 else []
        tok_periods = periods_in_range(tok_max_delta) if tok_max_delta >= DETREND_DEGREE + 2 else []
        if not char_periods:
            problems.append("width %d leaves a character profile of length %d, too short "
                            "to detrend or to carry a period of %d"
                            % (a, max_delta, PERIOD_MIN))
        if not tok_periods:
            problems.append("width %d leaves a token profile of length %d, too short to "
                            "detrend or to carry a period of %d"
                            % (a, tok_max_delta, PERIOD_MIN))
        entry = {
            "character_window_low": window_lo,
            "character_window_high": a,
            "n_labels": n_labels,
            "min_training_positions_per_label":
                int(min(char_counts.get(v, 0) for v in range(window_lo, a + 1))),
            "max_delta": max_delta,
            "committed_profile_length": committed,
            "profile_length_matches_committed": max_delta == committed,
            "n_periods_in_range": len(char_periods),
            "longest_period": (char_periods[0][1] if char_periods else None),
            "shortest_period": (char_periods[-1][1] if char_periods else None),
            "token_window_high": token_hi,
            "token_max_delta": tok_max_delta,
            "token_n_periods_in_range": len(tok_periods),
            "token_longest_period": (tok_periods[0][1] if tok_periods else None),
            "token_shortest_period": (tok_periods[-1][1] if tok_periods else None),
        }
        entry.update(ceiling)
        per_aperture[a] = entry
        print("[calibrate] width %d: character labels %d to %d, profile length %d "
              "(committed %d), %d periods; token labels 1 to %d, ceiling %d against an "
              "observed maximum of %d, %d counts excluded, profile length %d, %d periods"
              % (a, window_lo, a, max_delta, committed, len(char_periods), token_hi,
                 token_hi, ceiling["token_label_observed_max"],
                 ceiling["token_labels_excluded_above_ceiling"], tok_max_delta,
                 len(tok_periods)))

    calib_decision = "CALIBRATION_OK" if not problems else "CALIBRATION_UNUSABLE"
    values = {
        "ridge_strength": alpha_star,
        "ridge_grid": RIDGE_GRID,
        "ridge_argmax_interior_on_grid": bool(interior),
        "training_widths": SELECT_WIDTHS,
        "cross_validated_quality_by_strength_and_readout":
            {"%g|%d" % (a, l): round(cv_mean[(a, l)], 6)
             for a in RIDGE_GRID for l in range(n_read)},
        "cross_validated_best_per_strength":
            {"%g" % a: {"readout": per_alpha_best[a][0],
                        "quality": round(per_alpha_best[a][1], 6)} for a in RIDGE_GRID},
        "cross_validated_best_readout": train_best_layer,
        "cross_validated_best_quality": round(train_best_r2, 6),
        "fold_count": N_FOLDS,
        "base_seed": SEED,
        "fold_split_seed": SEED,
        "detrend_method": "polynomial",
        "detrend_degree": DETREND_DEGREE,
        "shortest_period": PERIOD_MIN,
        "profile_length_rule": "the label count minus %d" % PROFILE_MARGIN,
        "committed_profile_lengths": {str(k): v for k, v in COMMITTED_PROFILE_LENGTH.items()},
        "min_training_positions_per_label": MIN_TRAIN_POSITIONS_PER_LABEL,
        "token_window_min_training_positions": TOKEN_WINDOW_MIN_TRAIN,
        "token_ceiling_rule": ("a token count enters the window only with at least %d "
                               "training positions; higher counts are excluded"
                               % TOKEN_WINDOW_MIN_TRAIN),
        "floor_gate_floor": FLOOR_R2,
        "dimension_threshold": DIMENSION_THRESHOLD,
        "variance_target_gated": VARIANCE_TARGET,
        "variance_targets_recorded_only": VARIANCE_TARGETS_RECORDED,
        "shuffle_draws": N_DRAWS,
        "shuffle_seed_streams":
            {"%d|%s" % (a, u): shuffle_seed_entropy(a, u)
             for a in APERTURES for u in sorted(LABELLING_CODES)},
        "per_aperture": {str(a): per_aperture[a] for a in sorted(per_aperture)},
        "problems": problems,
    }
    notes = [
        "Every number here comes from training documents only. No held-out number exists yet.",
        "The training totals hold the seven wrap widths 20 through 140 only, so this "
        "strength search is pooled over exactly those seven widths.",
        "The readout chosen by this cross-validation is recorded and is not the run's "
        "chosen readout. The design chooses that on held-out documents, in the next stage.",
        "The label window at a probe window ends at that width, which is what makes the "
        "comparison across windows a comparison of probe windows.",
        "The window's low end can sit above 1, because the first column of a line needs a "
        "one-character opening token and is thin. How far above 1 is recorded per window.",
        "A character label enters a window only if training text carries it often enough "
        "that the held-out text, a third the size, is very unlikely to leave it empty. An "
        "empty held-out label has no average vector, so the character reading would halt "
        "and the token reading would be recorded as not computable.",
        "The token ceiling is committed by correction 3 of the design: at least 40 "
        "training positions. The ceiling is recorded above beside the observed maximum "
        "token count and the number of counts the cut removed, so the cut is visible. The "
        "previous rung applied the same 40 without stating the rule.",
        "The profile length is committed by the design as the label count minus 20. The "
        "design also states it as 120, 180 and 260. Both are checked, and a disagreement "
        "halts rather than picking one.",
        "There is no search band. The statistic ranges over every whole number of cycles "
        "whose period is at least 4 and at most the profile length. That range is applied "
        "in each labelling's own units, so the token control searches token periods.",
        "The fold split and every shuffle stream derive from the one committed base seed, "
        "0, with a separate stream per probe window and labelling. That is the gate "
        "record's answer 4, and the values are recorded above.",
        "Whether the chosen strength sits inside the grid rather than at an end is "
        "recorded above. No levelling-off is required.",
    ]
    if not interior:
        notes.append("The chosen ridge strength sits at an end of the grid, so a wider grid "
                     "might score higher. This is stated, not fixed.")
    notes.append(RERUN_NOTE)

    write_record(
        prov, out_dir, base_cell("calibrate", "constants"),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=[(Path(work_dir) / "totals.npz", n_read)]),
        metric="Every remaining constant of the fit and of the pattern readout, fixed on training documents only.",
        values=values,
        threshold={"shortest_period": PERIOD_MIN, "detrend_degree": DETREND_DEGREE,
                   "token_window_min_training_positions": TOKEN_WINDOW_MIN_TRAIN,
                   "committed_profile_lengths":
                       {str(k): v for k, v in COMMITTED_PROFILE_LENGTH.items()}},
        decision_rule=RULE_CALIBRATION,
        decision=calib_decision,
        notes=" ".join(notes),
    )
    with open(Path(out_dir) / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump({"decision": calib_decision, "values": values}, fh, indent=2, sort_keys=True)
    if calib_decision != "CALIBRATION_OK":
        raise Halt("calibration is unusable: " + "; ".join(problems))
    return 0


# ---------------------------------------------------------------------------
# Part 10. Analysis helpers for the last stage.
# ---------------------------------------------------------------------------

def read_aperture_layer(work_dir, aperture, layer, n_rows, hidden=EXPECT_HIDDEN):
    import numpy as np
    path = act_file(work_dir, aperture, layer)
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != n_rows * hidden:
        raise Halt("saved vectors for width %d readout %d hold %d numbers, expected %d"
                   % (aperture, layer, arr.size, n_rows * hidden))
    return arr.reshape(n_rows, hidden)


def means_by_slot(x, slots, n_slots):
    """Average vector per slot, and how many positions entered each. Needs numpy."""
    import numpy as np
    keep = slots >= 0
    xk = x[keep]
    sk = slots[keep]
    ind = np.zeros((n_slots, xk.shape[0]), dtype=np.float32)
    ind[sk, np.arange(xk.shape[0])] = 1.0
    counts = ind.sum(axis=1)
    if float(counts.min()) < 1.0:
        empty = [int(i) for i in np.nonzero(counts < 1.0)[0]]
        raise Halt("label slots %r hold no held-out position, so their average vector is "
                   "undefined" % empty[:10])
    sums = ind @ xk
    return (sums / counts[:, None]).astype(np.float64), counts.astype(int)


def cosine_matrix(mat):
    """Cosine similarity between every pair of rows, as nested lists. Needs numpy."""
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if float(norms.min()) <= 0:
        raise Halt("a per-label average vector is exactly zero, so cosine similarity is "
                   "undefined")
    unit = mat / norms
    return (unit @ unit.T).tolist()


def slots_array(labels, window_lo, n_labels):
    import numpy as np
    off = labels - window_lo
    off = np.where((off >= 0) & (off < n_labels), off, -1)
    return off.astype(np.int64)


def variance_shares(centred):
    """Share of the spread held by each direction, largest first. Needs numpy."""
    import numpy as np
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv ** 2
    return (var / var.sum()).tolist()


def dimension_at(share, target):
    """How many directions are needed to reach the target share of the spread."""
    cum = 0.0
    dim = len(share)
    for i, s in enumerate(share):
        cum += s
        if cum >= target:
            dim = i + 1
            break
    return dim


def ambient_dimension(centred, target=VARIANCE_TARGET):
    """Directions holding the target share, and the whole share list. Needs numpy."""
    share = variance_shares(centred)
    return dimension_at(share, target), share


def dimensions_at_targets(share, gated=VARIANCE_TARGET,
                          recorded=tuple(VARIANCE_TARGETS_RECORDED)):
    """The gated count and the recorded-only counts, keyed by their variance target.

    The gate record's answer 1 fixes 95 percent as the one convention the reading uses,
    and requires 90 and 99 percent to be recorded beside it, deciding nothing.
    """
    out = {"%g" % gated: dimension_at(share, gated)}
    for t in recorded:
        out["%g" % t] = dimension_at(share, t)
    return out


def shuffle_within_docs(labels, docs, rng):
    """Permute labels among the positions of each document.

    Every label keeps exactly the same number of positions overall, so no slot can
    become empty in a draw that was not empty in the observed data.
    """
    import numpy as np
    out = labels.copy()
    for d in np.unique(docs):
        sel = np.nonzero(docs == d)[0]
        out[sel] = labels[sel][rng.permutation(sel.size)]
    return out


def pattern_with_null(x, labels, docs, window_lo, n_slots, max_delta, n_draws, rng,
                      spectra_path, unit):
    """Observed largest power and its shuffle draws, on one set of positions.

    Every draw's whole spectrum is written to `spectra_path`, one line per draw, as the
    design requires. The observed spectrum is the first line. Per-draw average vectors
    are deliberately not written; only the observed ones are, and by the caller.
    """
    import numpy as np
    slots = slots_array(labels, window_lo, n_slots)
    mat, counts = means_by_slot(x, slots, n_slots)
    grand = mat.mean(axis=0, keepdims=True)
    centred = mat - grand
    cos = cosine_matrix(centred)
    observed = max_power_stat(cos, max_delta, keep_profile=True)
    share = variance_shares(centred)
    dims = dimensions_at_targets(share)

    with open(spectra_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "kind": "observed", "draw": None, "unit": unit,
            "profile_length": observed["profile_length"],
            "max_power": observed["max_power"], "wavelength": observed["wavelength"],
            "cycles": observed["cycles"],
            "spectrum": observed["spectrum"]}) + "\n")
        draws = []
        for i in range(n_draws):
            shuffled = shuffle_within_docs(labels, docs, rng)
            s2 = slots_array(shuffled, window_lo, n_slots)
            m2, _ = means_by_slot(x, s2, n_slots)
            c2 = m2 - m2.mean(axis=0, keepdims=True)
            stat = max_power_stat(cosine_matrix(c2), max_delta)
            draws.append(stat["max_power"])
            fh.write(json.dumps({
                "kind": "shuffle", "draw": i, "unit": unit,
                "profile_length": stat["profile_length"],
                "max_power": stat["max_power"], "wavelength": stat["wavelength"],
                "cycles": stat["cycles"],
                "spectrum": stat["spectrum"]}) + "\n")
    rank = null_rank(observed["max_power"], draws)
    return {"observed": observed, "draws": draws, "rank": rank,
            "counts": [int(c) for c in counts], "means": mat, "grand_mean": grand[0],
            "cosine": cos, "ambient_dimension": dims["%g" % VARIANCE_TARGET],
            "ambient_dimension_by_target": dims, "variance_share": share}


def bin_correlation_rows(pred, labels, bins, with_roles):
    """Per-bin correlation rows, and the low and high lists when roles apply.

    Needs numpy. `pred` and `labels` are one entry per held-out position.
    """
    rows = []
    lows = []
    highs = []
    for (lo, hi) in bins:
        sel = (labels >= lo) & (labels <= hi)
        r, n_in = pearson_corr(pred[sel], labels[sel])
        row = {"bin_low": lo, "bin_high": hi, "n_positions": n_in,
               "pearson_correlation": (None if r is None else round(float(r), 6))}
        if with_roles:
            role = bin_role(lo, hi)
            row["role"] = role
            if role == "low":
                lows.append(r)
            elif role == "high":
                highs.append(r)
        rows.append(row)
    return rows, lows, highs


# ---------------------------------------------------------------------------
# Part 11. Stage three: the per-readout held-out fit and its floor gate, the pattern
# at each probe window, the shuffles, the token-position control, and the two
# bin-correlation checks.
# ---------------------------------------------------------------------------

def stage_endpoints(args):
    import numpy as np

    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    gate = gate_record_path()
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        raise Halt("no calibration record at %s; the calibrate stage must run first"
                   % calib_path)
    with open(calib_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    if calib.get("decision") != "CALIBRATION_OK":
        raise Halt("the calibration record says %r, so no held-out number may be computed"
                   % calib.get("decision"))
    cv = calib["values"]
    alpha = float(cv["ridge_strength"])
    per_ap = {int(k): v for k, v in cv["per_aperture"].items()}

    tot = load_totals(work_dir)
    n_read = tot["n_read"]

    # ---- The readout sweep, and the floor gate. ------------------------------
    # The chosen readout and the gate call both come from held-out fit quality pooled
    # over the seven wrap widths 20 through 140. That is one number, used for both. The
    # width-140-only quality and the width-200 and width-280 qualities are recorded at
    # every readout and never gated. The record carries rounded numbers for reading; the
    # call is taken from the unrounded ones, so no rounding can move a decision.
    per_layer = []
    exact = []
    for layer in range(n_read):
        g, v, c, ls, lq, n = sum_groups(tot, TRAIN_GROUPS, layer)
        w, off = ridge_fit_from_stats(g, v, c, ls, n, alpha)
        gs, vs, cs, lss, lqs, ns = sum_groups(tot, SELECT_TEST_GROUPS, layer)
        pooled_seven, _, _ = r2_from_stats(w, off, gs, vs, cs, lss, lqs, ns)
        ga, va, ca, lsa, lqa, na = sum_groups(tot, TEST_GROUPS, layer)
        pooled_nine, _, _ = r2_from_stats(w, off, ga, va, ca, lsa, lqa, na)
        row = {"readout": layer,
               "held_out_quality_pooled_widths_20_to_140": round(float(pooled_seven), 6),
               "held_out_quality_pooled_all_nine_widths": round(float(pooled_nine), 6)}
        by_width = {}
        for width in WIDTHS:
            gw, vw, cw, lsw, lqw, nw = sum_groups(tot, ["test_w%d" % width], layer)
            r2w, _, _ = r2_from_stats(w, off, gw, vw, cw, lsw, lqw, nw)
            by_width[width] = float(r2w)
            row["held_out_quality_w%d" % width] = round(float(r2w), 6)
        per_layer.append(row)
        exact.append({"readout": layer, "pooled_seven": float(pooled_seven),
                      "pooled_nine": float(pooled_nine), "by_width": by_width})
    best_exact = max(exact, key=lambda r: r["pooled_seven"])
    anchor_layer = int(best_exact["readout"])
    anchor_pooled = best_exact["pooled_seven"]
    best = per_layer[anchor_layer]
    floor_decision = decide_floor(anchor_pooled)
    floor_block = floor_values(floor_decision, anchor_pooled, anchor_layer)
    n_layers_over_floor = sum(1 for r in exact if r["pooled_seven"] >= FLOOR_R2)
    print("[endpoints] chosen readout %d, pooled quality over widths 20 to 140 %.6f -> %s"
          % (anchor_layer, anchor_pooled, floor_decision))
    print("[endpoints] at that readout: width 140 %.6f, width 200 %.6f, width 280 %.6f "
          "(recorded, never gated)"
          % (best_exact["by_width"][140], best_exact["by_width"][200],
             best_exact["by_width"][280]))
    if floor_decision != "FLOOR_CLEARED":
        print("[endpoints] the floor failed, so every pattern result below is void as "
              "evidence about the model and carries that flag")
    floor_gate_values = {"chosen_readout": anchor_layer,
                         "held_out_quality_pooled_widths_20_to_140": round(anchor_pooled, 6),
                         "held_out_quality_pooled_all_nine_widths":
                             best["held_out_quality_pooled_all_nine_widths"],
                         "held_out_quality_w140_only": best["held_out_quality_w140"],
                         "held_out_quality_w200": best["held_out_quality_w200"],
                         "held_out_quality_w280": best["held_out_quality_w280"],
                         "n_readouts_reaching_floor": n_layers_over_floor,
                         "selection_widths": SELECT_WIDTHS,
                         "widths_recorded_never_gated": WIDE_WIDTHS,
                         "ridge_strength": alpha,
                         "ridge_argmax_interior_on_grid": cv["ridge_argmax_interior_on_grid"],
                         "cross_validated_best_readout": cv["cross_validated_best_readout"],
                         "training_positions": int(sum(tot["n_pos"][k] for k in TRAIN_GROUPS)),
                         "held_out_positions_widths_20_to_140":
                             int(sum(tot["n_pos"][k] for k in SELECT_TEST_GROUPS)),
                         "held_out_positions_all_widths":
                             int(sum(tot["n_pos"][k] for k in TEST_GROUPS)),
                         "prior_410m_pooled_quality": PRIOR_410M_POOLED,
                         "prior_410m_pooled_quality_first_run": PRIOR_410M_POOLED_FIRST_RUN,
                         "prior_410m_readout": PRIOR_410M_READOUT,
                         "prior_reproduction_pythia_160m": PRIOR_REPRODUCTION_PYTHIA_160M,
                         "prior_reproduction_gpt2": PRIOR_REPRODUCTION_GPT2}
    floor_gate_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "floor_gate", endpoint=1),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=[(Path(work_dir) / "totals.npz", n_read)]),
        metric="Best-readout held-out fit quality for the character-count curve, pooled over wrap widths 20 to 140.",
        values=floor_gate_values,
        threshold={"at_least": FLOOR_R2},
        decision_rule=RULE_FLOOR,
        decision=floor_decision,
        rows=per_layer,
        notes=("This gate is an existence floor, not a reproduction band, because no "
               "outside measurement of this model exists. It is therefore weaker than the "
               "410-million-weight rung's replication anchor, and says so. The floor 0.60 "
               "is the design's own guess, motivated by 0.781031 twice on the "
               "410-million-weight model, 0.744 on a 160-million-weight model of the same "
               "family, and 0.363 on GPT-2. Those three numbers are context, not gates. "
               "The readout is chosen by the same pooled number the gate reads, which is "
               "the committed construct, recorded here plainly; the readout's location is "
               "recorded and never judged. The gate is calibrated on widths 20 through 140 "
               "and then contextualises probe windows out to 280, which the committed "
               "record accepts as a stated scope limit; the width-280 bin check below is "
               "the compensating instrument. The per-readout quality curve, the "
               "width-140-only quality and the width-200 and width-280 qualities are in "
               "the rows below, and none of them is gated. " + RERUN_NOTE),
    )

    # ---- The pattern at each probe window, against its own shuffle draws. ----
    results = {}
    for a in APERTURES:
        info = per_ap[a]
        window_lo = int(info["character_window_low"])
        n_labels = int(info["n_labels"])
        max_delta = int(info["max_delta"])
        z = np.load(labels_file(work_dir, a), allow_pickle=False)
        test_mask = z["is_test"] == 1
        n_rows = int(test_mask.sum())
        x = read_aperture_layer(work_dir, a, anchor_layer, n_rows)
        docs_test = z["doc"][test_mask]
        char_test = z["char_label"][test_mask]
        token_test = z["token_label"][test_mask]

        spectra = Path(out_dir) / ("null_spectra_aperture-%d_labels-characters.jsonl" % a)
        entropy = shuffle_seed_entropy(a, "characters")
        rng = np.random.default_rng(entropy)
        t0 = time.time()
        res = pattern_with_null(x, char_test, docs_test, window_lo, n_labels, max_delta,
                                N_DRAWS, rng, spectra, "characters")
        decision = decide_cleared(res["rank"])
        print("[endpoints] width %d: largest power %.6g at period %.2f characters, rank "
              "%d of %d -> %s; directions at 95 percent %d (%.1f s)"
              % (a, res["observed"]["max_power"], res["observed"]["wavelength"],
                 res["rank"], N_DRAWS, decision, res["ambient_dimension"],
                 time.time() - t0))

        means_name = "mean_vectors_aperture-%d_labels-characters.npz" % a
        np.savez(Path(out_dir) / means_name,
                 labels=np.arange(window_lo, a + 1, dtype=np.int32),
                 counts=np.array(res["counts"], dtype=np.int32),
                 means=res["means"], grand_mean=res["grand_mean"],
                 cosine=np.array(res["cosine"], dtype=np.float64),
                 readout=np.array([anchor_layer]), aperture=np.array([a]))

        pattern_values = {"aperture": a, "readout": anchor_layer,
                          "max_power": res["observed"]["max_power"],
                          "wavelength_chars": res["observed"]["wavelength"],
                          "cycles": res["observed"]["cycles"],
                          "profile_length": res["observed"]["profile_length"],
                          "committed_profile_length": info["committed_profile_length"],
                          "n_periods_in_range": res["observed"]["n_periods_in_range"],
                          "rank": res["rank"], "draws": N_DRAWS,
                          "empirical_share": round(res["rank"] / (N_DRAWS + 1.0), 5),
                          "null_max": max(res["draws"]),
                          "null_median": sorted(res["draws"])[len(res["draws"]) // 2],
                          "character_window_low": window_lo, "character_window_high": a,
                          "n_labels": n_labels,
                          "held_out_rows": n_rows,
                          "positions_per_label_min": min(res["counts"]),
                          "positions_per_label_max": max(res["counts"]),
                          "ambient_dimension_at_95_percent": res["ambient_dimension"],
                          "ambient_dimension_by_variance_target_recorded_only":
                              res["ambient_dimension_by_target"],
                          "variance_share_first_ten":
                              [round(s, 6) for s in res["variance_share"][:10]],
                          "profile": [round(v, 8) for v in res["observed"]["profile"]],
                          "spectrum": res["observed"]["spectrum"],
                          "shuffle_base_seed": SEED,
                          "shuffle_seed_entropy": entropy,
                          "spectra_file": spectra.name,
                          "observed_mean_vectors_file": means_name,
                          "per_draw_mean_vectors_saved": False}
        pattern_values.update(floor_block)
        write_record(
            prov, out_dir,
            base_cell("endpoints", "pattern_vs_shuffle", endpoint=3, aperture=a),
            inputs=record_inputs(prov, stim, design, gate,
                                 extra=[(labels_file(work_dir, a), n_rows),
                                        (spectra, N_DRAWS + 1)]),
            metric="Rank of the observed largest power among 200 character-count label shuffles, at one probe window.",
            values=pattern_values,
            threshold={"rank_at_most": NULL_RANK_MAX, "draws": N_DRAWS},
            decision_rule=RULE_CLEARED,
            decision=decision,
            rows=[{"draw": i, "max_power": v} for i, v in enumerate(res["draws"])],
            notes=("Shuffling permutes the character-count labels inside each held-out "
                   "document, so every label keeps exactly the same number of positions. "
                   "With 200 draws the finest share this can resolve is about 0.5 percent. "
                   "The seed values above all derive from the one committed base seed, 0. "
                   "Every draw's whole spectrum is in the named file beside this record. "
                   "The average vectors of the observed labelling are saved too. The "
                   "average vectors of each draw are not saved, as the previous rung's "
                   "gate record settles. The count of directions at 95 percent feeds the "
                   "pre-registered reading; the counts at 90 and 99 percent above are "
                   "recorded only. " + VOID_NOTE + " " + RERUN_NOTE),
        )
        results[a] = {"char": res, "char_decision": decision,
                      "x": x, "docs": docs_test, "token": token_test,
                      "char_labels": char_test, "n_rows": n_rows}

    # ---- How many directions the count code uses, at the chosen readout. -----
    dims_by_window = {a: results[a]["char"]["ambient_dimension"] for a in APERTURES}
    dimension_decision = decide_dimension(dims_by_window)
    print("[endpoints] directions at 95 percent by window %r -> %s"
          % (dims_by_window, dimension_decision))
    dim_values = {"readout": anchor_layer,
                  "variance_target_gated": VARIANCE_TARGET,
                  "dimension_threshold": DIMENSION_THRESHOLD,
                  "ambient_dimension_at_95_percent_by_aperture":
                      {str(a): dims_by_window[a] for a in APERTURES},
                  "largest_ambient_dimension_at_95_percent": max(dims_by_window.values()),
                  "ambient_dimension_by_variance_target_recorded_only":
                      {str(a): results[a]["char"]["ambient_dimension_by_target"]
                       for a in APERTURES},
                  "variance_share_first_ten_by_aperture":
                      {str(a): [round(s, 6) for s in results[a]["char"]["variance_share"][:10]]
                       for a in APERTURES},
                  "prior_410m_ambient_dimension_by_aperture":
                      {str(k): v for k, v in PRIOR_410M_DIMENSIONS.items()},
                  "published_pattern_dimension": PRIOR_PUBLISHED_DIMENSIONS}
    dim_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "ambient_dimension", endpoint=2),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Directions holding 95 percent of the spread in the per-count average vectors, at the chosen readout, per probe window.",
        values=dim_values,
        threshold={"reaches_at_any_window": DIMENSION_THRESHOLD,
                   "variance_target": VARIANCE_TARGET},
        decision_rule=RULE_DIMENSION,
        decision=dimension_decision,
        notes=("The counting convention is the smallest number of directions reaching 95 "
               "percent of the spread, which is the gate record's answer 1 and every prior "
               "run's convention. The same counts at 90 and at 99 percent are recorded "
               "above and decide nothing; the reading uses 95 only. The threshold of 4 is "
               "the design's own guess, motivated by about 6 directions in the published "
               "pattern against 1 to 3 in the 410-million-weight model. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- The token-position control, one record per probe window. -------------
    variants_path = Path(out_dir) / "rows_variants.jsonl"
    if not variants_path.exists():
        raise Halt("no per-variant rows at %s; the extract stage must run first"
                   % variants_path)
    chars_by_width = {a: 0 for a in APERTURES}
    with open(variants_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == "test" and row["width"] in chars_by_width:
                chars_by_width[row["width"]] += row["n_chars"]

    for a in APERTURES:
        info = per_ap[a]
        res = results[a]
        obs_wl = res["char"]["observed"]["wavelength"]
        char_cleared = res["char_decision"] == "NULL_CLEARED"
        token_hi = int(info["token_window_high"])
        tok_max_delta = int(info["token_max_delta"])
        n_rows = res["n_rows"]
        chars_per_token = chars_by_width[a] / float(n_rows) if n_rows else None
        vals = {"aperture": a, "readout": anchor_layer,
                "token_window_high": token_hi,
                "token_ceiling": info["token_ceiling"],
                "token_label_observed_max": info["token_label_observed_max"],
                "token_labels_excluded_above_ceiling":
                    info["token_labels_excluded_above_ceiling"],
                "token_ceiling_min_training_positions":
                    info["token_ceiling_min_training_positions"],
                "token_max_delta": tok_max_delta,
                "held_out_characters": chars_by_width[a],
                "held_out_positions": n_rows,
                "chars_per_token": (round(chars_per_token, 4) if chars_per_token else None),
                "character_wavelength_chars": obs_wl,
                "character_cleared_shuffle": char_cleared,
                "character_rank": res["char"]["rank"],
                "shuffle_base_seed": SEED,
                "per_draw_mean_vectors_saved": False}
        note = ["The characters-per-token rate is measured on this probe window's own "
                "held-out documents, because the tokenisation shifts a little with the "
                "wrap width."]
        tok_wl_chars = None
        tok_cleared = None
        try:
            if not chars_per_token:
                raise Halt("no held-out positions at this probe window")
            spectra = Path(out_dir) / ("null_spectra_aperture-%d_labels-tokens.jsonl" % a)
            entropy = shuffle_seed_entropy(a, "tokens")
            rng = np.random.default_rng(entropy)
            t0 = time.time()
            tres = pattern_with_null(res["x"], res["token"], res["docs"], 1, token_hi,
                                     tok_max_delta, N_DRAWS, rng, spectra, "tokens")
            tok_cleared = decide_cleared(tres["rank"]) == "NULL_CLEARED"
            tok_wl_chars = tres["observed"]["wavelength"] * chars_per_token
            tok_means_name = "mean_vectors_aperture-%d_labels-tokens.npz" % a
            np.savez(Path(out_dir) / tok_means_name,
                     labels=np.arange(1, token_hi + 1, dtype=np.int32),
                     counts=np.array(tres["counts"], dtype=np.int32),
                     means=tres["means"], grand_mean=tres["grand_mean"],
                     cosine=np.array(tres["cosine"], dtype=np.float64),
                     readout=np.array([anchor_layer]), aperture=np.array([a]))
            vals.update({
                "token_max_power": tres["observed"]["max_power"],
                "token_wavelength_tokens": tres["observed"]["wavelength"],
                "token_wavelength_chars": round(tok_wl_chars, 4),
                "token_cycles": tres["observed"]["cycles"],
                "token_profile_length": tres["observed"]["profile_length"],
                "token_n_periods_in_range": tres["observed"]["n_periods_in_range"],
                "token_shortest_period_searched_tokens": PERIOD_MIN,
                "token_rank": tres["rank"], "draws": N_DRAWS,
                "token_empirical_share": round(tres["rank"] / (N_DRAWS + 1.0), 5),
                "token_null_max": max(tres["draws"]),
                "token_positions_per_label_min": min(tres["counts"]),
                "token_positions_per_label_max": max(tres["counts"]),
                "token_ambient_dimension_at_95_percent": tres["ambient_dimension"],
                "token_ambient_dimension_by_variance_target_recorded_only":
                    tres["ambient_dimension_by_target"],
                "token_cleared_shuffle": tok_cleared,
                "token_shuffle_seed_entropy": entropy,
                "token_spectra_file": spectra.name,
                "token_observed_mean_vectors_file": tok_means_name})
            note.append("The token period is turned into characters with that measured "
                        "rate, because the design's margin of 3 is in characters.")
            note.append("The shortest period of 4 is applied in each labelling's own "
                        "units. So this control searches token periods from 4 tokens up "
                        "to its own profile length, and never reuses the character "
                        "numbers.")
            note.append("The token ceiling above is the committed rule of correction 3: "
                        "at least 40 training positions. It is recorded beside the "
                        "observed maximum token count, so the size of the cut is visible.")
            print("[endpoints] width %d token control: period %.2f tokens, %.2f characters, "
                  "rank %d of %d, ceiling %d against an observed maximum of %d (%.1f s)"
                  % (a, tres["observed"]["wavelength"], tok_wl_chars, tres["rank"],
                     N_DRAWS, token_hi, info["token_label_observed_max"],
                     time.time() - t0))
        except Halt as exc:
            vals["not_computable_reason"] = str(exc)
            vals["token_cleared_shuffle"] = None
            note.append("The token-position control could not be computed: %s" % exc)
        attribution = decide_attribution(obs_wl, tok_wl_chars, char_cleared, tok_cleared)
        note.append("One line at this width holds far fewer tokens than characters, so the "
                    "two labellings cannot range over the same number of periods.")
        note.append("Both labellings' ranks and periods are recorded here whatever the "
                    "character result was.")
        note.append("Two absent patterns can never be told apart, so neither labelling "
                    "clearing reads NO_PATTERN_TO_ATTRIBUTE, which corrects the previous "
                    "rung. A token-only clearing reads TOKEN_ONLY_PATTERN and does not "
                    "enter the invariance fork.")
        note.append(VOID_NOTE)
        note.append(RERUN_NOTE)
        vals.update(floor_block)
        print("[endpoints] width %d labelling attribution -> %s" % (a, attribution))
        write_record(
            prov, out_dir,
            base_cell("endpoints", "labelling_attribution", endpoint=4, aperture=a),
            inputs=record_inputs(prov, stim, design, gate,
                                 extra=[(labels_file(work_dir, a), n_rows)]),
            metric="Which labelling, character-count or token-position, clears its own 200 shuffles at one probe window, and at what period.",
            values=vals,
            threshold={"margin_chars": CONFOUND_MARGIN_CHARS,
                       "rank_at_most": NULL_RANK_MAX},
            decision_rule=RULE_ATTRIBUTION,
            decision=attribution,
            notes=" ".join(note),
        )
        results[a]["attribution"] = attribution
        results[a]["token_cleared"] = tok_cleared

    # ---- The two bin-correlation checks. -------------------------------------
    # One fit at the chosen readout serves both. The width-280 check has thresholds and
    # a named reading. The width-140 check is recorded only, with no threshold at all.
    g, v, c, ls, lq, n = sum_groups(tot, TRAIN_GROUPS, anchor_layer)
    w, off = ridge_fit_from_stats(g, v, c, ls, n, alpha)

    a280 = 280
    z = np.load(labels_file(work_dir, a280), allow_pickle=False)
    test_mask = z["is_test"] == 1
    labels_280 = z["char_label"][test_mask].astype(np.float64)
    pred_280 = results[a280]["x"] @ w + off
    bin_rows, low_corrs, high_corrs = bin_correlation_rows(
        pred_280, labels_280, label_bins(), with_roles=True)
    range_decision = decide_range(low_corrs, high_corrs)
    print("[endpoints] bin correlations at width 280 -> %s" % range_decision)
    range_values = {"readout": anchor_layer, "ridge_strength": alpha,
                    "bin_width": BIN_WIDTH, "n_bins": N_BINS,
                    "statistic": "Pearson correlation between predicted and true count",
                    "low_bin_correlations":
                        [None if r is None else round(float(r), 6) for r in low_corrs],
                    "high_bin_correlations":
                        [None if r is None else round(float(r), 6) for r in high_corrs],
                    "excluded_bin": "141 to 175",
                    "low_bin_definition": "ends at or below label %d" % LOW_BIN_MAX_LABEL,
                    "high_bin_definition": "starts above label %d" % HIGH_BIN_MIN_LABEL,
                    "measurable_floor": RANGE_MEASURABLE_FLOOR,
                    "low_floor": RANGE_LOW_FLOOR, "high_ceiling": RANGE_HIGH_CEIL}
    range_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "measurable_range", endpoint=5, aperture=a280),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=[(labels_file(work_dir, a280), int(test_mask.sum()))]),
        metric="Pearson correlation between the main fit's predicted count and the true count, inside each bin of 35 label values at width 280.",
        values=range_values,
        threshold={"any_high_bin_at_least": RANGE_MEASURABLE_FLOOR,
                   "every_low_bin_at_least": RANGE_LOW_FLOOR,
                   "every_high_bin_below": RANGE_HIGH_CEIL},
        decision_rule=RULE_RANGE,
        decision=range_decision,
        rows=bin_rows,
        notes=("A correlation inside a bin, rather than a fit quality inside a bin, is "
               "what the design commits to here. A global fit's within-bin fit quality is "
               "bounded by one minus the squared ratio of the global residual spread to "
               "the bin's own spread, which for these numbers could not reach its own "
               "thresholds; the correlation has no such bound. The bin from 141 to 175 "
               "holds labels on both sides of 175, so it is recorded and left out of this "
               "decision. The decision uses the four bins ending at or below label 140 "
               "against the three bins starting above label 175. This check exists at "
               "width 280 only, as the design says, so it cannot classify a null result at "
               "width 200 on its own. It is the committed compensating instrument for the "
               "gate being calibrated on widths 20 through 140: its reading decides "
               "whether wide-window nulls count as evidence at all. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    a140 = 140
    z140 = np.load(labels_file(work_dir, a140), allow_pickle=False)
    test_mask_140 = z140["is_test"] == 1
    labels_140 = z140["char_label"][test_mask_140].astype(np.float64)
    pred_140 = results[a140]["x"] @ w + off
    bin_rows_140, _, _ = bin_correlation_rows(
        pred_140, labels_140, label_bins_140(), with_roles=False)
    print("[endpoints] bin correlations at width 140 (recorded only): %r"
          % [r["pearson_correlation"] for r in bin_rows_140])
    bins140_values = {"readout": anchor_layer, "ridge_strength": alpha,
                      "bin_width": BIN_WIDTH, "n_bins": N_BINS_140,
                      "statistic": "Pearson correlation between predicted and true count",
                      "bin_correlations":
                          [r["pearson_correlation"] for r in bin_rows_140],
                      "bin_edges": ["%d to %d" % (lo, hi) for lo, hi in label_bins_140()],
                      "prior_410m_width_280_low_bins": [0.648, 0.219, 0.153, 0.271]}
    bins140_values.update(floor_block)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "bin_correlations_width_140", aperture=a140),
        inputs=record_inputs(prov, stim, design, gate,
                             extra=[(labels_file(work_dir, a140),
                                     int(test_mask_140.sum()))]),
        metric="Pearson correlation between the main fit's predicted count and the true count, inside each bin of 35 label values across characters 1 to 140.",
        values=bins140_values,
        threshold={},
        decision_rule=RULE_BINS_140,
        decision="RECORDED_NO_DECISION",
        rows=bin_rows_140,
        notes=("The 410-million-weight rung found fine resolution collapsing early even "
               "inside its fitted range, at width 280: 0.648 then 0.219, 0.153 and 0.271. "
               "Whether the same collapse holds at width 140 bears on whether a fine-scale "
               "pattern was ever measurable, and no run had recorded it. " +
               RECORDED_ONLY_NOTE + " " + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- Whether anything cleared, under either labelling. -------------------
    cleared = [(a, results[a]["char"]["observed"]["wavelength"]) for a in APERTURES
               if results[a]["char_decision"] == "NULL_CLEARED"]
    token_cleared_windows = [a for a in APERTURES if results[a].get("token_cleared")]
    any_decision = decide_any_cleared(len(cleared), len(token_cleared_windows))
    any_values = {"n_character_windows_cleared": len(cleared),
                  "character_windows_cleared": [a for a, _ in cleared],
                  "n_token_windows_cleared": len(token_cleared_windows),
                  "token_windows_cleared": token_cleared_windows,
                  "per_aperture_character_decision":
                      {str(a): results[a]["char_decision"] for a in APERTURES},
                  "per_aperture_character_rank":
                      {str(a): results[a]["char"]["rank"] for a in APERTURES},
                  "per_aperture_character_wavelength_chars":
                      {str(a): results[a]["char"]["observed"]["wavelength"]
                       for a in APERTURES},
                  "per_aperture_token_cleared":
                      {str(a): results[a].get("token_cleared") for a in APERTURES},
                  "per_aperture_attribution":
                      {str(a): results[a]["attribution"] for a in APERTURES},
                  "width_280_bin_reading": range_decision}
    any_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "any_window_cleared", endpoint=3),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Whether any of the three probe windows cleared its own 200 shuffle draws, under either labelling.",
        values=any_values,
        threshold={"rank_at_most": NULL_RANK_MAX, "draws": N_DRAWS,
                   "windows": APERTURES},
        decision_rule=RULE_ANY_CLEARED,
        decision=any_decision,
        notes=("Three windows tested at 5 percent each carry roughly a 14 percent chance "
               "that at least one clears by luck alone. The headline reading absorbs that, "
               "because it conditions on which windows cleared and under which labelling. "
               "The width-280 bin reading is carried above because it decides whether a "
               "null at the wide windows counts as evidence at all. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- The previous rung's fork, over character-count clearings only. ------
    gap, ratio_gap = pairwise_gaps(cleared)
    gap_decision = decide_wavelength_gap(gap)
    gap_values = {"largest_gap_chars": (None if gap is None else round(float(gap), 4)),
                  "cleared_apertures": [a for a, _ in cleared],
                  "wavelengths_chars": {str(a): wl for a, wl in cleared}}
    gap_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "wavelength_gap"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Largest gap in characters between the pattern periods of any two probe windows that both cleared.",
        values=gap_values,
        threshold={"at_most_chars": INVARIANCE_MAX_GAP_CHARS},
        decision_rule=RULE_WAVELENGTH_GAP,
        decision=gap_decision,
        notes=("A period drifting by, say, 12 characters would be called not fixed here, "
               "even if it were one mechanism plus noise. The 10-character cutoff is the "
               "previous rung's own guess and is binding. This is one half of that rung's "
               "fork, which this design reuses and does not redefine. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    ratio_decision = decide_ratio_gap(ratio_gap)
    ratio_values = {"largest_ratio_gap":
                        (None if ratio_gap is None else round(float(ratio_gap), 6)),
                    "cleared_apertures": [a for a, _ in cleared],
                    "ratios": {str(a): round(wl / float(a), 6) for a, wl in cleared}}
    ratio_values.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "ratio_gap"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="Largest gap between period-to-width ratios of any two probe windows that both cleared.",
        values=ratio_values,
        threshold={"at_most": PROPORTION_MAX_RATIO_GAP},
        decision_rule=RULE_RATIO_GAP,
        decision=ratio_decision,
        notes=("A mixed, unclassifiable result needs both this gap above 0.1 and the "
               "period gap above 10 characters. Both cutoffs are the previous rung's own "
               "guesses and are binding. This is the other half of that rung's fork. "
               + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- One call for the whole run, from the calls already made. ------------
    setup_path = Path(out_dir) / (prov.cell_slug(base_cell("extract", "setup_check")) + ".json")
    setup_decision = "RECORD_ABSENT"
    if setup_path.exists():
        with open(setup_path, "r", encoding="utf-8") as fh:
            setup_decision = json.load(fh).get("decision", "RECORD_UNREADABLE")
    headline = decide_headline(floor_decision, cleared, bool(token_cleared_windows),
                               dimension_decision)
    summary = {
        "setup_check": setup_decision,
        "calibration": calib.get("decision"),
        "endpoint_1_floor_gate": floor_decision,
        "endpoint_2_ambient_dimension": dimension_decision,
        "endpoint_3_pattern_vs_shuffle":
            {str(a): results[a]["char_decision"] for a in APERTURES},
        "endpoint_3_any_window_cleared": any_decision,
        "endpoint_4_labelling_attribution":
            {str(a): results[a]["attribution"] for a in APERTURES},
        "endpoint_5_bin_correlations_width_280": range_decision,
        "recorded_only_bin_correlations_width_140":
            [r["pearson_correlation"] for r in bin_rows_140],
        "recorded_only_ambient_dimension_by_variance_target":
            {str(a): results[a]["char"]["ambient_dimension_by_target"] for a in APERTURES},
        "invariance_fork_wavelength_gap": gap_decision,
        "invariance_fork_ratio_gap": ratio_decision,
        "chosen_readout": anchor_layer,
        "held_out_quality_pooled_widths_20_to_140": round(anchor_pooled, 6),
        "held_out_quality_w140_only": best["held_out_quality_w140"],
        "held_out_quality_w200": best["held_out_quality_w200"],
        "held_out_quality_w280": best["held_out_quality_w280"],
        "floor": FLOOR_R2,
        "ridge_strength": alpha,
        "ridge_argmax_interior_on_grid": cv["ridge_argmax_interior_on_grid"],
        "n_character_windows_cleared": len(cleared),
        "n_token_windows_cleared": len(token_cleared_windows),
        "wavelengths_chars": {str(a): results[a]["char"]["observed"]["wavelength"]
                              for a in APERTURES},
        "ambient_dimension_at_95_percent_by_aperture":
            {str(a): dims_by_window[a] for a in APERTURES},
        "dimension_threshold": DIMENSION_THRESHOLD,
        "largest_gap_chars": (None if gap is None else round(float(gap), 4)),
        "largest_ratio_gap": (None if ratio_gap is None else round(float(ratio_gap), 6)),
        "token_ceiling_by_aperture":
            {str(a): per_ap[a]["token_ceiling"] for a in APERTURES},
        "token_label_observed_max_by_aperture":
            {str(a): per_ap[a]["token_label_observed_max"] for a in APERTURES},
        "generation_performed": False,
        "single_run_not_replicated": True,
        "headline": headline,
    }
    summary.update(floor_block)
    write_record(
        prov, out_dir, base_cell("endpoints", "run_summary"),
        inputs=record_inputs(prov, stim, design, gate),
        metric="One call per endpoint, and the pre-registered headline reading, from the records written beside this one.",
        values=summary,
        threshold={},
        decision_rule=RULE_HEADLINE,
        decision=headline,
        notes=("The four pre-registered readings are CURVE_TOO_WEAK, "
               "PRECONDITION_ABSENT_AT_1P4B, PRECONDITION_WITHOUT_RIPPLE, and the previous "
               "rung's fork. A failed floor makes every pattern number here "
               "uninterpretable as evidence about the model, so the whole rung is void in "
               "that case and the next rung is a human decision. A clearing under the "
               "token-position labelling alone reads TOKEN_ONLY_PATTERN and does not enter "
               "the fork. Exactly one clearing window is left undetermined by design, not "
               "decided. This rung is a single run, unlike the twice-measured rung below "
               "it, and could land differently on a repeat elsewhere. " + RERUN_NOTE),
    )
    print("[endpoints] summary %s" % json.dumps(summary))
    return 0


# ---------------------------------------------------------------------------
# Part 12. Self-test. Standard library only. No model, no graphics card. Heavy
# packages are imported inside guards, so a missing one skips its block rather
# than failing the run.
# ---------------------------------------------------------------------------

def _check(fails, name, got, want, tol=0.0):
    if isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print("  %-64s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def selftest() -> int:
    fails = []

    print("== position labeller (the reasoning-tag parser of this experiment) ==")
    text = "ab\ncd"
    spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    ch, tk, ln = labels_for_spans(text, spans)
    _check(fails, "columns on a two-line text", ch, [1, 2, 3, 1, 2])
    _check(fails, "token counts on a two-line text", tk, [1, 2, 3, 1, 2])
    _check(fails, "line numbers on a two-line text", ln, [0, 0, 0, 1, 1])
    # The opening boundary of the first line is implied, not present in the text. That
    # is the same shape as a chat template that already emitted the opening marker, so
    # only the closing marker appears in what the model produced.
    ch1, tk1, _ = labels_for_spans("abcd", [(0, 2), (2, 4)])
    _check(fails, "opening boundary implied, never present: columns", ch1, [2, 4])
    _check(fails, "opening boundary implied, never present: token counts", tk1, [1, 2])
    ch1b, tk1b, ln1b = labels_for_spans("abcd\nef", [(0, 2), (2, 4), (4, 5), (5, 7)])
    _check(fails, "only the closing boundary appears: columns", ch1b, [2, 4, 5, 2])
    _check(fails, "only the closing boundary appears: token counts", tk1b, [1, 2, 3, 1])
    _check(fails, "only the closing boundary appears: line numbers", ln1b, [0, 0, 0, 1])
    # A line break swallowed inside one token: the boundary sits inside a unit.
    ch2, tk2, ln2 = labels_for_spans("ab\ncd", [(0, 2), (2, 4), (4, 5)])
    _check(fails, "swallowed line break, columns", ch2, [2, 1, 2])
    _check(fails, "swallowed line break, token counts", tk2, [1, 1, 2])
    _check(fails, "swallowed line break, line numbers", ln2, [0, 1, 1])
    # A line break as its own token takes the column after the last visible character,
    # which is the width plus one, and every label window ends at the width itself.
    ch3, _, _ = labels_for_spans("abc\nd", [(0, 3), (3, 4), (4, 5)])
    _check(fails, "line-break token takes the next column", ch3, [3, 4, 1])
    ch5, _, _ = labels_for_spans("abcd\ne", [(0, 4), (4, 5), (5, 6)])
    _check(fails, "line-break token at width plus one", ch5, [4, 5, 1])
    _check(fails, "the width-plus-one label falls outside a window ending at the width",
           label_slot(5, 1, 4), None)
    ch4, tk4, _ = labels_for_spans("ab", [(0, 0), (0, 2)])
    _check(fails, "empty range is unlabelled", ch4, [None, 2])
    _check(fails, "empty range is not counted", tk4, [None, 1])

    print("== degenerate-input detector ==")
    _check(fails, "length-cap case flags", degenerate_flags(2048, 2048, ["a", "b"]),
           ["token_cap"])
    _check(fails, "under the length cap does not flag",
           degenerate_flags(700, 2048, ["a", "b", "c"]), [])
    rep = ["x", "y"] * 7
    _check(fails, "repeated-sequence case flags", degenerate_flags(14, 2048, rep),
           ["repeated_sequence"])
    _check(fails, "seven back-to-back copies counted", max_block_repeats(rep), 7)
    _check(fails, "five back-to-back copies is below the cutoff",
           degenerate_flags(10, 2048, ["x", "y"] * 5), [])
    _check(fails, "prose-like tokens are clean",
           degenerate_flags(9, 2048, ["It", " will", " be", " seen", " that", " this",
                                      " mere", " b", "\n"]), [])
    _check(fails, "both cases can flag together",
           degenerate_flags(2048, 2048, ["q"] * 6), ["token_cap", "repeated_sequence"])

    print("== exact-match checks and exact slot assignment ==")
    _check(fails, "hash equal", exact_hash_match("a" * 64, "a" * 64), True)
    _check(fails, "hash one character off", exact_hash_match("a" * 63 + "b", "a" * 64), False)
    _check(fails, "hash absent", exact_hash_match(None, "a" * 64), False)
    _check(fails, "configuration match at 24 layers and hidden 2048",
           exact_config_match(24, 2048), True)
    _check(fails, "configuration layer mismatch", exact_config_match(23, 2048), False)
    _check(fails, "the previous rung's hidden size is now a mismatch",
           exact_config_match(24, 1024), False)
    _check(fails, "lowest window value maps to slot 0", label_slot(1, 1, 140), 0)
    _check(fails, "highest window value maps to the last slot", label_slot(140, 1, 140), 139)
    _check(fails, "one below the window is outside", label_slot(0, 1, 140), None)
    _check(fails, "one above the window is outside", label_slot(141, 1, 140), None)
    _check(fails, "a fractional label is never accepted", label_slot(1.0, 1, 140), None)
    _check(fails, "a true value is never accepted as 1", label_slot(True, 1, 140), None)
    counts, slots = slot_counts([1, 2, 2, 5, 0], 1, 4)
    _check(fails, "slot counts", counts, [1, 2, 0, 0])
    _check(fails, "slot indices", slots, [0, 1, 1, None, None])
    _check(fails, "window start when the lowest value is thin",
           window_from_counts({1: 3, 2: 20, 3: 20}, 3, 20), 2)
    _check(fails, "window start when every value is thick",
           window_from_counts({1: 20, 2: 20, 3: 20}, 3, 20), 1)
    _check(fails, "no window when a middle value is thin",
           window_from_counts({1: 20, 2: 1, 3: 20}, 3, 20, max_start=1), None)

    print("== the committed token ceiling: at least 40 training positions ==")
    _check(fails, "token window stops at the first thin value",
           token_window_top({1: 9, 2: 9, 3: 1}, 5), 2)
    planted = {v: 100 for v in range(1, 42)}
    planted.update({v: 5 for v in range(42, 51)})
    rep_ceiling = token_ceiling_report(planted)
    _check(fails, "the ceiling is the highest count reaching 40 positions",
           rep_ceiling["token_ceiling"], 41)
    _check(fails, "the observed maximum is recorded beside it",
           rep_ceiling["token_label_observed_max"], 50)
    _check(fails, "the size of the cut is recorded",
           rep_ceiling["token_labels_excluded_above_ceiling"], 9)
    _check(fails, "the rule's own minimum is recorded",
           rep_ceiling["token_ceiling_min_training_positions"], 40)
    _check(fails, "positions just above the ceiling are recorded",
           rep_ceiling["token_positions_one_above_ceiling"], 5)
    _check(fails, "exactly 40 positions is inside the ceiling",
           token_ceiling_report({1: 40, 2: 40, 3: 39})["token_ceiling"], 2)
    _check(fails, "no count reaching 40 leaves an empty window",
           token_ceiling_report({1: 39})["token_ceiling"], 0)
    _check(fails, "the committed minimum is 40", TOKEN_WINDOW_MIN_TRAIN, 40)

    print("== the committed profile length: the label count minus 20 ==")
    _check(fails, "profile length at 140 labels", profile_length_for(140), 120)
    _check(fails, "profile length at 200 labels", profile_length_for(200), 180)
    _check(fails, "profile length at 280 labels", profile_length_for(280), 260)
    _check(fails, "the rule reproduces every committed number",
           [profile_length_for(a) == COMMITTED_PROFILE_LENGTH[a] for a in APERTURES],
           [True, True, True])
    _check(fails, "a shorter window no longer matches the committed number",
           profile_length_for(139) == COMMITTED_PROFILE_LENGTH[140], False)

    print("== label bins of both bin-correlation checks ==")
    bins = label_bins()
    _check(fails, "eight bins at width 280", len(bins), 8)
    _check(fails, "first bin", bins[0], (1, 35))
    _check(fails, "fourth bin ends at 140", bins[3], (106, 140))
    _check(fails, "fifth bin straddles 175", bins[4], (141, 175))
    _check(fails, "sixth bin starts at 176", bins[5], (176, 210))
    _check(fails, "last bin ends at 280", bins[7], (246, 280))
    _check(fails, "roles", [bin_role(lo, hi) for lo, hi in bins],
           ["low", "low", "low", "low", "straddles", "high", "high", "high"])
    _check(fails, "four low bins", sum(1 for lo, hi in bins if bin_role(lo, hi) == "low"), 4)
    _check(fails, "three high bins", sum(1 for lo, hi in bins if bin_role(lo, hi) == "high"), 3)
    _check(fails, "the excluded bin is the only straddling one",
           sum(1 for lo, hi in bins if bin_role(lo, hi) == "straddles"), 1)
    _check(fails, "four bins across characters 1 to 140", label_bins_140(),
           [(1, 35), (36, 70), (71, 105), (106, 140)])
    _check(fails, "the recorded-only bins are the low bins of the wide check",
           label_bins_140(), bins[:4])

    print("== shuffle seed streams, all from base seed 0 ==")
    _check(fails, "characters at width 140", shuffle_seed_entropy(140, "characters"), [0, 140, 0])
    _check(fails, "tokens at width 140", shuffle_seed_entropy(140, "tokens"), [0, 140, 1])
    _check(fails, "characters at width 280", shuffle_seed_entropy(280, "characters"), [0, 280, 0])
    streams = [shuffle_seed_entropy(a, u) for a in APERTURES for u in sorted(LABELLING_CODES)]
    _check(fails, "six distinct streams", len(set(tuple(s) for s in streams)), 6)
    _check(fails, "every stream starts from the committed base seed",
           all(s[0] == SEED for s in streams), True)
    try:
        shuffle_seed_entropy(140, "words")
        fails.append("an unknown labelling should halt")
        print("  %-64s FAIL no halt" % "an unknown labelling halts")
    except Halt:
        print("  %-64s ok" % "an unknown labelling halts")

    print("== which text variants enter the fit ==")
    _check(fails, "training text at width 140 enters", in_training_fit("train", 140), True)
    _check(fails, "training text at width 200 stays out", in_training_fit("train", 200), False)
    _check(fails, "training text at width 280 stays out", in_training_fit("train", 280), False)
    _check(fails, "held-out text at width 280 is kept", in_training_fit("test", 280), True)
    _check(fails, "the seven selection widths", SELECT_WIDTHS,
           [20, 40, 60, 80, 100, 120, 140])
    _check(fails, "the two widths never trained on or gated", WIDE_WIDTHS, [200, 280])
    _check(fails, "the seven held-out groups pooled for the gate",
           SELECT_TEST_GROUPS, ["test_w20", "test_w40", "test_w60", "test_w80",
                                "test_w100", "test_w120", "test_w140"])

    print("== the memory bill of the running totals at hidden 2048 ==")
    _check(fails, "one group of totals is about 840 megabytes",
           round(totals_bytes(n_groups=1) / 1e6), 840)
    _check(fails, "fourteen groups are about 11.8 gigabytes",
           round(totals_bytes() / 1e9, 1), 11.8, 0.05)
    _check(fails, "the totals are four times the previous rung's hidden 1024",
           round(totals_bytes() / totals_bytes(hidden=1024)), 4)

    print("== repeating-pattern statistic, no search band ==")
    n = 40
    want = [1.0 / (1 + d) for d in range(n)]
    mat = [[want[abs(i - j)] for j in range(n)] for i in range(n)]
    prof = offdiag_profile(mat, 20)
    _check(fails, "profile at distance 1", prof[0], want[1], 1e-12)
    _check(fails, "profile at distance 20", prof[19], want[20], 1e-12)
    line = [3.0 + 0.5 * i for i in range(60)]
    _check(fails, "a straight line detrends to zero",
           max(abs(v) for v in poly_detrend(line, 2)), 0.0, 1e-9)
    curve = [1.0 - 0.01 * i + 0.0002 * i * i for i in range(60)]
    _check(fails, "a curve of degree 2 detrends to zero",
           max(abs(v) for v in poly_detrend(curve, 2)), 0.0, 1e-9)
    wave = [math.cos(2 * math.pi * i / 12.0) for i in range(60)]
    power = dft_power(poly_detrend(wave, 2))
    peak = max(power, key=lambda k: power[k])
    _check(fails, "a 12-step wave peaks at 5 cycles over 60 steps", peak, 5)
    _check(fails, "that peak is a period of 12", 60 / peak, 12.0, 1e-9)
    _check(fails, "the period range reaches the profile length",
           periods_in_range(120)[0], (1, 120.0))
    _check(fails, "the period range stops at 4", periods_in_range(120)[-1], (30, 4.0))
    _check(fails, "periods below 4 are excluded",
           all(p >= 4.0 for _, p in periods_in_range(121)), True)
    # The token control searches its own units. Its profile is far shorter, so its
    # period range is a different set of numbers, never the character one.
    _check(fails, "a token profile of length 21 ranges over 5 periods",
           len(periods_in_range(21)), 5)
    _check(fails, "its longest period is 21 tokens", periods_in_range(21)[0][1], 21.0, 1e-9)
    _check(fails, "its shortest period is at least 4 tokens",
           periods_in_range(21)[-1][1] >= 4.0, True)
    _check(fails, "the character range at 120 is a different set",
           len(periods_in_range(120)) == len(periods_in_range(21)), False)
    # A distance-dependent matrix carrying a known period recovers that period.
    n = 160
    entries = [1.0 - 0.004 * d + 0.05 * math.cos(2 * math.pi * d / 40.0) for d in range(n)]
    mat = [[entries[abs(i - j)] for j in range(n)] for i in range(n)]
    stat = max_power_stat(mat, 120)
    _check(fails, "planted period of 40 recovered", stat["wavelength"], 40.0, 1e-9)
    _check(fails, "whole spectrum returned", len(stat["spectrum"]), 60)
    _check(fails, "periods ranged over", stat["n_periods_in_range"], 30)
    # The profile length itself is reachable. The planted single cycle starts at zero
    # and rises, because a single cycle that starts at its peak is shaped like a bowl,
    # and the degree-2 detrend the design fixes absorbs most of a bowl. That is a
    # property of the fixed detrend, checked in the next lines rather than worked around.
    entries = [1.0 + 0.05 * math.sin(2 * math.pi * d / 120.0) for d in range(n)]
    mat = [[entries[abs(i - j)] for j in range(n)] for i in range(n)]
    stat = max_power_stat(mat, 120)
    _check(fails, "a period equal to the profile length is reachable",
           stat["wavelength"], 120.0, 1e-9)
    _check(fails, "that period is one cycle", stat["cycles"], 1)
    entries = [1.0 + 0.05 * math.cos(2 * math.pi * d / 120.0) for d in range(n)]
    mat = [[entries[abs(i - j)] for j in range(n)] for i in range(n)]
    _check(fails, "a bowl-shaped single cycle is largely absorbed by the detrend",
           max_power_stat(mat, 120)["cycles"], 2)
    try:
        max_power_stat(mat, 3)
        fails.append("too short a profile should halt")
        print("  %-64s FAIL no halt" % "too short a profile halts")
    except Halt:
        print("  %-64s ok" % "too short a profile halts")

    print("== endpoint arithmetic on planted rows with known calls ==")
    _check(fails, "setup, all three match", decide_setup(True, 24, 2048), "SETUP_OK")
    _check(fails, "setup, hash differs", decide_setup(False, 24, 2048), "SETUP_FAILURE")
    _check(fails, "setup, layer count differs", decide_setup(True, 12, 2048), "SETUP_FAILURE")
    _check(fails, "setup, width differs", decide_setup(True, 24, 1024), "SETUP_FAILURE")

    print("-- endpoint 1, the existence floor at 0.60 --")
    _check(fails, "floor, exactly 0.60", decide_floor(0.60), "FLOOR_CLEARED")
    _check(fails, "floor, just under 0.60", decide_floor(0.5999), "CURVE_TOO_WEAK")
    _check(fails, "floor, the 410-million-weight value 0.781031",
           decide_floor(PRIOR_410M_POOLED), "FLOOR_CLEARED")
    _check(fails, "floor, the 160-million-weight reproduction 0.744",
           decide_floor(PRIOR_REPRODUCTION_PYTHIA_160M), "FLOOR_CLEARED")
    _check(fails, "floor, the GPT-2 reproduction 0.363",
           decide_floor(PRIOR_REPRODUCTION_GPT2), "CURVE_TOO_WEAK")
    _check(fails, "floor, below zero", decide_floor(-0.2), "CURVE_TOO_WEAK")
    _check(fails, "floor, absent", decide_floor(None), "CURVE_TOO_WEAK")
    _check(fails, "floor, not a number", decide_floor(float("nan")), "CURVE_TOO_WEAK")
    _check(fails, "the floor is not a two-sided band: a very high value still clears",
           decide_floor(0.95), "FLOOR_CLEARED")

    print("-- the floor gate's decision inside another record's values --")
    passed = floor_values("FLOOR_CLEARED", 0.72, 12)
    failed = floor_values("CURVE_TOO_WEAK", 0.42, 5)
    _check(fails, "a passing gate is carried", passed["floor_gate_decision"],
           "FLOOR_CLEARED")
    _check(fails, "a passing gate marks the record usable",
           passed["this_record_voided_by_floor_gate"], False)
    _check(fails, "a failing gate marks the record void",
           failed["this_record_voided_by_floor_gate"], True)
    _check(fails, "the gate's own value travels with it",
           failed["floor_gate_quality_pooled_widths_20_to_140"], 0.42, 1e-12)
    _check(fails, "the floor travels with it", passed["floor_gate_floor"], 0.60, 1e-12)
    _check(fails, "the readout travels with it", failed["floor_gate_readout"], 5)

    print("-- endpoint 2, directions at 95 percent reaching 4 --")
    _check(fails, "the 410-million-weight dimensions 2, 3, 1 stay below 4",
           decide_dimension({140: 2, 200: 3, 280: 1}), "DIMENSION_BELOW_4")
    _check(fails, "exactly 3 everywhere stays below 4",
           decide_dimension({140: 3, 200: 3, 280: 3}), "DIMENSION_BELOW_4")
    _check(fails, "exactly 4 at one window reaches the threshold",
           decide_dimension({140: 1, 200: 4, 280: 2}), "DIMENSION_REACHES_4")
    _check(fails, "the published pattern's 6 directions reach it",
           decide_dimension({140: 6}), "DIMENSION_REACHES_4")
    _check(fails, "nothing computed is not a call",
           decide_dimension({}), "DIMENSION_NOT_COMPUTABLE")
    _check(fails, "the threshold is 4", DIMENSION_THRESHOLD, 4)
    _check(fails, "the gated variance target is 95 percent", VARIANCE_TARGET, 0.95, 1e-12)
    _check(fails, "the recorded-only targets are 90 and 99 percent",
           VARIANCE_TARGETS_RECORDED, [0.90, 0.99])
    share = [0.5, 0.3, 0.15, 0.04, 0.01]
    _check(fails, "directions at 90 percent on a planted spread",
           dimension_at(share, 0.90), 3)
    _check(fails, "directions at 95 percent on the same spread",
           dimension_at(share, 0.95), 3)
    _check(fails, "directions at 99 percent on the same spread",
           dimension_at(share, 0.99), 4)
    share2 = [0.45, 0.3, 0.16, 0.05, 0.035, 0.005]
    _check(fails, "the convention can move the answer across the threshold",
           [dimension_at(share2, t) for t in (0.90, 0.95, 0.99)], [3, 4, 5])
    _check(fails, "all three counts are recorded together",
           dimensions_at_targets(share2), {"0.95": 4, "0.9": 3, "0.99": 5})
    _check(fails, "only the 95 percent count feeds the call",
           decide_dimension({140: dimensions_at_targets(share2)["0.95"]}),
           "DIMENSION_REACHES_4")

    print("-- endpoint 3, rank among 200 shuffle draws --")
    draws = [float(i) for i in range(200)]                      # 0.0 up to 199.0
    _check(fails, "rank of a value above every draw", null_rank(500.0, draws), 1)
    _check(fails, "rank of a value above all but nine", null_rank(190.5, draws), 10)
    _check(fails, "rank of a value above all but ten", null_rank(189.5, draws), 11)
    _check(fails, "rank counts ties against the observed value", null_rank(199.0, draws), 2)
    _check(fails, "clearing at rank 1", decide_cleared(1), "NULL_CLEARED")
    _check(fails, "clearing at rank 10", decide_cleared(10), "NULL_CLEARED")
    _check(fails, "not clearing at rank 11", decide_cleared(11), "NULL_NOT_CLEARED")
    _check(fails, "the 410-million-weight ranks 199, 201 and 34 all fail",
           [decide_cleared(r) for r in (199, 201, 34)],
           ["NULL_NOT_CLEARED"] * 3)
    _check(fails, "no window under either labelling", decide_any_cleared(0, 0),
           "NO_WINDOW_CLEARED")
    _check(fails, "one character window", decide_any_cleared(1, 0), "ANY_WINDOW_CLEARED")
    _check(fails, "one token window only", decide_any_cleared(0, 1),
           "ANY_WINDOW_CLEARED")

    print("-- endpoint 4, which labelling owns the pattern --")
    _check(fails, "neither labelling clears",
           decide_attribution(60.0, 57.5, False, False), "NO_PATTERN_TO_ATTRIBUTE")
    _check(fails, "neither clears even with far-apart periods",
           decide_attribution(60.0, 20.0, False, False), "NO_PATTERN_TO_ATTRIBUTE")
    _check(fails, "only the token labelling clears",
           decide_attribution(60.0, 57.5, False, True), "TOKEN_ONLY_PATTERN")
    _check(fails, "only the character labelling clears",
           decide_attribution(60.0, 57.5, True, False), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "both clear, periods 60 and 31",
           decide_attribution(60.0, 31.0, True, True), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "both clear, periods 60 and 57.5",
           decide_attribution(60.0, 57.5, True, True), "CONFOUNDED_WITH_TOKEN_POSITION")
    _check(fails, "both clear, periods 60 and 56.5",
           decide_attribution(60.0, 56.5, True, True), "CHARACTER_COUNT_DISTINGUISHED")
    _check(fails, "both clear, periods differ by exactly 3",
           decide_attribution(60.0, 57.0, True, True), "CONFOUNDED_WITH_TOKEN_POSITION")
    _check(fails, "the control cannot be computed",
           decide_attribution(60.0, None, True, None), "TOKEN_CONTROL_NOT_COMPUTABLE")
    _check(fails, "the previous rung's own numbers no longer read as distinguished",
           decide_attribution(120.0, 28.5585, False, False), "NO_PATTERN_TO_ATTRIBUTE")

    print("-- endpoint 5, bin correlations at width 280 --")
    good_low = [0.82, 0.79, 0.71, 0.62]
    dead_high = [0.11, 0.05, 0.02]
    live_high = [0.11, 0.62, 0.02]
    _check(fails, "low bins strong and high bins dead",
           decide_range(good_low, dead_high), "WIDE_COUNTS_NOT_REPRESENTED")
    _check(fails, "one high bin alive",
           decide_range(good_low, live_high), "WIDE_COUNTS_MEASURABLE")
    _check(fails, "a high bin exactly at 0.5",
           decide_range(good_low, [0.11, 0.5, 0.02]), "WIDE_COUNTS_MEASURABLE")
    _check(fails, "a high bin at 0.35 matches neither reading",
           decide_range(good_low, [0.11, 0.35, 0.02]), "RANGE_PATTERN_UNMATCHED")
    _check(fails, "a high bin exactly at 0.2",
           decide_range(good_low, [0.11, 0.2, 0.02]), "RANGE_PATTERN_UNMATCHED")
    _check(fails, "a high bin just under 0.2",
           decide_range(good_low, [0.11, 0.199, 0.02]), "WIDE_COUNTS_NOT_REPRESENTED")
    _check(fails, "a low bin exactly at 0.5 still counts as reaching it",
           decide_range([0.82, 0.5, 0.71, 0.62], dead_high), "WIDE_COUNTS_NOT_REPRESENTED")
    _check(fails, "a low bin just under 0.5",
           decide_range([0.82, 0.499, 0.71, 0.62], dead_high), "RANGE_PATTERN_UNMATCHED")
    _check(fails, "the 410-million-weight low bins were already weak",
           decide_range([0.648, 0.219, 0.153, 0.271], dead_high),
           "RANGE_PATTERN_UNMATCHED")
    _check(fails, "a bin not computable",
           decide_range(good_low, [0.1, None, 0.1]), "RANGE_CHECK_INCOMPLETE")
    _check(fails, "the low floor is 0.5", RANGE_LOW_FLOOR, 0.5, 1e-12)
    _check(fails, "the high ceiling is 0.2", RANGE_HIGH_CEIL, 0.2, 1e-12)

    print("-- the previous rung's fork, over character clearings only --")
    fixed = [(140, 60.0), (200, 62.0), (280, 66.0)]             # gaps up to 6 characters
    grows = [(140, 60.0), (200, 85.0), (280, 120.0)]            # ratios 0.4286, 0.425, 0.4286
    messy = [(140, 60.0), (200, 130.0), (280, 90.0)]            # neither rule holds
    g_fixed, r_fixed = pairwise_gaps(fixed)
    _check(fails, "planted fixed periods, largest gap", g_fixed, 6.0, 1e-9)
    g_grows, r_grows = pairwise_gaps(grows)
    _check(fails, "planted growing periods, largest gap", g_grows, 60.0, 1e-9)
    _check(fails, "planted growing periods, largest ratio gap", r_grows, 0.003571, 1e-6)
    _check(fails, "one window leaves the gap undefined", pairwise_gaps([(140, 60.0)]),
           (None, None))
    _check(fails, "gap within 10 characters", decide_wavelength_gap(g_fixed),
           "WAVELENGTH_GAP_WITHIN_10")
    _check(fails, "gap of exactly 10 characters", decide_wavelength_gap(10.0),
           "WAVELENGTH_GAP_WITHIN_10")
    _check(fails, "gap above 10 characters", decide_wavelength_gap(g_grows),
           "WAVELENGTH_GAP_EXCEEDS_10")
    _check(fails, "gap undefined", decide_wavelength_gap(None),
           "GAP_UNDEFINED_FEWER_THAN_TWO_CLEARED")
    _check(fails, "ratio gap within 0.1", decide_ratio_gap(r_grows), "RATIO_GAP_WITHIN_0.1")
    _check(fails, "ratio gap of exactly 0.1", decide_ratio_gap(0.1), "RATIO_GAP_WITHIN_0.1")
    _check(fails, "ratio gap above 0.1", decide_ratio_gap(0.3), "RATIO_GAP_EXCEEDS_0.1")
    _check(fails, "ratio gap undefined", decide_ratio_gap(None),
           "GAP_UNDEFINED_FEWER_THAN_TWO_CLEARED")
    _check(fails, "planted fixed periods classify as fixed", decide_invariance(fixed),
           "INVARIANT")
    _check(fails, "planted growing periods classify as proportional",
           decide_invariance(grows), "PROPORTIONAL")
    _check(fails, "planted messy periods classify as mixed", decide_invariance(messy),
           "MIXED")
    _check(fails, "two windows, close periods", decide_invariance([(140, 60.0), (280, 64.0)]),
           "INVARIANT")
    _check(fails, "one window is undetermined", decide_invariance([(200, 60.0)]),
           "UNDETERMINED")
    _check(fails, "no window cleared", decide_invariance([]), "NO_WIDTH_CLEARED")

    print("== the four pre-registered headline readings, on planted values ==")
    _check(fails, "a failed floor voids the rung",
           decide_headline("CURVE_TOO_WEAK", fixed, True, "DIMENSION_REACHES_4"),
           "CURVE_TOO_WEAK")
    _check(fails, "floor passed, nothing cleared, directions below 4",
           decide_headline("FLOOR_CLEARED", [], False, "DIMENSION_BELOW_4"),
           "PRECONDITION_ABSENT_AT_1P4B")
    _check(fails, "floor passed, nothing cleared, directions reached 4",
           decide_headline("FLOOR_CLEARED", [], False, "DIMENSION_REACHES_4"),
           "PRECONDITION_WITHOUT_RIPPLE")
    _check(fails, "a character window cleared, fixed periods",
           decide_headline("FLOOR_CLEARED", fixed, False, "DIMENSION_REACHES_4"),
           "INVARIANT")
    _check(fails, "a character window cleared, growing periods",
           decide_headline("FLOOR_CLEARED", grows, False, "DIMENSION_BELOW_4"),
           "PROPORTIONAL")
    _check(fails, "a character window cleared, messy periods",
           decide_headline("FLOOR_CLEARED", messy, False, "DIMENSION_BELOW_4"), "MIXED")
    _check(fails, "exactly one character window cleared",
           decide_headline("FLOOR_CLEARED", [(140, 60.0)], False, "DIMENSION_BELOW_4"),
           "UNDETERMINED")
    _check(fails, "only the token labelling cleared, so no fork is entered",
           decide_headline("FLOOR_CLEARED", [], True, "DIMENSION_BELOW_4"),
           "TOKEN_ONLY_PATTERN")
    _check(fails, "a token clearing never overrides a character clearing",
           decide_headline("FLOOR_CLEARED", fixed, True, "DIMENSION_BELOW_4"), "INVARIANT")
    _check(fails, "the 410-million-weight result, replayed through this rule",
           decide_headline("FLOOR_CLEARED", [], False,
                           decide_dimension(PRIOR_410M_DIMENSIONS)),
           "PRECONDITION_ABSENT_AT_1P4B")

    print("== supporting checks ==")
    _check(fails, "the model of this rung", MODEL_ID, "EleutherAI/pythia-1.4b-deduped")
    _check(fails, "the checkpoint of this rung", REVISION, "step143000")
    _check(fails, "the expected layer count", EXPECT_LAYERS, 24)
    _check(fails, "the expected hidden size", EXPECT_HIDDEN, 2048)
    _check(fails, "the three probe windows are among the nine wrap widths",
           all(a in WIDTHS for a in APERTURES), True)
    _check(fails, "the nine wrap widths are the seven plus the two wide ones",
           sorted(SELECT_WIDTHS + WIDE_WIDTHS), WIDTHS)
    _check(fails, "the ridge grid is the committed one", RIDGE_GRID,
           [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    _check(fails, "the floor is 0.60", FLOOR_R2, 0.60, 1e-12)
    _check(fails, "the eight bins cover labels 1 to 280", label_bins()[-1][1], 280)
    _check(fails, "the base seed is zero", SEED, 0)
    _check(fails, "the shuffle draws per window per labelling", N_DRAWS, 200)
    _check(fails, "clearing means rank 10 or better of 200", NULL_RANK_MAX, 10)
    _check(fails, "1200 draws in all",
           N_DRAWS * len(APERTURES) * len(LABELLING_CODES), 1200)
    _check(fails, "five totals groups for training documents", len(TRAIN_GROUPS), 5)
    _check(fails, "one totals group per wrap width for held-out documents",
           len(TEST_GROUPS), 9)
    _check(fails, "the totals groups are the training and held-out groups together",
           group_names(), list(TRAIN_GROUPS) + list(TEST_GROUPS))
    _check(fails, "every totals group name is distinct",
           len(set(group_names())), len(group_names()))
    _check(fails, "the held-out group for width 140 exists", "test_w140" in TEST_GROUPS, True)
    _check(fails, "the results folder of this rung", RESULTS_DIR_DEFAULT, "results_seedD3")
    _check(fails, "no output calls a same-box rerun a replication",
           "replication" in RERUN_NOTE and "not a replication" in RERUN_NOTE, True)
    try:
        prov = load_provenance()
        print("  %-64s ok" % "record writer importable")
        for name in ("write_result", "describe_input", "cell_slug"):
            if not hasattr(prov, name):
                fails.append("record writer lacks " + name)
                print("  %-64s FAIL" % ("record writer has " + name))
    except Exception as exc:                                    # noqa: BLE001
        fails.append("record writer import: %s" % exc)
        print("  %-64s FAIL %s" % ("record writer importable", exc))

    try:
        import numpy as np
    except ImportError:
        print("  %-64s skipped (numpy absent)" % "fits, correlations, analysis plumbing")
        print("selftest:", "PASS" if not fails else "%d FAILURES" % len(fails))
        for f in fails:
            print("FAIL", f)
        return 1 if fails else 0

    print("== ridge fit and fit quality from running totals ==")
    import random as _r
    _r.seed(7)
    rows = [[_r.uniform(-1, 1), _r.uniform(-1, 1)] for _ in range(50)]
    ys = [2.0 * a - 3.0 * b + 0.5 + _r.uniform(-0.01, 0.01) for a, b in rows]
    xm = np.array(rows)
    ym = np.array(ys)
    gram = xm.T @ xm
    vec = xm.sum(0)
    cross = xm.T @ ym
    w, off = ridge_fit_from_stats(gram, vec, cross, float(ym.sum()), len(ys), 1e-6)
    r2, sse, sstot = r2_from_stats(w, off, gram, vec, cross, float(ym.sum()),
                                   float((ym * ym).sum()), len(ys))
    pred = xm @ w + off
    direct_sse = float(((pred - ym) ** 2).sum())
    direct_r2 = 1.0 - direct_sse / float(((ym - ym.mean()) ** 2).sum())
    _check(fails, "sum of squared errors from running totals", sse, direct_sse, 1e-6)
    _check(fails, "fit quality from running totals", r2, direct_r2, 1e-9)
    _check(fails, "planted weights recovered",
           float(abs(w[0] - 2.0) + abs(w[1] + 3.0)), 0.0, 0.02)

    print("== per-bin Pearson correlation of both bin checks ==")
    truth = np.array([10.0, 12.0, 14.0, 16.0])
    _check(fails, "a perfect prediction correlates 1",
           pearson_corr(truth.copy(), truth)[0], 1.0, 1e-12)
    _check(fails, "a shifted and stretched prediction still correlates 1",
           pearson_corr(3.0 * truth + 100.0, truth)[0], 1.0, 1e-12)
    _check(fails, "a reversed prediction correlates minus 1",
           pearson_corr(-truth, truth)[0], -1.0, 1e-12)
    _check(fails, "a flat prediction has no correlation",
           pearson_corr(np.full(4, 13.0), truth)[0], None)
    _check(fails, "a bin with one position is not computable",
           pearson_corr(np.array([1.0]), np.array([1.0]))[0], None)
    _check(fails, "a bin with no label spread is not computable",
           pearson_corr(np.array([1.0, 2.0]), np.array([5.0, 5.0]))[0], None)
    _check(fails, "the count of positions in a bin is reported",
           pearson_corr(truth.copy(), truth)[1], 4)
    # A biased flat-plus-noise prediction can still correlate well, which is exactly
    # why the design changed this check from a within-bin fit quality.
    noisy = truth * 0.2 + 200.0 + np.array([0.01, -0.01, 0.02, -0.02])
    _check(fails, "a badly scaled prediction still correlates above 0.5",
           pearson_corr(noisy, truth)[0] > 0.5, True)
    # Planted bin rows: a prediction that tracks the count up to 140 and is noise above
    # 175. The wide check must read that as measurable only where a high bin is alive.
    lab_all = np.arange(1.0, 281.0)
    pred_all = lab_all.copy()
    pred_all[lab_all > 175] = 0.0
    pred_all[lab_all > 175] += np.arange(int((lab_all > 175).sum())) % 2
    rows280, lows280, highs280 = bin_correlation_rows(pred_all, lab_all, label_bins(),
                                                      with_roles=True)
    _check(fails, "eight bin rows at width 280", len(rows280), 8)
    _check(fails, "four low correlations and three high ones",
           (len(lows280), len(highs280)), (4, 3))
    _check(fails, "the straddling bin is recorded with its role",
           rows280[4]["role"], "straddles")
    _check(fails, "a perfect low half reads 1", round(lows280[0], 6), 1.0, 1e-9)
    _check(fails, "planted rows read as not represented",
           decide_range(lows280, highs280), "WIDE_COUNTS_NOT_REPRESENTED")
    rows140, lows140, highs140 = bin_correlation_rows(lab_all[:140], lab_all[:140],
                                                      label_bins_140(), with_roles=False)
    _check(fails, "four bin rows across characters 1 to 140", len(rows140), 4)
    _check(fails, "the recorded-only rows carry no role", "role" in rows140[0], False)
    _check(fails, "every recorded-only bin has a correlation",
           all(r["pearson_correlation"] is not None for r in rows140), True)

    print("== analysis plumbing on small planted arrays ==")
    import tempfile
    labels = np.array([1, 2, 3, 4, 5, 6])
    docs = np.array([0, 0, 0, 1, 1, 1])
    out = shuffle_within_docs(labels, docs, np.random.default_rng(SEED))
    _check(fails, "shuffling preserves the label multiset",
           sorted(out.tolist()), [1, 2, 3, 4, 5, 6])
    _check(fails, "shuffling stays inside each document",
           sorted(out[:3].tolist()) == [1, 2, 3] and sorted(out[3:].tolist()) == [4, 5, 6],
           True)
    _check(fails, "labels map to slots and outsiders to minus one",
           slots_array(np.array([0, 1, 2, 3]), 1, 2).tolist(), [-1, 0, 1, -1])
    xs = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    mat, counts = means_by_slot(xs, np.array([0, 0, 1, 1]), 2)
    _check(fails, "average vector of the first slot", mat[0].tolist(), [2.0, 0.0])
    _check(fails, "average vector of the second slot", mat[1].tolist(), [0.0, 3.0])
    _check(fails, "positions per slot", counts.tolist(), [2, 2])
    try:
        means_by_slot(xs, np.array([0, 0, 0, 0]), 2)
        fails.append("an empty slot should halt")
        print("  %-64s FAIL no halt" % "an empty slot halts")
    except Halt:
        print("  %-64s ok" % "an empty slot halts")
    cos = cosine_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]))
    _check(fails, "identical directions have similarity 1", cos[0][1], 1.0, 1e-12)
    _check(fails, "square directions have similarity 0", cos[0][2], 0.0, 1e-12)
    one_way = np.outer(np.arange(1.0, 9.0), np.array([1.0, 2.0, 3.0]))
    _check(fails, "a one-direction pattern measures one direction",
           ambient_dimension(one_way - one_way.mean(axis=0, keepdims=True))[0], 1)
    _check(fails, "a one-direction pattern measures one at every target",
           dimensions_at_targets(
               variance_shares(one_way - one_way.mean(axis=0, keepdims=True))),
           {"0.95": 1, "0.9": 1, "0.99": 1})

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        groups = ["fold0", "fold1"]
        payload = {"groups": np.array(groups), "label_sum": np.array([3.0, 7.0]),
                   "label_sq": np.array([9.0, 49.0]), "n_pos": np.array([2, 5]),
                   "n_read": np.array([2]), "hidden": np.array([3])}
        for i, g in enumerate(groups):
            payload["gram__" + g] = np.full((2, 3, 3), float(i + 1))
            payload["vec_sum__" + g] = np.full((2, 3), float(i + 1))
            payload["cross__" + g] = np.full((2, 3), float(i + 1))
        np.savez(d / "totals.npz", **payload)
        tot = load_totals(d)
        _check(fails, "reloaded readout count", tot["n_read"], 2)
        g, v, c, ls, lq, n = sum_groups(tot, groups, 1)
        _check(fails, "gram of two groups added", float(g[0][0]), 3.0, 1e-12)
        _check(fails, "labels of two groups added", ls, 10.0, 1e-12)
        _check(fails, "positions of two groups added", n, 7)
        (d / "act_a140_layer03.f32").write_bytes(np.arange(12, dtype=np.float32).tobytes())
        back = read_aperture_layer(d, 140, 3, 4, hidden=3)
        _check(fails, "saved vectors reload with the right shape", back.shape, (4, 3))
        _check(fails, "saved vectors reload with the right values",
               back[1].tolist(), [3.0, 4.0, 5.0])
        try:
            read_aperture_layer(d, 140, 3, 5, hidden=3)
            fails.append("a wrong row count should halt")
            print("  %-64s FAIL no halt" % "a wrong row count halts")
        except Halt:
            print("  %-64s ok" % "a wrong row count halts")

        # The whole pattern path, with a period planted in the average vectors. Forty
        # labels give a profile of length 20 under the committed rule, the label count
        # minus 20. The planted period is 10 over that length, which is a whole number
        # of cycles. A period off that grid, such as 8 over 20, cannot be returned: the
        # nearest whole-cycle periods are 10 and about 6.67.
        n_slots = 40
        planted_period = 10.0
        base = np.zeros((n_slots, 6), dtype=np.float32)
        for s in range(n_slots):
            base[s, 0] = float(s)
            base[s, 1] = 6.0 * math.sin(2 * math.pi * s / planted_period)
            base[s, 2] = 6.0 * math.cos(2 * math.pi * s / planted_period)
        reps = 5
        x = np.repeat(base, reps, axis=0)
        lab = np.repeat(np.arange(1, n_slots + 1, dtype=np.int32), reps)
        dcs = np.tile(np.arange(reps, dtype=np.int32), n_slots)
        spectra = d / "spectra.jsonl"
        res = pattern_with_null(x, lab, dcs, 1, n_slots, profile_length_for(n_slots), 20,
                                np.random.default_rng(shuffle_seed_entropy(140, "characters")),
                                spectra, "characters")
        lines = spectra.read_text().strip().split("\n")
        _check(fails, "one saved spectrum per draw plus the observed", len(lines), 21)
        first = json.loads(lines[0])
        _check(fails, "the observed spectrum is the first line", first["kind"], "observed")
        _check(fails, "every saved line carries a whole spectrum",
               all(len(json.loads(l)["spectrum"]) == 10 for l in lines), True)
        _check(fails, "no per-draw average vectors are written anywhere",
               sorted(p.name for p in d.glob("*.npz")), ["totals.npz"])
        _check(fails, "the planted period is recovered", res["observed"]["wavelength"],
               planted_period, 1e-9)
        _check(fails, "the observed beats every draw of pure noise", res["rank"], 1)
        _check(fails, "positions per label are preserved", min(res["counts"]), reps)
        _check(fails, "the average vectors have one row per label",
               res["means"].shape, (n_slots, 6))
        # The planted average vectors hold a rising ramp plus one full cycle, so three
        # directions carry everything: the ramp and the cycle's two parts.
        _check(fails, "the planted ramp and cycle count as three directions",
               res["ambient_dimension"], 3)
        _check(fails, "the same count is reported at all three targets",
               sorted(res["ambient_dimension_by_target"]), ["0.9", "0.95", "0.99"])

    print("selftest:", "PASS" if not fails else "%d FAILURES" % len(fails))
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Part 13. Entry point.
# ---------------------------------------------------------------------------

STAGE_ORDER = ["extract", "calibrate", "endpoints"]

# The build brief names the middle position "generate". Nothing is generated here, so
# that name is only a second way to ask for the calibration stage. Asking for it prints
# what the stage really does, then runs it.
STAGES = {"extract": stage_extract, "calibrate": stage_calibrate,
          "generate": stage_calibrate, "endpoints": stage_endpoints}

NO_GENERATION_MESSAGE = (
    "No text is generated in this experiment. The design says one deterministic forward "
    "pass per document per wrap width, over fixed committed text. The stage in this "
    "position fixes every remaining constant on training documents only, before any "
    "held-out number exists. It is also reachable as --stage calibrate."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES) + ["all"],
                    help="extract, calibrate (also spelled generate), endpoints, or all")
    ap.add_argument("--selftest", action="store_true",
                    help="standard-library checks only; no model, no graphics card")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", RESULTS_DIR_DEFAULT))
    ap.add_argument("--stimulus", default=None, help="path to the committed text file")
    ap.add_argument("--design-record", default=None, help="path to the committed design record")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.stage:
        ap.error("give --stage extract, calibrate, generate, endpoints, or all; "
                 "or give --selftest")
    if args.stage == "generate":
        print(NO_GENERATION_MESSAGE)

    order = STAGE_ORDER if args.stage == "all" else [args.stage]
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
