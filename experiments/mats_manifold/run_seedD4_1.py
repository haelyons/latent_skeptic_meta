#!/usr/bin/env python3
"""Measures when a character-count reader forms during training, and whether it moves.

WHAT THIS RUNS. Twelve public checkpoints of one open language model, 1.4 billion
weights, read the same fixed committed text. The text is prose hard-wrapped at nine
line widths; this run reads the seven widths from 20 to 140 characters. At every token
position we know how many characters have passed since the last line break. A ridge
regression is a linear fit with a penalty on large weights. One such fit predicts that
count from the model's internal vector at that position. Fit quality is the share of the
count's spread the fit explains on documents the fit never saw. It is measured at every
readout, from the embedding output through the last of the 24 transformer layers.

THE QUESTION. Two committed measurements already show that the final checkpoint of this
model carries such a counter and a sharp precision boundary near 70 characters: pooled
quality 0.876302 at readout 5 on this text, and 0.879879 on a second, disjoint text.
Neither says when either appeared. So this run measures the same things at twelve named
checkpoints of the same training run, from initialization to the end. Two rival accounts
are on the table. One says both are fossils of early training pressure: they form early
and then hold still. The other says they keep drifting after they first appear.

WHY THIS MODEL AND NOT THE SMALLER SIBLING. The previous rung swept the same twelve
checkpoints of the 410-million-weight sibling. It measured two things that force this
rung's shape. First, that model's precision boundary at its final checkpoint reads
-0.090673, so the boundary is absent there: it is scale-emergent, and a formation sweep
for it must run where the object exists. Second, the untrained checkpoint of that model
already decodes character position at pooled quality 0.344634. A randomly started network
carries partly decodable positional structure. So the untrained checkpoint here is a
BASELINE measurement, not an alarm, and every formation threshold is measured up from it.

WHAT IS MEASURED AT EVERY CHECKPOINT.
  a. Held-out fit quality at every readout, pooled over the seven wrap widths 20 to 140.
     The best readout is the one with the highest such quality, chosen per checkpoint.
  b. A four-part precision profile at that best readout, on the width-140 held-out
     documents. Inside each range of counts, 1 to 35, 36 to 70, 71 to 105 and 106 to
     140, we take the Pearson correlation between the predicted count and the true one.
     The boundary ratio is the mean of the first two parts minus the mean of the last two.
  c. The number of directions holding 90, 95 and 99 percent of the spread in the
     per-count average vectors. Recorded at every checkpoint, and gated nowhere.
  d. The best readout's location, the chosen ridge strength, and whether either sits at
     the edge of its search range.
  e. Two shape signatures the untrained sibling showed, recorded and gated nowhere: how
     flat fit quality is across readouts, and how it runs across wrap widths, where the
     narrowest width read -2.479186 on the untrained sibling.

THE TWO INTEGRITY GATES, IN THE COMMITTED ORDER. The final checkpoint (step143000) is
measured FIRST, and must pass two bars at once. Its pooled quality must land within 0.02
of 0.876302, and its boundary ratio must reach 0.3; the committed prior value of that
ratio on this text is 0.514227, from the committed parts 0.645989, 0.8196, 0.210583 and
0.226552. Missing either bar voids the whole sweep as an instrument fault and stops the
run. The initialization checkpoint (step0) is measured SECOND, and its pooled quality is
recorded as the baseline floor. It is not required to be near zero. It voids the sweep
only by reaching 0.7 times the anchor quality, which is 0.6134114: above that the fit
would be reading most of its signal from untrained structure. A range of 0.2 to 0.45 is
expected there, carried from the sibling's measured 0.344634, and it is recorded rather
than required. Only then are the remaining ten measured, earliest to latest.

THE FORMATION THRESHOLDS ARE MEASURED UP FROM THE BASELINE. Write the baseline floor as
B and the committed anchor quality as 0.876302. The formation step is the first
checkpoint whose pooled quality reaches B plus half the distance from B to the anchor.
The formed checkpoint is the first reaching B plus nine tenths of that distance, and the
stability test starts there for BOTH channels. The precision boundary keeps an absolute
bar of 0.3, with no baseline correction, because the untrained sibling's ratio read
-0.080316, which is effectively zero.

THE READINGS THIS RUN CAN RETURN. Across the checkpoints at or after the formed one, the
boundary ratio's range must stay within 0.15 and the pooled quality's range within 0.10.
Inside both, the reading is that both held still. Outside either, with the movement
running one way in training steps, the reading is drift. Outside either and not one-way,
the outcome is unclassifiable. The pre-registered readings are therefore
FORMED_EARLY_AND_HELD, FORMED_LATE_AND_HELD, FORMED_AND_DRIFTED, NOT_FORMED_BY_FINAL,
UNSTABLE_NOT_CLASSIFIABLE, VOID_ANCHOR_FAILED and VOID_CONTROL_FAILED. The boundary's own
timing is a second, separate reading: BOUNDARY_WITH_COUNTER when the ratio reaches 0.3 at
or before the formation step, BOUNDARY_AFTER_COUNTER when later, and
BOUNDARY_NOT_FORMED_BY_FINAL when it never does, which contradicts the gate cell and so
names an instrument fault.

WHICH MARKER DECIDES EARLY VERSUS LATE. The committed gate record answers this in its
second answer of record: the half-distance crossing locates when the counter becomes
usable and decides early against late, and the nine-tenths crossing only starts the
stability test. Both are computed and both are recorded. The design object's third
endpoint states the early-late comparison against the nine-tenths crossing instead, so
that comparison is recorded beside the reading, and the reading follows the committed
answer.

THE LABEL-REACHABILITY CONTROL, BINDING. The committed gate record settles one open
question with a model-free control, records/CONTROL_d4_1_label_reachability.txt: five
synthetic trajectories, read only at the twelve grid steps with these bands, produce all
five formation-and-stability labels, so none is forced by the grid. Those five
trajectories and their five labels are replayed inside this file's self-test, and a
disagreement fails the self-test before any graphics-card time is spent.

WHAT IS RECORDED AND GATES NOTHING. The direction counts at three cutoffs. The best
readout's trajectory, and the counter's localization moment, defined as the first
checkpoint whose best readout is interior AND whose peak beats the median readout's
quality by at least 0.1. The per-width qualities and the flatness of quality across
readouts. The chosen ridge strength. Whether the boundary already stood at the formation
step. The stability ranges an alternative reading would use.

THE COORDINATE CONTROL, BINDING. The previous rung's committed gate record adopts one
control as binding, and this rung's design carries it forward. At every checkpoint the
selected best readout and the selected ridge strength are logged beside the pooled
quality and the boundary ratio. A checkpoint whose best readout sits at an end of the
search range, readout 0 or readout 24, or whose ridge strength sits at an end of its
grid, 1 or 100000, is flagged in its own record as sitting at a coordinate edge. The
initialization checkpoint is EXPECTED to trip those flags, because a random model has no
meaningful best readout; on the sibling it selected readout 24 and strength 10000. That
expectation is on the record in advance, so the flag there reads as confirmation.

NO TEXT IS GENERATED ANYWHERE. The design says forward passes only: hidden states come
from teacher-forced encoding of the one fixed text file, with no decoding and no
sampling. So this file has no sampling settings, no decoding and no answer parsing. The
build brief that produced this file asks for a generation step, a reasoning-tag parser
and an answer extractor. Those three things cannot exist for this experiment. That brief
is a fixed template shared with a different, text-generating project (pipeline.js line
104). The three closest real equivalents are implemented and self-tested instead, and
each is named where it is defined below. The same choice was made, for the same reason,
by the six earlier runners of this chain. Nothing is batched either: each text variant is
one whole document and its labels are per position, so there is nothing to group.

STAGES, in order, each runnable on its own:
  --stage extract    the twelve-checkpoint sweep. Per checkpoint: download, check the
                     configuration, one forward pass per text variant, the running
                     totals, the ridge strength from training documents only, the best
                     readout from held-out documents, the four-part profile, the
                     direction counts, the records, then delete that checkpoint's
                     weights and working vectors before the next one starts.
  --stage generate   nothing is generated. The build brief names this position; the name
                     is an alias for the sweep, and asking for it prints why first.
  --stage endpoints  the cross-checkpoint arithmetic: the baseline-relative formation
                     thresholds, the formation step, the formed checkpoint, the
                     boundary-ratio trajectory and its own crossing, the stability test
                     at ranges 0.15 and 0.10, and the two pre-registered readings.
  --stage all        extract, then endpoints.
  --selftest         standard library only. Loads no model, needs no graphics card, and
                     exits non-zero on any failure.

WHY THE FIT LIVES INSIDE THE SWEEP. A checkpoint's fit reads running totals that are
about 10.1 gigabytes for that checkpoint alone at hidden size 2048. Keeping twelve sets
would cost about 121 gigabytes, and the execution directive requires the weights and
working vectors to be deleted before the next checkpoint starts. So each checkpoint's
totals are held in memory, used, and dropped, and every held-out number for that
checkpoint is computed while they exist. The last stage then needs only the small
per-checkpoint numbers, which the sweep writes to results_seedD4_1/out/sweep_table.json
beside the records.

Only one model is resident at any moment. Each is loaded in half precision, used, and
released, and its downloaded files are deleted before the next download starts.

EXPECTED WALL CLOCK, AND A CONFLICT ON THE RECORD. Measured numbers from this same
instrument on this same model at hidden size 2048: 432 forward passes took 289.1 seconds,
and one strength search over five folds, 25 readouts and six strengths took 221.1
seconds. This rung reads 336 variants per checkpoint, so one checkpoint costs roughly
eight minutes and twelve cost roughly 100 minutes. The design's own compute note assumes
one box under a 90-minute cap. Those two numbers disagree, and this file does not resolve
the disagreement: it measures the twelve checkpoints the design names, in the order the
design fixes, and the launcher's cap is a launcher input. The gate cell runs first and the
baseline second, so a cap reached early loses the tail of the sweep, and the last stage
refuses to read a sweep that did not finish. No partial reading is ever emitted.

Every result stage writes one self-describing record through provenance.py. A record
carries its own identity, its inputs with content hashes, its measurement, the rule that
judges it, and the resulting call. Every rule text here is written before any value is
known. Every per-checkpoint record and every trajectory record also carries both gate
decisions and a voided-by-gate flag, so no such record can be read without its void
status.

A second run of this script on the same rented box is a rerun of one instrument. It is
not a replication. No output here calls it one. This sweep is a single run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Constants. The source of each is marked.
#   DESIGN  the committed design object in records/seedD4_1_design_diff_triage.json.
#   GATE    a committed answer in records/GATE_COMMIT_seedD4_1.md.
#   CONTROL the committed label-reachability control,
#           records/CONTROL_d4_1_label_reachability.txt.
#   MEMORY  recalled from a paper, and checked against the download.
#   MEASURED a number already on this chain's record, carried as context only.
#   GUESS   fixed by this file because the design leaves it to implementation.
# Anything the design leaves unnamed is bound by the committed records of the previous
# rungs, records/GATE_COMMIT_seedD3_1.md and records/GATE_COMMIT_seedD4.md, as this
# rung's own design and gate record both say.
# ---------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"          # DESIGN
MODEL_SLUG = "pythia-1.4b-deduped"                   # the same name, safe in a filename
EXPECT_LAYERS = 24                                   # MEMORY, Pythia paper arXiv 2304.01373 table 1
EXPECT_HIDDEN = 2048                                 # MEMORY, same table

# The twelve named checkpoints, in training order. The release provides step0, the
# powers of two up to 512, then every 1000 steps (MEMORY, same paper).
REVISIONS = ["step0", "step512", "step1000", "step2000", "step4000", "step8000",
             "step16000", "step32000", "step64000", "step96000", "step128000",
             "step143000"]                            # DESIGN
FINAL_REVISION = "step143000"                         # DESIGN, the gate cell
BASELINE_REVISION = "step0"                           # DESIGN, the baseline floor
# Measurement order is committed and load-bearing: the gate cell first, the baseline
# second, then the remaining ten from earliest to latest.
MEASURE_ORDER = ([FINAL_REVISION, BASELINE_REVISION]
                 + [r for r in REVISIONS if r not in (FINAL_REVISION, BASELINE_REVISION)])

STIMULUS_NAME = "stimulus_linewrap_v2.json"          # DESIGN, the text behind both anchors
STIMULUS_SHA256 = "b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd"   # DESIGN
DESIGN_SHA256 = "1de4f659006e36b5ac280543d2a9cbb6cd014cbb8d371a36ba7e5730ae6fe282"     # this rung's design record, hashed on the workstation 2026-08-22
GATE_SHA256 = "604c934583da8e3fcb1b430b10409f43667cf9d6af7569baf5dc0b57e082f13a"       # this rung's gate record, same hashing
CONTROL_SHA256 = "9d48648a09ad3b3fca5255aab1b989a610baf8d4f6ecca0c301dc33397e027e0"    # the label-reachability control, same hashing

STIMULUS_WIDTHS = [20, 40, 60, 80, 100, 120, 140, 200, 280]   # DESIGN, what the file holds
USE_WIDTHS = [20, 40, 60, 80, 100, 120, 140]         # GATE answer 1, the seven this rung reads
PROFILE_WIDTH = 140                                  # DESIGN, the width the profile uses
N_DOCS = 48                                          # DESIGN
N_TRAIN_DOCS = 36                                    # DESIGN, documents 0 to 35
N_TEST_DOCS = 12                                     # DESIGN, documents 36 to 47

RIDGE_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]   # DESIGN
N_FOLDS = 5                                          # DESIGN, five-fold cross-validation
SEED = 0                                             # GATE answer 3, the one base seed

# Gate one: the final checkpoint must reproduce both committed anchors at once.
ANCHOR_POOLED = 0.876302                             # DESIGN, the committed anchor quality
ANCHOR_TOLERANCE = 0.02                              # DESIGN GUESS, the band's half-width
ANCHOR_RATIO_FLOOR = 0.3                             # DESIGN, the boundary bar, absolute

# Gate two: the untrained checkpoint is a baseline, and voids only from above.
BASELINE_VOID_SHARE = 0.7                            # DESIGN, share of the anchor quality
BASELINE_VOID_FLOOR = BASELINE_VOID_SHARE * ANCHOR_POOLED   # DESIGN, 0.6134114
BASELINE_EXPECTED_LOW = 0.2                          # DESIGN GUESS, recorded, never gated
BASELINE_EXPECTED_HIGH = 0.45                        # DESIGN GUESS, recorded, never gated

# The trajectory thresholds. Both formation crossings are shares of the distance from
# the measured baseline floor up to the committed anchor quality.
FORMATION_SHARE = 0.5                                # DESIGN, the formation crossing
FORMED_SHARE = 0.9                                   # DESIGN, the crossing that starts the hold test
BOUNDARY_PRESENT_FLOOR = 0.3                         # DESIGN, absolute, no baseline correction
HOLD_BOUNDARY_RANGE = 0.15                           # DESIGN GUESS, stability, boundary
HOLD_QUALITY_RANGE = 0.10                            # DESIGN GUESS, stability, quality
EARLY_CUTOFF_STEP = 16000                            # DESIGN, the early-late cutoff
LOCALIZATION_MARGIN = 0.1                            # DESIGN GUESS, recorded, never gated

BIN_WIDTH = 35                                       # DESIGN, width of one profile part
N_BINS = 4                                           # DESIGN, four parts over 1 to 140
VARIANCE_TARGETS = [0.90, 0.95, 0.99]                # DESIGN, all recorded, none gated

# The coordinate control carried forward as binding.
LAYER_EDGES = (0, EXPECT_LAYERS)                     # GATE, readout 0 and readout 24
RIDGE_EDGES = (RIDGE_GRID[0], RIDGE_GRID[-1])        # GATE, strength 1 and 100000

MIN_TRAIN_POSITIONS_PER_LABEL = 20                   # GUESS, carried from the prior rungs
MAX_WINDOW_START = 8                                 # GUESS, carried from the prior rungs
REPEAT_BLOCK_MAX = 8                                 # GUESS, longest repeating block checked
REPEAT_MIN_RUNS = 6                                  # GUESS, that many repeats is degenerate
CONTEXT_LIMIT_FALLBACK = 2048                        # MEMORY, Pythia context length

# Numbers already on this chain's record, carried only as context inside records.
PRIOR_POOLED_SAME_TEXT = 0.876302                    # MEASURED, this model, this text, readout 5
PRIOR_READOUT_SAME_TEXT = 5                          # MEASURED, where it landed
PRIOR_RIDGE_SAME_TEXT = 1000.0                       # MEASURED, the strength that run selected
PRIOR_PROFILE_SAME_TEXT = [0.645989, 0.8196, 0.210583, 0.226552]   # MEASURED, the committed parts
PRIOR_RATIO_SAME_TEXT_AS_COMMITTED = 0.514           # MEASURED, as the design states it
PRIOR_POOLED_FRESH_TEXT = 0.879879                   # MEASURED, this model, a second text
PRIOR_PROFILE_FRESH_TEXT = [0.649801, 0.83044, 0.19771, 0.244373]  # MEASURED, that text's parts
SIBLING_MODEL_ID = "EleutherAI/pythia-410m-deduped"  # the smaller sibling, previous rung
SIBLING_FINAL_POOLED = 0.781031                      # MEASURED, sibling, this text
SIBLING_FINAL_RATIO = -0.090673                      # MEASURED, sibling, boundary absent there
SIBLING_BASELINE_POOLED = 0.344634                   # MEASURED, sibling, untrained checkpoint
SIBLING_BASELINE_RATIO = -0.080316                   # MEASURED, sibling, untrained checkpoint
SIBLING_BASELINE_READOUT = 24                        # MEASURED, at the sweep edge
SIBLING_BASELINE_RIDGE = 10000.0                     # MEASURED, the strength selected there
SIBLING_BASELINE_NARROW_WIDTH_QUALITY = -2.479186    # MEASURED, sibling, width 20, untrained
CORPUS_WRAP_LENGTH_CHARS = "about 70 to 72"          # the corpus's dominant hard wrap

RESULTS_DIR_DEFAULT = "results_seedD4_1"

# This runner's own room-to-work guards, not design numbers. One checkpoint holds about
# 10.1 gigabytes of running totals in memory, and about 2.9 gigabytes of download plus
# about 2.2 gigabytes of per-position vectors on disk.
DISK_FLOOR_MB = 24576                                # GUESS, per-checkpoint peak plus headroom
MEMORY_FLOOR_MB = 24576                              # GUESS, the running totals plus headroom

RERUN_NOTE = ("A further run of this script on the same box is a rerun of the same "
              "instrument, not a replication.")
VOID_NOTE = ("If either integrity gate recorded in these values failed, this record is "
             "void as evidence about the model, and is an instrument fault only.")
RECORDED_ONLY_NOTE = ("This quantity is recorded only. It carries no threshold and "
                      "decides nothing, as the committed design states.")
COORDINATE_NOTE = (
    "The selected best readout and the selected ridge strength are logged beside the "
    "pooled quality and the boundary ratio, and a selection sitting at an end of its "
    "search range is flagged. The ends are readout 0, readout 24, strength 1 and "
    "strength 100000. The previous rung's committed gate record adopts this control as "
    "binding and this rung's design carries it forward. Both state in advance that the "
    "initialization checkpoint is expected to trip it, because a random model has no "
    "meaningful best readout: on the smaller sibling it selected readout 24 at strength "
    "10000. There the flag is confirmation, not alarm.")
ANCHOR_NOTE = (
    "The gate cell must clear two bars at once, because two committed measurements of "
    "this model on this text are being reproduced: pooled quality 0.876302 at readout 5, "
    "and the boundary ratio 0.514227 from the parts 0.645989, 0.8196, 0.210583 and "
    "0.226552. Quality within 0.02 and ratio at least 0.3 are the two bars. A fresh, "
    "disjoint text gave 0.879879 and 0.519079 on the same model, which is context and "
    "not a bar.")
BASELINE_NOTE = (
    "The initialization checkpoint is a baseline measurement, not a near-zero alarm. The "
    "smaller sibling's untrained checkpoint decoded character position at pooled quality "
    "0.344634, with quality nearly flat across readouts, the best readout at the sweep "
    "edge, and quality inverted across widths down to -2.479186 at width 20. So an "
    "untrained network already carries partly decodable positional structure. Only a "
    "baseline at or above 0.7 times the anchor quality, which is 0.6134114, voids the "
    "sweep: above that the fit would be reading most of its signal from untrained "
    "structure. The expected range of 0.2 to 0.45 is carried from the sibling's measured "
    "0.344634 and is recorded, never required.")
BOUNDARY_NOTE = (
    "The boundary ratio is the mean of the profile's first two parts minus the mean of "
    "its last two. Its bar stays absolute at 0.3, with no baseline correction, because "
    "the smaller sibling's untrained checkpoint read -0.080316, which is effectively "
    "zero. On that sibling's final checkpoint the ratio reads -0.090673, so the boundary "
    "is absent at 410 million weights and present at 1.4 billion. That is why this "
    "formation sweep runs on the larger model.")
MARKER_NOTE = (
    "Two crossings are computed, both measured up from the baseline floor toward the "
    "committed anchor quality. The pooled quality first reaching the half-distance point "
    "locates when the counter becomes usable; that crossing decides whether formation "
    "counts as early or late against step16000. The pooled quality first reaching the "
    "nine-tenths point marks where the stability test starts, for both channels. The "
    "committed gate record assigns each crossing that job in its second and fourth "
    "answers of record. The design object's third endpoint states the early-late "
    "comparison against the nine-tenths crossing instead, so both comparisons are "
    "recorded and the reading follows the committed answer.")
BOUNDARY_TIMING_NOTE = (
    "The boundary's own first crossing of 0.3 is a separate recorded observable, not the "
    "start of its stability window; the committed gate record settles that in its fourth "
    "answer of record. Comparing that crossing against the formation step asks whether "
    "fine-grained precision arrives with usability or only later.")
CONTROL_REPLAY_NOTE = (
    "The committed label-reachability control settles, before any spend, that the coarse "
    "twelve-point grid and these bands do not force one label: five synthetic "
    "trajectories read only at the twelve grid steps produce all five formation-and-"
    "stability labels. Those five trajectories and their five labels are replayed by this "
    "file's self-test, which fails if any label or crossing moves.")
NO_CORPUS_NOTE = (
    "The training corpus is served pre-shuffled, so the hard-wrapped share of each batch "
    "is steady and exposure grows in step with the step count. Formation timing in steps "
    "is therefore formation timing in exposure. No corpus is downloaded or inspected "
    "here, and the design forbids both.")
FORCED_OUTCOME_NOTE = (
    "One thing about these thresholds is on the record in advance. The gate cell forces "
    "the final checkpoint to sit within 0.02 of the anchor quality, and the formed "
    "threshold is nine tenths of the way from the baseline to that same anchor. So the "
    "final checkpoint satisfies the formed threshold by arithmetic, and the reading "
    "NOT_FORMED_BY_FINAL cannot fire once the gate passes: it names an instrument fault, "
    "not a possible finding. Which of the four live labels fires still depends entirely "
    "on the ten intermediate checkpoints, and the committed label-reachability control "
    "shows all of them are reachable.")


# The per-checkpoint numbers the trajectory stage needs from every checkpoint. Declared
# once so the sweep, the self-test and the trajectory stage all check the same shape.
ROW_KEYS_NEEDED = ("revision", "step", "pooled_quality", "best_layer", "ridge_strength",
                   "boundary_ratio", "profile_correlations", "dimensions",
                   "coordinate_flags", "layer_median_quality", "peak_over_layer_median",
                   "localization_decision")


class Halt(Exception):
    """Raised where the design says halt. Never caught to keep a run going."""


def step_of(revision: str) -> int:
    """The training step a revision name states. Anything else halts."""
    if not isinstance(revision, str) or not revision.startswith("step"):
        raise Halt("revision %r is not one of the release's named checkpoints" % revision)
    tail = revision[4:]
    if not tail.isdigit():
        raise Halt("revision %r has no whole-number step in its name" % revision)
    return int(tail)


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
    restarts at zero after each line break.
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
# The tokenizer is the same for every checkpoint, so both readings are the same at
# every checkpoint. They are computed and recorded at each one anyway, because a
# recorded fact per record is cheaper than an argument about which one to trust.
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
# for. Nothing here is fuzzy, normalised, or rounded. Four things must match exactly:
# the text file's content hash, the downloaded configuration's layer count and width,
# the revision name against the committed list of twelve, and the whole-number label of
# a position against one profile part or one slot of the per-count average vectors. An
# off-by-one in the last one would move every row of that table with no error anywhere.
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
    """Whether a download reports exactly 24 layers and hidden size 2048."""
    return n_layers == EXPECT_LAYERS and hidden_size == EXPECT_HIDDEN


def exact_revision_match(revision) -> bool:
    """Whether a revision name is exactly one of the twelve the design names."""
    return isinstance(revision, str) and revision in REVISIONS


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
    1, and how far above is recorded.
    """
    for lo in range(1, max_start + 2):
        if all(counts_by_label.get(v, 0) >= min_count for v in range(lo, top + 1)):
            return lo
    return None


def counts_from_labels(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Part 4. The four-part precision profile and the boundary ratio.
#
# The parts are fixed by the design: counts 1 to 35, 36 to 70, 71 to 105 and 106 to
# 140. The statistic inside a part is the Pearson correlation between the predicted
# count and the true count, which the previous rung's committed gate record confirms in
# its fourth answer of record. The boundary ratio is the mean of the first two parts
# minus the mean of the last two.
# ---------------------------------------------------------------------------

def label_bins(bin_width: int = BIN_WIDTH, n_bins: int = N_BINS):
    """The four count ranges of the precision profile, as (low, high) pairs."""
    return [(1 + bin_width * i, bin_width * (i + 1)) for i in range(n_bins)]


def boundary_ratio(corrs):
    """Mean of the first two profile parts minus the mean of the last two.

    Returns None when any part is absent, which is the design's un-measurable case.
    """
    vals = list(corrs)
    if len(vals) != N_BINS or any(v is None for v in vals):
        return None
    return (vals[0] + vals[1]) / 2.0 - (vals[2] + vals[3]) / 2.0


# ---------------------------------------------------------------------------
# Part 5. The coordinate control carried forward as binding, and the recorded
# localization moment that reads the same two numbers.
# ---------------------------------------------------------------------------

def coordinate_flags(best_layer, ridge_strength) -> list:
    """Names of the coordinate edges a checkpoint's selection sits on.

    An empty list means both selections sit inside their search ranges. The readout
    range runs from 0, the embedding output, to 24, the last transformer layer. The
    ridge grid runs from 1 to 100000.
    """
    flags = []
    if best_layer in LAYER_EDGES:
        flags.append("best_layer_at_readout_%d" % best_layer)
    if ridge_strength is not None:
        for edge in RIDGE_EDGES:
            if float(ridge_strength) == float(edge):
                flags.append("ridge_strength_at_%g" % edge)
    return flags


def decide_coordinates(best_layer, ridge_strength) -> str:
    """Whether a checkpoint's selections sit inside their search ranges."""
    return ("COORDINATES_AT_GRID_EDGE" if coordinate_flags(best_layer, ridge_strength)
            else "COORDINATES_INTERIOR")


def median_of(values):
    """Middle value of a list, absent values dropped. None when nothing is left."""
    vals = sorted(float(v) for v in values if v is not None and v == v)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


# ---------------------------------------------------------------------------
# Part 6. Ridge fitting from running totals.
#
# Storing every internal vector of every position at every checkpoint would need
# thousands of gigabytes. It is not needed. A ridge fit and a fit-quality score depend
# on the data only through five running totals per readout: the sum of outer products of
# the vectors, the sum of the vectors, the sum of vector times label, the sum of labels,
# and the sum of squared labels. So the sweep accumulates those, per readout, per group
# of documents, for one checkpoint at a time.
#
# At hidden size 2048 one group's outer-product totals hold 25 readouts times 2048 times
# 2048 double-precision numbers, about 839 megabytes. Twelve groups hold about 10.1
# gigabytes, which is why they are held for one checkpoint only and dropped before the
# next download starts.
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
    """Pearson correlation inside one part, and the count it used. Needs numpy.

    Returns (None, n) when fewer than two positions fall in the part, or when either
    side has no spread at all. That is the design's un-measurable case, recorded as
    absent rather than as a number.
    """
    import math
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


def bin_correlation_rows(pred, labels, bins):
    """One row per profile part: its range, its count of positions, its correlation.

    Needs numpy. `pred` and `labels` are one entry per held-out position.
    """
    rows = []
    for (lo, hi) in bins:
        sel = (labels >= lo) & (labels <= hi)
        r, n_in = pearson_corr(pred[sel], labels[sel])
        rows.append({"bin_low": lo, "bin_high": hi, "n_positions": n_in,
                     "pearson_correlation": (None if r is None else round(float(r), 6))})
    return rows


# ---------------------------------------------------------------------------
# Part 7. The per-count average vectors and how many directions they use.
# ---------------------------------------------------------------------------

def slots_array(labels, window_lo, n_labels):
    import numpy as np
    off = labels - window_lo
    off = np.where((off >= 0) & (off < n_labels), off, -1)
    return off.astype(np.int64)


def means_by_slot(x, slots, n_slots):
    """Average vector per count, and how many positions entered each. Needs numpy."""
    import numpy as np
    keep = slots >= 0
    xk = x[keep]
    sk = slots[keep]
    ind = np.zeros((n_slots, xk.shape[0]), dtype=np.float32)
    ind[sk, np.arange(xk.shape[0])] = 1.0
    counts = ind.sum(axis=1)
    if float(counts.min()) < 1.0:
        empty = [int(i) for i in np.nonzero(counts < 1.0)[0]]
        raise Halt("count slots %r hold no held-out position, so their average vector "
                   "is undefined" % empty[:10])
    sums = ind @ xk
    return (sums / counts[:, None]).astype(np.float64), counts.astype(int)


def variance_shares(centred):
    """Share of the spread held by each direction, largest first. Needs numpy.

    Returns None when the whole table has no spread at all. That cannot happen for a
    trained checkpoint, and is guarded because a randomly started one is measured too.
    """
    import numpy as np
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv ** 2
    total = float(var.sum())
    if not (total > 0.0):
        return None
    return (var / total).tolist()


def dimension_at(share, target):
    """How many directions are needed to reach the target share of the spread."""
    if share is None:
        return None
    cum = 0.0
    dim = len(share)
    for i, s in enumerate(share):
        cum += s
        if cum >= target:
            dim = i + 1
            break
    return dim


def dimensions_at_targets(share, targets=tuple(VARIANCE_TARGETS)):
    """The direction count at each recorded cutoff, keyed by the cutoff."""
    return {"%g" % t: dimension_at(share, t) for t in targets}


# ---------------------------------------------------------------------------
# Part 8. Endpoint arithmetic. Every function here is a pure function of numbers, so
# the self-test can plant values whose correct call is known in advance. Every rule
# text is written before any value is known.
# ---------------------------------------------------------------------------

RULE_SETUP = (
    "The text file's hash must equal the committed hash exactly, the revision must be "
    "one of the twelve the design names, the layer count must equal 24, and the hidden "
    "size must equal 2048. Anything else is SETUP_FAILURE and halts the run before any "
    "measurement."
)
RULE_EXTRACT = (
    "All 48 documents at all seven wrap widths from 20 to 140 must produce one forward "
    "pass each, 336 in total, with no variant flagged as reaching the context limit and "
    "no internal vector holding a value that is not finite. Anything else is "
    "EXTRACT_INCOMPLETE and halts."
)
RULE_CALIBRATION = (
    "The ridge strength must come from five-fold cross-validation over training "
    "documents only, before any held-out number for that checkpoint exists. Every count "
    "from the window's start to 140 must carry at least 20 training positions, and every "
    "one of the four profile parts must hold at least two held-out positions. Anything "
    "else is CALIBRATION_UNUSABLE and halts that checkpoint."
)
RULE_ANCHOR = (
    "The gate cell is the final checkpoint (step143000), and it must clear two bars at "
    "once. Held-out fit quality, pooled over the seven wrap widths 20 to 140 at that "
    "checkpoint's own best readout, must lie within 0.02 of the committed 0.876302. The "
    "boundary ratio at that same readout, width 140, must reach 0.3; its committed prior "
    "value on this text is 0.514227. Missing either bar is VOID_ANCHOR_FAILED: an "
    "instrument fault that voids the whole sweep, stops the run, and is not evidence "
    "about the model. The quality band is tighter than any fresh-text band because this "
    "is the same text, the same instrument and the same model as the committed value."
)
RULE_BASELINE = (
    "The initialization checkpoint (step0) is a baseline measurement, not a near-zero "
    "alarm. Its held-out pooled quality over the same seven widths, at its own best "
    "readout, is recorded as the baseline floor B, and every formation threshold is "
    "measured up from it. The one void trigger is B at or above 0.7 times the anchor "
    "quality, which is 0.6134114: at or above that the fit would be reading most of its "
    "signal from untrained structure, which is VOID_CONTROL_FAILED, and the whole sweep "
    "is void. A range of 0.2 to 0.45 is expected, carried from the smaller sibling's "
    "measured 0.344634; that range is recorded and is not a pass bar."
)
RULE_CHECKPOINT = (
    "Recorded for one checkpoint, with no pass-or-fail call of its own: the held-out "
    "pooled quality at every readout, the best readout, the selected ridge strength, the "
    "four-part precision profile at that readout, the boundary ratio, the direction "
    "counts at 90, 95 and 99 percent, how flat quality is across readouts, and the "
    "per-width qualities. A selection sitting at an end of its search range is flagged. "
    "The two integrity gates decide whether these numbers may be read at all, and both "
    "decisions are carried in these values."
)
RULE_FORMATION_EXISTS = (
    "Some checkpoint's pooled held-out quality must reach the formed threshold, the "
    "baseline floor B plus nine tenths of the distance from B to the committed anchor "
    "quality 0.876302. No checkpoint reaching it is NOT_FORMED_BY_FINAL. Two committed "
    "measurements of this model at its final checkpoint, 0.876302 on this text and "
    "0.879879 on a second text, already establish that the counter exists there, so a "
    "sweep contradicting both impeaches the sweep: it is treated as an instrument fault, "
    "not as a genuine null result. The gate cell also forces the final checkpoint inside "
    "0.02 of the anchor, so once the gate passes this reading cannot fire."
)
RULE_BOUNDARY_PRESENT = (
    "The boundary ratio at the final checkpoint (step143000) must reach 0.3. The bar is "
    "absolute, with no baseline correction, because the smaller sibling's untrained "
    "checkpoint read -0.080316 there, which is effectively zero. Below 0.3 the reading is "
    "BOUNDARY_NOT_FORMED_BY_FINAL, which contradicts the same checkpoint's committed "
    "0.514227 and is therefore an instrument fault; the gate cell tests the same number "
    "first, so a passing gate makes this reading unreachable."
)
RULE_HOLD_BOUNDARY = (
    "Across every checkpoint at or after the formed checkpoint, the boundary ratio's "
    "range, its largest value minus its smallest, must stay within 0.15. The window "
    "starts at the shared formed checkpoint for both channels, which the committed gate "
    "record settles in its fourth answer of record. A wider range means the boundary "
    "keeps moving, which feeds a drift reading when the movement runs one way in training "
    "steps and an unclassifiable reading when it does not. If the boundary is absent at "
    "the final checkpoint this channel is recorded as NOT_PRESENT_AT_FINAL and takes no "
    "part in the stability call. A checkpoint at or after the formed one with no "
    "measurable ratio leaves the range unavailable, which is recorded as "
    "RANGE_NOT_COMPUTABLE and likewise takes no part in that call."
)
RULE_HOLD_QUALITY = (
    "Across every checkpoint at or after the formed checkpoint, the pooled quality's "
    "range, its largest value minus its smallest, must stay within 0.10. A wider range "
    "means the counter keeps moving, which feeds a drift reading when the movement runs "
    "one way in training steps and an unclassifiable reading when it does not. A "
    "checkpoint with no measurable quality would leave the range unavailable, recorded "
    "as RANGE_NOT_COMPUTABLE, which cannot arise while every checkpoint reports a number."
)
RULE_TIMING = (
    "The first checkpoint whose pooled quality reaches the formation threshold, the "
    "baseline floor B plus half the distance from B to the committed anchor quality, is "
    "the formation step. At or before step16000 it counts as early; after step16000 it "
    "counts as late. The committed gate record assigns that crossing this job in its "
    "second answer of record. The first checkpoint reaching the nine-tenths crossing is "
    "recorded beside it, with its own comparison against step16000, because the design "
    "object's third endpoint states the comparison that way. Both are recorded; this "
    "call follows the half-distance crossing."
)
RULE_BOUNDARY_TIMING = (
    "The first checkpoint whose boundary ratio reaches 0.3 is compared against the "
    "formation step. At or before it, the reading is BOUNDARY_WITH_COUNTER: fine-grained "
    "precision arrives with usability. Strictly later, the reading is "
    "BOUNDARY_AFTER_COUNTER: precision arrives only after the counter is already usable. "
    "A ratio that never reaches 0.3 is BOUNDARY_NOT_FORMED_BY_FINAL, which contradicts "
    "the gate cell and names an instrument fault. If no formation step exists the "
    "comparison has no left-hand side, which is recorded as "
    "BOUNDARY_TIMING_UNDEFINED_NO_FORMATION; a passing gate makes that unreachable, and "
    "it is named here rather than left to be invented later."
)
RULE_HEADLINE = (
    "A gate cell missing either of its two bars reads VOID_ANCHOR_FAILED and voids the "
    "sweep. A baseline at or above 0.6134114 reads VOID_CONTROL_FAILED and voids the "
    "sweep. Otherwise, no checkpoint reaching the formed threshold reads "
    "NOT_FORMED_BY_FINAL, treated as an instrument fault. Otherwise the stability test "
    "decides: inside both ranges reads FORMED_EARLY_AND_HELD when the formation step is "
    "at or before step16000 and FORMED_LATE_AND_HELD when it is later; outside either "
    "range with the movement running one way in steps reads FORMED_AND_DRIFTED; outside "
    "either range and not one-way reads UNSTABLE_NOT_CLASSIFIABLE, which is the "
    "stability test's own committed name for that outcome. If neither channel yields a "
    "range at all the stability test reads HOLD_TEST_NOT_COMPUTABLE, which twelve "
    "measured checkpoints cannot produce and which is named here rather than left to be "
    "invented later."
)
RULE_RECORDED_ONLY = (
    "Recorded only, with no threshold and no pass-or-fail call: the best readout's "
    "trajectory across the twelve checkpoints and the counter's localization moment, the "
    "direction counts at 90, 95 and 99 percent at each one, the selected ridge strength "
    "at each one, how flat quality is across readouts, the per-width qualities, whether "
    "the boundary already stood at the formation step, and the stability ranges two "
    "alternative readings of the design text would have used. The committed design gates "
    "on none of them."
)


def not_below(value, floor) -> bool:
    """Whether a value reaches a floor, judged in decimals rather than in bits.

    Every quantity compared here is recorded to six decimals, and every threshold is a
    short decimal too. Binary arithmetic cannot hold either exactly, which would put a
    value sitting exactly on a stated edge outside its own band. Rounding the difference
    to nine decimals removes that noise and moves no threshold: nine decimals is a
    thousand times finer than anything recorded.
    """
    return round(float(value) - float(floor), 9) >= 0.0


def not_above(value, limit) -> bool:
    """Whether a value stays within a limit, judged in decimals rather than in bits."""
    return round(float(value) - float(limit), 9) <= 0.0


def decide_setup(hash_ok: bool, revision, n_layers, hidden_size) -> str:
    if hash_ok and exact_revision_match(revision) and exact_config_match(n_layers, hidden_size):
        return "SETUP_OK"
    return "SETUP_FAILURE"


def decide_extract(n_variants: int, expected: int, flagged: bool) -> str:
    return "EXTRACT_OK" if (n_variants == expected and not flagged) else "EXTRACT_INCOMPLETE"


def decide_calibration(problems) -> str:
    return "CALIBRATION_OK" if not problems else "CALIBRATION_UNUSABLE"


def decide_anchor(pooled, ratio) -> str:
    """The gate cell, on both its bars. An absent or not-a-number value fails."""
    if pooled is None or pooled != pooled:
        return "VOID_ANCHOR_FAILED"
    if ratio is None or ratio != ratio:
        return "VOID_ANCHOR_FAILED"
    quality_ok = not_above(abs(float(pooled) - ANCHOR_POOLED), ANCHOR_TOLERANCE)
    ratio_ok = not_below(float(ratio), ANCHOR_RATIO_FLOOR)
    return "ANCHOR_IN_BAND" if (quality_ok and ratio_ok) else "VOID_ANCHOR_FAILED"


def decide_baseline(pooled) -> str:
    """The baseline floor. It voids only from above, never for being low."""
    if pooled is None or pooled != pooled:
        return "VOID_CONTROL_FAILED"
    return ("VOID_CONTROL_FAILED" if not_below(pooled, BASELINE_VOID_FLOOR)
            else "BASELINE_RECORDED")


def baseline_inside_expected(pooled) -> bool:
    """Whether the baseline landed in the expected range. Recorded, never gated."""
    if pooled is None or pooled != pooled:
        return False
    return (not_below(pooled, BASELINE_EXPECTED_LOW)
            and not_above(pooled, BASELINE_EXPECTED_HIGH))


def formation_thresholds(baseline, anchor=ANCHOR_POOLED):
    """The two crossings, measured up from the baseline floor toward the anchor.

    Returns (formation threshold, formed threshold), or (None, None) when the baseline
    is absent. The first is half the distance from the baseline to the anchor, the
    second nine tenths of it.
    """
    if baseline is None or baseline != baseline:
        return None, None
    b = float(baseline)
    gap = float(anchor) - b
    return b + FORMATION_SHARE * gap, b + FORMED_SHARE * gap


def decide_formation_exists(best_quality, formed_threshold) -> str:
    if (best_quality is None or best_quality != best_quality
            or formed_threshold is None):
        return "NOT_FORMED_BY_FINAL"
    return ("FORMATION_REACHED" if not_below(best_quality, formed_threshold)
            else "NOT_FORMED_BY_FINAL")


def decide_boundary_presence(ratio_at_final) -> str:
    if ratio_at_final is None or ratio_at_final != ratio_at_final:
        return "BOUNDARY_NOT_FORMED_BY_FINAL"
    return ("BOUNDARY_PRESENT" if not_below(ratio_at_final, BOUNDARY_PRESENT_FLOOR)
            else "BOUNDARY_NOT_FORMED_BY_FINAL")


def decide_hold_channel(spread, limit) -> str:
    """One stability channel: inside its range, outside it, or not computable."""
    if spread is None or spread != spread:
        return "RANGE_NOT_COMPUTABLE"
    return "WITHIN_BAND" if not_above(spread, limit) else "OUTSIDE_BAND"


def series_range(values):
    """Largest value minus smallest. None when any value is absent."""
    vals = list(values)
    if not vals or any(v is None or v != v for v in vals):
        return None
    return float(max(vals)) - float(min(vals))


def is_monotone(values) -> bool:
    """Whether a series only ever rises or only ever falls, ties allowed.

    An absent value makes the question unanswerable, and that reads as not one-way.
    """
    vals = list(values)
    if any(v is None or v != v for v in vals):
        return False
    ups = all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1))
    downs = all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
    return bool(ups or downs)


def decide_hold(boundary_channel: str, quality_channel: str,
                boundary_monotone: bool, quality_monotone: bool) -> str:
    """The fork's decider, over whichever channels are live.

    A channel is live when it produced a range to judge. The boundary channel is not
    live when the boundary is absent at the final checkpoint, which the gate cell
    already forbids; the quality readings would still stand on their own.
    """
    live = []
    if boundary_channel in ("WITHIN_BAND", "OUTSIDE_BAND"):
        live.append(("boundary", boundary_channel, boundary_monotone))
    if quality_channel in ("WITHIN_BAND", "OUTSIDE_BAND"):
        live.append(("quality", quality_channel, quality_monotone))
    if not live:
        return "HOLD_TEST_NOT_COMPUTABLE"
    outside = [item for item in live if item[1] == "OUTSIDE_BAND"]
    if not outside:
        return "HELD"
    if all(item[2] for item in outside):
        return "DRIFTED"
    return "UNSTABLE_NOT_CLASSIFIABLE"


def decide_timing(formation_step) -> str:
    if formation_step is None:
        return "TIMING_UNDEFINED_NO_FORMATION"
    return ("EARLY_AT_OR_BEFORE_STEP16000" if int(formation_step) <= EARLY_CUTOFF_STEP
            else "LATE_AFTER_STEP16000")


def decide_boundary_timing(crossing_step, formation_step) -> str:
    """Where the boundary's own first crossing sits against the formation step."""
    if crossing_step is None:
        return "BOUNDARY_NOT_FORMED_BY_FINAL"
    if formation_step is None:
        return "BOUNDARY_TIMING_UNDEFINED_NO_FORMATION"
    return ("BOUNDARY_WITH_COUNTER" if int(crossing_step) <= int(formation_step)
            else "BOUNDARY_AFTER_COUNTER")


def decide_localization(best_layer, best_quality, median_quality) -> str:
    """Whether one checkpoint shows a clear interior peak across readouts.

    Recorded only. The peak must sit off both ends of the readout range and must beat
    the median readout's quality by at least 0.1.
    """
    if best_layer is None or best_quality is None or median_quality is None:
        return "LOCALIZATION_NOT_COMPUTABLE"
    if best_quality != best_quality or median_quality != median_quality:
        return "LOCALIZATION_NOT_COMPUTABLE"
    interior = int(best_layer) not in LAYER_EDGES
    margin = float(best_quality) - float(median_quality)
    if interior and not_below(margin, LOCALIZATION_MARGIN):
        return "LOCALIZED_INTERIOR_PEAK"
    return "NOT_YET_LOCALIZED"


def decide_headline(anchor: str, baseline: str, formation: str, timing: str,
                    hold: str) -> str:
    """One call for the whole sweep, from the calls already made."""
    if anchor != "ANCHOR_IN_BAND":
        return "VOID_ANCHOR_FAILED"
    if baseline != "BASELINE_RECORDED":
        return "VOID_CONTROL_FAILED"
    if formation != "FORMATION_REACHED":
        return "NOT_FORMED_BY_FINAL"
    if hold == "HELD":
        return ("FORMED_EARLY_AND_HELD" if timing == "EARLY_AT_OR_BEFORE_STEP16000"
                else "FORMED_LATE_AND_HELD")
    if hold == "DRIFTED":
        return "FORMED_AND_DRIFTED"
    return hold


def first_reaching(rows, threshold, key: str = "pooled_quality"):
    """The earliest checkpoint whose named quantity reaches a threshold.

    `rows` is one dict per checkpoint, each carrying its step, and it is read in step
    order. Returns (step, revision) or (None, None).
    """
    if threshold is None:
        return None, None
    for row in sorted(rows, key=lambda r: r["step"]):
        v = row.get(key)
        if v is None or v != v:
            continue
        if not_below(v, threshold):
            return int(row["step"]), row["revision"]
    return None, None


def first_localized(rows):
    """The earliest checkpoint showing a clear interior peak. Recorded only."""
    for row in sorted(rows, key=lambda r: r["step"]):
        if row.get("localization_decision") == "LOCALIZED_INTERIOR_PEAK":
            return int(row["step"]), row["revision"]
    return None, None


def sweep_readings(rows, anchor_decision: str, baseline_decision: str) -> dict:
    """Every cross-checkpoint number and call, from the per-checkpoint rows.

    `rows` is one dict per checkpoint with the keys named in ROW_KEYS_NEEDED. Pure
    arithmetic over numbers, so the self-test plants rows whose calls are known.
    """
    ordered = sorted(rows, key=lambda r: r["step"])
    steps = [int(r["step"]) for r in ordered]
    if len(set(steps)) != len(steps):
        raise Halt("two rows carry the same training step, so the trajectory is ambiguous")
    quality = {int(r["step"]): r.get("pooled_quality") for r in ordered}
    ratio = {int(r["step"]): r.get("boundary_ratio") for r in ordered}

    final_step = step_of(FINAL_REVISION)
    baseline_step = step_of(BASELINE_REVISION)
    if final_step not in quality:
        raise Halt("the final checkpoint (%s) is missing from the trajectory"
                   % FINAL_REVISION)
    if baseline_step not in quality:
        raise Halt("the initialization checkpoint (%s) is missing, so the baseline floor "
                   "every formation threshold is measured from does not exist"
                   % BASELINE_REVISION)
    final_quality = quality[final_step]
    baseline = quality[baseline_step]
    qualities = [q for q in quality.values() if q is not None and q == q]
    best_quality = max(qualities) if qualities else None

    formation_threshold, formed_threshold = formation_thresholds(baseline)
    formation_step, formation_revision = first_reaching(ordered, formation_threshold)
    formed_step, formed_revision = first_reaching(ordered, formed_threshold)

    # Recorded only, and gating nothing: the same two crossings computed against this
    # run's own measured final quality instead of the committed anchor quality. The two
    # readings of the design's word "anchor" differ by at most the gate's own 0.02 band.
    alt_formation_threshold, alt_formed_threshold = formation_thresholds(
        baseline, anchor=(ANCHOR_POOLED if final_quality is None else final_quality))
    alt_formation_step, _ = first_reaching(ordered, alt_formation_threshold)
    alt_formed_step, _ = first_reaching(ordered, alt_formed_threshold)

    after = [r for r in ordered if formed_step is not None and int(r["step"]) >= formed_step]
    quality_after = [r.get("pooled_quality") for r in after]
    ratio_after = [r.get("boundary_ratio") for r in after]

    presence = decide_boundary_presence(ratio.get(final_step))
    quality_spread = series_range(quality_after)
    ratio_spread = series_range(ratio_after)
    quality_channel = decide_hold_channel(quality_spread, HOLD_QUALITY_RANGE)
    if presence == "BOUNDARY_PRESENT":
        boundary_channel = decide_hold_channel(ratio_spread, HOLD_BOUNDARY_RANGE)
    else:
        boundary_channel = "NOT_PRESENT_AT_FINAL"
    quality_monotone = is_monotone(quality_after)
    ratio_monotone = is_monotone(ratio_after)

    crossing_step, crossing_revision = first_reaching(ordered, BOUNDARY_PRESENT_FLOOR,
                                                      key="boundary_ratio")
    # Recorded only: the boundary's stability measured from its own first crossing
    # instead of the shared formed checkpoint. The committed gate record's fourth answer
    # settles that the shared checkpoint governs; this number makes that choice visible.
    from_crossing = [r for r in ordered
                     if crossing_step is not None and int(r["step"]) >= crossing_step]
    ratio_from_crossing = series_range([r.get("boundary_ratio") for r in from_crossing])

    localization_step, localization_revision = first_localized(ordered)

    formation = decide_formation_exists(best_quality, formed_threshold)
    hold = decide_hold(boundary_channel, quality_channel, ratio_monotone,
                       quality_monotone)
    timing = decide_timing(formation_step)
    timing_formed_marker = decide_timing(formed_step)
    headline = decide_headline(anchor_decision, baseline_decision, formation, timing,
                               hold)
    boundary_timing = decide_boundary_timing(crossing_step, formation_step)

    return {
        "n_checkpoints": len(ordered),
        "steps_in_order": steps,
        "revisions_in_order": [r["revision"] for r in ordered],
        "pooled_quality_by_step": {str(s): quality[s] for s in steps},
        "boundary_ratio_by_step": {str(s): ratio[s] for s in steps},
        "best_layer_by_step": {str(int(r["step"])): r.get("best_layer") for r in ordered},
        "ridge_strength_by_step":
            {str(int(r["step"])): r.get("ridge_strength") for r in ordered},
        "dimensions_by_step": {str(int(r["step"])): r.get("dimensions") for r in ordered},
        "coordinate_flags_by_step":
            {str(int(r["step"])): list(r.get("coordinate_flags") or []) for r in ordered},
        "degenerate_coordinates_by_step":
            {str(int(r["step"])): bool(r.get("degenerate_coordinates")) for r in ordered},
        "layer_median_quality_by_step":
            {str(int(r["step"])): r.get("layer_median_quality") for r in ordered},
        "peak_over_layer_median_by_step":
            {str(int(r["step"])): r.get("peak_over_layer_median") for r in ordered},
        "localization_by_step":
            {str(int(r["step"])): r.get("localization_decision") for r in ordered},
        "quality_by_width_by_step":
            {str(int(r["step"])): r.get("quality_by_width") for r in ordered},
        "layer_quality_spread_by_step":
            {str(int(r["step"])): r.get("layer_quality_spread") for r in ordered},
        "final_step": final_step,
        "final_quality": final_quality,
        "baseline_step": baseline_step,
        "baseline_quality": baseline,
        "baseline_void_floor": BASELINE_VOID_FLOOR,
        "baseline_expected_range": [BASELINE_EXPECTED_LOW, BASELINE_EXPECTED_HIGH],
        "baseline_inside_expected_range": baseline_inside_expected(baseline),
        "committed_anchor_quality": ANCHOR_POOLED,
        "highest_quality_any_checkpoint": best_quality,
        "formation_share_of_gap": FORMATION_SHARE,
        "formation_threshold": formation_threshold,
        "formation_step": formation_step,
        "formation_revision": formation_revision,
        "formed_share_of_gap": FORMED_SHARE,
        "formed_threshold": formed_threshold,
        "formed_step": formed_step,
        "formed_revision": formed_revision,
        "formation_threshold_from_measured_final_recorded_only": alt_formation_threshold,
        "formation_step_from_measured_final_recorded_only": alt_formation_step,
        "formed_threshold_from_measured_final_recorded_only": alt_formed_threshold,
        "formed_step_from_measured_final_recorded_only": alt_formed_step,
        "checkpoints_at_or_after_formed": [r["revision"] for r in after],
        "boundary_ratio_at_final": ratio.get(final_step),
        "boundary_ratio_at_baseline": ratio.get(baseline_step),
        "boundary_ratio_at_formation_step":
            (None if formation_step is None else ratio.get(formation_step)),
        "boundary_ratio_at_formed_step":
            (None if formed_step is None else ratio.get(formed_step)),
        "boundary_crossing_step": crossing_step,
        "boundary_crossing_revision": crossing_revision,
        "boundary_crossing_floor": BOUNDARY_PRESENT_FLOOR,
        "boundary_range_from_its_own_crossing_recorded_only": ratio_from_crossing,
        "localization_step_recorded_only": localization_step,
        "localization_revision_recorded_only": localization_revision,
        "localization_margin": LOCALIZATION_MARGIN,
        "quality_range_after_formed": quality_spread,
        "boundary_range_after_formed": ratio_spread,
        "quality_series_one_way_in_steps": quality_monotone,
        "boundary_series_one_way_in_steps": ratio_monotone,
        "hold_test_quality_channel": quality_channel,
        "hold_test_boundary_channel": boundary_channel,
        "hold_test_quality_limit": HOLD_QUALITY_RANGE,
        "hold_test_boundary_limit": HOLD_BOUNDARY_RANGE,
        "boundary_presence_decision": presence,
        "boundary_presence_floor": BOUNDARY_PRESENT_FLOOR,
        "boundary_timing_decision": boundary_timing,
        "formation_exists_decision": formation,
        "hold_test_decision": hold,
        "formation_timing_decision": timing,
        "formation_timing_from_formed_marker_recorded_only": timing_formed_marker,
        "early_late_cutoff_step": EARLY_CUTOFF_STEP,
        "anchor_gate_decision": anchor_decision,
        "baseline_decision": baseline_decision,
        "headline": headline,
        "boundary_trajectory_reading": boundary_timing,
    }


# ---------------------------------------------------------------------------
# The committed label-reachability control, replayed as data.
#
# records/CONTROL_d4_1_label_reachability.txt is committed by the gate record and states
# five synthetic trajectories, their five labels, and the crossings of the first four:
#   early-sigmoid-flat  -> FORMED_EARLY_AND_HELD     (formation 2000, formed 4000)
#   late-sigmoid-flat   -> FORMED_LATE_AND_HELD      (formation 64000, formed 96000)
#   form-then-drift-up  -> FORMED_AND_DRIFTED        (formation 2000, formed 4000)
#   noisy-plateau       -> UNSTABLE_NOT_CLASSIFIABLE (formation 2000, formed 4000)
#   never-forms         -> NOT_FORMED_BY_FINAL
# The quality series below start at the planted baseline floor, so the same shapes can
# be replayed at several floors. The self-test replays all five at three floors inside
# the design's expected range and checks every label and every crossing.
# ---------------------------------------------------------------------------

CONTROL_BASELINE = 0.35        # the floor these shapes are replayed at, near the
                              # sibling's measured 0.344634
CONTROL_BASELINES_ALSO_CHECKED = [0.25, 0.40]

# Eleven values each, for the eleven checkpoints after the baseline one.
CONTROL_QUALITY_AFTER_BASELINE = {
    "early-sigmoid-flat": [0.40, 0.55, 0.65, 0.84, 0.85, 0.855, 0.86, 0.865, 0.87,
                           0.874, 0.876],
    "late-sigmoid-flat": [0.36, 0.37, 0.38, 0.40, 0.43, 0.47, 0.55, 0.70, 0.85, 0.87,
                          0.876],
    "form-then-drift-up": [0.40, 0.55, 0.65, 0.84, 0.85, 0.855, 0.86, 0.865, 0.87,
                           0.874, 0.876],
    "noisy-plateau": [0.40, 0.55, 0.65, 0.84, 0.70, 0.86, 0.72, 0.87, 0.74, 0.88,
                      0.876],
    "never-forms": [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56],
}

# Twelve values each: the boundary ratio at every checkpoint, starting at initialization.
CONTROL_RATIO = {
    "early-sigmoid-flat": [-0.08, -0.05, 0.20, 0.35, 0.50, 0.50, 0.50, 0.51, 0.50,
                           0.51, 0.50, 0.514],
    "late-sigmoid-flat": [-0.08, -0.07, -0.05, 0.00, 0.10, 0.20, 0.30, 0.40, 0.48,
                          0.50, 0.51, 0.514],
    "form-then-drift-up": [-0.08, -0.05, 0.20, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55,
                           0.60, 0.65, 0.70],
    "noisy-plateau": [-0.08, -0.05, 0.20, 0.35, 0.50, 0.50, 0.50, 0.51, 0.50, 0.51,
                      0.50, 0.514],
    "never-forms": [-0.08, -0.07, -0.05, 0.00, 0.10, 0.20, 0.30, 0.40, 0.48, 0.50,
                    0.51, 0.514],
}

CONTROL_EXPECTED = {
    "early-sigmoid-flat": ("FORMED_EARLY_AND_HELD", 2000, 4000),
    "late-sigmoid-flat": ("FORMED_LATE_AND_HELD", 64000, 96000),
    "form-then-drift-up": ("FORMED_AND_DRIFTED", 2000, 4000),
    "noisy-plateau": ("UNSTABLE_NOT_CLASSIFIABLE", 2000, 4000),
    "never-forms": ("NOT_FORMED_BY_FINAL", None, None),
}
CONTROL_NAMES = ["early-sigmoid-flat", "late-sigmoid-flat", "form-then-drift-up",
                 "noisy-plateau", "never-forms"]


def control_trajectory(name: str, baseline: float = CONTROL_BASELINE):
    """One committed synthetic trajectory as (quality series, ratio series)."""
    if name not in CONTROL_QUALITY_AFTER_BASELINE:
        raise Halt("no committed control trajectory named %r" % name)
    return ([float(baseline)] + list(CONTROL_QUALITY_AFTER_BASELINE[name]),
            list(CONTROL_RATIO[name]))


# ---------------------------------------------------------------------------
# Part 9. Shared plumbing.
# ---------------------------------------------------------------------------

TRAIN_GROUPS = ["fold%d" % f for f in range(N_FOLDS)]
TEST_GROUPS = ["test_w%d" % w for w in USE_WIDTHS]


def group_names():
    """The groups the running totals are kept for, for one checkpoint at a time.

    Five groups split the training documents, which is what the strength search reads.
    One further group per wrap width holds the held-out documents at that width, so the
    pooled quality and the per-width qualities come from the same totals.
    """
    return list(TRAIN_GROUPS) + list(TEST_GROUPS)


def totals_bytes(n_read: int = EXPECT_LAYERS + 1, hidden: int = EXPECT_HIDDEN,
                 n_groups: int = None) -> int:
    """Size of one checkpoint's running totals, so the log states it before the work."""
    if n_groups is None:
        n_groups = len(group_names())
    per_group = n_read * hidden * hidden * 8 + 2 * n_read * hidden * 8
    return per_group * n_groups


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
    p = find_first([STIMULUS_NAME, "records/" + STIMULUS_NAME])
    if p is None:
        raise Halt("%s is absent; the design says halt, never regenerate, re-wrap, or "
                   "reconstruct it" % STIMULUS_NAME)
    return p


def design_record_path(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return find_first([
        "seedD4_1_design_diff_triage.json",              # box layout, flat
        "records/seedD4_1_design_diff_triage.json",      # workstation layout
    ])


def gate_record_path():
    return find_first(["GATE_COMMIT_seedD4_1.md", "records/GATE_COMMIT_seedD4_1.md"])


def control_record_path():
    """The committed label-reachability control, hashed into every record when shipped."""
    return find_first(["CONTROL_d4_1_label_reachability.txt",
                       "records/CONTROL_d4_1_label_reachability.txt"])


def hf_token():
    """Access token, and the name of the variable it came from.

    The build brief names HF_KEY_ONE. The launcher forwards the same secret as
    HF_TOKEN, and the bootstrap script also exports HUGGING_FACE_HUB_TOKEN. All three
    are tried, in that order. These checkpoints are public, so an absent token is fine
    and is recorded as such. No token value is ever printed or written.
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
    if list(meta.get("widths", [])) != STIMULUS_WIDTHS:
        problems.append("wrap widths %r, expected %r" % (meta.get("widths"),
                                                        STIMULUS_WIDTHS))
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
        missing = [str(w) for w in USE_WIDTHS if str(w) not in d.get("wrapped", {})]
        if missing:
            problems.append("document %r lacks widths %r" % (d.get("id"), missing))
            break
    if problems:
        raise Halt("the text file does not match the design: " + "; ".join(problems))
    return meta, docs


def fold_of_documents(train_ids):
    """Which of the five groups each training document belongs to.

    Splitting by document, not by position, keeps sentences of one document out of both
    sides of a split. The one fixed base seed makes the split repeatable, and it is the
    same split at every checkpoint.
    """
    import numpy as np
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(train_ids)))
    assign = {}
    for rank, pos in enumerate(order):
        assign[train_ids[int(pos)]] = rank % N_FOLDS
    return assign


def sum_groups(tot, names, layer):
    """Running totals of several groups added together, at one readout."""
    g = sum(tot["gram"][k][layer] for k in names)
    v = sum(tot["vec_sum"][k][layer] for k in names)
    c = sum(tot["cross"][k][layer] for k in names)
    ls = sum(tot["label_sum"][k] for k in names)
    lq = sum(tot["label_sq"][k] for k in names)
    n = sum(tot["n_pos"][k] for k in names)
    return g, v, c, ls, lq, n


def paths_for(results_dir):
    out = Path(results_dir) / "out"
    work = Path(results_dir) / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def record_inputs(prov, stim, design, gate, control, extra=()):
    items = [prov.describe_input(stim, n_items=N_DOCS * len(USE_WIDTHS))]
    for path in (design, gate, control):
        if path is not None:
            items.append(prov.describe_input(path, n_items=1))
    for path, n in extra:
        items.append(prov.describe_input(path, n_items=n))
    return items


def base_cell(stage, arm, revision, endpoint=None):
    cell = {"model": MODEL_SLUG, "revision": revision, "stage": stage, "arm": arm}
    if endpoint is not None:
        cell["endpoint"] = endpoint
    return cell


def write_record(prov, out_dir, cell, **kwargs):
    path = Path(out_dir) / (prov.cell_slug(cell) + ".json")
    prov.write_result(path, cell=cell, **kwargs)
    print("[record] %s -> %s" % (kwargs.get("decision", ""), path))
    return path


def gate_block(anchor_decision: str, baseline_decision: str, anchor_quality=None,
               anchor_ratio=None, baseline_quality=None) -> dict:
    """Both integrity gates' own decisions, for embedding in another record's values.

    No record of this sweep can be quoted without its void status attached. While the
    sweep is still measuring, a gate that has not been reached yet says so plainly
    rather than passing silently.
    """
    voided = (anchor_decision not in ("ANCHOR_IN_BAND", "NOT_YET_MEASURED")
              or baseline_decision not in ("BASELINE_RECORDED", "NOT_YET_MEASURED"))
    return {
        "anchor_gate_decision": anchor_decision,
        "anchor_gate_anchor_value": ANCHOR_POOLED,
        "anchor_gate_tolerance": ANCHOR_TOLERANCE,
        "anchor_gate_ratio_floor": ANCHOR_RATIO_FLOOR,
        "anchor_gate_quality":
            (None if anchor_quality is None else round(float(anchor_quality), 6)),
        "anchor_gate_boundary_ratio":
            (None if anchor_ratio is None else round(float(anchor_ratio), 6)),
        "baseline_decision": baseline_decision,
        "baseline_void_floor": BASELINE_VOID_FLOOR,
        "baseline_quality":
            (None if baseline_quality is None else round(float(baseline_quality), 6)),
        "this_record_voided_by_a_gate": voided,
    }


def act_file(work_dir, revision, layer):
    return Path(work_dir) / ("act_%s_layer%02d.f32" % (revision, layer))


def labels_file(work_dir, revision):
    return Path(work_dir) / ("labels_%s.npz" % revision)


def cache_dir_for(work_dir, revision):
    return Path(work_dir) / "downloads" / revision


def sweep_table_path(out_dir):
    return Path(out_dir) / "sweep_table.json"


def write_sweep_table(out_dir, rows, anchor_decision, baseline_decision, complete):
    """The plain hand-off from the sweep to the last stage, rewritten as it grows."""
    payload = {"model_id": MODEL_ID, "revisions_in_training_order": REVISIONS,
               "measurement_order": MEASURE_ORDER,
               "anchor_gate_decision": anchor_decision,
               "baseline_decision": baseline_decision,
               "n_rows": len(rows), "complete": bool(complete), "rows": rows}
    with open(sweep_table_path(out_dir), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def read_sweep_table(out_dir):
    path = sweep_table_path(out_dir)
    if not path.exists():
        raise Halt("no per-checkpoint table at %s; the sweep must run first" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def delete_tree(path):
    """Remove a folder and everything under it, if it is there."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    return not p.exists()


def delete_file(path):
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            return False
    return not p.exists()


# ---------------------------------------------------------------------------
# Part 10. The sweep: one checkpoint at a time, in the committed order.
#
# Per checkpoint: download, check the configuration, one forward pass per text variant,
# the running totals, the ridge strength from training documents only, the best readout
# from held-out documents, the four-part profile and the direction counts at that
# readout, the records, then delete the weights and the working vectors. The weights are
# deleted as soon as the forward passes end, so the disk holds one checkpoint's download
# at a time.
# ---------------------------------------------------------------------------

def measure_revision(revision, order_index, prov, out_dir, work_dir, stim, design,
                     gate, control, docs, folds, anchor_decision, baseline_decision):
    """Measure one checkpoint. Returns the row the trajectory stage reads."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    step = step_of(revision)
    n_read = EXPECT_LAYERS + 1                        # readout 0 is the embedding output
    expected_variants = N_DOCS * len(USE_WIDTHS)
    cache = cache_dir_for(work_dir, revision)
    cache.mkdir(parents=True, exist_ok=True)
    token, token_env = hf_token()
    kw = {"revision": revision, "cache_dir": str(cache)}
    if token:
        kw["token"] = token
    gates = gate_block(anchor_decision, baseline_decision)

    print("[sweep] checkpoint %s (step %d), measurement %d of %d, downloading into %s"
          % (revision, step, order_index + 1, len(MEASURE_ORDER), cache))
    config = AutoConfig.from_pretrained(MODEL_ID, **kw)
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    context_limit = int(getattr(config, "max_position_embeddings", CONTEXT_LIMIT_FALLBACK)
                        or CONTEXT_LIMIT_FALLBACK)
    actual_hash = sha256_file(stim)
    hash_ok = exact_hash_match(actual_hash, STIMULUS_SHA256)
    setup_decision = decide_setup(hash_ok, revision, n_layers, hidden)
    setup_values = {"stimulus_sha256": actual_hash,
                    "stimulus_sha256_expected": STIMULUS_SHA256,
                    "stimulus_hash_match": hash_ok,
                    "revision": revision, "training_step": step,
                    "revision_is_one_of_the_twelve": exact_revision_match(revision),
                    "measurement_order_index": order_index,
                    "n_layers": n_layers, "hidden_size": hidden,
                    "n_layers_expected": EXPECT_LAYERS,
                    "hidden_size_expected": EXPECT_HIDDEN,
                    "context_limit": context_limit,
                    "model_id": MODEL_ID,
                    "design_record_found": str(design) if design else None,
                    "design_record_expected_sha256": DESIGN_SHA256,
                    "gate_record_found": str(gate) if gate else None,
                    "gate_record_expected_sha256": GATE_SHA256,
                    "control_record_found": str(control) if control else None,
                    "control_record_expected_sha256": CONTROL_SHA256,
                    "access_token_source": token_env}
    setup_values.update(gates)
    write_record(
        prov, out_dir, base_cell("extract", "setup_check", revision),
        inputs=record_inputs(prov, stim, design, gate, control),
        metric="Committed text file hash, the revision name, and the downloaded configuration's layer count and hidden size.",
        values=setup_values,
        threshold={"stimulus_sha256": STIMULUS_SHA256, "n_layers": EXPECT_LAYERS,
                   "hidden_size": EXPECT_HIDDEN, "revision_one_of": REVISIONS},
        decision_rule=RULE_SETUP,
        decision=setup_decision,
        notes=("Layer count 24 and hidden size 2048 are recalled from the Pythia paper, "
               "arXiv 2304.01373 table 1, and checked against this checkpoint's own "
               "download, which is authoritative. A mismatch halts before any "
               "measurement. The text is loaded exactly as committed and is never "
               "regenerated or re-wrapped; it is the same file behind the committed "
               "0.876302 and 0.514227. " + NO_CORPUS_NOTE + " " + RERUN_NOTE),
    )
    if setup_decision != "SETUP_OK":
        raise Halt("the setup check failed at %s: hash match=%s layers=%r hidden=%r "
                   "(expected %d layers, hidden %d)"
                   % (revision, hash_ok, n_layers, hidden, EXPECT_LAYERS, EXPECT_HIDDEN))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    if not getattr(tok, "is_fast", False):
        raise Halt("this tokenizer reports no character offsets, so labels cannot be "
                   "computed")
    tokenizer_class = type(tok).__name__

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    dtype_name = "float16" if device == "cuda" else "float32"
    print("[sweep] %s: loading in %s on %s" % (revision, dtype_name, device))
    if device != "cuda":
        print("[sweep] WARNING: no graphics card visible, so half precision is replaced "
              "by single precision on the processor")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **kw)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    groups = group_names()
    print("[sweep] %s: running totals %d groups times %d readouts at hidden %d = %.2f GB"
          % (revision, len(groups), n_read, EXPECT_HIDDEN, totals_bytes(n_read) / 1e9))
    gram = {g: np.zeros((n_read, EXPECT_HIDDEN, EXPECT_HIDDEN), dtype=np.float64)
            for g in groups}
    vec_sum = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    cross = {g: np.zeros((n_read, EXPECT_HIDDEN), dtype=np.float64) for g in groups}
    label_sum = {g: 0.0 for g in groups}
    label_sq = {g: 0.0 for g in groups}
    n_pos = {g: 0 for g in groups}

    # Per-position internal vectors are kept for the width-140 held-out documents only.
    # The best readout is not known until the fit is done, so all 25 are written and all
    # but the best are deleted as soon as it is. That is about 2.2 gigabytes for a few
    # seconds, against about 54 gigabytes if every width were kept.
    act_files = [open(act_file(work_dir, revision, l), "wb") for l in range(n_read)]
    held = {"doc": [], "char": [], "token": [], "line": [], "tok_index": [],
            "tok_text": []}
    train_profile_counts = {}
    rows_var_path = Path(out_dir) / ("rows_variants_%s.jsonl" % revision)
    rows_var = open(rows_var_path, "w", encoding="utf-8")

    started = time.time()
    n_variants = 0
    all_flags = {}
    try:
        for doc in docs:
            did = int(doc["id"])
            split = doc["split"]
            for width in USE_WIDTHS:
                group = ("test_w%d" % width) if split == "test" else ("fold%d" % folds[did])
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
                want_act = (width == PROFILE_WIDTH) and (split == "test")
                for l in range(n_read):
                    x = states[l][0].index_select(0, idx).float()
                    if not bool(torch.isfinite(x).all()):
                        raise Halt("checkpoint %s produced a value that is not finite at "
                                   "readout %d on document %d at width %d, so its "
                                   "totals cannot be trusted" % (revision, l, did, width))
                    gram[group][l] += (x.T @ x).double().cpu().numpy()
                    vec_sum[group][l] += x.sum(0).double().cpu().numpy()
                    cross[group][l] += (x.T @ y_t).double().cpu().numpy()
                    if want_act:
                        x.cpu().numpy().astype(np.float32).tofile(act_files[l])
                label_sum[group] += float(y.sum())
                label_sq[group] += float((y * y).sum())
                n_pos[group] += len(keep)

                if width == PROFILE_WIDTH:
                    if split == "test":
                        held["doc"].extend([did] * len(keep))
                        held["char"].extend([char_labels[i] for i in keep])
                        held["token"].extend([token_labels[i] for i in keep])
                        held["line"].extend([line_nums[i] for i in keep])
                        held["tok_index"].extend(keep)
                        held["tok_text"].extend([pieces[i] for i in keep])
                    else:
                        for i in keep:
                            lab = char_labels[i]
                            train_profile_counts[lab] = train_profile_counts.get(lab, 0) + 1
                rows_var.write(json.dumps({
                    "revision": revision, "training_step": step,
                    "doc": did, "width": width, "split": split, "group": group,
                    "n_chars": len(text), "n_tokens": len(ids), "n_labelled": len(keep),
                    "n_lines": text.count("\n") + 1,
                    "degenerate_flags": flags,
                    "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text}, ensure_ascii=False) + "\n")
                n_variants += 1
                del out, states
        elapsed_pass = time.time() - started
    finally:
        rows_var.close()
        for fh in act_files:
            fh.close()

    # The weights go now, before any analysis. One checkpoint's download at a time.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    weights_deleted = delete_tree(cache)
    print("[sweep] %s: %d variants in %.1f s; downloaded weights deleted: %s"
          % (revision, n_variants, elapsed_pass, weights_deleted))

    np.savez(labels_file(work_dir, revision),
             doc=np.array(held["doc"], dtype=np.int32),
             char_label=np.array(held["char"], dtype=np.int32),
             token_label=np.array(held["token"], dtype=np.int32),
             line=np.array(held["line"], dtype=np.int32),
             tok_index=np.array(held["tok_index"], dtype=np.int32))
    n_held_rows = len(held["char"])

    capped = any("token_cap" in f for f in all_flags.values())
    extract_decision = decide_extract(n_variants, expected_variants, capped)
    extract_values = {"revision": revision, "training_step": step,
                      "measurement_order_index": order_index,
                      "n_variants": n_variants,
                      "n_variants_expected": expected_variants,
                      "widths_read": USE_WIDTHS,
                      "widths_in_the_file_not_read": [w for w in STIMULUS_WIDTHS
                                                     if w not in USE_WIDTHS],
                      "n_positions_by_group": {g: n_pos[g] for g in groups},
                      "n_positions_total": int(sum(n_pos.values())),
                      "n_readouts": n_read, "hidden_size": EXPECT_HIDDEN,
                      "device": device, "dtype": dtype_name,
                      "tokenizer_class": tokenizer_class,
                      "running_totals_bytes": totals_bytes(n_read),
                      "per_position_vectors_saved_for_width": PROFILE_WIDTH,
                      "per_position_vectors_saved_for_split": "test",
                      "held_out_profile_rows": n_held_rows,
                      "seconds_forward_passes": round(elapsed_pass, 1),
                      "degenerate_variants": all_flags,
                      "downloaded_weights_deleted": weights_deleted,
                      "generation_performed": False,
                      "verbatim_variant_rows_file": rows_var_path.name}
    extract_values.update(gates)
    write_record(
        prov, out_dir, base_cell("extract", "forward_passes", revision),
        inputs=record_inputs(prov, stim, design, gate, control),
        metric="Counts from one forward pass per text variant at one checkpoint, and the running totals saved for its fits.",
        values=extract_values,
        threshold={"n_variants": expected_variants},
        decision_rule=RULE_EXTRACT,
        decision=extract_decision,
        notes=("Readout 0 is the embedding output and readout 24 is the last layer. This "
               "library applies the model's final normalisation to readout 24 only, "
               "which is a property of the library, recorded here rather than changed. "
               "Every one of the 336 text variants gets one deterministic forward pass, "
               "one variant at a time; there is nothing to group, because each variant is "
               "a whole document and its labels are per position. The two wide widths in "
               "the committed file, 200 and 280, are not read at this rung: the committed "
               "gate record's first answer names the seven the file defines, and nothing "
               "here is fitted, gated or recorded on the other two. Every variant's text "
               "is saved verbatim with its own hash in the file named above. No text is "
               "generated at any point. " + RERUN_NOTE),
    )
    if extract_decision != "EXTRACT_OK":
        raise Halt("only %d of %d text variants were read at %s"
                   % (n_variants, expected_variants, revision))
    if not weights_deleted:
        raise Halt("the downloaded weights of %s are still at %s after deletion; eleven "
                   "more downloads would fill the disk, so the sweep stops here"
                   % (revision, cache))

    tot = {"gram": gram, "vec_sum": vec_sum, "cross": cross, "label_sum": label_sum,
           "label_sq": label_sq, "n_pos": n_pos, "n_read": n_read}

    # ---- The ridge strength, from training documents only. ------------------
    table = {}
    t_cv = time.time()
    for held_fold in range(N_FOLDS):
        fit_names = [g for g in TRAIN_GROUPS if g != "fold%d" % held_fold]
        eval_names = ["fold%d" % held_fold]
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
        best_l = max(range(n_read), key=lambda l: cv_mean[(alpha, l)])
        per_alpha_best[alpha] = (best_l, cv_mean[(alpha, best_l)])
    alpha_star = max(RIDGE_GRID, key=lambda a: per_alpha_best[a][1])
    train_best_layer, train_best_r2 = per_alpha_best[alpha_star]
    print("[sweep] %s: ridge strength %g, cross-validated best readout %d at %.4f "
          "(%.1f s)" % (revision, alpha_star, train_best_layer, train_best_r2,
                        time.time() - t_cv))

    # ---- The count window and the profile parts, from counts alone. ---------
    window_lo = window_from_counts(train_profile_counts, PROFILE_WIDTH,
                                  MIN_TRAIN_POSITIONS_PER_LABEL)
    labels_held = np.load(labels_file(work_dir, revision),
                          allow_pickle=False)["char_label"]
    bins = label_bins()
    bin_positions = [int(((labels_held >= lo) & (labels_held <= hi)).sum())
                     for (lo, hi) in bins]
    problems = []
    if window_lo is None:
        problems.append("no count window ending at %d whose values all reach %d training "
                        "positions" % (PROFILE_WIDTH, MIN_TRAIN_POSITIONS_PER_LABEL))
    thin = [i for i, n in enumerate(bin_positions) if n < 2]
    if thin:
        problems.append("profile parts %r hold fewer than two held-out positions" % thin)
    n_labels = None if window_lo is None else PROFILE_WIDTH - window_lo + 1
    calib_decision = decide_calibration(problems)
    calib_values = {"revision": revision, "training_step": step,
                    "ridge_strength": alpha_star,
                    "ridge_grid": RIDGE_GRID,
                    "ridge_strength_at_grid_edge":
                        float(alpha_star) in (float(RIDGE_EDGES[0]), float(RIDGE_EDGES[1])),
                    "cross_validated_quality_by_strength_and_readout":
                        {"%g|%d" % (a, l): round(cv_mean[(a, l)], 6)
                         for a in RIDGE_GRID for l in range(n_read)},
                    "cross_validated_best_per_strength":
                        {"%g" % a: {"readout": per_alpha_best[a][0],
                                    "quality": round(per_alpha_best[a][1], 6)}
                         for a in RIDGE_GRID},
                    "cross_validated_best_readout": train_best_layer,
                    "cross_validated_best_quality": round(train_best_r2, 6),
                    "fold_count": N_FOLDS, "base_seed": SEED,
                    "training_widths": USE_WIDTHS,
                    "count_window_low": window_lo,
                    "count_window_high": PROFILE_WIDTH,
                    "count_window_n_labels": n_labels,
                    "min_training_positions_per_count": MIN_TRAIN_POSITIONS_PER_LABEL,
                    "profile_parts": ["%d to %d" % (lo, hi) for lo, hi in bins],
                    "profile_part_held_out_positions": bin_positions,
                    "prior_run_ridge_strength_same_text": PRIOR_RIDGE_SAME_TEXT,
                    "seconds_strength_search": round(time.time() - t_cv, 1),
                    "problems": problems}
    calib_values.update(gates)
    write_record(
        prov, out_dir, base_cell("extract", "constants", revision),
        inputs=record_inputs(prov, stim, design, gate, control),
        metric="The ridge strength and the count window for one checkpoint, fixed on training documents only.",
        values=calib_values,
        threshold={"ridge_grid": RIDGE_GRID, "folds": N_FOLDS,
                   "min_training_positions_per_count": MIN_TRAIN_POSITIONS_PER_LABEL},
        decision_rule=RULE_CALIBRATION,
        decision=calib_decision,
        notes=("Every number here comes from training documents only, or from counts of "
               "positions, and is written before any held-out number for this checkpoint "
               "exists. All randomness streams from base seed 0, which the committed gate "
               "record fixes in its third answer of record. The readout chosen by this "
               "cross-validation is recorded and is not this checkpoint's chosen readout: "
               "the design chooses that on held-out documents, immediately below. The "
               "window's low end can sit above 1, because the first column of a line "
               "needs a one-character opening token and is thin; how far above is "
               "recorded. The four profile parts are fixed by the design at 1 to 35, 36 "
               "to 70, 71 to 105 and 106 to 140. " + COORDINATE_NOTE + " " + RERUN_NOTE),
    )
    if calib_decision != "CALIBRATION_OK":
        raise Halt("calibration is unusable at %s: %s" % (revision, "; ".join(problems)))

    # ---- The readout sweep on held-out documents. ---------------------------
    per_layer = []
    exact = []
    weights_by_layer = {}
    t_score = time.time()
    for layer in range(n_read):
        g, v, c, ls, lq, n = sum_groups(tot, TRAIN_GROUPS, layer)
        w, off = ridge_fit_from_stats(g, v, c, ls, n, alpha_star)
        weights_by_layer[layer] = (w, off)
        gt, vt, ct, lst, lqt, nt = sum_groups(tot, TEST_GROUPS, layer)
        pooled, _, _ = r2_from_stats(w, off, gt, vt, ct, lst, lqt, nt)
        row = {"readout": layer,
               "held_out_quality_pooled_widths_20_to_140": round(float(pooled), 6)}
        by_width = {}
        for width in USE_WIDTHS:
            gw, vw, cw, lsw, lqw, nw = sum_groups(tot, ["test_w%d" % width], layer)
            r2w, _, _ = r2_from_stats(w, off, gw, vw, cw, lsw, lqw, nw)
            by_width[width] = float(r2w)
            row["held_out_quality_w%d" % width] = round(float(r2w), 6)
        per_layer.append(row)
        exact.append({"readout": layer, "pooled": float(pooled), "by_width": by_width})
    best_exact = max(exact, key=lambda r: r["pooled"])
    best_layer = int(best_exact["readout"])
    best_pooled = best_exact["pooled"]
    layer_qualities = [r["pooled"] for r in exact]
    median_quality = median_of(layer_qualities)
    peak_margin = (None if median_quality is None else best_pooled - median_quality)
    layer_spread = max(layer_qualities) - min(layer_qualities)
    localization = decide_localization(best_layer, best_pooled, median_quality)
    narrow_width = USE_WIDTHS[0]
    narrow_quality = best_exact["by_width"][narrow_width]
    print("[sweep] %s: best readout %d, pooled quality over widths 20 to 140 %.6f, "
          "median readout %.6f, peak above median %.6f, %s (%.1f s)"
          % (revision, best_layer, best_pooled, median_quality, peak_margin,
             localization, time.time() - t_score))

    # Free the totals before the vector work; they are the largest thing in memory.
    del gram, vec_sum, cross, tot
    w_best, off_best = weights_by_layer[best_layer]

    # ---- Keep the best readout's vectors, delete the other twenty-four. -----
    kept = act_file(work_dir, revision, best_layer)
    dropped = 0
    for layer in range(n_read):
        if layer == best_layer:
            continue
        if delete_file(act_file(work_dir, revision, layer)):
            dropped += 1
    print("[sweep] %s: kept the vectors of readout %d, deleted %d other readouts"
          % (revision, best_layer, dropped))

    x = np.fromfile(kept, dtype=np.float32)
    if x.size != n_held_rows * EXPECT_HIDDEN:
        raise Halt("saved vectors for %s readout %d hold %d numbers, expected %d"
                   % (revision, best_layer, x.size, n_held_rows * EXPECT_HIDDEN))
    x = x.reshape(n_held_rows, EXPECT_HIDDEN)
    labels_f = labels_held.astype(np.float64)
    pred = x @ w_best + off_best

    bin_rows = bin_correlation_rows(pred, labels_f, bins)
    corrs = [r["pearson_correlation"] for r in bin_rows]
    ratio = boundary_ratio(corrs)

    slots = slots_array(labels_held, window_lo, n_labels)
    means, counts = means_by_slot(x, slots, n_labels)
    share = variance_shares(means - means.mean(axis=0, keepdims=True))
    dims = dimensions_at_targets(share)
    means_name = "mean_vectors_%s.npz" % revision
    np.savez(Path(out_dir) / means_name,
             labels=np.arange(window_lo, PROFILE_WIDTH + 1, dtype=np.int32),
             counts=np.array(counts, dtype=np.int32), means=means,
             readout=np.array([best_layer]), step=np.array([step]))
    rows_pos_path = Path(out_dir) / ("rows_positions_w%d_%s.jsonl" % (PROFILE_WIDTH,
                                                                     revision))
    with open(rows_pos_path, "w", encoding="utf-8") as fh:
        for i in range(n_held_rows):
            fh.write(json.dumps({
                "revision": revision, "training_step": step, "readout": best_layer,
                "doc": int(held["doc"][i]), "tok_index": int(held["tok_index"][i]),
                "tok_text": held["tok_text"][i],
                "char_label": int(held["char"][i]),
                "predicted_count": round(float(pred[i]), 6)}, ensure_ascii=False) + "\n")

    flags_coord = coordinate_flags(best_layer, alpha_star)
    coord_decision = decide_coordinates(best_layer, alpha_star)
    print("[sweep] %s: profile %r, boundary ratio %s, directions %r, coordinates %s"
          % (revision, corrs, ("absent" if ratio is None else round(ratio, 6)), dims,
             coord_decision))

    row = {"revision": revision, "step": step, "measurement_order_index": order_index,
           "pooled_quality": round(float(best_pooled), 6),
           "best_layer": best_layer,
           "ridge_strength": float(alpha_star),
           "boundary_ratio": (None if ratio is None else round(float(ratio), 6)),
           "profile_correlations": corrs,
           "profile_part_positions": [r["n_positions"] for r in bin_rows],
           "dimensions": dims,
           "coordinate_flags": flags_coord,
           "coordinate_decision": coord_decision,
           "degenerate_coordinates": bool(flags_coord),
           "layer_median_quality": round(float(median_quality), 6),
           "peak_over_layer_median": round(float(peak_margin), 6),
           "layer_quality_spread": round(float(layer_spread), 6),
           "best_layer_interior": best_layer not in LAYER_EDGES,
           "localization_decision": localization,
           "quality_at_narrowest_width": round(float(narrow_quality), 6),
           "narrow_width_quality_negative": bool(narrow_quality < 0.0),
           "quality_by_width": {str(k): round(float(v), 6)
                                for k, v in best_exact["by_width"].items()},
           "cross_validated_best_layer": train_best_layer,
           "count_window_low": window_lo,
           "held_out_profile_rows": n_held_rows,
           "seconds": round(time.time() - started, 1)}

    missing_keys = [k for k in ROW_KEYS_NEEDED if k not in row]
    if missing_keys:
        raise Halt("this checkpoint's row lacks %r, which the trajectory stage reads"
                   % missing_keys)

    reading_values = dict(row)
    reading_values.update({
        "variance_share_first_ten":
            (None if share is None else [round(s, 6) for s in share[:10]]),
        "variance_targets_recorded_only": VARIANCE_TARGETS,
        "profile_parts": ["%d to %d" % (lo, hi) for lo, hi in bins],
        "boundary_ratio_definition":
            "mean of the parts 1 to 35 and 36 to 70 minus mean of 71 to 105 and 106 to 140",
        "localization_margin": LOCALIZATION_MARGIN,
        "narrowest_width_read": narrow_width,
        "prior_run_profile_same_text": PRIOR_PROFILE_SAME_TEXT,
        "prior_run_profile_fresh_text": PRIOR_PROFILE_FRESH_TEXT,
        "corpus_dominant_wrap_length_chars": CORPUS_WRAP_LENGTH_CHARS,
        "prior_run_readout_same_text": PRIOR_READOUT_SAME_TEXT,
        "prior_run_pooled_quality_same_text": PRIOR_POOLED_SAME_TEXT,
        "best_layer_equals_prior_run_readout": best_layer == PRIOR_READOUT_SAME_TEXT,
        "sibling_untrained_pooled_quality": SIBLING_BASELINE_POOLED,
        "sibling_untrained_narrow_width_quality": SIBLING_BASELINE_NARROW_WIDTH_QUALITY,
        "mean_vectors_file": means_name,
        "per_position_rows_file": rows_pos_path.name,
        "verbatim_variant_rows_file": rows_var_path.name,
    })
    reading_values.update(gates)
    write_record(
        prov, out_dir, base_cell("extract", "checkpoint_reading", revision),
        inputs=record_inputs(prov, stim, design, gate, control,
                             extra=[(labels_file(work_dir, revision), n_held_rows)]),
        metric="One checkpoint's held-out pooled quality at every readout, its best readout, its four-part precision profile, its boundary ratio, its direction counts, and its two shape signatures.",
        values=reading_values,
        threshold={"recorded_not_gated": True},
        decision_rule=RULE_CHECKPOINT,
        decision=coord_decision,
        rows=per_layer,
        notes=("The best readout is this checkpoint's own, chosen by this checkpoint's "
               "own pooled quality, which the previous rung's committed gate record "
               "settles in its third answer of record: fixing one readout everywhere "
               "would presume the answer, and the readout's trajectory is itself a "
               "recorded observable. The profile statistic is the Pearson correlation "
               "between predicted and true count inside each part, which the same record "
               "settles in its fourth answer. The two shape signatures the untrained "
               "sibling showed are recorded here at every checkpoint and gate nothing: "
               "how flat quality is across readouts, and the per-width qualities, where "
               "the narrowest width read -2.479186 on that untrained sibling. "
               + BOUNDARY_NOTE + " " + COORDINATE_NOTE + " The direction counts at 90, 95 "
               "and 99 percent are recorded and gate nothing. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- Delete this checkpoint's working vectors before the next download. -
    delete_file(kept)
    delete_file(labels_file(work_dir, revision))
    delete_tree(cache)
    leftovers = sorted(p.name for p in Path(work_dir).glob("act_%s_*" % revision))
    if leftovers:
        raise Halt("working vectors for %s survived deletion: %r" % (revision, leftovers))
    print("[sweep] %s done in %.1f s; working vectors and weights deleted"
          % (revision, time.time() - started))
    return row


SWEEP_CELL_REVISION = "sweep_twelve_checkpoints"


def write_void_summary(prov, out_dir, inputs, rows, anchor_decision, baseline_decision,
                       headline):
    """The whole-sweep record when an integrity gate stopped the run.

    Nothing beyond the checkpoints already measured is computed, because the design says
    no other reading is trusted. The cells for the trajectory endpoints are simply never
    written, so their absence is visible as an absence rather than as a value.
    """
    values = {"headline": headline,
              "sweep_stopped_at_gate": True,
              "n_checkpoints_measured": len(rows),
              "n_checkpoints_planned": len(MEASURE_ORDER),
              "measurement_order": MEASURE_ORDER,
              "checkpoints_measured": [r["revision"] for r in rows],
              "pooled_quality_by_revision":
                  {r["revision"]: r["pooled_quality"] for r in rows},
              "boundary_ratio_by_revision":
                  {r["revision"]: r["boundary_ratio"] for r in rows},
              "best_layer_by_revision": {r["revision"]: r["best_layer"] for r in rows},
              "ridge_strength_by_revision":
                  {r["revision"]: r["ridge_strength"] for r in rows},
              "coordinate_flags_by_revision":
                  {r["revision"]: r["coordinate_flags"] for r in rows},
              "localization_by_revision":
                  {r["revision"]: r["localization_decision"] for r in rows},
              "trajectory_endpoints_computed": False,
              "generation_performed": False,
              "single_run_not_replicated": True}
    values.update(gate_block(anchor_decision, baseline_decision))
    write_record(
        prov, out_dir,
        base_cell("extract", "run_summary", SWEEP_CELL_REVISION),
        inputs=inputs,
        metric="The whole-sweep reading, stopped by an integrity gate before the trajectory was measured.",
        values=values,
        threshold={"anchor_within": ANCHOR_TOLERANCE,
                   "anchor_ratio_at_least": ANCHOR_RATIO_FLOOR,
                   "baseline_below": BASELINE_VOID_FLOOR},
        decision_rule=RULE_HEADLINE,
        decision=headline,
        notes=("An integrity gate failed, so the sweep stopped where the committed order "
               "says it must. The failure is an instrument fault and is not evidence "
               "about the model. No trajectory endpoint was computed, and no record for "
               "one exists. " + VOID_NOTE + " " + RERUN_NOTE),
    )


def stage_extract(args):
    """The twelve-checkpoint sweep, in the committed measurement order."""
    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    gate = gate_record_path()
    control = control_record_path()

    actual_hash = sha256_file(stim)
    print("[sweep] text file %s" % stim)
    print("[sweep] hash %s (expected %s) match=%s"
          % (actual_hash, STIMULUS_SHA256, exact_hash_match(actual_hash, STIMULUS_SHA256)))
    if not exact_hash_match(actual_hash, STIMULUS_SHA256):
        raise Halt("the committed text file hashes %s, expected %s; halt, never "
                   "regenerate it" % (actual_hash, STIMULUS_SHA256))
    meta, docs = load_stimulus(stim)
    train_ids = [int(d["id"]) for d in docs if d["split"] == "train"]
    folds = fold_of_documents(train_ids)
    inputs = record_inputs(prov, stim, design, gate, control)

    print("[sweep] model %s, %d checkpoints, measured in this order: %s"
          % (MODEL_ID, len(MEASURE_ORDER), ", ".join(MEASURE_ORDER)))
    print("[sweep] the final checkpoint is first because its quality must land within "
          "%.2f of %.6f AND its boundary ratio must reach %.1f, or the sweep is void"
          % (ANCHOR_TOLERANCE, ANCHOR_POOLED, ANCHOR_RATIO_FLOOR))
    print("[sweep] the initialization checkpoint is second because its quality is the "
          "baseline floor every formation threshold is measured from; it voids the sweep "
          "only at or above %.7f" % BASELINE_VOID_FLOOR)
    print("[sweep] widths read %r; text source %s" % (USE_WIDTHS, meta.get("source")))
    if control is None:
        print("[sweep] WARNING: the committed label-reachability control was not shipped, "
              "so records carry no hash for it; its five trajectories are constants in "
              "this file and the self-test replays them either way")

    rows = []
    anchor_decision = "NOT_YET_MEASURED"
    baseline_decision = "NOT_YET_MEASURED"
    anchor_quality = None
    anchor_ratio = None
    baseline_quality = None
    started = time.time()

    for order_index, revision in enumerate(MEASURE_ORDER):
        row = measure_revision(revision, order_index, prov, out_dir, work_dir,
                               stim, design, gate, control, docs, folds,
                               anchor_decision, baseline_decision)
        rows.append(row)

        if revision == FINAL_REVISION:
            anchor_quality = row["pooled_quality"]
            anchor_ratio = row["boundary_ratio"]
            anchor_decision = decide_anchor(anchor_quality, anchor_ratio)
            values = {"revision": revision, "training_step": row["step"],
                      "held_out_quality_pooled_widths_20_to_140": anchor_quality,
                      "boundary_ratio": anchor_ratio,
                      "best_layer": row["best_layer"],
                      "ridge_strength": row["ridge_strength"],
                      "committed_anchor": ANCHOR_POOLED,
                      "tolerance": ANCHOR_TOLERANCE,
                      "band_low": round(ANCHOR_POOLED - ANCHOR_TOLERANCE, 6),
                      "band_high": round(ANCHOR_POOLED + ANCHOR_TOLERANCE, 6),
                      "difference_from_anchor":
                          round(float(anchor_quality) - ANCHOR_POOLED, 6),
                      "boundary_ratio_floor": ANCHOR_RATIO_FLOOR,
                      "committed_prior_profile_same_text": PRIOR_PROFILE_SAME_TEXT,
                      "committed_prior_ratio_as_stated_in_the_design":
                          PRIOR_RATIO_SAME_TEXT_AS_COMMITTED,
                      "committed_prior_ratio_recomputed_from_the_parts":
                          (None if boundary_ratio(PRIOR_PROFILE_SAME_TEXT) is None
                           else round(boundary_ratio(PRIOR_PROFILE_SAME_TEXT), 6)),
                      "fresh_text_quality_same_model": PRIOR_POOLED_FRESH_TEXT,
                      "prior_run_readout_same_text": PRIOR_READOUT_SAME_TEXT,
                      "prior_run_ridge_strength_same_text": PRIOR_RIDGE_SAME_TEXT,
                      "best_layer_equals_prior_run_readout":
                          row["best_layer"] == PRIOR_READOUT_SAME_TEXT,
                      "profile_correlations": row["profile_correlations"],
                      "coordinate_flags": row["coordinate_flags"]}
            values.update(gate_block(anchor_decision, baseline_decision, anchor_quality,
                                     anchor_ratio))
            write_record(
                prov, out_dir,
                base_cell("extract", "anchor_gate", revision, endpoint=1),
                inputs=inputs,
                metric="Held-out pooled quality and boundary ratio at the final checkpoint, on the same text as the committed 0.876302 and 0.514227, at that checkpoint's own best readout.",
                values=values,
                threshold={"anchor": ANCHOR_POOLED, "within": ANCHOR_TOLERANCE,
                           "ratio_at_least": ANCHOR_RATIO_FLOOR},
                decision_rule=RULE_ANCHOR,
                decision=anchor_decision,
                notes=(ANCHOR_NOTE + " This is the same text, the same instrument and the "
                       "same model as both committed values, so the quality band is "
                       "tighter than any fresh-text band would be. " + BOUNDARY_NOTE + " "
                       + RERUN_NOTE),
            )
            if anchor_decision != "ANCHOR_IN_BAND":
                write_sweep_table(out_dir, rows, anchor_decision, baseline_decision, False)
                write_void_summary(prov, out_dir, inputs, rows, anchor_decision,
                                   baseline_decision, "VOID_ANCHOR_FAILED")
                raise Halt("the final checkpoint measured quality %.6f and ratio %s; the "
                           "gate needs quality within %.2f of %.6f and ratio at least "
                           "%.1f, so the sweep is void as an instrument fault and stops "
                           "here" % (anchor_quality,
                                     "absent" if anchor_ratio is None else "%.6f" % anchor_ratio,
                                     ANCHOR_TOLERANCE, ANCHOR_POOLED, ANCHOR_RATIO_FLOOR))

        if revision == BASELINE_REVISION:
            baseline_quality = row["pooled_quality"]
            baseline_decision = decide_baseline(baseline_quality)
            formation_threshold, formed_threshold = formation_thresholds(baseline_quality)
            values = {"revision": revision, "training_step": row["step"],
                      "held_out_quality_pooled_widths_20_to_140": baseline_quality,
                      "baseline_floor": baseline_quality,
                      "best_layer": row["best_layer"],
                      "ridge_strength": row["ridge_strength"],
                      "void_share_of_anchor": BASELINE_VOID_SHARE,
                      "void_floor": BASELINE_VOID_FLOOR,
                      "margin_below_void_floor":
                          round(BASELINE_VOID_FLOOR - float(baseline_quality), 6),
                      "expected_range": [BASELINE_EXPECTED_LOW, BASELINE_EXPECTED_HIGH],
                      "inside_expected_range_recorded_only":
                          baseline_inside_expected(baseline_quality),
                      "formation_threshold_from_this_baseline":
                          (None if formation_threshold is None
                           else round(formation_threshold, 6)),
                      "formed_threshold_from_this_baseline":
                          (None if formed_threshold is None
                           else round(formed_threshold, 6)),
                      "profile_correlations": row["profile_correlations"],
                      "boundary_ratio": row["boundary_ratio"],
                      "layer_median_quality": row["layer_median_quality"],
                      "peak_over_layer_median": row["peak_over_layer_median"],
                      "layer_quality_spread": row["layer_quality_spread"],
                      "localization_decision": row["localization_decision"],
                      "quality_by_width": row["quality_by_width"],
                      "quality_at_narrowest_width": row["quality_at_narrowest_width"],
                      "narrow_width_quality_negative":
                          row["narrow_width_quality_negative"],
                      "sibling_untrained_pooled_quality": SIBLING_BASELINE_POOLED,
                      "sibling_untrained_boundary_ratio": SIBLING_BASELINE_RATIO,
                      "sibling_untrained_best_readout": SIBLING_BASELINE_READOUT,
                      "sibling_untrained_ridge_strength": SIBLING_BASELINE_RIDGE,
                      "sibling_untrained_narrow_width_quality":
                          SIBLING_BASELINE_NARROW_WIDTH_QUALITY,
                      "coordinate_flags": row["coordinate_flags"],
                      "coordinate_flags_expected_here": True}
            values.update(gate_block(anchor_decision, baseline_decision, anchor_quality,
                                     anchor_ratio, baseline_quality))
            write_record(
                prov, out_dir,
                base_cell("extract", "baseline", revision, endpoint=2),
                inputs=inputs,
                metric="Held-out pooled quality at the initialization checkpoint, before any training: the baseline floor every formation threshold is measured from.",
                values=values,
                threshold={"void_at_or_above": BASELINE_VOID_FLOOR,
                           "expected_range_recorded_only":
                               [BASELINE_EXPECTED_LOW, BASELINE_EXPECTED_HIGH]},
                decision_rule=RULE_BASELINE,
                decision=baseline_decision,
                notes=(BASELINE_NOTE + " " + COORDINATE_NOTE + " The coordinate flags on "
                       "this record are therefore expected, and both committed gate "
                       "records say so in advance. The two shape signatures are recorded "
                       "here and gate nothing. " + RERUN_NOTE),
            )
            if baseline_decision != "BASELINE_RECORDED":
                write_sweep_table(out_dir, rows, anchor_decision, baseline_decision, False)
                write_void_summary(prov, out_dir, inputs, rows, anchor_decision,
                                   baseline_decision, "VOID_CONTROL_FAILED")
                raise Halt("the initialization checkpoint measured %.6f, at or above the "
                           "%.7f void floor; the sweep is void and stops here"
                           % (baseline_quality, BASELINE_VOID_FLOOR))

        write_sweep_table(out_dir, rows, anchor_decision, baseline_decision,
                          len(rows) == len(MEASURE_ORDER))
        print("[sweep] %d of %d checkpoints measured, %.1f minutes so far"
              % (len(rows), len(MEASURE_ORDER), (time.time() - started) / 60.0))

    print("[sweep] all %d checkpoints measured in %.1f minutes"
          % (len(rows), (time.time() - started) / 60.0))
    return 0


# ---------------------------------------------------------------------------
# Part 11. The trajectory: formation, the boundary, and the stability test.
#
# The record names here follow this rung's own design object, which lists five
# endpoints: the gate cell, the baseline floor, formation and stability of quality, the
# boundary-ratio trajectory, and the boundary's own timing against the formation step.
# The first two are written inside the sweep, because a gate that fired afterwards would
# already have spent the box. The last three are written here.
# ---------------------------------------------------------------------------

def stage_endpoints(args):
    """The cross-checkpoint arithmetic and the two pre-registered readings."""
    prov = load_provenance()
    out_dir, work_dir = paths_for(args.results_dir)
    stim = stimulus_path(args.stimulus)
    design = design_record_path(args.design_record)
    gate = gate_record_path()
    control = control_record_path()
    table = read_sweep_table(out_dir)
    rows = table.get("rows", [])
    inputs = record_inputs(prov, stim, design, gate, control,
                           extra=[(sweep_table_path(out_dir), len(rows))])
    if not table.get("complete") or len(rows) != len(REVISIONS):
        raise Halt("the per-checkpoint table holds %d of %d checkpoints and is marked "
                   "%s; the trajectory may not be read from part of a sweep"
                   % (len(rows), len(REVISIONS),
                      "complete" if table.get("complete") else "incomplete"))
    present = sorted(r["revision"] for r in rows)
    if present != sorted(REVISIONS):
        raise Halt("the table names %r, which is not the twelve committed checkpoints"
                   % present)

    for r in rows:
        missing_keys = [k for k in ROW_KEYS_NEEDED if k not in r]
        if missing_keys:
            raise Halt("the row for %r lacks %r, so the trajectory cannot be read"
                       % (r.get("revision"), missing_keys))

    anchor_row = [r for r in rows if r["revision"] == FINAL_REVISION][0]
    baseline_row = [r for r in rows if r["revision"] == BASELINE_REVISION][0]
    anchor_decision = decide_anchor(anchor_row["pooled_quality"],
                                    anchor_row["boundary_ratio"])
    baseline_decision = decide_baseline(baseline_row["pooled_quality"])
    for name, here, there in (("gate cell", anchor_decision,
                               table.get("anchor_gate_decision")),
                              ("baseline floor", baseline_decision,
                               table.get("baseline_decision"))):
        if there is not None and there != here:
            raise Halt("the %s reads %r here and %r in the table the sweep wrote; "
                       "a run may not be read through two different calls"
                       % (name, here, there))
    if anchor_decision != "ANCHOR_IN_BAND" or baseline_decision != "BASELINE_RECORDED":
        raise Halt("an integrity gate failed, so the sweep is void and no trajectory "
                   "endpoint may be computed: gate cell %s, baseline %s"
                   % (anchor_decision, baseline_decision))

    read = sweep_readings(rows, anchor_decision, baseline_decision)
    gates = gate_block(anchor_decision, baseline_decision,
                       anchor_row["pooled_quality"], anchor_row["boundary_ratio"],
                       baseline_row["pooled_quality"])
    if read["formed_step"] is None:
        raise Halt("no checkpoint reaches the formed threshold %r, which the gate cell "
                   "makes impossible; the table is inconsistent"
                   % read["formed_threshold"])

    print("[trajectory] baseline floor %r; formation threshold %r; formed threshold %r"
          % (read["baseline_quality"], read["formation_threshold"],
             read["formed_threshold"]))
    print("[trajectory] quality by step %r" % (read["pooled_quality_by_step"],))
    print("[trajectory] boundary ratio by step %r" % (read["boundary_ratio_by_step"],))
    print("[trajectory] best readout by step %r" % (read["best_layer_by_step"],))
    print("[trajectory] formation step %r, formed step %r, boundary crossing %r, "
          "localization %r" % (read["formation_step"], read["formed_step"],
                               read["boundary_crossing_step"],
                               read["localization_step_recorded_only"]))

    # ---- Design endpoint 3, first part: did the counter ever form. ----------
    v3 = {"highest_quality_any_checkpoint": read["highest_quality_any_checkpoint"],
          "baseline_quality": read["baseline_quality"],
          "committed_anchor_quality": ANCHOR_POOLED,
          "formed_threshold": read["formed_threshold"],
          "formed_share_of_gap": FORMED_SHARE,
          "pooled_quality_by_step": read["pooled_quality_by_step"],
          "final_quality": read["final_quality"],
          "n_checkpoints_reaching_formed_threshold":
              sum(1 for q in read["pooled_quality_by_step"].values()
                  if q is not None and read["formed_threshold"] is not None
                  and not_below(q, read["formed_threshold"])),
          "committed_anchors_this_model": [PRIOR_POOLED_SAME_TEXT,
                                          PRIOR_POOLED_FRESH_TEXT]}
    v3.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "formation_existence", SWEEP_CELL_REVISION, endpoint=3),
        inputs=inputs,
        metric="Highest held-out pooled quality reached by any of the twelve checkpoints, against the baseline-relative formed threshold.",
        values=v3, threshold={"at_least": read["formed_threshold"]},
        decision_rule=RULE_FORMATION_EXISTS,
        decision=read["formation_exists_decision"],
        notes=("Two committed measurements of this model at its final checkpoint, "
               "0.876302 on this text and 0.879879 on a second text, already establish "
               "that the counter exists there. So a sweep that reached the formed "
               "threshold nowhere would impeach the sweep rather than the counter. "
               + FORCED_OUTCOME_NOTE + " " + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- Design endpoint 3, second part: the quality stability channel. -----
    after = read["checkpoints_at_or_after_formed"]
    v6 = {"quality_range_after_formed": read["quality_range_after_formed"],
          "limit": HOLD_QUALITY_RANGE,
          "checkpoints_at_or_after_formed": after,
          "formed_revision": read["formed_revision"],
          "formed_step": read["formed_step"],
          "formed_threshold": read["formed_threshold"],
          "pooled_quality_by_step": read["pooled_quality_by_step"],
          "series_one_way_in_steps": read["quality_series_one_way_in_steps"]}
    v6.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "hold_test_quality", SWEEP_CELL_REVISION, endpoint=3),
        inputs=inputs,
        metric="Range of the held-out pooled quality across every checkpoint at or after the formed checkpoint.",
        values=v6, threshold={"within": HOLD_QUALITY_RANGE},
        decision_rule=RULE_HOLD_QUALITY,
        decision=read["hold_test_quality_channel"],
        notes=(MARKER_NOTE + " " + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- Design endpoint 3, third part: early or late. ----------------------
    v7 = {"formation_step": read["formation_step"],
          "formation_revision": read["formation_revision"],
          "formation_threshold": read["formation_threshold"],
          "formation_share_of_gap": FORMATION_SHARE,
          "baseline_quality": read["baseline_quality"],
          "committed_anchor_quality": ANCHOR_POOLED,
          "formed_step": read["formed_step"],
          "formed_revision": read["formed_revision"],
          "formed_threshold": read["formed_threshold"],
          "formed_share_of_gap": FORMED_SHARE,
          "cutoff_step": EARLY_CUTOFF_STEP,
          "call_from_the_formed_marker_recorded_only":
              read["formation_timing_from_formed_marker_recorded_only"],
          "formation_step_from_measured_final_recorded_only":
              read["formation_step_from_measured_final_recorded_only"],
          "formed_step_from_measured_final_recorded_only":
              read["formed_step_from_measured_final_recorded_only"],
          "pooled_quality_by_step": read["pooled_quality_by_step"]}
    v7.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "formation_timing", SWEEP_CELL_REVISION, endpoint=3),
        inputs=inputs,
        metric="The first checkpoint whose pooled quality reaches the baseline-relative half-distance threshold, against step16000.",
        values=v7, threshold={"at_or_before_step": EARLY_CUTOFF_STEP},
        decision_rule=RULE_TIMING,
        decision=read["formation_timing_decision"],
        notes=(MARKER_NOTE + " A late formation routes the reading to the late variant "
               "even when the stability test passes, which weakens the early-formation "
               "half of the fossil account. Recorded beside the call and gating nothing: "
               "the same two crossings computed against this run's own measured final "
               "quality instead of the committed anchor quality. " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- Design endpoint 4, first part: is the boundary there at the end. ---
    v4 = {"boundary_ratio_at_final": read["boundary_ratio_at_final"],
          "floor": BOUNDARY_PRESENT_FLOOR,
          "profile_at_final": anchor_row["profile_correlations"],
          "boundary_ratio_by_step": read["boundary_ratio_by_step"],
          "boundary_ratio_at_baseline": read["boundary_ratio_at_baseline"],
          "committed_prior_profile_same_text": PRIOR_PROFILE_SAME_TEXT,
          "committed_prior_profile_fresh_text": PRIOR_PROFILE_FRESH_TEXT,
          "sibling_final_boundary_ratio": SIBLING_FINAL_RATIO,
          "sibling_untrained_boundary_ratio": SIBLING_BASELINE_RATIO,
          "corpus_dominant_wrap_length_chars": CORPUS_WRAP_LENGTH_CHARS}
    v4.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "boundary_presence", SWEEP_CELL_REVISION, endpoint=4),
        inputs=inputs,
        metric="Boundary ratio at the final checkpoint: mean of the first two profile parts minus mean of the last two.",
        values=v4, threshold={"at_least": BOUNDARY_PRESENT_FLOOR},
        decision_rule=RULE_BOUNDARY_PRESENT,
        decision=read["boundary_presence_decision"],
        notes=(BOUNDARY_NOTE + " Recorded beside it, and gating nothing: the corpus's "
               "dominant hard-wrap length is about 70 to 72 characters, which is where "
               "the second profile part ends. " + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- Design endpoint 4, second part: the boundary stability channel. ----
    v5 = {"boundary_range_after_formed": read["boundary_range_after_formed"],
          "limit": HOLD_BOUNDARY_RANGE,
          "checkpoints_at_or_after_formed": after,
          "formed_revision": read["formed_revision"],
          "formed_step": read["formed_step"],
          "boundary_ratio_by_step": read["boundary_ratio_by_step"],
          "series_one_way_in_steps": read["boundary_series_one_way_in_steps"],
          "boundary_presence_decision": read["boundary_presence_decision"],
          "boundary_crossing_step": read["boundary_crossing_step"],
          "boundary_range_from_its_own_crossing_recorded_only":
              read["boundary_range_from_its_own_crossing_recorded_only"]}
    v5.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "hold_test_boundary", SWEEP_CELL_REVISION, endpoint=4),
        inputs=inputs,
        metric="Range of the boundary ratio across every checkpoint at or after the shared formed checkpoint.",
        values=v5, threshold={"within": HOLD_BOUNDARY_RANGE},
        decision_rule=RULE_HOLD_BOUNDARY,
        decision=read["hold_test_boundary_channel"],
        notes=(BOUNDARY_NOTE + " " + MARKER_NOTE + " The design object's fourth endpoint "
               "would start this window at the boundary's own first crossing instead; the "
               "committed gate record's fourth answer of record puts both windows at the "
               "shared formed checkpoint, so the range from the boundary's own crossing is "
               "recorded here and gates nothing. " + VOID_NOTE + " " + RERUN_NOTE),
    )

    # ---- Design endpoint 5: the boundary's own timing. ----------------------
    v8 = {"boundary_crossing_step": read["boundary_crossing_step"],
          "boundary_crossing_revision": read["boundary_crossing_revision"],
          "boundary_crossing_floor": BOUNDARY_PRESENT_FLOOR,
          "formation_step": read["formation_step"],
          "formation_revision": read["formation_revision"],
          "formation_threshold": read["formation_threshold"],
          "boundary_ratio_by_step": read["boundary_ratio_by_step"],
          "boundary_ratio_at_formation_step": read["boundary_ratio_at_formation_step"],
          "boundary_ratio_at_formed_step": read["boundary_ratio_at_formed_step"],
          "pooled_quality_by_step": read["pooled_quality_by_step"]}
    v8.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "boundary_timing", SWEEP_CELL_REVISION, endpoint=5),
        inputs=inputs,
        metric="The first checkpoint whose boundary ratio reaches 0.3, compared against the formation step.",
        values=v8, threshold={"at_or_before_step": read["formation_step"]},
        decision_rule=RULE_BOUNDARY_TIMING,
        decision=read["boundary_timing_decision"],
        notes=(BOUNDARY_TIMING_NOTE + " " + BOUNDARY_NOTE + " " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- Recorded only: the trajectories nothing gates. ---------------------
    v9 = {"degenerate_coordinates_by_step": read["degenerate_coordinates_by_step"],
          "best_layer_by_step": read["best_layer_by_step"],
          "ridge_strength_by_step": read["ridge_strength_by_step"],
          "dimensions_by_step": read["dimensions_by_step"],
          "coordinate_flags_by_step": read["coordinate_flags_by_step"],
          "layer_median_quality_by_step": read["layer_median_quality_by_step"],
          "peak_over_layer_median_by_step": read["peak_over_layer_median_by_step"],
          "layer_quality_spread_by_step": read["layer_quality_spread_by_step"],
          "localization_by_step": read["localization_by_step"],
          "localization_step_recorded_only": read["localization_step_recorded_only"],
          "localization_revision_recorded_only":
              read["localization_revision_recorded_only"],
          "localization_margin": LOCALIZATION_MARGIN,
          "quality_by_width_by_step": read["quality_by_width_by_step"],
          "variance_targets_recorded": VARIANCE_TARGETS,
          "boundary_ratio_at_formation_step": read["boundary_ratio_at_formation_step"],
          "boundary_ratio_at_formed_step": read["boundary_ratio_at_formed_step"],
          "boundary_ratio_at_final": read["boundary_ratio_at_final"],
          "baseline_inside_expected_range": read["baseline_inside_expected_range"],
          "baseline_expected_range": read["baseline_expected_range"],
          "quality_by_width_at_final": anchor_row.get("quality_by_width"),
          "prior_run_readout_same_text": PRIOR_READOUT_SAME_TEXT}
    v9.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "recorded_trajectories", SWEEP_CELL_REVISION),
        inputs=inputs,
        metric="The best readout and its localization moment, the ridge strength, the direction counts at three cutoffs, the per-width qualities, and where the boundary already stood, checkpoint by checkpoint.",
        values=v9, threshold={},
        decision_rule=RULE_RECORDED_ONLY,
        decision="RECORDED_NO_DECISION",
        notes=("The counting convention for directions is the smallest number reaching a "
               "given share of the spread, which is every prior run of this chain's "
               "convention. All three cutoffs are recorded and none is gated: the count "
               "is a property of the cutoff. The localization moment is the first "
               "checkpoint whose best readout sits off both ends of the readout range and "
               "beats the median readout by at least 0.1; on the untrained sibling "
               "neither held, which is what a flat readout profile looks like. "
               + COORDINATE_NOTE + " " + RECORDED_ONLY_NOTE + " " + VOID_NOTE + " "
               + RERUN_NOTE),
    )

    # ---- One call for the whole sweep. -------------------------------------
    summary = dict(read)
    summary.update({
        "endpoint_1_anchor_gate": anchor_decision,
        "endpoint_2_baseline": baseline_decision,
        "endpoint_3_formation_existence": read["formation_exists_decision"],
        "endpoint_3_hold_test_quality": read["hold_test_quality_channel"],
        "endpoint_3_formation_timing": read["formation_timing_decision"],
        "endpoint_4_boundary_presence": read["boundary_presence_decision"],
        "endpoint_4_hold_test_boundary": read["hold_test_boundary_channel"],
        "endpoint_5_boundary_timing": read["boundary_timing_decision"],
        "hold_test": read["hold_test_decision"],
        "profile_at_final": anchor_row["profile_correlations"],
        "profile_at_initialization": baseline_row["profile_correlations"],
        "text_file": STIMULUS_NAME,
        "widths_read": USE_WIDTHS,
        "profile_width": PROFILE_WIDTH,
        "generation_performed": False,
        "single_run_not_replicated": True,
    })
    summary.update(gates)
    write_record(
        prov, out_dir,
        base_cell("endpoints", "run_summary", SWEEP_CELL_REVISION),
        inputs=inputs,
        metric="One call per endpoint, and the two pre-registered readings, from the records written beside this one.",
        values=summary, threshold={"formation_threshold": read["formation_threshold"],
                                   "formed_threshold": read["formed_threshold"],
                                   "boundary_floor": BOUNDARY_PRESENT_FLOOR,
                                   "hold_boundary_within": HOLD_BOUNDARY_RANGE,
                                   "hold_quality_within": HOLD_QUALITY_RANGE,
                                   "early_at_or_before_step": EARLY_CUTOFF_STEP},
        decision_rule=RULE_HEADLINE,
        decision=read["headline"],
        notes=("The pre-registered readings are FORMED_EARLY_AND_HELD, "
               "FORMED_LATE_AND_HELD, FORMED_AND_DRIFTED, NOT_FORMED_BY_FINAL, "
               "UNSTABLE_NOT_CLASSIFIABLE, VOID_ANCHOR_FAILED and VOID_CONTROL_FAILED. "
               "The boundary's own timing is a second reading, recorded beside this one: "
               "BOUNDARY_WITH_COUNTER, BOUNDARY_AFTER_COUNTER or "
               "BOUNDARY_NOT_FORMED_BY_FINAL. Both rival accounts have a live refuting "
               "outcome here: a held reading refutes drift, and a drifted reading refutes "
               "the fossil account. " + CONTROL_REPLAY_NOTE + " " + FORCED_OUTCOME_NOTE
               + " " + MARKER_NOTE + " " + BOUNDARY_NOTE + " " + NO_CORPUS_NOTE
               + " Twelve named checkpoints cannot resolve anything finer than the gaps "
               "between them, and one training run cannot speak for other training "
               "seeds. " + RERUN_NOTE),
    )
    print("[trajectory] summary %s" % json.dumps({
        "headline": read["headline"],
        "boundary_trajectory_reading": read["boundary_trajectory_reading"],
        "baseline_quality": read["baseline_quality"],
        "formation_step": read["formation_step"],
        "formed_step": read["formed_step"],
        "boundary_crossing_step": read["boundary_crossing_step"],
        "hold_test": read["hold_test_decision"],
        "boundary_presence": read["boundary_presence_decision"],
        "quality_range_after_formed": read["quality_range_after_formed"],
        "boundary_range_after_formed": read["boundary_range_after_formed"]}))
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
    print("  %-74s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def _planted_row(revision, quality, ratio, layer=5, ridge=1000.0, median=None):
    """One planted per-checkpoint row, with a known correct call."""
    med = (quality - 0.2) if median is None else median
    return {"revision": revision, "step": step_of(revision),
            "pooled_quality": quality, "boundary_ratio": ratio,
            "best_layer": layer, "ridge_strength": ridge,
            "dimensions": {"0.9": 1, "0.95": 2, "0.99": 5},
            "profile_correlations": (None if ratio is None else
                                     [0.8, 0.8, round(0.8 - ratio, 6),
                                      round(0.8 - ratio, 6)]),
            "coordinate_flags": coordinate_flags(layer, ridge),
            "degenerate_coordinates": bool(coordinate_flags(layer, ridge)),
            "layer_median_quality": med,
            "peak_over_layer_median": round(quality - med, 6),
            "layer_quality_spread": 0.5,
            "best_layer_interior": layer not in LAYER_EDGES,
            "localization_decision": decide_localization(layer, quality, med),
            "quality_by_width": {"20": -1.0, "140": quality}}


def _planted_sweep(qualities, ratios):
    """Twelve planted rows, one per committed checkpoint, in training order."""
    if len(qualities) != len(REVISIONS) or len(ratios) != len(REVISIONS):
        raise Halt("a planted sweep needs one quality and one ratio per checkpoint")
    return [_planted_row(rev, q, r)
            for rev, q, r in zip(REVISIONS, qualities, ratios)]


def _control_rows(name, baseline=CONTROL_BASELINE):
    """One committed synthetic trajectory as twelve planted rows."""
    q, r = control_trajectory(name, baseline)
    return _planted_sweep(q, r)


def selftest() -> int:
    fails = []

    print("== position labeller (the reasoning-tag parser of this experiment) ==")
    ch, tk, ln = labels_for_spans("ab\ncd", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
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
    ch3, _, _ = labels_for_spans("abc\nd", [(0, 3), (3, 4), (4, 5)])
    _check(fails, "line-break token takes the next column", ch3, [3, 4, 1])
    _check(fails, "the width-plus-one label falls outside a window ending at the width",
           label_slot(141, 1, 140), None)
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
    _check(fails, "hash one character off", exact_hash_match("a" * 63 + "b", "a" * 64),
           False)
    _check(fails, "hash absent", exact_hash_match(None, "a" * 64), False)
    _check(fails, "configuration match at 24 layers and hidden 2048",
           exact_config_match(24, 2048), True)
    _check(fails, "configuration layer mismatch", exact_config_match(23, 2048), False)
    _check(fails, "the smaller sibling's hidden size is a mismatch here",
           exact_config_match(24, 1024), False)
    _check(fails, "a named checkpoint matches", exact_revision_match("step16000"), True)
    _check(fails, "an unnamed checkpoint never matches",
           exact_revision_match("step17000"), False)
    _check(fails, "a missing revision never matches", exact_revision_match(None), False)
    _check(fails, "lowest window value maps to slot 0", label_slot(1, 1, 140), 0)
    _check(fails, "highest window value maps to the last slot",
           label_slot(140, 1, 140), 139)
    _check(fails, "one below the window is outside", label_slot(0, 1, 140), None)
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
    _check(fails, "counts from labels", counts_from_labels([1, 1, 2]), {1: 2, 2: 1})

    print("== the twelve committed checkpoints and the committed measurement order ==")
    _check(fails, "twelve checkpoints", len(REVISIONS), 12)
    _check(fails, "the step in a name is read exactly", step_of("step143000"), 143000)
    _check(fails, "initialization is step 0", step_of("step0"), 0)
    try:
        step_of("stepfinal")
        fails.append("a name with no whole-number step was accepted")
        print("  %-74s FAIL" % "a name with no whole-number step halts")
    except Halt:
        print("  %-74s ok" % "a name with no whole-number step halts")
    _check(fails, "the gate cell is measured first", MEASURE_ORDER[0], "step143000")
    _check(fails, "the baseline is measured second", MEASURE_ORDER[1], "step0")
    _check(fails, "the remaining ten run earliest to latest",
           [step_of(r) for r in MEASURE_ORDER[2:]],
           sorted(step_of(r) for r in MEASURE_ORDER[2:]))
    _check(fails, "the order covers exactly the twelve", sorted(MEASURE_ORDER),
           sorted(REVISIONS))
    _check(fails, "the steps rise through the committed ladder",
           [step_of(r) for r in REVISIONS],
           [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 143000])
    _check(fails, "seven wrap widths are read, as the gate record's first answer says",
           USE_WIDTHS, [20, 40, 60, 80, 100, 120, 140])
    _check(fails, "the two wide widths are not read",
           [w for w in STIMULUS_WIDTHS if w not in USE_WIDTHS], [200, 280])
    _check(fails, "forward passes per checkpoint", N_DOCS * len(USE_WIDTHS), 336)

    print("== the four profile parts and the boundary ratio ==")
    _check(fails, "the parts are the committed ranges", label_bins(),
           [(1, 35), (36, 70), (71, 105), (106, 140)])
    _check(fails, "a strong low half and a dead high half give a large ratio",
           boundary_ratio([1.0, 1.0, 0.0, 0.0]), 1.0, 1e-12)
    _check(fails, "the committed parts on this text reproduce 0.514227",
           round(boundary_ratio(PRIOR_PROFILE_SAME_TEXT), 6), 0.514227, 1e-9)
    _check(fails, "the design states that same ratio as 0.514",
           round(boundary_ratio(PRIOR_PROFILE_SAME_TEXT), 3),
           PRIOR_RATIO_SAME_TEXT_AS_COMMITTED, 1e-9)
    _check(fails, "the fresh-text parts on this model reproduce 0.519079",
           round(boundary_ratio(PRIOR_PROFILE_FRESH_TEXT), 6), 0.519079, 1e-9)
    _check(fails, "a flat profile gives no boundary",
           boundary_ratio([0.4, 0.4, 0.4, 0.4]), 0.0, 1e-12)
    _check(fails, "an absent part leaves the ratio absent",
           boundary_ratio([1.0, None, 0.0, 0.0]), None)
    _check(fails, "a profile of the wrong length is refused",
           boundary_ratio([1.0, 1.0, 0.0]), None)

    print("== the coordinate control, and the recorded localization moment ==")
    _check(fails, "an interior selection is clean", coordinate_flags(5, 1000.0), [])
    _check(fails, "the embedding output is an edge", coordinate_flags(0, 1000.0),
           ["best_layer_at_readout_0"])
    _check(fails, "the last layer is an edge", coordinate_flags(24, 1000.0),
           ["best_layer_at_readout_24"])
    _check(fails, "the weakest strength is an edge", coordinate_flags(5, 1.0),
           ["ridge_strength_at_1"])
    _check(fails, "the strongest strength is an edge", coordinate_flags(5, 100000.0),
           ["ridge_strength_at_100000"])
    _check(fails, "both edges together, the case expected at initialization",
           coordinate_flags(0, 100000.0),
           ["best_layer_at_readout_0", "ridge_strength_at_100000"])
    _check(fails, "an interior selection reads interior", decide_coordinates(5, 1000.0),
           "COORDINATES_INTERIOR")
    _check(fails, "an edge selection reads as an edge", decide_coordinates(24, 1.0),
           "COORDINATES_AT_GRID_EDGE")
    _check(fails, "the prior run's own selection on this model is interior",
           decide_coordinates(PRIOR_READOUT_SAME_TEXT, PRIOR_RIDGE_SAME_TEXT),
           "COORDINATES_INTERIOR")
    _check(fails, "the middle of five values is the median", median_of([1, 5, 2, 4, 3]),
           3.0, 1e-12)
    _check(fails, "an absent value is dropped before the median",
           median_of([1.0, None, 3.0]), 2.0, 1e-12)
    _check(fails, "nothing left leaves no median", median_of([None]), None)
    _check(fails, "an interior peak a clear margin above the median is localized",
           decide_localization(5, 0.80, 0.60), "LOCALIZED_INTERIOR_PEAK")
    _check(fails, "a margin of exactly 0.1 counts",
           decide_localization(5, 0.70, 0.60), "LOCALIZED_INTERIOR_PEAK")
    _check(fails, "a margin just under 0.1 does not",
           decide_localization(5, 0.699999999, 0.60), "NOT_YET_LOCALIZED")
    _check(fails, "a peak at the sweep edge is never localized",
           decide_localization(24, 0.90, 0.30), "NOT_YET_LOCALIZED")
    _check(fails, "the untrained sibling's readout profile is not yet localized",
           decide_localization(SIBLING_BASELINE_READOUT, SIBLING_BASELINE_POOLED,
                               0.311371), "NOT_YET_LOCALIZED")
    _check(fails, "an absent median leaves localization uncomputable",
           decide_localization(5, 0.8, None), "LOCALIZATION_NOT_COMPUTABLE")

    print("== comparisons judged in decimals, not in bits ==")
    _check(fails, "a value on a floor reaches it", not_below(0.3, 0.3), True)
    _check(fails, "a value under a floor does not", not_below(0.29999, 0.3), False)
    _check(fails, "a value on a limit stays within it", not_above(0.15, 0.15), True)
    _check(fails, "a value over a limit does not", not_above(0.15001, 0.15), False)
    _check(fails, "the low band edge is not pushed out by binary noise",
           not_above(abs((ANCHOR_POOLED - ANCHOR_TOLERANCE) - ANCHOR_POOLED),
                     ANCHOR_TOLERANCE), True)
    _check(fails, "a range of exactly the quality limit stays within it",
           not_above(series_range([0.776302, 0.876302]), HOLD_QUALITY_RANGE), True)

    print("== endpoint 1, the gate cell on both its bars ==")
    _check(fails, "the two committed values reproduce themselves",
           decide_anchor(ANCHOR_POOLED, boundary_ratio(PRIOR_PROFILE_SAME_TEXT)),
           "ANCHOR_IN_BAND")
    _check(fails, "the low edge of the quality band is inside it",
           decide_anchor(ANCHOR_POOLED - ANCHOR_TOLERANCE, 0.5), "ANCHOR_IN_BAND")
    _check(fails, "the high edge of the quality band is inside it",
           decide_anchor(ANCHOR_POOLED + ANCHOR_TOLERANCE, 0.5), "ANCHOR_IN_BAND")
    _check(fails, "just under the quality band voids the sweep",
           decide_anchor(ANCHOR_POOLED - ANCHOR_TOLERANCE - 1e-6, 0.5),
           "VOID_ANCHOR_FAILED")
    _check(fails, "just over the quality band voids the sweep",
           decide_anchor(ANCHOR_POOLED + ANCHOR_TOLERANCE + 1e-6, 0.5),
           "VOID_ANCHOR_FAILED")
    _check(fails, "the fresh-text quality on this model would also pass this band",
           decide_anchor(PRIOR_POOLED_FRESH_TEXT, 0.5), "ANCHOR_IN_BAND")
    _check(fails, "a ratio exactly on its floor passes",
           decide_anchor(ANCHOR_POOLED, ANCHOR_RATIO_FLOOR), "ANCHOR_IN_BAND")
    _check(fails, "a ratio just under its floor voids the sweep",
           decide_anchor(ANCHOR_POOLED, ANCHOR_RATIO_FLOOR - 1e-9),
           "VOID_ANCHOR_FAILED")
    _check(fails, "the smaller sibling's own final ratio would void this gate",
           decide_anchor(ANCHOR_POOLED, SIBLING_FINAL_RATIO), "VOID_ANCHOR_FAILED")
    _check(fails, "an absent quality voids the sweep", decide_anchor(None, 0.5),
           "VOID_ANCHOR_FAILED")
    _check(fails, "an absent ratio voids the sweep",
           decide_anchor(ANCHOR_POOLED, None), "VOID_ANCHOR_FAILED")
    _check(fails, "a not-a-number quality voids the sweep",
           decide_anchor(float("nan"), 0.5), "VOID_ANCHOR_FAILED")

    print("== endpoint 2, the baseline floor at initialization ==")
    _check(fails, "the sibling's measured floor is recorded, not an alarm",
           decide_baseline(SIBLING_BASELINE_POOLED), "BASELINE_RECORDED")
    _check(fails, "a quality near zero is recorded", decide_baseline(0.01),
           "BASELINE_RECORDED")
    _check(fails, "a negative quality is recorded", decide_baseline(-0.4),
           "BASELINE_RECORDED")
    _check(fails, "0.3 no longer voids anything, unlike the previous rung",
           decide_baseline(0.3), "BASELINE_RECORDED")
    _check(fails, "the top of the expected range is recorded",
           decide_baseline(BASELINE_EXPECTED_HIGH), "BASELINE_RECORDED")
    _check(fails, "just below the void floor is recorded",
           decide_baseline(BASELINE_VOID_FLOOR - 1e-9), "BASELINE_RECORDED")
    _check(fails, "exactly at the void floor voids the sweep",
           decide_baseline(BASELINE_VOID_FLOOR), "VOID_CONTROL_FAILED")
    _check(fails, "above the void floor voids the sweep", decide_baseline(0.7),
           "VOID_CONTROL_FAILED")
    _check(fails, "an absent quality voids the sweep", decide_baseline(None),
           "VOID_CONTROL_FAILED")
    _check(fails, "the void floor is 0.7 times the anchor quality",
           round(BASELINE_VOID_FLOOR, 7), 0.6134114, 1e-12)
    _check(fails, "the design states that floor as about 0.613",
           round(BASELINE_VOID_FLOOR, 3), 0.613, 1e-12)
    _check(fails, "the sibling's floor sits inside the expected range",
           baseline_inside_expected(SIBLING_BASELINE_POOLED), True)
    _check(fails, "a floor above the expected range is still only recorded",
           baseline_inside_expected(0.55), False)

    print("== the two formation thresholds, measured up from the baseline ==")
    t_form, t_formed = formation_thresholds(0.35)
    _check(fails, "half of the distance from 0.35 to the anchor", round(t_form, 6),
           0.613151, 1e-9)
    _check(fails, "nine tenths of that distance", round(t_formed, 6), 0.823672, 1e-9)
    _check(fails, "a floor at the anchor leaves both thresholds at the anchor",
           formation_thresholds(ANCHOR_POOLED), (ANCHOR_POOLED, ANCHOR_POOLED))
    _check(fails, "an absent floor leaves both thresholds absent",
           formation_thresholds(None), (None, None))
    t_form0, t_formed0 = formation_thresholds(0.0)
    _check(fails, "a zero floor reduces to half the anchor", round(t_form0, 6),
           round(0.5 * ANCHOR_POOLED, 6), 1e-9)
    _check(fails, "a zero floor reduces to nine tenths of the anchor",
           round(t_formed0, 6), round(0.9 * ANCHOR_POOLED, 6), 1e-9)
    _check(fails, "reaching the formed threshold is formation",
           decide_formation_exists(t_formed, t_formed), "FORMATION_REACHED")
    _check(fails, "just under it is not",
           decide_formation_exists(t_formed - 1e-9, t_formed), "NOT_FORMED_BY_FINAL")
    _check(fails, "the committed anchor is past a 0.35 floor's formed threshold",
           decide_formation_exists(ANCHOR_POOLED, t_formed), "FORMATION_REACHED")
    _check(fails, "an absent threshold leaves formation unreached",
           decide_formation_exists(0.9, None), "NOT_FORMED_BY_FINAL")

    print("== endpoint 4, is the boundary there at the end ==")
    _check(fails, "reaching the boundary floor is presence",
           decide_boundary_presence(BOUNDARY_PRESENT_FLOOR), "BOUNDARY_PRESENT")
    _check(fails, "just under it is absence",
           decide_boundary_presence(BOUNDARY_PRESENT_FLOOR - 1e-9),
           "BOUNDARY_NOT_FORMED_BY_FINAL")
    _check(fails, "this model's committed ratio reads as present",
           decide_boundary_presence(boundary_ratio(PRIOR_PROFILE_SAME_TEXT)),
           "BOUNDARY_PRESENT")
    _check(fails, "the smaller sibling's final ratio reads as absent",
           decide_boundary_presence(SIBLING_FINAL_RATIO),
           "BOUNDARY_NOT_FORMED_BY_FINAL")
    _check(fails, "the untrained sibling's ratio reads as absent",
           decide_boundary_presence(SIBLING_BASELINE_RATIO),
           "BOUNDARY_NOT_FORMED_BY_FINAL")
    _check(fails, "an absent ratio reads as absent", decide_boundary_presence(None),
           "BOUNDARY_NOT_FORMED_BY_FINAL")

    print("== the two stability channels, both starting at the shared formed checkpoint ==")
    _check(fails, "a range at the boundary limit is inside it",
           decide_hold_channel(HOLD_BOUNDARY_RANGE, HOLD_BOUNDARY_RANGE), "WITHIN_BAND")
    _check(fails, "a range past the boundary limit is outside it",
           decide_hold_channel(HOLD_BOUNDARY_RANGE + 1e-9, HOLD_BOUNDARY_RANGE),
           "OUTSIDE_BAND")
    _check(fails, "a range at the quality limit is inside it",
           decide_hold_channel(HOLD_QUALITY_RANGE, HOLD_QUALITY_RANGE), "WITHIN_BAND")
    _check(fails, "a range past the quality limit is outside it",
           decide_hold_channel(0.2, HOLD_QUALITY_RANGE), "OUTSIDE_BAND")
    _check(fails, "an absent range is not computable",
           decide_hold_channel(None, HOLD_QUALITY_RANGE), "RANGE_NOT_COMPUTABLE")
    _check(fails, "a range is the largest minus the smallest",
           series_range([0.70, 0.78, 0.74]), 0.08, 1e-12)
    _check(fails, "one absent value leaves no range",
           series_range([0.70, None, 0.74]), None)
    _check(fails, "an empty series leaves no range", series_range([]), None)
    _check(fails, "a rising series runs one way", is_monotone([0.1, 0.2, 0.2, 0.3]), True)
    _check(fails, "a falling series runs one way", is_monotone([0.3, 0.2, 0.1]), True)
    _check(fails, "a series that turns around does not",
           is_monotone([0.1, 0.3, 0.2]), False)
    _check(fails, "a flat series runs one way", is_monotone([0.2, 0.2, 0.2]), True)
    _check(fails, "an absent value makes the question unanswerable",
           is_monotone([0.2, None]), False)
    _check(fails, "both channels inside their bands hold",
           decide_hold("WITHIN_BAND", "WITHIN_BAND", True, True), "HELD")
    _check(fails, "one channel outside and one-way drifts",
           decide_hold("OUTSIDE_BAND", "WITHIN_BAND", True, True), "DRIFTED")
    _check(fails, "one channel outside and not one-way is unclassifiable",
           decide_hold("OUTSIDE_BAND", "WITHIN_BAND", False, True),
           "UNSTABLE_NOT_CLASSIFIABLE")
    _check(fails, "an absent boundary leaves the quality channel to decide",
           decide_hold("NOT_PRESENT_AT_FINAL", "WITHIN_BAND", False, True), "HELD")
    _check(fails, "an absent boundary and a drifting quality still drifts",
           decide_hold("NOT_PRESENT_AT_FINAL", "OUTSIDE_BAND", False, True), "DRIFTED")
    _check(fails, "neither channel live leaves the stability test uncomputable",
           decide_hold("NOT_PRESENT_AT_FINAL", "RANGE_NOT_COMPUTABLE", False, False),
           "HOLD_TEST_NOT_COMPUTABLE")

    print("== endpoint 3, early against late ==")
    _check(fails, "the cutoff step itself counts as early",
           decide_timing(EARLY_CUTOFF_STEP), "EARLY_AT_OR_BEFORE_STEP16000")
    _check(fails, "one step later counts as late",
           decide_timing(EARLY_CUTOFF_STEP + 1), "LATE_AFTER_STEP16000")
    _check(fails, "initialization counts as early", decide_timing(0),
           "EARLY_AT_OR_BEFORE_STEP16000")
    _check(fails, "no formation leaves the timing undefined", decide_timing(None),
           "TIMING_UNDEFINED_NO_FORMATION")

    print("== endpoint 5, the boundary's own timing against the formation step ==")
    _check(fails, "the same checkpoint counts as arriving with the counter",
           decide_boundary_timing(2000, 2000), "BOUNDARY_WITH_COUNTER")
    _check(fails, "an earlier crossing counts as arriving with the counter",
           decide_boundary_timing(1000, 2000), "BOUNDARY_WITH_COUNTER")
    _check(fails, "a later crossing counts as arriving after it",
           decide_boundary_timing(4000, 2000), "BOUNDARY_AFTER_COUNTER")
    _check(fails, "a boundary that never crosses names an instrument fault",
           decide_boundary_timing(None, 2000), "BOUNDARY_NOT_FORMED_BY_FINAL")
    _check(fails, "no formation step leaves the comparison undefined",
           decide_boundary_timing(2000, None),
           "BOUNDARY_TIMING_UNDEFINED_NO_FORMATION")

    print("== the pre-registered readings, from the calls already made ==")
    _check(fails, "a failed gate cell voids everything",
           decide_headline("VOID_ANCHOR_FAILED", "BASELINE_RECORDED",
                           "FORMATION_REACHED", "EARLY_AT_OR_BEFORE_STEP16000", "HELD"),
           "VOID_ANCHOR_FAILED")
    _check(fails, "a baseline above its void floor voids everything",
           decide_headline("ANCHOR_IN_BAND", "VOID_CONTROL_FAILED", "FORMATION_REACHED",
                           "EARLY_AT_OR_BEFORE_STEP16000", "HELD"),
           "VOID_CONTROL_FAILED")
    _check(fails, "no formation reads as not formed",
           decide_headline("ANCHOR_IN_BAND", "BASELINE_RECORDED",
                           "NOT_FORMED_BY_FINAL", "TIMING_UNDEFINED_NO_FORMATION",
                           "HELD"), "NOT_FORMED_BY_FINAL")
    _check(fails, "early and holding reads as the fossil-consistent reading",
           decide_headline("ANCHOR_IN_BAND", "BASELINE_RECORDED", "FORMATION_REACHED",
                           "EARLY_AT_OR_BEFORE_STEP16000", "HELD"),
           "FORMED_EARLY_AND_HELD")
    _check(fails, "late and holding reads as the late variant",
           decide_headline("ANCHOR_IN_BAND", "BASELINE_RECORDED", "FORMATION_REACHED",
                           "LATE_AFTER_STEP16000", "HELD"), "FORMED_LATE_AND_HELD")
    _check(fails, "drift reads as the equilibrium-consistent reading",
           decide_headline("ANCHOR_IN_BAND", "BASELINE_RECORDED", "FORMATION_REACHED",
                           "EARLY_AT_OR_BEFORE_STEP16000", "DRIFTED"),
           "FORMED_AND_DRIFTED")
    _check(fails, "an unclassifiable stability test keeps its own name",
           decide_headline("ANCHOR_IN_BAND", "BASELINE_RECORDED", "FORMATION_REACHED",
                           "EARLY_AT_OR_BEFORE_STEP16000", "UNSTABLE_NOT_CLASSIFIABLE"),
           "UNSTABLE_NOT_CLASSIFIABLE")

    print("== the committed label-reachability control, replayed at three baseline floors ==")
    for floor in [CONTROL_BASELINE] + list(CONTROL_BASELINES_ALSO_CHECKED):
        seen = []
        for name in CONTROL_NAMES:
            want_label, want_form, want_formed = CONTROL_EXPECTED[name]
            got = sweep_readings(_control_rows(name, floor), "ANCHOR_IN_BAND",
                                 "BASELINE_RECORDED")
            seen.append(got["headline"])
            _check(fails, "floor %.2f, %s reads its committed label" % (floor, name),
                   got["headline"], want_label)
            _check(fails, "floor %.2f, %s formation step" % (floor, name),
                   got["formation_step"], want_form)
            _check(fails, "floor %.2f, %s formed checkpoint" % (floor, name),
                   got["formed_step"], want_formed)
            _check(fails, "floor %.2f, %s planted floor is read back" % (floor, name),
                   got["baseline_quality"], floor, 1e-12)
        _check(fails, "floor %.2f, all five committed labels are reachable" % floor,
               sorted(set(seen)),
               sorted(set(v[0] for v in CONTROL_EXPECTED.values())))
        _check(fails, "floor %.2f, five labels for five trajectories" % floor,
               len(set(seen)), 5)

    early = sweep_readings(_control_rows("early-sigmoid-flat"), "ANCHOR_IN_BAND",
                           "BASELINE_RECORDED")
    _check(fails, "the early control's boundary arrives with the counter",
           early["boundary_timing_decision"], "BOUNDARY_WITH_COUNTER")
    _check(fails, "the early control's boundary crossing is its formation step",
           early["boundary_crossing_step"], 2000)
    _check(fails, "the early control's quality range is inside its band",
           early["hold_test_quality_channel"], "WITHIN_BAND")
    _check(fails, "the early control's boundary range is inside its band",
           early["hold_test_boundary_channel"], "WITHIN_BAND")
    _check(fails, "the early control's boundary is present at the end",
           early["boundary_presence_decision"], "BOUNDARY_PRESENT")
    _check(fails, "eight checkpoints sit at or after the early control's formed one",
           len(early["checkpoints_at_or_after_formed"]), 8)
    late = sweep_readings(_control_rows("late-sigmoid-flat"), "ANCHOR_IN_BAND",
                          "BASELINE_RECORDED")
    _check(fails, "the late control's boundary crosses before its counter forms",
           late["boundary_crossing_step"], 16000)
    _check(fails, "so the late control's boundary still arrives with the counter",
           late["boundary_timing_decision"], "BOUNDARY_WITH_COUNTER")
    drifted = sweep_readings(_control_rows("form-then-drift-up"), "ANCHOR_IN_BAND",
                             "BASELINE_RECORDED")
    _check(fails, "the drifting control's boundary range leaves its band",
           drifted["hold_test_boundary_channel"], "OUTSIDE_BAND")
    _check(fails, "the drifting control's boundary runs one way in steps",
           drifted["boundary_series_one_way_in_steps"], True)
    noisy = sweep_readings(_control_rows("noisy-plateau"), "ANCHOR_IN_BAND",
                           "BASELINE_RECORDED")
    _check(fails, "the noisy control's quality range leaves its band",
           noisy["hold_test_quality_channel"], "OUTSIDE_BAND")
    _check(fails, "the noisy control's quality does not run one way",
           noisy["quality_series_one_way_in_steps"], False)
    never = sweep_readings(_control_rows("never-forms"), "ANCHOR_IN_BAND",
                           "BASELINE_RECORDED")
    _check(fails, "the never-forming control has no formed checkpoint",
           never["formed_step"], None)
    _check(fails, "so its stability test has nothing to read",
           never["hold_test_decision"], "HOLD_TEST_NOT_COMPUTABLE")
    _check(fails, "each control trajectory carries twelve quality values",
           [len(control_trajectory(n)[0]) for n in CONTROL_NAMES], [12] * 5)
    _check(fails, "each control trajectory carries twelve ratio values",
           [len(control_trajectory(n)[1]) for n in CONTROL_NAMES], [12] * 5)
    try:
        control_trajectory("no-such-shape")
        fails.append("an unnamed control trajectory was accepted")
        print("  %-74s FAIL" % "an unnamed control trajectory halts")
    except Halt:
        print("  %-74s ok" % "an unnamed control trajectory halts")

    print("== the whole trajectory, on planted checkpoints with known calls ==")
    fossil_q = [0.35, 0.40, 0.55, 0.65, 0.84, 0.85, 0.855, 0.86, 0.865, 0.87, 0.874,
                0.876]
    fossil_r = [0.00, 0.05, 0.48, 0.49, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    read = sweep_readings(_planted_sweep(fossil_q, fossil_r), "ANCHOR_IN_BAND",
                          "BASELINE_RECORDED")
    _check(fails, "the planted floor is read from the initialization checkpoint",
           read["baseline_quality"], 0.35, 1e-12)
    _check(fails, "planted early formation is found at the half-distance crossing",
           read["formation_step"], 2000)
    _check(fails, "planted formed checkpoint is the nine-tenths crossing",
           read["formed_step"], 4000)
    _check(fails, "the formed threshold is measured up from the planted floor",
           read["formed_threshold"], 0.35 + 0.9 * (ANCHOR_POOLED - 0.35), 1e-12)
    _check(fails, "planted rows read as formed early and holding", read["headline"],
           "FORMED_EARLY_AND_HELD")
    _check(fails, "the boundary already stood at the formation step",
           read["boundary_ratio_at_formation_step"], 0.49, 1e-12)
    _check(fails, "the planted localization moment is the earliest checkpoint",
           read["localization_step_recorded_only"], 0)
    _check(fails, "a planted row carries every number the trajectory stage reads",
           [k for k in ROW_KEYS_NEEDED if k not in _planted_row("step0", 0.1, 0.2)], [])
    _check(fails, "the planted profile matches the planted boundary ratio",
           boundary_ratio(_planted_row("step0", 0.1, 0.2)["profile_correlations"]),
           0.2, 1e-9)
    _check(fails, "an interior checkpoint is not flagged degenerate",
           read["degenerate_coordinates_by_step"]["143000"], False)
    edge = sweep_readings([_planted_row(r, 0.5, 0.5, layer=0, ridge=1.0)
                           for r in REVISIONS], "ANCHOR_IN_BAND", "BASELINE_RECORDED")
    _check(fails, "a checkpoint at a coordinate edge is flagged degenerate",
           edge["degenerate_coordinates_by_step"]["0"], True)
    _check(fails, "a sweep whose peak never leaves the edge is never localized",
           edge["localization_step_recorded_only"], None)
    _check(fails, "the direction counts travel with the rows",
           read["dimensions_by_step"]["143000"], {"0.9": 1, "0.95": 2, "0.99": 5})
    _check(fails, "the alternative crossings against the measured final are recorded",
           read["formed_step_from_measured_final_recorded_only"], 4000)

    read_void = sweep_readings(_planted_sweep(fossil_q, fossil_r), "VOID_ANCHOR_FAILED",
                               "BASELINE_RECORDED")
    _check(fails, "a failed gate cell voids the trajectory reading too",
           read_void["headline"], "VOID_ANCHOR_FAILED")
    read_void2 = sweep_readings(_planted_sweep(fossil_q, fossil_r), "ANCHOR_IN_BAND",
                                "VOID_CONTROL_FAILED")
    _check(fails, "a voided baseline voids the trajectory reading too",
           read_void2["headline"], "VOID_CONTROL_FAILED")
    _check(fails, "the earliest checkpoint reaching a floor is found",
           first_reaching(_planted_sweep(fossil_q, fossil_r), 0.5), (1000, "step1000"))
    _check(fails, "a floor nobody reaches has no checkpoint",
           first_reaching(_planted_sweep(fossil_q, fossil_r), 0.99), (None, None))
    _check(fails, "an absent threshold finds no checkpoint",
           first_reaching(_planted_sweep(fossil_q, fossil_r), None), (None, None))
    _check(fails, "the earliest checkpoint reaching a ratio floor is found",
           first_reaching(_planted_sweep(fossil_q, fossil_r), 0.3,
                          key="boundary_ratio"), (1000, "step1000"))
    for dropped, why in ((FINAL_REVISION, "the final checkpoint"),
                         (BASELINE_REVISION, "the initialization checkpoint")):
        try:
            sweep_readings([r for r in _planted_sweep(fossil_q, fossil_r)
                            if r["revision"] != dropped], "ANCHOR_IN_BAND",
                           "BASELINE_RECORDED")
            fails.append("a trajectory without %s was accepted" % why)
            print("  %-74s FAIL" % ("a trajectory missing %s halts" % why))
        except Halt:
            print("  %-74s ok" % ("a trajectory missing %s halts" % why))
    flat_r = [0.0, 0.0, 0.05, 0.05, 0.06, 0.05, 0.04, 0.05, 0.05, 0.06, 0.05, 0.05]
    read_flat = sweep_readings(_planted_sweep(fossil_q, flat_r), "ANCHOR_IN_BAND",
                               "BASELINE_RECORDED")
    _check(fails, "a boundary absent at the end drops out of the stability test",
           read_flat["hold_test_boundary_channel"], "NOT_PRESENT_AT_FINAL")
    _check(fails, "and that absence names an instrument fault of its own",
           read_flat["boundary_timing_decision"], "BOUNDARY_NOT_FORMED_BY_FINAL")
    _check(fails, "the quality readings still stand when the boundary is absent",
           read_flat["headline"], "FORMED_EARLY_AND_HELD")
    late_boundary_r = [0.0, 0.0, 0.05, 0.05, 0.10, 0.20, 0.35, 0.40, 0.42, 0.43, 0.44,
                       0.45]
    read_late_b = sweep_readings(_planted_sweep(fossil_q, late_boundary_r),
                                 "ANCHOR_IN_BAND", "BASELINE_RECORDED")
    _check(fails, "a boundary crossing after the formation step reads as later",
           read_late_b["boundary_timing_decision"], "BOUNDARY_AFTER_COUNTER")
    _check(fails, "that crossing is recorded with its own checkpoint",
           read_late_b["boundary_crossing_step"], 16000)

    print("== constants, notes, and the record writer ==")
    _check(fails, "the model of this rung", MODEL_ID, "EleutherAI/pythia-1.4b-deduped")
    _check(fails, "the layer count expected of it", EXPECT_LAYERS, 24)
    _check(fails, "the hidden size expected of it", EXPECT_HIDDEN, 2048)
    _check(fails, "the smaller sibling of the previous rung", SIBLING_MODEL_ID,
           "EleutherAI/pythia-410m-deduped")
    _check(fails, "the text file of this rung", STIMULUS_NAME, "stimulus_linewrap_v2.json")
    _check(fails, "the committed hash of that text file", STIMULUS_SHA256,
           "b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd")
    _check(fails, "the results folder of this rung", RESULTS_DIR_DEFAULT,
           "results_seedD4_1")
    _check(fails, "the committed anchor quality", ANCHOR_POOLED, 0.876302, 1e-12)
    _check(fails, "the committed anchor tolerance", ANCHOR_TOLERANCE, 0.02, 1e-12)
    _check(fails, "the gate's boundary bar", ANCHOR_RATIO_FLOOR, 0.3, 1e-12)
    _check(fails, "the baseline void share of the anchor", BASELINE_VOID_SHARE, 0.7,
           1e-12)
    _check(fails, "the recorded expected baseline range",
           [BASELINE_EXPECTED_LOW, BASELINE_EXPECTED_HIGH], [0.2, 0.45])
    _check(fails, "the committed formation share of the gap", FORMATION_SHARE, 0.5,
           1e-12)
    _check(fails, "the committed formed share of the gap", FORMED_SHARE, 0.90, 1e-12)
    _check(fails, "the committed boundary floor", BOUNDARY_PRESENT_FLOOR, 0.3, 1e-12)
    _check(fails, "the committed boundary stability range", HOLD_BOUNDARY_RANGE, 0.15,
           1e-12)
    _check(fails, "the committed quality stability range", HOLD_QUALITY_RANGE, 0.10,
           1e-12)
    _check(fails, "the committed early-late cutoff", EARLY_CUTOFF_STEP, 16000)
    _check(fails, "the recorded localization margin", LOCALIZATION_MARGIN, 0.1, 1e-12)
    _check(fails, "the committed ridge grid", RIDGE_GRID,
           [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    _check(fails, "the committed fold count", N_FOLDS, 5)
    _check(fails, "the one committed base seed", SEED, 0)
    _check(fails, "the three recorded variance cutoffs", VARIANCE_TARGETS,
           [0.90, 0.95, 0.99])
    _check(fails, "twelve groups of running totals", len(group_names()), 12)
    _check(fails, "every group name is distinct", len(set(group_names())),
           len(group_names()))
    _check(fails, "one checkpoint's totals in gigabytes",
           round(totals_bytes() / 1e9, 2), 10.08, 0.01)
    _check(fails, "one group's totals in megabytes",
           round(totals_bytes(n_groups=1) / 1e6), 840)
    _check(fails, "the room-to-work floors clear those totals",
           bool(MEMORY_FLOOR_MB * 1e6 > totals_bytes()), True)
    _check(fails, "no output calls a same-box rerun a replication",
           "replication" in RERUN_NOTE and "not a replication" in RERUN_NOTE, True)
    _check(fails, "the coordinate note states the expectation at initialization",
           "expected to trip" in COORDINATE_NOTE, True)
    _check(fails, "the baseline note says the untrained checkpoint is not an alarm",
           "not a near-zero alarm" in BASELINE_NOTE, True)
    _check(fails, "the boundary note says why the bar stays absolute",
           "no baseline correction" in BOUNDARY_NOTE, True)
    _check(fails, "the marker note names which crossing decides early against late",
           "half-distance" in MARKER_NOTE, True)
    _check(fails, "the control note names the committed reachability control",
           "five synthetic" in CONTROL_REPLAY_NOTE, True)
    _check(fails, "the forced-outcome note says which reading cannot fire",
           "cannot fire once the gate passes" in FORCED_OUTCOME_NOTE, True)
    _check(fails, "the design forbids downloading the corpus",
           "No corpus is downloaded" in NO_CORPUS_NOTE, True)
    try:
        prov = load_provenance()
        print("  %-74s ok" % "record writer importable")
        for name in ("write_result", "describe_input", "cell_slug"):
            if not hasattr(prov, name):
                fails.append("record writer lacks " + name)
                print("  %-74s FAIL" % ("record writer has " + name))
        slug = prov.cell_slug(base_cell("extract", "anchor_gate", "step143000",
                                        endpoint=1))
        _check(fails, "a record's name carries its own identity",
               slug, "arm-anchor_gate_endpoint-1_model-pythia-1.4b-deduped_"
                     "revision-step143000_stage-extract")
    except Exception as exc:                                    # noqa: BLE001
        fails.append("record writer import: %s" % exc)
        print("  %-74s FAIL %s" % ("record writer importable", exc))

    try:
        import numpy as np
    except ImportError:
        print("  %-74s skipped (numpy absent)" % "fits, correlations, average vectors")
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
    _check(fails, "adding two groups' totals fits their union",
           float(abs(sum_groups({"gram": {"a": gram[None, :, :], "b": gram[None, :, :]},
                                 "vec_sum": {"a": vec[None, :], "b": vec[None, :]},
                                 "cross": {"a": cross[None, :], "b": cross[None, :]},
                                 "label_sum": {"a": 1.0, "b": 2.0},
                                 "label_sq": {"a": 3.0, "b": 4.0},
                                 "n_pos": {"a": 5, "b": 6}}, ["a", "b"], 0)[3] - 3.0)),
           0.0, 1e-12)

    print("== the four-part profile on planted rows with a known boundary ==")
    truth = np.array([10.0, 12.0, 14.0, 16.0])
    _check(fails, "a perfect prediction correlates 1",
           pearson_corr(truth.copy(), truth)[0], 1.0, 1e-12)
    _check(fails, "a shifted and stretched prediction still correlates 1",
           pearson_corr(3.0 * truth + 100.0, truth)[0], 1.0, 1e-12)
    _check(fails, "a reversed prediction correlates minus 1",
           pearson_corr(-truth, truth)[0], -1.0, 1e-12)
    _check(fails, "a flat prediction has no correlation",
           pearson_corr(np.full(4, 13.0), truth)[0], None)
    _check(fails, "a part with one position is not computable",
           pearson_corr(np.array([1.0]), np.array([1.0]))[0], None)
    _check(fails, "the count of positions in a part is reported",
           pearson_corr(truth.copy(), truth)[1], 4)
    labs = np.arange(1.0, 141.0)
    preds = labs.copy()
    high = labs > 70
    preds[high] = 100.0
    preds[high] += np.arange(int(high.sum())) % 2          # alternating, so no trend
    rows140 = bin_correlation_rows(preds, labs, label_bins())
    corrs = [r["pearson_correlation"] for r in rows140]
    _check(fails, "four planted profile rows", len(rows140), 4)
    _check(fails, "each part holds its 35 counts",
           [r["n_positions"] for r in rows140], [35, 35, 35, 35])
    _check(fails, "the planted low half reads 1", corrs[:2], [1.0, 1.0])
    _check(fails, "the planted high half is near zero",
           bool(abs(corrs[2]) < 0.05 and abs(corrs[3]) < 0.05), True)
    _check(fails, "the planted boundary ratio is near 1",
           bool(abs(boundary_ratio(corrs) - 1.0) < 0.05), True)
    _check(fails, "the planted profile reads as a present boundary",
           decide_boundary_presence(boundary_ratio(corrs)), "BOUNDARY_PRESENT")

    print("== average vectors per count, and how many directions they use ==")
    labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    x = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0], [9.0, 0.0],
                  [11.0, 0.0]], dtype=np.float32)
    slots_arr = slots_array(labels, 1, 3)
    _check(fails, "slots for a window starting at 1", slots_arr.tolist(),
           [0, 0, 1, 1, 2, 2])
    means, counts = means_by_slot(x, slots_arr, 3)
    _check(fails, "one average vector per count", means.shape, (3, 2))
    _check(fails, "each count averaged its own positions",
           [round(float(v), 6) for v in means[:, 0]], [2.0, 3.0, 10.0])
    _check(fails, "the positions per count are reported", counts.tolist(), [2, 2, 2])
    share = variance_shares(means - means.mean(axis=0, keepdims=True))
    _check(fails, "a one-direction table needs one direction",
           dimension_at(share, 0.95), 1)
    _check(fails, "the three cutoffs are all reported",
           sorted(dimensions_at_targets(share)), ["0.9", "0.95", "0.99"])
    _check(fails, "a table with no spread has no direction count",
           variance_shares(np.zeros((3, 2))), None)
    _check(fails, "an absent share leaves the count absent",
           dimension_at(None, 0.95), None)
    ramp = np.zeros((6, 3))
    ramp[:, 0] = np.arange(6.0)
    ramp[:, 1] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    share2 = variance_shares(ramp - ramp.mean(axis=0, keepdims=True))
    _check(fails, "a ramp plus an alternation needs two directions",
           dimension_at(share2, 0.99), 2)
    empty_slots = np.array([0, 0, 2, 2], dtype=np.int64)
    try:
        means_by_slot(np.ones((4, 2), dtype=np.float32), empty_slots, 3)
        fails.append("an empty count slot was accepted")
        print("  %-74s FAIL" % "an empty count slot halts")
    except Halt:
        print("  %-74s ok" % "an empty count slot halts")

    print("== the fold split from the one committed base seed ==")
    assign = fold_of_documents(list(range(N_TRAIN_DOCS)))
    _check(fails, "every training document lands in a group", len(assign), N_TRAIN_DOCS)
    _check(fails, "five groups are used", sorted(set(assign.values())), [0, 1, 2, 3, 4])
    sizes = sorted(sum(1 for v in assign.values() if v == f) for f in range(N_FOLDS))
    _check(fails, "the groups are the same size", sizes, [7, 7, 7, 7, 8])
    _check(fails, "the split repeats from the same seed",
           fold_of_documents(list(range(N_TRAIN_DOCS))), assign)

    print("selftest:", "PASS" if not fails else "%d FAILURES" % len(fails))
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Part 13. Entry point.
# ---------------------------------------------------------------------------

STAGE_ORDER = ["extract", "endpoints"]

# The build brief names a middle position "generate". Nothing is generated here, so that
# name is only a second way to ask for the sweep. Asking for it prints what the sweep
# really does, then runs it.
STAGES = {"extract": stage_extract, "generate": stage_extract,
          "endpoints": stage_endpoints}

NO_GENERATION_MESSAGE = (
    "No text is generated in this experiment. The design says forward passes only: one "
    "deterministic pass per document per wrap width, over fixed committed text, with the "
    "model in evaluation mode. The position the build brief calls generation is filled "
    "here by the sweep, which also fixes each checkpoint's ridge strength on training "
    "documents only before any held-out number for that checkpoint exists. It is also "
    "reachable as --stage extract, and running it twice measures the same twelve "
    "checkpoints twice."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES) + ["all"],
                    help="extract (also spelled generate), endpoints, or all")
    ap.add_argument("--selftest", action="store_true",
                    help="standard-library checks only; no model, no graphics card")
    ap.add_argument("--results-dir",
                    default=os.environ.get("RESULTS_DIR", RESULTS_DIR_DEFAULT))
    ap.add_argument("--stimulus", default=None, help="path to the committed text file")
    ap.add_argument("--design-record", default=None,
                    help="path to the committed design record")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.stage:
        ap.error("give --stage extract, generate, endpoints, or all; or give --selftest")
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
