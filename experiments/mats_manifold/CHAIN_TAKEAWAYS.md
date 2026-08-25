# Does a Small Language Model's Character Counter Ripple? A Five-Run Chain.

Written by the orchestrating session on 2026-08-22, from the committed records
only. Every number below lives in a result record inside the per-run output
folders or a committed file in the records folder; nothing is restated from
memory. Scope markers: [1] means the base 410-million-weight model at its
final public training checkpoint; [2] means the base 1.4-billion-weight model
at its final public training checkpoint; [3] means the chain as a whole.

## The question, and where it came from

A study of a large closed model found that a language model internally tracks
how many characters have passed since the last line break, as a smooth curve
in its hidden activity, and that the curve's similarity structure shows a
faint repeating wave, a "ripple", explained as the cost of squeezing a rich
structure into about six usable directions. Nobody had measured that ripple
on an open model. This chain asked: does it exist in open models whose full
training history is public, where a later study could watch it form?

Five runs executed on rented single-graphics-card machines, total compute
cost about 3.60 dollars [3]: two runs on the base 410-million-weight model
and three on the base 1.4-billion-weight model, all at final public training
checkpoints, using committed, hash-pinned text stimuli, three mutually
disjoint text samples, and thresholds registered before each run.

## What was found

1. The counter is real and replicates almost exactly. A linear reading of the
   character count from hidden activity, fit on training documents and scored
   on held-out ones (pooled over line widths 20 to 140), reached 0.780903 and
   then 0.781031 on two disjoint texts, a gap of 0.000128, both at layer 6 of
   24 [1]. The larger model reached 0.876302 and then 0.879879 on two further
   disjoint texts, both at layer 5 [2]. Scale improves the counter.

2. The counter's fine resolution ends near seventy characters. Within
   stretches of 35 count values, the correlation between predicted and true
   count on held-out text reads 0.646, 0.820, 0.211, 0.227 across counts 1 to
   140 in one run, and 0.650, 0.830, 0.198, 0.244 in the next, on different
   text [2]. Exact counting collapses past roughly seventy characters, which
   is where the training corpus's dominant hard-wrapped line length sits
   (about 70 to 72 characters). Past that boundary the model knows roughly
   how far along the line it is, not exactly. On 280-character lines, counts
   above 175 correlate at only 0.10 to 0.31 [2]: long lines are
   out-of-distribution wholesale.

3. No ripple, anywhere tested. Under the final committed statistic (the
   strongest repeating component with at least two full cycles, judged
   against 200 label-shuffled comparison runs per case), no probe window
   (140, 200, or 280 characters) cleared its comparison on either base model,
   in either the character-count or the token-position labelling [3]. At most
   windows the real structure is smoother than nearly every shuffled version
   (ranks up to 201 of 200) [3], which is what a clean low-dimensional curve
   with nothing folded into it looks like. The original study's ripple
   account did not transfer to these models at these scales.

4. The one recurring "signal" is the curve itself. A one-cycle, falling
   component (dropping about 0.36 to minus 0.16 across the widest window,
   correlation with position minus 0.92 to minus 0.998) beat the shuffle
   comparison twice at the widest window [2]. It is not a repeating pattern:
   it is the counter's own smooth similarity decay, incompletely removed by
   the trend-removal step. The chain's one statistic change (barring the
   one-cycle bin) separates the established curve from the question asked on
   top of it.

5. The ripple's precondition is absent. The count structure occupies one to
   three directions at 95 percent of its spread in nearly every measurement
   on both base models [3]; the original study's ripple lived at about six.
   The single contrary reading (15 directions, one run, one window) collapsed
   to 2 on fresh text and varied tenfold with the percentage convention [2]:
   it was measurement noise, exposed by the registered alternate conventions.

## What this means for the parent question

The parent program asked whether the ripple's geometry is an equilibrium or a
fossil, to be decided by sweeping the smaller model family's 154 public
training checkpoints. That sweep has no object at the scales tested here:
there is no ripple to track on either base model [3]. The counter itself, and
its seventy-character resolution boundary, are sweepable objects, and the
boundary's location (at the training corpus's wrap pressure) is itself a
formation question. The transfer question stays open: the ripple exists in a
far larger closed model, and the strongest-curved anchored open model (a
nine-billion-weight model, reproduction quality 0.868 on its base form)
releases no training checkpoints. Both forks are a human decision.

## The instrument's own history, kept honest

The chain corrected itself five times, each correction committed before the
run that used it [3]. A first run's search band censored the predicted
periods (caught by registered-prediction verification against the source
paper). A second run voided itself at its own anchor gate after the anchor
was built from a mismatched construct (caught by the gate, diagnosed from
the records). A corrected rerun replicated the anchor to four decimal places
on the base model [1]. A scale rung's only positive was a trend artifact in a
single spectral bin (caught by an independent reader re-deriving the winning
shape from the committed record) [2]. A closing rung with that bin barred
returned the committed null [2]. One run in five produced its intended
headline on the first try [3]. Every failure became a named, committed
correction; nothing was re-scored after the fact; voided numbers stayed void.

## Where everything lives

Seed questions sit in the seeds folder. Committed designs, gate answers,
deviations, and registered-prediction files sit in the records folder.
Per-run results sit in the per-run output folders, as self-describing
records that carry their own inputs, rules, and calls, alongside every
shuffle draw's full spectrum, every per-count average vector, and every raw
position row. The sealed predictions ledger at the repository root carries
every prediction and its scored outcome. Each run also has its own reader
report at the repository root, and the budget ledger sits in the records
folder (about 3.60 dollars spent of a 100-dollar cap, across both base
models) [3].

## The formation chapter, added after two further rungs

The chain then asked its parent question of the counter itself: when did it
form, and did it move afterward? A first sweep attempt voided itself at its
own negative control and returned two discoveries [1]: a randomly initialized
network already decodes character position at quality 0.345 (untrained
attention carries positional structure, flat across layers and failing on
narrow lines), so the formation baseline is about 0.35, not zero; and the
seventy-character precision boundary is absent in the smaller model (ratio
minus 0.091 at its final checkpoint) [1] - the boundary is scale-emergent.

The corrected sweep ran twelve checkpoints of the larger base model [2], with
both gates passing exactly (anchor 0.876302 reproduced to the last digit;
initialization baseline 0.341, nearly identical to the smaller model's 0.345)
[2]. The trajectory answered the fossil-or-drift question with a shape neither
story predicted, in four phases [2]: a crude counter appears deep in the
network within 512 steps - 94 percent of final quality inside the first 0.4
percent of training, the entire rise hidden below the grid's first interval;
then a destructive reorganization between steps 2,000 and 8,000, where quality
regresses from 0.851 to 0.715, fine precision falls below its initialization
value (boundary ratio minus 0.248), and the representation re-expands to about
a hundred directions; then relocation, the best layer jumping from layer 22 to
layer 6 by step 16,000; then slow consolidation - the precision boundary forms
last, crossing its bar only at step 64,000, forty-five percent of training,
and holding within 0.014 thereafter, while the dominant dimensionality keeps
compressing from twelve directions to three through the very last checkpoint
[2].

The committed rules read this as not classifiable, correctly: the hold window
opened at the premature step-512 milestone, and no rule assumed formation
could regress. The recorded-only fields carry the resolution: measured from
its own crossing, the boundary is dead still [2]. So the fossil-or-equilibrium
answer is per-property. Usability is near-instant and never leaves. Location
and precision fossilize, but only after a mid-training teardown and rebuild -
the transient, build-then-demolish shape the formation literature predicts,
here measured on a named, twice-anchored structure. Dimensionality behaves
like an equilibrium still settling when training ends.

Honesty items from the final reading, all in the records [2]: the
boundary-stability label flips with the committed startpoint choice (both
startpoints recorded); formation is bounded above by step 512, not timed; the
narrowest wrap length never works at any checkpoint (quality minus 0.888 at
the end) while the pooled headline stays 0.876, a pooling effect no design
stated; per-position rows exist only for the widest length, so six of seven
widths are not re-derivable from raw output; and one record note misnames the
quality statistic - settled by re-deriving both statistics from raw rows,
with every recorded number reproducing under its true construct.

## Gate outputs, verbatim [3]

```
$ gate_density.py
CHAIN_TAKEAWAYS.md: 0.0 codes/1000 words (0 codes, 1529 words); threshold 10.0
(exit 0)

$ gate_scope.py
0 unscoped result sentence(s)
(exit 0)
```
