# Takeaways: does this model's line-position counter wobble?

(This is about the test recorded in the project as seed D2. An earlier, related attempt is recorded as seed D1.)

## The idea being tested

A 2025 paper, testing a different and much larger AI model, reported a hidden
signal inside that model's internal activity. The signal tracked how many
characters had gone by since the start of the current line of text. The paper
also saw a faint, repeating wobble in how that signal was organized. Its guess
was that a naturally richer pattern was being squeezed into too few internal
dimensions — a structure that needed about 150 dimensions forced into about 6.

An earlier test in this project looked for that same wobble, in a different
model, using one single window of text. It found nothing. But that test could
only ever have detected a wobble of one specific size — the window's own size
— so a plain "nothing found" there proved very little either way.

This experiment fixes that flaw. It repeats the search on the base checkpoint
of a mid-size open language model, Pythia-410m-deduped, this time using three
separate window widths: 140, 200, and 280 characters. If a wobble shows up and
keeps the same size, in characters, across all three widths, that points to
something real and fixed inside the model. If the wobble's size instead grows
in step with the window width, that points to a side effect of the
measurement itself, not a real fixed pattern. If no wobble beats chance at any
width, that would be a clean negative — unless a separate check shows the
wider windows could not have detected anything anyway, in which case the
negative would prove nothing.

## What was predicted, and what happened

Before trusting any wobble result, this base checkpoint first had to pass one
required check: how well its internal signal predicted character counts on
text it had not trained on, at the 140-character window. That check had to
score between 0.701 and 0.861 on this base checkpoint. That band came from an
earlier run's score of 0.781, plus or minus 0.08, using the same base
checkpoint lineage, on different text.

This base checkpoint measured only 0.549167 at that window, far below the
required floor. By the design's own rule, that shortfall makes the entire run
void, and every wobble number that follows is not evidence about the model.

## But three "clean" results got reported anyway

Even so, each of the three window-level result files still reports its own
wobble as beating chance, with no void flag attached to any of them. Only one
file, the overall run summary, records the whole run as void. A reader who
opened just the three window files would see three confident-looking positive
results, and no warning at all.

## Those three results do not agree with each other

Setting the failed check aside for a moment: the three window widths did each
report a wobble size, but the three sizes do not agree with each other.

- At the 140-character window, on this base checkpoint, the wobble repeated
  every 40 characters.
- At the 200-character window, on this base checkpoint, the wobble repeated
  only about every 16.3636 characters.
- At the 280-character window, on this base checkpoint, the wobble repeated
  every 130 characters.

The gap between the smallest and largest of these three sizes comes to
113.6364 characters, on this base checkpoint. The design's own limit for
calling a size fixed was only 10 characters, so that rules out a single fixed
size.

A second check compares each wobble size to its own window width, as a ratio.
Those ratios were 0.285714 at the 140-window, 0.081818 at the 200-window, and
0.464286 at the 280-window, on this base checkpoint. The largest gap between
any two of those ratios is 0.382468 on this base checkpoint, well past the
design's own limit of 0.1, so that also rules out a size that simply scales
with the window.

Neither the plain sizes nor the ratios line up. The study's own three-way
outcome list — a fixed size, a scaling size, or no pattern at all — has no box
for three different, non-scaling sizes. The three results read as three
separate, unrelated blips rather than one genuine repeating pattern.

## A barely-passing result hiding inside "three for three"

The three results also differ in how convincingly each beat its own chance
baseline of 200 shuffled trials. At the 140-window, on this base checkpoint,
the result beat all but a share of 0.00498 of the chance trials, ranking 1st
of 200. At the 280-window, on this base checkpoint, the result ranked 3rd of
200, with a share of 0.01493. At the 200-window, on this base checkpoint, the
result ranked 10th of 200, with a share of 0.04975.

On this base checkpoint, that 200-window result sits right at the study's own
0.05 cutoff for counting as a beat-chance result at all. So the headline
claim, that all three widths clear chance, rests on one result that barely
clears it and two that clear it by a wide margin.

## The best-fitting window is the least trustworthy one

At the 140-window, the wobble cannot be told apart from a simpler
explanation: the model may just be counting sub-word chunks (tokens), not raw
characters. At the 200- and 280-windows, the wobble is clearly distinct from
that simpler explanation.

This pairing is awkward. This base checkpoint's fit at the 140-window,
0.549167, sits closest to trustworthy of the three, yet it is the one window
confounded with token counting. So the best-fitting window is also the
weakest evidence that this is really about characters rather than tokens.

The reason traces to a matching margin: the two kinds of wobble (character-
based and token-based) must differ by more than 3 characters to count as
genuinely distinct. At the 140-window, on this base checkpoint, the
token-based wobble converts to about 42.8378 characters, only 2.8378
characters away from the character-based wobble's 40. That gap sits under the
3-character margin, which is why the 140-window counts as confounded. At the
200- and 280-windows, on this base checkpoint, that same gap grows to roughly
57.96 and 103.43 characters, comfortably over the margin.

## Exploratory: the size of the failure depends on which layer-picking rule is trusted

This next point is exploratory, not a pre-registered test. Which internal
layer got used depends on which of two disagreeing selection steps is
trusted. An early calibration step, using only training documents, preferred
a different layer than the one this study's own held-out rule picked and
used.

On the training-only measure, the calibration step's preferred layer scored
0.672781 on this base checkpoint, versus 0.657823 for the layer the study
actually used. But the study's own rule for picking the layer uses held-out,
unseen documents instead, which is why it used the lower-scoring layer.

The difference matters a lot at the 140-window: on this base checkpoint, the
layer actually used scored 0.549167 there, while the calibration step's
preferred layer would have scored 0.700601. That is a shortfall of just
0.000399 against the required 0.701 floor, on this same base checkpoint. So
the size of the failure — a wide miss or a near miss — depends on which of two
disagreeing procedures chose the layer, not only on what the model's activity
shows.

## The planned fallback check also came up empty

The design included one pre-registered fallback: a check for whether wide
character counts, above 150, could be measured at all on this base
checkpoint. If they could not, a "nothing found" result at the wider windows
would mean little. That fallback also failed to produce a usable answer.

In the four chunks covering counts up to 140, fit quality on this base
checkpoint came out at roughly -24.34, -35.17, -18.83, and -3.83. The design
required those four scores to sit above 0.5, so all four fell far short, on
this same base checkpoint.

In the three chunks covering counts above 175, fit quality on this base
checkpoint came out at roughly -39.37, -90.25, and -146.59. Those three were
required to sit below 0.3, on this base checkpoint, and they did. But that
pass happened only because the fit was poor everywhere, not because wide
counts were specifically shown to be unmeasurable.

So this planned fallback could not tell a genuinely blind window apart from a
real absence of a pattern. The one pre-registered escape hatch this design
relied on did not work.

## Two process notes, not results about the model

First, one rule was decided only after the study began, not written down
beforehand. That rule fixes each window's usable pattern length at the
window's own width minus 20 characters. For the three widths tested, on this
base checkpoint, that works out to 120, 180, and 260 characters of usable
pattern. That specific rule shows up only in free-form notes from the setup
step, not in the plan that was fixed before the run started.

Second, an already-flagged open question about how to measure the
character-to-token conversion rate was never explicitly settled beforehand.
The code answered it by its own default: measuring a separate rate for each
window, rather than one rate for the whole study. On this base checkpoint,
those three separate rates came out to 4.0798, 4.1294, and 4.1685 characters
per token, one per window. A choice the design's own review had explicitly
flagged as open got resolved silently, by whatever the code happened to do.

## Bottom line

The one check meant to decide whether any of this was trustworthy failed: on
this base checkpoint, 0.549167 against a required 0.701 to 0.861. By the
study's own rule, that failure voids every wobble result below it.

Even setting that failure aside, the three window-by-window results do not
agree with each other, one of the three "beats chance" results barely clears
its own bar, the single window closest to a trustworthy fit is also the one
that cannot be told apart from simple token counting, and the one
pre-registered fallback check meant to rescue a negative result did not work
either. This run does not provide credible evidence, either way, about
whether this base checkpoint's character-tracking signal really wobbles.

---

## Gate checks

Both commands below were run against this same base checkpoint's takeaways
file, and both are pasted in full, verbatim.

```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_density.py TAKEAWAYS_seedD2.md
TAKEAWAYS_seedD2.md: 0.0 codes/1000 words (0 codes, 1777 words); threshold 10.0
```

The second command checks the same base checkpoint's takeaways file for
unscoped numbers, and its output is pasted below, verbatim.

```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_scope.py TAKEAWAYS_seedD2.md
0 unscoped result sentence(s)
```
