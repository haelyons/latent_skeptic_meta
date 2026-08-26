# Operating session's reading of record, seed C1

The pipeline's own write-up is `TAKEAWAYS_seedC1.md`, produced by the write stage
and passed through both language gates untouched. This file is separate on
purpose: it is the operating session's reading, and it corrects one thing the
write stage got wrong about the run's own records. It changes no measured number
and no decision.

## 1. The answer to the research question

The question asked: at probe windows of 140, 200 and 280 characters, where a
periodicity clears its shuffle null, is its wavelength INVARIANT in characters or
PROPORTIONAL to the window?

**The run is void, and the question is not answered.** The anchor gate failed, and
the question's own rule says an out-of-band anchor voids every ripple result as an
instrument fault, not as evidence about the model. That rule was written before the
run and is not being softened now.

For the record, the void numbers point at PROPORTIONAL, and they do so about as
hard as numbers can. All three windows cleared their null at rank 1 of 200. The
wavelengths were 70, 100 and 140 characters at windows 140, 200 and 280. Every one
is exactly half its own window. The wavelength-to-window ratio is 0.5 three times
over, so the ratio gap is exactly 0. None of this is admissible.

## 2. Why the anchor failed, and why "the instrument drifted" is the wrong reading

The anchor statistic came out at 0.859740 against a band of 0.767394 to 0.794606.
It failed high, not low. The write stage read that as the measuring setup having
drifted from the run it was calibrated against. The run's own records rule that
reading out.

This run reproduces the previous asking's instrument almost exactly, on code an
agent wrote from scratch without seeing that previous run:

| quantity | seed D2, committed 2026-08-21 | seed C1, this run |
|---|---|---|
| best readout, held-out, pooled over nine widths | 5 | 5 |
| held-out fit there, pooled over nine widths | 0.657823 | 0.657823 |
| held-out fit there, width 140 alone | 0.549167 | 0.549167 |
| ridge strength | 1000 | 1000 |
| held-out positions, all nine widths | 99692 | 99692 |
| range-check bins at or below label 140 | -24.343564, -35.165575, -18.826312, -3.827423 | -24.343593, -35.165610, -18.826348, -3.827435 |
| range-check bins above label 175 | -39.367425, -90.250193, -146.593113 | -39.367383, -90.250139, -146.593033 |

Nine numbers, two independent implementations, agreeing to between five and nine
significant figures. The residual difference is the size expected from storing
activations at reduced precision. An instrument that drifted would not do this.

What actually happened is a construct mismatch, and the run's own
recorded-never-gated readings show it directly. Four different fits all get called
"held-out fit at width 140" in this project's history, and they are not the same
quantity:

| what is fitted, and where the layer comes from | scored at width 140 | source |
|---|---|---|
| ridge on train pooled over nine widths, layer chosen on held-out pooled over nine widths | 0.549167 | seed D2, and this run |
| the same probe at seed D2's calibration-preferred layer | 0.700601 | seed D2 |
| ridge on train pooled over widths 20 to 140, layer 6 | 0.664024 | seed D2.1 |
| ridge on train at width 140 alone, layer chosen on train at width 140 alone | **0.859740** | this run, first measurement of this construct |

The band's centre, 0.781, is a fifth quantity again: a fit pooled over the widths
20 to 140 and scored pooled over those same widths (0.780903 in seed D1, 0.781031
in seed D2.1). This run measured its own version of that, 0.612773, but with the
nine-width probe, which is a sixth thing.

So the anchor gate compared a statistic that had never been measured before against
a band centred on a different statistic. The gate did its job; the centre was wrong
for the quantity it was gating.

The gate commit's answer A6 chose the reading that produced 0.859740, and the
pre-run grounding note (gate commit B1) predicted the run would void from below,
citing 0.700601, 0.549 and 0.664024. That prediction was wrong in direction, and it
was wrong for an instructive reason: a probe fitted and selected at one wrap width
alone is a *better* probe at that width than any probe that has to serve nine
widths at once. Specialising the fit raised the number; it did not lower it.

## 3. The dispersion rule worked, and its own narrow-band flag fired

The five resplits gave 0.867072, 0.876217, 0.866433, 0.861897 and 0.860696. Their
median is 0.866433 and their unscaled median absolute deviation is 0.004535, giving
a half-width of 0.013606. The measured spread was not zero, so
`WRONG_VARIANCE_SOURCE` correctly did not fire; the five held-out document sets
differ, and their fit qualities differ with them.

All five resplits independently chose the same readout (7) and the same ridge
strength (100). That is a stability result about the selection, not a zero-variance
error.

The run's own `band_half_width_below_0.02` flag came out true, which the gate
commit required to be quoted whenever the anchor fails. Quoting it: the band was
0.0136 wide either side, and the anchor statistic sits 17.4 median-absolute-
deviations above the band's centre, and only 1.5 of them from the median of the
resplits that measured the spread. Read plainly: the statistic is highly
reproducible and the centre is in the wrong place.

## 4. The finding that matters most, even though it is void

Every one of the five computable periodicity readouts, at three windows and under
two labellings, put its maximum power at exactly **two cycles across the profile**.
Not at the same wavelength. At the same cycle count.

- window 140, characters: 2 cycles, 70 characters, rank 1 of 200
- window 200, characters: 2 cycles, 100 characters, rank 1 of 200
- window 280, characters: 2 cycles, 140 characters, rank 1 of 200
- window 200, tokens: 2 cycles, 33.5 tokens, rank 1 of 200
- window 280, tokens: 2 cycles, 45.5 tokens, rank 1 of 200

The saved similarity profile says why. At window 140 it runs 0.1814 at distance 1,
falls steadily to -0.1103 at distance 100, then turns and rises to 0.1622 at
distance 140. That is one smooth arc, not a wave. A degree-2 polynomial cannot
absorb an arc of that shape, and what it leaves behind has its power at the lowest
few cycle counts: 6.03e-05 at one cycle, 8.84e-05 at two, 7.03e-05 at three. Two
cycles wins by a nose over its own neighbours.

The shuffle null cannot catch this. Shuffling labels within a document destroys all
structure, so a null profile is flat, and *any* smooth non-polynomial shape beats
all 200 draws. The test as written asks "is there structure here that a parabola
does not explain", not "is there a periodicity here". Those are different
questions, and a decaying similarity curve answers yes to the first while having no
wave in it at all.

Two consequences follow, and both are about the design rather than the model.

First, the invariance fork cannot separate its two horns with this statistic. A
fixed cycle count is proportional to the window by construction, so PROPORTIONAL is
what this readout returns whenever the profile is a smooth arc. The fork's
"artifact of the probe window" horn is exactly right here, and it is right about
the readout, not about the model.

Second, the token-position control cannot do its job either. Both labellings landed
on two cycles. The control converts the token period to characters and asks whether
the two differ by more than 3 characters. Two cycles in token units and two cycles
in character units convert to different character numbers whenever the profile
lengths differ, so the control credits characters (gaps of 15.28 and 17.74
characters) while both labellings are showing the same artifact. The control
measures a unit conversion, not a distinction.

## 5. Two premises of the question did not hold

The question's reasoning rests on seed D1's measured ambient dimension of 1, with
the first component carrying 95.3 percent. This run measured 93, 141 and 196
directions to reach 95 percent, at windows 140, 200 and 280, on centred per-label
mean vectors at readout 5. The first component carries 0.437, 0.305 and 0.200.
Whatever seed D1 measured, it is not the same object as this, and the "spacing
equals the window divided by one" argument that motivated the three-aperture design
does not survive contact with 93 to 196 directions.

The represented-range escape hatch also did not behave as its two-sided rule
imagined. Bins are represented, then not, then represented again as the label grows:
1-35, 36-70, 71-105 and 106-140 represented; 141-175 (excluded by rule) not;
176-210 not; 211-245 and 246-280 represented. The rule's decision,
ABOVE_150_REPRESENTED, is correct as written and closes the escape hatch. But a
pattern that switches off and back on is not the "signal fades with distance"
picture the hatch was built around.

## 6. What a next asking would have to change

This is a recommendation, not a result. The question's own rule says a failed
anchor blocks any next seed, so nothing below is licensed until the anchor is
re-specified.

1. The anchor's band centre must come from the same construct as its statistic.
   Either centre the band on a number measured with the anchor's own procedure, or
   define the anchor as the construct 0.781 came from. Mixing them is what voided
   this run, and it has now voided two askings in a row for two different reasons.
2. The periodicity statistic needs a null that keeps the smooth part and destroys
   only the periodic part. A within-document label shuffle destroys everything, so
   it cannot tell an arc from a wave. A phase-randomised or trend-preserving
   surrogate would put the two on the same footing.
3. A readout whose answer is a fixed cycle count cannot decide a fork between
   "fixed in characters" and "proportional to the window". Either the statistic must
   be reported in absolute units with the cycle count free, or the fork needs a
   different discriminator.
4. The probe pooled over all nine wrap widths is a poor probe: it scores -7.99 at
   width 20 and -1.18 at width 40. Every downstream number, the represented-range
   bins included, inherits that. Pooling only over widths that share a label range
   is the change seed D2.1 already made, and dropping it here cost accuracy.
