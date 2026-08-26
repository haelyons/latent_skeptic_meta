# Gate commit, seed C1 (experiment directory `manifold_c`)

Committed by the operating session, 2026-08-26, before any build and before any
rented box is launched. It answers, of record, every item the design pass
queued: five departure questions from the diff phase, one grounding verdict that
did not reproduce, one run-queue item, one author-queue item, and seven
instrument validations. "Accepted as-is" is an answer here and is used where the
design's own choice stands unchanged.

Binding source of the design: `out/seedC1_design_diff_triage.json`.
Budget: approved at 4 USD against a total cap of 15 USD.
Substrate: one Lambda Cloud box, `gpu_1x_a100_sxm4`, `us-west-2`, 1.99 USD per
hour, launched and torn down by a per-launch copy of the frozen launcher, with
teardown confirmed through the cloud API.

---

## A. Diff departure questions

### A1. "Should the number of periods searched at each window depend on the longest line length in training text, or be fixed some other way?"

Answer: it depends on the training label range, as the design says. ACCEPTED AS
IS in principle, with one arithmetic clamp that is required for the quantity to
exist at all, and one extra recorded number.

The design commits, per probe window, a profile length taken from the largest
label value that occurs at that wrap width in training documents. That is kept.
Two things are pinned so the stage is computable and auditable:

* Let `L` be the number of labels in that window's committed label window. The
  off-diagonal profile `m(delta)` is defined for `delta = 1 .. L - 1`. There is
  no `delta = L`: it would average zero label pairs. The committed profile
  length is therefore `min(committed_max_label_span, L - 1)`, and the clamp, if
  it bites, is written into the calibration record.
* For every `delta` the run records how many label pairs were averaged into
  `m(delta)`. The far tail of the profile rests on a handful of pairs, and that
  thinness must be visible in the artifact rather than inferred.

The periods searched are unchanged and are not narrowed: every whole-cycle
period `P = profile_length / k` for integer `k`, keeping `P >= 4` and
`P <= profile_length`. No search band exists.

Why the design's own rule stands rather than a tighter one: the thin tail
inflates the observed maximum power, but it inflates every shuffle draw's
maximum power in the same way, because the null draws are built from per-label
mean vectors over the same positions and the same label counts. The rank test
therefore stays honest; what a thin tail costs is sensitivity, not validity.
Narrowing the search to protect sensitivity would reintroduce exactly the band
censorship this question was written to remove.

### A2. "Should every chance-comparison shuffle start from the same fixed random point, or should several different starting points be checked?"

Answer: ACCEPTED AS IS. One fixed base seed, value 0, with a separate derived
stream per probe window and per labelling, recorded in the calibration record
before any held-out number exists.

Reason: several starting points would produce several rank tests with no
pre-registered rule for combining them, which is a larger threat to this design
than seed sensitivity is. The seed sensitivity that remains is already carried
by the headline rule: the decision is the invariance fork across three windows,
not any one window's pass or fail.

### A3. "Is this list of test strengths (1, 10, 100, 1000, 3000, 10000, 30000, 100000) fine enough to trust a pass or fail?"

Answer: ACCEPTED AS IS. The question fixes only that the grid must extend beyond
1000, and it does. Two recorded, never gated, additions: whether the winning
strength sits at an interior grid point or at an edge, recorded separately for
the main selection and for the anchor selection; and the full quality surface
over strength and layer, so a later reader can see how flat the ridge is rather
than take the argmax on trust.

### A4. "Is holding out 6 of the 36 training documents, five separate times, the right amount to set the pass-fail range?"

Answer: ACCEPTED AS IS, with the procedure pinned so that the dispersion
measures what the band has to absorb.

Each of the five resplits runs the WHOLE anchor procedure on its own split, not
just the final fit: it selects the anchor layer on its own sub-training
documents at width 140 alone, fits the ridge on those same sub-training
documents, and scores held-out fit quality on its own six held-out documents at
width 140 alone. The dispersion therefore includes the noise of the layer
selection, which is part of the anchor's definition.

The spread is the median absolute deviation of those five values about their own
median, unscaled (no 1.4826 factor). The half-width is three times that. A
measured spread of exactly zero halts the run as WRONG_VARIANCE_SOURCE; it is
never emitted as a zero-width band.

Acknowledged limitation, of record: five values is a coarse dispersion estimate,
and the six-document held-out sets are half the size of the twelve-document test
set the band will judge. That is a stated weakness of this run, and it is
carried into the answer of A6 and B1 below rather than fixed by changing the
question's rule.

### A5. "Is a $4 budget on one rented graphics computer, for about one hour, enough to safely finish all three probe windows?"

Answer: yes, and approved. The same instrument, on the same committed text and
the same checkpoint, measured about 37 seconds for the 432 forward passes and
about two minutes for the whole measurement in a prior run of this project. The
launcher's default hard cap is 5400 seconds with a self-destruct backstop past
it. At 1.99 USD per hour the expected charge is well under one hour of box time.
The total cap for this seed is 15 USD, so one full retry launch is affordable
without a further approval.

### A6. Departure raised by the operating session, not by the diff phase: which fit the anchor statistic uses

The question's definition 8 says "held-out fit at width 140 alone, judged at the
layer the calibration stage selects on train documents at width 140 alone". Two
readings exist. Reading one: the single probe of definition 2, fit on training
documents pooled over all nine wrap widths, scored only on width-140 test
documents. Reading two: a ridge fit on width-140 training documents alone, at a
layer selected on width-140 training documents alone.

Answer: reading two, the design's own reading, stands. It is adopted because
definition 8 states that the selection procedure is part of the anchor's
definition, which makes the anchor a self-contained width-140 procedure rather
than a restriction applied to the pooled probe.

Because the two readings give different numbers, and because the band's centre
came from neither of them, the run must RECORD, never gate, all of these at the
same time, so that a void can be told apart from a construct mismatch:

1. the anchor statistic itself (reading two), with its band and its decision;
2. the same held-out fit under reading one: the pooled nine-width probe at the
   main layer, scored on width-140 test documents alone;
3. held-out fit of the pooled nine-width probe at the main layer, pooled over
   the seven wrap widths 20 through 140 (the construct behind the committed
   value 0.781031 in this project's records);
4. held-out fit of the pooled nine-width probe at the main layer, per wrap
   width, all nine widths;
5. the main layer index and the anchor layer index.

None of these five may enter any decision rule. They exist so the record can be
read.

---

## B. Grounding verdict that did not reproduce

### B1. Verdict "diverges": the anchor band's centre belongs to a different construct from the anchor statistic

The reader's note: the design centres the width-140-only anchor band on 0.781,
while this project's committed measurements of width-140-only fits are 0.700601
and 0.549 (recorded in the seed D2 records) and 0.664024 (recorded in the seed
D2.1 records), and the value 0.781031 belongs to a fit pooled over the seven
widths 20 through 140. The reader predicts the run voids.

Answer, of record: the divergence is REAL, ACKNOWLEDGED, and NOT REPAIRED.

The band's centre is fixed by the question at 0.781 and the operating session
does not move it. Moving a pre-registered band because a prediction says it will
fail is the precise failure this pipeline exists to prevent. The half-width is
measured, not guessed, exactly as the question requires.

Three consequences are committed here instead:

* Every number listed in A6 is recorded, so that an out-of-band anchor can be
  read either as instrument drift or as a construct mismatch between the
  anchor statistic and the number its band is centred on. Neither reading is
  chosen in advance.
* If the anchor lands outside its band, the run's own rule stands: every ripple,
  control and represented-range result is void as an instrument fault, the void
  flag is written into every record produced after the gate, and the void is
  reported as the run's outcome. It is not evidence about the model, and the
  write-up must say so in those words.
* The reader's own prediction, that the run voids, is recorded here before the
  run. If the anchor instead lands inside its band, that prediction was wrong
  and the write-up says so.

A separate note on the reader's second observation, that no output artifacts
exist yet: correct, and expected. This gate is committed before the build, so
the only artifacts at this moment are the committed question, the committed
stimulus file, the design record and this file.

---

## C. Run queue

### C1. "Execute the pre-registered pipeline (layer sweep 0-24, ridge grid {1,10,...,100000}) and report main/anchor layer indices, whether ridge strength lands interior or at grid edge, and the anchor-gate R^2 vs its band."

Answer: this item IS this run, and it is approved to execute. It is not a
separate control. The four things it asks for are required outputs: the main
layer index, the anchor layer index, an interior-or-edge flag for each
selection's winning ridge strength, and the anchor statistic printed beside its
own measured band. All four go into the calibration and anchor records and into
the roll-up.

---

## D. Author queue

### D1. "Report the 5 anchor-resplit R^2 values, their MAD, the resulting band, and the observed width-140 test-set R^2 -- confirm the band width is not so narrow that ordinary train/test sampling variance alone would fail it."

Answer: ACCEPTED, and it becomes a required part of the calibration record
rather than a separately authored control, because everything it asks for is
produced by the calibration stage itself.

The calibration record must carry: the five resplit values in order, the five
document identities held out in each resplit, their median, their median
absolute deviation, the half-width (three times that), and the band's two
endpoints. The endpoints record must carry the observed anchor statistic beside
that band and the pass-or-fail decision.

One flag is added, RECORDED and NEVER GATED: `band_half_width_below_0.02`,
true when the measured half-width is under 0.02. A true value does not change
any decision. It is the marker that says the anchor gate, on this run, was
decided by a dispersion estimate too tight for ordinary sampling variation, and
it must be quoted in the write-up whenever the anchor fails.

---

## E. Instrument validations

Seven endpoint readouts were queued because no version of them carries a
committed validation artifact. One artifact answers all seven, and it is
required by the build brief in any case: the built runner's `--selftest`, which
is standard-library only, runs on the processor before any graphics-card time,
both locally and again on the box, and exits non-zero on any failure. Its
output is saved into the fetched results as the validation artifact.

For each queued instrument the planted ground truth is named here, of record.

### E1. Anchor gate readout
Planted rows with a known fit quality and a known band. Cases: a value inside
the band decides in-band; a value outside decides out-of-band AND marks every
record written afterwards `run_valid=false`, per-window files included; a
planted set of five identical resplit values, whose spread is exactly zero,
halts WRONG_VARIANCE_SOURCE rather than emitting a zero-width band.

### E2. Maximum ripple power readout
Two planted profiles with known answers. First, a synthetic profile carrying one
injected sine wave of a known whole-cycle period plus a known degree-2 trend:
after detrending, the statistic must return that injected period as the location
of maximum power. Second, a flat profile with no injected period, judged against
planted null draws whose maxima all exceed it: it must NOT clear. This is the
end-to-end validation of detrend, transform, period set and maximum.

### E3. Count of windows clearing their own null
Planted per-window decisions covering counts of 0, 1, 2 and 3, each mapping to
the stated outcome, including the case where 0 defers to the represented-range
check.

### E4. Largest wavelength gap in characters
Planted wavelength sets: one whose pairwise gaps are all at or under 10
characters must read INVARIANT; one with a gap over 10 must not; a set with
fewer than two clearing windows must return the undefined-gap value rather than
a number.

### E5. Largest wavelength-to-window ratio gap
Planted ratio sets: all within 0.1 reads PROPORTIONAL once INVARIANT is ruled
out; a wider spread with wavelength gaps also over 10 characters reads MIXED;
exactly one clearing window reads UNDETERMINED and never MIXED.

### E6. Represented-range readout
Planted bins with known within-bin fit qualities and known 50-draw null
distributions: a bin above its own 95th percentile counts as represented, one
below does not, and the bin spanning label 150 is recorded and excluded from the
decision. The eight bins are fixed at label values 1-35, 36-70, 71-105, 106-140,
141-175, 176-210, 211-245 and 246-280; the straddling bin is 141-175. Labels
above 280, which occur only on line-final newline tokens, fall in no bin and are
recorded as excluded. Within-bin fit quality uses the main probe's predictions
on aperture-280 test positions in that bin, with that bin's own label variance
as the denominator; each null draw permutes those same positions' labels among
themselves and recomputes the same quantity.

### E7. Token-position control readout
Planted pairs of wavelengths with a known characters-per-token rate: a converted
gap over 3 characters credits characters; a gap at or under 3 characters with
both labellings clearing credits token position; disagreement on clearing
credits characters regardless of the gap. The characters-per-token rate is
measured per probe window, on that window's own labelled test positions, and all
three rates are recorded. This resolves, of record, a choice a previous asking
of this question left to whatever the code happened to do.

---

## F. Ordering, pinned because two rules in the question depend on it

The stages run extract, then calibrate, then endpoints, and never in another
order.

The calibration stage reads training documents only. It fixes and writes, in one
committed record: the ridge strength and its interior-or-edge flag, the label
window per probe window, the profile length per probe window with its pair
counts, the shuffle seed streams, the anchor layer, the five resplit values,
their median absolute deviation, and the anchor band. That record is complete
and on disk before the endpoints stage reads its first test-document row.

The endpoints stage is the only stage that reads test documents. It selects the
main layer by held-out fit pooled over all nine wrap widths, computes the anchor
statistic and its gate, then the ripple statistic, the nulls, the token-position
control and the represented-range check, in that order.

A failed stage marks every stage that depends on it as skipped, records every
exit code, and exits non-zero.
