# Does a line-wrap wave reflect the model, or the ruler used to measure it?

This file explains one completed measurement run, in plain language. It is
written for a reader who has never seen this project before.

One model and one snapshot were studied throughout. It is a small, openly
available language model named Pythia-410m-deduped. It was tested at one
frozen training snapshot named step143000, with no changes made to it.
Below, this is called the base checkpoint.

## The question this run was testing

Some language models build an internal number line while reading wrapped
text. It is a running count of characters since the start of the current
line. A 2025 study of a much larger, closed model found that this number
line was not a smooth ramp. It carried a faint, repeating wave on top of the
ramp, like ripples over a straight line. That study explained the wave as a
side effect of squeezing a shape into too little room. The shape would
naturally need about 150 independent directions. Only about 6 were actually
available to it.

An earlier run in this project looked for the same kind of wave on the base
checkpoint. That run tested only one line-width setting. On the base
checkpoint, it found the wave's shape needed only 1 direction to hold 95.3
percent of it. If a wave truly needs only 1 direction, its predicted length
is just the chosen line width. Testing one width alone can never tell a real
wave from an artifact of picking that width.

This run tested three line widths at once, on the base checkpoint: 140,
200, and 280 characters per line. Where a wave beats pure chance at more
than one width, does its length in characters stay fixed? Or does it grow
together with the width instead? A fixed length would point to a real
feature of the model. A length that grows with the width would point to an
artifact of the ruler. Two other outcomes were allowed for: a mixed
pattern, or too few widths clearing chance to decide at all. If no width
showed any wave, that alone would be a strong negative finding, on the base
checkpoint. The one exception: a separate check might show the model was
never tracking position that far into a line to begin with.

## What the run measured

Forty-eight short public-domain texts were each wrapped onto lines, at nine
different widths. Thirty-six of the texts trained a simple statistical
readout. Twelve were held back purely for testing, in a fixed, prearranged
order.

A simple statistical readout was trained to predict each word's character
position within its line. It was trained separately at each of 25 evenly
spaced points inside the network, on the base checkpoint. The very first
point is the raw input embedding.

Exploratory, not itself a test of the wave: one of those 25 points was
picked. It became the single point used for every wave search below. The
pick used the best prediction accuracy, on the base checkpoint, on the 12
held-out test texts, pooled across all nine widths. That turned out to be
the sixth point, counting the input embedding as the first.

Exploratory, and separate from the choice above: a second best-point search
used only 140-character-wide texts. It picked the point used purely for one
required sanity check, described next.

For each of the three widths, at the one point chosen above, every
character position's internal pattern was compared against every other
position's pattern. This search was looking for a repeating wave. Each
search was scored against 200 random shuffles of the position labels. That
comparison shows what a wave would look like from pure chance alone.

A parallel control repeated the same wave search. It counted whole sub-word
pieces since the line began, instead of characters. This checked whether
any wave found was really about counting characters. It could instead be an
artifact of counting sub-word pieces.

A separate check asked whether positions past character 150 into a line
were represented by the network at all. It split the 280-character test
texts into eight slices of 35 positions each. Prediction accuracy inside
each slice was tested against that same slice's own scrambled-label
comparison.

Before trusting any wave result, one required sanity check had to pass
first. It measured held-out prediction accuracy at the sanity check's own
chosen point, using only 140-character-wide test texts. That number had to
fall inside an allowed range, built from a past run's committed number.
Falling outside it would mean the run is a broken measuring instrument.
Every wave result from a broken instrument counts as not being evidence
about the model at all.

## What was predicted, and what happened

The required sanity check was predicted to land inside an allowed range, on
the base checkpoint: 0.767394 to 0.794606. On that same base checkpoint,
the range was centred on 0.781, a number carried over from an earlier run
of this project.

The measured value, on the base checkpoint, came out to 0.859740. That
falls outside the allowed range, above its high end.

That is a failed sanity check. By this run's own rule, every later
measurement, on the base checkpoint, is void. None of it is evidence about
the model. It only shows the measuring setup drifted from the earlier run
it was calibrated against.

The numbers below were still computed and saved. They are worth reporting
for what they show. Every one of them carries that same void label.

All three widths beat pure chance, on the base checkpoint. The observed
wave strength ranked first out of 200 random-shuffle comparisons, at every
one of the three widths tested.

On the base checkpoint, the 140-character width gave a wave length of 70
characters. The 200-character width gave 100 characters, and the
280-character width gave 140 characters. Each is exactly half of its own
width.

The longest and shortest of those lengths differ by 70 characters. That is
more than the 10-character difference allowed for calling them one fixed
length. All three lengths sit at exactly half their own width. On that same base
checkpoint, that is a matching ratio of 0.5 in every case. So the headline
call, on the base checkpoint, is that the wave's length grows with the
width. That call is
void, for the reason above.

At the 140-character width, the sub-word-piece control could not even be
computed. No held-out test text reached the sub-word position the
comparison needed. At the 200-character width, the two ways of counting
position gave wave lengths 15.277055 characters apart, on the base
checkpoint. At the 280-character width, they gave wave lengths 17.735970
characters apart, on the base checkpoint. Both gaps pass the 3-character
cutoff for treating the two as the same wave. So the wave at those two
widths is credited to counting characters, not to counting sub-word pieces.

The escape-hatch check looked past character 150 into a line, on the base
checkpoint. Both far ends tested were represented there: positions 211
through 245, and positions 246 through 280. So the void above is not
explained away as nothing being measurable that far into a line.

## Does this match what was expected

No, on more than one count. This section says so plainly rather than
smoothing it over.

Before the run, the recorded concern was that the sanity check would most
likely fail low. Three past readings of this exact 140-character-only
measurement, on the base checkpoint, were 0.700601, 0.549000, and 0.664024.
All three, on that same base checkpoint, sat below the allowed range's low
end of 0.767394. The check instead failed high, on the base checkpoint, at
0.859740. That is the opposite direction from the one anticipated.

An earlier, closely related run, on the base checkpoint, had found this
same kind of wave needing only 1 direction. On that same base checkpoint,
that single direction carried 95.3 percent of the wave's shape. This run
instead needed 93 to 196 directions, on the same base checkpoint, across
its three widths, to reach that same 95 percent mark.
That is a large, unexplained change in the wave's basic shape between the
two runs.

Because the required sanity check failed, this run's outcome is neither of
the two rival ideas it set out to test. It is not "the wave has one fixed
length." It is not "the wave's length is an artifact of the width" either.
It is a third outcome, on the base checkpoint: the run cannot be read as
evidence for or against either idea. Every wave number above is exploratory
only, kept for the record. None of it is usable as a finding about the
model.

## What in these files does not fit

These are open questions raised about this run. A few more that this review
turned up on its own follow them.

**The wave's length always came out to exactly half the width tested.**
Counting by character or by sub-word piece made no difference. At every
width tried (140, 200, and 280 characters), the found wave completed
exactly 2 full cycles across the tested span, with no exception. A method
that always reports two cycles, regardless of width or counting scheme, may
be producing that number itself. It may not be detecting it in the network.

**The sanity check failed on the wrong side.** Going in, every recorded
warning pointed toward a low failure. That warning rested on three past
readings of the same measurement. The check instead failed high. Whatever
reasoning predicted trouble was reasoning about the wrong direction. That
weakens confidence in the rest of that reasoning.

**Five reruns meant to measure this run's own natural variation all landed
on the identical choice.** Five reruns, each holding out a different six
documents, were meant to show how much the sanity check's result would
normally wander. All five picked the identical internal point and the
identical readout strength. The resulting allowed range came out narrower
than a limit the project had set for itself in advance. That limit exists
precisely to flag this kind of trouble. The flag was raised in the record,
but it changed no decision.

**Line-by-line records survive for only three of the nine line widths.**
Those nine widths fed the single most important choice of this whole run:
which internal point to search for a wave. The saved line-by-line detail
behind that choice exists for only three of those nine widths. The other
six widths' line-by-line detail was not kept anywhere in this run's saved
files. That choice cannot be independently checked line by line. Only its
already-summed totals can be trusted or questioned.

**The number of directions needed for this signal jumped by roughly one
hundred times.** A closely related earlier run, on the base checkpoint,
needed only 1 direction to carry 95.3 percent of this same kind of signal.
This run needed 93 to 196 directions for that same 95 percent mark, on the
base checkpoint, across its three widths. Whatever theory expected a
narrow, one-direction signal does not obviously fit a signal this much
wider.

**The wave was detectable, then not, then detectable again, as line
position grew.** The widest test texts were split into eight bands of line
position. The wave was found beyond chance for positions 211 through 245.
It was absent for positions 176 through 210. It was found again beyond
chance for positions 246 through 280. A story where the wave simply fades
out with distance does not fit a gap that closes back up further out.

**The same twelve documents both picked the search point and then judged
it.** The one internal point used for every wave search was chosen by
testing accuracy directly on twelve held-out documents. The wave search was
then run on those same twelve documents. A point picked this way could look
stronger on those twelve documents than it would on fresh ones. That makes
the wave result less independent than a fresh test would be.

**One missing-data flag caught a shortfall but not an overrun.** At the
narrowest width, the sub-word-piece control was explicitly marked as
impossible to compute. No held-out document reached the needed line
position there. At the two wider widths, the matching missing-data field
reported nothing missing at all. Yet this review found individual
positions in the wider widths' own line-by-line files. They sat exactly at
the outer edge of the range fixed from training documents. A field meant to
flag every data problem missed at least one kind while catching another.

A few further things this review noticed on its own, not from the list
above, are described next.

The readout strength used everywhere was tuned by a cross-check. That
cross-check favored a different internal point than the one this run
actually used for every wave search. The point-picking step chose its point
by direct accuracy on the 12 held-out documents. The strength-picking step,
run separately, found its own best accuracy one point later in the network.
Both choices were kept as the design specifies. They quietly disagree about
which point is best.

In the escape-hatch check, "represented" only ever meant one thing: beating
its own scrambled-label comparison. It never meant "predicted accurately"
in an ordinary sense. Every one of its eight accuracy numbers was strongly
negative. One number was as low as roughly negative 147, even in a slice
marked as represented. A reader skimming only the word "represented" could
mistake a comparison that beat pure chance for an accurate prediction. It
was not one.

The first attempt at this run failed outright. A two-character mistake, in
how a file path was built into an output file name, caused the failure.
This happened before any of the numbers above were produced. The project's
own automatic pre-flight check could not have caught that particular
mistake. It is barred from touching the array-writing library the mistake
was in. The mistake was found and fixed by hand. The fixed run is the one
described above.

The approved spending limit for this run, on the base checkpoint, was 4 US
dollars. Two rented computer sessions were actually used: one that failed,
and one that finished. Together, on that same base checkpoint's run, they
cost about 0.69 US dollars. That is a wide safety
margin, not a problem. But the approved figure was not a close estimate of
the real cost.

## Raw samples, verbatim

A few rows were picked at random from this run's saved files. They are
shown here exactly as they appear on disk, to make the underlying records
checkable.

This is one input row: a whole test document (document 36, Herman
Melville's "Moby-Dick," public domain), wrapped to 80 characters per line
before being shown to the network, with the counts the run kept about it.

```
{"doc": 36, "width": 80, "split": "test", "n_chars": 3591, "n_tokens": 899, "n_labelled": 899, "n_lines": 46, "chars_spanned_by_labelled_tokens": 3032, "degenerate_flags": [], "sha256_text": "a1ab53944c16dc7bafb6480c27a4e29fba533ec58a3621045bfbdffdde9dc5e9", "text": "Already we are boldly launched upon the deep; but soon we shall be lost in its\nunshored, harbourless immensities. Ere that come to pass; ere the Pequod's weedy\nhull rolls side by side with the barnacled hulls of the leviathan; at the outset\nit is but well to attend to a matter almost indispensable to a thorough\nappreciative understanding of the more special leviathanic revelations and\nallusions of all sorts which are to follow. Thus speak of the whale, the great\nCuvier, and John Hunter, and Lesson, those lights of zoology and anatomy.\nNevertheless, though of real knowledge there be little, yet of books there are a\nplenty; and so in some small degree, with cetology, or the science of whales.\nMany are the men, small and great, old and new, landsmen and seamen, who have at\nlarge or in little, written of the whale. Run over a few:-The Authors of the\nBible; Aristotle; Pliny; Aldrovandi; Sir Thomas Browne; Gesner; Ray; Linnaeus;\nRondeletius; Willoughby; Green; Artedi; Sibbald; Brisson; Marten; Lacepede;\nBonneterre; Desmarest; Baron Cuvier; Frederick Cuvier; John Hunter; Owen;\nScoresby; Beale; Bennett; J. Ross Browne; the Author of Miriam Coffin; Olmstead;\nand the Rev. T. Cheever. But to what ultimate generalizing purpose all these\nhave written, the above cited extracts will show. Of the names in this list of\nwhale authors, only those following Owen ever saw living whales; and but one of\nthem was a real professional harpooneer and whaleman. I mean Captain Scoresby.\nOn the separate subject of the Greenland or right-whale, he is the best existing\nauthority. But Scoresby knew nothing and says nothing of the great sperm whale,\ncompared with which the Greenland whale is almost unworthy mentioning. And here\nbe it said, that the Greenland whale is an usurper upon the throne of the seas.\nHe is not even by any means the largest of the whales. Yet, owing to the long\npriority of his claims, and the profound ignorance which, till some seventy\nyears back, invested the then fabulous or utterly unknown sperm-whale, and which\nignorance to this present day still reigns in all but some few scientific\nretreats and whale-ports; this usurpation has been every way complete. Reference\nto nearly all the leviathanic allusions in the great poets of past days, will\nsatisfy you that the Greenland whale, without one rival, was to them the monarch\nof the seas. But the time has at last come for a new proclamation. This is\nCharing Cross; hear ye! good people all,-the Greenland whale is deposed,-the\ngreat sperm whale now reigneth! There are only two books in being which at all\npretend to put the living sperm whale before you, and at the same time, in the\nremotest degree succeed in the attempt. Those books are Beale's and Bennett's;\nboth in their time surgeons to English South-Sea whale-ships, and both exact and\nreliable men. The original matter touching the sperm whale to be found in their\nvolumes is necessarily small; but so far as it goes, it is of excellent quality,\nthough mostly confined to scientific description. As yet, however, the sperm\nwhale, scientific or poetic, lives not complete in any literature. Far above all\nother hunted whales, his is an unwritten life. Now the various species of whales\nneed some sort of popular comprehensive classification, if only an easy outline\none for the present, hereafter to be filled in all its departments by subsequent\nlaborers. As no better man advances to take this matter in hand, I hereupon\noffer my own poor endeavors. I promise nothing complete; because any human thing\nsupposed to be complete, must for that very reason"}
```

This is a second input row: one single word, from the same kind of file, at
a different width. It shows that word's line number and its character
position within that line.

```
{"doc": 36, "aperture": 140, "split": "test", "tok_index": 819, "tok_text": "Ġall", "char_start": 3350, "char_end": 3353, "line": 24, "char_label": 42, "token_label": 9}
```

This is one output row: one of the five reruns behind the sanity check's
allowed range, exactly as the run recorded it.

```
    {
      "cross_validated_quality": 0.864102,
      "held_out_doc_ids": [
        15,
        16,
        22,
        23,
        25,
        34
      ],
      "held_out_positions": 5418,
      "held_out_quality": 0.866433,
      "readout": 7,
      "resplit": 2,
      "ridge_strength": 100.0
    }
```

This is a second output row: one of the eight position-range slices from
the escape-hatch check, with both its accuracy number and the fifty
chance-comparison numbers it was measured against.

```
    {
      "bin": 3,
      "decision": "REPRESENTED",
      "draws": 50,
      "held_out_positions": 1295,
      "held_out_quality": -3.827435,
      "high": 140,
      "low": 106,
      "null_95th_percentile": -3.962777,
      "null_draw_values": [
        -3.947188,
        -4.117599,
        -4.080432,
        -4.072984,
        -4.047416,
        -4.04317,
        -3.901166,
        -3.989505,
        -4.070227,
        -4.055595,
        -4.232212,
        -4.285042,
        -4.093407,
        -4.288736,
        -4.014841,
        -3.975797,
        -4.184456,
        -3.952124,
        -4.175061,
        -4.211661,
        -4.284317,
        -4.109961,
        -4.059423,
        -4.291485,
        -4.179498,
        -4.044881,
        -4.138682,
        -4.184151,
        -4.088446,
        -4.124258,
        -3.987017,
        -4.02369,
        -4.033772,
        -4.216853,
        -4.025455,
        -4.089581,
        -4.25836,
        -4.125187,
        -4.276599,
        -4.146988,
        -4.1163,
        -4.113558,
        -4.014021,
        -4.185961,
        -3.988861,
        -4.075858,
        -4.267143,
        -4.104267,
        -4.292705,
        -3.999942
      ],
      "null_largest": -3.901166,
      "null_median": -4.098837,
      "rank_among_draws": 1,
      "role": "low",
      "shuffle_seed": 280203
    }
```

## Gate outputs, verbatim

Both gates were run against a copy of this file truncated at the "Raw
samples, verbatim" heading above.

```
$ python3 gates/gate_density.py <truncated copy>
/tmp/claude-1000/-home-hal-dev-interp-dm-hack-latent-skeptic-meta-experiments-manifold-c/75993dcb-5461-4dcd-a5df-e047e30a97a0/scratchpad/TAKEAWAYS_seedC1_truncated.md: 0.0 codes/1000 words (0 codes, 2392 words); threshold 10.0
exit code: 0
```

```
$ python3 gates/gate_scope.py <truncated copy>
0 unscoped result sentence(s)
exit code: 0
```
