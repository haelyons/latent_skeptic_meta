# Does a base model's counting sense form early and stay fixed?

Every number below comes from one base language model: EleutherAI's
pythia-1.4b-deduped, a pretrained model with 1.4 billion weights, never
fine-tuned or instruction-tuned. Researchers saved this base model at twelve
points during its training. The points ran from before training started to
its final trained state. At each point, a simple readout used the base
model's internal activity to guess a character's position in a line of
text.

The prediction: this base model's sense of position forms early in
training. A sharp precision break inside it was predicted to form early
too. Both were predicted to then hold fixed, like a fossil left over from
early training pressure. The rival prediction: both keep drifting through
later training instead. This is what the study tested. Here is what
actually happened, including where the story breaks down.

## The reproduction check passed

Before trusting anything else, the study ran one check first. This base
model's fully trained state had already been measured twice before, in
earlier work. The check re-measured its combined position-guessing score
and its precision-break size. The fully trained base model scored 0.876302,
matching the required range of 0.856302 to 0.896302. This base model's
precision-break size scored 0.514227, clearing the required minimum of 0.3.
Both passed, so the sweep across this base model's twelve saved points was
allowed to continue.

## The untrained floor passed, but see the metric problem below

The untrained version of this base model, before any training steps,
scored 0.341495 on the same position-guessing test. The study would have
voided itself if this base model's floor score reached 0.613411 or higher.
That would have meant the test was mostly reading untrained structure, not
trained counting. It did not, so the floor passed. But hold this number
loosely. A problem described further down shows this base model's untrained
score may understate real signal already present before training.

## Two milestones landed on the very same checkpoint

The study set two milestones for this base model's climb toward its final
score. The climb runs from this base model's untrained floor up to its
final score. The first milestone is a halfway point, 0.608899, marking when this base
model's position-sensing becomes usable. The second is a nine-tenths point,
0.822821, marking when this base model's position-sensing counts as
essentially formed.

Both milestones were first cleared at the exact same checkpoint for this
base model. That checkpoint came after 512 training steps. At that point,
this base model's score was already 0.827079. The very next checkpoint
after the untrained state already cleared both milestones. So the entire
climb happened somewhere inside the gap between zero and 512 steps. That
gap was never measured by the study.

The two-milestone system was meant to tell a slow climb apart from a sudden
jump. Here it cannot do that job. Both milestones fall on one grid point
only because of where checkpoints were placed. It is not because the data
actually showed a slow climb losing to a fast one.

## The steadiness check failed, and openly could not be classified

After this base model's score was judged formed, the study tracked its
steadiness. It tracked whether the precision-break size stayed steady
across later checkpoints. It also tracked whether the combined score stayed
steady across later checkpoints. Neither did, by the rules actually
applied.

This base model's precision-break size ranged from -0.247611 up to
0.526207. That spread, 0.773818, is far outside this base model's allowed
spread of 0.15. This base model's combined score ranged from 0.715426 up to
0.876302. That spread, 0.160876, is also outside this base model's allowed
spread of 0.10.

Movement was not one-directional in either case. So the honest, pre-written
verdict was that this base model's trajectory could not be classified. It
could not be called either steady or drifting. That refuting result is the
real headline of this experiment, not a clean story in either direction.

## One hidden choice would flip the precision-break verdict

The steadiness check above started its clock at one shared milestone. That
milestone came after 512 training steps. It is the same checkpoint used for
the score-formation call.

But the study's own written rule said something different. It said the
precision-break's steadiness clock should start later. It should start at
this base model's own crossing of the 0.3 threshold. That crossing happened
at the checkpoint taken after 64,000 training steps.

Measured from that later starting point instead, the picture changes. This
base model's precision-break spread across later checkpoints shrinks to
just 0.014179. That is comfortably inside this base model's allowed spread
of 0.15. So this one specific check would read as steady instead of
unclassifiable. That is true only if it started counting where its own rule
said to start. The underlying measurements did not change; only the
starting point did.

## The bigger problem: the score is not the correlation it claims to be

Every record in this study names its combined score the same way. It calls
the score "the Pearson correlation between predicted and true count." A
Pearson correlation is a plain measure of whether two sets of numbers rise
and fall together.

Recomputing that plain measure directly from this base model's saved raw
predictions tells a different story. At the widest line length, before any
training, this base model's true correlation is 0.2800. That is a real,
positive signal. But the number this base model's untrained state actually
recorded for that same measurement is -0.007607, not 0.2800.

A different formula reproduces that recorded number exactly. This formula
scores how close predictions land to the true numbers. It does not score
whether predictions and truth move together. It reproduces -0.007607
exactly, and does so at all twelve of this base model's checkpoints.

The score deciding every threshold in this study is that closeness-to-truth
formula wearing a correlation's name. This matters because that formula
punishes badly-scaled guesses far more harshly than true correlation would.
This base model's real untrained signal at the widest line length was
0.2800, not close to zero. Some of the mid-training wobble may not be real
instability at all. It may instead reflect this base model's guesses being
miscalibrated, not lost information.

## Exploratory: a dip, a sign flip, and a layer jump together

Between the checkpoints taken after 2,000 and 16,000 training steps, three
things happen to this base model at once. This base model's combined score
falls to its lowest point since formation, 0.715426, at 4,000 steps.

This base model's precision-break size turns negative in the same window.
This base model's break reaches as low as -0.247611 at 8,000 steps. That is
the opposite sign from when the break is genuinely present.

And one more thing shifts in this base model at the same time. The internal
layer that best supports position-sensing jumps to a much earlier layer. It
starts at very late layers, 22 or 23 out of 24. By 16,000 steps it lands on
layer 6 instead.

Whether this is one disturbance touching three signals at once, or three
unrelated coincidences, is not settled by anything recorded here. This
observation stays exploratory: it is not named or explained anywhere in the
study's design.

## The shortest line length fails quietly, every time

At all twelve checkpoints of this base model, one line length always
fails. The shortest wrapped line scores a strongly negative result every
time, even at the final checkpoint. It reached -0.887613 even at the end of
training on this base model.

Yet this base model's combined score spanning all seven line lengths
together stayed strongly positive throughout, reaching 0.876302 at the end.
A plain average of this base model's seven final per-length scores comes to
about 0.565, far below that reported combined figure. This means the
combined score must be built by merging raw predictions across line
lengths before scoring. It is not built by averaging each length's own
score. That method hides a line length that never works, at every point in
this base model's training.

## Six of seven line lengths cannot be independently checked

Raw per-guess predictions were saved to disk for only the widest line
length tested on this base model. The other six line lengths feed the
combined score used for every threshold in this study. None of those six
have saved raw numbers anywhere in the results. So most of what decides
pass or fail for this base model cannot be checked against raw output at
all. Only the one saved line length can be checked this way. No single
check in this study rests on that line length alone.

## Exploratory: a shrinking bottleneck, never explained

The number of directions needed to capture 90 percent of this base model's
internal variation shrinks sharply during training. It falls from 77 at the
untrained state to as low as 1 by later checkpoints. That is roughly a
fifty-fold shrink.

The number needed for 99 percent barely moves for this base model, staying
between 80 and 130 throughout training. This pattern is recorded at every
one of this base model's checkpoints. It is never interpreted or connected
to the position-sensing or precision-break story anywhere in the design.

It remains an exploratory observation, not a tested claim. It is exactly
the kind of unexplained structural change the original hypothesis was
meant to account for.

## Bottom line

This base model passed its integrity checks. Its position-sensing looked
already present the moment measurement began. It cleared both formation
milestones at the very first checkpoint after the untrained state.

But nearly every threshold call behind that story leans on something other
than clean data. The combined score is not the correlation it claims to
be. The measurement grid is too coarse to see the real formation moment.
The steadiness check's verdict depends on which checkpoint you start
counting from.

The pre-written verdict stands as the honest result of this experiment.
This base model's trajectory could not be classified as steady or
drifting. The reasons behind that verdict say as much about how this study
measured things as about what this base model actually did.

## Gate checks

These checks were run against this base model's takeaways file above:

```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_density.py TAKEAWAYS_seedD4_1.md
TAKEAWAYS_seedD4_1.md: 0.0 codes/1000 words (0 codes, 1722 words); threshold 10.0

$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_scope.py TAKEAWAYS_seedD4_1.md
0 unscoped result sentence(s)
```
