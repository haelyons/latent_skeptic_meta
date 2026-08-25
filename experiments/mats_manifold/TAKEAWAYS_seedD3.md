# Does This Model Count Characters in a Repeating Way?

## What this test was

This test used the base version of a language model called Pythia, sized at 1.4 billion parameters.

No fine-tuning, no added instructions, and no comparison to other models are included here.

The test checked whether the model, while reading plain text, keeps an internal track of how many characters it has read so far.

It then asked whether that internal count shows up as a repeating up-and-down wave, rather than as one smooth trend.

A repeating wave, if real, would mean the model reuses a small, cyclic code for large counts. That would work the way a clock face reuses twelve positions to track many different hours.

Each test looked inside a fixed span of characters, called a measurement window here, of either 140, 200, or 280 characters.

All scores below were measured on text kept aside and never used while fitting the count reader.

This report covers one single run on this base model. It was not repeated, and no other model is compared inside it.

## What was predicted, and what actually happened

Before the test ran, three yes-or-no questions were set, each with a numeric bar fixed in advance.

### Question 1: does a character-count signal exist at all?

The base model needed a fit-quality score of at least 0.60 to pass this first check.

This score can be zero, meaning no better than guessing the average count, one, meaning a perfect prediction, or negative, meaning worse than guessing.

The base model actually reached 0.876302, comfortably above the required 0.60.

This first check passed, clearly.

### Question 2: does the count signal use four or more separate directions?

The second question asked whether the count signal spreads across four or more independent directions inside the base model, at any window.

Using the plan's own way of counting, the largest number of directions found was fifteen, at the 200-character window.

That cleared the four-direction bar, but only at one of the three windows tested.

However, that count of directions depends heavily on which share-of-signal cutoff gets chosen.

At a stricter cutoff, the same base model needed 81, 127, and 129 directions at the three windows, in that order.

At a looser cutoff, every single window needed only one direction.

So the four-or-more finding rests entirely on picking the one middle cutoff out of three that were tried.

### Question 3: does any window beat 200 random comparisons?

The third question asked whether any window's pattern strength beat 200 random reshufflings of the same base model's own count labels.

Clearing meant ranking in the top ten out of 201 total values: the real result plus the 200 random ones.

Only one window, the 280-character one, cleared that bar, in this base model.

Its strength ranked first out of all 201 values there.

The other two windows, 140 and 200 characters, did not clear the bar.

Both ranked closer to the bottom than the top: 196th and 120th out of 201, respectively.

That means both looked smoother than nearly all of their own random comparisons, not more wave-like.

At every window, a separate check using token position instead of character count also failed to clear its own bar.

### The headline result

Because exactly one window cleared, the plan's own rule marks the overall outcome as undecided, not as a clear pass.

That rule was written into the plan before the test ran, specifically to stop one narrow result from being over-read.

So, formally, this base model's headline result is a mixed one. The existence check passed, the direction count reached four at one window under one cutoff, and the repeating-pattern check stayed undecided.

## A closer look at the one result that beat chance

The points below did not come from the original, pre-registered plan.

They come from a later review of the same recorded data, and stay labelled here as exploratory, open questions.

**Is it really a repeating pattern, or just a leftover trend?**

Looked at closely, the winning shape at the 280-character window is one long, smooth slide, not an up-and-down wave.

In the base model, it falls from about 0.36 down to about -0.16 across the whole window, with no repeat inside it at all.

That raises a real question: does a single, one-time drop count as a repeating pattern at all?

If it does not, this base model's one positive-looking result was never a repeating pattern to begin with.

**How thin is the margin behind that one result?**

The base model's winning strength there measured 0.00008673.

In the same base-model test, the strongest of 200 random comparisons reached 0.00008027, about 8 percent lower.

A different batch of 200 random comparisons could easily have scored higher than that, purely by chance.

If so, the one positive result in this whole run rests on a coin-flip-thin margin, not a clear gap from chance.

**Can the model even read counts reliably at that same window?**

At that same 280-character window, the base model's own fit-quality score for reading counts was negative: -0.793553.

A negative score means guessing the average count every time would have worked better than this base model's own readout there.

If the model cannot reliably read counts at that window, a pattern found there is hard to credit as a true character-count signal.

**Does the winning window just have noisier data?**

The 280-character window also has the fewest examples backing each character-count value, as few as 12, and at most 52.

The 140-character window's thinnest count has 27 examples, and the 200-character window's thinnest has 26.

So the one window with a positive-looking result is also the noisiest of the three, with roughly half as many examples as the other two.

Noisier averages are more likely to throw up one stray high value purely by chance.

**Does the four-or-more-directions finding hold up on its own?**

As already shown above, that finding rests on picking one specific cutoff out of three that were tried.

This makes the reach-four finding look like an artifact of cutoff choice, not a stable fact about the base model.

**Does the model's own range check agree with crediting character count here?**

There is a direct conflict inside the base model's own recorded checks.

A separate check at the 280-character window asked whether counts above 175 characters can be read at all there.

In that base-model check, every one of those higher-count groups scored below a correlation of 0.5, the bar this same check set for readable.

In the base model, the highest of those groups reached only 0.31, still under that 0.5 bar.

Yet the same window's labelling check still credits character count with the one pattern that beat random chance.

If counts above 175 characters cannot be read reliably there, that credit is hard to trust as genuine evidence of a character-count signal.

**Does fine detail hold up even inside the range this test trusts most?**

One more check looked inside characters 1 through 140, the range this test trusts most.

There, in the base model, correlation between predicted and true count ran 0.645989, then 0.8196, in the first two groups of 35 characters.

It then dropped to 0.210583 and 0.226552 in the next two groups, characters 71 through 140, in that same base model.

That is a real drop inside the base model, even though the whole range's combined score was a strong 0.876302.

So the strong overall pass may mainly reflect coarse, big-versus-small counting, not fine, accurate tracking across the whole trusted range.

**Should one narrow win outweigh two windows that looked smoother than chance?**

Stepping back, two of the three windows point the other way entirely.

At the 140- and 200-character windows, this base model's real pattern was smoother than nearly all of its own random comparisons.

An unusually smooth signal, almost never repeating, in two tries out of three, is itself a pattern, just not the predicted one.

Weighed against that, the one narrow exception at the 280-character window looks more like noise than a genuine, repeating signal.

## How this compares to a smaller version of the same model

A smaller, 410-million-parameter version of the same model family was tested before this one, using the same method twice.

That smaller base model's count signal used only one to three directions at every window, and no window ever beat random chance there.

This larger base model's headline numbers look different on the surface: up to fifteen directions at one window, and one window beating chance.

But, as shown above, both of those surface differences turn out to be fragile once examined closely, in this one run.

## Caveats worth keeping in mind

This whole test is one single run on one specific, frozen snapshot of the base model, never repeated.

Unlike the smaller model's twice-confirmed result, a repeat of this exact test on this base model could land differently.

There is also no outside, independently measured result to check this base model's existence-check number against.

So the base model's 0.60 pass mark is this project's own informed guess, not a proven, external boundary.

Finally, the existence check itself was calibrated using measurements only up to 140 characters.

That same pass-or-fail result was then used to license or void every later measurement out to 280 characters, double the calibrated range.

That is a real scope gap between what was calibrated and what was later judged, worth remembering when weighing every result above.

## Bottom line

Put simply: this base model clearly does track how many characters it has read, in some form.

But whether that tracking repeats in a wave-like way, rather than trending in one smooth direction, remains unresolved.

The one result that looked like a repeating pattern is real in the data, but thin, narrow, and hard to fully trust.

The pre-registered plan itself calls the overall outcome undecided, and this closer, exploratory look mostly supports leaving it that way.

## Gate checks

Both required gate scripts were run directly against this base-model write-up, and their exact output follows below
```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_density.py TAKEAWAYS_seedD3.md
TAKEAWAYS_seedD3.md: 0.0 codes/1000 words (0 codes, 1736 words); threshold 10.0
```
That plain-language density check passed cleanly.

This same base model's write-up was then checked a second time, for unscoped numeric claims, giving this result
```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_scope.py TAKEAWAYS_seedD3.md
0 unscoped result sentence(s)
```
That scope check passed cleanly too.
