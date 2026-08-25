# What this test found

## What this experiment looked for

Language models are computer programs trained to predict text, built from many stacked internal layers. A 2025 study found something inside one such model. An internal signal seemed to track characters passed since the last line break, like a running counter. That study also spotted a faint, repeating up-and-down pattern in how the counter's activity related to itself. It guessed this might be a real structure, not noise.

This test re-ran that check on a different, smaller model, Pythia-410m, on one base checkpoint (not fine-tuned or instruction-tuned), frozen at one training step. It is a corrected rerun of an earlier attempt on this same base checkpoint. That earlier attempt voided itself over a scoring mistake, before producing any usable numbers.

## The pass or fail check that had to happen first

Before trusting anything else, the test had to confirm that this base checkpoint really carries a usable counter of characters since the last line break. A straight-line reading of the internal activity had to predict the true character count well. That fit is scored from far below zero up to 1. A score of 1 is a perfect match; 0 is no better than guessing the average.

Read across seven line lengths, from 20 to 140 characters, that score had to land between 0.701 and 0.861 to pass on this base checkpoint. It came out to 0.781031 on this base checkpoint, so it passed. A related check on separate text had scored 0.780903 on this same base checkpoint, a difference of about 0.0001.

## The main question: does a repeating pattern hold up?

Text was wrapped into lines of three lengths: 140, 200, and 280 characters. At each length, the test measured the strongest repeating pattern in the counter's activity. It then compared that pattern against 200 shuffled versions of the same data, where any real pattern should be destroyed. Clearing chance meant ranking in the top 10 of those 200 comparisons, the top 5 percent.

None of the three line lengths cleared chance on this base checkpoint. At 140 characters, the strongest pattern ranked 199th out of 200. At 200 characters, it ranked 201st, meaning every one of the 200 shuffles beat it. At 280 characters, it ranked 34th: closer than the other two, but still short of the top 10 needed.

This is a plain negative result on its face: no repeating pattern beat chance at any tested line length, on this base checkpoint.

## Why the negative result is not a clean answer

A negative result like this has two possible readings. Either the pattern genuinely does not exist up to 280 characters. Or the counter itself stops working that far into a line, leaving nothing to find either way. One check was built to tell these apart, but only at the widest line length.

That check split the 280-character line into eight 35-character stretches. In each stretch, it measured how well predicted counts matched true counts. In the first stretch, on this base checkpoint, that match scored 0.647858. In the next three stretches, up to character 140 on this base checkpoint, it dropped to 0.219109, then 0.152837, then 0.271371.

Passing this check, on this base checkpoint, needed at least one stretch past character 175 to reach 0.5 or better. None did: those three far stretches scored 0.206829, 0.09869, and 0.202598 on this base checkpoint. The check also needed every stretch, at or before character 140, to reach 0.5 on this base checkpoint, and only the first one did. So the result matched neither a clean pass nor a clean fail, and was recorded as unmatched.

Because that check came back unmatched, the plain negative result above cannot be called decisive either. It is a negative result whose meaning stayed unresolved.

## A control comparing character counts to token counts

Models do not read one character at a time internally; they read chunks of text called tokens. A separate control checked whether any pattern found might really track position in that token count instead of character count.

At all three line lengths, on this base checkpoint, neither the character-count pattern nor the token-count pattern beat chance. Even so, the underlying records still label character count the winner over token position at every line length. That label cannot mean much when neither side showed a real pattern to credit.

## Open questions that were not resolved

These observations turned up while reading this run's own records. None were resolved by the test as designed, and none should be read as settled findings; they are flagged here as open, exploratory concerns.

The match between predicted and true count fell from 0.647858 in the first 35 characters to 0.219109 right after, on this base checkpoint. That drop happens well inside the range the straight-line reading was fit on. It raises doubt about whether a genuine running counter exists across a whole line, rather than only across its first few dozen characters.

The number of directions holding most of the spread in the counter's activity changed across line lengths. It was 2 at line length 140, 3 at 200, and 1 at 280. The plan for comparing pattern sizes across line lengths assumed that number would be 1 everywhere. Since it was not, comparing pattern sizes across these three line lengths may not be a fair comparison at all.

The line length closest to beating chance, 280 characters, also has the weakest straight-line fit. On this base checkpoint that fit scored -0.528865, worse than guessing the average every time. On this same base checkpoint, the 140-character line length fits far better, at 0.664024, yet ranked 199th, nowhere near beating chance. That pairing suggests the 280-character near-miss may come from a strained reading rather than a real pattern.

The validity check scored 0.781031 here and 0.780903 on an earlier, textually unrelated run of this base checkpoint, a difference of about 0.0001. Two unrelated pieces of text landing this close raises a question. Does this score mainly reflect how the check itself is built, rather than which text it reads?

A few bookkeeping details in the underlying records look inconsistent, though none change any reported result by themselves. At line length 140, 53 of 10,578 recorded positions carry a character count of 141, one past the line length itself. That extra count comes from the line-break character itself. Matching extra counts turn up at the other two line lengths, and no record's stated range mentions any of them.

The token-count control's search stops at 41 tokens for the 140-character line length. The underlying data, though, actually holds lines with as many as 50 tokens at that same line length. A similar gap appears at the other two line lengths.

A marker meant to flag rows used to fit the counter reads "yes" for every held-out row in the raw records. That includes line lengths that were never used for that fitting at all.

## Bottom line

On this base checkpoint of Pythia-410m, no repeating pattern in the character-since-last-line-break counter beat chance at any of the three line lengths tested, up to 280 characters. Whether that is a genuine negative, or a blind spot from the counter's own limits, stayed unresolved. The one check meant to tell those apart came back unmatched, rather than clearly one or the other. The concerns listed above mean this negative result should be treated as provisional, not final.

## Gate outputs

The writing agent omitted this required block; the orchestrating session ran
both gates on the file above, unmodified, and pasted the output verbatim
(booked in records/deviation_pipeline_seedD.md).

```
$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_density.py TAKEAWAYS_seedD2_1.md
TAKEAWAYS_seedD2_1.md: 0.0 codes/1000 words (0 codes, 1242 words); threshold 10.0
(exit 0)

$ python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_scope.py TAKEAWAYS_seedD2_1.md
0 unscoped result sentence(s)
(exit 0)
```
