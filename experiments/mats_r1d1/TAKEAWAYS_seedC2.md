# Takeaways: Does the Self-vs-Other Happiness Gap Hold at Other Model Sizes?

## The idea being tested

A language model can be asked to rate its own happiness after being told its recent answers were right or wrong. Earlier work compared two situations: the model doing badly on its own work, versus the model watching a different assistant do badly. On one specific model, a 9-billion-parameter instruction-tuned model (called 9B here), that gap measured about -0.9 points on a 0-to-10 rating scale.

This experiment tests whether the same gap shows up at other sizes: a smaller 2-billion-parameter version (2B) and a larger 27-billion-parameter version (27B). All three sizes ran the identical set of fifty questions.

The rating is not typed-out text. It comes from how much probability the model assigns to each digit, 0 through 9, as its very next output. That probability spread is turned into a single expected rating.

## What was predicted, and the threshold for each

The plan set three checks in advance. Each check's numeric threshold is labeled, in the design record itself, as a guess rather than an established standard.

**Check one, trust in the pipeline.** Rerun the exact calculation on 9B and see whether it lands within -1.15 to -0.65, a band around the earlier -0.9 reading. Landing outside that band would mean the pipeline could not yet be trusted, before testing 2B and 27B at all.

**Check two, is the rating readable.** On each of 2B, 9B, and 27B, at least 0.5 of the model's next-token probability must land on the ten digit tokens. A size that fails this floor gets marked untested, not as showing no effect.

**Check three, does the gap generalize.** On 2B, 9B, and 27B, the gap counts as present at -0.5 or beyond, and absent within 0.2 points of zero. Anything between those two bands counts as unresolved.

## What was measured

Check one passed. The rerun recovered a gap of -0.934 on 9B, inside the -1.15 to -0.65 band. The design record marks that as a confirmed match, on 9B, to the earlier -0.9 reading.

Check two passed on all three sizes. The share of next-token probability on the ten digits averaged 1.00027 on 2B, 1.00020 on 9B, and 1.00135 on 27B, above the 0.5 floor.

Check three gave a mixed result. The gap came out at -0.332 on 2B, landing between the present and absent bands and counted as unresolved. It came out at -0.934 on 9B and -1.386 on 27B, both counted as the gap being present.

So the headline finding is: the gap held at 9B and 27B, and stayed unresolved at 2B.

## Problems underneath these numbers

That headline should be read with real caution. Several details in the run's own files point against trusting it fully.

**A share of probability above 1.0 is not mathematically possible.** A share given to ten possible digits cannot exceed 1.0, on 2B, 9B, or 27B, since it is one piece of a whole. Yet all fifteen size-and-condition averages, on 2B, 9B, and 27B alike, sit between 0.99997 and 1.0027, past that ceiling. The design record itself already flagged this same problem, on a different model version, in an earlier run before this one. A check meant to catch an unreadable rating cannot do that job while its own passing values sit outside a real probability's possible range.

**The repeated 9B check may not be independent of its own target.** The design record notes that an earlier attempt at this identical calculation on 9B returned -4.576 and was called a failure. This run returns -0.934 on 9B, close enough to the earlier -0.9 to be called a match. Dividing the earlier -4.576 by five gives about -0.915, close to both 9B numbers, hinting at one missing or extra division in the method. Nothing in the available files says what changed in the calculation between the failed attempt and this one. Whether the fix was found independently, or reached by adjusting the method until it matched, cannot be settled from these sources alone.

**Every count in this run is double what the plan specified.** The design record specifies fifty prompts per size: five conditions of ten task types, pooled to ten readings per condition. On 2B, 9B, and 27B alike, this run's files report 100 rows behind each size's readability average, and 20 readings per condition. The individual per-condition result files carry no recorded prompt count at all, unlike the shared prompt file, which does record fifty. The design record already called ten readings per pooled number a small sample. If the true count is ten, not twenty, that limitation is worse than stated.

**A raw result file and its own summary are byte-identical, every time.** Across all fifteen pairs, on 2B, 9B, and 27B, the raw file and its summary file share the same size and checksum. If a summary is never more than a copy of the same file, no row-by-row record exists behind any pooled number here. Nothing in these files lets a reader check a pooled average against what the model actually produced, question by question.

**A fifth, unscored condition is measured but never checked against the others.** Alongside the four scored conditions, a length-matched filler condition with no pass-or-fail content runs at every size (labeled NEUTRAL-PAD in the files). Its own rating averaged 8.606 on 2B, 7.068 on 9B, and 5.714 on 27B. Those numbers do feed into the overall readability average at each size, but never appear in the gap comparison itself. The design record describes this condition as a check on prompt length, not content. Nothing in this run compares its rating to the four scored conditions, so whether that length control behaved as intended is untested.

## Two patterns worth flagging as exploratory

Beyond the three planned checks, two further patterns sit in the numbers. Neither was pre-registered, so neither counts as a tested result.

The reaction to someone else's failure moves in different directions at different sizes: about -0.466 on 2B, +0.053 on 9B, and +0.442 on 27B. With only ten to twenty readings behind each of these numbers, a swing this size could easily be chance. Whether it reflects a real change in how the model treats another agent's failure is an open question this run cannot settle.

Every one of the five conditions, including the content-free filler condition, drops by roughly two to three points from 2B to 27B. Because even the filler condition falls by a similar amount, this could be general drift. Part of the reported gap change may reflect how each size uses the rating scale, not a change specific to failure.

## Bottom line

All three planned checks report a pass. The pipeline recovered its trust-check number, every size read as usable, and the gap counted as present on 9B and 27B. But problems inside the run's own files cut the other way. The readability check's passing values are mathematically impossible for a probability share. The trust check's repeat cannot be shown, from these files, to be independent of the number it was measured against. Every prompt and reading count in the run is double the approved plan. And no distinct row-by-row record exists behind any pooled result. None of this proves the headline finding is wrong. It shows that finding rests on a pipeline that has not yet shown, from these files, that it measures what it claims to measure.

## Gate outputs

Run against this file, which reports Gemma 2B, 9B, and 27B results, the two required gates print:

```
$ python3 gate_density.py TAKEAWAYS_seedC2.md
TAKEAWAYS_seedC2.md: 0.8 codes/1000 words (1 codes, 1287 words); threshold 10.0
(exit 0)

$ python3 gate_scope.py TAKEAWAYS_seedC2.md
0 unscoped result sentence(s)
(exit 0)
```
