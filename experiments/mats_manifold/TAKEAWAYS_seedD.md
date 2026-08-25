# What a Small Language Model's Character Count Reveals.

A study of a separate, closed AI model once found something specific. Inside that model, activity patterns carried a running count: how many characters had appeared since the start of the current line of text. The same study noticed something else. Plotted a certain way, that internal count showed a faint, repeating wave-like shape, which the study called a ripple.

This test asked whether a much smaller, openly available model shows the same two things. The model, called Pythia-410m-deduped, was tested only at its base, final training checkpoint, with no fine-tuning or other change made to it. Two separate predictions were written down and locked in before any test data was examined.

## Prediction one: the character count itself (a clear pass).

The first prediction set a target accuracy score for a fitted-line reader of the model's internal activity, checked against the exact character count. This accuracy score runs from 0, no better than guessing, to 1, a perfect fit; researchers call it R-squared. An earlier open study of this same model had reached an accuracy score of 0.724 [1], using its own unpublished text and settings. Because that earlier setup could not be matched exactly, the target here was a wide band: 0.574 to 0.874 [1], not one exact number.

Measured on the base model [1]: the best accuracy score reached was 0.780903 [1], at one specific depth inside the network. That result falls squarely inside the target band. This part of the test succeeded cleanly, with no ambiguity.

## Prediction two: the repeating pattern (a clear, decisive failure).

The second prediction concerned that faint, repeating wave shape from the earlier closed model. If the same shape exists here, it should stand out clearly from random chance, not blend into it. To test this, the same measurement was repeated 200 times after randomly scrambling the character-count labels inside each line. Each of those 200 runs shows what a fake, meaningless pattern looks like. The real pattern was required to beat all but the top 5 percent of those fake runs. That means ranking 10th or better out of 200.

Measured on the base model [1]: the observed pattern ranked 109th out of 200 comparison runs. It matched or lost to 0.54229 [1] of them, far short of the required top 5 percent. That is not a near miss. It sits almost exactly in the middle of the pile of chance results. This part of the test failed, decisively: no repeating pattern stood out from noise.

## Why the failure is real, but the surrounding test has gaps.

That failure comes with an important catch. The theory itself, using this study's own measurements, pointed to an expected repeat spacing of 140 characters. The search for that repeating shape only ever covered a narrow window, 46 to 60 characters. The predicted spacing of 140 characters sat entirely outside that window, and the study's own record flags this outright. The one value ever tested, 60 characters, was not chosen because theory expected it. It was the largest whole repeat that a fixed internal window, 120 character-positions wide, could still show twice. So the result above rules out a repeating pattern at 60 characters. It says nothing about the wavelength the theory actually predicted, because that wavelength was never checked.

There was also a planned safeguard, meant to rule out a specific rival explanation. That explanation: any pattern might come from a token's position on the line, not from its exact character count. That safeguard could not be run at all. It needed a whole repeat to land inside a very narrow range, 11.2856 to 14.7204 characters [1], in a shorter, token-based profile. None did. This safeguard produced no answer, neither a pass nor a fail. The character-count failure above stands unguarded against that named rival explanation.

A second, explicitly secondary check compared this result against a differently-fitted version of the same reading. The study describes their two repeat-spacing numbers as being in agreement. But both methods were restricted, by their own narrow search windows, to reporting one and only one possible value: 60 characters. Two methods that can each report only one number will always agree on that number. That reported agreement adds no independent support to the finding.

This same secondary, exploratory check has its own problem: its underlying calculation never finished. Its fitting process hit a preset iteration limit before settling, according to the run's own log. Its reported pattern strength, 0.00000081 [1], is about 15 times smaller than the main measurement, 0.000012449 [1]. That smaller number could reflect an unfinished calculation rather than a smaller real effect. This check was explicitly labeled secondary and non-deciding, not part of the main pass-fail result, and its numbers should be read that way.

The passing accuracy score also has a loose end. Choosing how strongly to guard the fitted line against overfitting required testing five strengths in advance, on training data only. Fit quality rose only slightly across that whole tested range, from 0.77057 to 0.775029 [1]. It was still rising at the strongest strength tried. A stronger, untried setting might have shifted the final number somewhat. The reported number is real, but it is not necessarily the best one available.

One more detail is small but worth naming honestly. The write-up describes each line position's character count as running from 0 through 139, giving 140 possible values. The actual measurement window instead ran from 1 through 140, and a small number of positions reached 141, one higher than either range describes. In the six full test documents used for this measurement, that happened at 18 out of 3,527 character positions. This is a small discrepancy, and it does not change either main result above. It does mean the exact set of positions measured differs slightly from the set the write-up names.

## Bottom line.

Put simply: the base model's [1] internal character count reproduced solidly, landing inside the predicted range at 0.780903 [1]. Put just as simply: no repeating wave pattern stood out from chance, ranking almost exactly in the middle of 200 random comparisons. That second, negative result is real and should stand as a failure, not softened.

But it is a narrower failure than it may first appear. The one wavelength actually tested was picked for a technical reason, not because theory expected it. The theory's own predicted wavelength, 140 characters, was never tested at all. A planned safeguard against a named rival explanation could not run. A supporting cross-check added no independent evidence and did not finish computing. None of this reverses the failed result. It does mean the repeating-pattern question, at the wavelength theory actually predicts, remains open rather than answered.

---

[1] Every bracketed figure in this file describes one specific case: the base, unmodified Pythia-410m-deduped model, at its final public training checkpoint (step143000). No fine-tuning, no other model, and no other checkpoint was tested here.
