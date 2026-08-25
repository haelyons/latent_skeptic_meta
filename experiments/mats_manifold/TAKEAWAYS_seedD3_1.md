# Second Look: Does a Language Model's Sense of Text Length Repeat?

## What this experiment tested

Some language models build an inner sense of how many characters into a passage they currently are. One earlier study found that sense moving in a repeating, wave-like rhythm, inside a large model made by another lab. This experiment asked whether that same rhythm shows up in a much smaller, openly available model. One earlier ambiguous reading was measured again, on brand-new text.

The model under test was a base language model called Pythia, sized at 1.4 billion parameters. It was tested at the base model's final training snapshot (recorded in the files as step143000). Nothing was fine-tuned, instructed, or steered. This was the plain base model reading its own internal activity. The text was 48 fresh passages, about 3,600 characters each, pulled from the public-domain novel Moby Dick. None of it was reused from any earlier round. Thirty-six passages calibrated the reading method. Twelve were held back purely for testing.

The method used three window sizes: 140, 200, and 280 characters. At each one, a statistical read-out was trained to guess the running character count. That guess came from the base model's internal activity. Then a wave-shape check looked for a repeating cycle in how well the guess tracked the true count across the window. A real cycle only counts as genuine if its strength beats 200 comparison versions. Those comparisons come from randomly reshuffling which count-label went with which internal reading, inside each held-out passage. Beating means ranking in the top ten of two hundred one, roughly the top five percent.

One rule was fixed before any measurement ran. The very slowest possible cycle, one full rise-and-fall across the entire window, could never count as a repeating pattern. After removing the main trend, that slowest cycle only measures leftover curve shape, not true repetition. This rule applied identically to the real data and to every shuffled comparison.

## What was predicted

Zero windows clearing the top-ten threshold would close the question. The answer would be a clean "no repeating pattern," echoing an earlier finding on a smaller base model. Exactly one window clearing would leave the question formally unresolved. Two or more clearing would reopen a wider search across different training snapshots.

## What happened

The base model's overall fit-quality score reached 0.879879, well past the required 0.60 floor. That score had to clear 0.60 for the base model before any pattern check could be trusted. The prior round reported 0.876302 for this same base model. This round's score for the base model landed only 0.003577 away. That is comfortably inside a recorded comparison range which decides nothing on its own.

None of the three windows cleared the top-ten threshold. At 140 characters the real cycle strength ranked 201st of 201, dead last. At 200 characters it ranked 198th of 201. At 280 characters it ranked 18th of 201, closer but still short. Zero of three windows cleared. So the pre-registered "no repeating pattern" answer applies, matching the smaller base model's earlier result.

That is the reported headline. Seven follow-up checks probed whether it actually holds up.

## Where the closing call gets shaky

**Barred measurement, counted back in.** One rule fixed beforehand barred the slowest cycle from counting as a pattern. Counted back in, under the exact same top-ten rule used everywhere else, that excluded value tells a different story. At the 200-character window it would have ranked 8th of 201. At the 280-character window it would have ranked 1st of 201. Both clear the same threshold used for every other result. Only the 140-character window's excluded value, ranked 134th of 201, would still have failed. Under the study's own rule, two windows clearing reopens the wider search instead of closing the question.

**A familiar shape, not a fresh artifact.** At the 280-character window, in the base model, the read-out slides almost in a straight line. In the base model, it falls from 0.365 near the window's start to -0.163 near its end, a line-fit strength of -0.917. An earlier round on this same base model found nearly the same slide. That base-model round measured 0.36 down to -0.16, line-fit strength -0.914. Smaller windows show the same downward slide even more sharply. The base model's line-fit strength reaches -0.9985 at 140 characters and -0.9924 at 200 characters. A rule meant to screen out one stray shape is instead removing a shape that keeps recurring.

**A position count that outruns its own data.** The summary record claims text-position counts up to 58, 73, and 97 across three window widths. The full position-by-position data file covers every held-out passage in the base model's test set. That file never goes above 44, 66, and 85 at those same three widths. Using the higher, unsupported figures, the record drops 17, 16, and 22 rare positions as too sparse to use. Using the actual highest values found in the data, only 3, 9, and 10 positions would need dropping.

**A stated safeguard that never actually fires.** A second accuracy check, at the widest window, is described as able to invalidate the whole run. It is described as carrying the same weight as the main fit-quality floor. Its own reading came back negative, in the base model's own record. It matched neither the pattern expected if wide counts were readable, nor the pattern expected if they were not. The run's own "not invalidated" marking stayed unchanged regardless, the same whether this check passed or failed.

**The same check flips sign on identical text.** Scored inside the 140-character window, one stretch gets a positive reading in the base model. That stretch runs from character 106 to 140, and its base-model reading is 0.244373. Scored inside the 280-character window, that exact same stretch gets a negative reading instead. The base model's second reading for characters 106 to 140 is -0.058152. The sign flips completely for what both records call the same measurement.

**A complexity count that swings sevenfold.** One count tracks how many independent directions capture most of the read-out's spread. At the 200-character window, that count fell from 15 in the earlier round to 2 in this round. Both rounds used the same base model family and the same window. Only the text and the random comparisons changed between them. Such a swing suggests the count depends on the specific draw, not on the model.

**A quiet gap inside the widest check's own window.** Before declaring its mismatch verdict, the widest-window accuracy check skips part of its own window. It leaves out characters 141 to 175 of the 280-character span, in the base model's own record. That gap is one eighth of the window's full length. Both halves of the comparison exclude it, with no stated reason given.

## Bottom line

The reported closing answer is that no repeating pattern was found, in this base model, up to 280 characters. That matches what an earlier, smaller base model showed. The answer follows directly from the study's own stated rules. Every one of the seven checks above lands on a place where a different reading of the same base-model data was equally defensible. That reading would have changed the closing call, or at least weakened it. This was one run, on one draw of text and one draw of random comparisons. The record itself says a second run, on the same computer, would only repeat the same instrument. It would not independently confirm the finding.

---

## Gate output

Run of `gate_density.py` on this base-model takeaway file:

```
TAKEAWAYS_seedD3_1.md: 0.0 codes/1000 words (0 codes, 1247 words); threshold 10.0
```

Run of `gate_scope.py` on the same base-model takeaway file:

```
0 unscoped result sentence(s)
```

Both gate scripts above exited clean on this base-model file.

