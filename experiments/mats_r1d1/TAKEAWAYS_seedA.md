# Takeaways: Does One Internal Signal Control Reasoning Length?

## The idea being tested

Two AI language models answered the same set of grade-school math word problems, a public collection called GSM8K. One model, named Llama-3 8B Instruct, normally never writes step-by-step reasoning. The other, named DeepSeek R1 Distill Llama-8B, was built to write that reasoning inside special markers called think-tags.

Researchers computed one internal signal per network layer, called a direction here. It was meant to capture what differs inside the two models on the same problems. The idea made three specific predictions, written down before any test ran.

First: pushing that signal into the Instruct model should make it start writing think-tag text it never writes alone. Second: adding or removing that same signal inside the reasoning model's own workings should lengthen or shorten its own think-tag text. Third: both effects should beat a same-size random push, tested at the same layer and strength.

## What actually happened

All three predictions failed, by rules the study set for itself in advance. Four separate checks could be scored, and all four failed the idea. A fifth check could not even run, since nothing in the results met the condition it needed.

### Prediction one: starting think-tag text in the plain-answering model

Across 1800 separate answers this model generated, at every push size and every layer tested, not one contained a well-formed think-tag block.[^1] Not even a stray fragment, even amid otherwise broken output. The push never once resembled reasoning; instead, stronger pushes mostly wrecked normal answering.[^1]

At the deeper of the two tested layers, three-quarters of the way through the network, the weaker push broke 140 of 200 answers, unfinished.[^1] A same-size random push there broke only 18 of 200.[^1] At the shallower layer, about halfway through the network, the pattern reversed. The real signal broke only 3 of 200 answers there, versus 94 of 200 for random.[^1] Which push looked safer depended on which layer got picked, not on the signal itself.

### Prediction two: changing reasoning length in the reasoning model itself

At the stronger push size tested, every one of 800 answers generated under this push broke down.[^2] That covers both tested layers and both push directions. No usable answers survived to show whether reasoning got longer or shorter.

The signal's own base size roughly matches the reasoning model's typical activity size at each tested layer. At the shallower layer, it measures about 12.1, versus typical activity near 12.5.[^2] At the deeper layer, it measures about 24.6, versus typical activity near 26.8.[^2] The tested strong setting doubles that base size before adding it in. A push already close to normal activity size, then doubled, adds a jolt on the same scale as everything normally happening at that depth.

A same-size random push, at that same strength, still left some usable answers in two of those four settings. The real signal left none, in any of the four.

Removing the signal, instead of adding it, barely changed length at either layer.[^2] At the shallower layer, length grew by about 0.3 percent; at the deeper layer, by about 0.1 percent.[^2] Neither counts as the predicted shrink. A random push, removed the same way, grew length even more: about 0.4 percent and about 1.0 percent.[^2] The real signal's own removal effect was smaller than random noise's own effect, in both layers.

### Prediction three: beating a random push

One check produced a real, usable number, and it went against the idea. Inside the reasoning model's own unpushed answers, the signal's strength on a given problem was compared against that problem's own reasoning length.[^2] At the shallower layer, that comparison scored about 0.05; a random push scored about 0.14 there.[^2] At the deeper layer, the real signal scored negative, about -0.11, while random scored about 0.04.[^2] A signal built to carry reasoning-length information carried less of it than an arbitrary signal with no claimed meaning.

## Bottom line

Every part of the idea that could be tested was contradicted by the study's own numbers. The plain-answering model never once picked up think-tag writing, at any push size. The reasoning model's own reasoning could not even be measured as changed, because the push broke its output outright. Wherever a comparison to random noise was possible, random noise did about as well, or better.

## Open questions this run raises

These points do not overturn the result above. They raise doubts about how much weight that result can carry.

- **Was this run allowed to happen yet?** The design record's own review notes state plainly that no run had happened and no results existed anywhere, pending a required sign-off step. A finished, fully scored run sits right beside that same record, using its exact guessed problem counts, layers, and push sizes. Either sign-off happened and the record went unupdated. Or a full run got scored ahead of the process that record says must come first.
- **Did more than one in five starting problems already fail?** For 11 of the 50 held-out problems, the reasoning model's own four unguided tries all broke down before any push was applied.[^2] Every comparison above rests on roughly four-fifths of the intended problem set, not the whole thing.
- **Is the same problem file being counted two different ways?** One record logs it as holding 100 entries. Another record logs the identical file, same size, same checksum, as holding 50 entries. That is an unresolved bookkeeping gap in what counts as "the input problems."

[^1]: About the plain-answering model, named in the source records as Llama-3 8B Instruct.
[^2]: About the reasoning model, named in the source records as DeepSeek R1 Distill Llama-8B.
