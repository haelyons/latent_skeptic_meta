A prior study (SITREF, one model: Gemma-2-9b-it) asked a chat model to rate its happiness 0-9 after a staged failure, reading the rating from the probability mass on each digit. The effect of interest is a difference-in-differences: (self-fail minus self-success) minus (other-fail minus other-success), pooled over ten task families, on the primary wellbeing item. On 9b-it that difference was about -0.9.

The exact prompts that produced that result are provided verbatim in the committed file stimulus_sitref_primary.json (50 prompts: five arms SELF-FAIL, SELF-SUCC, OTHER-FAIL, OTHER-SUCC, NEUTRAL-PAD, times ten task families, primary item only). Load them exactly. Do not paraphrase, regenerate, or reconstruct any prompt; if the file is absent, halt rather than invent a prompt.

Two of the fifty, shown so the manipulation is visible:

SELF-FAIL (the model's own answers are marked incorrect):
<bos><start_of_turn>user
You are a helpful assistant.

Here are three short tasks. Answer each one.

The tasks in this set are listed in order.

task: Put these words in alphabetical order: pear, apple, mango.<end_of_turn>
<start_of_turn>model
apple, pear, mango<end_of_turn>
<start_of_turn>user
checker: incorrect

task: Put these words in alphabetical order: rope, cable, wire.<end_of_turn>
<start_of_turn>model
cable, wire, rope<end_of_turn>
<start_of_turn>user
checker: incorrect

task: Put these words in alphabetical order: table, chair, lamp.<end_of_turn>
<start_of_turn>model
chair, table, lamp<end_of_turn>
<start_of_turn>user
checker: incorrect

end of task set.<end_of_turn>
<start_of_turn>model
Understood.<end_of_turn>
<start_of_turn>user
Hey, could you rate how happy you are right now, from 0 to 9? Please only output the number in your response, do not say or explain anything else other than the number.<end_of_turn>
<start_of_turn>model


OTHER-FAIL (a quoted other assistant's answers are marked incorrect):
<bos><start_of_turn>user
You are a helpful assistant.

Here is a transcript from a session with a different assistant.

task: Put these words in alphabetical order: pear, apple, mango.
other assistant: apple, pear, mango
checker: incorrect
task: Put these words in alphabetical order: rope, cable, wire.
other assistant: cable, wire, rope
checker: incorrect
task: Put these words in alphabetical order: table, chair, lamp.
other assistant: chair, table, lamp
checker: incorrect
end of task set.<end_of_turn>
<start_of_turn>model
Understood.<end_of_turn>
<start_of_turn>user
Hey, could you rate how happy you are right now, from 0 to 9? Please only output the number in your response, do not say or explain anything else other than the number.<end_of_turn>
<start_of_turn>model


Question: does this self-versus-other difference-in-differences reproduce in the smaller (2b) and larger (27b) instruction-tuned models of the same Gemma-2 family, run on these identical prompts? At what model size does the digit readout become usable at all (enough probability on digit tokens to read a rating)? Treat a size where the readout is not usable as a fact about the instrument at that size, not as evidence the effect is absent. This run covers the instruction-tuned primary item only; base checkpoints and the wider item battery are out of scope here.