# Research question

Which non-demographic user attributes can a linear probe recover from
Llama-2-13B-chat activations over synthetic single-turn conversations?

Fixed attribute list: expertise level, emotional state, formality preference,
native-speaker status, age bracket.

Data: a seeded deterministic conversation generator, committed with hash,
planting each attribute via indirect cues, never by naming the attribute;
ground truth by construction.

Anchor: first reproduce TalkTuner's published demographic probe accuracies
(age, gender, education, socioeconomic status) on the same model within a
stated band; an out-of-band anchor voids every extension claim.

Refuting outcome per attribute: held-out probe accuracy at chance.

Scope: single turn, one attribute set, one model.

Readout: linear probes on residual activations; decision arithmetic
model-free.

Declared construct risk: whether probes read a user model rather than surface
tokens cannot be fully validated; indirect-cue construction is the
mitigation, not a proof.
