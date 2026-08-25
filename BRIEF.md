# Program brief - agent-run falsification pipeline (as of 2026-08-24)

One research question goes in; a pre-registered experiment, adversarial audits,
and plain-language takeaways come out. Agents speak only through typed
contracts; anything unfaithful returns a refusal (halt, void, on-hold), never
an answer. A human appears three times: choose the question, approve budget,
read the open questions.

The stack: doctrine (four heuristics: independence of actors, decisiveness of
numbers, groundedness in raw artifacts, the-instrument-is-a-claim) -> a
claim-triage tool (fresh skeptics, one confound each; an isolated reader
re-derives every cited number) -> a pipeline (design, diff, triage, human
gate, build, box run, elicit, takeaways) -> since this week, a delegated
manager session that drives all of it from a 150-word prompt.

## Six seeds, one line each

- A (reasoning direction, two 8B models): all testable endpoints refuted at
  guessed coordinates; the refutation is scoped to those coordinates.
- B (deliberately flawed twin): halted on a licence-gated dataset the design
  had invented - a question built on unnamed data cannot run faithfully.
- C (self-vs-other happiness at 3 sizes, data described in prose): ran clean,
  anchor fired (recovered 4.58 vs recorded 0.9) - the runner had reconstructed
  its own stimulus. Everything on hold.
- C2 (same question, exact prompts embedded): anchor passed (-0.934 vs -0.9);
  effect present at 9B (-0.93) and 27B (-1.39), indeterminate at 2B (-0.33),
  monotone with scale - but its own takeaways showed the readout reporting
  impossible probability shares above 1.0.
- D (character-count ripple, Pythia-410m): curve replicated (0.781 in band),
  ripple null - but the post phase showed the predicted 140-character period
  was never testable by the instrument. Seed amended before execute.
- D2 (aperture invariance, built from D's failures): shuffle nulls cleared at
  all three windows - and the anchor fired (0.549 vs band 0.701-0.861), so
  the whole run is void by its own rule. Second anchor save.

Lesson distilled: a good question carries five things nothing downstream can
conjure - committed data (or a committed generator), an anchor number with a
band (or "anchor absent" declared), a refuting outcome per claim, one-cell
scope, and declared construct risks. Everything else the machinery can build.

## This week's additions (v2.2, two branches)

Consumption law: a run halts unless the human gate answered, of record, every
queued question, non-reproducing grounding note, and instrument validation.
Forced by: a pre-run note flagged C2's readout as non-physical and nothing
consumed it. Also: instrument validation wired (H4), degenerate skeptic
findings rejected (a committed finding once read confound "test", crux
"test"), void flags propagate to every record (D2's void lived only in its
summary while per-window files said "null cleared").

## The two live experiments

1. Idea-to-question calibration: one blind agent harvested 20 research ideas
   from Neel Nanda's writings; a second blind agent filtered them against our
   criteria. Five survived; all replication-shaped. Finding: our criteria
   select runnable-and-honest, not interesting - taste is a missing criterion
   (a GPT-2 idea survived that the source explicitly disfavours).
2. The manager experiment: a fresh session, seeded with only the chosen
   question (user-attribute probing on Llama-2-13B-chat), the frozen tool,
   and GPU keys, ran the entire arc itself for $0.67: pre-registered bands
   before launch, built and hashed its own dataset generator, validated its
   instruments (first actor ever to complete that loop), probed all five
   attributes far above chance (0.81-1.00) - and returned VOID, because its
   anchor fired: the published gender-probe accuracy (0.94) reproduced at
   0.4467 on template-disjoint families. Its own diagnosis: the probes learn
   template-family surface cues, not a user model - exactly the construct
   risk the question declared.

## The wording A/B and the instability finding

Two branches differ by three skeptic-prompt sentences ("Most strong effects
are confounded", "Think before answering; do not guess", "State your
conclusion; do not then hedge it away"). Both audited the manager's claims
with identical inputs. Statuses agree in 66 percent of 59 matched cells;
flips run both directions and mostly cancel (net drift about 3 cells). No
wrongful acquittals, no hedging difference. The real finding: single-sample
skeptic verdicts are unstable on roughly a third of discretionary cells,
which dwarfs any wording effect - and no same-text repeat has ever been run
to measure that noise floor. The pre-registered decision (which wording
merges) is therefore still open, along with: the original known-claim
protocol cells, and a fresh sandbox (auto_interp_2) ready for a new question
on the noisier branch.
