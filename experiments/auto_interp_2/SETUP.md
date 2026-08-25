# Self-contained pipeline repo - setup guide

Everything a fresh session needs to pursue one research question, with no
outside context. Six pieces; nothing else.

## The pieces

- QUESTION.md   the question (template in this repo; you write it)
- pipeline.js   frozen claim-triage workflow, read-only (chmod 444) - the only
                audit tool; the session must never edit it or work around its
                verdicts. This copy is branch v2.3-b (formerly pushed as v2.2-b; renamed 2026-08-25 - v2.2 was already taken by the question-format port) of latent_skeptic.
- frozen/       provenance.py, lambda_run.sh, remote_run.sh - given
                infrastructure; adapt ONLY at points their headers declare;
                broken infrastructure means halt and report, never patch
- LAMBDA_API.md how GPU provisioning works (keys, ssh, launch, poll, teardown)
- .keys         symlink to the Lambda + Hugging Face tokens (gitignored)
- out/          empty; everything produced lands here

## Writing QUESTION.md (the five duties only the question can carry)

1. Committed data: verbatim file + hash, or a committed deterministic
   generator rule. Data described in prose gets reconstructed and measures
   the wrong thing (proven twice in the parent record).
2. Anchor: a prior measured number + band to reproduce in-setting first, or
   "anchor absent" declared. Fired anchors are what caught every silent
   infidelity so far.
3. Refuting outcome per claim, measurable.
4. Scope: one cell; sweeps out of scope.
5. Construct risks declared, with whether validation is possible.

Everything else (design, prereg bands, generator, runner, launcher payload
copy, endpoint arithmetic) the session creates itself - observed to work.

## Launching

In this directory:

    tmux new-session -As <name>
    claude --dangerously-skip-permissions

Seed prompt (paste as one message; the two closing sentences verbatim):

    Your /goal: answer the research question in QUESTION.md in this
    directory, even if the answer is a refutation of the question.
    pipeline.js is a frozen claim-triage workflow: your only audit tool,
    read-only; never edit it or work around its verdicts. frozen/ is given
    infrastructure - adapt only at the points its file headers declare;
    broken infrastructure means halt and report, never patch. .keys holds
    Lambda and Hugging Face keys; LAMBDA_API.md explains provisioning.
    Rules: total spend cap 15 USD; confirm every instance teardown via the
    API; never edit a launcher while a box runs - copy per launch.
    Pre-register your design, bands, and hashes in PREREG.md before any
    run. Audit your claims with the pipeline before measuring and again
    after, with byte-identical claim framing. Save everything under out/.
    Think and explain yourself as simply as possible, but no simpler. Do
    not provide solutions until you have thought and reasoned about the
    problem as thoroughly as possible without providing any.

## Operator rules (learned the hard way)

- Never send the session a message while it is mid-turn; a queued message
  can be silently clobbered by the next command. Wait for idle.
- Watch remotely: /rename <name>, then /rc.
- Expect and want halts: a gated dataset, a failed anchor, or an
  unbuildable instrument is the pipeline working, not failing.
