latent_skeptic_meta
===================

Private, permanently. The session transcripts in this repo contain full
tool output, local paths, keys-adjacent commands, and infrastructure
detail. Never public, never mirrored.

The tool is latent_skeptic/ (submodule, pinned; an experiment that used a
different version says so in its own records). Each directory under
experiments/ is one experiment: question, ledgers, records, results, and
the transcript of the session that drove it, side by side. Transcripts
that drove several experiments sit at the root. To answer "how was this
done": ledger first, records second, transcript last. Heavy raw artifacts
(activations, generations archives) stay on disk but out of git; every
provenance record and per-row result file is tracked.

Sessions
--------
Root jsonl files are the DM-HACK-RETRO lineage driven from the dm_hack
directory (question-elicitation experiments A-C and the mats_r1d1 seeds;
exact letter labels were not recorded per file, dates order them):
  f52c84de 2026-08-16 | e82f17f1 2026-08-16 | bf292b7e 2026-08-17
  6b6694c2 2026-08-18 | 9d2b7dee 2026-08-18 | f8dd62b0 2026-08-19
  7c6fa1c3 2026-08-21 | e8928553 2026-08-25 (RETRO-C-ii; snapshot of a
  live session) | 4a66f7ec 2026-08-25 (RETRO-E; snapshot of a live
  session; drove auto_interp, auto_interp_2, and latent_skeptic v2.3)
Colocated transcripts:
  experiments/mats_manifold/2fb15c4d.jsonl  RETRO-D, seeds D through D4.1
  experiments/auto_interp/4b5dae78.jsonl    RETRO-E-manager, the $0.67 run

New experiment
--------------
Make experiments/<name>/ with: QUESTION.md (committed data or generator
rule; anchor + band, or "anchor absent" declared; refuting outcome per
claim; one-cell scope; declared construct risks), pipeline.js (frozen copy
from the latent_skeptic pin, chmod 444), frozen/ (provenance.py,
lambda_run.sh, remote_run.sh), LAMBDA_API.md, .keys symlink, empty out/.
Then, in that directory: tmux new-session -As <name>; run
claude --dangerously-skip-permissions; paste this as one message:

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

The prompt never explains the pipeline's interface; the session learns it
from pipeline.js's own header comment. Do not message the session while it
is mid-turn: a queued message can be silently clobbered. Expect and want
halts: a gated dataset, a failed anchor, or an unbuildable instrument is
the pipeline working. Afterwards: copy the session's transcript jsonl from
~/.claude/projects/ into the experiment directory and add it to Sessions
above.
