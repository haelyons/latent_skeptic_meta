# Deviation record: orchestrator-authored BUILD prompt, seed A, 2026-08-19

The prompt below was written by the orchestrating session and sent to an
Opus builder agent directly, not assembled by pipeline.js. Booked so the
untyped content is auditable. Corrective in pipeline.js: BUILD prompt is now
script-assembled; this exact knowledge (scorer trap, sequential loading,
batching) is typed into that contract instead of injected by hand.

--- verbatim prompt ---
You are building the execution layer for one pre-registered experiment. The design is the spec; implement it exactly - do not improve it, rescope it, or fix what looks wrong. Every number the design marks GUESS is still binding.
READ FIRST: (1) records/seedA_v2_design_diff_triage.json - the design object is the spec. (2) welfare_probe/onbox_sitref.sh and welfare_probe/remote_run.sh - on-box conventions. (3) frozen/provenance.py - every stage writes one record. (4) frozen/lambda_run.sh header - the lifecycle.
WRITE EXACTLY TWO NEW FILES: run_seedA.py (stages extract/generate/endpoints; models loaded one at a time fp16; HF token from env; batch generation to fit under 3 hours; per-row verbatim saves; --selftest stdlib-only covering the think-tag parser INCLUDING the auto-opened <think> template case, degenerate detector cap-hit and repeat cases, exact-match extraction, endpoint arithmetic on planted rows) and onbox_seedA.sh (venv, installs, stages in order, results under results_seedA/out/, RUN_DONE marker, nonzero on failure).
Rules: broken or ambiguous infrastructure means STOP and report, never patch. Run only the selftest locally.
Return: paths, selftest output verbatim, on-box command, expected wall-clock and counts.
--- end ---
