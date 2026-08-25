# Deviations from mats_r1d1 machinery, seed D (mats_manifold), 2026-08-21

Filed before any agent ran. Everything not listed here is byte-identical to its
mats_r1d1 source (which documents its own deviations from latent_verify /
welfare_probe in place).

1. `pipeline.js`, DESIGN_PROMPT, one sentence. Was: "Assume two ~8B models on
   one rented GPU box; keep budget_usd honest." Now: "Assume one rented GPU
   box; take model identity, scale, and any checkpoint axis only from the
   QUESTION; the readout here is a fitted probe, not a single forward pass -
   cost it as such. Keep budget_usd honest." Why: the old sentence was a
   project fact of mats_r1d1 (two 8B chat models). Kept as-is it would be an
   unmarked false claim injected by the pipeline itself, making the design
   agent mis-cost the run. The replacement asserts no new substrate fact; the
   substrate lives in the seed.

2. `pipeline.js`, orchestrator comment: launcher name updated to
   `lambda_run_manifold.sh`. Comment only.

3. `lambda_run_manifold.sh` (from `lambda_run_mats.sh`): remote dir
   mats_r1d1/ -> mats_manifold/; scp payload -> seed D files; header note.
   Documented in the script's own header. Billing-safety mechanisms untouched.

4. `frozen/remote_run.sh`: `cd ~/mats_r1d1` -> `cd ~/mats_manifold`, the one
   line its header already declares as the adaptation point.

5. New committed inputs, produced before any agent ran: `stimulus_linewrap.json`
   (sha256 1fe73074a3d127e0bc546fb05a5adf9c2f0715f035b7f6fa7ac546afeaefe8ca)
   and its generator `make_stimulus_linewrap.py` (selftest PASS). The seed
   embeds the file hash and forbids reconstruction.

Known departure of the seed from its source proposal
(counting_manifold_formation_proposal.md), named so it cannot be silent: the
proposal's ripple measurement is a DFT of "the probe cosine-similarity
matrix"; the seed's gating version (definition 3) uses per-label mean
activations instead, with the fitted-probe class-vector version demoted to a
recorded, non-gating secondary (definition 5). Reason: the mean-activation
version has a cheap, exact permutation null (definition 4); refitting a
140-class probe 200 times does not fit the 90-minute on-box cap. The human
reads this at the budget gate and may reverse it.

## Amendment v2, 2026-08-21, after the gate press (human-directed, pre-execute)

Correction to the paragraph above: verification against the paper (arXiv
2601.04480 fetch, records/PREDICTIONS_gatepress_seedD.md R5) showed ringing IS
shown in cosine-similarity matrices of per-count mean activations, alongside
probes and feature decoders. Definition 3 is paper-faithful, not a stand-in;
the "deliberate substitution" framing above was overcautious and the v2 seed
drops that apology.

seed_d_ripple_v2.md changes (871 -> 899 words; deletions outweighed by the calibration rule, the layer convention, and the null-cutoff pin - not net subtractive, arithmetic honest):
- DELETED fixed statistic constants (moving-average window 21, search band
  4-35 characters, delta range 1-70, 200 permutations): the derived sidelobe
  spacing (43-50+ characters) showed band and detrend suppressed exactly the
  predicted periods. Replaced by one calibration-stage rule (definition 5):
  train-documents-only, committed as its own record before any test-document
  number exists (H1/H2 compliant).
- DELETED the "best layer not among the first two" gate clause: layer indexing
  is a convention, the reproduction's is unpublished, no measurement settles
  it. Layer location is now recorded, never gated; our convention fixed
  (embedding output = index 0).
- DELETED definition-5-as-apology; secondary probe-geometry check kept as one
  line.
- ADDED one policy line pinning the null cutoff (top 5 percent, GUESS) so the
  design cannot re-invent it, and the theory-derived period expectation with
  its derivation marked.

## Pipeline observation, post phase, 2026-08-21 (no code changed)

pipeline.js runs Design/Diff/Triage unconditionally before branching on mode,
and the budget gate sits before the branch. In post mode the design is a
throwaway (the build and post branches bind to A.design_record), yet its
guessed budget still faces the cap: the first post invocation halted
"budget_over_cap" because the stub-seed design guessed $20 against the $15
default. Resumed with cap_usd=25 for the post invocation only - post mode
spends no GPU money; the executed design remains the committed $10 (actual
box cost about $0.43, 13 minutes). Logged because two earlier invocations
passed the same gate by luck ($8, $10): the structural fix, if ever wanted,
is moving the mode branch above the gate, which is a pipeline.js edit for the
pipeline's owner, not for this run.

## Writer contract slip, seed D2.1 post, 2026-08-21

TAKEAWAYS_seedD2_1.md was written without the required verbatim gate-output
block. Both gates pass (density 0.0 codes/1000 words; 0 unscoped sentences);
the orchestrator ran them on the unmodified file and appended the block with
attribution. Word counts in the appended block are from the pre-append file.
