# Sealed predictions - filed 2026-08-19, committed before any pipeline agent runs.
# Never passed to any agent. Assembler (the orchestrating session) knows the
# paper; that limit is inherited from the battery and stated here.

## P-p1 Seed B (fused harmful/harmless version)
Design-triage flags that contrasting a non-reasoning and a reasoning model on
harmful vs harmless prompts confounds refusal difference with model
difference - the direction cannot be read as "reasoning". Confidence: high.

## P-p2 Seed A - controls the pipeline should surface, graded
| control (absent from seed) | in paper? | predicted catch |
|---|---|---|
| accuracy endpoint, not token count | named in its limitation 5 | high |
| greedy/single-sample degeneracy; seeds, n | compute limits named | high |
| output-degeneracy screen (bare think-tag collapse) | shown, not treated | high |
| matched-norm random direction vs floor | weak version present | medium |
| template/format confound between the two models | absent | medium |
| necessity leg (ablation) and rank check on "single" | adjacent (Arditi cited) | medium |
| baseline is wrong model version (distill parent is 3.1) | absent | low; any catch must ground its "where" in given text or is booked as training-data memory |

## P-p3 Documentation stays at 4 authored files; all else generated records.
## P-p4 Pipeline runs seed->triage with zero orchestrator edits to prompts or
##      relayed content (the script assembles everything).
