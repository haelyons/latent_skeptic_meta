# Budget ledger, mats_manifold (cap: $100 total, set by the human 2026-08-21)
| when | what | est | actual |
|---|---|---|---|
| 2026-08-21 | seed D1 box (A100 sxm4, 13 min, instance 9e34393a) | $0.70 | $0.43 |
| 2026-08-21 | seed D2 box (planned: one A100-class box, 432 forward passes + 3-aperture ripple; under the 90-min cap) | $1-4 | - |
Committed spend so far: $0.43. Remaining headroom: ~$99.6.
Design-mode cap_usd passed as 30 (pipeline arg, not a code change); the executed
design's own budget_usd governs the actual run.

Note 2026-08-21: D2 design-mode first pass halted budget_over_cap (design
budget_usd 40 vs cap_usd 30). Resumed same run with cap_usd 60 - same cached
design, gate passed. The $40 is the design agent's conservative estimate;
actuals booked here after the box run.
| 2026-08-21 | seed D2 box (A100 sxm4, instance 80aa729f, ~15-20 min) | $1-4 | ~$0.55 |
| 2026-08-21 | seed D2.1 box (A100 sxm4, instance 47f310a6) | $1 | ~$0.6 |
Total actuals to date: ~$1.6 of $100.
| 2026-08-21 | seed D3 box (A100 sxm4, instance e39c47ef, pythia-1.4b) | $2 | ~$1.0 |
Total actuals to date: ~$2.6 of $100.
| 2026-08-22 | seed D3.1 box (A100 sxm4, instance df20b119) | $1.5 | ~$1.0 |
Total actuals to date: ~$3.6 of $100.
| 2026-08-22 | seed D3.1 box + seed D4 box (instances df20b119, d2fa0a08) | $2 | ~$1.3 |
Total actuals to date: ~$4.9 of $100.
| 2026-08-22 | seed D4.1 box (A100 sxm4, instance 7e92f510, 12 ckpts 1.4b, ~100 min) | $3.5 | ~$3.4 |
Total actuals to date: ~$8.3 of $100.
