# Gate commit, seed D3.1 (closing rung), started 2026-08-21, committed 2026-08-22

Standing human directive: autonomous, tools frozen, seeds mutable, $100 cap.
Part 1 (stimulus construction) was filed with the seed, before any D3.1 agent
ran. Part 2 (diff answers) is appended at the design gate.

## Part 1. stimulus_linewrap_v3.json construction, inlined

Built 2026-08-21 from the same cached Gutenberg #2701 source
(source_sha256 recorded in the file's meta), by importing the committed
make_stimulus_linewrap_v2.py and replaying its packing twice before packing:

    paras = paragraphs(raw)                       # v2 generator's own functions
    _, c1 = pack_docs(paras, 24, 2400)            # replay seed D1's packing
    v2docs, c2 = pack_docs(paras[c1:], 48, 3600)  # replay seed D2/D2.1/D3's packing
    docs, _  = pack_docs(paras[c1:][c2:], 48, 3600)   # v3: starts after both
    assert every v2 document's first 200 chars absent from the joined v3 text

Same widths [20,40,60,80,100,120,140,200,280], same 36/12 block split, same
wrap rule, same per-width asserts (no overlong line; character stream
round-trips). Result: 48 docs, w140 test lines 320 (mean length 133.9),
w280 test lines 156 (mean length 275.7).
sha256(stimulus_linewrap_v3.json) =
05db25f7a97dd9c3963c7eeb3b8d9738c707a2929a18b2503e56b1f43003aea1

## Part 2. Design gate, 2026-08-22

Commits records/seedD3_1_design_diff_triage.json (budget_usd 15) WITH TWO
AMENDMENTS, struck at the gate as the pipeline's own gate text allows:

A. Arm 10 ("ripple replication leg: look for the original study's roughly
   six-dimensional periodic signal") is STRUCK. The seed defines no such leg;
   nothing in the committed instrument computes one. Design invention.
B. Arm 7's "capped at 40 token positions" is read per the committed ceiling
   rule (seed D3 correction 3): a token count enters the window only with at
   least 40 TRAINING POSITIONS; ceiling, observed maximum, and cut size
   recorded. Not a 40-position cap.

Answers of record to the diff's five questions:
1. Both committed constraints stand together: cycle count at least 2 AND
   period at least 4 in the labelling's own units.
2. The token-position control runs three times, once per window, as in seed D3.
3. A floor-gate quality outside 0.796-0.956 changes no decision; it is
   recorded and must be named in the takeaways. The gate stays the 0.60 floor.
4. The $15 estimate is accepted; the ledger (records/BUDGET.md) governs.
5. No extra follow-up check: two or more clearings enter the fork; INVARIANT
   reopens the sweep as a QUESTION - any sweep would be its own seed with its
   own gate.

Post-build ratification, 2026-08-22: the builder's judgment call is confirmed -
a token-only clearing reads TOKEN_ONLY_PATTERN and never enters the fork, per
the seed's clause binding everything unnamed to seed D3's committed rung.
Local re-verification: 308-check selftest PASS (rc 0), onbox syntax clean,
constants delta limited to the committed changes.
