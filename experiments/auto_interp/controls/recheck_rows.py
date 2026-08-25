#!/usr/bin/env python3
"""Read-only control: re-derive held-out accuracy from the per-item rows of a
probe_summary json and compare to its recorded values.test_acc.
Decision rule (neutral, stated here): |rows_acc - recorded| < 1e-9 -> MATCH,
else MISMATCH. Prints one line; writes nothing."""
import json, sys
r = json.load(open(sys.argv[1]))
rows = r["rows"]
acc = sum(x["label"] == x["pred"] for x in rows) / len(rows)
rec = r["values"]["test_acc"]
verdict = "MATCH" if abs(acc - rec) < 1e-9 else "MISMATCH"
print(f"{r['cell']['task']} rows_acc={acc:.6f} recorded={rec:.6f} n={len(rows)} -> {verdict}")
