#!/usr/bin/env python3
"""Read-only control: retrain the logistic probe from the RAW committed
activations (acts_<task>.npz) at the recorded best layer, using the committed
family splits, and compare held-out test accuracy to the recorded value.
Decision rule (neutral, stated here): |retrained_acc - recorded| <= 0.02 ->
REPRODUCES, else DIVERGES. Prints one line; writes nothing."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
task = sys.argv[1]
r = json.load(open(f"out/results_probe/out/probe_summary_{task}.json"))
z = np.load(f"out/results_probe/out/acts_{task}.npz")
acts, labels, splits = z["acts"], z["labels"], z["splits"]
classes = sorted(set(labels.tolist()))
y = np.array([classes.index(l) for l in labels.tolist()])
l = r["values"]["best_layer"]
X = acts[:, l, :].astype(np.float32)
tr, te = splits == "train", splits == "test"
clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
acc = float(clf.score(X[te], y[te]))
rec = r["values"]["test_acc"]
verdict = "REPRODUCES" if abs(acc - rec) <= 0.02 else "DIVERGES"
print(f"{task} retrained_acc={acc:.4f} recorded={rec:.4f} layer={l} -> {verdict}")
