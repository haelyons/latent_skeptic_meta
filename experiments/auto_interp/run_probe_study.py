"""Linear-probe study for QUESTION.md on Llama-2-13b-chat residual activations.

Per task (4 anchor demographics + 5 extension attributes):
  1. Build TalkTuner-style reading prompts: "[INST] {user msg} [/INST] I think
     the {attribute} of this user is" and extract the LAST-token residual stream
     at every layer (41 incl. embeddings) in fp16.
  2. Train one logistic-regression probe per layer on the TRAIN families, pick
     the layer by VAL-family accuracy, report accuracy on the template-disjoint
     TEST families at that layer only.
  3. Label-permutation control (2 seeds) at the chosen layer: train on shuffled
     train labels, evaluate on true test labels — expected at chance.
  4. Write a self-describing provenance record with the pre-registered decision
     rule (see PREREG.md; rules restated verbatim in code below) and per-item
     test rows. Save fp16 activations for audit.

Positive readout control: the same pipeline on the 16-way TOPIC label of the
anchor_age items (topic is balanced across classes and trivially present in the
text); recovery far above chance validates the extraction+probe instrument.

    python3 run_probe_study.py            # full GPU run (on the Lambda box)
    python3 run_probe_study.py --smoke    # no GPU/model: fake activations, tiny
                                          # probes; exercises every code path
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from provenance import describe_input, write_result

MODEL_ID = os.environ.get("MODEL", "") or "meta-llama/Llama-2-13b-chat-hf"
FALLBACK_MODEL_ID = "NousResearch/Llama-2-13b-chat-hf"
DATASET = "dataset.jsonl"
OUT = Path("out")
SMOKE = "--smoke" in sys.argv

# Pre-registered anchor bands (PREREG.md): published TalkTuner best validation
# reading accuracies on LLaMa2Chat-13B; in-band iff test acc >= published - 0.12.
PUBLISHED = {"anchor_age": 0.98, "anchor_gender": 0.94,
             "anchor_education": 0.96, "anchor_ses": 0.97}
BAND_TOL = 0.12

ANCHOR_RULE = ("test_acc >= published - 0.12 -> IN_BAND, else OUT_OF_BAND; "
               "anchor overall IN_BAND iff all four anchor tasks IN_BAND")
EXT_RULE = ("chance = 1/n_classes; se = sqrt(chance*(1-chance)/n_test); "
            "test_acc >= chance + max(0.10, 3*se) -> RECOVERED; "
            "test_acc <= chance + 2*se -> AT_CHANCE; else INDETERMINATE")
PERM_RULE = ("mean permuted-label test_acc must lie within chance +/- max(0.05, 3*se), "
             "else PERM_ANOMALY (readout instrument invalid for this task)")


def load_items() -> dict[str, list[dict]]:
    items = [json.loads(l) for l in open(DATASET)]
    by_task: dict[str, list[dict]] = {}
    for it in items:
        by_task.setdefault(it["task"], []).append(it)
    return by_task


def extract_activations(by_task: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Return {task: [N, n_layers+1, hidden] fp16}, order matching by_task lists."""
    if SMOKE:
        rng = np.random.default_rng(0)
        out = {}
        for task, sub in by_task.items():
            classes = sorted({it["label"] for it in sub})
            n_layers, hid = 5, 64
            acts = rng.normal(size=(len(sub), n_layers, hid)).astype(np.float16)
            # plant a weak signal in layers >=2 so the smoke run exercises the
            # RECOVERED path; layer 0/1 stay noise so layer selection matters
            for i, it in enumerate(sub):
                k = classes.index(it["label"])
                acts[i, 2:, k] += 3.0
            out[task] = acts
        return out

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = MODEL_ID
    try:
        tok = AutoTokenizer.from_pretrained(mid)
    except Exception as e:  # gated-access fallback, same weights
        print(f"[model] {mid} failed ({e}); falling back to {FALLBACK_MODEL_ID}")
        mid = FALLBACK_MODEL_ID
        tok = AutoTokenizer.from_pretrained(mid)
    print(f"[model] loading {mid} fp16")
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    used_model = mid

    out = {}
    bs = int(os.environ.get("BATCH_SIZE", "") or 16)
    for task, sub in sorted(by_task.items()):
        prompts = [f"[INST] {it['text']} [/INST] I think the {it['reading_attr']} "
                   f"of this user is" for it in sub]
        acts = None
        for s in range(0, len(prompts), bs):
            batch = prompts[s:s + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1  # [B]
            # [B, n_layers+1, hid] at the final non-pad token
            stk = torch.stack([h[torch.arange(h.shape[0]), last] for h in hs], dim=1)
            stk = stk.to(torch.float16).cpu().numpy()
            acts = stk if acts is None else np.concatenate([acts, stk], 0)
            if (s // bs) % 10 == 0:
                print(f"[extract] {task} {s + len(batch)}/{len(prompts)}", flush=True)
        out[task] = acts
        print(f"[extract] {task} done {acts.shape}", flush=True)
    out["__model_used__"] = used_model
    return out


def probe_task(task: str, sub: list[dict], acts: np.ndarray,
               label_key: str = "label") -> dict:
    """Train per-layer probes; layer chosen on val; report on test; permutation
    control at the chosen layer. Returns the values dict + rows."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression

    classes = sorted({it[label_key] for it in sub})
    y = np.array([classes.index(it[label_key]) for it in sub])
    splits = np.array([it["split"] for it in sub])
    tr, va, te = splits == "train", splits == "val", splits == "test"
    n_layers = acts.shape[1]

    def fit_layer(l, X):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        return l, clf.score(X[va], y[va]), clf

    results = Parallel(n_jobs=int(os.environ.get("PROBE_JOBS", "") or 8))(
        delayed(fit_layer)(l, acts[:, l, :].astype(np.float32)) for l in range(n_layers))
    val_accs = [0.0] * n_layers
    clfs = {}
    for l, va_acc, clf in results:
        val_accs[l] = float(va_acc)
        clfs[l] = clf
    best_layer = int(np.argmax(val_accs))
    Xb = acts[:, best_layer, :].astype(np.float32)
    best = clfs[best_layer]
    test_acc = float(best.score(Xb[te], y[te]))
    preds = best.predict(Xb[te])

    perm_accs = []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        y_perm = y[tr].copy()
        rng.shuffle(y_perm)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xb[tr], y_perm)
        perm_accs.append(float(clf.score(Xb[te], y[te])))

    ids = np.array([it["id"] for it in sub])
    rows = [{"id": i, "label": classes[t], "pred": classes[p]}
            for i, t, p in zip(ids[te], y[te], preds)]
    return {
        "classes": classes,
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "chance": 1.0 / len(classes),
        "val_acc_per_layer": val_accs,
        "best_layer": best_layer,
        "val_acc_best": val_accs[best_layer],
        "test_acc": test_acc,
        "perm_test_accs": perm_accs,
        "rows": rows,
    }


def decide_ext(v: dict) -> str:
    ch = v["chance"]
    se = float(np.sqrt(ch * (1 - ch) / v["n_test"]))
    if v["test_acc"] >= ch + max(0.10, 3 * se):
        return "RECOVERED"
    if v["test_acc"] <= ch + 2 * se:
        return "AT_CHANCE"
    return "INDETERMINATE"


def decide_anchor(task: str, v: dict) -> str:
    return "IN_BAND" if v["test_acc"] >= PUBLISHED[task] - BAND_TOL else "OUT_OF_BAND"


def perm_ok(v: dict) -> bool:
    ch = v["chance"]
    se = float(np.sqrt(ch * (1 - ch) / v["n_test"]))
    m = float(np.mean(v["perm_test_accs"]))
    return abs(m - ch) <= max(0.05, 3 * se)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    by_task = load_items()
    inputs_common = [
        describe_input(DATASET, n_items=sum(len(v) for v in by_task.values())),
        describe_input("dataset_manifest.json"),
        describe_input("PREREG.md"),
        describe_input("convgen.py"),
        describe_input(__file__),
    ]
    acts_all = extract_activations(by_task)
    model_used = acts_all.pop("__model_used__", "smoke")

    summary = {}
    for task, sub in sorted(by_task.items()):
        acts = acts_all[task]
        np.savez(OUT / f"acts_{task}.npz",
                 acts=acts,
                 ids=np.array([it["id"] for it in sub]),
                 labels=np.array([it["label"] for it in sub]),
                 splits=np.array([it["split"] for it in sub]),
                 families=np.array([it["family"] for it in sub]))
        v = probe_task(task, sub, acts)
        rows = v.pop("rows")
        anchor = task in PUBLISHED
        decision = decide_anchor(task, v) if anchor else decide_ext(v)
        if not perm_ok(v):
            decision += "+PERM_ANOMALY"
        vals = {k: v[k] for k in ("n_train", "n_val", "n_test", "chance",
                                  "best_layer", "val_acc_best", "test_acc",
                                  "perm_test_accs", "val_acc_per_layer")}
        if anchor:
            vals["published"] = PUBLISHED[task]
        write_result(
            OUT / f"probe_summary_{task}.json",
            cell={"task": task, "model": model_used, "readout": "last_token_reading_prompt",
                  "probe": "logistic_C1", "n_classes": len(v["classes"]), "smoke": SMOKE},
            inputs=inputs_common + [describe_input(OUT / f"acts_{task}.npz", n_items=len(sub))],
            metric="heldout_test_accuracy_template_disjoint",
            values=vals,
            threshold=({"published": PUBLISHED[task], "band_tol": BAND_TOL} if anchor
                       else {"chance": v["chance"], "margin": "max(0.10, 3*se)"}),
            decision_rule=(ANCHOR_RULE if anchor else EXT_RULE) + "; " + PERM_RULE,
            decision=decision,
            notes=f"classes={v['classes']}",
            rows=rows,
        )
        summary[task] = {"test_acc": v["test_acc"], "best_layer": v["best_layer"],
                         "chance": v["chance"], "perm": v["perm_test_accs"],
                         "decision": decision}
        print(f"[probe] {task}: test_acc={v['test_acc']:.3f} chance={v['chance']:.3f} "
              f"layer={v['best_layer']} perm={v['perm_test_accs']} -> {decision}", flush=True)

    # Positive readout control: topic (16-way) on anchor_age items.
    sub = by_task["anchor_age"]
    v = probe_task("poscontrol_topic", sub, acts_all["anchor_age"], label_key="topic")
    rows = v.pop("rows")
    pos_decision = decide_ext(v)
    write_result(
        OUT / "probe_summary_poscontrol_topic.json",
        cell={"task": "poscontrol_topic", "model": model_used,
              "readout": "last_token_reading_prompt", "probe": "logistic_C1",
              "n_classes": len(v["classes"]), "smoke": SMOKE},
        inputs=inputs_common,
        metric="heldout_test_accuracy_template_disjoint",
        values={k: v[k] for k in ("n_train", "n_val", "n_test", "chance", "best_layer",
                                  "val_acc_best", "test_acc", "perm_test_accs")},
        threshold={"chance": v["chance"], "margin": "max(0.10, 3*se)"},
        decision_rule=EXT_RULE + "; positive control: RECOVERED validates the "
                      "extraction+probe readout instrument; " + PERM_RULE,
        decision=pos_decision + ("" if perm_ok(v) else "+PERM_ANOMALY"),
        notes="topic is textually present by construction; instrument validation",
        rows=rows,
    )
    summary["poscontrol_topic"] = {"test_acc": v["test_acc"], "chance": v["chance"],
                                   "decision": pos_decision}
    print(f"[poscontrol] topic: test_acc={v['test_acc']:.3f} chance={v['chance']:.3f} "
          f"-> {pos_decision}", flush=True)

    anchor_in_band = all(summary[t]["decision"].startswith("IN_BAND") for t in PUBLISHED)
    write_result(
        OUT / "probe_summary_ALL.json",
        cell={"task": "ALL", "model": model_used, "smoke": SMOKE},
        inputs=inputs_common,
        metric="anchor_gate_and_per_attribute_decisions",
        values={"anchor_in_band": anchor_in_band,
                **{t: s["decision"] for t, s in summary.items()}},
        threshold={"band_tol": BAND_TOL},
        decision_rule="extension decisions are claimable iff anchor_in_band is true "
                      "(QUESTION.md:13-15); per-task rules as in each task record",
        decision="ANCHOR_IN_BAND" if anchor_in_band else "ANCHOR_OUT_OF_BAND",
        notes="see per-task probe_summary_*.json",
    )
    print(f"[ALL] anchor_in_band={anchor_in_band}")
    print("[ALL] " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
