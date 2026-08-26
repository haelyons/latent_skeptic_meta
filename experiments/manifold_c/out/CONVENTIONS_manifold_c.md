# On-box conventions and the provenance writer (experiment directory `manifold_c`)

This file is the second binding input to the build stage. The design record is
the specification of WHAT to measure. This file is the specification of HOW the
two built files must sit inside the frozen infrastructure that will run them.
Nothing here may be "improved". Where this file and the design record disagree
about a number, the design record wins; where they disagree about a path, a
marker, or a file layout, this file wins.

## 1. The two files you write

    run_<TAG>.py      the measurement, stages `extract`, `calibrate`, `endpoints`
    onbox_<TAG>.sh    the on-box driver that runs those stages in order

`<TAG>` is given to you in the build call. Write these two files in the
experiment directory (the current working directory) and edit nothing that
already exists. In particular `pipeline.js`, `triage_workflow.js`, `gates/`,
`frozen/` and `.claude/agents/` are frozen and read-only.

## 2. The frozen infrastructure you are running inside

Three frozen files, in `frozen/`, are shipped to the rented box unchanged:

  * `frozen/provenance.py`   the record writer (section 4)
  * `frozen/remote_run.sh`   the box bootstrap. It does `cd ~/mats_manifold`,
    installs `pillow jinja2 numpy==1.26.4 scikit-learn transformers accelerate`
    on top of the image's preinstalled CUDA torch, asserts
    `torch.cuda.is_available()` and exits 3 if it is False, then execs the
    command it was handed. It creates no virtual environment. Do not create one
    either, and do not install or pin torch.
  * `frozen/lambda_run.sh`   the local launcher. A per-launch copy of it, with
    only its declared adaptation points changed, does: launch -> poll -> scp the
    payload -> run `bash remote_run.sh bash onbox_<TAG>.sh` detached under
    `setsid` and `timeout` -> poll for a marker -> fetch -> terminate.

Consequences you must respect, because the launcher is frozen:

  * The on-box working directory is `~/mats_manifold`. Your shell script must
    `cd ~/mats_manifold` and must halt loudly if it is missing.
  * The launcher writes `~/mats_manifold/RUN_DONE` itself, from your script's
    exit code, and polls for that file. Your script must NEVER create or remove
    `RUN_DONE` in that directory. Use a differently named marker of your own.
  * The launcher fetches, in this order: `~/mats_manifold/out/*summary*.json`,
    then `~/mats_manifold/out/*.log`, then the whole of `~/mats_manifold/out/`.
    Anything not under `~/mats_manifold/out/` is never fetched and is lost when
    the box is torn down.
  * The launcher tails `~/mats_manifold/out/run_detached.log` for progress, so
    print short one-line progress messages and flush them.
  * The run is capped by `timeout $REMOTE_TIMEOUT` (default 5400 seconds) and
    the box self-destructs at cap plus grace. Design for well under one hour.

## 3. Where results go

Write records under `results_<TAG>/out/` and large intermediates under
`results_<TAG>/work/`. Then copy every fetchable artifact into
`~/mats_manifold/out/`, because that is the only directory the launcher takes.
`work/` is deliberately not copied: it is gigabytes of intermediate arrays that
no record depends on after the run.

Give the single decision-bearing roll-up record a filename containing the word
`summary`, so the launcher fetches it first over a flaky link. The provenance
writer already emits a `<name>_summary.json` beside every record; that is what
satisfies this.

The local result directory is passed to the launcher as its fourth argument and
will be `out/results_<TAG>`; the fetch lands the box's `out/` inside it, giving
`out/results_<TAG>/out/...` locally. Do not hardcode any local path in the
runner.

## 4. The provenance writer: one home per fact

`frozen/provenance.py` is imported as `import provenance` (the launcher ships it
flat into `~/mats_manifold/`). Use exactly these entry points:

    provenance.describe_input(path, n_items=<int>) -> dict
    provenance.write_result(out_path, *, cell, inputs, metric, values,
                            threshold=None, decision_rule="", decision="",
                            notes="", rows=None) -> Path
    provenance.cell_slug(cell) -> str

Rules the writer enforces, and which you must therefore satisfy:

  * `cell` must be flat and JSON-scalar valued. It is the identity of the run.
    The convention in this project's prior seeds is
    `{"arm": ..., "model": ..., "revision": ..., "stage": ...}`, plus
    `"aperture": <int>` on any record that belongs to one probe width.
  * `decision_rule` is REQUIRED and is an argument written before the values are
    known. It states the neutral threshold in terms of the measurement only.
  * The filename must be `cell_slug(cell) + ".json"`, so the path and the
    contents cannot silently disagree.
  * `inputs` must list every file the result was computed from, each through
    `describe_input`, with its item count. This includes the stimulus file, the
    committed design record, the committed gate commit, and any intermediate
    array file the stage read.

Run `python3 provenance.py --selftest` on the box before any measurement, and
fail the run if it exits non-zero.

## 5. Stimulus file

`stimulus_linewrap_v2.json`, sha256
`b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd`. Shape:

    {"meta": {"widths": [20,40,60,80,100,120,140,200,280], "n_docs": 48,
              "train_docs": 36, ...},
     "docs": [{"id": 0, "split": "train"|"test", "chars": 3595,
               "sha256_plain": "...", "wrapped": {"20": "...", ..., "280": "..."}},
              ...]}

Load it exactly. Never regenerate, re-wrap, paraphrase or reconstruct the text.
If the file is absent, or its hash differs, halt with a non-zero exit before any
graphics-card time. The document split is the file's own `split` field; do not
re-derive it.

## 6. Model and checkpoint

`EleutherAI/pythia-410m-deduped` at HuggingFace revision `step143000`, loaded
with `output_hidden_states=True`. The token `HF_TOKEN` is exported into the run
by the launcher. Read the DOWNLOADED config for the layer count and hidden size
and halt if they disagree with the design's stated values: the config is
authoritative, the design's numbers are recalled and are only a cross-check.
Hidden state index 0 is the embedding output, so a model with `n` blocks yields
`n + 1` readouts.

## 7. Halt discipline

  * The stages are dependent, not parallel legs. A failed stage marks every
    stage that depends on it as skipped, records every exit code, and exits
    non-zero.
  * The calibration stage must write its committed record, from training
    documents only, BEFORE the endpoint stage computes any held-out number.
    That ordering is the only property that makes the held-out numbers worth
    reading. Do not swap those two lines.
  * Broken infrastructure is a halt, never a patch. If a frozen file is missing
    or a convention is ambiguous, stop and report rather than writing a
    replacement.
  * A void is contagious. Any gate or anchor endpoint that fails must write
    `run_valid=false` into EVERY record written after it, per-aperture files
    included, not only into the roll-up. A void that lives in one file is not a
    void: the prior asking of this question wrote three confident per-aperture
    records with no void flag on them, and only the roll-up said the run was
    void.
  * Never describe a second run of the same script on the same box as a
    replication. It is a rerun of the same instrument.

## 8. Self-test

`run_<TAG>.py --selftest` must be standard-library only, so guard `numpy`,
`torch`, `transformers` and `sklearn` behind the stage entry points rather than
importing them at module scope. It must cover every parser, every
degenerate-output detector, and each endpoint's arithmetic on planted rows with
known decisions. It must include a planted gate-failure row that has to
propagate `run_valid=false`, and, because this design contains a dispersion
rule, a planted zero-dispersion case that has to halt `WRONG_VARIANCE_SOURCE`.
Non-zero exit on any self-test failure. The self-tests run on the processor,
before any graphics-card time, both locally and again on the box.

## 9. Naming and language inside artifacts

Record fields and log lines are read later by an agent that has never seen this
project. Prefer plain words in `metric`, `decision_rule` and `notes`. Numbers
stay exact. Keep the code names for `cell` fields and file names only.
