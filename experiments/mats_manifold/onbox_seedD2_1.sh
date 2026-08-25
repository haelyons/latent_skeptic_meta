#!/usr/bin/env bash
# On-box runner for the corrected probe-width comparison of the character-count curve
# and its wobble. The complete specification is the design object inside
# records/seedD2_1_design_diff_triage.json, committed in records/GATE_COMMIT_seedD2_1.md.
# One public checkpoint of one open model, one committed text file, no text generation.
#
# WHAT THIS IS A CORRECTION OF. An earlier run of this experiment voided itself at its
# own validity gate. Its instrument is reused here with four corrections, each named by
# the committed design: the readout choice and the gate both pool the seven wrap widths
# 20 through 140 only; the wobble is read at that one readout; the measurable-range
# check uses a per-bin correlation with bins above label 175 against bins at or below
# label 140; and the profile length is committed as the label count minus 20.
#
# Same contract as onbox_sitref.sh in the sibling project, with three deliberate
# differences, each forced by this experiment rather than chosen:
#
#   1. No cell to select. That project runs two checkpoints as two legs of one
#      comparison, so it reads a variant name from the environment. Here the design
#      names exactly one model at exactly one revision, so there is nothing to select
#      and no variant is read.
#   2. The three steps are dependent, not parallel legs. That project deliberately
#      keeps going after a failure, because its legs are independent. Here step two
#      reads what step one saved. So a failed step marks the steps that depend on it as
#      skipped, records every exit code, and exits non-zero.
#   3. No virtual environment is created. See the block marked PACKAGES below.
#
# ORDER IS LOAD-BEARING, twice over.
#   a. The self-tests run before any graphics-card time. A broken instrument must show
#      on the processor, where it costs nothing.
#   b. The middle step fixes every remaining constant using training documents only,
#      and writes that record, before the last step computes any held-out number.
#      Swapping those two lines destroys the only property that makes the held-out
#      numbers worth reading.
#
# WHERE RESULTS GO. Records are written under results_seedD2_1/out/, which this script
# creates. The launcher's fetch step copies only mats_manifold/out/, so the collect
# block below copies every record there as well, and also under a name containing
# "summary", which the launcher fetches first over a flaky link. Four kinds of file
# beyond the records are required by the design and are fetched with them:
#   - every shuffle draw's whole power spectrum, one file per probe width per labelling;
#   - the observed per-label average vectors, one file per probe width per labelling;
#   - every text variant, saved verbatim with its own hash;
#   - one row per labelled held-out position at the three probe widths.
# The average vectors of the individual shuffle draws are deliberately NOT produced.
# The gate record's answer 5 settles that: about 230 megabytes per probe width, and
# nothing reads them.
# Large working files stay in results_seedD2_1/work/ and are never fetched. They hold
# about 7 gigabytes of intermediate numbers that no record depends on after the run.
#
# THE COMPLETION MARKER. The launcher writes mats_manifold/RUN_DONE itself, from the
# exit code of this script, and polls for that file. This script therefore never
# touches that file. Its own marker is results_seedD2_1/out/RUN_DONE, copied to
# out/RUN_DONE_seedD2_1, holding the per-step exit codes.
#
# EXPECTED WALL CLOCK on one A100 box. The same instrument on the same text measured
# about 1 minute for the 432 forward passes, about 30 seconds to fix the constants, and
# about 40 seconds for the wobble and its 1,200 shuffle draws. Add a few minutes for the
# checkpoint download on a cold box. That sits far inside the launcher's 90-minute cap,
# and far inside the 3-hour ceiling the build brief names.
#
# BEFORE THIS CAN RUN, THE LAUNCHER MUST SHIP THREE MORE FILES. As of this writing the
# copy list in lambda_run_manifold.sh names run_seedD2.py and the earlier records, not
# run_seedD2_1.py, records/seedD2_1_design_diff_triage.json, or
# records/GATE_COMMIT_seedD2_1.md. This script refuses to run without the first, and
# warns loudly about the other two. Fix the copy list; do not write replacements here.
#
#   usage, from the launcher:
#     bash lambda_run_manifold.sh gpu_1x_a100_sxm4 us-west-2 onbox_seedD2_1.sh results_seedD2_1
#   the launcher runs it as: bash remote_run.sh bash onbox_seedD2_1.sh
set -uo pipefail
cd ~/mats_manifold || { echo "FATAL: ~/mats_manifold missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RDIR="${RESULTS_DIR:-results_seedD2_1}"
mkdir -p out "$RDIR/out" "$RDIR/work"

echo "=== onbox_seedD2_1 model=EleutherAI/pythia-410m-deduped revision=step143000 results=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) commit=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# INPUTS. The text file and the runner must be present. The design says halt,
# never regenerate, re-wrap, or reconstruct the text.
# ---------------------------------------------------------------------------
[ -f stimulus_linewrap_v2.json ] || { echo "FATAL: stimulus_linewrap_v2.json absent; the design forbids regenerating it"; exit 2; }
if [ ! -f run_seedD2_1.py ]; then
  echo "FATAL: run_seedD2_1.py absent from the payload."
  echo "FATAL: add run_seedD2_1.py to the copy list in lambda_run_manifold.sh."
  echo "FATAL: the earlier run_seedD2.py is a DIFFERENT instrument and is not a substitute."
  exit 2
fi
if [ -f seedD2_1_design_diff_triage.json ]; then
  echo "[inputs] design record: seedD2_1_design_diff_triage.json, hash 189f87148c64... expected"
else
  echo "[inputs] WARNING: no design record shipped, so records will carry no design hash."
  echo "[inputs] Fix by adding records/seedD2_1_design_diff_triage.json to the launcher's"
  echo "[inputs] copy list. Do not write a replacement here."
fi
if [ -f GATE_COMMIT_seedD2_1.md ]; then
  echo "[inputs] gate record: GATE_COMMIT_seedD2_1.md, hash 50297fb3110d... expected"
else
  echo "[inputs] WARNING: records/GATE_COMMIT_seedD2_1.md was not shipped, so records will"
  echo "[inputs] carry no hash for it. Its five committed answers are already fixed as"
  echo "[inputs] constants inside run_seedD2_1.py, so the measurement is unaffected. Fix"
  echo "[inputs] the audit trail by adding that path to the launcher's copy list."
fi

# ---------------------------------------------------------------------------
# SELFTESTS, before any graphics-card time. Both are standard-library only.
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (no model, no graphics card; a broken instrument must show first) ==="
PROV="provenance.py"; [ -f "$PROV" ] || PROV="frozen/provenance.py"
for m in "$PROV" run_seedD2_1.py; do
  [ -f "$m" ] || { echo "  $m: ABSENT"; FAILED=1; RC_LINES="${RC_LINES}selftest_$(basename "$m" .py)=absent "; continue; }
  log="out/selftest_$(basename "$m" .py).log"
  python3 "$m" --selftest > "$log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_$(basename "$m" .py)=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "$log")"
  [ "$rc" != 0 ] && FAILED=1
done
if [ "$FAILED" != 0 ]; then
  echo "=== SELFTESTS FAILED -- refusing to spend graphics-card time on a broken instrument ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi

# ---------------------------------------------------------------------------
# PACKAGES. Read this before adding a virtual environment.
#
# The brief that produced this file asks for a virtual environment plus installs. The
# bootstrap script this runner starts inside, remote_run.sh, refuses to build one, and
# says why at length: this machine image ships a working graphics-card build of torch,
# and no pinned replacement resolves against its driver. A fresh environment without
# the system packages would silently fall back to processor-only work, and bill
# graphics-card hours for processor numbers. That is a recorded past incident, not a
# worry. The sibling project's own on-box runner, onbox_sitref.sh, builds no environment
# either. So none is created here, the installs stay in remote_run.sh, and this block
# only verifies the result and adds a pure-Python package if one is missing.
# ---------------------------------------------------------------------------
echo "=== PACKAGES (verify what remote_run.sh left; add only what is missing) ==="
need_install=0
python3 - <<'PY' || need_install=1
import importlib, sys
for m in ("numpy", "sklearn", "transformers"):
    try:
        importlib.import_module(m)
    except Exception as exc:
        print("[packages] missing:", m, exc, file=sys.stderr); sys.exit(1)
PY
if [ "$need_install" != 0 ]; then
  echo "[packages] installing the missing pure-Python packages, torch untouched"
  python3 -m pip install -q --upgrade "numpy==1.26.4" scikit-learn transformers || true
fi
python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("[packages] FATAL: torch unimportable:", exc, file=sys.stderr); sys.exit(3)
if not torch.cuda.is_available():
    print("[packages] FATAL: no graphics card visible to torch. The run would silently use",
          "the processor and bill graphics-card hours for processor numbers.",
          "torch:", torch.__version__, file=sys.stderr)
    sys.exit(3)
import numpy, sklearn, transformers
# Resolve the concrete model class now, not inside the download. This model family is
# GPT-NeoX; transformers imports lazily, so a broken module would otherwise surface
# only after the checkpoint had been pulled and the box paid for.
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM  # noqa: F401
print("[packages] torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0))
print("[packages] transformers", transformers.__version__, "numpy", numpy.__version__,
      "scikit-learn", sklearn.__version__, "GPTNeoXForCausalLM resolvable")
PY
rc=$?; RC_LINES="${RC_LINES}packages=${rc} "
if [ "$rc" != 0 ]; then
  echo "=== PACKAGE CHECK FAILED rc=$rc -- no measurement attempted ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi

# ---------------------------------------------------------------------------
# ACCESS TOKEN. This checkpoint is public, so the run works without one. The brief
# names the variable HF_KEY_ONE; the launcher forwards the same secret as HF_TOKEN and
# the bootstrap also exports HUGGING_FACE_HUB_TOKEN. The runner reads all three, in
# that order, and records which one it used. No value is ever printed.
# ---------------------------------------------------------------------------
if [ -n "${HF_KEY_ONE:-}" ]; then
  echo "[token] HF_KEY_ONE is set"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HF_KEY_ONE="$HF_TOKEN"
  echo "[token] HF_KEY_ONE taken from the forwarded HF_TOKEN"
else
  echo "[token] no access token in the environment; this checkpoint is public, so the run continues"
fi

# ---------------------------------------------------------------------------
# DISK. The first step writes about 3 gigabytes of running totals and about 3.2
# gigabytes of per-position vectors into the working folder. Running out of room
# halfway through costs the whole box, so the room is checked first. The 20-gigabyte
# floor is this runner's own guard, not a design number.
# ---------------------------------------------------------------------------
FREE_MB=$(df -Pm . | awk 'NR==2 {print $4}')
echo "=== DISK: ${FREE_MB} MB free where results are written ==="
if [ "${FREE_MB:-0}" -lt 20480 ]; then
  echo "=== FATAL: under 20 gigabytes free; the first step would die part-written ==="
  RC_LINES="${RC_LINES}disk=1 "
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi
RC_LINES="${RC_LINES}disk=0 "

# ---------------------------------------------------------------------------
# THE THREE STEPS, in order. Each gets its own log and its own exit code.
# ---------------------------------------------------------------------------
run_stage(){                    # run_stage <name> <cmd...>
  local name="$1"; shift
  local log="out/${name}.log"
  echo "=== $name RUNNING ==="
  "$@" > "$log" 2>&1
  local rc=$?
  echo "--- $name tail ---"; tail -15 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="; return 1
  else echo "=== $name OK ==="; return 0; fi
}

STEP1=1; STEP2=1
# Step one: check the text file and the download, then one forward pass per text
# variant, 432 in all. The model is resident only here, in half precision, and only one
# model exists in this experiment.
run_stage seedD2_1_extract python3 run_seedD2_1.py --stage extract --results-dir "$RDIR" && STEP1=0

# Step two: fix every remaining constant on training documents only, and commit it. The
# build brief calls this position the generation stage. Nothing is generated in this
# experiment, and the runner accepts either name for the same stage.
if [ "$STEP1" = 0 ]; then
  run_stage seedD2_1_calibrate python3 run_seedD2_1.py --stage calibrate --results-dir "$RDIR" && STEP2=0
else
  echo "=== seedD2_1_calibrate SKIPPED: it reads what the extract step saves ==="
  RC_LINES="${RC_LINES}seedD2_1_calibrate=skipped "
fi

# Step three: the per-readout held-out fit, the validity gate, the wobble at each of the
# three probe widths, the shuffle draws, the token-position control, and the
# measurable-range check.
if [ "$STEP2" = 0 ]; then
  run_stage seedD2_1_endpoints python3 run_seedD2_1.py --stage endpoints --results-dir "$RDIR"
else
  echo "=== seedD2_1_endpoints SKIPPED: it may not run before a committed calibration ==="
  RC_LINES="${RC_LINES}seedD2_1_endpoints=skipped "
fi

# ---------------------------------------------------------------------------
# COLLECT. The launcher fetches out/ only, smallest decision-bearing files first.
# The average-vector files and the shuffle-spectrum files are copied too, because
# the design requires both to be fetched, not merely computed.
# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR/out -> out/ ==="
cp "$RDIR"/out/*.json out/ 2>/dev/null || true
cp "$RDIR"/out/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/out/*.npz out/ 2>/dev/null || true
for f in "$RDIR"/out/*.json; do
  [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true
done

python3 - "$FAILED" "$RC_LINES" "$RDIR" <<'PY' > out/.seedD2_1_status.tmp 2>out/seedD2_1_status.err
import glob, json, os, sys
failed, rc_lines, rdir = sys.argv[1:4]
out = rdir + "/out"
json.dump({"runner": "onbox_seedD2_1.sh",
           "model": "EleutherAI/pythia-410m-deduped", "revision": "step143000",
           "probe_widths": [140, 200, 280],
           "selection_and_gate_widths": [20, 40, 60, 80, 100, 120, 140],
           "widths_recorded_never_gated": [200, 280],
           "generation_performed": False,
           "per_draw_mean_vectors_saved": False,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "records_present": sorted(os.path.basename(p) for p in glob.glob(out + "/*.json")),
           "spectra_files": sorted(os.path.basename(p)
                                   for p in glob.glob(out + "/null_spectra_*.jsonl")),
           "mean_vector_files": sorted(os.path.basename(p)
                                       for p in glob.glob(out + "/mean_vectors_*.npz")),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": ("A further run of this script on the same box is a rerun of the same "
                    "instrument, not a replication.")},
          sys.stdout, indent=2)
PY
[ -s out/.seedD2_1_status.tmp ] && mv out/.seedD2_1_status.tmp out/seedD2_1_onbox_summary.json || rm -f out/.seedD2_1_status.tmp

# This runner's own marker. The launcher owns mats_manifold/RUN_DONE and writes it from
# this script's exit code, so that file is never touched here.
{
  echo "state=$([ "$FAILED" = 0 ] && echo OK || echo FAILED)"
  echo "exit_codes=${RC_LINES}"
} > "$RDIR/out/RUN_DONE"
cp "$RDIR/out/RUN_DONE" out/RUN_DONE_seedD2_1 2>/dev/null || true

ls -la out/ | tail -30
du -sh "$RDIR/work" 2>/dev/null || true
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SEEDD2_1 WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SEEDD2_1 OK ==="
