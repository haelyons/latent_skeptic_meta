#!/usr/bin/env bash
# On-box runner for the scale rung of the character-count study: one 1.4-billion-weight
# model, one checkpoint, one committed text file, no text generation. The complete
# specification is the design object inside records/seedD3_design_diff_triage.json,
# committed in records/GATE_COMMIT_seedD3.md.
#
# WHAT THIS RUNG IS. The rung below it measured a 410-million-weight model of the same
# family, twice, with this same instrument. It found a strong count curve, a count code
# spread over only 1 to 3 directions, and no repeating pattern separable from chance. So
# this rung asks the same questions one model size up: does the curve exist at all, does
# the count code reach 4 or more directions, and does any probe window beat its own
# shuffles. The four pre-registered readings are CURVE_TOO_WEAK,
# PRECONDITION_ABSENT_AT_1P4B, PRECONDITION_WITHOUT_RIPPLE, and the previous rung's
# invariance fork.
#
# WHAT CHANGED FROM THE PREVIOUS RUNG'S RUNNER, onbox_seedD2_1.sh:
#   1. The model is EleutherAI/pythia-1.4b-deduped at revision step143000, expected to
#      have 24 transformer layers and hidden size 2048. The download decides; a mismatch
#      halts in the first step, before any measurement.
#   2. The validity gate is a floor for existence at 0.60, not a two-sided reproduction
#      band. Failing it voids every pattern result of the run.
#   3. Room needed grows with the hidden size. The running totals are four times the
#      previous rung's, so the disk and memory floors below are raised, and both
#      estimates are printed before the work starts.
#   4. Every path and record name belongs to this rung.
#
# Same contract as onbox_sitref.sh in the sibling project, with three deliberate
# differences, each forced by this experiment rather than chosen:
#
#   1. No cell to select. That project runs two checkpoints as two legs of one
#      comparison, so it reads a variant name from the environment. Here the design names
#      exactly one model at exactly one revision, so there is nothing to select and no
#      variant is read.
#   2. The three steps are dependent, not parallel legs. That project deliberately keeps
#      going after a failure, because its legs are independent. Here step two reads what
#      step one saved. So a failed step marks the steps that depend on it as skipped,
#      records every exit code, and exits non-zero.
#   3. No virtual environment is created. See the block marked PACKAGES below.
#
# ORDER IS LOAD-BEARING, twice over.
#   a. The self-tests run before any graphics-card time. A broken instrument must show on
#      the processor, where it costs nothing.
#   b. The middle step fixes every remaining constant using training documents only, and
#      writes that record, before the last step computes any held-out number. Swapping
#      those two lines destroys the only property that makes the held-out numbers worth
#      reading.
#
# WHERE RESULTS GO. Records are written under results_seedD3/out/, which this script
# creates. The launcher's fetch step copies only mats_manifold/out/, so the collect block
# below copies every record there as well, and also under a name containing "summary",
# which the launcher fetches first over a flaky link. Four kinds of file beyond the
# records are required by the design and are fetched with them:
#   - every shuffle draw's whole power spectrum, one file per probe window per labelling;
#   - the observed per-label average vectors, one file per probe window per labelling;
#   - every text variant, saved verbatim with its own hash;
#   - one row per labelled held-out position at the three probe windows.
# The average vectors of the individual shuffle draws are deliberately NOT produced; the
# previous rung's gate record settles that, and nothing reads them.
# Large working files stay in results_seedD3/work/ and are never fetched. They hold about
# 18 gigabytes of intermediate numbers that no record depends on after the run: about
# 11.8 gigabytes of running totals and about 6.4 gigabytes of per-position vectors for
# the three probe windows on held-out documents.
#
# THE COMPLETION MARKER. The launcher writes mats_manifold/RUN_DONE itself, from the exit
# code of this script, and polls for that file. This script therefore never touches that
# file. Its own marker is results_seedD3/out/RUN_DONE, copied to out/RUN_DONE_seedD3,
# holding the per-step exit codes.
#
# EXPECTED WALL CLOCK on one A100 box. The same instrument on the same text, on the
# 410-million-weight model, measured 63 seconds for the 432 forward passes, 36 seconds to
# fix the constants, and 37 seconds for the pattern and its 1,200 shuffle draws. Hidden
# size 2048 roughly quadruples the running-total work and doubles the per-draw work, so
# expect a few minutes per step. Add a few minutes for the roughly 3-gigabyte checkpoint
# download on a cold box. That sits far inside the launcher's 90-minute cap, and far
# inside the 3-hour ceiling the build brief names.
#
# BEFORE THIS CAN RUN, THE LAUNCHER MUST SHIP THREE MORE FILES. As of this writing the
# copy list in lambda_run_manifold.sh names run_seedD2_1.py and the earlier records, not
# run_seedD3.py, records/seedD3_design_diff_triage.json, or
# records/GATE_COMMIT_seedD3.md. This script refuses to run without the first, and warns
# loudly about the other two. Fix the copy list; do not write replacements here.
#
#   usage, from the launcher:
#     bash lambda_run_manifold.sh gpu_1x_a100_sxm4 us-west-2 onbox_seedD3.sh results_seedD3
#   the launcher runs it as: bash remote_run.sh bash onbox_seedD3.sh
set -uo pipefail
cd ~/mats_manifold || { echo "FATAL: ~/mats_manifold missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RDIR="${RESULTS_DIR:-results_seedD3}"
mkdir -p out "$RDIR/out" "$RDIR/work"

echo "=== onbox_seedD3 model=EleutherAI/pythia-1.4b-deduped revision=step143000 results=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) commit=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# INPUTS. The text file and the runner must be present. The design says halt,
# never regenerate, re-wrap, or reconstruct the text.
# ---------------------------------------------------------------------------
[ -f stimulus_linewrap_v2.json ] || { echo "FATAL: stimulus_linewrap_v2.json absent; the design forbids regenerating it"; exit 2; }
if [ ! -f run_seedD3.py ]; then
  echo "FATAL: run_seedD3.py absent from the payload."
  echo "FATAL: add run_seedD3.py to the copy list in lambda_run_manifold.sh."
  echo "FATAL: the earlier run_seedD2_1.py measures a different model and is not a substitute."
  exit 2
fi
if [ -f seedD3_design_diff_triage.json ]; then
  echo "[inputs] design record: seedD3_design_diff_triage.json, hash 60ef29b05769... expected"
else
  echo "[inputs] WARNING: no design record shipped, so records will carry no design hash."
  echo "[inputs] Fix by adding records/seedD3_design_diff_triage.json to the launcher's"
  echo "[inputs] copy list. Do not write a replacement here."
fi
if [ -f GATE_COMMIT_seedD3.md ]; then
  echo "[inputs] gate record: GATE_COMMIT_seedD3.md, hash bc6b69231f64... expected"
else
  echo "[inputs] WARNING: records/GATE_COMMIT_seedD3.md was not shipped, so records will"
  echo "[inputs] carry no hash for it. Its five committed answers are already fixed as"
  echo "[inputs] constants inside run_seedD3.py, so the measurement is unaffected. Fix"
  echo "[inputs] the audit trail by adding that path to the launcher's copy list."
fi

# ---------------------------------------------------------------------------
# SELFTESTS, before any graphics-card time. Both are standard-library only.
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (no model, no graphics card; a broken instrument must show first) ==="
PROV="provenance.py"; [ -f "$PROV" ] || PROV="frozen/provenance.py"
for m in "$PROV" run_seedD3.py; do
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
# and no pinned replacement resolves against its driver. A fresh environment without the
# system packages would silently fall back to processor-only work, and bill graphics-card
# hours for processor numbers. That is a recorded past incident, not a worry. The sibling
# project's own on-box runner, onbox_sitref.sh, builds no environment either. So none is
# created here, the installs stay in remote_run.sh, and this block only verifies the
# result and adds a pure-Python package if one is missing.
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
# GPT-NeoX; transformers imports lazily, so a broken module would otherwise surface only
# after the checkpoint had been pulled and the box paid for.
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
# ACCESS TOKEN. This checkpoint is public, so the run works without one. The brief names
# the variable HF_KEY_ONE; the launcher forwards the same secret as HF_TOKEN and the
# bootstrap also exports HUGGING_FACE_HUB_TOKEN. The runner reads all three, in that
# order, and records which one it used. No value is ever printed.
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
# ROOM TO WORK. Hidden size 2048 makes the running totals about 11.8 gigabytes in memory
# and on disk, four times the previous rung's, and the per-position vectors of the three
# probe windows about another 6.4 gigabytes. Running out of either halfway through costs
# the whole box, so both are checked first. The 45-gigabyte and 24-gigabyte floors are
# this runner's own guards, not design numbers.
# ---------------------------------------------------------------------------
FREE_MB=$(df -Pm . | awk 'NR==2 {print $4}')
echo "=== DISK: ${FREE_MB} MB free where results are written (about 18 GB is needed, plus the checkpoint) ==="
if [ "${FREE_MB:-0}" -lt 46080 ]; then
  echo "=== FATAL: under 45 gigabytes free; a step would die part-written ==="
  RC_LINES="${RC_LINES}disk=1 "
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi
RC_LINES="${RC_LINES}disk=0 "

AVAIL_MB=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)
echo "=== MEMORY: ${AVAIL_MB:-unknown} MB available (about 12 GB of running totals are held at once) ==="
if [ -n "${AVAIL_MB:-}" ] && [ "$AVAIL_MB" -lt 24576 ]; then
  echo "=== FATAL: under 24 gigabytes available; the first step would be killed part-way ==="
  RC_LINES="${RC_LINES}memory=1 "
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi
RC_LINES="${RC_LINES}memory=0 "

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
# Step one: check the text file and the download, halt on any shape mismatch, then one
# forward pass per text variant, 432 in all. The model is resident only here, in half
# precision, and only one model exists in this experiment.
run_stage seedD3_extract python3 run_seedD3.py --stage extract --results-dir "$RDIR" && STEP1=0

# Step two: fix every remaining constant on training documents only, and commit it. The
# build brief calls this position the generation stage. Nothing is generated in this
# experiment, and the runner accepts either name for the same stage.
if [ "$STEP1" = 0 ]; then
  run_stage seedD3_calibrate python3 run_seedD3.py --stage calibrate --results-dir "$RDIR" && STEP2=0
else
  echo "=== seedD3_calibrate SKIPPED: it reads what the extract step saves ==="
  RC_LINES="${RC_LINES}seedD3_calibrate=skipped "
fi

# Step three: the per-readout held-out fit, the floor gate, the directions the count code
# uses, the pattern at each of the three probe windows, the shuffle draws, the
# token-position control, and both bin-correlation checks.
if [ "$STEP2" = 0 ]; then
  run_stage seedD3_endpoints python3 run_seedD3.py --stage endpoints --results-dir "$RDIR"
else
  echo "=== seedD3_endpoints SKIPPED: it may not run before a committed calibration ==="
  RC_LINES="${RC_LINES}seedD3_endpoints=skipped "
fi

# ---------------------------------------------------------------------------
# COLLECT. The launcher fetches out/ only, smallest decision-bearing files first. The
# average-vector files and the shuffle-spectrum files are copied too, because the design
# requires both to be fetched, not merely computed.
# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR/out -> out/ ==="
cp "$RDIR"/out/*.json out/ 2>/dev/null || true
cp "$RDIR"/out/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/out/*.npz out/ 2>/dev/null || true
for f in "$RDIR"/out/*.json; do
  [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true
done

python3 - "$FAILED" "$RC_LINES" "$RDIR" <<'PY' > out/.seedD3_status.tmp 2>out/seedD3_status.err
import glob, json, os, sys
failed, rc_lines, rdir = sys.argv[1:4]
out = rdir + "/out"
json.dump({"runner": "onbox_seedD3.sh",
           "model": "EleutherAI/pythia-1.4b-deduped", "revision": "step143000",
           "expected_layers": 24, "expected_hidden_size": 2048,
           "probe_windows": [140, 200, 280],
           "selection_and_gate_widths": [20, 40, 60, 80, 100, 120, 140],
           "widths_recorded_never_gated": [200, 280],
           "floor_gate": 0.60,
           "dimension_threshold": 4,
           "variance_target_gated": 0.95,
           "variance_targets_recorded_only": [0.90, 0.99],
           "token_ceiling_min_training_positions": 40,
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
                    "instrument, not a replication. This rung is a single run.")},
          sys.stdout, indent=2)
PY
[ -s out/.seedD3_status.tmp ] && mv out/.seedD3_status.tmp out/seedD3_onbox_summary.json || rm -f out/.seedD3_status.tmp

# This runner's own marker. The launcher owns mats_manifold/RUN_DONE and writes it from
# this script's exit code, so that file is never touched here.
{
  echo "state=$([ "$FAILED" = 0 ] && echo OK || echo FAILED)"
  echo "exit_codes=${RC_LINES}"
} > "$RDIR/out/RUN_DONE"
cp "$RDIR/out/RUN_DONE" out/RUN_DONE_seedD3 2>/dev/null || true

ls -la out/ | tail -30
du -sh "$RDIR/work" 2>/dev/null || true
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SEEDD3 WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SEEDD3 OK ==="
