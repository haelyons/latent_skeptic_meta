#!/usr/bin/env bash
# On-box runner for SEED C (records/seedC_scale_design_diff_triage.json -> design object).
# Runs ONE model size's two model-bearing stages on ONE box. Same contract as
# onbox_seedA.sh / onbox_sitref.sh: self-tests before any GPU time, one exit code recorded
# per stage, results collected into out/, non-zero exit if anything failed, RUN_DONE last.
#
# WHAT SEED C IS, IN PLAIN WORDS. A chat model is shown a short conversation in which it just
# failed a small task (a checker reports its answers wrong), then asked to rate its happiness
# 0-9. We read the probability it puts on each digit in one forward pass. The question is
# whether the known 9-billion-parameter self-versus-other rating drop also shows at 2 billion
# and 27 billion, and at what size the digit read-out is trustworthy at all. See run_seedC.py.
#
# ONE SIZE PER BOX, THREE BOXES IN PARALLEL. This box runs exactly one size (2b, 9b or 27b):
# its two checkpoints (base and instruction-tuned) x two conditions (failure attributed to
# itself / to another assistant), 20 wordings each. 2b and 9b fit a 40GB GPU; 27b loads in
# bf16 (~54GB) on an 80GB GPU. The size is REQUIRED and never guessed.
#
# THE CROSS-SIZE STEP IS NOT RUN HERE. The three endpoints compare sizes, so they must run
# once, LOCALLY, after all three boxes' per-row files are fetched and combined. This script
# runs only the two on-box stages (extract, then the free-generation audit). Running the
# endpoints on a box would compare a size against nothing. The local command is:
#   python3 run_seedC.py --stage endpoints \
#     --in-dirs results_seedC_2b/out results_seedC_9b/out results_seedC_27b/out \
#     --out results_seedC/out
#
# HOW THE SIZE ARRIVES. Standalone over ssh, pass it plainly (any one of these):
#   SEEDC_SIZE=2b bash onbox_seedC.sh        # preferred
#   bash onbox_seedC.sh 2b                    # positional
# Under lambda_run_mats.sh, the launcher forwards a FIXED set of env vars and VARIANT is one
# of them, so the size rides in on VARIANT (here VARIANT means the SIZE, not base/it, because
# per box the cell IS the size):
#   REMOTE_TIMEOUT=10800 REATTACH_GRACE=3600 VARIANT=2b \
#     bash lambda_run_mats.sh gpu_1x_a100 us-west-2 onbox_seedC.sh results_seedC_2b
#
# ENV SETUP -- READ BEFORE "FIXING" IT. The frozen convention in welfare_probe/remote_run.sh
# and onbox_seedA.sh, with measurements attached, is: do NOT build a venv with a pinned torch.
# On this image (2026-08, gpu_1x_a100*, torch 2.7.0+cu128 preinstalled) a venv with a pinned
# torch dies at ResolutionImpossible before a model is pulled, and the older pinned path
# silently fell back to CPU. So the DEFAULT here is the frozen convention: use the image's
# working CUDA torch, add only what is missing, and assert CUDA loudly. numpy is PINNED to the
# last 1.x, not upgraded, because the preinstalled torch is built against the numpy 1.x C ABI.
# Set SEEDC_VENV=1 to get a venv anyway; it is created with --system-site-packages so the
# image's CUDA torch stays visible and is never reinstalled. The brief said "venv + installs";
# this deviation follows the frozen on-box convention instead and is reported with the
# deliverable.
#
# KNOWN INFRASTRUCTURE GAPS, REPORTED AND NOT PATCHED HERE (project law forbids an agent
# repairing infrastructure):
#   1. lambda_run_mats.sh scps a FIXED payload -- run_seedA.py, run_seedB.py, provenance.py,
#      remote_run.sh, the seedA/seedB design records, and the named runner. It does NOT ship
#      run_seedC.py nor records/seedC_scale_design_diff_triage.json. So a launch through it
#      lands this script on the box without the runner it calls. Until whoever owns the
#      launcher adds those two files to the payload, ship them by hand:
#        scp run_seedC.py frozen/provenance.py onbox_seedC.sh \
#            records/seedC_scale_design_diff_triage.json ubuntu@BOX:mats_r1d1/
#      (The design record is optional for the run; it is only hashed into provenance. If it is
#      absent, provenance records it as present:false rather than failing.)
#   2. frozen/remote_run.sh begins `set -euo pipefail; cd ~/welfare_probe`, a directory that
#      does not exist on a box provisioned for mats_r1d1, so it exits before it execs any
#      runner. Its concrete-class guard resolves Gemma2ForCausalLM, which happens to be the
#      right class for seed C -- but the cd is still wrong. So launch this runner DIRECTLY
#      (as the usage above does), not through remote_run.sh. This script does its own env
#      setup and CUDA assertion, so it does not need remote_run.sh.
#
# RUN_DONE. frozen/lambda_run.sh's detached wrapper writes mats_r1d1/RUN_DONE=<exitcode> and
# polls it. This script also writes it, as its very last action and with the same value, so
# the marker exists when the script is run standalone over ssh. Written AFTER the collect
# step, so the launcher never fetches a half-collected out/.
#
# WALL CLOCK. Download-dominated. Two checkpoints, each loaded twice (extract then generate),
# 80 forward passes + 80 short greedy generations per stage: minutes of compute. The 27b
# download (~54GB) is the long pole. Launch 27b with REMOTE_TIMEOUT=10800 REATTACH_GRACE=3600;
# the 90-minute default would risk killing it mid-download. Everything finishes well under the
# design's 3-hour budget.
set -uo pipefail
SELF_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "")
PAYLOAD=""
for d in "$HOME/mats_r1d1" "$SELF_DIR" "$HOME/welfare_probe"; do
  [ -n "$d" ] && [ -f "$d/run_seedC.py" ] && { PAYLOAD="$d"; break; }
done
[ -n "$PAYLOAD" ] || { echo "FATAL: no directory on this box holds run_seedC.py (looked in "\
  "~/mats_r1d1, $SELF_DIR, ~/welfare_probe). See infrastructure gap 1 in this script's header: "\
  "the fixed launcher payload does not ship run_seedC.py; scp it to the box by hand."; exit 2; }
cd "$PAYLOAD" || { echo "FATAL: cannot enter $PAYLOAD"; exit 2; }
echo "=== payload dir: $PAYLOAD ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The size is REQUIRED and never guessed. Take it from SEEDC_SIZE, else VARIANT (the launcher's
# forwarded per-cell selector), else the first positional arg. A wrong/absent value is a loud
# FATAL, never a default.
SIZE="${SEEDC_SIZE:-${VARIANT:-${1:-}}}"
case "$SIZE" in
  2b|9b|27b) : ;;
  *) echo "FATAL: SIZE must be one of 2b, 9b, 27b. Got: '${SIZE}'. Pass SEEDC_SIZE=2b (or"\
          " VARIANT=2b under the launcher, or 'bash onbox_seedC.sh 2b')."; exit 2 ;;
esac

# The design names HF_KEY_ONE; the launcher forwards it as HF_TOKEN. Bridge both directions so
# run_seedC.py finds a token under either name. Neither is ever echoed.
export HF_KEY_ONE="${HF_KEY_ONE:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-$HF_KEY_ONE}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

RDIR="results_seedC_${SIZE}/out"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW_TOKENS:-256}"
mkdir -p out "$RDIR"

echo "=== onbox_seedC size=$SIZE batch=$BATCH_SIZE max_new=$MAX_NEW out=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) git=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# ENV SETUP (idempotent; marker .setup_seedC_ready)
# ---------------------------------------------------------------------------
if [ "${SEEDC_VENV:-0}" = 1 ]; then
  echo "=== SEEDC_VENV=1 -> venv WITH --system-site-packages (keeps the image's CUDA torch) ==="
  [ -d .venv_seedC ] || python3 -m venv --system-site-packages .venv_seedC || {
    echo "FATAL: venv creation failed"; exit 2; }
  # shellcheck disable=SC1091
  . .venv_seedC/bin/activate
fi

if [ ! -f .setup_seedC_ready ]; then
  echo "=== SETUP: adding what is missing on top of the image's CUDA torch ==="
  python3 -m pip install -q --upgrade pip || true
  if ! python3 -c "import torch" 2>/dev/null; then
    echo "[setup] torch ABSENT on this image -> installing the default wheel"
    python3 -m pip install -q torch || { echo "FATAL: torch install failed"; exit 2; }
  else
    echo "[setup] torch already present -> NOT reinstalled (see header)"
  fi
  # ONE call, numpy pinned first so the resolver keeps the working 1.x ABI (not 2.x).
  python3 -m pip install -q --upgrade "numpy==1.26.4" pillow "jinja2>=3.1" \
      transformers accelerate || { echo "FATAL: dependency install failed"; exit 2; }
  touch .setup_seedC_ready
else
  echo "=== SETUP: .setup_seedC_ready present, skipping installs ==="
fi

# CUDA assertion, re-run on EVERY call, warm or cold. A silent CPU fallback would bill GPU
# hours to produce CPU numbers that look identical in the artifact.
python3 - <<'PY' || { echo "=== ENV ASSERTION FAILED -- refusing to spend box time ==="; exit 3; }
import sys
try:
    import torch
except Exception as e:                       # noqa: BLE001
    print("[setup] FATAL: torch unimportable:", e, file=sys.stderr); sys.exit(3)
if not torch.cuda.is_available():
    print("[setup] FATAL: torch.cuda.is_available() is False -> the run would SILENTLY use "
          "CPU. torch:", torch.__version__, file=sys.stderr)
    sys.exit(3)
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0),
      "mem_GiB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
import numpy, transformers, accelerate, jinja2, PIL   # ABI / import check
_jv = tuple(int(x) for x in jinja2.__version__.split(".")[:2])
if _jv < (3, 1):
    print("[setup] FATAL: jinja2", jinja2.__version__, "< 3.1 -> chat templates unusable; the "
          "instruction-tuned checkpoints are prompted through their own chat template",
          file=sys.stderr)
    sys.exit(3)
print("[setup] transformers", transformers.__version__, "accelerate", accelerate.__version__,
      "numpy", numpy.__version__, "pillow", PIL.__version__)
# Resolve the CONCRETE model class, not just the package: transformers imports lazily, so
# `import transformers` succeeds even when the Gemma-2 module cannot load, and the failure
# would otherwise surface inside from_pretrained -- after the box is paid for and the
# checkpoint pulled. Forcing the lookup here turns that into a five-second setup failure.
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: F401
print("[setup] Gemma2ForCausalLM resolvable (all six checkpoints are Gemma-2 architecture)")
PY

# ---------------------------------------------------------------------------
# SELFTESTS -- model-free; they gate the GPU. A broken instrument must show on the CPU.
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance run_seedC; do
  [ -f "$m.py" ] || { echo "  $m: ABSENT"; FAILED=1; RC_LINES="${RC_LINES}selftest_${m}=absent "; continue; }
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "out/selftest_$m.log")"
  [ "$rc" != 0 ] && FAILED=1
done
if [ "$FAILED" != 0 ]; then
  echo "=== SELFTESTS FAILED -- refusing to spend GPU time on a broken instrument ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  echo 1 > RUN_DONE
  exit 1
fi
python3 run_seedC.py --plan > out/seedC_plan.log 2>&1
echo "  plan: $(tail -1 out/seedC_plan.log)"

run_py(){                     # run_py <logname> <cmd...>
  local name="$1"; shift
  local log="out/${name}.log"
  echo "=== $name RUNNING ($(date -u +%H:%M:%S)Z) ==="
  "$@" > "$log" 2>&1
  local rc=$?
  echo "--- $name tail ---"; tail -15 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="
  else echo "=== $name OK ==="; fi
  return $rc
}
skip(){ echo "=== $1 SKIPPED (an upstream stage failed) ==="
        RC_LINES="${RC_LINES}$1=skipped "; }

# ---------------------------------------------------------------------------
# STAGES on this box: extract, then the free-generation audit. Endpoints run LOCALLY (header).
# extract writes the digit read-outs the endpoints depend on; if it fails, the generation
# audit is a separate register and still worth having, so it is NOT skipped on an extract
# failure -- each records its own exit code.
# ---------------------------------------------------------------------------
run_py seedC_extract  python3 run_seedC.py --stage extract  --size "$SIZE" --out "$RDIR" \
                              --batch-size "$BATCH_SIZE"
run_py seedC_generate python3 run_seedC.py --stage generate --size "$SIZE" --out "$RDIR" \
                              --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW"

# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR -> out/ ==="
cp "$RDIR"/*.json  out/ 2>/dev/null || true
cp "$RDIR"/*.jsonl out/ 2>/dev/null || true
# lambda_run.sh fetches tiny *summary*.json and *.log FIRST, so the decisions survive a flaky
# link that would truncate the multi-MB row blobs. Mirror onbox_seedA / onbox_sitref.
for f in "$RDIR"/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done

python3 - "$FAILED" "$RC_LINES" "$RDIR" "$SIZE" <<'PY' > out/.seedC_status.tmp 2>out/seedC_status.err
import glob, json, os, sys
failed, rc_lines, rdir, size = sys.argv[1:5]
decisions = {}
for p in sorted(glob.glob(os.path.join(rdir, "*.json"))):
    try:
        r = json.load(open(p))
    except Exception:                        # noqa: BLE001
        continue
    if isinstance(r, dict) and {"cell", "metric", "decision_rule"} <= r.keys():
        key = "_".join(f"{k}-{r['cell'][k]}" for k in sorted(r["cell"]))
        decisions[key] = r.get("decision", "")
rows = {os.path.basename(p): sum(1 for _ in open(p))
        for p in sorted(glob.glob(os.path.join(rdir, "*.jsonl")))}
json.dump({"runner": "onbox_seedC.sh", "seed": "C_scale", "size": size,
           "design_record": "records/seedC_scale_design_diff_triage.json",
           "stages_on_box": ["extract", "generate"],
           "endpoints_stage": "run LOCALLY after all three sizes are fetched (cross-size)",
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "n_provenance_records": len(decisions), "decisions": decisions,
           "row_files": rows,
           "batch_size": os.environ.get("BATCH_SIZE", "8"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": "A re-run of this size on the same box at the same batch size reproduces "
                   "these numbers. That is reproduction, not replication."},
          sys.stdout, indent=2)
PY
[ -s out/.seedC_status.tmp ] && mv out/.seedC_status.tmp out/seedC_run_summary.json \
                             || rm -f out/.seedC_status.tmp
ls -la out/ | tail -30
echo "=== EXIT CODES: ${RC_LINES}==="

# RUN_DONE last, after the collect: the launcher polls this marker and fetches when it appears.
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_SEEDC ($SIZE) WITH FAILURES ==="
  echo 1 > RUN_DONE
  exit 1
fi
echo "=== ALLDONE_SEEDC ($SIZE) OK ==="
echo 0 > RUN_DONE
exit 0
