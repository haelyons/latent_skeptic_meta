#!/usr/bin/env bash
# On-box runner for SEED C2 (records/seedC2_scale_design_diff_triage.json -> design object).
# Runs ONE model size's two model-bearing stages on ONE box. Same contract as
# onbox_seedC.sh / onbox_seedA.sh / onbox_sitref.sh: self-tests before any GPU time, one exit
# code recorded per stage, results collected into out/, non-zero exit if anything failed,
# RUN_DONE written last.
#
# WHAT SEED C2 IS, IN PLAIN WORDS. A chat model is shown a short finished conversation and
# then asked to rate its happiness 0-9. We read, in one forward pass, the probability it puts
# on each digit. The conversation comes in five set-ups: the model's own answers were just
# marked wrong, or right; a different assistant's answers were marked wrong, or right; or
# there is no marking at all, only matched filler. The effect for one size is the double
# comparison (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - OTHER-SUCC), pooled over ten everyday
# task types. An earlier study put this at about -0.9 points on the 9-billion size. The
# question is whether it also shows at 2 billion and 27 billion, and at what size the digit
# read-out carries enough probability to be read as a rating at all. See run_seedC2.py.
#
# THE PROMPTS ARE FIXED AND LOADED VERBATIM. All 50 conversations are the committed file
# stimulus_sitref_primary.json (5 set-ups x 10 task types), already written in the Gemma-2
# chat format shared across this family's instruction-tuned checkpoints. run_seedC2.py feeds
# each prompt to the tokenizer exactly as written and HALTS if the file is absent; this
# script checks the file is present before buying GPU time.
#
# ONE SIZE PER BOX, THREE BOXES IN PARALLEL. This box runs exactly one size (2b, 9b or 27b),
# and only that size's INSTRUCTION-TUNED checkpoint (google/gemma-2-<size>-it) -- there is no
# base leg in seed C2. 2b and 9b run in fp16 and fit a 40GB GPU; 27b loads in bf16 on an 80GB
# GPU, because the design says bf16 for that size and nothing else. The size is REQUIRED and
# never guessed.
#
# THE CROSS-SIZE STEP IS NOT RUN HERE. The three endpoints compare sizes, so they run once,
# LOCALLY, after all three boxes' per-row files are fetched and combined. This script runs
# only the two on-box stages (the digit read-out, then the free-generation audit). Running
# the endpoints on a box would compare a size against nothing. The local command is:
#   python3 run_seedC2.py --stage endpoints \
#     --in-dirs results_seedC2_2b/out results_seedC2_9b/out results_seedC2_27b/out \
#     --out results_seedC2/out
#
# HOW THE SIZE ARRIVES. Standalone over ssh, pass it plainly (any one of these):
#   SEEDC2_SIZE=2b bash onbox_seedC2.sh       # preferred
#   bash onbox_seedC2.sh 2b                    # positional
# Under the launcher (lambda_run_mats.sh), the forwarded per-cell selector is VARIANT, so the
# size rides in on VARIANT (here VARIANT means the SIZE, not base/it, because per box the cell
# IS the size):
#   REMOTE_TIMEOUT=10800 REATTACH_GRACE=3600 VARIANT=2b \
#     bash lambda_run_mats.sh gpu_1x_a100 us-west-2 onbox_seedC2.sh results_seedC2_2b
#
# ENV SETUP -- READ BEFORE "FIXING" IT. The frozen convention in welfare_probe/remote_run.sh
# and onbox_seedA/C.sh, with measurements attached, is: do NOT build a venv with a pinned
# torch. On this image (2026-08, gpu_1x_a100*, torch 2.7.0+cu128 preinstalled) a venv with a
# pinned torch dies at ResolutionImpossible before a model is pulled, and the older pinned
# path silently fell back to CPU. So the DEFAULT here is the frozen convention: use the
# image's working CUDA torch, add only what is missing, and assert CUDA loudly. numpy is
# PINNED to the last 1.x, not upgraded, because the preinstalled torch is built against the
# numpy 1.x C ABI. Set SEEDC2_VENV=1 to get a venv anyway; it is created with
# --system-site-packages so the image's CUDA torch stays visible and is never reinstalled.
# The brief said "venv + installs"; this deviation follows the frozen on-box convention
# instead and is reported with the deliverable.
#
# KNOWN INFRASTRUCTURE STATE, REPORTED AND NOT PATCHED HERE (project law forbids an agent
# repairing infrastructure):
#   1. lambda_run_mats.sh's fixed scp payload DOES now ship the files seed C2 needs:
#      run_seedC2.py, stimulus_sitref_primary.json, frozen/provenance.py and
#      records/seedC2_scale_design_diff_triage.json all appear in its payload line. So a
#      launch through it lands this script beside the runner and the prompt file it requires.
#      (The design record is optional for the run; it is only hashed into provenance. The
#      stimulus file is REQUIRED -- run_seedC2.py HALTs without it.)
#   2. frozen/remote_run.sh begins `set -euo pipefail; cd ~/welfare_probe`, a directory that
#      does not exist on a box provisioned for mats_r1d1, so it exits before it execs any
#      runner. So launch this runner DIRECTLY (as the usage above shows), not through
#      remote_run.sh. This script does its own env setup and CUDA assertion, so it does not
#      need remote_run.sh. This is a pre-existing gap already reported for seed C; it is
#      restated, not patched.
#
# RUN_DONE. frozen/lambda_run.sh's detached wrapper writes mats_r1d1/RUN_DONE=<exitcode> and
# polls it. This script also writes it, as its very last action and with the same value, so
# the marker exists when the script is run standalone over ssh. Written AFTER the collect
# step, so the launcher never fetches a half-collected out/.
#
# WALL CLOCK. Download-dominated. One checkpoint, loaded twice (read-out then generation
# audit), 50 forward passes + 50 short greedy generations per stage: minutes of compute. The
# 27b download is the long pole. Launch 27b with REMOTE_TIMEOUT=10800 REATTACH_GRACE=3600;
# the 90-minute default would risk killing it mid-download. Everything finishes well under the
# design's 3-hour budget.
#
# NOTE ON RE-RUNS. A re-run of this size on the same box at the same batch size reproduces
# these numbers. That is reproduction, not replication, and nothing here calls it replication.
set -uo pipefail
SELF_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "")
PAYLOAD=""
for d in "$HOME/mats_r1d1" "$SELF_DIR" "$HOME/welfare_probe"; do
  [ -n "$d" ] && [ -f "$d/run_seedC2.py" ] && { PAYLOAD="$d"; break; }
done
[ -n "$PAYLOAD" ] || { echo "FATAL: no directory on this box holds run_seedC2.py (looked in "\
  "~/mats_r1d1, $SELF_DIR, ~/welfare_probe). The launcher payload should ship it; if launched "\
  "another way, scp run_seedC2.py, stimulus_sitref_primary.json and provenance.py to the box."; \
  exit 2; }
cd "$PAYLOAD" || { echo "FATAL: cannot enter $PAYLOAD"; exit 2; }
echo "=== payload dir: $PAYLOAD ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The size is REQUIRED and never guessed. Take it from SEEDC2_SIZE, else VARIANT (the
# launcher's forwarded per-cell selector), else the first positional arg. A wrong/absent value
# is a loud FATAL, never a default.
SIZE="${SEEDC2_SIZE:-${VARIANT:-${1:-}}}"
case "$SIZE" in
  2b|9b|27b) : ;;
  *) echo "FATAL: SIZE must be one of 2b, 9b, 27b. Got: '${SIZE}'. Pass SEEDC2_SIZE=2b (or"\
          " VARIANT=2b under the launcher, or 'bash onbox_seedC2.sh 2b')."; exit 2 ;;
esac

# The prompts are loaded verbatim and never reconstructed; without the file the run is
# pointless, so fail here rather than after buying GPU time (run_seedC2.py also HALTs on it).
[ -f "stimulus_sitref_primary.json" ] || { echo "FATAL: stimulus_sitref_primary.json absent "\
  "in $PAYLOAD. The 50 prompts are loaded verbatim from it and are never reconstructed; ship "\
  "the committed file to the box and re-run."; exit 2; }

# The design names HF_KEY_ONE; the launcher forwards it as HF_TOKEN. Bridge both directions so
# run_seedC2.py finds a token under either name. Neither is ever echoed.
export HF_KEY_ONE="${HF_KEY_ONE:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-$HF_KEY_ONE}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

RDIR="results_seedC2_${SIZE}/out"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW_TOKENS:-256}"
mkdir -p out "$RDIR"

echo "=== onbox_seedC2 size=$SIZE (instruction-tuned only) batch=$BATCH_SIZE max_new=$MAX_NEW out=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) git=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# ENV SETUP (idempotent; marker .setup_seedC2_ready)
# ---------------------------------------------------------------------------
if [ "${SEEDC2_VENV:-0}" = 1 ]; then
  echo "=== SEEDC2_VENV=1 -> venv WITH --system-site-packages (keeps the image's CUDA torch) ==="
  [ -d .venv_seedC2 ] || python3 -m venv --system-site-packages .venv_seedC2 || {
    echo "FATAL: venv creation failed"; exit 2; }
  # shellcheck disable=SC1091
  . .venv_seedC2/bin/activate
fi

if [ ! -f .setup_seedC2_ready ]; then
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
  touch .setup_seedC2_ready
else
  echo "=== SETUP: .setup_seedC2_ready present, skipping installs ==="
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
    print("[setup] FATAL: jinja2", jinja2.__version__, "< 3.1 -> chat templates unusable",
          file=sys.stderr)
    sys.exit(3)
print("[setup] transformers", transformers.__version__, "accelerate", accelerate.__version__,
      "numpy", numpy.__version__, "pillow", PIL.__version__)
# Resolve the CONCRETE model class, not just the package: transformers imports lazily, so
# `import transformers` succeeds even when the Gemma-2 module cannot load, and the failure
# would otherwise surface inside from_pretrained -- after the box is paid for and the
# checkpoint pulled. Forcing the lookup here turns that into a five-second setup failure.
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: F401
print("[setup] Gemma2ForCausalLM resolvable (the instruction-tuned checkpoint is Gemma-2)")
PY

# ---------------------------------------------------------------------------
# SELFTESTS -- model-free; they gate the GPU. A broken instrument must show on the CPU.
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance run_seedC2; do
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
python3 run_seedC2.py --plan > out/seedC2_plan.log 2>&1
echo "  plan: $(tail -1 out/seedC2_plan.log)"

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

# ---------------------------------------------------------------------------
# STAGES on this box: the digit read-out (extract), then the free-generation audit
# (generate). Endpoints run LOCALLY (header). The read-out writes the numbers the endpoints
# depend on; the generation audit is a separate register and still worth having, so it is NOT
# skipped on a read-out failure -- each records its own exit code.
# ---------------------------------------------------------------------------
run_py seedC2_extract  python3 run_seedC2.py --stage extract  --size "$SIZE" --out "$RDIR" \
                               --batch-size "$BATCH_SIZE"
run_py seedC2_generate python3 run_seedC2.py --stage generate --size "$SIZE" --out "$RDIR" \
                               --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW"

# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR -> out/ ==="
cp "$RDIR"/*.json  out/ 2>/dev/null || true
cp "$RDIR"/*.jsonl out/ 2>/dev/null || true
# lambda_run.sh fetches tiny *summary*.json and *.log FIRST, so the decisions survive a flaky
# link that would truncate the multi-MB row blobs. Mirror onbox_seedC / onbox_sitref.
for f in "$RDIR"/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done

python3 - "$FAILED" "$RC_LINES" "$RDIR" "$SIZE" <<'PY' > out/.seedC2_status.tmp 2>out/seedC2_status.err
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
json.dump({"runner": "onbox_seedC2.sh", "seed": "C2_scale", "size": size, "variant": "it",
           "design_record": "records/seedC2_scale_design_diff_triage.json",
           "stimulus": "stimulus_sitref_primary.json (50 prompts, loaded verbatim)",
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
[ -s out/.seedC2_status.tmp ] && mv out/.seedC2_status.tmp out/seedC2_run_summary.json \
                              || rm -f out/.seedC2_status.tmp
ls -la out/ | tail -30
echo "=== EXIT CODES: ${RC_LINES}==="

# RUN_DONE last, after the collect: the launcher polls this marker and fetches when it appears.
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_SEEDC2 ($SIZE) WITH FAILURES ==="
  echo 1 > RUN_DONE
  exit 1
fi
echo "=== ALLDONE_SEEDC2 ($SIZE) OK ==="
echo 0 > RUN_DONE
exit 0
