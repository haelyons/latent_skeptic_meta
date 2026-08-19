#!/usr/bin/env bash
# On-box runner for SEED A, pipeline v2 (records/seedA_v2_design_diff_triage.json -> design).
# Same contract as onbox_sitref.sh: selftests before any GPU time, one exit code recorded per
# stage, results collected into out/, non-zero exit if anything failed.
#
# ORDER IS LOAD-BEARING, and here it is also a dependency chain:
#   1. Selftests run before any GPU time. A broken instrument must show on the CPU.
#   2. extract -> generate -> endpoints. Unlike the SITREF runner, a failed stage DOES skip
#      the stages below it: generate reads the direction extract writes, and endpoints reads
#      the rows generate writes. Running generate after a failed extract would burn ~2 GPU
#      hours to produce nothing. Every skipped stage is recorded as rc=skipped, so the
#      summary shows a half-finished run rather than a green 0 over one.
#
# ENV SETUP -- READ THIS BEFORE "FIXING" IT. The brief for this runner said "creates
# venv/installs (torch, transformers, accelerate, datasets)". The frozen convention in
# welfare_probe/remote_run.sh says the opposite, with measurements attached: on this image
# (2026-08-14, gpu_1x_a100_sxm4, torch 2.7.0+cu128 preinstalled, driver 570.148.08) a venv
# with a pinned torch dies at ResolutionImpossible before a single model is pulled, and the
# older pinned path silently fell back to CPU. So the DEFAULT here is the frozen convention:
# use the image's working CUDA torch, install only what is missing on top, and assert CUDA
# loudly. All four packages are guaranteed present either way -- torch is installed only if
# it is genuinely absent. Set SEEDA_VENV=1 to get a venv anyway; it is created with
# --system-site-packages so the working CUDA torch stays visible and is never reinstalled.
# This deviation from the brief is deliberate and is reported with the deliverable.
#
# NUMPY IS PINNED, NOT UPGRADED, and every package goes in ONE pip call so the resolver sees
# the pin: the preinstalled torch is built against the numpy 1.x C ABI, and `datasets` will
# happily pull numpy 2.x if asked separately (measured trap, remote_run.sh).
#
# RUN_DONE: frozen/lambda_run.sh's detached wrapper writes welfare_probe/RUN_DONE=<exitcode>
# and polls it. This script also writes it, as its very last action and with the same value,
# so the marker exists when the script is run standalone (ssh, no launcher). Written AFTER
# the collect step -- writing it earlier would make the launcher fetch a half-collected out/.
#
# WALL CLOCK: expect ~2 to 2.5 h (2 model downloads + 200 prompt forwards + 4,400 generations
# at up to 512 new tokens). The launcher defaults REMOTE_TIMEOUT to 5400s (90 min), which WOULD
# KILL THIS RUN mid-generate -- launch it with REMOTE_TIMEOUT=12600 REATTACH_GRACE=3600.
#
# REMOTE DIR. lambda_run_mats.sh scps the payload (run_seedA.py, provenance.py, remote_run.sh,
# the design record, this runner) into ~/mats_r1d1/ and polls ~/mats_r1d1/RUN_DONE. This script
# therefore lands in the directory that actually HOLDS run_seedA.py, checked rather than
# assumed, so it works under either launcher and standalone over ssh.
#
# KNOWN INFRASTRUCTURE DEFECT, REPORTED AND NOT PATCHED HERE: the shipped frozen/remote_run.sh
# is a byte copy of the welfare_probe bootstrap. It runs `set -euo pipefail` then
# `cd ~/welfare_probe`, a directory that does not exist on a box provisioned by
# lambda_run_mats.sh, so it exits before it ever execs this runner; and its concrete-class
# guard resolves Gemma2ForCausalLM, not the Llama class this run needs. Both are infrastructure
# and the project law forbids an agent repairing infrastructure, so they are flagged with the
# deliverable instead. Until that is resolved by whoever owns frozen/, launch the runner
# directly rather than through remote_run.sh (see the reported command).
#
#   usage (under the launcher):  REMOTE_TIMEOUT=12600 REATTACH_GRACE=3600 \
#                                  bash lambda_run_mats.sh gpu_1x_a100_sxm4 us-west-2 \
#                                  onbox_seedA.sh results_seedA
#   usage (standalone on box):   HF_KEY_ONE=hf_... bash onbox_seedA.sh
set -uo pipefail
SELF_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "")
PAYLOAD=""
for d in "$HOME/mats_r1d1" "$SELF_DIR" "$HOME/welfare_probe"; do
  [ -n "$d" ] && [ -f "$d/run_seedA.py" ] && { PAYLOAD="$d"; break; }
done
[ -n "$PAYLOAD" ] || { echo "FATAL: no directory on this box holds run_seedA.py (looked in "\
                            "~/mats_r1d1, $SELF_DIR, ~/welfare_probe)"; exit 2; }
cd "$PAYLOAD" || { echo "FATAL: cannot enter $PAYLOAD"; exit 2; }
echo "=== payload dir: $PAYLOAD ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The design names HF_KEY_ONE (frozen/lambda_run.sh reads it from ./.keys); the launcher
# forwards it into the run env as HF_TOKEN. Bridge both directions so run_seedA.py finds a
# token under either name, and neither is ever echoed.
export HF_KEY_ONE="${HF_KEY_ONE:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-$HF_KEY_ONE}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

RDIR=results_seedA/out
BATCH_SIZE="${BATCH_SIZE:-16}"
mkdir -p out "$RDIR"

echo "=== onbox_seedA seed=A pipeline=v2 batch=$BATCH_SIZE out=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) git=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# ENV SETUP (idempotent; marker .setup_seedA_ready)
# ---------------------------------------------------------------------------
if [ "${SEEDA_VENV:-0}" = 1 ]; then
  echo "=== SEEDA_VENV=1 -> venv WITH --system-site-packages (keeps the image's CUDA torch) ==="
  [ -d .venv_seedA ] || python3 -m venv --system-site-packages .venv_seedA || {
    echo "FATAL: venv creation failed"; exit 2; }
  # shellcheck disable=SC1091
  . .venv_seedA/bin/activate
fi

if [ ! -f .setup_seedA_ready ]; then
  echo "=== SETUP: adding what is missing on top of the image's CUDA torch ==="
  python3 -m pip install -q --upgrade pip || true
  if ! python3 -c "import torch" 2>/dev/null; then
    echo "[setup] torch ABSENT on this image -> installing the default wheel"
    python3 -m pip install -q torch || { echo "FATAL: torch install failed"; exit 2; }
  else
    echo "[setup] torch already present -> NOT reinstalled (see header)"
  fi
  # ONE call, numpy pinned first: the resolver must not swap the working 1.x ABI for 2.x.
  python3 -m pip install -q --upgrade "numpy==1.26.4" pillow "jinja2>=3.1" \
      transformers accelerate datasets || {
    echo "[setup] pinned install failed; retrying with a datasets range known to accept numpy 1.x"
    python3 -m pip install -q --upgrade "numpy==1.26.4" pillow "jinja2>=3.1" \
        transformers accelerate "datasets>=2.14,<3" || {
      echo "FATAL: dependency install failed"; exit 2; }
  }
  touch .setup_seedA_ready
else
  echo "=== SETUP: .setup_seedA_ready present, skipping installs ==="
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
import numpy, transformers, datasets, accelerate, jinja2, PIL   # ABI / import check
_jv = tuple(int(x) for x in jinja2.__version__.split(".")[:2])
if _jv < (3, 1):
    print("[setup] FATAL: jinja2", jinja2.__version__, "< 3.1 -> chat templates unusable; "
          "BOTH models here are prompted through their own chat template", file=sys.stderr)
    sys.exit(3)
print("[setup] transformers", transformers.__version__, "datasets", datasets.__version__,
      "accelerate", accelerate.__version__, "numpy", numpy.__version__,
      "pillow", PIL.__version__)
# Resolve the CONCRETE model class, not just the package: transformers imports lazily, so
# `import transformers` succeeds even when the Llama module cannot load, and the failure would
# otherwise surface inside from_pretrained -- after the box is paid for and 16GB is pulled.
from transformers.models.llama.modeling_llama import LlamaForCausalLM  # noqa: F401
print("[setup] LlamaForCausalLM resolvable (both checkpoints are Llama-architecture)")
PY

# ---------------------------------------------------------------------------
# SELFTESTS -- model-free; they gate the GPU
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance run_seedA; do
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
python3 run_seedA.py --plan > out/seedA_plan.log 2>&1
echo "  plan: $(tail -1 out/seedA_plan.log)"

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
skip(){ echo "=== $1 SKIPPED (an upstream stage failed; it has no inputs) ==="
        RC_LINES="${RC_LINES}$1=skipped "; }

# ---------------------------------------------------------------------------
# STAGES. extract -> generate -> endpoints, strictly in order.
# ---------------------------------------------------------------------------
if run_py seedA_extract python3 run_seedA.py --stage extract --out "$RDIR" \
                                --batch-size "$BATCH_SIZE"; then
  if run_py seedA_generate python3 run_seedA.py --stage generate --out "$RDIR" \
                                  --batch-size "$BATCH_SIZE"; then
    run_py seedA_endpoints python3 run_seedA.py --stage endpoints --out "$RDIR"
  else
    skip seedA_endpoints
  fi
else
  skip seedA_generate; skip seedA_endpoints
fi

# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR -> out/ ==="
cp "$RDIR"/*.json  out/ 2>/dev/null || true
cp "$RDIR"/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/*.npy   out/ 2>/dev/null || true
cp "$RDIR"/*.npz   out/ 2>/dev/null || true
# lambda_run.sh fetches tiny *summary*.json and *.log FIRST, so the decisions survive a flaky
# link that would truncate the multi-MB generation blobs. Same convention as onbox_sitref.sh.
for f in "$RDIR"/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done

python3 - "$FAILED" "$RC_LINES" "$RDIR" <<'PY' > out/.seedA_status.tmp 2>out/seedA_status.err
import glob, json, os, sys
failed, rc_lines, rdir = sys.argv[1:4]
decisions = {}
for p in sorted(glob.glob(os.path.join(rdir, "*.json"))):
    try:
        r = json.load(open(p))
    except Exception:                        # noqa: BLE001
        continue
    if isinstance(r, dict) and {"cell", "metric", "decision_rule"} <= r.keys():
        key = "_".join(f"{k}-{r['cell'][k]}" for k in sorted(r["cell"]))
        decisions[key] = r.get("decision", "")
gens = {os.path.basename(p): sum(1 for _ in open(p))
        for p in sorted(glob.glob(os.path.join(rdir, "*.jsonl")))}
json.dump({"runner": "onbox_seedA.sh", "seed": "A", "pipeline": "v2",
           "design_record": "records/seedA_v2_design_diff_triage.json",
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "n_provenance_records": len(decisions), "decisions": decisions,
           "generation_rows": gens, "rows_total": sum(gens.values()),
           "batch_size": os.environ.get("BATCH_SIZE", "16"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": "A rerun of this run on the same box at the same batch size reproduces "
                   "these numbers. That is reproduction, not replication."},
          sys.stdout, indent=2)
PY
[ -s out/.seedA_status.tmp ] && mv out/.seedA_status.tmp out/seedA_run_summary.json \
                             || rm -f out/.seedA_status.tmp
ls -la out/ | tail -30
echo "=== EXIT CODES: ${RC_LINES}==="

# RUN_DONE last, after the collect: the launcher polls this marker and fetches when it appears.
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_SEEDA WITH FAILURES ==="
  echo 1 > RUN_DONE
  exit 1
fi
echo "=== ALLDONE_SEEDA OK ==="
echo 0 > RUN_DONE
exit 0
