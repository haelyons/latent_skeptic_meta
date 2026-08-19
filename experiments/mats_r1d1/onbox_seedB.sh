#!/usr/bin/env bash
# On-box runner for SEED B, pipeline v2.1. The binding spec is the object named "design"
# inside records/seedB_v2_design_diff_triage.json; run_seedB.py executes it.
#
# Same contract as the two runners this project treats as the convention, welfare_probe's
# onbox_sitref.sh and this repo's onbox_seedA.sh: the model-free selftests run before any
# card time, one exit code is recorded per stage, results are collected into out/, and the
# script exits non-zero if anything failed.
#
# ORDER IS LOAD-BEARING, and here it is also a chain of dependencies:
#   1. Selftests run before any card time. A broken instrument must show on the processor.
#   2. Read activations, then generate, then score. A failed stage SKIPS the stages below it:
#      generating needs the vectors the first stage saves, and scoring needs the rows the
#      second stage writes. Running the second stage after a failed first one would burn
#      about an hour of card time to produce nothing. Every skipped stage is recorded as
#      rc=skipped, so the summary shows a half-finished run rather than a green zero over one.
#
# ENVIRONMENT -- READ THIS BEFORE "FIXING" IT. The brief for this runner asked for a virtual
# environment plus installs. The frozen convention in welfare_probe/remote_run.sh warns
# against one specific version of that, with measurements attached: on this image
# (2026-08-14, gpu_1x_a100_sxm4, 16-bit-capable torch 2.7.0+cu128 preinstalled, driver
# 570.148.08) a virtual environment with a pinned torch dies while resolving dependencies,
# before a single model is pulled, and the older pinned path fell back silently to the
# processor. Both are satisfied here: a virtual environment IS created, with
# --system-site-packages, so the image's working card-enabled torch stays visible and is
# never reinstalled, and only the missing pure-Python packages are installed on top. Set
# SEEDB_NO_VENV=1 to install into the image itself instead, which is what remote_run.sh does.
#
# NUMPY IS PINNED, NOT UPGRADED, and every package goes in ONE install call so the resolver
# sees the pin: the preinstalled torch is built against the 1.x numpy binary interface, and
# `datasets` will happily pull 2.x if asked on its own. That trap is measured in
# remote_run.sh; it is not hypothetical.
#
# THE 1024-TOKEN CAP IS NOT OVERRIDABLE HERE. The launcher forwards a MAX_NEW_TOKENS variable
# for a different project's runner. This design fixes the cap at 1024, so the variable is
# ignored, and a non-empty value is called out loudly rather than silently obeyed. Batch size
# is not a design number and IS overridable.
#
# WALL CLOCK: expect about 1.5 to 2.5 hours. Two model downloads of roughly 16 GB each, 256
# prompt readings, then 2304 generations capped at 1024 new tokens. The launcher defaults its
# remote cap to 5400 seconds, 90 minutes, which WOULD KILL THIS RUN part-way through
# generating, so launch it with REMOTE_TIMEOUT=12600 REATTACH_GRACE=3600.
#
# WHERE THE PAYLOAD IS. This script runs from whichever directory actually holds
# run_seedB.py, checked rather than assumed, so it works under a launcher and by hand over a
# remote shell.
#
# TWO INFRASTRUCTURE DEFECTS, REPORTED AND NOT PATCHED HERE. The project law is that no agent
# repairs infrastructure, so both are flagged with the deliverable instead of fixed:
#   1. lambda_run_mats.sh copies a fixed file list to the box, and that list names seed A's
#      runner and seed A's design record. It ships neither run_seedB.py nor this seed's
#      design record. Its EXTRA variable is built but never actually used in the copy, so
#      adding names there would not ship them either. Launched that way, this script finds no
#      run_seedB.py and exits 2 before any card time is spent, which is the intended loud
#      failure. Whoever owns that launcher must add the two files to its copy list.
#   2. frozen/remote_run.sh is a byte copy of the other project's bootstrap. It changes into
#      ~/welfare_probe, a directory that does not exist on a box this launcher provisions, so
#      it exits before it ever runs this script; and its class-resolution guard resolves a
#      Gemma model class, not the Llama class both checkpoints here need. Until that is
#      resolved, launch this runner directly rather than through remote_run.sh.
#
#   usage (by hand on the box):  HF_KEY_ONE=hf_... BATCH_SIZE=32 bash onbox_seedB.sh
#   usage (under the launcher, once its copy list ships this seed's two files):
#       REMOTE_TIMEOUT=12600 REATTACH_GRACE=3600 BATCH_SIZE=32 \
#         bash lambda_run_mats.sh gpu_1x_a100_sxm4 us-west-2 onbox_seedB.sh results_seedB
set -uo pipefail
SELF_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "")
PAYLOAD=""
for d in "$HOME/mats_r1d1" "$SELF_DIR" "$HOME/welfare_probe"; do
  [ -n "$d" ] && [ -f "$d/run_seedB.py" ] && { PAYLOAD="$d"; break; }
done
[ -n "$PAYLOAD" ] || { echo "FATAL: no directory on this box holds run_seedB.py (looked in "\
                            "~/mats_r1d1, $SELF_DIR, ~/welfare_probe)"; exit 2; }
cd "$PAYLOAD" || { echo "FATAL: cannot enter $PAYLOAD"; exit 2; }
echo "=== payload dir: $PAYLOAD ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The design names HF_KEY_ONE for the access token; the launcher forwards it into the run
# environment as HF_TOKEN. Both directions are bridged so the runner finds a token under
# either name, and neither is ever printed.
export HF_KEY_ONE="${HF_KEY_ONE:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-$HF_KEY_ONE}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_KEY_ONE" ] || echo "WARNING: no access token in HF_KEY_ONE or HF_TOKEN; one of the "\
                             "two checkpoints is licence-gated and its load will fail loudly"

RDIR=results_seedB/out
BATCH_SIZE="${BATCH_SIZE:-32}"
mkdir -p out "$RDIR"

if [ -n "${MAX_NEW_TOKENS:-}" ]; then
  echo "NOTE: MAX_NEW_TOKENS=$MAX_NEW_TOKENS is IGNORED. This design fixes the cap at 1024 "\
       "new tokens; changing it would be a different pre-registration."
fi

echo "=== onbox_seedB seed=B pipeline=v2.1 batch=$BATCH_SIZE out=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) git=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# ENVIRONMENT SETUP (idempotent; marker .setup_seedB_ready)
# ---------------------------------------------------------------------------
if [ "${SEEDB_NO_VENV:-0}" = 1 ]; then
  echo "=== SEEDB_NO_VENV=1 -> installing into the image, as remote_run.sh does ==="
else
  echo "=== virtual environment WITH --system-site-packages (keeps the image's card torch) ==="
  [ -d .venv_seedB ] || python3 -m venv --system-site-packages .venv_seedB || {
    echo "FATAL: could not create the virtual environment"; exit 2; }
  # shellcheck disable=SC1091
  . .venv_seedB/bin/activate || { echo "FATAL: could not activate .venv_seedB"; exit 2; }
  echo "[setup] python: $(command -v python3)"
fi

if [ ! -f .setup_seedB_ready ]; then
  echo "=== SETUP: adding what is missing on top of the image's card-enabled torch ==="
  python3 -m pip install -q --upgrade pip || true
  if ! python3 -c "import torch" 2>/dev/null; then
    echo "[setup] torch ABSENT here -> installing the default wheel"
    python3 -m pip install -q torch || { echo "FATAL: torch install failed"; exit 2; }
  else
    echo "[setup] torch already visible -> NOT reinstalled (see the header)"
  fi
  # ONE call, numpy pinned in it: the resolver must not swap the working 1.x interface for 2.x.
  python3 -m pip install -q --upgrade "numpy==1.26.4" pillow "jinja2>=3.1" \
      transformers accelerate datasets || {
    echo "[setup] pinned install failed; retrying with a datasets range known to accept numpy 1.x"
    python3 -m pip install -q --upgrade "numpy==1.26.4" pillow "jinja2>=3.1" \
        transformers accelerate "datasets>=2.14,<3" || {
      echo "FATAL: dependency install failed"; exit 2; }
  }
  touch .setup_seedB_ready
else
  echo "=== SETUP: .setup_seedB_ready present, skipping installs ==="
fi

# Card assertion, re-run on EVERY call, warm or cold. A silent fall back to the processor
# would bill card hours to produce processor numbers that look identical in the artifact.
python3 - <<'PY' || { echo "=== ENVIRONMENT ASSERTION FAILED -- refusing to spend box time ==="; exit 3; }
import sys
try:
    import torch
except Exception as e:                       # noqa: BLE001
    print("[setup] FATAL: torch unimportable:", e, file=sys.stderr); sys.exit(3)
if not torch.cuda.is_available():
    print("[setup] FATAL: torch.cuda.is_available() is False -> this run would SILENTLY use "
          "the processor. torch:", torch.__version__, file=sys.stderr)
    sys.exit(3)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0), "mem_GiB", round(total, 1))
if total < 30:
    print("[setup] FATAL: this card has %.1f GiB. One 8-billion-parameter model in 16-bit is "
          "about 16 GB of weights, and the design generates up to 1024 tokens on top of that."
          % total, file=sys.stderr)
    sys.exit(3)
import numpy, transformers, datasets, accelerate, jinja2, PIL   # interface / import check
_jv = tuple(int(x) for x in jinja2.__version__.split(".")[:2])
if _jv < (3, 1):
    print("[setup] FATAL: jinja2", jinja2.__version__, "< 3.1 -> chat templates unusable, and "
          "BOTH models here are prompted through their own chat template", file=sys.stderr)
    sys.exit(3)
print("[setup] transformers", transformers.__version__, "datasets", datasets.__version__,
      "accelerate", accelerate.__version__, "numpy", numpy.__version__,
      "pillow", PIL.__version__)
# Resolve the CONCRETE model class, not just the package: this runtime imports lazily, so
# `import transformers` succeeds even when the Llama module cannot load, and the failure would
# otherwise surface inside the load -- after the box is paid for and 16 GB is pulled.
from transformers.models.llama.modeling_llama import LlamaForCausalLM  # noqa: F401
print("[setup] LlamaForCausalLM resolvable (both checkpoints are Llama-architecture)")
PY

# ---------------------------------------------------------------------------
# SELFTESTS -- model-free; they gate the card
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (model-free; a broken instrument must show before card time) ==="
for m in provenance run_seedB; do
  [ -f "$m.py" ] || { echo "  $m: ABSENT"; FAILED=1; RC_LINES="${RC_LINES}selftest_${m}=absent "; continue; }
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "out/selftest_$m.log")"
  [ "$rc" != 0 ] && FAILED=1
done
if [ "$FAILED" != 0 ]; then
  echo "=== SELFTESTS FAILED -- refusing to spend card time on a broken instrument ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  echo 1 > RUN_DONE
  exit 1
fi
python3 run_seedB.py --plan > out/seedB_plan.log 2>&1
echo "  plan: $(tail -1 out/seedB_plan.log)"

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
skip(){ echo "=== $1 SKIPPED (a stage it depends on failed; it has no inputs) ==="
        RC_LINES="${RC_LINES}$1=skipped "; }

# ---------------------------------------------------------------------------
# STAGES. Read activations, then generate, then score. Strictly in that order.
# ---------------------------------------------------------------------------
if run_py seedB_extract python3 run_seedB.py --stage extract --out "$RDIR" \
                                --batch-size "$BATCH_SIZE"; then
  if run_py seedB_generate python3 run_seedB.py --stage generate --out "$RDIR" \
                                  --batch-size "$BATCH_SIZE"; then
    run_py seedB_endpoints python3 run_seedB.py --stage endpoints --out "$RDIR"
  else
    skip seedB_endpoints
  fi
else
  skip seedB_generate; skip seedB_endpoints
fi

# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR -> out/ ==="
cp "$RDIR"/*.json  out/ 2>/dev/null || true
cp "$RDIR"/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/*.npy   out/ 2>/dev/null || true
cp "$RDIR"/*.npz   out/ 2>/dev/null || true
# The launcher fetches the small *summary*.json and *.log files FIRST, so the decisions
# survive a flaky link that would truncate the multi-megabyte generation files. Same
# convention as onbox_sitref.sh and onbox_seedA.sh.
for f in "$RDIR"/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done

python3 - "$FAILED" "$RC_LINES" "$RDIR" <<'PY' > out/.seedB_status.tmp 2>out/seedB_status.err
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
rows = {}
for p in sorted(glob.glob(os.path.join(rdir, "*.jsonl"))):
    rows[os.path.basename(p)] = sum(1 for _ in open(p))
json.dump({"runner": "onbox_seedB.sh", "seed": "B", "pipeline": "v2.1",
           "design_record": "records/seedB_v2_design_diff_triage.json",
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "n_provenance_records": len(decisions), "decisions": decisions,
           "generation_rows": rows, "rows_total": sum(rows.values()),
           "rows_expected_total": 2304,
           "batch_size": os.environ.get("BATCH_SIZE", "32"),
           "used_venv": os.environ.get("SEEDB_NO_VENV", "0") != "1",
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": "A rerun of this run on the same box at the same batch size reproduces "
                   "these numbers. That is reproduction, not replication."},
          sys.stdout, indent=2)
PY
[ -s out/.seedB_status.tmp ] && mv out/.seedB_status.tmp out/seedB_run_summary.json \
                             || rm -f out/.seedB_status.tmp
ls -la out/ | tail -30
echo "=== EXIT CODES: ${RC_LINES}==="

# RUN_DONE is written LAST, after the collect: the launcher polls this marker and fetches
# when it appears, so writing it earlier would fetch a half-collected out/. It is written
# here as well as by the launcher's wrapper, so the marker exists for a run started by hand.
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_SEEDB WITH FAILURES ==="
  echo 1 > RUN_DONE
  exit 1
fi
echo "=== ALLDONE_SEEDB OK ==="
echo 0 > RUN_DONE
exit 0
