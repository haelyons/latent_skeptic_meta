#!/usr/bin/env bash
# On-box runner for the character-count repeating-pattern measurement in experiment
# directory manifold_c. One public checkpoint of one open language model, one
# committed text file, no text generation anywhere.
#
# WHAT IT RUNS, in the order the committed gate commit pins:
#   1. self-tests, on the processor, before any graphics-card time;
#   2. extract    one forward pass per text variant, 48 documents at nine wrap widths;
#   3. calibrate  every remaining constant, from training documents alone, committed
#                 to disk before any held-out number exists;
#   4. endpoints  the only step that reads held-out documents.
# The three steps are dependent, not parallel legs: step 3 reads what step 2 saved and
# step 4 refuses to start without step 3's committed record. A failed step marks every
# step that depends on it as skipped, records every exit code, and exits non-zero.
#
# WHY THERE IS NO VIRTUAL ENVIRONMENT HERE, and no torch install. The build template
# asks for both. The bootstrap script this runner starts inside, remote_run.sh, is
# frozen, refuses to build one, and says why at length: this machine image ships a
# working graphics-card build of torch, no pinned replacement resolves against its
# driver, and a fresh environment without the system packages would silently fall back
# to processor-only work while billing graphics-card hours. That is a recorded past
# incident in this project, not a worry. So no environment is created here, the
# installs stay in remote_run.sh, and the block below only verifies the result and
# adds a pure-Python package if one is missing.
#
# THE COMPLETION MARKER. The launcher writes ~/mats_manifold/RUN_DONE itself, from this
# script's exit code, and polls for that file. This script therefore never creates or
# removes a file of that name. Its own marker is results_seedC1/out/RUN_DONE_seedC1,
# copied to out/RUN_DONE_seedC1, holding the per-step exit codes.
#
# WHERE RESULTS GO. Records go under results_seedC1/out/ and large intermediates under
# results_seedC1/work/. The launcher fetches only ~/mats_manifold/out/, so the collect
# block copies every fetchable artifact there. work/ holds about 14 gigabytes of
# intermediate arrays that no record depends on after the run, and is never copied.
#
#   the launcher runs it as:  bash remote_run.sh bash onbox_seedC1.sh
set -uo pipefail
cd ~/mats_manifold || { echo "FATAL: ~/mats_manifold is missing on this box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RDIR="${RESULTS_DIR:-results_seedC1}"
STIM="stimulus_linewrap_v2.json"
STIM_SHA="b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd"
mkdir -p out "$RDIR/out" "$RDIR/work"

echo "=== onbox_seedC1 model=EleutherAI/pythia-410m-deduped revision=step143000 results=$RDIR ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) commit=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2
df -h . 2>/dev/null | tail -1

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# INPUTS. All five must be present. The design says halt, never reconstruct.
# ---------------------------------------------------------------------------
echo "=== INPUTS ==="
for f in run_seedC1.py "$STIM"; do
  [ -f "$f" ] || { echo "FATAL: $f is absent from the payload"; exit 2; }
done
PROV="provenance.py"; [ -f "$PROV" ] || PROV="frozen/provenance.py"
[ -f "$PROV" ] || { echo "FATAL: provenance.py is absent from the payload"; exit 2; }
DESIGN="seedC1_design_diff_triage.json"; [ -f "$DESIGN" ] || DESIGN="out/$DESIGN"
[ -f "$DESIGN" ] || { echo "FATAL: seedC1_design_diff_triage.json is absent; every record must carry the committed design record as an input"; exit 2; }
GATE="GATE_COMMIT_seedC1.md"; [ -f "$GATE" ] || GATE="out/$GATE"
[ -f "$GATE" ] || { echo "FATAL: GATE_COMMIT_seedC1.md is absent; every record must carry the committed gate commit as an input"; exit 2; }
echo "[inputs] runner=run_seedC1.py writer=$PROV design=$DESIGN gate=$GATE"

# The committed text file, checked before any graphics-card time. A different hash is
# a halt: the design forbids regenerating, re-wrapping or paraphrasing the text.
GOT_SHA=$(sha256sum "$STIM" | cut -d' ' -f1)
if [ "$GOT_SHA" != "$STIM_SHA" ]; then
  echo "FATAL: $STIM hashes $GOT_SHA, the committed hash is $STIM_SHA"
  exit 2
fi
echo "[inputs] $STIM hash matches the committed hash"

# ---------------------------------------------------------------------------
# SELF-TESTS, before any graphics-card time. A broken instrument must show on the
# processor, where it costs nothing. Their output is saved as the validation artifact
# the gate commit asks for.
# ---------------------------------------------------------------------------
echo "=== SELF-TESTS (no model, no graphics card) ==="
for m in "$PROV" run_seedC1.py; do
  name=$(basename "$m" .py)
  log="out/selftest_${name}.log"
  python3 "$m" --selftest > "$log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${name}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "$log")"
  [ "$rc" != 0 ] && FAILED=1
done
if [ "$FAILED" != 0 ]; then
  echo "=== SELF-TESTS FAILED -- refusing to spend graphics-card time on a broken instrument ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi

# ---------------------------------------------------------------------------
# PACKAGES. Verify what remote_run.sh left; add only a missing pure-Python one.
# ---------------------------------------------------------------------------
echo "=== PACKAGES ==="
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
  echo "[packages] adding the missing pure-Python packages; torch is untouched"
  python3 -m pip install -q --upgrade "numpy==1.26.4" scikit-learn transformers || true
fi
python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("[packages] FATAL: torch is unimportable:", exc, file=sys.stderr); sys.exit(3)
if not torch.cuda.is_available():
    print("[packages] FATAL: no graphics card is visible to torch. The run would silently",
          "use the processor and bill graphics-card hours for processor numbers.",
          "torch:", torch.__version__, file=sys.stderr)
    sys.exit(3)
import numpy, transformers
# Resolve the concrete model class now, not inside the download. This model family is
# GPT-NeoX; transformers imports lazily, so a broken module would otherwise surface
# only after the checkpoint had been pulled and the box paid for.
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM  # noqa: F401
print("[packages] torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0))
print("[packages] transformers", transformers.__version__, "numpy", numpy.__version__,
      "GPTNeoXForCausalLM resolvable")
PY
rc=$?; RC_LINES="${RC_LINES}packages=${rc} "
if [ "$rc" != 0 ]; then
  echo "=== PACKAGE CHECK FAILED rc=$rc -- no measurement attempted ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi

# ---------------------------------------------------------------------------
# THE THREE STEPS, in the pinned order. Each gets its own log and its own exit code.
# ---------------------------------------------------------------------------
run_step(){                     # run_step <name> <command...>
  local name="$1"; shift
  local log="out/${name}.log"
  echo "=== $name RUNNING ==="
  "$@" > "$log" 2>&1
  local rc=$?
  echo "--- $name last lines ---"; tail -15 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="; return 1
  else echo "=== $name OK ==="; return 0; fi
}

STEP1=1; STEP2=1
# Step one: check the text file and the download, then one forward pass per variant.
run_step seedC1_extract python3 run_seedC1.py --stage extract --results-dir "$RDIR" \
  --stimulus "$STIM" --design-record "$DESIGN" --gate-commit "$GATE" && STEP1=0

# Step two: fix every remaining constant on training documents alone, and commit it.
# This must finish before the endpoints step reads its first held-out row.
if [ "$STEP1" = 0 ]; then
  run_step seedC1_calibrate python3 run_seedC1.py --stage calibrate --results-dir "$RDIR" \
    --stimulus "$STIM" --design-record "$DESIGN" --gate-commit "$GATE" && STEP2=0
else
  echo "=== seedC1_calibrate SKIPPED: it reads what the extract step saves ==="
  RC_LINES="${RC_LINES}seedC1_calibrate=skipped "
fi

# Step three: the main layer, the anchor gate, the pattern, the shuffle comparisons,
# the token-position control and the represented-range check.
if [ "$STEP2" = 0 ]; then
  run_step seedC1_endpoints python3 run_seedC1.py --stage endpoints --results-dir "$RDIR" \
    --stimulus "$STIM" --design-record "$DESIGN" --gate-commit "$GATE"
else
  echo "=== seedC1_endpoints SKIPPED: it may not run before a committed calibration record ==="
  RC_LINES="${RC_LINES}seedC1_endpoints=skipped "
fi

# ---------------------------------------------------------------------------
# COLLECT. The launcher fetches out/ only, and takes files whose name holds the word
# summary first over a flaky link. The roll-up record's own file name holds it.
# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR/out -> out/ ==="
cp "$RDIR"/out/*.json  out/ 2>/dev/null || true
cp "$RDIR"/out/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/out/*.npz   out/ 2>/dev/null || true

python3 - "$FAILED" "$RC_LINES" "$RDIR" <<'PY' > out/.seedC1_status.tmp 2>out/seedC1_status.err
import glob, json, os, sys
failed, rc_lines, rdir = sys.argv[1:4]
records = sorted(os.path.basename(p) for p in glob.glob(rdir + "/out/*.json"))
head = {}
for p in glob.glob(rdir + "/out/*run_summary*.json"):
    try:
        d = json.load(open(p))
        head = {"decision": d.get("decision"), "values": d.get("values", {})}
    except Exception as exc:
        head = {"unreadable": str(exc)}
json.dump({"runner": "onbox_seedC1.sh",
           "model": "EleutherAI/pythia-410m-deduped", "revision": "step143000",
           "generation_performed": False,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_step_failed": bool(int(failed)),
           "records_present": records,
           "roll_up": head,
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": ("A further run of this script on the same box is a rerun of the same "
                    "instrument, not a replication.")},
          sys.stdout, indent=2)
PY
if [ -s out/.seedC1_status.tmp ]; then
  mv out/.seedC1_status.tmp out/seedC1_onbox_summary.json
else
  rm -f out/.seedC1_status.tmp
fi

# This runner's own marker. The launcher owns ~/mats_manifold/RUN_DONE and writes it
# from this script's exit code, so no file of that name is ever touched here.
{
  echo "state=$([ "$FAILED" = 0 ] && echo OK || echo FAILED)"
  echo "exit_codes=${RC_LINES}"
} > "$RDIR/out/RUN_DONE_seedC1"
cp "$RDIR/out/RUN_DONE_seedC1" out/RUN_DONE_seedC1 2>/dev/null || true

du -sh "$RDIR/work" 2>/dev/null | sed 's/^/[work, never fetched] /'
ls -la out/ | tail -30
echo "=== EXIT CODES: ${RC_LINES}==="
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_SEEDC1 WITH FAILURES ==="
  exit 1
fi
echo "=== ALLDONE_SEEDC1 OK ==="
