#!/usr/bin/env bash
# On-box runner for the formation rung of the character-count study: twelve named public
# checkpoints of one 410-million-weight model, one committed text file, no text
# generation. The complete specification is the design object inside
# records/seedD4_design_diff_triage.json, committed in records/GATE_COMMIT_seedD4.md.
#
# WHAT THIS RUNG ASKS. Two committed measurements already show that the final checkpoint
# of this model carries a character-count reader: held-out quality 0.780903 and 0.781031
# on two disjoint texts, both at readout 6. Neither says when that reader appeared. So
# this run measures the same thing at twelve checkpoints of the same training run, from
# initialization to the end, and asks whether the reader and its precision boundary form
# early and then hold still, or keep moving after they first appear.
#
# THE TWO INTEGRITY GATES, IN THE COMMITTED ORDER, AND WHY THE ORDER IS LOAD-BEARING.
#   1. The final checkpoint (step143000) is measured FIRST. Its pooled quality must land
#      within 0.02 of 0.781031 on this same text. Outside that band the whole sweep is
#      void as an instrument fault and the run stops, before eleven more downloads are
#      paid for.
#   2. The initialization checkpoint (step0) is measured SECOND. Its pooled quality must
#      stay below 0.3. At or above that, the fit is reading surface text statistics
#      rather than a trained counter: void, and the run stops.
#   3. Only then are the remaining ten measured, earliest to latest.
# Both gates are evaluated inside the sweep, because a gate that fired after the sweep
# would have already spent the box.
#
# WHAT THE RUNNER DELETES AND WHY. Each checkpoint's downloaded weights are deleted as
# soon as its forward passes end, and its per-position vectors are deleted as soon as its
# best readout is chosen. So the disk holds one checkpoint's download and one
# checkpoint's vectors at a time, about 3 gigabytes together, not twelve times that. The
# runner stops with a loud message if a deletion does not take.
#
# THE PRE-REGISTERED READINGS. FORMED_EARLY_AND_HELD, FORMED_LATE_AND_HELD,
# FORMED_AND_DRIFTED, NOT_FORMED_BY_FINAL, VOID_ANCHOR_FAILED, VOID_CONTROL_FAILED, with
# UNSTABLE_NOT_CLASSIFIABLE carried from the stability test's own definition for a
# movement that is not one-way in training steps.
#
# WHAT CHANGED FROM THE PREVIOUS RUNG'S RUNNER, onbox_seedD3_1.sh:
#   1. The runner is run_seedD4.py, the model is the 410-million-weight one, and the text
#      file is stimulus_linewrap_v2.json, the same file behind the committed 0.781031.
#      Both files are required. The text file's hash is checked here before any
#      graphics-card time, and again inside the runner, which records it.
#   2. There are two steps, not three, and the first one loops twelve checkpoints. Each
#      checkpoint's own calibration happens inside that loop, because its running totals
#      are dropped with its weights.
#   3. Records go under results_seedD4/out/, and every marker names this rung.
#   4. The room-to-work floors are smaller, because hidden size 1024 makes one
#      checkpoint's running totals about 2.5 gigabytes rather than 11.8.
#
# Same contract as onbox_sitref.sh in the sibling project, with three deliberate
# differences, each forced by this experiment rather than chosen:
#   1. No cell to select. That project runs two checkpoints as two legs of one
#      comparison, so it reads a variant name from the environment. Here the design names
#      one model and twelve revisions in a committed order, and the runner holds that
#      order, so nothing is read from the environment and no leg can be run alone.
#   2. The two steps are dependent, not parallel legs. That project deliberately keeps
#      going after a failure, because its legs are independent. Here the second step
#      reads what the first one wrote, so a failed first step marks the second as
#      skipped, records every exit code, and exits non-zero.
#   3. No virtual environment is created. See the block marked PACKAGES below.
#
# ORDER IS LOAD-BEARING, twice over.
#   a. The self-tests run before any graphics-card time. A broken instrument must show on
#      the processor, where it costs nothing.
#   b. Inside the sweep, each checkpoint's ridge strength is fixed on training documents
#      only, and that record is written, before any held-out number for that checkpoint
#      is computed. That ordering is the only thing that makes the held-out numbers worth
#      reading.
#
# WHERE RESULTS GO. Records are written under results_seedD4/out/, which this script
# creates. The launcher's fetch step copies only mats_manifold/out/, so the collect block
# below copies every record there as well, and also under a name containing "summary",
# which the launcher fetches first over a flaky link. Four kinds of file beyond the
# records are fetched with them:
#   - the twelve-row per-checkpoint table the trajectory stage reads;
#   - every text variant, saved verbatim with its own hash, one file per checkpoint;
#   - one row per held-out width-140 position with its predicted count, per checkpoint;
#   - the per-count average vectors at each checkpoint's best readout.
# Working files stay in results_seedD4/work/ and are never fetched. Nothing lives there
# between checkpoints: the weights and the per-position vectors are deleted as the sweep
# moves on.
#
# THE COMPLETION MARKER. The launcher writes mats_manifold/RUN_DONE itself, from the exit
# code of this script, and polls for that file. This script therefore never touches that
# file. Its own marker is results_seedD4/out/RUN_DONE, copied to out/RUN_DONE_seedD4,
# holding the per-step exit codes.
#
# EXPECTED WALL CLOCK on one A100 box. The same instrument on this same model and text
# measured 37 seconds for 432 forward passes, 36 seconds to fix the constants and 37
# seconds for its held-out numbers. This rung reads 336 variants per checkpoint, so one
# checkpoint costs roughly two minutes including its download, and twelve cost roughly
# twenty-five minutes. That sits inside the launcher's 90-minute cap and far inside the
# three-hour ceiling the build brief names. Nothing is generated and nothing is batched:
# each text variant is one whole document read once.
#
# BEFORE THIS CAN RUN, THE LAUNCHER MUST SHIP THREE MORE FILES. As of this writing the
# copy list in lambda_run_manifold.sh names the earlier runners and text files. It does
# not name run_seedD4.py, records/seedD4_design_diff_triage.json or
# records/GATE_COMMIT_seedD4.md. It does already ship stimulus_linewrap_v2.json, in the
# block guarded by run_seedD2.py. This script refuses to run without the runner, and
# warns loudly about the two records. Fix the copy list in the launcher; do not write
# replacements here.
#
#   usage, from the launcher:
#     bash lambda_run_manifold.sh gpu_1x_a100_sxm4 us-west-2 onbox_seedD4.sh results_seedD4
#   the launcher runs it as: bash remote_run.sh bash onbox_seedD4.sh
set -uo pipefail
cd ~/mats_manifold || { echo "FATAL: ~/mats_manifold missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RDIR="${RESULTS_DIR:-results_seedD4}"
STIM="stimulus_linewrap_v2.json"
STIM_SHA="b0aa4255b5038d6fbc2e7314decf1011a5b5a13990098142e16ab44c6bba41dd"
mkdir -p out "$RDIR/out" "$RDIR/work"

echo "=== onbox_seedD4 model=EleutherAI/pythia-410m-deduped results=$RDIR ==="
echo "=== twelve checkpoints, measured step143000 first, then step0, then step512 to step128000 ==="
echo "=== gate 1: step143000 within 0.02 of 0.781031, else the sweep is void and stops ==="
echo "=== gate 2: step0 below 0.3, else the sweep is void and stops ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) commit=${GIT_COMMIT:-unknown} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

# ---------------------------------------------------------------------------
# INPUTS. The text file and the runner must be present, and the text file must hash
# exactly. The design says halt, never regenerate, re-wrap, or reconstruct the text.
# The runner checks the same hash again and records it; this check only makes a wrong
# or truncated file fail here, before any graphics-card time is paid for.
# ---------------------------------------------------------------------------
if [ ! -f "$STIM" ]; then
  echo "FATAL: $STIM absent; the design forbids regenerating it."
  echo "FATAL: add $STIM to the copy list in lambda_run_manifold.sh."
  echo "FATAL: this is the text behind the committed 0.781031; the other two text files"
  echo "FATAL: of this chain are different text and are not substitutes."
  exit 2
fi
GOT_SHA=$(sha256sum "$STIM" | cut -d' ' -f1)
if [ "$GOT_SHA" != "$STIM_SHA" ]; then
  echo "FATAL: $STIM hashes $GOT_SHA, expected $STIM_SHA."
  echo "FATAL: the committed text file is not what arrived; halt, do not regenerate."
  exit 2
fi
echo "[inputs] $STIM present, hash matches the committed $STIM_SHA"
if [ ! -f run_seedD4.py ]; then
  echo "FATAL: run_seedD4.py absent from the payload."
  echo "FATAL: add run_seedD4.py to the copy list in lambda_run_manifold.sh."
  echo "FATAL: the earlier runners of this chain measure one checkpoint each and are not"
  echo "FATAL: substitutes for a twelve-checkpoint sweep."
  exit 2
fi
if [ -f seedD4_design_diff_triage.json ]; then
  echo "[inputs] design record: seedD4_design_diff_triage.json, hash 0f450072d015... expected"
else
  echo "[inputs] WARNING: no design record shipped, so records will carry no design hash."
  echo "[inputs] Fix by adding records/seedD4_design_diff_triage.json to the launcher's"
  echo "[inputs] copy list. Do not write a replacement here."
fi
if [ -f GATE_COMMIT_seedD4.md ]; then
  echo "[inputs] gate record: GATE_COMMIT_seedD4.md, hash 28940a20bf97... expected"
else
  echo "[inputs] WARNING: records/GATE_COMMIT_seedD4.md was not shipped, so records will"
  echo "[inputs] carry no hash for it. Its four answers of record and its binding"
  echo "[inputs] coordinate control are already fixed as constants inside run_seedD4.py,"
  echo "[inputs] so the measurement is unaffected. Fix the audit trail by adding that"
  echo "[inputs] path to the copy list."
fi

# ---------------------------------------------------------------------------
# SELFTESTS, before any graphics-card time. Both are standard-library only. The runner's
# own self-test covers the position labeller including the case where only a closing
# boundary appears, the degenerate-input detector in both its cases, every exact-match
# check, and every endpoint's arithmetic on planted checkpoints whose calls are known.
# ---------------------------------------------------------------------------
echo "=== SELFTESTS (no model, no graphics card; a broken instrument must show first) ==="
PROV="provenance.py"; [ -f "$PROV" ] || PROV="frozen/provenance.py"
for m in "$PROV" run_seedD4.py; do
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
# says why at length: this machine image ships a working graphics-card build of torch, and
# no pinned replacement resolves against its driver. A fresh environment without the
# system packages would silently fall back to processor-only work, and bill graphics-card
# hours for processor numbers. That is a recorded past incident, not a worry. The sibling
# project's own on-box runner, onbox_sitref.sh, builds no environment either, and neither
# did the five earlier runners of this chain. So none is created here, the installs stay
# in remote_run.sh, and this block only verifies the result and adds a pure-Python package
# if one is missing.
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
# Resolve the concrete model class now, not inside the first download. This model family
# is GPT-NeoX; transformers imports lazily, so a broken module would otherwise surface
# only after a checkpoint had been pulled and the box paid for.
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
# ACCESS TOKEN. These checkpoints are public, so the run works without one. The brief
# names the variable HF_KEY_ONE; the launcher forwards the same secret as HF_TOKEN and
# the bootstrap also exports HUGGING_FACE_HUB_TOKEN. The runner reads all three, in that
# order, and records which one it used. No value is ever printed.
# ---------------------------------------------------------------------------
if [ -n "${HF_KEY_ONE:-}" ]; then
  echo "[token] HF_KEY_ONE is set"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HF_KEY_ONE="$HF_TOKEN"
  echo "[token] HF_KEY_ONE taken from the forwarded HF_TOKEN"
else
  echo "[token] no access token in the environment; these checkpoints are public, so the run continues"
fi

# ---------------------------------------------------------------------------
# ROOM TO WORK. One checkpoint at a time needs about 1.6 gigabytes for its download and
# about 1.2 gigabytes for the per-position vectors of the width-140 held-out documents at
# all 25 readouts, of which 24 are deleted as soon as the best one is chosen. Its running
# totals are about 2.5 gigabytes in memory. Running out of either halfway through costs
# the whole box, so both are checked first. The 12-gigabyte and 8-gigabyte floors are this
# runner's own guards, not design numbers.
# ---------------------------------------------------------------------------
FREE_MB=$(df -Pm . | awk 'NR==2 {print $4}')
echo "=== DISK: ${FREE_MB} MB free where results are written (about 3 GB is live at once) ==="
if [ "${FREE_MB:-0}" -lt 12288 ]; then
  echo "=== FATAL: under 12 gigabytes free; a checkpoint would die part-written ==="
  RC_LINES="${RC_LINES}disk=1 "
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi
RC_LINES="${RC_LINES}disk=0 "

AVAIL_MB=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)
echo "=== MEMORY: ${AVAIL_MB:-unknown} MB available (about 2.5 GB of running totals are held at once) ==="
if [ -n "${AVAIL_MB:-}" ] && [ "$AVAIL_MB" -lt 8192 ]; then
  echo "=== FATAL: under 8 gigabytes available; the sweep would be killed part-way ==="
  RC_LINES="${RC_LINES}memory=1 "
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi
RC_LINES="${RC_LINES}memory=0 "

# ---------------------------------------------------------------------------
# THE TWO STEPS, in order. Each gets its own log and its own exit code.
# ---------------------------------------------------------------------------
run_stage(){                    # run_stage <name> <cmd...>
  local name="$1"; shift
  local log="out/${name}.log"
  echo "=== $name RUNNING ==="
  "$@" > "$log" 2>&1
  local rc=$?
  echo "--- $name tail ---"; tail -20 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="; return 1
  else echo "=== $name OK ==="; return 0; fi
}

STEP1=1
# Step one: the twelve-checkpoint sweep, in the committed measurement order. Per
# checkpoint: download, check the configuration against 24 layers and hidden size 1024,
# one forward pass per text variant, the running totals, the ridge strength from training
# documents only, the best readout from held-out documents, the four-part precision
# profile and the direction counts at that readout, the records, then delete that
# checkpoint's weights and working vectors. The build brief calls a middle position the
# generation stage; nothing is generated in this experiment, and the runner accepts either
# name for this step.
run_stage seedD4_sweep python3 run_seedD4.py --stage extract --results-dir "$RDIR" && STEP1=0

# Step two: the cross-checkpoint arithmetic. Formation step, the formed checkpoint, the
# boundary-ratio trajectory, the stability test at ranges 0.15 and 0.10, the early-late
# call against step16000, and the pre-registered reading. It reads the twelve-row table
# the sweep wrote and refuses to read a partial sweep.
if [ "$STEP1" = 0 ]; then
  run_stage seedD4_endpoints python3 run_seedD4.py --stage endpoints --results-dir "$RDIR"
else
  echo "=== seedD4_endpoints SKIPPED: it may not read a sweep that did not finish ==="
  echo "=== a sweep stopped by an integrity gate has already written its own void reading ==="
  RC_LINES="${RC_LINES}seedD4_endpoints=skipped "
fi

# ---------------------------------------------------------------------------
# COLLECT. The launcher fetches out/ only, smallest decision-bearing files first. The
# per-checkpoint table, the verbatim text rows, the per-position rows and the average
# vectors are copied too, because the design requires them fetched, not merely computed.
# ---------------------------------------------------------------------------
echo "=== COLLECT $RDIR/out -> out/ ==="
cp "$RDIR"/out/*.json out/ 2>/dev/null || true
cp "$RDIR"/out/*.jsonl out/ 2>/dev/null || true
cp "$RDIR"/out/*.npz out/ 2>/dev/null || true
for f in "$RDIR"/out/*.json; do
  [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true
done

python3 - "$FAILED" "$RC_LINES" "$RDIR" "$STIM" "$STIM_SHA" <<'PY' > out/.seedD4_status.tmp 2>out/seedD4_status.err
import glob, json, os, sys
failed, rc_lines, rdir, stim, stim_sha = sys.argv[1:6]
out = rdir + "/out"
table_path = os.path.join(out, "sweep_table.json")
table = None
if os.path.exists(table_path):
    try:
        with open(table_path, "r", encoding="utf-8") as fh:
            t = json.load(fh)
        table = {"n_rows": t.get("n_rows"), "complete": t.get("complete"),
                 "anchor_gate_decision": t.get("anchor_gate_decision"),
                 "negative_control_decision": t.get("negative_control_decision"),
                 "pooled_quality_by_revision":
                     {r["revision"]: r.get("pooled_quality") for r in t.get("rows", [])},
                 "best_layer_by_revision":
                     {r["revision"]: r.get("best_layer") for r in t.get("rows", [])},
                 "ridge_strength_by_revision":
                     {r["revision"]: r.get("ridge_strength") for r in t.get("rows", [])},
                 "boundary_ratio_by_revision":
                     {r["revision"]: r.get("boundary_ratio") for r in t.get("rows", [])},
                 "coordinate_flags_by_revision":
                     {r["revision"]: r.get("coordinate_flags") for r in t.get("rows", [])}}
    except Exception as exc:                                   # noqa: BLE001
        table = {"unreadable": str(exc)}
json.dump({"runner": "onbox_seedD4.sh",
           "model": "EleutherAI/pythia-410m-deduped",
           "revisions_in_training_order":
               ["step0", "step512", "step1000", "step2000", "step4000", "step8000",
                "step16000", "step32000", "step64000", "step96000", "step128000",
                "step143000"],
           "measurement_order":
               ["step143000", "step0", "step512", "step1000", "step2000", "step4000",
                "step8000", "step16000", "step32000", "step64000", "step96000",
                "step128000"],
           "expected_layers": 24, "expected_hidden_size": 1024,
           "text_file": stim, "text_file_sha256_expected": stim_sha,
           "widths_read": [20, 40, 60, 80, 100, 120, 140],
           "widths_in_the_file_not_read": [200, 280],
           "profile_width": 140,
           "profile_parts": ["1 to 35", "36 to 70", "71 to 105", "106 to 140"],
           "profile_statistic": "Pearson correlation between predicted and true count",
           "anchor_value": 0.781031, "anchor_tolerance": 0.02,
           "negative_control_ceiling": 0.3,
           "formation_floor": 0.5, "formed_share_of_final": 0.90,
           "boundary_presence_floor": 0.3,
           "hold_test_boundary_range": 0.15, "hold_test_quality_range": 0.10,
           "early_late_cutoff_step": 16000,
           "ridge_grid": [1, 10, 100, 1000, 10000, 100000],
           "folds": 5, "base_seed": 0,
           "best_readout_chosen_per_checkpoint": True,
           "coordinate_edges_flagged": {"readouts": [0, 24], "strengths": [1, 100000]},
           "coordinate_flags_expected_at_step0": True,
           "variance_targets_recorded_only": [0.90, 0.95, 0.99],
           "ambient_dimension_gated": False,
           "pre_registered_readings": ["FORMED_EARLY_AND_HELD",
                                       "FORMED_LATE_AND_HELD",
                                       "FORMED_AND_DRIFTED",
                                       "NOT_FORMED_BY_FINAL",
                                       "UNSTABLE_NOT_CLASSIFIABLE",
                                       "VOID_ANCHOR_FAILED",
                                       "VOID_CONTROL_FAILED"],
           "generation_performed": False,
           "weights_and_vectors_deleted_per_checkpoint": True,
           "sweep_table": table,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "records_present": sorted(os.path.basename(p) for p in glob.glob(out + "/*.json")),
           "verbatim_variant_files": sorted(os.path.basename(p)
                                            for p in glob.glob(out + "/rows_variants_*.jsonl")),
           "per_position_files": sorted(os.path.basename(p)
                                        for p in glob.glob(out + "/rows_positions_*.jsonl")),
           "mean_vector_files": sorted(os.path.basename(p)
                                       for p in glob.glob(out + "/mean_vectors_*.npz")),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
           "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
           "note": ("A further run of this script on the same box is a rerun of the same "
                    "instrument, not a replication. This sweep is a single run.")},
          sys.stdout, indent=2)
PY
[ -s out/.seedD4_status.tmp ] && mv out/.seedD4_status.tmp out/seedD4_onbox_summary.json || rm -f out/.seedD4_status.tmp

# This runner's own marker. The launcher owns mats_manifold/RUN_DONE and writes it from
# this script's exit code, so that file is never touched here.
{
  echo "state=$([ "$FAILED" = 0 ] && echo OK || echo FAILED)"
  echo "exit_codes=${RC_LINES}"
} > "$RDIR/out/RUN_DONE"
cp "$RDIR/out/RUN_DONE" out/RUN_DONE_seedD4 2>/dev/null || true

ls -la out/ | tail -30
du -sh "$RDIR/work" 2>/dev/null || true
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SEEDD4 WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SEEDD4 OK ==="
