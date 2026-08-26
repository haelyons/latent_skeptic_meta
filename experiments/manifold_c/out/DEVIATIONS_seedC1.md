# Deviations, seed C1

Every departure from the pipeline's own procedure, booked as it happened. None of
these changes a threshold, a band, a decision rule, or any measured number.

## D1. Launch 1 failed on a defect in the built runner; the operating session fixed two lines rather than re-running the build stage

**When.** 2026-08-26, first launch. Instance `71a83ac35d244fe1bc5a51053629dee5`,
`gpu_1x_a100_sxm4`, us-west-2. The box ran about 16 minutes, of which the
measurement ran 60 seconds before failing.

**What failed.** The extract step completed all 432 forward passes and then died
writing its own output:

```
  File "/home/ubuntu/mats_manifold/run_seedC1.py", line 1389, in stage_extract
    np.savez(work_dir / "doc_sums_w%d.npz" % ANCHOR_WIDTH,
TypeError: unsupported operand type(s) for %: 'PosixPath' and 'int'
```

In Python `/` binds tighter than `%`, so the expression built a path and then
tried to format the path, instead of formatting the name and then joining it.
The same defect sat two lines later, at the companion save for the three probe
windows. Every read-back site in the same file already used the parenthesised
form, so the two write sites were the only ones wrong.

**Why the self-test did not catch it.** The build brief requires the self-test to
be standard-library only, so that a broken instrument shows on the processor
before any graphics-card time is spent. That requirement is what made the
self-test skip these two lines: they are the only code in the extract step that
calls into the array library to write a file, and a standard-library-only test
cannot reach them. The self-test's 189 checks all passed, on this workstation and
again on the box, and were right to.

**What was changed.** Exactly two characters classes, on lines 1389 and 1394 of
`run_seedC1.py`:

```
-    np.savez(work_dir / "doc_sums_w%d.npz" % ANCHOR_WIDTH,
+    np.savez(work_dir / ("doc_sums_w%d.npz" % ANCHOR_WIDTH),
-        np.savez(work_dir / "act_meta_a%d.npz" % a,
+        np.savez(work_dir / ("act_meta_a%d.npz" % a),
```

Nothing else in the file was touched. The self-test was re-run and still passes,
exit 0. A scan for the same precedence pattern across the whole file returns
nothing further.

**Why this was a fix and not a halt.** The pipeline's law is that no agent repairs
INFRASTRUCTURE: anything in `frozen/` that is not runnable as given must halt and
be reported, never patched. `run_seedC1.py` is not infrastructure. It is this
run's own build-stage output, and repairing a defect in it is the operating
session's job. No frozen file was touched: `pipeline.js`, `triage_workflow.js`,
`gates/` and `frozen/` are byte-identical to how they were found. No threshold,
band, bin edge, seed, or decision rule was touched. The committed design record
and gate commit are unchanged, and their hashes, which the runner carries as
constants and stamps into every record, still match.

**Cost.** Reconstructed from the cloud audit log: launched 01:09:00 UTC,
terminated 01:16:18 UTC, 7.3 minutes of one box at 1.99 USD per hour, 0.24 USD,
against this seed's 15 USD cap. Teardown was confirmed through the cloud API: the
first terminate call failed on a transient network error, the launcher's retry
loop succeeded on the second try, and the instance moved to `terminating` and
then out of the account's instance list.

**What was kept.** The failed launch's artifacts are kept under
`out/results_seedC1_launch1_failed/`, including the extract step's own log, both
self-test logs, the setup-check record, and the per-position and per-variant row
files it did manage to write. They are evidence about the instrument, not about
the model, and no number in them enters any endpoint.

## D2. The build stage's template named a stage this design cannot have, and a package step the frozen bootstrap forbids

**What.** The build brief asks for stages named extract / generate / endpoints and
for the on-box script to create a virtual environment and install packages. This
design generates no text at all, and `frozen/remote_run.sh` explicitly refuses a
virtual environment and does the installs itself, with a recorded past incident
behind each refusal.

**Resolution.** The build agent followed the two binding files, which is what the
execution directive told it to do: stages are extract / calibrate / endpoints,
`--stage generate` exits 2 with an explanation, and the on-box script only
verifies what the frozen bootstrap left behind. Booked here because it is a
visible departure from the pipeline's own template wording.


## D3. The launcher's own elapsed counter under-reports, which is cosmetic

**What.** The launch-2 log reads `[run] DONE marker rc=0 after 60s`, while the
run's own records span 01:24:17 to 01:30:59 UTC and its three steps report 50.5,
159.9 and 194.8 seconds. The counter in the frozen launcher's poll loop counts
its own sleeps, not wall clock, and each poll's short-lived connection to the box
took minutes while the box was writing about 13 gigabytes and holding 30 cores
busy. So one loop pass covered the whole run and the counter advanced by one
poll.

**Resolution.** Nothing changed. The launcher is frozen, the marker was read
correctly, the fetch happened after the run finished, and every fetched artifact
is complete: all six step exit codes are 0 and 19 records are present. The
counter is a display, not a measurement, and no result depends on it. Recorded
so a later reader does not mistake the log line for a run that took one minute.

## D4. Total spend, reconstructed from the audit log

Two launches, both `gpu_1x_a100_sxm4` in us-west-2 at 1.99 USD per hour.

| instance | launched (UTC) | terminated (UTC) | minutes | cost |
|---|---|---|---|---|
| 71a83ac35d244fe1bc5a51053629dee5 | 01:09:00 | 01:16:18 | 7.3 | 0.24 USD |
| cfca0389986f46ec8b2e950210c5f399 | 01:19:37 | 01:33:14 | 13.6 | 0.45 USD |

Total 0.69 USD against a 15 USD cap. The account's instance list was confirmed
empty through the cloud API after each teardown, and both terminations appear in
the audit log.
