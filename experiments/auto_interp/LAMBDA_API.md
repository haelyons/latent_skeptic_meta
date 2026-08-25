# Lambda GPU access — how it worked, and the APIs used

How the GPU session for this project was provisioned, in plain terms, so it can
be reproduced. Date: 2026-06-15. All steps were driven from the local Windows
workstation (`LAPTOP-ULDACBRE`), **not** the Claude Code web sandbox — the
sandbox's HTTPS-only egress proxy blocks outbound SSH (see `archive/README_RUN.md`),
whereas the workstation has normal outbound network, which is what makes
"instrument from here over SSH" viable at all.

## The shape of it

1. Generate an SSH keypair locally.
2. Add the **public** key to the Lambda account via the Lambda Cloud API.
3. Launch a GPU instance via the API, selecting that key by name.
4. Lambda provisions the box and **injects the public key automatically** — no
   manual `ssh-copy-id`, no password. Once the instance is `active` you SSH in as
   `ubuntu@<ip>` with the matching private key.
5. Poll the API for the instance's IP, connect, run, then terminate.

The key point (and the reason no in-instance setup was needed for access):
**SSH access is completed by Lambda at provision time from the account's key
list.** You only have to get the public key onto the account once; every
instance launched thereafter with that key name is reachable immediately.

## Credentials

- Lambda API key: a `secret_…` bearer token, stored as `LAMBDA_KEY_ONE` in
  `./.keys` (single line `LAMBDA_KEY_ONE=secret_…`). `.keys` is **gitignored**
  (added to `.gitignore` this session) so the token can't be committed.
- SSH keypair: `~/.ssh/lambda_ed25519` (private) + `.pub`, ed25519, **no
  passphrase** (matches the existing `vastai_ed25519` pattern; enables
  non-interactive automation). Comment `lambda-latent_verify-helios-2026-06-15`.
  Treat the private key as sensitive; the rented box is ephemeral.
- The private key lives ONLY on the machine that generated it. A new
  workstation generates its own `~/.ssh/lambda_ed25519`, registers it under a
  NEW unique name via `POST /ssh-keys`, and launches with
  `SSH_KEY_NAME=<that name> bash lambda_run.sh …` (2026-07-21: this Linux
  workstation uses `latent_verify_hal_20260721`; the 06-15 name belongs to the
  Windows laptop). 1× H100 capacity flickers by the minute — poll
  `/instance-types` and launch into the window (see the capacity-watch pattern
  in the session logs); a 27b foldlisten cell needs its OWN box at a ≥5.5h cap
  (~4.3h PCIe, ~1.4h SXM5).

Load the token without echoing it:

```bash
LAMBDA_KEY_ONE=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
```

## Spend & GPU-hours

Real usage reconstructs from the audit log: `GET /api/v1/audit-events` (`launched`/`terminated` `event_time` + `additional_details.instance_type`), priced with `GET /api/v1/instance-types`. The dollar invoice is in the Lambda dashboard only.

**Budget cap (user-authorized, cumulative): $950 = 500 + 150 + 300.** Latest raise +$300 on 2026-07-07
to make the 27b Phase-4 transport feasible. Project-attributable spend to date and remaining headroom
reconstruct from the audit log (see "Spend & GPU-hours" above) — not restated here (it rots).

**Failed run addendum (2026-07-06): phase-4 2b transport launch `e032e06b…` failed** (cost/timing in the audit log). Lesson (proper home: a known-issue comment in the launcher, not here) — a multi-hour run driven from this workstation is fragile to a network drop / session end: the launcher dies, detached on-box results go unfetched, and the box bills until the on-box self-destruct backstop fires.
On launcher death, SSH-fetch from the still-live box BEFORE terminating; consider hardening the launcher (longer NOCONN tolerance + a fetch-only reattach; don't auto-terminate on LOST while the backstop is armed).

## Lambda Cloud API — endpoints leveraged

Base URL `https://cloud.lambda.ai/api/v1/`. Auth on every call:
`-H "Authorization: Bearer $LAMBDA_KEY_ONE"`.

| Method & path | Purpose | Body / notes |
|---|---|---|
| `GET /ssh-keys` | List keys on the account (also the auth smoke-test) | — |
| `POST /ssh-keys` | **Add** a public key | `{"name": "...", "public_key": "ssh-ed25519 AAAA… comment"}` → returns the key `id`. Names must be unique. |
| `GET /instance-types` | What's launchable + **where there's capacity** | Each entry has `instance_type` (price_cents_per_hour, gpu_description, specs) and `regions_with_capacity_available` (empty ⇒ no capacity). |
| `POST /instance-operations/launch` | **Launch** an instance | `{"region_name","instance_type_name","ssh_key_names":["..."],"name"}` → returns `data.instance_ids`. |
| `GET /instances/<id>` | Poll status / get the **IP** | `data.status` goes `booting` → `active`; `data.ip` populates during boot. |
| `POST /instance-operations/terminate` | **Tear down** (stops billing) | `{"instance_ids":["..."]}`. Run this when done. |

### Concrete calls used this session

Add the key:

```bash
PUB=$(cat ~/.ssh/lambda_ed25519.pub)
curl -sS -X POST https://cloud.lambda.ai/api/v1/ssh-keys \
  -H "Authorization: Bearer $LAMBDA_KEY_ONE" -H "Content-Type: application/json" \
  --data "{\"name\":\"latent_verify_helios\",\"public_key\":\"$PUB\"}"
# -> {"data":{"id":"fd0348a11da147fa9b213e624f1d230a","name":"latent_verify_helios",...}}
```

Check capacity:

```bash
curl -sS https://cloud.lambda.ai/api/v1/instance-types \
  -H "Authorization: Bearer $LAMBDA_KEY_ONE"
# filter to entries whose regions_with_capacity_available is non-empty
```

Launch (region fallback across the type's available regions):

```bash
curl -sS -X POST https://cloud.lambda.ai/api/v1/instance-operations/launch \
  -H "Authorization: Bearer $LAMBDA_KEY_ONE" -H "Content-Type: application/json" \
  --data '{"region_name":"us-west-2","instance_type_name":"gpu_1x_a100_sxm4","ssh_key_names":["latent_verify_helios"],"name":"latent_verify"}'
# -> {"data":{"instance_ids":["702fd675baa1447e8ee58eb774273484"]}}
```

Poll for the IP (background loop; exits when active):

```bash
ID=702fd675baa1447e8ee58eb774273484
for i in $(seq 1 40); do
  RESP=$(curl -sS https://cloud.lambda.ai/api/v1/instances/$ID \
           -H "Authorization: Bearer $LAMBDA_KEY_ONE")
  STATUS=$(echo "$RESP" | python -c 'import sys,json;print(json.load(sys.stdin)["data"].get("status",""))')
  IP=$(echo "$RESP" | python -c 'import sys,json;print(json.load(sys.stdin)["data"].get("ip") or "")')
  [ "$STATUS" = active ] && [ -n "$IP" ] && { echo "READY ip=$IP"; break; }
  sleep 15
done
```

## This session's instance

| field | value |
|---|---|
| SSH key name (Lambda) | `latent_verify_helios` (id `fd0348a11da147fa9b213e624f1d230a`) |
| local private key | `~/.ssh/lambda_ed25519` |
| instance id | `702fd675baa1447e8ee58eb774273484` |
| type | `gpu_1x_a100_sxm4` — A100 40 GB SXM4, 30 vCPU / 200 GiB |
| region | `us-west-2` |
| price | $1.99/hr |
| IP | `161.153.52.246` |
| login user | `ubuntu` |

## Connecting

```bash
ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/lambda_ed25519 ubuntu@161.153.52.246 nvidia-smi
```

Or via `~/.ssh/config` for a one-word handle:

```
Host lambda-lv
    HostName 161.153.52.246
    User ubuntu
    IdentityFile ~/.ssh/lambda_ed25519
    StrictHostKeyChecking accept-new
```

→ `ssh lambda-lv nvidia-smi`

## Teardown (do this when finished — it's billed per hour)

```bash
curl -sS -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -H "Authorization: Bearer $LAMBDA_KEY_ONE" -H "Content-Type: application/json" \
  --data '{"instance_ids":["702fd675baa1447e8ee58eb774273484"]}'
```

Removing the SSH key from the account (optional) is `DELETE /ssh-keys/<id>`.

## Reproduction note — `job_sycophancy.py` it/chat run (2026-06-17, A10)

The sycophancy it/chat half (`FRAMING_NOTES §11`) was run on the cheapest adequate
box, **`gpu_1x_a10`** ($1.29/hr, us-east-1) — 2b/2b-it is tiny and the job is
minutes. Stack = `circuit-tracer@041a9b2` (pulls transformer_lens). Two traps cost a
restart each; avoid them:

- **Use a plain venv** (`python3 -m venv .venv`), **not** `--system-site-packages`.
  The Lambda image's system `pandas` is built against a different `numpy` than the one
  pip pulls, so `import transformer_lens` throws `ValueError: numpy.dtype size changed`.
- **Do not** install the default PyPI torch — it resolves to `2.12.0+cu130`, too new
  for the A10 driver (570 / CUDA 12.8), so `torch.cuda.is_available()` returns `False`
  and the run **silently falls back to CPU**. Install the cu124 wheel:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124` (→ `2.6.0+cu124`,
  cuda True). Resolved interp stack this run: transformer_lens 3.2.1 / transformers 4.57.3
  (§10 records 3.4 / 5.12 — the pin drifted; faithfulness is behavioural, see next).
- **Validation gate:** run `--model base` first and confirm it reproduces the committed
  CPU controls (Δ_syc −4.55, salience +6.52, counter −3.03, bare −2.33) to bf16 rounding
  before trusting `--model it`. (This run: −4.62 / +6.58 / −2.98 / −2.30.)
- HF token (`HF_KEY_ONE`) is needed for the gated `google/gemma-2-2b{,-it}`. **Terminate**
  the instance when done and confirm no active instances — it bills per hour.
</content>
</invoke>
