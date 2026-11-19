# Local service and single-VM deployment

The service source and commands in this guide refer to the
[GitLab service repository](https://gitlab.com/olip78/energy-optimization).
The public GitHub research repository also carries this guide and the report.

## Implemented boundary

The service runs fresh CatBoost inference and the existing deterministic MILP
at each operating hour. It does not read saved economic decisions. The offline
report and frozen backtests remain independent. No replanning threshold is enabled.
The original hourly settlement and overnight battery accounting are retained.

Four models form an immutable, checksummed release. The initial release uses
2024 only; its latest 28 days are validation and are not refitted. PV residual
training uses group-out-of-fold predictions of its compatible PV head. This
serving training configuration is versioned separately from the report's models.
Validation residuals provide descriptive uncertainty bands; these are not the
research scenario generator or calibrated conditional quantiles.

The prepared data contains historical forecast feature views and labels.
Providers gate input availability, disclose completed-hour PV only, and never
pass label columns to model prediction. Known auction-cleared prices become
available after the auction, separately from future delivery observations.
Research's conservative one-hour forecast issue convention is preserved.

March--June is replayed once. July--September loops with the July checkpoint's
model release restored atomically with virtual time. Cycles have separate
operational histories; training selects rows by virtual label cutoff. An audit
record and each state advance commit in one SQLite transaction. A shared
single-host lock serializes writers in the worker and both API slots.
SQLite lives on one local disk, never NFS and never multiple VM replicas.

Every 14 virtual calendar days the worker requests an idempotent training job.
It trains four candidates on newly available local history, logs them to MLflow,
and compares forecast MAE and a three-day economic replay with the champion.
The latter is a short smoke/evaluation gate, not a full statistical proof of
economic improvement. Candidate failure leaves the champion running. A new
release requires manual promotion at a day boundary; automatic promotion and
continuous online shadow inference are intentionally not implemented yet.
The worker pauses virtual progression while it trains, which keeps this demo
small and deterministic. It is not intended for uninterrupted real market control.

Data monitoring currently measures feature completeness, non-finite values,
standardized mean shift against release training statistics and all-zero PV days.
Model monitoring records completed-day forecast MAE. The immutable historical
file provider has no live source freshness SLA; a displayed zero-minute lag
means local historical input, not an independently measured external feed.

## Native local launch

Run from the repository root with Python 3.11. No server credentials are needed.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,train,service]'

# Export a small portable input snapshot from the existing research checkout.
.venv/bin/energy-service --root runtime prepare --source-root /path/to/research/project
.venv/bin/energy-service --root runtime init --start 2025-03-01 --iterations 200
.venv/bin/energy-service --root runtime doctor

# Optional: compute March-June once, including scheduled candidate jobs,
# and stop exactly at the first July checkpoint.
.venv/bin/energy-service --root runtime warmup

# Terminal 1: serves the existing dashboard and its API.
.venv/bin/energy-service --root runtime serve --port 8080
# Terminal 2: one virtual operating hour per five wall-clock seconds.
.venv/bin/energy-service --root runtime worker --interval 5
```

Open `http://127.0.0.1:8080`. The dashboard polls the service every five seconds.
Its test clock controls are hidden in service mode because the worker owns progression.
Tomorrow's plan is withheld until its D-1 inputs become available. Actual PV
and intraday prices beyond committed hours remain null. At a new day's 06:00
state no hours have executed yet. The standalone static dashboard remains
available via `make dashboard-serve` and is a separate frozen demonstration.

`runtime/data/manifest.json` contains checksums. Export to a new root when
preparing another snapshot; do not overwrite a running dataset.

Finite replay, model lifecycle and inspection:

```bash
.venv/bin/energy-service --root runtime worker --interval 0 --steps 32
.venv/bin/energy-service --root runtime train
.venv/bin/energy-service --root runtime promote --release RELEASE_ID
.venv/bin/energy-service --root runtime rollback --release PREVIOUS_RELEASE_ID
```

`train`, `promote` and `rollback` require a day boundary. Candidate training
uses labels strictly before the current virtual day. Promotion also requires
passing forecast and economic gates. Local MLflow metadata/artifacts are under
`runtime/mlflow`; local model inference reads `runtime/releases` and its manifest.
The MLflow run and four registered model versions record provenance; serving
does not query mutable registry aliases during an hourly decision.

## Docker Compose on one VM

Compose is the deployment target for this first VM. k3s remains an optional
later exercise, not a prerequisite. Start with Linux, Docker Engine and Compose
v2. For ENERGY alone, 4 vCPU / 8 GB RAM is an initial sizing estimate, not a
measured memory guarantee. When also hosting YNYT, 16 GB is a comfortable
starting point; 8 GB is a budget option with prepared taxi aggregates, one
active API slot and serialized training jobs. Allow 40--60 GB SSD for a demo
without raw taxi archives. Measure container peaks before reducing memory
limits. No GPU is used.

Before starting, create `.env` from `.env.example`, set a random admin token,
and set the runtime UID/GID to the directory owner's numeric IDs. Keep `.env`
out of Git. Use a local runtime disk with enough room for datasets and releases.

```bash
mkdir -p runtime/mlflow
docker compose config --quiet
docker compose build

# Export from a mounted research directory (ENERGY_SOURCE_DIR in .env).
docker compose run --rm tools prepare --source-root /source
docker compose up -d mlflow
docker compose run --rm tools init --start 2025-03-01 --iterations 200
docker compose up -d api-blue worker proxy
```

The bind-mounted runtime directory must be writable by ENERGY_UID/ENERGY_GID.
On a fresh server use `/srv/energy/app` for this checkout and
`ENERGY_RUNTIME_DIR=/srv/energy/runtime` for data and state. You can copy ONLY
the prepared `runtime/data` snapshot from the laptop, then run container `init`
on the server. Do not copy laptop MLflow SQLite metadata: its local artifact
URIs refer to laptop paths. Server initialization registers the release with
the server MLflow instance and container artifact paths.

By default only `127.0.0.1:8080` is public-facing and MLflow binds
`127.0.0.1:5000`. API slots have no host ports. Use SSH forwarding for the first
VM validation. Configure domain/TLS before changing ENERGY_BIND to a public
interface. The proxy permits reads and blocks replay administration and audit
routes. Admin API access also requires `Authorization: Bearer ...`; if the
token is empty, administration is disabled. A public portfolio viewer cannot
change virtual state, trigger training or promote models. MLflow stays private.

## Blue/green and CI/CD

GitLab CI/CD is prepared in `.gitlab-ci.yml`; see [GitLab setup](gitlab.md)
for Registry authentication, protected variables and the first manual rollout.
The GitHub workflows below remain available as an alternative.

```bash
docker pull ghcr.io/OWNER/REPO:COMMIT_SHA
python3 scripts/switch_slot.py green --image ghcr.io/OWNER/REPO:COMMIT_SHA
# Later rollback the application using the old image in the inactive slot.
python3 scripts/switch_slot.py blue --image ghcr.io/OWNER/REPO:PREVIOUS_SHA
```

The script refuses the active slot, starts the inactive API, waits for its
model-aware readiness, checks Nginx configuration and reloads the proxy. A proxy
failure restores the original upstream. The sole worker is updated after the
switch. No schema migrations are performed by a rollout: backward-incompatible
DB/model schema changes require a separately reviewed migration. The previous
API stays available for rollback. Only one worker may be running.

CI runs lint/tests and builds the image. Main/tag builds publish a SHA-tagged
image to GHCR with GitHub's built-in token; nothing is published from this local
session. The separate manual Deploy VM workflow uses the `production`
environment and these secrets once a VM exists:

- `VM_HOST`, `VM_USER`, `VM_SSH_KEY`, `VM_KNOWN_HOSTS`;
- server-side GHCR login if the package is private.

Bootstrap `/srv/energy/app`, runtime, `.env`, and the initial active upstream
once before running CD. The workflow preserves credentials, active-slot config
and runtime. Build the SHA image before dispatching deployment for that commit.
Keep environment approval enabled for the first deployments.

## Checks and operations

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/ready
curl --fail http://127.0.0.1:8080/v1/state/current
curl --fail http://127.0.0.1:8080/metrics
docker compose logs --tail=100 worker api-blue mlflow
.venv/bin/python -m pytest -q
```

`/health` proves the HTTP process is alive; `/ready` additionally loads and
checks the snapshot and all active model artifacts. `/metrics` exports replay
step/cycle, pause and error indicators. JSON logs include run/decision/release
IDs; Docker rotates and compresses them with the `local` driver (three 2 MB
files per container, approximately 6 MB before compression plus overhead).
The demo defaults to `ENERGY_LOG_LEVEL=WARNING`; use `INFO` temporarily for
diagnostics. Nginx access logging is disabled. Detailed current-cycle metrics and release candidates
are in the dashboard/API. Replay errors preserve the previous committed step
and are audited; the worker retries without silently advancing a failed hour.

Docker log-driver behavior and rotation options are described in the
[Docker local driver documentation](https://docs.docker.com/engine/logging/drivers/local/).

For an initial backup, stop `worker` and both API slots, copy the entire runtime
directory including SQLite WAL files, then restart the previous active stack.
Keep snapshot hashes and release manifests with the backup. Restore into a new
directory and run `doctor` before exposing it. Online backups remain a follow-up.

## Bounded storage and demo maintenance

### Berlin operating clock

Compose defaults to `ENERGY_CLOCK=berlin`. One complete historical operating
day maps to one real Berlin operating day, from 06:00 to the next 06:00.
At 07:00 the 06:00--07:00 interval is committed, and so on through 22:00.
After 22:00 the full completed context and graphs remain visible overnight;
at 06:00 the next complete historical day opens. September's final completed
day is also held overnight before July's checkpoint is restored next morning.
Berlin DST changes are handled by `zoneinfo`, independently of historical
summer-date UTC offsets.

The worker polls every 30 seconds. On first attachment the current historical
day is assigned to the current Berlin operating day; the saved progress is
never rewound. A persistent `wall_day` anchor survives restarts. Missed
intervals/days are calculated in sequence, with bounded catch-up per tick;
long downtimes or training may therefore temporarily lag real time. A paused
replay stays paused. UI distinguishes real Berlin time, simulation time and
the last settled interval. Simulation buttons are hidden on the server.

The native `worker --interval ...` and `warmup` retain their finite/test replay
behavior. For accelerated Compose testing explicitly set `ENERGY_CLOCK=interval`.
Switching back from accelerated mode to Berlin mode after an earlier attachment
retains the existing anchor; do not mix clock modes on the production runtime.

### Retention

The worker applies retention at startup, at each loop transition and once per
wall-clock hour, including while paused. It keeps the real March--June chart
prehistory, the current and previous operating cycles, and the latest 2,000
audit rows. Initialization and lifecycle events for the retained chart history
are additionally preserved. Older completed/failed cycle training jobs are
removed; running jobs, current state and model pointers remain unchanged.
Maintenance errors are logged and do not stop replay. This is a bounded demo
audit, not permanent compliance storage.

Preview or apply explicitly without loading datasets/models:

```bash
docker compose run --rm tools cleanup
docker compose run --rm tools cleanup --apply --compact
```

The regular cleanup reuses freed SQLite pages; nightly `--compact` also returns
space to disk and truncates the WAL. It holds the existing single-host writer
lock, so state advances may briefly wait. Dataset snapshots, model releases
and MLflow artifacts are never deleted by this command. Deterministic release
IDs reuse existing models for repeated cutoffs/settings in the finite replay
instead of producing copies every loop. Changed data/settings can still add
releases; automatic MLflow artifact garbage collection is not implemented.

Host maintenance is prepared for Linux at `/srv/energy/app`:

```bash
# Preview commands; no Docker changes are made.
bash scripts/cleanup_host.sh
# Install on the VM, after initialization and with its .env configured.
sudo install -m 644 deploy/systemd/energy-cleanup.service /etc/systemd/system/
sudo install -m 644 deploy/systemd/energy-cleanup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now energy-cleanup.timer
```

The timer runs daily around 04:00 server time. It compacts operational history,
prunes ENERGY-labelled images older than seven days that no container uses,
and removes unused host build cache older than seven days with a 512 MB cache
retention target. This target is not a hard total cache cap: newer/in-use cache
can remain. Old images need the `io.energy.demo=true` label from the prepared
Dockerfile. All containers, including the stopped rollback slot, keep their
images referenced. A second pass removes old untagged, unreferenced images
across the demo host, including replaced YNYT `:latest` images. It does not
prune containers, volumes, unrelated tagged images or bind-mounted data. Build cache is shared across projects and is
disposable. The script stops on errors and uses `flock` against duplicate runs.
No cleanup has been run against the laptop's live runtime or Docker daemon.

For the VM's system journal (also used by the current YNYT stack), install the
optional host-wide retention policy:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
sudo install -m 644 deploy/journald-demo.conf /etc/systemd/journald.conf.d/demo-retention.conf
sudo systemctl restart systemd-journald
sudo journalctl --rotate --vacuum-size=100M --vacuum-time=3d
```

This limits persistent journal use to roughly 100 MB and retains up to three
days, with a separate 50 MB runtime-journal budget. It affects the whole VM;
it does not delete application datasets. YNYT code/images are not modified by
these ENERGY changes. Its cron currently appends to `/tmp/cron.log`, which is
outside Docker/journald rotation. A prepared `deploy/ynyt-logging.patch` redirects
cron output to Docker stdout/stderr and enables a bounded log driver for that
service. Apply it in the YNYT checkout, rebuild its cron image and recreate that
service during YNYT deployment; it has not been applied to the external checkout.

For CPU/RAM, the serving release cache is capped at four loaded releases,
CatBoost inference uses at most two threads per prediction, and BLAS/OpenMP
defaults to one thread. MLflow uses one server worker. `/tmp` is a 256 MB
tmpfs limit per application container, not a memory reservation. Build images
in CI and pull them to the VM; keep replay slow and avoid overlapping YNYT
preprocessing with ENERGY training. An inactive API can be stopped after a
successful blue/green switch (`docker compose stop api-blue` when green is
active, or conversely); keeping its stopped container preserves the rollback
image. Load levels alone do not establish a safe smaller RAM limit.

The complete four-month pseudo-history is a real replay run, not a fabricated
chart seed. Start from March and let the worker build it; the July checkpoint
is captured at the boundary. `--start 2025-07-01` is a smoke/demo shortcut that
uses the initial release as checkpoint and deliberately has no pseudo-history.
With the currently prepared source snapshot, March--June has 111 complete
delivery days and July--September has 91. Incomplete source days are quarantined;
calendar gaps are recorded in interval audit rows. They are not presented as
successful operation or filled using saved decisions.
