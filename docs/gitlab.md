# GitLab CI/CD on the existing VM

The prepared pipeline runs on GitLab.com or a self-managed GitLab with a
Container Registry and a runner supporting privileged Docker-in-Docker.
GitLab itself and its build runner need not run on the small demo VM.

## Pipeline

1. Branch and merge-request pipelines run Ruff and pytest with the pinned
   service dependencies. JUnit reports expire after seven days.
2. Build an amd64 image on an amd64 runner, check `/health` without private
   datasets, then push `$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA` to GitLab Registry.
3. The default branch offers a manual production deployment. Its resource
   group serializes deployments. The deployment copies code/configuration,
   pulls the SHA image and selects the inactive API slot automatically.
   After model-aware readiness, Nginx switches and the sole worker is updated.

The VM retains `.env`, `deploy/active`, runtime datasets, models and SQLite
state. Failed API readiness leaves the old upstream in place; failed Nginx
validation/reload restores it. Worker-update failures require inspection:
the script does not claim transactional rollback of the entire stack.
Changing MLflow configuration requires a separate service recreation; ordinary
application deployment does not restart MLflow. Schema migrations are separate.

## GitLab project and variables

The service project is [olip78/energy-optimization](https://gitlab.com/olip78/energy-optimization),
with Container Registry enabled. For a new installation, create an empty
project and review files for publication before connecting its remote.
Runtime data and credentials remain excluded from Git.

Protect the default branch and configure these variables under Settings > CI/CD:

| Variable | Type | Value |
| --- | --- | --- |
| `VM_HOST` | Variable, protected | `81.26.188.18` |
| `VM_USER` | Variable, protected | `olip78` |
| `VM_SSH_KEY` | File, protected | Dedicated deployment private key, ending in a newline |
| `VM_KNOWN_HOSTS` | File, protected | Previously verified SSH host-key record |

Use a dedicated deployment key. Configure the corresponding public key for VM
login and test it from the Mac first. If Yandex OS Login governs access, configure
the key through that access mechanism. Do not disable host-key verification or
generate trusted host records from unverified `ssh-keyscan` inside CI.
Deploy variables should only be available to trusted production/default-branch
jobs. A runner can access job secrets: use trusted runners.

Before deployment, authenticate Docker on the VM once with a GitLab deploy token
scoped to `read_registry` (not a short-lived CI job token):

```bash
docker login registry.gitlab.com -u YOUR_DEPLOY_TOKEN_USERNAME
```

Enter the token at the password prompt. Use your own registry hostname for a
self-managed GitLab. The VM must already have the initialized runtime and the
working `/srv/energy/app/.env`; CI does not initialize or reset simulation.

Push the reviewed source to the selected GitLab project, wait for tests/build,
then run the `deploy` job. Check `/ready`, dashboard, worker logs and replay
progress. Confirm the previous slot remains available for application rollback.

To roll back an application image already present in the registry:

```bash
cd /srv/energy/app
docker pull registry.gitlab.com/GROUP/PROJECT:PREVIOUS_SHA
python3 scripts/switch_slot.py auto --image registry.gitlab.com/GROUP/PROJECT:PREVIOUS_SHA
```

After the first verified deployment, automatic deployment can be enabled by
changing the default-branch deploy rule from `when: manual` to `when: on_success`.
Keep `resource_group` and enable GitLab's protection against outdated deployment
jobs before using automatic delivery. The first complete pipeline and manual
production rollout were verified on 19 November 2026: tests, JavaScript checks,
image build/smoke test, Registry publication and deployment all passed. The VM
switched from blue to green with its model release and runtime preserved;
daily host maintenance was subsequently enabled with systemd.

References: [Docker builds](https://docs.gitlab.com/ci/docker/using_docker_build/),
[SSH keys](https://docs.gitlab.com/ci/jobs/ssh_keys/),
[serialized deployments](https://docs.gitlab.com/ci/resource_groups/).
