# Release and deploy

## Model

Continuous deployment, pull-based. Nothing in CI can reach the homelab.

```
merge to main  ->  CI gates  ->  build + push GHCR  ->  smoke test  ->  tag + GitHub Release
                                                                              |
                                        homelab watchtower polls :latest  <---+
                                                   |
                                        docker pull + recreate web (~5 min)
```

The server needs **outbound network only**. There is no inbound path, no deploy key in
GitHub, and no self-hosted runner. This matters because the repo is public: a
self-hosted runner would let any fork PR execute code on the box that also runs
`ollama` and `chatgpt-rag-kb`.

## Versioning

`v<major>.<minor>.<commit-count-on-main>`, e.g. `v0.1.36`.

The major/minor pair comes from `version` in `pyproject.toml`; bump it there to start a
new line. The patch number is `git rev-list --count HEAD`, which is monotonic and
survives workflow re-creation (unlike `github.run_number`).

Every release publishes three tags to `ghcr.io/chauhanprateek89/vram-sherpa`:

| Tag | Use |
|---|---|
| `v0.1.36` | Immutable release. Pin this to roll back. |
| `sha-abc1234` | Exact commit, for tracing a running container back to source. |
| `latest` | What the homelab tracks. |

## What gates a release

`release.yml` calls `ci.yml` via `workflow_call`, so main is gated on exactly the same
checks as a PR — not a copy that can drift. After the image is pushed, a smoke test runs
the real published image against a real Postgres and fails the release if any of these
do not hold:

- `/healthz` responds
- the catalog actually seeds (`Seeded catalog_version=` in the logs)
- the results page renders
- `APP_ENV=prod` rejects an unknown `Host` header with 400

A broken image therefore never gets tagged or released, so `latest` never moves to it.

## Catalog refreshes

`data/seed_*.json` is baked into the image, so the catalog is part of the immutable
release and a rollback restores the exact catalog that shipped with it. The weekly
`catalog-refresh.yml` opens a PR; merging it is a normal merge to main and therefore
cuts a release and deploys automatically.

## First-time server setup

Requires only Docker (already present, usable without sudo). The GHCR package must be
**public** so the server needs no registry credentials — set that once, after the first
release, under the package settings on GitHub.

```bash
mkdir -p ~/src/vram-sherpa && cd ~/src/vram-sherpa
curl -fsSLO https://raw.githubusercontent.com/chauhanprateek89/vram-sherpa/main/docker-compose.prod.yml

cat > .env <<'EOF'
APP_ENV=prod
PUBLIC_HOSTNAME=vramsherpa.example.com
ALLOWED_HOSTS=localhost,127.0.0.1,web
POSTGRES_PASSWORD=<generate one>
EOF
chmod 600 .env

docker compose -f docker-compose.prod.yml up -d
```

Then, in the Cloudflare Zero Trust dashboard, add a Public Hostname on the **existing**
tunnel pointing at `http://localhost:8000`. Do not start a cloudflared sidecar container
on this host: cloudflared already runs there as a systemd service, and a second tunnel
would register alongside it rather than replacing it.

`PUBLIC_HOSTNAME` must match the DNS name exactly. Because `APP_ENV=prod`, any other
`Host` is rejected with 400 — including Cloudflare's, if the name is wrong.

## Rolling back

```bash
cd ~/src/vram-sherpa
echo "IMAGE_TAG=v0.1.35" >> .env            # the known-good release
docker compose -f docker-compose.prod.yml up -d web
docker compose -f docker-compose.prod.yml stop watchtower   # stop it re-updating
```

Remove the `IMAGE_TAG` line and restart watchtower to resume tracking releases.

## Operational notes

- **Watchtower scope.** It runs with `--label-enable`, so it only touches containers
  carrying `com.centurylinklabs.watchtower.enable=true`. Only `web` has it. `ollama` and
  `db` are never auto-updated — an unattended Postgres major bump would be worse than a
  stale one.
- **Watchtower holds the Docker socket**, which is effectively root on the host. That is
  the cost of pull-based deploys without inbound access; `--label-enable` is what keeps
  its blast radius to this stack.
- **Schema changes.** `create_all` creates missing tables but never alters existing ones,
  so a release that adds a column fails to seed against an existing volume. See
  "Upgrading an existing database" in the README.
- **Deploy lag** is up to the 5-minute poll interval. `docker compose -f
  docker-compose.prod.yml pull web && ... up -d web` forces it immediately.
