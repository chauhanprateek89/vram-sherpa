# Cloudflare Workers (Containers)

Runs the same container image as a Cloudflare Container behind a Worker. This is
what the `Workers Builds: vram-sherpa` check on every pull request builds.

## Layout

| File | Role |
|---|---|
| `wrangler.toml` | Worker + container config |
| `worker/index.js` | Worker entrypoint; exports the `VramSherpaApp` container class |
| `Dockerfile` | the container image, shared with the Docker Compose deploys |
| `package.json` | pins `@cloudflare/containers` and `wrangler` |

A Container needs three things that must agree, or the build fails: a
`[[containers]]` entry, a `[[durable_objects.bindings]]` entry, and a Durable
Object class of that name actually exported from the Worker. The config that was
previously in the repo declared the first two but no Worker and no class, which
is why the check failed on every branch — including the one that added
`wrangler.toml`.

## Database

The container runs **SQLite on its own ephemeral disk**, not Postgres. The
catalog is read-only and rebuilt from the seed files baked into the image on
every start, so there is nothing to persist, no database to provision and no
secret to store. This is the one real difference from the homelab deploy, which
runs Postgres.

## Hostname

`APP_ENV=prod` rejects any `Host` outside its allowlist with 400, so the
hostname the Worker is served on must be in `PUBLIC_HOSTNAME`:

```toml
[vars]
PUBLIC_HOSTNAME = "vram-sherpa.<your-subdomain>.workers.dev"
```

The Worker folds that into the container's `ALLOWED_HOSTS`. Leave it empty and
every request 404s at the Worker or 400s at the container.

The Worker deliberately forwards the **original** hostname rather than rewriting
it to localhost: Starlette's `url_for` builds absolute CSS and JS URLs from the
request, so a rewritten host would point every asset at localhost.

For the same reason the Worker sets `X-Forwarded-Proto` explicitly and the image
runs uvicorn with `--proxy-headers`. Without it the app emits `http://` asset
URLs onto an `https://` page and browsers block them as mixed content — which
breaks all styling and interactivity while the HTML still returns 200.

## Instance size

`instance_type = "basic"` (1/4 vCPU, 1 GiB memory, 4 GB disk). An image may not
exceed the instance's disk, and this one is ~800 MB; `lite` offers 2 GB disk but
only 256 MiB memory, which is too little for Python, FastAPI and SQLAlchemy.

## Local checks

Wrangler needs **Node.js 22+**, and `--dry-run` still builds the image, so
Docker with buildx must be available:

```bash
npm install
npx wrangler deploy --dry-run    # validates config, bundles, builds the image
```

A successful dry-run prints the bindings and the container it found:

```
env.VRAM_SHERPA (VramSherpaApp)   Durable Object
env.PUBLIC_HOSTNAME               Environment Variable
- vram-sherpa-vramsherpaapp (./Dockerfile)
```

To exercise the image with exactly the environment the Worker injects:

```bash
docker run --rm -p 8000:8000 \
  -e APP_ENV=prod \
  -e DATABASE_URL="sqlite+pysqlite:////tmp/vramsherpa.db" \
  -e ALLOWED_HOSTS="localhost,127.0.0.1,<your hostname>" \
  ghcr.io/chauhanprateek89/vram-sherpa:latest
```

## Relationship to the homelab deploy

These are two independent deploy targets for one image: Workers builds on every
pull request, while `docker-compose.prod.yml` runs the released image on the
homelab behind the existing cloudflared tunnel. Keeping both means the Dockerfile
has to stay self-sufficient — it seeds and serves on its own, with no compose
file layering a command on top.
