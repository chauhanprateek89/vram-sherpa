import { DurableObject } from "cloudflare:workers";

const CONTAINER_PORT = 8000;
// Stop the container once it has been idle this long. Containers bill for
// runtime, so without this it would run forever after the first request.
const IDLE_SHUTDOWN_MS = 15 * 60 * 1000;

/**
 * Runs the VRAM Sherpa FastAPI image as a Cloudflare Container.
 *
 * This deliberately uses the built-in `cloudflare:workers` module rather than
 * the `@cloudflare/containers` helper. The build environment installs Python
 * for this repo (pyproject.toml and .python-version are present) and may not
 * run `npm install`, in which case an npm import fails to bundle with
 * "Could not resolve". A built-in module cannot fail that way.
 *
 * The catalog is read-only and rebuilt from the seed files baked into the image
 * on every start, so the container runs SQLite on its own ephemeral disk: no
 * external database to provision and no secret to store.
 */
export class VramSherpaApp extends DurableObject {
  async fetch(request) {
    const container = this.ctx.container;

    if (!container.running) {
      container.start({
        env: {
          APP_ENV: "prod",
          DATABASE_URL: "sqlite+pysqlite:////tmp/vramsherpa.db",
          // APP_ENV=prod rejects any Host outside this list with 400, and the
          // real hostname has to stay intact because Starlette's url_for builds
          // absolute CSS and JS URLs from it. Set PUBLIC_HOSTNAME in [vars].
          ALLOWED_HOSTS: ["localhost", "127.0.0.1", this.env.PUBLIC_HOSTNAME]
            .filter(Boolean)
            .join(","),
        },
      });
    }

    // Push the idle deadline out on every request.
    await this.ctx.storage.setAlarm(Date.now() + IDLE_SHUTDOWN_MS);

    return container.getTcpPort(CONTAINER_PORT).fetch(request);
  }

  async alarm() {
    if (this.ctx.container.running) {
      await this.ctx.container.destroy();
    }
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // The container is reached with its original URL, so without an explicit
    // hint the app reports the scheme as http and url_for emits http:// asset
    // URLs onto an https page, which browsers block as mixed content. uvicorn
    // runs with --proxy-headers, so this header decides the scheme it reports.
    const headers = new Headers(request.headers);
    headers.set("X-Forwarded-Proto", url.protocol === "https:" ? "https" : "http");

    const id = env.VRAM_SHERPA.idFromName("vram-sherpa-singleton");
    return env.VRAM_SHERPA.get(id).fetch(new Request(request, { headers }));
  },
};
