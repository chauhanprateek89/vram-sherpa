import { Container, getContainer } from "@cloudflare/containers";

/**
 * Runs the VRAM Sherpa FastAPI image as a Cloudflare Container.
 *
 * The catalog is read-only and rebuilt from the seed files baked into the image
 * on every start, so the container runs SQLite on its own ephemeral disk. There
 * is no external database to provision and no secret to store.
 */
export class VramSherpaApp extends Container {
  defaultPort = 8000;
  sleepAfter = "15m";

  constructor(ctx, env) {
    super(ctx, env);

    // The container is reached with the public hostname intact, because
    // Starlette's url_for builds absolute asset URLs from it. APP_ENV=prod
    // rejects any Host outside this list with 400, so the hostname the Worker
    // is served on has to be in it: set PUBLIC_HOSTNAME in [vars].
    const allowedHosts = ["localhost", "127.0.0.1"];
    if (env.PUBLIC_HOSTNAME) {
      allowedHosts.push(env.PUBLIC_HOSTNAME);
    }

    this.envVars = {
      APP_ENV: "prod",
      DATABASE_URL: "sqlite+pysqlite:////tmp/vramsherpa.db",
      ALLOWED_HOSTS: allowedHosts.join(","),
      // Honoured once config.py gains PUBLIC_HOSTNAME; harmless before that,
      // since the hostname is already covered by ALLOWED_HOSTS above.
      PUBLIC_HOSTNAME: env.PUBLIC_HOSTNAME || "",
    };
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // containerFetch forwards request.url with `https:` rewritten to `http:`.
    // Starlette's url_for builds absolute asset URLs from the request scheme, so
    // without an explicit hint the container emits http:// CSS and JS onto an
    // https page, which browsers block as mixed content. uvicorn runs with
    // --proxy-headers so this header decides the scheme it reports.
    const headers = new Headers(request.headers);
    headers.set("X-Forwarded-Proto", url.protocol === "https:" ? "https" : "http");

    return getContainer(env.VRAM_SHERPA).fetch(new Request(request, { headers }));
  },
};
