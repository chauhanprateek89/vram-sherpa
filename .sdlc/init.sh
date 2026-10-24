#!/usr/bin/env bash
# Start VRAM Sherpa locally for testing and wait until /healthz reports ready.
#
# Local only: uses docker-compose.yml (builds from source, Postgres in a container).
# Never touches docker-compose.prod.yml, GHCR or the homelab.
# Idempotent: exits 0 at once if the local app is already healthy.
# Stop with: docker compose -f docker-compose.yml down   (add -v to reset the database)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_URL="http://localhost:8000"
HEALTH_URL="${LOCAL_URL}/healthz"
TIMEOUT_SECONDS=120
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/docker-compose.yml")

healthy() {
  curl -fsS -m 5 "$HEALTH_URL" >/dev/null 2>&1
}

if healthy; then
  echo "Already healthy: $LOCAL_URL"
  exit 0
fi

echo "Starting local stack (docker compose up -d --build)..."
"${COMPOSE[@]}" up -d --build

deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if healthy; then
    echo "Healthy: $LOCAL_URL"
    exit 0
  fi
  sleep 3
done

echo "App did not become healthy within ${TIMEOUT_SECONDS}s ($HEALTH_URL)." >&2
"${COMPOSE[@]}" ps >&2 || true
"${COMPOSE[@]}" logs --tail 40 web >&2 || true
echo "If seeding failed with UndefinedColumn, the local volume has an old schema:" >&2
echo "  docker compose -f docker-compose.yml down -v && .sdlc/init.sh" >&2
exit 1
