from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    app_env: str
    allowed_hosts: tuple[str, ...]


def _parse_allowed_hosts(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("localhost", "127.0.0.1", "testserver")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL"),
        app_env=os.getenv("APP_ENV", "dev"),
        allowed_hosts=_parse_allowed_hosts(os.getenv("ALLOWED_HOSTS")),
    )
