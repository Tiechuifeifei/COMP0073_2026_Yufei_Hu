"""Shared WRDS connection helpers for fundamental_pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import wrds

SCRIPT_DIR = Path(__file__).resolve().parent


def load_local_env(env_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from fundamental_pipeline/.env without printing secrets."""
    path = env_path or SCRIPT_DIR / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def connect_wrds() -> wrds.Connection:
    """Open a WRDS connection using .env / environment credentials or .pgpass."""
    load_local_env()

    username = os.environ.get("WRDS_USERNAME", "").strip()
    password = os.environ.get("WRDS_PASSWORD", os.environ.get("PGPASSWORD", "")).strip()

    kwargs: dict[str, str] = {}
    if username:
        kwargs["wrds_username"] = username
    if password:
        kwargs["wrds_password"] = password

    try:
        return wrds.Connection(**kwargs)
    except EOFError as exc:
        raise RuntimeError(
            "WRDS connection requires credentials in non-interactive mode. "
            "Set WRDS_USERNAME and WRDS_PASSWORD in fundamental_pipeline/.env "
            "or configure ~/.pgpass, then rerun."
        ) from exc
