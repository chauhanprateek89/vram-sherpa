"""ADR-0002 / ADR-0003: every static file and template ships in the installed package.

The Docker image runs `pip install .` (not editable), so a file under `static/` or
`templates/` that no `[tool.setuptools.package-data]` glob matches is missing from the image
while tests, which import from `src/`, still pass (HLD "Packaging trap").
"""

from __future__ import annotations

import os
import tomllib
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "vramsherpa"
SHIPPED_DIRS = ("static", "templates")

# The globs before ADR-0002 and ADR-0003 added fonts and images.
OLD_GLOBS = ["templates/*.html", "static/css/*.css", "static/js/*.js"]


def _package_data_globs() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    return config["tool"]["setuptools"]["package-data"]["vramsherpa"]


def _shipped(package_dir: Path, patterns: list[str]) -> set[str]:
    """Files the patterns ship, expanded as setuptools' `build_py.find_data_files` does:
    `glob(..., recursive=True)` relative to the package directory, files only. So `*` does
    not cross `/` and does not match a leading dot; only `**` crosses directories."""
    matched: set[str] = set()
    for pattern in patterns:
        for path in glob(os.path.join(package_dir, pattern), recursive=True):
            if os.path.isfile(path):
                matched.add(Path(path).relative_to(package_dir).as_posix())
    return matched


def _unshipped(package_dir: Path, patterns: list[str]) -> list[str]:
    """Files under `static/` and `templates/` that no pattern ships."""
    present = {
        path.relative_to(package_dir).as_posix()
        for top in SHIPPED_DIRS
        for path in (package_dir / top).rglob("*")
        if path.is_file()
    }
    return sorted(present - _shipped(package_dir, patterns))


def test_package_data_lists_fonts_and_images_and_keeps_existing_globs() -> None:
    globs = _package_data_globs()
    for pattern in [*OLD_GLOBS, "static/fonts/*", "static/img/*"]:
        assert pattern in globs


def test_every_static_file_and_template_matches_a_package_data_glob() -> None:
    missing = _unshipped(PACKAGE_DIR, _package_data_globs())
    assert not missing, (
        "These files would be missing from the installed package (add a glob under "
        f"[tool.setuptools.package-data] vramsherpa in pyproject.toml): {missing}"
    )


def test_guard_catches_a_file_the_old_globs_would_not_ship(tmp_path: Path) -> None:
    for rel in ("templates/base.html", "static/css/styles.css", "static/fonts/x.woff2"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")

    assert _unshipped(tmp_path, OLD_GLOBS) == ["static/fonts/x.woff2"]
    assert _unshipped(tmp_path, _package_data_globs()) == []


def test_guard_does_not_let_star_cross_a_directory(tmp_path: Path) -> None:
    for rel in ("static/img/favicon.svg", "static/img/sub/nested.png", "templates/a/b.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")

    assert _unshipped(tmp_path, ["static/img/*", "templates/*.html"]) == [
        "static/img/sub/nested.png",
        "templates/a/b.html",
    ]
