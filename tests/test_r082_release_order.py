"""R-082: `latest` and the version image tag move only after the release smoke test passes.

Static checks on `.github/workflows/release.yml`; no network, no database (rule 19).
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"
SMOKE_STEP = "Smoke test the published image"
VERSION_REF = "steps.version.outputs.version"
DIGEST_REF = "steps.build.outputs.digest"


@pytest.fixture(scope="module")
def steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow["jobs"]["release"]["steps"]


def _smoke_index(steps: list[dict]) -> int:
    names = [step.get("name") for step in steps]
    assert SMOKE_STEP in names, f"job release has no step named {SMOKE_STEP!r}"
    return names.index(SMOKE_STEP)


def _text(step: dict) -> str:
    """The parts of a step that could push or tag an image: `with.tags` and `run`."""
    tags = (step.get("with") or {}).get("tags") or ""
    return f"{tags}\n{step.get('run') or ''}"


def _pushes_release_tag(text: str) -> bool:
    if ":latest" in text or "imagetools" in text:
        return True
    # A tag built from the version output, e.g. `name:${{ steps.version.outputs.version }}`.
    for line in text.splitlines():
        if VERSION_REF in line and ":${{" in line.replace(" ", "") and "outputs.name" in line:
            return True
    return False


def test_workflow_parses_and_has_smoke_step(steps):
    _smoke_index(steps)


def test_no_release_tag_pushed_before_smoke_test(steps):
    smoke = _smoke_index(steps)
    for step in steps[:smoke]:
        assert not _pushes_release_tag(_text(step)), (
            f"step {step.get('name') or step.get('uses')!r} pushes `latest` or the version tag "
            "before the smoke test"
        )


def test_build_step_pushes_only_sha_tag(steps):
    build = next(s for s in steps if s.get("name") == "Build and push")
    assert build.get("id") == "build"
    assert build["uses"] == "docker/build-push-action@v6"
    tags = [t.strip() for t in build["with"]["tags"].splitlines() if t.strip()]
    assert tags == ["${{ steps.image.outputs.name }}:${{ steps.version.outputs.sha_tag }}"]


def test_smoke_test_runs_by_digest(steps):
    smoke = steps[_smoke_index(steps)]
    assert DIGEST_REF in smoke["run"]
    assert '"${{ steps.image.outputs.name }}@${{ steps.build.outputs.digest }}"' in smoke["run"]
    assert "if" not in smoke and "continue-on-error" not in smoke


def test_one_tagging_step_after_smoke_and_before_git_tag(steps):
    smoke = _smoke_index(steps)
    tagging = [
        i for i, s in enumerate(steps) if i > smoke and "imagetools create" in (s.get("run") or "")
    ]
    assert len(tagging) == 1, "want exactly one `imagetools create` step after the smoke test"
    idx = tagging[0]
    step = steps[idx]
    run = step["run"]
    assert ":latest" in run
    assert VERSION_REF in run
    assert DIGEST_REF in run
    assert "if" not in step
    assert "continue-on-error" not in step

    git_tag = [i for i, s in enumerate(steps) if "git tag" in (s.get("run") or "")]
    assert git_tag, "no step runs `git tag`"
    assert idx < min(git_tag)
    release = steps[min(git_tag)]
    assert "if" not in release and "continue-on-error" not in release


def test_smoke_checks_unchanged(steps):
    run = steps[_smoke_index(steps)]["run"]
    for needle in (
        "/healthz",
        "Seeded catalog_version=",
        "X-Forwarded-Proto: https",
        "evil.example.com",
        "/static/",
        "/results?vram_gb=24&context_tokens=4096",
        "summary-banner",
        "Manual VRAM (24 GB)",
    ):
        assert needle in run, f"smoke test lost the {needle!r} check"
    assert "cache-control" in run.lower()

    # The /results check must see the real status: a redirect is a failure, not a pass
    # (ADR-0013 point 4).
    results_lines = [line for line in run.splitlines() if "/results" in line and "curl" in line]
    assert results_lines, "no curl line requests /results"
    for line in results_lines:
        words = line.split()
        assert "-L" not in words and "--location" not in line, (
            f"the /results request follows redirects: {line.strip()!r}"
        )
        assert "-f" not in words and "-fsS" not in words, (
            f"the /results request uses curl -f, which hides the status: {line.strip()!r}"
        )
        assert "%{http_code}" in line, f"the /results request does not read the status: {line!r}"
