"""QA acceptance checks for R-077 (accessibility through the redesign).

Contrast is computed here from the design tokens in `styles.css` (WCAG 2.x formula), against the
glass worst case: the pane's fill laid over `--backdrop-worst`. The browser measurements, which
also include the backdrop's saturation filter, focus rendering, reduced motion and the real pixels,
are in .sdlc/qa/R-077/report.md.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
CSS = re.sub(
    r"/\*.*?\*/",
    "",
    (ROOT / "src" / "vramsherpa" / "static" / "css" / "styles.css").read_text(encoding="utf-8"),
    flags=re.S,
)

Colour = tuple[float, float, float, float]


def _block(open_brace: int) -> str:
    depth = 0
    for index in range(open_brace, len(CSS)):
        depth += {"{": 1, "}": -1}.get(CSS[index], 0)
        if depth == 0:
            return CSS[open_brace + 1 : index]
    raise AssertionError("unbalanced braces")


def _tokens(selector_pattern: str) -> dict[str, str]:
    match = re.search(selector_pattern + r"\s*\{", CSS)
    assert match, selector_pattern
    return {
        name: " ".join(value.split())
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", _block(match.end() - 1))
    }


LIGHT = _tokens(r":root,\s*:root\[data-theme=\"light\"\]")
DARK = _tokens(r":root\[data-theme=\"dark\"\]")


def _colour(value: str) -> Colour:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
        return (*(float(int(value[i : i + 2], 16)) for i in (0, 2, 4)), 1.0)
    numbers = [float(n) for n in re.findall(r"[\d.]+", value)]
    return (numbers[0], numbers[1], numbers[2], numbers[3] if len(numbers) > 3 else 1.0)


def _over(top: Colour, below: Colour) -> Colour:
    alpha = top[3]
    return (*(top[i] * alpha + below[i] * (1 - alpha) for i in range(3)), 1.0)


def _luminance(c: Colour) -> float:
    def channel(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(c[0]) + 0.7152 * channel(c[1]) + 0.0722 * channel(c[2])


def _ratio(a: Colour, b: Colour) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _surfaces(theme: dict[str, str]) -> dict[str, Colour]:
    t = {k: _colour(v) for k, v in theme.items() if not k.startswith(("--sh", "--ic"))}
    worst = t["--backdrop-worst"]
    pane = _over(t["--panel"], worst)  # a glass pane over the worst backdrop colour
    lite_phone = _over((*t["--panel-solid"][:3], 0.86), worst)  # .lite below 900 px: no blur
    return {
        "pane over worst backdrop": pane,
        "lite pane on a phone over worst backdrop": lite_phone,
        "inset sheet on a pane": _over(t["--inset"], pane),
        "solid pane (fallbacks)": t["--panel-solid"],
        "field": t["--field"],
    }


THEMES = [("light", LIGHT), ("dark", DARK)]


@pytest.mark.parametrize(("theme_name", "theme"), THEMES)
def test_body_text_tokens_reach_4_5_on_every_surface(
    theme_name: str, theme: dict[str, str]
) -> None:
    t = {k: _colour(v) for k, v in theme.items() if k.startswith("--") and "sh" not in k[:4]}
    for surface_name, surface in _surfaces(theme).items():
        for token in ("--ink", "--muted", "--accent"):
            assert _ratio(t[token], surface) >= 4.5, (theme_name, token, surface_name)
    assert _ratio(t["--muted"], t["--field"]) >= 4.5  # the placeholder


@pytest.mark.parametrize(("theme_name", "theme"), THEMES)
def test_text_on_filled_controls_and_pills_reaches_4_5(
    theme_name: str, theme: dict[str, str]
) -> None:
    t = {k: _colour(v) for k, v in theme.items() if k.startswith("--") and "sh" not in k[:4]}
    pairs = [("--on-btn", "--btn"), ("--on-pressed", "--pressed"), ("--on-accent", "--accent")]
    pairs += [(f"--{k}-fg", f"--{k}-bg") for k in ("fits", "tight", "wont", "gate", "info")]
    for fg, bg in pairs:
        assert _ratio(t[fg], t[bg]) >= 4.5, (theme_name, fg, bg)


@pytest.mark.parametrize(("theme_name", "theme"), THEMES)
def test_borders_focus_gauge_and_reserve_reach_3_on_every_surface(
    theme_name: str, theme: dict[str, str]
) -> None:
    t = {k: _colour(v) for k, v in theme.items() if k.startswith("--") and "sh" not in k[:4]}
    surfaces = _surfaces(theme)
    track = _over(t["--track"], surfaces["pane over worst backdrop"])
    for surface_name, surface in surfaces.items():
        assert _ratio(t["--line"], surface) >= 3.0, (
            theme_name,
            "field/button border",
            surface_name,
        )
        assert _ratio(t["--focus"], surface) >= 3.0, (theme_name, "focus ring", surface_name)
        assert _ratio(t["--resv"], surface) >= 3.0, (theme_name, "reserve band", surface_name)
        for verdict in ("fits", "tight", "wont"):
            assert _ratio(t[f"--{verdict}-bar"], surface) >= 3.0, (
                theme_name,
                verdict,
                surface_name,
            )
            assert _ratio(t[f"--{verdict}-fg"], surface) >= 3.0, (theme_name, verdict, surface_name)
    for verdict in ("fits", "tight", "wont"):
        assert _ratio(t[f"--{verdict}-bar"], track) >= 3.0, (theme_name, verdict, "track")
        # the verdict icon against the pill it sits in
        assert _ratio(t[f"--{verdict}-fg"], t[f"--{verdict}-bg"]) >= 3.0


def test_the_unseeded_info_icon_cuts_its_i_out_in_the_sheet_colour() -> None:
    # The agreed departure of docs/decisions.md: the dark unseeded icon's "i" must show.
    assert re.search(r"\.empty \.empty-status\s*\{[^}]*--pill-bg:\s*var\(--panel-solid\)", CSS)
    for theme_name, theme in THEMES:
        t = {k: _colour(v) for k, v in theme.items() if k.startswith("--") and "sh" not in k[:4]}
        assert _ratio(t["--ink"], t["--panel-solid"]) >= 3.0, theme_name


def test_one_three_pixel_focus_ring_on_every_control() -> None:
    assert re.search(
        r":focus-visible\s*\{\s*outline:\s*3px solid var\(--focus\);\s*outline-offset:\s*2px;", CSS
    )
    removed = re.findall(r"([^{}]+):focus-visible\s*\{\s*outline:\s*(?:none|0)\b", CSS)
    assert [s.strip() for s in removed] == [".pick h3 a"]  # the whole tile carries the ring instead
    assert re.search(
        r"\.pick:has\(a:focus-visible\)\s*\{\s*outline:\s*3px solid var\(--focus\)", CSS
    )
    assert re.search(
        r"\.seg label:has\(input:focus-visible\)\s*\{\s*outline:\s*3px solid var\(--focus\)", CSS
    )


def test_reduced_motion_removes_every_transition_and_animation() -> None:
    bodies = [
        _block(m.end() - 1)
        for m in re.finditer(r"@media \(prefers-reduced-motion: reduce\)\s*\{", CSS)
    ]
    assert bodies
    assert any(
        re.search(
            r"\*,\s*\*::before,\s*\*::after\s*\{[^}]*transition:\s*none !important;"
            r"[^}]*animation:\s*none !important",
            body,
        )
        for body in bodies
    )


def test_reduced_transparency_and_missing_backdrop_filter_make_panes_solid() -> None:
    match = re.search(r"@media \(prefers-reduced-transparency: reduce\)\s*\{", CSS)
    assert match
    body = _block(match.end() - 1)
    assert ".pane, .lite" in body and "var(--panel-solid) !important" in body
    assert "backdrop-filter: none !important" in body
    assert re.search(
        r"@supports not \(\(backdrop-filter: blur\(1px\)\) or "
        r"\(-webkit-backdrop-filter: blur\(1px\)\)\)\s*\{\s*\.pane, \.lite\s*\{\s*"
        r"background: var\(--panel-solid\);",
        CSS,
    )


@pytest.fixture()
def client(app: FastAPI, seeded_session: Session) -> TestClient:
    return TestClient(app)


def test_quantisation_buttons_carry_exactly_one_aria_pressed_true_per_card(
    client: TestClient,
) -> None:
    page = client.get("/results?vram_gb=24&context_tokens=4096").text
    groups = re.findall(r'<div class="quants"[^>]*role="group"[^>]*>(.*?)</div>', page, re.S)
    assert len(groups) > 10
    for group in groups:
        states = re.findall(r'aria-pressed="(true|false)"', group)
        assert states.count("true") == 1 and len(states) >= 2


def test_every_gauge_is_an_image_with_the_needs_of_gigabytes_label(client: TestClient) -> None:
    page = client.get("/results?vram_gb=24&context_tokens=4096").text
    tracks = re.findall(r'<div\s+class="track"\s+role="img"\s+aria-label="([^"]+)"', page)
    assert len(tracks) > 40
    for label in tracks:
        assert re.fullmatch(
            r"Needs [\d.]+ of [\d.]+ gigabytes, with [\d.]+ gigabytes held in reserve", label
        )


def test_input_errors_are_one_alert_and_gpu_feedback_is_one_status(client: TestClient) -> None:
    errors = client.get("/results?vram_gb=abc").text
    assert errors.count('role="alert"') == 1
    assert re.search(r'role="alert">.*?Please fix the following input issues:', errors, re.S)
    assert 'aria-invalid="true"' in errors
    detail = client.get("/models/model_qwen25_7b_instruct?vram_gb=abc").text
    assert detail.count('role="alert"') == 1
    feedback = html.unescape(client.get("/results?gpu_search=RTX").text)
    assert re.search(
        r'role="status">\s*<svg[^>]*>.*?</svg>\s*<p>GPU search "RTX" matched', feedback, re.S
    )
    assert feedback.count('GPU search "RTX" matched') == 1
    assert 'role="alert"' not in feedback


def test_theme_toggle_and_navigation_keep_their_states(client: TestClient) -> None:
    page = client.get("/how-it-works").text
    assert re.search(r'<button id="theme-toggle"[^>]*aria-pressed="false"', page)
    assert page.count('aria-current="page"') == 1
    assert re.search(r'aria-current="page"[^>]*>\s*How it works', page)


def test_the_how_it_works_gauge_is_labelled(client: TestClient) -> None:
    page = client.get("/how-it-works").text
    assert re.search(
        r'class="track"\s+role="img"\s+aria-label="With [\d.]+ gigabytes of VRAM', page
    )
