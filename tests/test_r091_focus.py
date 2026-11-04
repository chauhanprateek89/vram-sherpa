"""Focus and scroll after a view link, typed text kept, modified clicks left to the browser
(R-091, R-090, ADR-0018 points 7 to 9). pytest cannot run htmx, so this reads app.js."""

from __future__ import annotations

import re
from pathlib import Path

from vramsherpa import main

APP_JS = Path(__file__).resolve().parents[1] / "src/vramsherpa/static/js/app.js"


def _js() -> str:
    return APP_JS.read_text()


def _fn(name: str) -> str:
    return _js().split(f"function {name}")[1].split("\nfunction ")[0]


def test_one_shot_flag_set_at_before_request_and_consumed() -> None:
    js = _js()
    assert "var focusPending = false" in js
    before = js.split('"htmx:beforeRequest"')[1].split('"htmx:afterRequest"')[0]
    assert "focusPending = !!pendingViewKind" in before
    body = _fn("focusAfterViewLink")
    assert "!focusPending || !isShellRequest(event)" in body
    assert "focusPending = false" in body


def test_heading_focus_prevents_scroll_then_scrolls_with_explicit_behavior() -> None:
    body = _fn("focusAfterViewLink")
    assert 'getElementById("list-heading")' in body
    assert "if (!heading)" in body
    assert "heading.focus({ preventScroll: true })" in body
    assert body.index("preventScroll") < body.index("scrollIntoView")
    assert 'block: "start"' in body
    assert 'prefersReducedMotion() ? "auto" : "smooth"' in body


def test_collapse_focuses_the_control_with_the_same_value() -> None:
    body = _fn("focusAfterViewLink")
    assert 'pendingViewKind === "collapse"' in body
    assert "data-verdict-control" in body
    assert "=== pendingControl" in body
    assert ".focus()" in body


def test_failures_and_aborts_clear_the_flag() -> None:
    js = _js()
    after = js.split('"htmx:afterRequest"')[1].split("function showFailure")[0]
    assert "!event.detail.successful" in after and "focusPending = false" in after
    assert re.search(r"function showFailure[^}]*focusPending = false", js, re.S)


def test_view_link_settle_skips_form_syncs() -> None:
    js = _js()
    assert "function isViewLinkSettle" in js
    for name in ("syncVramInvalid", "syncBoxVram"):
        body = _fn(name)
        assert body.index("isViewLinkSettle(event)") < body.index("getElementById")
    assert 'addEventListener("htmx:afterSettle", syncVramInvalid)' in js


def test_modified_click_is_left_to_the_browser_in_a_capture_listener() -> None:
    body = _fn("initViewLinkModifiers")
    assert 'document.addEventListener(\n    "click"' in body
    assert "[data-view-link]" in body
    for key in ("ctrlKey", "metaKey", "shiftKey"):
        assert f"event.{key}" in body
    assert "event.stopPropagation()" in body
    assert "preventDefault" not in body
    assert "    true,\n  );" in body
    assert "initViewLinkModifiers();" in _js().split('addEventListener("DOMContentLoaded"')[1]


def test_asset_version_bumped() -> None:
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
    assert main.ASSET_VERSION > "20261103h"
    assert main.ASSET_VERSION != "20261028f"


def test_app_js_still_parses() -> None:
    from test_app import test_app_js_has_valid_syntax

    test_app_js_has_valid_syntax()
