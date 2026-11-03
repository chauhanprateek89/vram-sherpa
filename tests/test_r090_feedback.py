"""Loading and failure feedback outside the swap (R-090, ADR-0018 points 4 to 6).

pytest cannot run htmx, so this reads the rendered HTML, styles.css and app.js, and runs the
listeners alone in node against a stub page, in the order htmx 1.9.12 fires them. QA drives the
real thing in a browser.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from vramsherpa import main

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "vramsherpa" / "static" / "js" / "app.js"
STYLES = ROOT / "src" / "vramsherpa" / "static" / "css" / "styles.css"

LOADING = "Loading models"
FAILED = "Could not load the models. Try again."
GIST = "/results?vram_gb=8&context_tokens=4096"
LIST = "/results?vram_gb=24&context_tokens=4096&verdict=fits&page=2"


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


class _Page(HTMLParser):
    """Every start tag with the open tags above it, and the text inside any `#results-feedback`."""

    VOID = {"area", "br", "col", "hr", "img", "input", "link", "meta", "source", "use", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict[str, str]] = []
        self.found: list[tuple[dict[str, str], list[dict[str, str]]]] = []
        self.text: list[str] = []
        self.feedback_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = {"#tag": tag, **{name: value or "" for name, value in attrs}}
        self.found.append((element, list(self.stack)))
        if tag not in self.VOID:
            self.stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["#tag"] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if any(element.get("id") == "results-feedback" for element in self.stack):
            self.feedback_text.append(data)


def _page(text: str) -> _Page:
    parser = _Page()
    parser.feed(text)
    parser.close()
    return parser


def _one(page: _Page, element_id: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    [found] = [pair for pair in page.found if pair[0].get("id") == element_id]
    return found


@pytest.mark.parametrize("url", [GIST, LIST])
def test_feedback_is_rendered_as_a_sibling_of_the_shell(client: TestClient, url: str) -> None:
    page = _page(client.get(url).text)
    _, feedback_parents = _one(page, "results-feedback")
    _, shell_parents = _one(page, "results-shell")
    assert feedback_parents == shell_parents
    assert feedback_parents[-1].get("class") == "results-main"
    assert not any(parent.get("id") == "results-shell" for parent in feedback_parents)


@pytest.mark.parametrize("url", [GIST, LIST])
def test_paragraphs_carry_the_strings_word_for_word(client: TestClient, url: str) -> None:
    html = client.get(url).text
    assert f'<p id="results-loading" role="status" data-text="{LOADING}">' in html
    assert f'<p id="results-error" role="alert" data-text="{FAILED}">' in html

    page = _page(html)
    for element_id in ("results-loading", "results-error"):
        element, parents = _one(page, element_id)
        assert element["#tag"] == "p"
        assert parents[-1].get("id") == "results-feedback"
        # A live region has to be in the page before its text changes (ADR-0018 point 4).
        assert "hidden" not in element


@pytest.mark.parametrize("url", [GIST, LIST])
def test_no_visible_text_until_a_script_fills_them(client: TestClient, url: str) -> None:
    html = client.get(url).text
    page = _page(html)
    assert "".join(page.feedback_text).strip() == ""
    for string in (LOADING, FAILED):
        assert not any(string in chunk for chunk in page.text)
        assert string not in html.replace(f'data-text="{string}"', "")


def test_the_partial_does_not_carry_the_feedback(client: TestClient) -> None:
    # Inside the swap it would be replaced on each refresh (ADR-0018).
    html = client.get(
        "/results/partials/content?vram_gb=8&context_tokens=4096",
        headers={"HX-Request": "true"},
    ).text
    assert "results-feedback" not in html
    assert "results-loading" not in html
    assert "results-error" not in html


def test_the_landing_page_has_no_feedback(client: TestClient) -> None:
    assert "results-feedback" not in client.get("/").text


def test_app_js_names_the_listeners_and_the_target_filter() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    for name in (
        "htmx:beforeRequest",
        "htmx:afterRequest",
        "htmx:responseError",
        "htmx:sendError",
    ):
        assert f'document.body.addEventListener("{name}"' in source, name
    assert 'event.detail.target.id === "results-shell"' in source
    assert 'addEventListener("htmx:sendAbort"' not in source
    # The text comes from data-text, so the template holds the only copy.
    assert LOADING not in source
    assert FAILED not in source
    assert "dataset.text" in source


def test_there_is_no_dismiss_or_retry_control(client: TestClient) -> None:
    page = _page(client.get(GIST).text)
    inside = [
        element
        for element, parents in page.found
        if any(parent.get("id") == "results-feedback" for parent in parents)
    ]
    assert inside
    assert not {"a", "button", "input"} & {element["#tag"] for element in inside}


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLES.read_text(encoding="utf-8"), flags=re.S)


def _rule(selector: str, css: str | None = None) -> str:
    source = _css() if css is None else css
    match = re.search(r"(?m)(?:^|\}\s*)" + re.escape(selector) + r"\s*\{([^{}]*)\}", source)
    assert match, selector
    return " ".join(match.group(1).split())


def test_the_feedback_is_fixed_to_the_viewport() -> None:
    assert _rule("#results-feedback") == (
        "position: fixed; z-index: 60; left: 0; right: 0; top: 12px; display: grid; "
        "justify-items: center; gap: 8px; padding: 0 16px; pointer-events: none;"
    )
    wide = re.search(
        r"@media \(min-width: 900px\) \{\s*#results-feedback \{\s*top: 80px;\s*\}\s*\}", _css()
    )
    assert wide


def test_no_box_is_drawn_until_filled() -> None:
    empty = _rule("#results-feedback p")
    # Off the grid, so an empty paragraph adds no gap above a filled one, and still in the
    # accessibility tree: display: none or visibility: hidden would take it out.
    assert "position: absolute" in empty
    assert "margin: 0" in empty
    for word in ("display", "visibility", "padding", "border", "background", "min-height"):
        assert word not in empty, word
    hidden_parts = re.search(
        r"#results-feedback p:not\(\.is-on\) \.fb-spin,\s*"
        r"#results-feedback p:not\(\.is-on\) svg\s*\{\s*display: none;\s*\}",
        _css(),
    )
    assert hidden_parts


def test_the_filled_look_is_the_mockups() -> None:
    assert _rule("#results-feedback .is-on") == (
        "position: static; display: flex; align-items: center; gap: 10px; "
        "max-width: min(100%, 440px); padding: 8px 16px 8px 12px; min-height: 40px; "
        "border-radius: 20px; background: var(--panel-solid); color: var(--ink); "
        "border: 1px solid var(--line); box-shadow: var(--sh-3); font-size: var(--fs-small); "
        "font-weight: 560;"
    )
    assert _rule("#results-feedback .fb-spin") == (
        "width: 16px; height: 16px; border-radius: 50%; border: 2.5px solid var(--accent); "
        "border-top-color: transparent; animation: spin .8s linear infinite; flex: none;"
    )
    assert _rule("#results-feedback .fb-alert") == (
        "background: var(--wont-bg); color: var(--wont-fg); "
        "border-color: color-mix(in srgb, var(--wont-fg) 45%, transparent); "
        "--pill-bg: var(--wont-bg); pointer-events: auto;"
    )
    css = _css()
    # Equal specificity, so the alert has to come after the filled look to win its colours.
    assert css.index("#results-feedback .is-on") < css.index("#results-feedback .fb-alert")


def test_pending_and_busy_rules_are_the_mockups() -> None:
    assert _rule(".is-pending::after") == (
        'content: ""; position: absolute; top: 10px; right: 10px; width: 14px; height: 14px; '
        "border-radius: 50%; border: 2px solid currentColor; border-top-color: transparent; "
        "animation: spin .8s linear infinite;"
    )
    assert _rule(".pager .is-pending") == "padding-right: 34px;"
    assert _rule(".pager .is-pending::after") == "top: 15px; right: 12px;"
    assert _rule('[aria-busy="true"] .card-grid') == "cursor: progress;"


def test_a_pending_size_link_anchors_its_spinner_inside_itself() -> None:
    # `.vt` and `.pager a` are positioned already; `.sizes-seg a` is not, and has no room to spare.
    assert "position: relative" in _rule(".sizes-seg .is-pending")
    assert _rule(".sizes-seg .is-pending::after") == "top: 15px; right: 4px;"


def test_feedback_rules_sit_before_utilities_and_motion() -> None:
    raw = STYLES.read_text(encoding="utf-8")
    utilities = raw.index("Utilities and motion")
    for needle in ("#results-feedback", ".is-pending", '[aria-busy="true"] .card-grid'):
        assert raw.rindex(needle) < utilities, needle


NODE_HARNESS = """
const fs = require("fs");
const vm = require("vm");
const [appJs, stepsJson] = process.argv.slice(1);

function makeElement(id, attrs) {
  const classes = new Set();
  const attributes = Object.assign({}, attrs);
  return {
    id,
    attributes,
    dataset: {},
    inDocument: true,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
    },
    classes,
    getAttribute: (name) => (name in attributes ? attributes[name] : null),
    setAttribute: (name, value) => { attributes[name] = value; },
    removeAttribute: (name) => { delete attributes[name]; },
    hasAttribute: (name) => name in attributes,
  };
}

function makeFeedback(id, text) {
  const element = makeElement(id, {});
  element.dataset.text = text;
  element.inner = { textContent: "" };
  element.querySelector = (selector) => (selector === "[data-fb-text]" ? element.inner : null);
  return element;
}

const shell = makeElement("results-shell", {});
const why = makeElement("why-1", {});
const loading = makeFeedback("results-loading", "Loading models");
const failed = makeFeedback("results-error", "Could not load the models. Try again.");
const byId = { "results-shell": shell, "results-loading": loading, "results-error": failed };
const requesters = {
  verdict: makeElement("link-verdict", { "data-view-link": "verdict" }),
  collapse: makeElement("link-collapse", { "data-view-link": "collapse" }),
  size: makeElement("link-size", { "data-view-link": "size" }),
  page: makeElement("link-page", { "data-view-link": "page" }),
  form: makeElement("search-form", {}),
  summary: makeElement("why-summary", {}),
};
const listeners = {};
const sandbox = {
  window: { addEventListener() {} },
  document: {
    addEventListener() {},
    body: {
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
    },
    getElementById: (id) => byId[id] || null,
    querySelectorAll: (selector) =>
      selector === ".is-pending"
        ? Object.values(requesters).filter(
            (el) => el.inDocument && el.classList.contains("is-pending"),
          )
        : [],
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appJs, "utf8"), sandbox);
sandbox.initResultsFeedback();

function state() {
  return {
    busy: shell.getAttribute("aria-busy"),
    loading: loading.inner.textContent,
    loadingOn: loading.classes.has("is-on"),
    error: failed.inner.textContent,
    errorOn: failed.classes.has("is-on"),
    errorAlert: failed.classes.has("fb-alert"),
    pending: Object.keys(requesters).filter((k) => requesters[k].classes.has("is-pending")),
    kind: vm.runInContext("pendingViewKind", sandbox),
  };
}

const out = JSON.parse(stepsJson).map((step) => {
  if (step.leave) {
    requesters[step.leave].inDocument = false;
    return state();
  }
  const target = step.target === "why" ? why : shell;
  (listeners[step.event] || []).forEach((fn) =>
    fn({ target: requesters[step.from], detail: { target } }),
  );
  return state();
});
console.log(JSON.stringify(out));
"""

IDLE = {
    "busy": None,
    "loading": "",
    "loadingOn": False,
    "error": "",
    "errorOn": False,
    "errorAlert": False,
    "pending": [],
    "kind": None,
}


def _run(*steps: dict[str, str]) -> list[dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "-e", NODE_HARNESS, str(APP_JS), json.dumps(list(steps))],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _ev(event: str, source: str, target: str = "shell") -> dict[str, str]:
    return {"event": f"htmx:{event}", "from": source, "target": target}


def _loading(kind: str | None, pending: list[str]) -> dict[str, object]:
    return {
        **IDLE,
        "busy": "true",
        "loading": LOADING,
        "loadingOn": True,
        "pending": pending,
        "kind": kind,
    }


def _failed(kind: str | None) -> dict[str, object]:
    return {**IDLE, "error": FAILED, "errorOn": True, "errorAlert": True, "kind": kind}


@pytest.mark.parametrize("kind", ["verdict", "collapse", "size", "page"])
def test_before_request_marks_the_shell_and_the_link(kind: str) -> None:
    [state] = _run(_ev("beforeRequest", kind))
    assert state == _loading(kind, [kind])


def test_a_form_refresh_shows_loading_with_no_pending_mark() -> None:
    # Every results refresh, including form changes and ones ending on the gist (Gate 2).
    before, after = _run(_ev("beforeRequest", "form"), _ev("afterRequest", "form"))
    assert before == _loading(None, [])
    assert after == IDLE


def test_after_request_clears_the_loading_state() -> None:
    _, after = _run(_ev("beforeRequest", "page"), _ev("afterRequest", "page"))
    assert after == {**IDLE, "kind": "page"}


def test_after_request_heard_on_an_ancestor_once_the_link_has_left() -> None:
    # After a swap the link is gone and htmx fires afterRequest on the closest ancestor as well.
    states = _run(
        _ev("beforeRequest", "verdict"),
        {"leave": "verdict"},
        _ev("afterRequest", "verdict"),
        _ev("afterRequest", "form"),
    )
    assert states[-1]["busy"] is None
    assert states[-1]["loading"] == ""
    assert states[-1]["loadingOn"] is False
    # The removed link keeps its class: only elements still in the document are touched.
    assert states[-1]["pending"] == ["verdict"]


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(("responseError", "afterRequest"), id="response-error"),
        pytest.param(("afterRequest", "sendError"), id="send-error"),
    ],
)
def test_a_failure_shows_the_alert_and_the_next_request_clears_it(
    order: tuple[str, str],
) -> None:
    states = _run(
        _ev("beforeRequest", "page"),
        _ev(order[0], "page"),
        _ev(order[1], "page"),
        _ev("beforeRequest", "verdict"),
    )
    assert states[2] == _failed("page")
    assert states[3] == _loading("verdict", ["verdict"])


def test_an_abort_is_not_a_failure_and_the_new_request_keeps_its_loading_state() -> None:
    # Under hx-sync replace the old request's events fire before the new beforeRequest.
    states = _run(
        _ev("beforeRequest", "page"),
        _ev("afterRequest", "page"),
        _ev("sendAbort", "page"),
        _ev("beforeRequest", "verdict"),
    )
    assert states[2] == {**IDLE, "kind": "page"}
    assert states[3] == _loading("verdict", ["verdict"])


def test_a_breakdown_request_changes_nothing() -> None:
    steps = [
        _ev(event, "summary", target="why")
        for event in ("beforeRequest", "afterRequest", "responseError", "sendError")
    ]
    for state in _run(*steps):
        assert state == IDLE


def test_a_breakdown_request_leaves_a_refresh_in_progress_alone() -> None:
    first, *rest = _run(
        _ev("beforeRequest", "page"),
        _ev("beforeRequest", "summary", target="why"),
        _ev("afterRequest", "summary", target="why"),
        _ev("sendError", "summary", target="why"),
    )
    assert first == _loading("page", ["page"])
    assert rest == [first, first, first]


def test_asset_version_is_bumped() -> None:
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
    assert main.ASSET_VERSION > "20261103g"
    assert main.ASSET_VERSION != "20261028f"
