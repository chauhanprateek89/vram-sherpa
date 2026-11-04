"""The form keeps the view on a filter change and shares the links' hx-sync (R-089, ADR-0016).

pytest cannot run htmx, so this reads app.js and the rendered HTML, and runs the listener alone in
node against a stub page. QA drives the real thing in a browser.
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

APP_JS = Path(__file__).resolve().parents[1] / "src" / "vramsherpa" / "static" / "js" / "app.js"
SYNC = "#results-shell:replace"
GIST = "/results?vram_gb=8&context_tokens=4096"
LIST = f"{GIST}&verdict=tight&page_size=10"


@pytest.fixture()
def client(app, seeded_session: Session) -> TestClient:
    return TestClient(app)


class _Elements(HTMLParser):
    """Every start tag, with its attributes and the open tags above it."""

    VOID = {"area", "br", "col", "hr", "img", "input", "link", "meta", "source", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict[str, str]] = []
        self.found: list[tuple[dict[str, str], list[dict[str, str]]]] = []

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


def _elements(text: str) -> list[tuple[dict[str, str], list[dict[str, str]]]]:
    parser = _Elements()
    parser.feed(text)
    parser.close()
    return parser.found


def _by_id(text: str, element_id: str) -> dict[str, str]:
    [element] = [el for el, _ in _elements(text) if el.get("id") == element_id]
    return element


def _listener_source() -> str:
    source = APP_JS.read_text(encoding="utf-8")
    match = re.search(r"^function addFormView\(event\) \{\n.*?^\}\n", source, re.M | re.S)
    assert match
    return match.group(0)


@pytest.mark.parametrize("url", [GIST, LIST])
def test_form_has_the_links_sync_and_keeps_its_interface(client: TestClient, url: str) -> None:
    form = _by_id(client.get(url).text, "search-form")
    assert form["hx-sync"] == SYNC
    assert form["hx-get"] == "/results/partials/content"
    assert form["hx-target"] == "#results-shell"
    assert form["hx-push-url"] == "true"
    assert form["hx-trigger"] == (
        "submit, change from:.js-auto-filter, keyup changed delay:350ms from:.js-auto-filter-text"
    )


def test_sync_is_not_on_the_shell_or_above_the_breakdowns(client: TestClient) -> None:
    text = client.get(LIST).text
    elements = _elements(text)
    assert "hx-sync" not in _by_id(text, "results-shell")

    breakdowns = [
        (element, parents)
        for element, parents in elements
        if element.get("hx-trigger") == "click once"
    ]
    assert breakdowns
    for element, parents in breakdowns:
        assert element["hx-get"].startswith("/results/why/")
        assert "hx-sync" not in element
        assert not any("hx-sync" in parent for parent in parents)


def test_no_form_field_inherits_the_sync_into_a_request(client: TestClient) -> None:
    elements = _elements(client.get(LIST).text)
    in_form = [
        element
        for element, parents in elements
        if any(parent.get("id") == "search-form" for parent in parents)
    ]
    assert in_form
    assert not any("hx-get" in element for element in in_form)


def test_the_form_has_no_verdict_page_or_size_field(client: TestClient) -> None:
    # A field would be sent by every action, and a `page` would leave page 1 (ADR-0016).
    elements = _elements(client.get(f"{LIST}&page=2").text)
    names = {
        element.get("name")
        for element, parents in elements
        if any(parent.get("id") == "search-form" for parent in parents)
    }
    assert not names & {"verdict", "page", "page_size"}


@pytest.mark.parametrize(
    ("query", "verdict", "size"),
    [
        ("&verdict=tight&page_size=10", "tight", "10"),
        ("&verdict=tight", "tight", None),
        ("&verdict=green&page_size=10", None, None),
        ("", None, None),
    ],
)
def test_banner_holds_only_the_canonical_view(
    client: TestClient, query: str, verdict: str | None, size: str | None
) -> None:
    banner = _by_id(client.get(f"{GIST}{query}").text, "summary-banner")
    assert banner.get("data-verdict") == verdict
    assert banner.get("data-page-size") == size


def test_listener_has_the_submit_branch_and_the_two_dataset_reads() -> None:
    body = _listener_source()
    assert '.type === "submit"' in body
    assert "banner.dataset.verdict" in body
    assert "banner.dataset.pageSize" in body
    assert 'detail.parameters["verdict"] =' in body
    assert 'detail.parameters["page_size"] =' in body
    assert 'getElementById("summary-banner")' in body


def test_listener_never_reads_the_address_or_adds_a_page() -> None:
    body = _listener_source()
    assert "location" not in body
    assert "URLSearchParams" not in body
    assert 'parameters["page"]' not in body
    assert "location.search" not in APP_JS.read_text(encoding="utf-8")


def test_listener_is_on_config_request_and_runs_at_load() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'addEventListener("htmx:configRequest", addFormView)' in source
    ready = source[source.index('document.addEventListener("DOMContentLoaded"') :]
    assert "initFormView();" in ready.split("});", 1)[0]


def test_app_js_still_parses() -> None:
    from test_app import test_app_js_has_valid_syntax

    test_app_js_has_valid_syntax()


NODE_HARNESS = """
const fs = require("fs");
const vm = require("vm");
const [appJs, casesJson] = process.argv.slice(1);
const listeners = {};
let banner = null;
const sandbox = {
  window: {},
  // An invalid value in the address must not be forwarded (ADR-0016 point 4).
  location: { search: "?verdict=green&page=9&page_size=999" },
  document: {
    addEventListener() {},
    body: {
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
    },
    getElementById: (id) => (id === "summary-banner" ? banner : null),
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appJs, "utf8"), sandbox);
sandbox.initFormView();
const out = JSON.parse(casesJson).map((c) => {
  banner = c.dataset ? { dataset: c.dataset } : null;
  const detail = { triggeringEvent: { type: c.trigger }, parameters: { context_tokens: "2048" } };
  listeners["htmx:configRequest"].forEach((fn) => fn({ target: { id: c.target }, detail }));
  return detail.parameters;
});
console.log(JSON.stringify(out));
"""

BASE = {"context_tokens": "2048"}
KEPT = {**BASE, "verdict": "tight", "page_size": "10"}
TIGHT_10 = {"verdict": "tight", "pageSize": "10"}


@pytest.mark.parametrize(
    ("target", "trigger", "dataset", "expected"),
    [
        pytest.param("search-form", "change", TIGHT_10, KEPT, id="change-keeps-view"),
        pytest.param("search-form", "keyup", TIGHT_10, KEPT, id="keyup-keeps-view"),
        pytest.param("search-form", "submit", TIGHT_10, BASE, id="submit-adds-nothing"),
        pytest.param(
            "search-form",
            "change",
            {"verdict": "tight"},
            {**BASE, "verdict": "tight"},
            id="size-5-leaves-page-size-out",
        ),
        pytest.param("search-form", "change", {}, BASE, id="gist-stays-on-gist"),
        pytest.param("search-form", "change", None, BASE, id="no-banner"),
        pytest.param("results-shell", "click", TIGHT_10, BASE, id="other-request-untouched"),
    ],
)
def test_listener_in_node(
    target: str, trigger: str, dataset: dict[str, str] | None, expected: dict[str, str]
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    cases = json.dumps([{"target": target, "trigger": trigger, "dataset": dataset}])
    result = subprocess.run(
        [node, "-e", NODE_HARNESS, str(APP_JS), cases],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [expected]


# Each step is one request htmx is about to issue from the form: it fires htmx:confirm first, and
# only a request that passes goes on to hx-sync's abort of the one in flight (htmx 1.9.12).
SUBMIT_HARNESS = """
const fs = require("fs");
const vm = require("vm");
const [appJs, stepsJson] = process.argv.slice(1);
const listeners = {};
const sandbox = {
  window: {},
  URLSearchParams,
  FormData: function (form) {
    return form.fields;
  },
  document: {
    addEventListener() {},
    body: {
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
    },
    getElementById: () => null,
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appJs, "utf8"), sandbox);
sandbox.initFormView();
const sent = JSON.parse(stepsJson).map((step) => {
  let cancelled = false;
  const event = {
    target: { id: step.target || "search-form", fields: step.fields },
    detail: { triggeringEvent: { type: step.trigger } },
    preventDefault() {
      cancelled = true;
    },
  };
  (listeners["htmx:confirm"] || []).forEach((fn) => fn(event));
  return !cancelled;
});
console.log(JSON.stringify(sent));
"""


def _step(trigger: str, min_params: str = "6", target: str = "search-form") -> dict[str, object]:
    fields = [["context_tokens", "4096"], ["family", "llama"], ["min_params_b", min_params]]
    return {"trigger": trigger, "fields": fields, "target": target}


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param(
            [_step("submit"), _step("keyup")],
            [True, False],
            id="late-keyup-with-the-submitted-values-is-dropped",
        ),
        pytest.param(
            [_step("submit"), _step("keyup"), _step("keyup")],
            [True, False, False],
            id="every-late-keyup-is-dropped",
        ),
        pytest.param(
            [_step("keyup"), _step("keyup", "7")],
            [True, True],
            id="typing-without-a-submit-refreshes",
        ),
        pytest.param(
            [_step("submit"), _step("keyup", "7")],
            [True, True],
            id="a-new-value-after-a-submit-refreshes",
        ),
        pytest.param(
            [_step("submit"), _step("keyup", "7"), _step("keyup", "6")],
            [True, True, True],
            id="going-back-to-the-submitted-value-refreshes",
        ),
        pytest.param(
            [_step("submit"), _step("change", "7"), _step("keyup", "6")],
            [True, True, True],
            id="a-filter-change-ends-the-submit",
        ),
        pytest.param(
            [_step("submit"), _step("change")],
            [True, True],
            id="only-a-keyup-is-dropped",
        ),
        pytest.param(
            [_step("submit"), _step("submit")],
            [True, True],
            id="a-submit-is-never-dropped",
        ),
        pytest.param(
            [_step("submit"), _step("keyup", "7"), _step("submit", "7"), _step("keyup", "7")],
            [True, True, True, False],
            id="the-latest-submit-is-the-one-compared",
        ),
        pytest.param(
            [_step("submit"), _step("keyup", target="results-shell")],
            [True, True],
            id="other-request-untouched",
        ),
    ],
)
def test_keyup_after_submit_in_node(steps: list[dict[str, object]], expected: list[bool]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "-e", SUBMIT_HARNESS, str(APP_JS), json.dumps(steps)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_a_late_keyup_is_dropped_before_hx_sync_can_abort_the_submit() -> None:
    # htmx aborts the request in flight before htmx:configRequest, so cancelling there would lose a
    # submit still waiting for its response. htmx:confirm fires before the abort.
    source = APP_JS.read_text(encoding="utf-8")
    assert 'addEventListener("htmx:confirm", dropKeyupAfterSubmit)' in source
    assert "preventDefault" not in _listener_source()


def test_asset_version_is_bumped() -> None:
    assert re.fullmatch(r"\d{8}[a-z]", main.ASSET_VERSION)
    assert main.ASSET_VERSION > "20261103i"
    assert main.ASSET_VERSION != "20261028f"
