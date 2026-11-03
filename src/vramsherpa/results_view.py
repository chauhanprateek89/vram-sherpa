"""Verdict filter and paging for the results view (R-087, R-088, ADR-0014).

Pure functions: no session, no request, nothing imported from main.py.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote_plus

VERDICTS = ("fits", "tight", "wont-fit")
PAGE_SIZES = (5, 10)
DEFAULT_PAGE_SIZE = 5
MAX_PAGE = 2**63 - 1
_OWN_KEYS = frozenset({"verdict", "page", "page_size", "apply_view"})
_DIGITS = re.compile("[0-9]+")


@dataclass(frozen=True)
class ResultsView:
    verdict: str | None = None
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE


@dataclass(frozen=True)
class Pagination:
    page: int
    first: int
    last: int
    total: int
    last_page: int
    page_size: int


def _last(params, key: str) -> str | None:
    values = params.getlist(key)
    return values[-1] if values else None


def _parse_page(raw: str | None) -> int:
    if raw is None or not _DIGITS.fullmatch(raw):
        return 1
    digits = raw.lstrip("0")
    if len(digits) > 19:  # also keeps int() clear of Python's 4300-digit limit
        return MAX_PAGE
    return min(max(int(digits or "0"), 1), MAX_PAGE) if digits else 1


def parse_results_view(params) -> ResultsView:
    verdict = _last(params, "verdict")
    if verdict not in VERDICTS:
        return ResultsView()
    size_raw = _last(params, "page_size")
    page_size = int(size_raw) if size_raw in ("5", "10") else DEFAULT_PAGE_SIZE
    return ResultsView(verdict, _parse_page(_last(params, "page")), page_size)


def paginate(total: int, view: ResultsView) -> Pagination:
    size = view.page_size
    last_page = max(1, -(-total // size))
    page = min(view.page, last_page)
    if total <= 0:
        return Pagination(page, 0, 0, 0, last_page, size)
    return Pagination(page, (page - 1) * size + 1, min(page * size, total), total, last_page, size)


def page_window(page: int, last_page: int) -> list[int | None]:
    wanted = sorted({1, last_page} | {n for n in (page - 1, page, page + 1) if 1 <= n <= last_page})
    out: list[int | None] = []
    for n in wanted:
        if out and n - out[-1] > 1:  # type: ignore[operator]
            out.append(None)
        out.append(n)
    return out


def view_query(
    raw_query: str, verdict: str | None = None, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
) -> str:
    kept = [
        part
        for part in raw_query.split("&")
        if part and unquote_plus(part.split("=", 1)[0]) not in _OWN_KEYS
    ]
    if verdict is not None:
        kept.append(f"verdict={verdict}")
        if page != 1:
            kept.append(f"page={page}")
        if page_size != DEFAULT_PAGE_SIZE:
            kept.append(f"page_size={page_size}")
    return "&".join(kept)
