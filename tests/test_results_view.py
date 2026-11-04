import pytest
from starlette.datastructures import QueryParams

from vramsherpa.results_view import (
    ResultsView,
    page_window,
    paginate,
    parse_apply_view,
    parse_results_view,
    view_query,
    without_apply_view,
)


def parse(qs):
    return parse_results_view(QueryParams(qs))


@pytest.mark.parametrize("v", ["fits", "tight", "wont-fit"])
def test_verdict_accepted(v):
    assert parse(f"verdict={v}").verdict == v


@pytest.mark.parametrize("v", ["FITS", "green", "", "%3Cscript%3E", "fits+", "wont_fit"])
def test_verdict_rejected(v):
    assert parse(f"verdict={v}&page=3&page_size=10") == ResultsView(None, 1, 5)


def test_verdict_last_wins():
    assert parse("verdict=fits&verdict=tight").verdict == "tight"
    assert parse("verdict=fits&verdict=green").verdict is None
    assert parse("").verdict is None


@pytest.mark.parametrize(
    "s,want", [("5", 5), ("10", 10), ("7", 5), ("abc", 5), ("1000", 5), ("", 5)]
)
def test_page_size(s, want):
    assert parse(f"verdict=fits&page_size={s}").page_size == want


def test_page_size_last_wins():
    assert parse("verdict=fits&page_size=10&page_size=5").page_size == 5
    assert parse("verdict=fits&page_size=5&page_size=10").page_size == 10


@pytest.mark.parametrize(
    "p,want",
    [
        ("2", 2),
        ("0012", 12),
        ("abc", 1),
        ("0", 1),
        ("-1", 1),
        ("1.5", 1),
        ("+2", 1),
        ("1_000", 1),
        ("٣", 1),
        ("", 1),
        (" 2", 1),
    ],
)
def test_page(p, want):
    assert parse(f"verdict=fits&page={p}").page == want


def test_page_last_wins():
    assert parse("verdict=fits&page=2&page=3").page == 3


def test_huge_page_does_not_raise():
    v = parse("verdict=fits&page=" + "9" * 5000)
    assert v.page >= 2**63 - 1
    p = paginate(71, v)
    assert p.page == 15 and (p.first, p.last) == (71, 71)
    assert parse("verdict=fits&page=" + "0" * 5000 + "3").page == 3


@pytest.mark.parametrize(
    "total,size,page,want",
    [
        (71, 5, 2, (2, 6, 10, 71, 15)),
        (71, 5, 99, (15, 71, 71, 71, 15)),
        (71, 10, 8, (8, 71, 71, 71, 8)),
        (9, 5, 1, (1, 1, 5, 9, 2)),
        (9, 10, 1, (1, 1, 9, 9, 1)),
    ],
)
def test_paginate(total, size, page, want):
    p = paginate(total, ResultsView("fits", page, size))
    assert (p.page, p.first, p.last, p.total, p.last_page) == want
    assert p.page_size == size


def test_paginate_empty():
    p = paginate(0, ResultsView("fits", 3, 5))
    assert (p.page, p.first, p.last, p.last_page) == (1, 0, 0, 1)


@pytest.mark.parametrize(
    "page,last,want",
    [
        (1, 15, [1, 2, None, 15]),
        (6, 15, [1, None, 5, 6, 7, None, 15]),
        (15, 15, [1, None, 14, 15]),
        (10, 19, [1, None, 9, 10, 11, None, 19]),
        (3, 15, [1, 2, 3, 4, None, 15]),
        (1, 1, [1]),
        (2, 3, [1, 2, 3]),
    ],
)
def test_page_window(page, last, want):
    assert page_window(page, last) == want


def test_view_query_replaces_own_keys_and_keeps_rest():
    raw = "family=a&vram_gb=abc&family=b&verdict=fits&page=2&page_size=10&apply_view=1&&q=%20x+y"
    assert view_query(raw, "tight", 3, 10) == (
        "family=a&vram_gb=abc&family=b&q=%20x+y&verdict=tight&page=3&page_size=10"
    )


def test_view_query_defaults_omitted():
    assert view_query("a=1", "fits") == "a=1&verdict=fits"
    assert view_query("a=1", "fits", 1, 10) == "a=1&verdict=fits&page_size=10"
    assert view_query("a=1", "fits", 2, 5) == "a=1&verdict=fits&page=2"


def test_view_query_no_verdict_appends_nothing():
    assert view_query("a=1&verdict=fits&page=2", None, 4, 10) == "a=1"
    assert view_query("") == ""


def test_view_query_decodes_keys_but_keeps_bytes():
    assert view_query("%76erdict=fits&pa+ge=1&page%5Fsize=10&x%20y=%41", "fits") == (
        "pa+ge=1&x%20y=%41&verdict=fits"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fits:5", ("fits", 5)),
        ("fits:10", ("fits", 10)),
        ("tight:5", ("tight", 5)),
        ("wont-fit:10", ("wont-fit", 10)),
    ],
)
def test_parse_apply_view_accepts_every_verdict_and_size(value, expected):
    assert parse_apply_view(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        ":",
        "fits",
        "fits:",
        ":5",
        "green:5",
        "fits:7",
        "fits:05",
        "fits:5:5",
        "FITS:5",
        " fits:5",
        "fits:5 ",
        "fits:5\n",
        "fits:\u0665",
        "wont_fit:5",
    ],
)
def test_parse_apply_view_rejects_everything_else(value):
    assert parse_apply_view(value) is None


def test_parse_apply_view_takes_the_last_of_repeated_values():
    assert parse_apply_view(["green:5", "tight:10"]) == ("tight", 10)
    assert parse_apply_view(["tight:10", "green:5"]) is None
    assert parse_apply_view([]) is None
    assert parse_apply_view(
        QueryParams("apply_view=a&apply_view=fits:5").getlist("apply_view")
    ) == (
        "fits",
        5,
    )


def test_without_apply_view_drops_only_that_key_and_keeps_bytes():
    raw = "a=%41&&%61pply_view=fits:5&b=x+y&apply_view&verdict=fits&page=2"
    assert without_apply_view(raw) == "a=%41&&b=x+y&verdict=fits&page=2"
    assert without_apply_view("") == ""
