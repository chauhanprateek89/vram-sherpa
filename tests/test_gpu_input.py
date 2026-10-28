from types import SimpleNamespace

import pytest

from vramsherpa.gpu_input import (
    BoxDecision,
    decide_box,
    exact_gpu_match,
    gpu_option_label,
    normalize_for_match,
    parse_box_number,
)

RTX = SimpleNamespace(vendor="NVIDIA", name="RTX 3060", vram_gb=12.0)
ODD = SimpleNamespace(vendor="Acme", name="12", vram_gb=12.0)
GPUS = [RTX, SimpleNamespace(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0)]


def test_label_and_normalize():
    assert gpu_option_label(RTX) == "NVIDIA RTX 3060 (12.0 GB)"
    assert normalize_for_match("  NVIDIA   RTX\t3060 ") == "nvidia rtx 3060"


@pytest.mark.parametrize(
    "text", ["NVIDIA RTX 3060 (12.0 GB)", "nvidia  rtx 3060", "rtx 3060", " RTX   3060 "]
)
def test_exact_match(text):
    assert exact_gpu_match(text, GPUS) is RTX


def test_exact_match_none_or_ambiguous():
    assert exact_gpu_match("rtx", GPUS) is None
    assert exact_gpu_match("rtx 3060", [RTX, RTX]) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12", "12"),
        ("12 GB", "12"),
        ("12gb", "12"),
        ("12.5 GB", "12.5"),
        ("24GB", "24"),
        ("0", "0"),
        ("0.0", "0.0"),
        ("  12  ", "12"),
        ("12  GB", "12"),
    ],
)
def test_box_numbers(text, expected):
    assert parse_box_number(text) == expected


@pytest.mark.parametrize(
    "text",
    ["12 G", "12 GiB", "12,5", ".5", "-4", "12 GB VRAM", "RTX 3060 12GB", "１２", "", "GB"],
)
def test_not_box_numbers(text):
    assert parse_box_number(text) is None


@pytest.mark.parametrize("text", ["", "   ", "\t"])
def test_decide_blank(text):
    assert decide_box(text, GPUS) == BoxDecision("blank")


def test_decide_gpu_vram_search():
    assert decide_box("rtx 4090", GPUS).kind == "gpu"
    assert decide_box("RTX 3060", GPUS).gpu is RTX
    d = decide_box("12 GB", GPUS)
    assert (d.kind, d.vram_text, d.gpu) == ("vram", "12", None)
    assert decide_box("rtx", GPUS) == BoxDecision("search")


def test_exact_label_wins_over_number():
    d = decide_box("12", [ODD])
    assert d.kind == "gpu" and d.gpu is ODD


def test_decision_is_frozen():
    with pytest.raises(AttributeError):
        BoxDecision("blank").kind = "gpu"  # type: ignore[misc]
