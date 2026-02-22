from __future__ import annotations

import pytest

from vramsherpa.estimation import FitBadge, classify_fit, estimate_required_vram_gb, reserve_gb


def test_estimate_required_vram_formula() -> None:
    required = estimate_required_vram_gb(
        params_b=8,
        bits_effective=4.5,
        kv_gb_per_1k_ctx=0.18,
        context_tokens=2048,
    )
    # weights=4.95, kv=0.36864, overhead=1.4
    assert required == pytest.approx(6.71864)


def test_reserve_floor_and_percent() -> None:
    assert reserve_gb(4.0) == pytest.approx(0.75)
    assert reserve_gb(16.0) == pytest.approx(1.6)


def test_classification_boundaries() -> None:
    available = 10.0
    # reserve=1.0, fit threshold=9.0
    assert classify_fit(9.0, available) == FitBadge.FITS
    assert classify_fit(9.0001, available) == FitBadge.TIGHT
    assert classify_fit(10.0, available) == FitBadge.TIGHT
    assert classify_fit(10.0001, available) == FitBadge.WONT_FIT
