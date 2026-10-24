from __future__ import annotations

import pytest

from vramsherpa.estimation import (
    FitBadge,
    classify_fit,
    estimate_decode_speed,
    estimate_required_vram_gb,
    reserve_gb,
)


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


# ── decode speed ──────────────────────────────────────────────────────────────


def test_decode_speed_scales_with_bandwidth() -> None:
    """Twice the bandwidth, twice the tokens per second."""
    slow = estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=500.0)
    fast = estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=1000.0)
    assert slow is not None and fast is not None
    assert fast.mid_tps == pytest.approx(slow.mid_tps * 2)


def test_decode_speed_falls_with_precision() -> None:
    """FP16 reads roughly 3.5x the bytes of Q4, so it decodes proportionally slower."""
    q4 = estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=1008.0)
    fp16 = estimate_decode_speed(params_b=8.0, bits_effective=16.0, bandwidth_gb_s=1008.0)
    assert q4 is not None and fp16 is not None
    assert q4.mid_tps == pytest.approx(fp16.mid_tps * (16.0 / 4.5))


def test_decode_speed_uses_active_params_for_mixture_of_experts() -> None:
    """An MoE keeps every expert resident but reads only the routed ones per token."""
    moe = estimate_decode_speed(
        params_b=41.87, bits_effective=4.5, bandwidth_gb_s=1008.0, active_params_b=6.6
    )
    dense = estimate_decode_speed(params_b=41.87, bits_effective=4.5, bandwidth_gb_s=1008.0)
    assert moe is not None and dense is not None
    assert moe.is_mixture_of_experts is True
    assert dense.is_mixture_of_experts is False
    # it decodes at the speed its active size implies, not its total size
    assert moe.mid_tps == pytest.approx(dense.mid_tps * (41.87 / 6.6))


def test_decode_speed_is_none_without_bandwidth() -> None:
    """A hand-typed VRAM figure says nothing about speed, so nothing is claimed."""
    assert estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=None) is None
    assert estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=0.0) is None


def test_decode_speed_range_brackets_the_midpoint() -> None:
    speed = estimate_decode_speed(params_b=14.0, bits_effective=4.5, bandwidth_gb_s=1008.0)
    assert speed is not None
    assert speed.low_tps < speed.mid_tps < speed.high_tps


def test_decode_speed_is_plausible_for_a_known_pairing() -> None:
    """An RTX 4090 running an 8B model at Q4 lands near reported llama.cpp figures."""
    speed = estimate_decode_speed(params_b=8.0, bits_effective=4.5, bandwidth_gb_s=1008.0)
    assert speed is not None
    assert 100 <= speed.low_tps <= 200
    assert 120 <= speed.high_tps <= 220
