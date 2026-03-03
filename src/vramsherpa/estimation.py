from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FitBadge(StrEnum):
    FITS = "Fits"
    TIGHT = "Tight"
    WONT_FIT = "Won't fit"


@dataclass(frozen=True)
class Estimate:
    required_vram_gb: float
    available_vram_gb: float
    margin_gb: float
    reserve_gb: float
    classification: FitBadge


@dataclass(frozen=True)
class EstimateBreakdown:
    weights_gb: float
    kv_cache_gb: float
    runtime_overhead_gb: float
    required_vram_gb: float
    available_vram_gb: float
    margin_gb: float
    reserve_gb: float
    classification: FitBadge


def estimate_required_vram_gb(
    params_b: float,
    bits_effective: float,
    kv_gb_per_1k_ctx: float,
    context_tokens: int,
) -> float:
    weights_gb = params_b * bits_effective / 8 * 1.10
    kv_cache_gb = kv_gb_per_1k_ctx * (context_tokens / 1000)
    runtime_overhead_gb = 1.0 + 0.05 * params_b
    return weights_gb + kv_cache_gb + runtime_overhead_gb


def reserve_gb(available_vram_gb: float) -> float:
    return max(0.75, 0.10 * available_vram_gb)


def classify_fit(required_vram_gb: float, available_vram_gb: float) -> FitBadge:
    reserve = reserve_gb(available_vram_gb)
    if required_vram_gb <= available_vram_gb - reserve:
        return FitBadge.FITS
    if required_vram_gb <= available_vram_gb:
        return FitBadge.TIGHT
    return FitBadge.WONT_FIT


def estimate_variant(
    *,
    params_b: float,
    bits_effective: float,
    kv_gb_per_1k_ctx: float,
    context_tokens: int,
    available_vram_gb: float,
) -> Estimate:
    breakdown = estimate_breakdown(
        params_b=params_b,
        bits_effective=bits_effective,
        kv_gb_per_1k_ctx=kv_gb_per_1k_ctx,
        context_tokens=context_tokens,
        available_vram_gb=available_vram_gb,
    )
    return Estimate(
        required_vram_gb=breakdown.required_vram_gb,
        available_vram_gb=breakdown.available_vram_gb,
        margin_gb=breakdown.margin_gb,
        reserve_gb=breakdown.reserve_gb,
        classification=breakdown.classification,
    )


def estimate_breakdown(
    *,
    params_b: float,
    bits_effective: float,
    kv_gb_per_1k_ctx: float,
    context_tokens: int,
    available_vram_gb: float,
) -> EstimateBreakdown:
    weights_gb = params_b * bits_effective / 8 * 1.10
    kv_cache_gb = kv_gb_per_1k_ctx * (context_tokens / 1000)
    runtime_overhead_gb = 1.0 + 0.05 * params_b
    required = weights_gb + kv_cache_gb + runtime_overhead_gb
    reserve = reserve_gb(available_vram_gb)
    classification = classify_fit(required, available_vram_gb)
    return EstimateBreakdown(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        runtime_overhead_gb=runtime_overhead_gb,
        required_vram_gb=required,
        available_vram_gb=available_vram_gb,
        margin_gb=available_vram_gb - required,
        reserve_gb=reserve,
        classification=classification,
    )
