"""T-036: one card per model, counts tally the cards, verdict lists sort on the server."""

from __future__ import annotations

from sqlalchemy.orm import Session

from vramsherpa.main import (
    ResultRow,
    _build_results,
    _default_variant,
    _group_results_by_model,
    _summary_counts,
    _verdict_cards,
)

FITS, TIGHT, WONT = "Fits", "Tight", "Won't fit"


def _row(bucket: str, verdict: str, recommended: bool = False, **kw) -> ResultRow:
    base = dict(
        variant_id=f"m-{bucket}",
        model_id="m",
        model_name="M",
        family="f",
        params_b=7.0,
        quant_bucket=bucket,
        quant_label=bucket,
        bits_effective=4.0,
        required_vram_gb=10.0,
        available_vram_gb=8.0,
        reserve_gb=1.0,
        margin_gb=-2.0,
        classification=verdict,
        recommended=recommended,
    )
    base.update(kw)
    return ResultRow(**base)


def test_default_prefers_usable_recommended() -> None:
    rows = [_row("Q4", FITS), _row("Q5", TIGHT, True), _row("Q8", WONT)]
    assert _default_variant(rows).quant_bucket == "Q5"


def test_default_first_usable_when_recommended_not_usable() -> None:
    rows = [_row("Q4", WONT), _row("Q5", TIGHT), _row("Q8", WONT, True)]
    assert _default_variant(rows).quant_bucket == "Q5"


def test_default_none_usable_takes_recommended() -> None:
    rows = [_row("Q4", WONT), _row("Q5", WONT, True), _row("Q8", WONT)]
    assert _default_variant(rows).quant_bucket == "Q5"


def test_default_none_usable_no_recommended_takes_first() -> None:
    rows = [_row("Q4", WONT), _row("Q5", WONT)]
    assert _default_variant(rows).quant_bucket == "Q4"


def test_no_usable_model_still_gets_a_card_under_wont_fit() -> None:
    cards = _group_results_by_model([_row("Q8", WONT), _row("Q4", WONT)])
    assert len(cards) == 1
    assert cards[0].default_variant.quant_bucket == "Q4"
    assert _summary_counts(cards) == {FITS: 0, TIGHT: 0, WONT: 1}


def test_wont_fit_sorts_by_shortfall_then_size_then_code_point_name() -> None:
    def card(mid: str, name: str, params: float, need: float):
        return _group_results_by_model(
            [
                _row(
                    "Q4",
                    WONT,
                    model_id=mid,
                    model_name=name,
                    params_b=params,
                    variant_id=mid,
                    required_vram_gb=need,
                )
            ]
        )[0]

    cards = [
        card("a", "gemma-3-1b-it", 1.0, 10.0),
        card("b", "Llama-3.2-1B-Instruct", 1.0, 10.0),
        card("c", "Big", 5.0, 10.0),
        card("d", "Near", 1.0, 9.0),
    ]
    names = [c.model_name for c in _verdict_cards(cards, WONT)]
    assert names == ["Near", "Big", "Llama-3.2-1B-Instruct", "gemma-3-1b-it"]


def _cards(session: Session, vram: float, ctx: int, **kw):
    rows = _build_results(
        session,
        available_vram_gb=vram,
        context_tokens=ctx,
        families=kw.get("families", []),
        quant_buckets=kw.get("quant_buckets", []),
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
    )
    return _group_results_by_model(rows)


def _names(cards, label, lo, hi):
    return [c.model_name for c in _verdict_cards(cards, label)][lo - 1 : hi]


def test_reference_counts_and_orders(seeded_session: Session) -> None:
    cards = _cards(seeded_session, 24, 4096)
    assert _summary_counts(cards) == {FITS: 71, TIGHT: 0, WONT: 24}
    assert _names(cards, FITS, 1, 5) == [
        "DeepSeek-Coder-V2-Lite-Instruct",
        "DeepSeek-V2-Lite",
        "DeepSeek-V2-Lite-Chat",
        "Qwen2.5 14B Instruct",
        "DeepSeek-R1-Distill-Qwen-14B",
    ]
    assert _names(cards, WONT, 1, 5) == [
        "Qwen3-30B-A3B",
        "Qwen3-30B-A3B-Instruct-2507",
        "Qwen3-Coder-30B-A3B-Instruct",
        "DeepSeek-R1-Distill-Qwen-32B",
        "Qwen2.5-32B-Instruct",
    ]
    assert _names(cards, WONT, 16, 20) == [
        "DeepSeek-V4-Flash-0731",
        "DeepSeek-R1",
        "DeepSeek-V3",
        "DeepSeek-R1-0528",
        "DeepSeek-V3-0324",
    ]
    assert _names(cards, WONT, 21, 24) == [
        "DeepSeek-V3.1",
        "DeepSeek-V3.2",
        "DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro-0813",
    ]


def test_one_gb_counts_and_order(seeded_session: Session) -> None:
    cards = _cards(seeded_session, 1, 4096)
    assert _summary_counts(cards) == {FITS: 0, TIGHT: 0, WONT: 95}
    assert _names(cards, WONT, 6, 10) == [
        "Qwen3-0.6B-Base",
        "Llama-3.2-1B",
        "Llama-3.2-1B-Instruct",
        "gemma-3-1b-it",
        "phi-1_5",
    ]


def test_family_filter_counts(seeded_session: Session) -> None:
    cards = _cards(seeded_session, 8, 2048, families=["llama"])
    assert _summary_counts(cards) == {FITS: 6, TIGHT: 4, WONT: 4}


def test_q8_first_wont_fit_is_over_by_20_9(seeded_session: Session) -> None:
    cards = _cards(seeded_session, 24, 4096, quant_buckets=["Q8"])
    first = _verdict_cards(cards, WONT)[0]
    assert first.model_name == "Qwen3-30B-A3B"
    assert f"{-first.default_variant.margin_gb:.1f}" == "20.9"
