from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, distinct, select
from sqlalchemy.orm import Session

from vramsherpa.config import Settings, get_settings
from vramsherpa.database import configure_engine, create_db_and_tables, get_session
from vramsherpa.estimation import FitBadge, estimate_breakdown, estimate_variant
from vramsherpa.models import GPU, CatalogMeta, Model, Variant

CONTEXT_OPTIONS = (2048, 4096, 8192)
ASSET_VERSION = "20260303d"
PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


@dataclass
class ResultRow:
    variant_id: str
    model_id: str
    model_name: str
    family: str
    params_b: float
    quant_bucket: str
    quant_label: str
    bits_effective: float
    required_vram_gb: float
    available_vram_gb: float
    margin_gb: float
    classification: str
    recommended: bool

    @property
    def classification_slug(self) -> str:
        return self.classification.lower().replace("'", "").replace(" ", "-")


@dataclass
class TopPick:
    title: str
    description: str
    row: ResultRow


@dataclass
class ActiveFilterChip:
    label: str
    remove_url: str


@dataclass
class ExampleChip:
    label: str
    gpu_id: str | None
    vram_gb: float | None


@dataclass
class ParsedFloatResult:
    value: float | None
    error: str | None


def on_startup() -> None:
    create_db_and_tables()


async def host_guard(request: Request, call_next):
    settings: Settings = request.app.state.settings
    allowed_hosts = settings.allowed_hosts
    if "*" not in allowed_hosts:
        host = request.headers.get("host", "").split(":", 1)[0]
        if host and host not in allowed_hosts:
            return JSONResponse({"detail": "Host not allowed"}, status_code=400)
    return await call_next(request)


def _catalog_version(session: Session) -> str:
    meta = session.get(CatalogMeta, "catalog_version")
    return meta.value if meta else "unseeded"


def _all_families(session: Session) -> list[str]:
    values = session.scalars(select(distinct(Model.family)).order_by(Model.family)).all()
    return [str(value) for value in values]


def _all_quant_buckets(session: Session) -> list[str]:
    values = session.scalars(
        select(distinct(Variant.quant_bucket)).order_by(Variant.quant_bucket)
    ).all()
    return [str(value) for value in values]


def _load_gpus(session: Session) -> list[GPU]:
    return list(session.scalars(select(GPU).order_by(GPU.vram_gb, GPU.name)).all())


def _resolve_available_vram(
    session: Session,
    gpu_id: str | None,
    vram_gb: float | None,
    gpu_search: str | None = None,
    prefer_gpu_id_when_both: bool = False,
) -> tuple[float, str | None, list[GPU], str | None]:
    gpus = _load_gpus(session)

    normalized_gpu_id = gpu_id.strip() if isinstance(gpu_id, str) and gpu_id.strip() else None
    normalized_gpu_search = (
        gpu_search.strip() if isinstance(gpu_search, str) and gpu_search.strip() else None
    )
    selected_gpu = session.get(GPU, normalized_gpu_id) if normalized_gpu_id else None
    if selected_gpu is not None and prefer_gpu_id_when_both:
        return float(selected_gpu.vram_gb), selected_gpu.id, gpus, None

    if vram_gb is not None:
        return float(vram_gb), None, gpus, None

    if selected_gpu is not None:
        return float(selected_gpu.vram_gb), selected_gpu.id, gpus, None

    if normalized_gpu_search:
        needle = _normalize_for_match(normalized_gpu_search)
        exact_matches = [
            gpu
            for gpu in gpus
            if needle
            in {
                _normalize_for_match(_gpu_option_label(gpu)),
                _normalize_for_match(f"{gpu.vendor} {gpu.name}"),
                _normalize_for_match(gpu.name),
            }
        ]
        if len(exact_matches) == 1:
            matched_gpu = exact_matches[0]
            return float(matched_gpu.vram_gb), matched_gpu.id, gpus, None

        partial_matches = [
            gpu
            for gpu in gpus
            if needle in _normalize_for_match(_gpu_option_label(gpu))
            or needle in _normalize_for_match(f"{gpu.vendor} {gpu.name}")
            or needle in _normalize_for_match(gpu.name)
        ]
        if len(partial_matches) == 1:
            matched_gpu = partial_matches[0]
            return float(matched_gpu.vram_gb), matched_gpu.id, gpus, None

        if len(partial_matches) > 1:
            suggestions = ", ".join(_gpu_option_label(gpu) for gpu in partial_matches[:3])
            return (
                8.0,
                None,
                gpus,
                f'GPU search "{normalized_gpu_search}" matched multiple entries ({suggestions}). '
                "Select an exact GPU from the list or enter VRAM manually.",
            )
        hinted_vram = _extract_vram_hint(normalized_gpu_search)
        if hinted_vram is not None:
            return (
                hinted_vram,
                None,
                gpus,
                (
                    f'GPU "{normalized_gpu_search}" was not found. '
                    f"Using {hinted_vram:g} GB inferred from your input."
                ),
            )
        return (
            8.0,
            None,
            gpus,
            (
                f'GPU "{normalized_gpu_search}" was not found. '
                "Select a listed GPU or enter VRAM manually."
            ),
        )
    return 8.0, None, gpus, None


def _variant_query(
    *,
    families: list[str],
    quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
) -> Select[tuple[Variant, Model]]:
    stmt = select(Variant, Model).join(Model, Variant.model_id == Model.id)
    if families:
        stmt = stmt.where(Model.family.in_(families))
    if quant_buckets:
        stmt = stmt.where(Variant.quant_bucket.in_(quant_buckets))
    if min_params_b is not None:
        stmt = stmt.where(Model.params_b >= min_params_b)
    if max_params_b is not None:
        stmt = stmt.where(Model.params_b <= max_params_b)
    if recommended_only:
        stmt = stmt.where(Variant.recommended.is_(True))
    return stmt


def _build_results(
    session: Session,
    *,
    available_vram_gb: float,
    context_tokens: int,
    families: list[str],
    quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    stmt = _variant_query(
        families=families,
        quant_buckets=quant_buckets,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )
    for variant, model in session.execute(stmt).all():
        estimate = estimate_variant(
            params_b=float(model.params_b),
            bits_effective=float(variant.bits_effective),
            kv_gb_per_1k_ctx=float(model.kv_gb_per_1k_ctx),
            context_tokens=context_tokens,
            available_vram_gb=available_vram_gb,
        )
        rows.append(
            ResultRow(
                variant_id=variant.id,
                model_id=model.id,
                model_name=model.name,
                family=model.family,
                params_b=float(model.params_b),
                quant_bucket=variant.quant_bucket,
                quant_label=variant.quant_label,
                bits_effective=float(variant.bits_effective),
                required_vram_gb=estimate.required_vram_gb,
                available_vram_gb=estimate.available_vram_gb,
                margin_gb=estimate.margin_gb,
                classification=estimate.classification.value,
                recommended=bool(variant.recommended),
            )
        )

    class_rank = {FitBadge.FITS.value: 0, FitBadge.TIGHT.value: 1, FitBadge.WONT_FIT.value: 2}
    rows.sort(
        key=lambda row: (
            0 if row.recommended else 1,
            class_rank[row.classification],
            -row.params_b,
            -row.bits_effective,
            row.model_name,
        )
    )
    return rows


def _selected_gpu_label(gpus: list[GPU], selected_gpu_id: str | None) -> str:
    if selected_gpu_id is None:
        return ""
    for gpu in gpus:
        if gpu.id == selected_gpu_id:
            return _gpu_option_label(gpu)
    return ""


def _context_label(context_tokens: int) -> str:
    labels = {2048: "2k", 4096: "4k", 8192: "8k"}
    return labels.get(context_tokens, str(context_tokens))


def _selected_hardware_label(
    *,
    selected_gpu_label: str,
    manual_vram: float | None,
    available_vram_gb: float,
) -> str:
    if selected_gpu_label:
        return selected_gpu_label
    if manual_vram is not None:
        return f"Manual VRAM ({manual_vram:g} GB)"
    return f"Assumed VRAM ({available_vram_gb:g} GB)"


def _gpu_option_label(gpu: GPU) -> str:
    return f"{gpu.vendor} {gpu.name} ({gpu.vram_gb:.1f} GB)"


def _normalize_for_match(value: str) -> str:
    return " ".join(value.lower().split())


def _extract_vram_hint(value: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*gb\b", value.lower())
    if match is None:
        return None
    parsed = float(match.group(1))
    if parsed <= 0:
        return None
    return parsed


def _parse_optional_float(
    raw_value: str | float | int | None,
    *,
    field_name: str,
    min_value: float | None = None,
) -> float | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        parsed = float(raw_value)
    else:
        value = raw_value.strip()
        if value == "":
            return None
        try:
            parsed = float(value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid value for {field_name}.") from exc

    if min_value is not None and parsed < min_value:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be greater than or equal to {min_value}.",
        )
    return parsed


def _safe_parse_optional_float(
    raw_value: str | float | int | None,
    *,
    field_name: str,
    min_value: float | None = None,
) -> ParsedFloatResult:
    try:
        parsed = _parse_optional_float(raw_value, field_name=field_name, min_value=min_value)
    except HTTPException as exc:
        return ParsedFloatResult(value=None, error=str(exc.detail))
    return ParsedFloatResult(value=parsed, error=None)


def _query_items(
    *,
    selected_gpu_id: str | None,
    gpu_search: str | None,
    manual_vram: float | None,
    selected_context: int,
    selected_families: list[str],
    selected_quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = [("context_tokens", str(selected_context))]
    if selected_gpu_id:
        items.append(("gpu_id", selected_gpu_id))
    elif gpu_search:
        items.append(("gpu_search", gpu_search))
    if manual_vram is not None:
        items.append(("vram_gb", f"{manual_vram:g}"))
    for family in selected_families:
        items.append(("family", family))
    for quant in selected_quant_buckets:
        items.append(("quant_bucket", quant))
    if min_params_b is not None:
        items.append(("min_params_b", f"{min_params_b:g}"))
    if max_params_b is not None:
        items.append(("max_params_b", f"{max_params_b:g}"))
    if recommended_only:
        items.append(("recommended_only", "true"))
    return items


def _results_url(items: list[tuple[str, str]]) -> str:
    encoded = urlencode(items, doseq=True)
    return f"/results?{encoded}" if encoded else "/results"


def _active_filter_chips(
    *,
    selected_gpu_id: str | None,
    gpu_search: str | None,
    manual_vram: float | None,
    selected_context: int,
    selected_families: list[str],
    selected_quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
) -> list[ActiveFilterChip]:
    chips: list[ActiveFilterChip] = []

    for index, family in enumerate(selected_families):
        updated_families = selected_families[:index] + selected_families[index + 1 :]
        items = _query_items(
            selected_gpu_id=selected_gpu_id,
            gpu_search=gpu_search,
            manual_vram=manual_vram,
            selected_context=selected_context,
            selected_families=updated_families,
            selected_quant_buckets=selected_quant_buckets,
            min_params_b=min_params_b,
            max_params_b=max_params_b,
            recommended_only=recommended_only,
        )
        chips.append(ActiveFilterChip(label=f"Family: {family}", remove_url=_results_url(items)))

    for index, quant in enumerate(selected_quant_buckets):
        updated_quant = selected_quant_buckets[:index] + selected_quant_buckets[index + 1 :]
        items = _query_items(
            selected_gpu_id=selected_gpu_id,
            gpu_search=gpu_search,
            manual_vram=manual_vram,
            selected_context=selected_context,
            selected_families=selected_families,
            selected_quant_buckets=updated_quant,
            min_params_b=min_params_b,
            max_params_b=max_params_b,
            recommended_only=recommended_only,
        )
        chips.append(ActiveFilterChip(label=f"Quant: {quant}", remove_url=_results_url(items)))

    if min_params_b is not None:
        items = _query_items(
            selected_gpu_id=selected_gpu_id,
            gpu_search=gpu_search,
            manual_vram=manual_vram,
            selected_context=selected_context,
            selected_families=selected_families,
            selected_quant_buckets=selected_quant_buckets,
            min_params_b=None,
            max_params_b=max_params_b,
            recommended_only=recommended_only,
        )
        chips.append(
            ActiveFilterChip(
                label=f"Min params >= {min_params_b:g}B",
                remove_url=_results_url(items),
            )
        )

    if max_params_b is not None:
        items = _query_items(
            selected_gpu_id=selected_gpu_id,
            gpu_search=gpu_search,
            manual_vram=manual_vram,
            selected_context=selected_context,
            selected_families=selected_families,
            selected_quant_buckets=selected_quant_buckets,
            min_params_b=min_params_b,
            max_params_b=None,
            recommended_only=recommended_only,
        )
        chips.append(
            ActiveFilterChip(
                label=f"Max params <= {max_params_b:g}B",
                remove_url=_results_url(items),
            )
        )

    if recommended_only:
        items = _query_items(
            selected_gpu_id=selected_gpu_id,
            gpu_search=gpu_search,
            manual_vram=manual_vram,
            selected_context=selected_context,
            selected_families=selected_families,
            selected_quant_buckets=selected_quant_buckets,
            min_params_b=min_params_b,
            max_params_b=max_params_b,
            recommended_only=False,
        )
        chips.append(ActiveFilterChip(label="Recommended only", remove_url=_results_url(items)))

    return chips


def _summary_counts(rows: list[ResultRow]) -> dict[str, int]:
    return {
        FitBadge.FITS.value: sum(1 for row in rows if row.classification == FitBadge.FITS.value),
        FitBadge.TIGHT.value: sum(1 for row in rows if row.classification == FitBadge.TIGHT.value),
        FitBadge.WONT_FIT.value: sum(
            1 for row in rows if row.classification == FitBadge.WONT_FIT.value
        ),
    }


def _top_picks(rows: list[ResultRow]) -> list[TopPick]:
    fits = [row for row in rows if row.classification == FitBadge.FITS.value]
    viable = [
        row for row in rows if row.classification in (FitBadge.FITS.value, FitBadge.TIGHT.value)
    ]
    if not fits or len(viable) < 3:
        return []

    best_quality = fits[0]
    best_balance = next((row for row in viable if row.variant_id != best_quality.variant_id), None)
    if best_balance is None:
        return []

    used_variant_ids = {best_quality.variant_id, best_balance.variant_id}
    best_small = next(
        (row for row in reversed(viable) if row.variant_id not in used_variant_ids),
        None,
    )
    if best_small is None:
        return []

    return [
        TopPick(
            title="Best quality that fits",
            description="Highest-ranked variant that remains in the Fits band.",
            row=best_quality,
        ),
        TopPick(
            title="Best balance",
            description="Next highest-ranked practical option by current ranking order.",
            row=best_balance,
        ),
        TopPick(
            title="Best small",
            description="Lowest-ranked viable option, usually the lightest footprint.",
            row=best_small,
        ),
    ]


def _example_chips(gpus: list[GPU]) -> list[ExampleChip]:
    rtx_4060 = next(
        (gpu for gpu in gpus if "4060" in gpu.name and float(gpu.vram_gb) <= 8.1),
        None,
    )
    if rtx_4060 is not None:
        gpu_chip = ExampleChip(label="RTX 4060 8GB", gpu_id=rtx_4060.id, vram_gb=None)
    else:
        fallback_gpu = gpus[0] if gpus else None
        gpu_chip = ExampleChip(
            label="Example GPU",
            gpu_id=fallback_gpu.id if fallback_gpu else None,
            vram_gb=None,
        )
    return [
        gpu_chip,
        ExampleChip(label="8GB", gpu_id=None, vram_gb=8.0),
        ExampleChip(label="24GB", gpu_id=None, vram_gb=24.0),
    ]


def _results_context(
    request: Request,
    session: Session,
    *,
    available_vram_gb: float,
    gpus: list[GPU],
    results_rows: list[ResultRow],
    selected_gpu_id: str | None,
    gpu_search: str | None,
    manual_vram: float | None,
    selected_context: int,
    selected_families: list[str],
    selected_quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
    form_errors: list[str] | None = None,
    gpu_search_feedback: str | None = None,
    manual_vram_input: str | None = None,
    min_params_b_input: str | None = None,
    max_params_b_input: str | None = None,
) -> dict:
    visible_results = [row for row in results_rows if row.classification != FitBadge.WONT_FIT.value]
    selected_gpu_label = _selected_gpu_label(gpus, selected_gpu_id)
    hardware_items = _query_items(
        selected_gpu_id=selected_gpu_id,
        gpu_search=gpu_search,
        manual_vram=manual_vram,
        selected_context=selected_context,
        selected_families=[],
        selected_quant_buckets=[],
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
    )
    context = _base_context(request, session)
    context.update(
        {
            "gpus": gpus,
            "results": visible_results,
            "families": _all_families(session),
            "quant_options": _all_quant_buckets(session),
            "selected_gpu_id": selected_gpu_id,
            "selected_gpu_label": selected_gpu_label,
            "gpu_search_text": selected_gpu_label or (gpu_search or ""),
            "manual_vram": manual_vram,
            "manual_vram_input": (
                manual_vram_input
                if manual_vram_input is not None
                else (f"{manual_vram:g}" if manual_vram is not None else "")
            ),
            "selected_context": selected_context,
            "selected_families": selected_families,
            "selected_quant_buckets": selected_quant_buckets,
            "min_params_b": min_params_b,
            "max_params_b": max_params_b,
            "min_params_b_input": (
                min_params_b_input
                if min_params_b_input is not None
                else (f"{min_params_b:g}" if min_params_b is not None else "")
            ),
            "max_params_b_input": (
                max_params_b_input
                if max_params_b_input is not None
                else (f"{max_params_b:g}" if max_params_b is not None else "")
            ),
            "recommended_only": recommended_only,
            "form_errors": form_errors or [],
            "gpu_search_feedback": gpu_search_feedback,
            "summary_counts": _summary_counts(results_rows),
            "selected_hardware_label": _selected_hardware_label(
                selected_gpu_label=selected_gpu_label,
                manual_vram=manual_vram,
                available_vram_gb=available_vram_gb,
            ),
            "selected_context_label": _context_label(selected_context),
            "top_picks": _top_picks(visible_results),
            "active_filter_chips": _active_filter_chips(
                selected_gpu_id=selected_gpu_id,
                gpu_search=gpu_search,
                manual_vram=manual_vram,
                selected_context=selected_context,
                selected_families=selected_families,
                selected_quant_buckets=selected_quant_buckets,
                min_params_b=min_params_b,
                max_params_b=max_params_b,
                recommended_only=recommended_only,
            ),
            "hardware_query_string": urlencode(hardware_items, doseq=True),
            "example_chips": _example_chips(gpus),
        }
    )
    return context


def _base_context(request: Request, session: Session) -> dict:
    return {
        "request": request,
        "context_options": CONTEXT_OPTIONS,
        "catalog_version": _catalog_version(session),
        "asset_version": ASSET_VERSION,
        "current_path": request.url.path,
    }


def _results_page_context(
    request: Request,
    session: Session,
    *,
    gpu_id: str | None,
    gpu_search: str | None,
    vram_gb: str | None,
    context_tokens: int,
    family: list[str],
    quant_bucket: list[str],
    min_params_b: str | None,
    max_params_b: str | None,
    recommended_only: bool,
) -> dict:
    form_errors: list[str] = []
    parsed_vram_result = _safe_parse_optional_float(
        vram_gb, field_name="vram_gb", min_value=0.000001
    )
    parsed_min_params_result = _safe_parse_optional_float(
        min_params_b, field_name="min_params_b", min_value=0
    )
    parsed_max_params_result = _safe_parse_optional_float(
        max_params_b, field_name="max_params_b", min_value=0
    )

    if parsed_vram_result.error is not None:
        form_errors.append(parsed_vram_result.error)
    if parsed_min_params_result.error is not None:
        form_errors.append(parsed_min_params_result.error)
    if parsed_max_params_result.error is not None:
        form_errors.append(parsed_max_params_result.error)

    parsed_vram_gb = parsed_vram_result.value
    parsed_min_params_b = parsed_min_params_result.value
    parsed_max_params_b = parsed_max_params_result.value

    if (
        parsed_min_params_b is not None
        and parsed_max_params_b is not None
        and parsed_min_params_b > parsed_max_params_b
    ):
        form_errors.append("min_params_b must be less than or equal to max_params_b.")
        parsed_min_params_b = None
        parsed_max_params_b = None

    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    available_vram_gb, selected_gpu_id, gpus, gpu_search_feedback = _resolve_available_vram(
        session,
        gpu_id=gpu_id,
        vram_gb=parsed_vram_gb,
        gpu_search=gpu_search,
    )
    results_rows = _build_results(
        session,
        available_vram_gb=available_vram_gb,
        context_tokens=context_tokens,
        families=family,
        quant_buckets=quant_bucket,
        min_params_b=parsed_min_params_b,
        max_params_b=parsed_max_params_b,
        recommended_only=recommended_only,
    )
    return _results_context(
        request,
        session,
        available_vram_gb=available_vram_gb,
        gpus=gpus,
        results_rows=results_rows,
        selected_gpu_id=selected_gpu_id,
        gpu_search=gpu_search,
        manual_vram=parsed_vram_gb,
        selected_context=context_tokens,
        selected_families=family,
        selected_quant_buckets=quant_bucket,
        min_params_b=parsed_min_params_b,
        max_params_b=parsed_max_params_b,
        recommended_only=recommended_only,
        form_errors=form_errors,
        gpu_search_feedback=gpu_search_feedback,
        manual_vram_input=vram_gb if isinstance(vram_gb, str) else None,
        min_params_b_input=min_params_b if isinstance(min_params_b, str) else None,
        max_params_b_input=max_params_b if isinstance(max_params_b, str) else None,
    )


def home(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    families: list[str] = []
    quant_buckets: list[str] = []
    available_vram_gb, selected_gpu_id, gpus, _ = _resolve_available_vram(
        session, gpu_id=None, vram_gb=None, gpu_search=None
    )
    results = _build_results(
        session,
        available_vram_gb=available_vram_gb,
        context_tokens=2048,
        families=families,
        quant_buckets=quant_buckets,
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
    )
    context = _results_context(
        request,
        session,
        available_vram_gb=available_vram_gb,
        gpus=gpus,
        results_rows=results,
        selected_gpu_id=selected_gpu_id,
        gpu_search=None,
        manual_vram=None,
        selected_context=2048,
        selected_families=families,
        selected_quant_buckets=quant_buckets,
        min_params_b=None,
        max_params_b=None,
        recommended_only=False,
    )
    return templates.TemplateResponse(request, "index.html", context)


def results(
    request: Request,
    gpu_id: str | None = Query(default=None),
    gpu_search: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    family: list[str] = Query(default=[]),
    quant_bucket: list[str] = Query(default=[]),
    min_params_b: str | None = Query(default=None),
    max_params_b: str | None = Query(default=None),
    recommended_only: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    context = _results_page_context(
        request,
        session,
        gpu_id=gpu_id,
        gpu_search=gpu_search,
        vram_gb=vram_gb,
        context_tokens=context_tokens,
        family=family,
        quant_bucket=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(request, "results_content.html", context)
    return templates.TemplateResponse(request, "index.html", context)


def results_partial_content(
    request: Request,
    gpu_id: str | None = Query(default=None),
    gpu_search: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    family: list[str] = Query(default=[]),
    quant_bucket: list[str] = Query(default=[]),
    min_params_b: str | None = Query(default=None),
    max_params_b: str | None = Query(default=None),
    recommended_only: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    context = _results_page_context(
        request,
        session,
        gpu_id=gpu_id,
        gpu_search=gpu_search,
        vram_gb=vram_gb,
        context_tokens=context_tokens,
        family=family,
        quant_bucket=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )
    return templates.TemplateResponse(request, "results_content.html", context)


def results_partial_summary(
    request: Request,
    gpu_id: str | None = Query(default=None),
    gpu_search: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    family: list[str] = Query(default=[]),
    quant_bucket: list[str] = Query(default=[]),
    min_params_b: str | None = Query(default=None),
    max_params_b: str | None = Query(default=None),
    recommended_only: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    context = _results_page_context(
        request,
        session,
        gpu_id=gpu_id,
        gpu_search=gpu_search,
        vram_gb=vram_gb,
        context_tokens=context_tokens,
        family=family,
        quant_bucket=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )
    return templates.TemplateResponse(request, "results_summary.html", context)


def results_partial_list(
    request: Request,
    gpu_id: str | None = Query(default=None),
    gpu_search: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    family: list[str] = Query(default=[]),
    quant_bucket: list[str] = Query(default=[]),
    min_params_b: str | None = Query(default=None),
    max_params_b: str | None = Query(default=None),
    recommended_only: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    context = _results_page_context(
        request,
        session,
        gpu_id=gpu_id,
        gpu_search=gpu_search,
        vram_gb=vram_gb,
        context_tokens=context_tokens,
        family=family,
        quant_bucket=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )
    return templates.TemplateResponse(request, "results_list.html", context)


def variant_breakdown(
    request: Request,
    variant_id: str,
    gpu_id: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    variant = session.get(Variant, variant_id)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    parsed_vram_gb = _parse_optional_float(vram_gb, field_name="vram_gb", min_value=0.000001)
    available_vram_gb, _, _, _ = _resolve_available_vram(
        session,
        gpu_id=gpu_id,
        vram_gb=parsed_vram_gb,
    )
    model = variant.model
    breakdown = estimate_breakdown(
        params_b=float(model.params_b),
        bits_effective=float(variant.bits_effective),
        kv_gb_per_1k_ctx=float(model.kv_gb_per_1k_ctx),
        context_tokens=context_tokens,
        available_vram_gb=available_vram_gb,
    )
    context = _base_context(request, session)
    context.update(
        {
            "variant": variant,
            "model": model,
            "breakdown": breakdown,
        }
    )
    return templates.TemplateResponse(request, "results_why.html", context)


def model_detail(
    request: Request,
    model_id: str,
    gpu_id: str | None = Query(default=None),
    vram_gb: str | None = Query(default=None),
    context_tokens: int = Query(default=2048),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    model = session.get(Model, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    form_errors: list[str] = []
    parsed_vram_result = _safe_parse_optional_float(
        vram_gb, field_name="vram_gb", min_value=0.000001
    )
    if parsed_vram_result.error is not None:
        form_errors.append(parsed_vram_result.error)
    parsed_vram_gb = parsed_vram_result.value

    available_vram_gb, selected_gpu_id, gpus, _ = _resolve_available_vram(
        session,
        gpu_id=gpu_id,
        vram_gb=parsed_vram_gb,
        prefer_gpu_id_when_both=True,
    )

    variant_rows: list[dict] = []
    for variant in sorted(model.variants, key=lambda item: (item.quant_bucket, item.quant_label)):
        estimate = estimate_variant(
            params_b=float(model.params_b),
            bits_effective=float(variant.bits_effective),
            kv_gb_per_1k_ctx=float(model.kv_gb_per_1k_ctx),
            context_tokens=context_tokens,
            available_vram_gb=available_vram_gb,
        )
        variant_rows.append(
            {
                "id": variant.id,
                "quant_bucket": variant.quant_bucket,
                "quant_label": variant.quant_label,
                "bits_effective": float(variant.bits_effective),
                "required_vram_gb": estimate.required_vram_gb,
                "available_vram_gb": estimate.available_vram_gb,
                "margin_gb": estimate.margin_gb,
                "classification": estimate.classification.value,
                "recommended": bool(variant.recommended),
                "notes": variant.notes,
                "sources": variant.sources,
            }
        )

    context = _base_context(request, session)
    context.update(
        {
            "model": model,
            "variants": variant_rows,
            "gpus": gpus,
            "selected_gpu_id": selected_gpu_id,
            "manual_vram": parsed_vram_gb,
            "manual_vram_input": vram_gb if isinstance(vram_gb, str) else "",
            "selected_context": context_tokens,
            "form_errors": form_errors,
        }
    )
    return templates.TemplateResponse(request, "model_detail.html", context)


def how_it_works(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    return templates.TemplateResponse(request, "how_it_works.html", _base_context(request, session))


def healthz() -> dict[str, str]:
    return {"status": "ok"}


def create_app(settings: Settings | None = None, *, database_url: str | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_database_url = database_url or resolved_settings.database_url
    if not resolved_database_url:
        if resolved_settings.app_env != "test":
            msg = "DATABASE_URL must be set when APP_ENV is not 'test'."
            raise RuntimeError(msg)
        msg = "DATABASE_URL must be provided when APP_ENV is 'test'."
        raise RuntimeError(msg)

    configure_engine(resolved_database_url)

    app = FastAPI(title="VRAM Sherpa")
    app.state.settings = resolved_settings
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
    app.add_event_handler("startup", on_startup)
    app.middleware("http")(host_guard)
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/results", results, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route(
        "/results/partials/content",
        results_partial_content,
        methods=["GET"],
        response_class=HTMLResponse,
    )
    app.add_api_route(
        "/results/partials/summary",
        results_partial_summary,
        methods=["GET"],
        response_class=HTMLResponse,
    )
    app.add_api_route(
        "/results/partials/list",
        results_partial_list,
        methods=["GET"],
        response_class=HTMLResponse,
    )
    app.add_api_route(
        "/results/why/{variant_id}",
        variant_breakdown,
        methods=["GET"],
        response_class=HTMLResponse,
    )
    app.add_api_route(
        "/models/{model_id}", model_detail, methods=["GET"], response_class=HTMLResponse
    )
    app.add_api_route("/how-it-works", how_it_works, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    return app
