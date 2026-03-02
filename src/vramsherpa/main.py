from __future__ import annotations

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
ASSET_VERSION = "20260303a"
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
) -> tuple[float, str | None, list[GPU]]:
    gpus = _load_gpus(session)
    if vram_gb is not None:
        return float(vram_gb), None, gpus

    selected_gpu = None
    if gpu_id is not None:
        selected_gpu = session.get(GPU, gpu_id)
    if selected_gpu is None and gpus:
        selected_gpu = gpus[0]
    if selected_gpu is None:
        return 8.0, None, gpus
    return float(selected_gpu.vram_gb), selected_gpu.id, gpus


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


def _gpu_option_label(gpu: GPU) -> str:
    return f"{gpu.vendor} {gpu.name} ({gpu.vram_gb:.1f} GB)"


def _query_items(
    *,
    selected_gpu_id: str | None,
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
        row
        for row in rows
        if row.classification in (FitBadge.FITS.value, FitBadge.TIGHT.value)
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
    gpus: list[GPU],
    results_rows: list[ResultRow],
    selected_gpu_id: str | None,
    manual_vram: float | None,
    selected_context: int,
    selected_families: list[str],
    selected_quant_buckets: list[str],
    min_params_b: float | None,
    max_params_b: float | None,
    recommended_only: bool,
) -> dict:
    hardware_items = _query_items(
        selected_gpu_id=selected_gpu_id,
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
            "results": results_rows,
            "families": _all_families(session),
            "quant_options": _all_quant_buckets(session),
            "selected_gpu_id": selected_gpu_id,
            "selected_gpu_label": _selected_gpu_label(gpus, selected_gpu_id),
            "manual_vram": manual_vram,
            "selected_context": selected_context,
            "selected_families": selected_families,
            "selected_quant_buckets": selected_quant_buckets,
            "min_params_b": min_params_b,
            "max_params_b": max_params_b,
            "recommended_only": recommended_only,
            "summary_counts": _summary_counts(results_rows),
            "top_picks": _top_picks(results_rows),
            "active_filter_chips": _active_filter_chips(
                selected_gpu_id=selected_gpu_id,
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
    }


def home(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    families: list[str] = []
    quant_buckets: list[str] = []
    available_vram_gb, selected_gpu_id, gpus = _resolve_available_vram(
        session, gpu_id=None, vram_gb=None
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
        gpus=gpus,
        results_rows=results,
        selected_gpu_id=selected_gpu_id,
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
    vram_gb: float | None = Query(default=None, gt=0),
    context_tokens: int = Query(default=2048),
    family: list[str] = Query(default=[]),
    quant_bucket: list[str] = Query(default=[]),
    min_params_b: float | None = Query(default=None, gt=0),
    max_params_b: float | None = Query(default=None, gt=0),
    recommended_only: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    available_vram_gb, selected_gpu_id, gpus = _resolve_available_vram(
        session, gpu_id=gpu_id, vram_gb=vram_gb
    )
    results_rows = _build_results(
        session,
        available_vram_gb=available_vram_gb,
        context_tokens=context_tokens,
        families=family,
        quant_buckets=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )

    context = _results_context(
        request,
        session,
        gpus=gpus,
        results_rows=results_rows,
        selected_gpu_id=selected_gpu_id,
        manual_vram=vram_gb,
        selected_context=context_tokens,
        selected_families=family,
        selected_quant_buckets=quant_bucket,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        recommended_only=recommended_only,
    )

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(request, "results_content.html", context)
    return templates.TemplateResponse(request, "index.html", context)


def variant_breakdown(
    request: Request,
    variant_id: str,
    gpu_id: str | None = Query(default=None),
    vram_gb: float | None = Query(default=None, gt=0),
    context_tokens: int = Query(default=2048),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    variant = session.get(Variant, variant_id)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    available_vram_gb, _, _ = _resolve_available_vram(session, gpu_id=gpu_id, vram_gb=vram_gb)
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
    vram_gb: float | None = Query(default=None, gt=0),
    context_tokens: int = Query(default=2048),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    model = session.get(Model, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    if context_tokens not in CONTEXT_OPTIONS:
        context_tokens = 2048

    available_vram_gb, selected_gpu_id, gpus = _resolve_available_vram(
        session, gpu_id=gpu_id, vram_gb=vram_gb
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
            "manual_vram": vram_gb,
            "selected_context": context_tokens,
        }
    )
    return templates.TemplateResponse(request, "model_detail.html", context)


def how_it_works(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    return templates.TemplateResponse(request, "how_it_works.html", _base_context(request, session))


def healthz() -> dict[str, str]:
    return {"status": "ok"}


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    database_url = resolved_settings.database_url
    if not database_url:
        if resolved_settings.app_env != "test":
            msg = "DATABASE_URL must be set when APP_ENV is not 'test'."
            raise RuntimeError(msg)
        msg = "DATABASE_URL must be provided when creating the test app."
        raise RuntimeError(msg)

    configure_engine(database_url)

    app = FastAPI(title="VRAM Sherpa")
    app.state.settings = resolved_settings
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
    app.add_event_handler("startup", on_startup)
    app.middleware("http")(host_guard)
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/results", results, methods=["GET"], response_class=HTMLResponse)
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
