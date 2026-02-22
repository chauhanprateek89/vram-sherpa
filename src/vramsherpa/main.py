from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, distinct, select
from sqlalchemy.orm import Session

from vramsherpa.database import create_db_and_tables, get_session
from vramsherpa.estimation import FitBadge, estimate_variant
from vramsherpa.models import GPU, CatalogMeta, Model, Variant

CONTEXT_OPTIONS = (2048, 4096, 8192)
PACKAGE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="VRAM Sherpa")
app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


@app.middleware("http")
async def host_guard(request: Request, call_next):
    from vramsherpa.config import get_settings

    settings = get_settings()
    allowed_hosts = settings.allowed_hosts
    if "*" not in allowed_hosts:
        host = request.headers.get("host", "").split(":", 1)[0]
        if host and host not in allowed_hosts:
            return JSONResponse({"detail": "Host not allowed"}, status_code=400)
    return await call_next(request)


@dataclass
class ResultRow:
    variant_id: int
    model_id: int
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
    gpu_id: int | None,
    vram_gb: float | None,
) -> tuple[float, int | None, list[GPU]]:
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


def _base_context(request: Request, session: Session) -> dict:
    return {
        "request": request,
        "context_options": CONTEXT_OPTIONS,
        "catalog_version": _catalog_version(session),
    }


@app.get("/", response_class=HTMLResponse)
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
    context = _base_context(request, session)
    context.update(
        {
            "gpus": gpus,
            "results": results,
            "families": _all_families(session),
            "quant_options": _all_quant_buckets(session),
            "selected_gpu_id": selected_gpu_id,
            "manual_vram": None,
            "selected_context": 2048,
            "selected_families": families,
            "selected_quant_buckets": quant_buckets,
            "min_params_b": None,
            "max_params_b": None,
            "recommended_only": False,
        }
    )
    return templates.TemplateResponse("index.html", context)


@app.get("/results", response_class=HTMLResponse)
def results(
    request: Request,
    gpu_id: int | None = Query(default=None),
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

    context = _base_context(request, session)
    context.update(
        {
            "gpus": gpus,
            "results": results_rows,
            "families": _all_families(session),
            "quant_options": _all_quant_buckets(session),
            "selected_gpu_id": selected_gpu_id,
            "manual_vram": vram_gb,
            "selected_context": context_tokens,
            "selected_families": family,
            "selected_quant_buckets": quant_bucket,
            "min_params_b": min_params_b,
            "max_params_b": max_params_b,
            "recommended_only": recommended_only,
        }
    )

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse("results_table.html", context)
    return templates.TemplateResponse("index.html", context)


@app.get("/models/{model_id}", response_class=HTMLResponse)
def model_detail(
    request: Request,
    model_id: int,
    gpu_id: int | None = Query(default=None),
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
    return templates.TemplateResponse("model_detail.html", context)


@app.get("/how-it-works", response_class=HTMLResponse)
def how_it_works(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    return templates.TemplateResponse("how_it_works.html", _base_context(request, session))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
