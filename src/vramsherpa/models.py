from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from vramsherpa.database import Base


class GPU(Base):
    __tablename__ = "gpus"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    vram_gb: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    family: Mapped[str] = mapped_column(String(120), nullable=False)
    params_b: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False)
    model_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    license: Mapped[str | None] = mapped_column(String(120), nullable=True)
    kv_gb_per_1k_ctx: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)

    variants: Mapped[list[Variant]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


class Variant(Base):
    __tablename__ = "variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    quant_bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    quant_label: Mapped[str] = mapped_column(String(100), nullable=False)
    bits_effective: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    model: Mapped[Model] = relationship(back_populates="variants")


class CatalogMeta(Base):
    __tablename__ = "catalog_meta"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(200), nullable=False)
