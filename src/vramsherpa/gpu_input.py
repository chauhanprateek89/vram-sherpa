"""Pure rule for the landing box: is the text a GPU, a VRAM number, or a search?"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol


class GPULike(Protocol):
    vendor: str
    name: str
    vram_gb: float


@dataclass(frozen=True)
class BoxDecision:
    kind: Literal["blank", "gpu", "vram", "search"]
    gpu: GPULike | None = None
    vram_text: str | None = None


# [0-9], not \d: \d also matches other scripts' digits.
_BOX_NUMBER = re.compile(r"([0-9]+(?:\.[0-9]+)?)(?:\s*gb)?", re.IGNORECASE)


def gpu_option_label(gpu: GPULike) -> str:
    return f"{gpu.vendor} {gpu.name} ({gpu.vram_gb:.1f} GB)"


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def exact_gpu_match(text: str, gpus: list[GPULike]) -> GPULike | None:
    needle = normalize_for_match(text)
    matches = [
        gpu
        for gpu in gpus
        if needle
        in {
            normalize_for_match(gpu_option_label(gpu)),
            normalize_for_match(f"{gpu.vendor} {gpu.name}"),
            normalize_for_match(gpu.name),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def parse_box_number(text: str) -> str | None:
    match = _BOX_NUMBER.fullmatch(text.strip())
    return match.group(1) if match else None


def decide_box(text: str, gpus: list[GPULike]) -> BoxDecision:
    if not text.strip():
        return BoxDecision("blank")
    gpu = exact_gpu_match(text, gpus)
    if gpu is not None:
        return BoxDecision("gpu", gpu=gpu)
    number = parse_box_number(text)
    if number is not None:
        return BoxDecision("vram", vram_text=number)
    return BoxDecision("search")
