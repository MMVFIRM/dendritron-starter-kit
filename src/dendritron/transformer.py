"""Optional helpers for frozen-backbone Transformer memory packs.

This module intentionally imports the heavy Transformer stack lazily. The full
SmolLM2-360M v0.4.2 experiment is preserved in benchmarks/archive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def require_transformer_stack() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "Install the Transformer integration with: pip install -e '.[transformer]'"
        ) from error
    return torch, transformers, peft


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_sha256(module: Any, *, exclude: tuple[str, ...] = ("lora_", "adapter")) -> str:
    """Hash non-adapter tensors to prove frozen-backbone retention."""

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if any(token in name.lower() for token in exclude):
            continue
        digest.update(name.encode())
        tensor = parameter.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class FrozenBackboneGuard:
    def __init__(self, module: Any) -> None:
        self.before = module_sha256(module)

    def assert_unchanged(self, module: Any) -> None:
        after = module_sha256(module)
        if after != self.before:
            raise RuntimeError("frozen backbone changed while training a memory pack")
