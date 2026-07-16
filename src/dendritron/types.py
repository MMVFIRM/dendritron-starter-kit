"""Shared types for ownership, verification, and recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class LifecycleState(str, Enum):
    CANDIDATE = "candidate"
    QUARANTINED = "quarantined"
    REGISTERED = "registered"
    DISABLED = "disabled"
    ARCHIVED = "archived"


class RecallMode(str, Enum):
    FAST = "fast"
    EFFICIENT = "efficient"
    RELIABLE = "reliable"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Certificate:
    """Bounded evidence used to admit, verify, or repair a functional owner."""

    inputs: np.ndarray
    targets: np.ndarray
    name: str = "certificate"
    minimum_accuracy: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.inputs)
        y = np.asarray(self.targets)
        if x.ndim < 1 or len(x) != len(y):
            raise ValueError("certificate inputs and targets must align")
        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError("minimum_accuracy must be in [0, 1]")
        object.__setattr__(self, "inputs", x.copy())
        object.__setattr__(self, "targets", y.copy())


@dataclass(frozen=True)
class RegistrationReceipt:
    owner: str
    state: LifecycleState
    accuracy: float
    signature: str
    reason: str = ""


@dataclass(frozen=True)
class InferenceTrace:
    candidates: tuple[str, ...]
    selected: str | None
    mode: RecallMode
    address_probabilities: dict[str, float]
    verifier_scores: dict[str, float]
    abstained: bool = False
