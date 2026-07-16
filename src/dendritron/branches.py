"""Locally owned nonlinear branches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Chart, Geometry, chart_distance, mixture_distance


@dataclass
class LocalBranch:
    """A local RBF compartment with explicit functional ownership."""

    owner: str
    center: np.ndarray
    output: np.ndarray
    sigma: float = 1.0
    charts: tuple[Chart, ...] = field(
        default_factory=lambda: (Chart("euclidean", Geometry.EUCLIDEAN),)
    )
    chart_weights: np.ndarray | None = None
    branch_id: str = "branch"
    active: bool = True
    count: int = 1

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).copy()
        self.output = np.asarray(self.output, dtype=np.float64).reshape(-1).copy()
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not self.charts:
            raise ValueError("a branch needs at least one chart")
        if self.chart_weights is None:
            self.chart_weights = np.ones(len(self.charts), dtype=np.float64) / len(self.charts)
        else:
            weights = np.asarray(self.chart_weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(self.charts) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("invalid chart weights")
            self.chart_weights = weights / weights.sum()

    def distance(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if len(self.charts) == 1:
            return chart_distance(values, self.center, self.charts[0])
        assert self.chart_weights is not None
        return mixture_distance(values, self.center, self.charts, self.chart_weights)

    def activation(self, values: np.ndarray) -> np.ndarray:
        if not self.active:
            values = np.asarray(values)
            return np.zeros(1 if values.ndim == 1 else len(values))
        distance = self.distance(values)
        return np.exp(-0.5 * (distance / max(self.sigma, 1e-8)) ** 2)

    def response(self, values: np.ndarray) -> np.ndarray:
        return self.activation(values)[:, None] * self.output[None, :]

    def update_local(self, value: np.ndarray, learning_rate: float | None = None) -> None:
        """Update only this branch's center; no global parameter is touched."""

        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.center.shape:
            raise ValueError("value shape does not match branch center")
        self.count += 1
        rate = min(0.1, 1.0 / math.sqrt(self.count)) if learning_rate is None else learning_rate
        if not 0.0 < rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.center = (1.0 - rate) * self.center + rate * value


@dataclass(frozen=True)
class MintermBranch:
    """Exact Boolean minterm compartment."""

    pattern: np.ndarray
    value: float = 1.0

    def activation(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        literals = np.where(np.asarray(self.pattern)[None, :] == 1, values, 1.0 - values)
        return np.prod(literals, axis=1)
