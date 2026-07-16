"""Numerically defensive Euclidean and Poincare-ball chart operations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

EPS = 1e-12


class Geometry(str, Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"


@dataclass(frozen=True)
class Chart:
    """A local coordinate chart attached to a branch, not to its owner."""

    name: str
    geometry: Geometry | str
    dims: tuple[int, ...] | None = None
    curvature: float = 1.0
    tangent_scale: float = 0.18
    input_is_ball: bool = False

    def __post_init__(self) -> None:
        geometry = Geometry(self.geometry)
        if self.curvature <= 0:
            raise ValueError("curvature must be positive")
        if self.dims is not None and len(set(self.dims)) != len(self.dims):
            raise ValueError("chart dimensions must be unique")
        object.__setattr__(self, "geometry", geometry)


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def clip_ball(values: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    values = _finite(values, "values")
    max_norm = (1.0 - 1e-7) / math.sqrt(curvature)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values * np.minimum(1.0, max_norm / np.maximum(norm, EPS))


def expmap0(tangent: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    tangent = _finite(tangent, "tangent")
    sqrt_c = math.sqrt(curvature)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    coefficient = np.tanh(sqrt_c * norm) / np.maximum(sqrt_c * norm, EPS)
    return clip_ball(coefficient * tangent, curvature)


def poincare_distance(a: np.ndarray, b: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """Broadcasted Poincare-ball distance over the final dimension."""

    a = clip_ball(a, curvature)
    b = clip_ball(b, curvature)
    difference_squared = np.sum((a - b) ** 2, axis=-1)
    a_squared = np.sum(a * a, axis=-1)
    b_squared = np.sum(b * b, axis=-1)
    denominator = np.maximum((1.0 - curvature * a_squared) * (1.0 - curvature * b_squared), EPS)
    argument = 1.0 + 2.0 * curvature * difference_squared / denominator
    return np.arccosh(np.maximum(argument, 1.0)) / math.sqrt(curvature)


def project(values: np.ndarray, chart: Chart) -> np.ndarray:
    values = _finite(values, "values")
    projected = values if chart.dims is None else values[..., list(chart.dims)]
    if chart.geometry is Geometry.EUCLIDEAN:
        return projected
    if chart.input_is_ball:
        return clip_ball(projected, chart.curvature)
    return expmap0(chart.tangent_scale * projected, chart.curvature)


def chart_distance(values: np.ndarray, center: np.ndarray, chart: Chart) -> np.ndarray:
    values_projected = project(values, chart)
    center_projected = project(np.asarray(center)[None, :], chart)[0]
    if chart.geometry is Geometry.EUCLIDEAN:
        return np.linalg.norm(values_projected - center_projected, axis=-1)
    return poincare_distance(values_projected, center_projected, chart.curvature)


def mixture_distance(
    values: np.ndarray,
    center: np.ndarray,
    charts: Sequence[Chart],
    weights: np.ndarray,
) -> np.ndarray:
    if not charts:
        raise ValueError("at least one chart is required")
    weights = _finite(weights, "weights").reshape(-1)
    if len(weights) != len(charts) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and align with charts")
    weights = weights / weights.sum()
    distances = np.stack([chart_distance(values, center, chart) for chart in charts], axis=-1)
    return distances @ weights
