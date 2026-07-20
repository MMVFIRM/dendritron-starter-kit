"""Acceptance-gate evaluation for benchmark integrity checks.

Integrity gates compare a reloaded or reinstalled model against a recorded
reference. Raw-logit equality is stricter than functional equivalence: a model
whose predictions, candidate sets, and parameter hashes all survive a reload
can still show O(1) absolute logit deltas when it computes in a reduced
precision dtype such as bfloat16, because kernel reduction order is not
guaranteed to be identical across model instances.

This module evaluates logit equivalence against three calibrated criteria,
in order:

1. an absolute tolerance (appropriate for float32 compute);
2. a measured run-to-run noise floor, scaled by a safety factor;
3. a scale-relative tolerance, ``max |a - b| / max(1, RMS(a))`` (default
   0.15: bf16 accumulation over a deep stack yields scaled errors of a few
   percent, so 15% gives roughly 3x headroom over observed reload noise
   while still rejecting genuine weight changes, which produce deltas on
   the order of the logit scale itself).

The scale-relative criterion matches the reduced-precision noise model:
bfloat16 rounding error is proportional to the magnitude of the values
flowing through the network, so the expected absolute error on any one logit
scales with the overall logit scale, not with that logit's own value. (A
per-element relative criterion would spuriously fail near-zero logits.)

Every check records which criterion decided it, so a failed gate is
diagnosable from the summary artifact alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "LogitEquivalence",
    "evaluate_gate",
    "evaluate_logit_equivalence",
    "logit_scale",
    "scaled_logit_delta",
]


def logit_scale(reference: np.ndarray) -> float:
    """Root-mean-square logit magnitude, floored at 1."""

    reference = np.asarray(reference, dtype=np.float64)
    if reference.size == 0:
        return 1.0
    return float(max(1.0, np.sqrt(np.mean(reference * reference))))


def scaled_logit_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Largest absolute elementwise delta normalized by the table's scale."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference.size == 0:
        return 0.0
    return float(np.max(np.abs(reference - candidate))) / logit_scale(reference)


@dataclass
class LogitEquivalence:
    """Recorded outcome of one reload/reinstall logit comparison."""

    name: str
    passed: bool
    basis: str
    max_abs_delta: float
    scaled_delta: float
    scale: float
    tolerance_used: float
    noise_floor: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_logit_equivalence(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.15,
    noise_floor: float | None = None,
    noise_factor: float = 4.0,
) -> LogitEquivalence:
    """Decide whether two logit tables are equivalent up to compute noise.

    The check passes on the first criterion that accepts the delta:

    - ``absolute``: ``max |a - b| <= absolute_tolerance``;
    - ``noise_floor``: ``max |a - b| <= noise_floor * noise_factor``, when a
      measured run-to-run noise floor is supplied;
    - ``scale_relative``: ``max |a - b| / max(1, RMS(a)) <=
      relative_tolerance``.

    If none accept, the check fails with basis ``"failed"`` and the relative
    tolerance is reported as the binding constraint.
    """

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if noise_factor < 1.0:
        raise ValueError("noise_factor must be >= 1")
    if noise_floor is not None and noise_floor < 0:
        raise ValueError("noise_floor must be non-negative")

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    abs_delta = float(np.max(np.abs(reference - candidate))) if reference.size else 0.0
    scale = logit_scale(reference)
    scaled = abs_delta / scale

    if abs_delta <= absolute_tolerance:
        return LogitEquivalence(
            name, True, "absolute", abs_delta, scaled, scale, absolute_tolerance, noise_floor
        )
    if noise_floor is not None and abs_delta <= noise_floor * noise_factor:
        return LogitEquivalence(
            name,
            True,
            "noise_floor",
            abs_delta,
            scaled,
            scale,
            noise_floor * noise_factor,
            noise_floor,
        )
    if scaled <= relative_tolerance:
        return LogitEquivalence(
            name, True, "scale_relative", abs_delta, scaled, scale, relative_tolerance, noise_floor
        )
    return LogitEquivalence(
        name, False, "failed", abs_delta, scaled, scale, relative_tolerance, noise_floor
    )


def evaluate_gate(
    results: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    upper_bound_keys: frozenset[str] | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate scalar gate criteria and report every failure by name.

    A criterion passes when ``results[key] >= threshold``, except for keys in
    ``upper_bound_keys`` (for example average candidate counts), which pass
    when ``results[key] <= threshold``. Missing or non-finite values fail.
    """

    upper = upper_bound_keys or frozenset()
    failures: list[str] = []
    for key, threshold in thresholds.items():
        if key not in results:
            failures.append(f"{key}:missing")
            continue
        value = float(results[key])
        if not np.isfinite(value):
            failures.append(f"{key}:non-finite")
            continue
        if key in upper:
            if value > threshold:
                failures.append(f"{key}:{value}>{threshold}")
        elif value < threshold:
            failures.append(f"{key}:{value}<{threshold}")
    return (not failures), failures
