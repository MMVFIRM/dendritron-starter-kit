"""Contract tests for the benchmark acceptance-gate evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from dendritron.gating import (
    evaluate_gate,
    evaluate_logit_equivalence,
    logit_scale,
    scaled_logit_delta,
)


def test_scaled_delta_normalizes_by_table_scale() -> None:
    reference = np.array([[10.0, 0.0], [-20.0, 0.0]])
    candidate = reference + np.array([[1.0, 0.4], [2.0, 0.3]])
    # RMS scale = sqrt((100 + 0 + 400 + 0) / 4) = sqrt(125); max abs delta = 2.
    assert logit_scale(reference) == pytest.approx(np.sqrt(125.0))
    assert scaled_logit_delta(reference, candidate) == pytest.approx(2.0 / np.sqrt(125.0))


def test_scaled_delta_tolerates_noise_on_near_zero_logits() -> None:
    # bf16-scale noise is proportional to network magnitudes, not to each
    # logit's own value; a near-zero logit must not dominate the criterion.
    reference = np.array([[15.0, 0.01], [-14.0, 0.0]])
    candidate = reference + 0.5
    assert scaled_logit_delta(reference, candidate) < 0.05


def test_scaled_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        scaled_logit_delta(np.zeros((2, 2)), np.zeros(3))


def test_absolute_criterion_passes_float32_grade_match() -> None:
    reference = np.array([[1.0, 2.0]])
    candidate = reference + 1e-4
    check = evaluate_logit_equivalence(
        "checkpoint", reference, candidate, absolute_tolerance=2e-3
    )
    assert check.passed
    assert check.basis == "absolute"
    assert check.tolerance_used == pytest.approx(2e-3)


def test_noise_floor_criterion_accepts_delta_within_measured_noise() -> None:
    reference = np.array([[12.0, -8.0]])
    candidate = reference + 0.05
    check = evaluate_logit_equivalence(
        "checkpoint",
        reference,
        candidate,
        absolute_tolerance=2e-3,
        noise_floor=0.02,
        noise_factor=4.0,
    )
    assert check.passed
    assert check.basis == "noise_floor"
    assert check.tolerance_used == pytest.approx(0.08)


def test_scale_relative_criterion_accepts_bf16_scale_drift() -> None:
    reference = np.array([[16.0, -24.0]])
    candidate = reference + np.array([[1.0, -0.75]])
    check = evaluate_logit_equivalence(
        "reinstall",
        reference,
        candidate,
        absolute_tolerance=2e-3,
        relative_tolerance=0.10,
    )
    assert check.passed
    assert check.basis == "scale_relative"
    assert check.max_abs_delta == pytest.approx(1.0)
    assert check.scaled_delta == pytest.approx(1.0 / logit_scale(reference))


def test_real_disagreement_fails_all_criteria() -> None:
    reference = np.array([[1.0, 2.0]])
    candidate = reference + 5.0
    check = evaluate_logit_equivalence(
        "checkpoint",
        reference,
        candidate,
        absolute_tolerance=2e-3,
        relative_tolerance=0.10,
        noise_floor=0.02,
    )
    assert not check.passed
    assert check.basis == "failed"


def test_invalid_inputs_raise() -> None:
    reference = np.zeros((1, 2))
    with pytest.raises(ValueError):
        evaluate_logit_equivalence("x", reference, reference, absolute_tolerance=-1.0)
    with pytest.raises(ValueError):
        evaluate_logit_equivalence(
            "x", reference, reference, absolute_tolerance=1.0, noise_factor=0.5
        )
    with pytest.raises(ValueError):
        evaluate_logit_equivalence("x", reference, np.zeros(3), absolute_tolerance=1.0)


def test_evaluate_gate_reports_each_failure() -> None:
    thresholds = {"accuracy": 0.9, "hash_retention": 1.0, "average_candidates": 3.2}
    results = {"accuracy": 0.97, "hash_retention": 0.5, "average_candidates": 4.0}
    passed, failures = evaluate_gate(
        results, thresholds, upper_bound_keys=frozenset({"average_candidates"})
    )
    assert not passed
    assert any(f.startswith("hash_retention:") for f in failures)
    assert any(f.startswith("average_candidates:") for f in failures)
    assert not any(f.startswith("accuracy") for f in failures)


def test_evaluate_gate_missing_and_nonfinite_values_fail() -> None:
    passed, failures = evaluate_gate(
        {"accuracy": float("nan")}, {"accuracy": 0.9, "hash_retention": 1.0}
    )
    assert not passed
    assert "accuracy:non-finite" in failures
    assert "hash_retention:missing" in failures


def test_evaluate_gate_passes_when_all_criteria_hold() -> None:
    passed, failures = evaluate_gate(
        {"accuracy": 0.97, "logit_equivalence": 1.0}, {"accuracy": 0.9, "logit_equivalence": 1.0}
    )
    assert passed
    assert failures == []
