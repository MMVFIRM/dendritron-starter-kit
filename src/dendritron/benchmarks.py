"""Fast smoke benchmarks for the installable package."""

from __future__ import annotations

from typing import Any

import numpy as np

from .boolean import BooleanDendritron, ParityTissue, boolean_cube
from .geometry import Chart, Geometry, poincare_distance
from .memory import MemoryRegistry, make_memory_pack
from .plasticity import PlasticDendritronWeb


def boolean_smoke() -> dict[str, Any]:
    cube = boolean_cube(2)
    checked = 0
    for code in range(16):
        targets = np.array([(code >> index) & 1 for index in range(4)], dtype=np.int8)
        model = BooleanDendritron.fit(cube, targets, name=f"f-{code}")
        assert np.array_equal(model(cube), targets)
        assert model.verify_explicit_branch_equivalence()
        checked += 1
    parity_values = boolean_cube(8)
    parity = ParityTissue(8)
    accuracy = float(np.mean(parity(parity_values) == parity_values.sum(axis=1) % 2))
    return {"functions_checked": checked, "parity_8_accuracy": accuracy, "passed": accuracy == 1.0}


def plasticity_smoke(seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    first = rng.normal((-3.0, 0.0), 0.25, size=(100, 2))
    second = rng.normal((3.0, 0.0), 0.25, size=(100, 2))
    model = PlasticDendritronWeb(2, sigma=0.8, seed=seed)
    model.partial_fit(first, np.zeros(len(first), dtype=int))
    before = float(np.mean(model.predict(first) == 0))
    model.partial_fit(second, np.ones(len(second), dtype=int))
    after = float(np.mean(model.predict(first) == 0))
    combined = float(
        np.mean(
            model.predict(np.vstack([first, second]))
            == np.concatenate([np.zeros(100), np.ones(100)])
        )
    )
    return {"old_before": before, "old_after": after, "combined": combined, "passed": after >= 0.95}


def memory_smoke(seed: int = 11) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    registry = MemoryRegistry(reliable_k=2, critical_mass=0.99)
    registry.register(make_memory_pack("left", lambda x: f"left:{x}", rng.normal(-2, 0.2, (80, 3))))
    registry.register(
        make_memory_pack("right", lambda x: f"right:{x}", rng.normal(2, 0.2, (80, 3)))
    )
    value, trace = registry.execute("cue", np.full(3, 2.0), "reliable")
    return {"selected": trace.selected, "value": value, "passed": trace.selected == "right"}


def geometry_smoke() -> dict[str, Any]:
    chart = Chart("tree", Geometry.HYPERBOLIC)
    del chart
    a = np.array([[0.1, 0.0], [0.2, 0.0]])
    distance = poincare_distance(a, np.zeros(2))
    passed = bool(distance[1] > distance[0] > 0)
    return {"distances": distance.tolist(), "passed": passed}


def run_smoke_suite() -> dict[str, dict[str, Any]]:
    return {
        "boolean": boolean_smoke(),
        "plasticity": plasticity_smoke(),
        "memory": memory_smoke(),
        "geometry": geometry_smoke(),
    }
