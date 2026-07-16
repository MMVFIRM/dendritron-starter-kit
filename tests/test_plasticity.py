import numpy as np

from dendritron import Chart, Geometry, MixedGeometryWeb, PlasticDendritronWeb


def two_clusters(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.normal((-3, 0), 0.2, size=(150, 2))
    second = rng.normal((3, 0), 0.2, size=(150, 2))
    return first, second


def test_new_owner_does_not_erase_old_owner() -> None:
    first, second = two_clusters()
    model = PlasticDendritronWeb(2, sigma=0.8, seed=4)
    model.partial_fit(first, np.zeros(len(first), dtype=int))
    model.partial_fit(second, np.ones(len(second), dtype=int))
    assert np.mean(model.predict(first) == 0) >= 0.95
    assert np.mean(model.predict(second) == 1) >= 0.95


def test_local_damage_and_repair() -> None:
    first, second = two_clusters()
    model = PlasticDendritronWeb(2, sigma=0.8, seed=4)
    model.partial_fit(first, np.zeros(len(first), dtype=int))
    model.partial_fit(second, np.ones(len(second), dtype=int))
    damaged = model.damage(0, fraction=1.0)
    assert damaged
    model.repair(0)
    assert np.mean(model.predict(first) == 0) >= 0.95


def test_mixed_geometry_keeps_explicit_chart_usage() -> None:
    first, second = two_clusters()
    charts = (
        Chart("flat", Geometry.EUCLIDEAN),
        Chart("tree", Geometry.HYPERBOLIC, tangent_scale=0.15),
    )
    model = MixedGeometryWeb(2, charts=charts, sigma=0.8, seed=4)
    model.partial_fit(first, np.zeros(len(first), dtype=int))
    model.partial_fit(second, np.ones(len(second), dtype=int))
    usage = model.chart_usage()
    assert set(usage) == {"flat", "tree"}
    assert np.isclose(sum(usage.values()), 1.0)
