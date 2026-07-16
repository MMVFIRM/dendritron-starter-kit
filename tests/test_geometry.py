import numpy as np

from dendritron import Chart, Geometry, expmap0, poincare_distance
from dendritron.geometry import chart_distance, clip_ball


def test_expmap_stays_inside_poincare_ball() -> None:
    values = expmap0(np.array([[0.0, 0.0], [100.0, 0.0]]))
    assert np.all(np.linalg.norm(values, axis=1) < 1.0)


def test_poincare_distance_is_symmetric_and_zero_on_diagonal() -> None:
    a = np.array([[0.1, 0.2], [0.2, -0.1]])
    b = np.array([[0.2, 0.1], [-0.1, 0.2]])
    np.testing.assert_allclose(poincare_distance(a, b), poincare_distance(b, a))
    np.testing.assert_allclose(poincare_distance(a, a), 0.0, atol=1e-8)


def test_compartment_dimensions_are_respected() -> None:
    chart = Chart("first-two", Geometry.EUCLIDEAN, dims=(0, 1))
    values = np.array([[1.0, 2.0, 1000.0]])
    center = np.array([1.0, 2.0, -1000.0])
    np.testing.assert_allclose(chart_distance(values, center, chart), 0.0)


def test_clip_ball_is_finite() -> None:
    clipped = clip_ball(np.array([[1e30, 1e30]]))
    assert np.all(np.isfinite(clipped))
    assert np.linalg.norm(clipped) < 1.0
