"""One web with flat and hierarchy-preserving local charts."""

import numpy as np

from dendritron import Chart, Geometry, MixedGeometryWeb

charts = (
    Chart("flat", Geometry.EUCLIDEAN),
    Chart("hierarchy", Geometry.HYPERBOLIC, curvature=1.0, tangent_scale=0.16),
)
web = MixedGeometryWeb(2, charts=charts, sigma=0.7, seed=3)
rng = np.random.default_rng(3)

left = rng.normal((-2, 0), 0.15, size=(120, 2))
right = rng.normal((2, 0), 0.15, size=(120, 2))
web.partial_fit(left, np.zeros(len(left), dtype=int))
web.partial_fit(right, np.ones(len(right), dtype=int))

values = np.vstack([left, right])
targets = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
print("Accuracy:", float(np.mean(web.predict(values) == targets)))
print("Chart usage:", web.chart_usage())
