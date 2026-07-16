"""Add a new local function without retraining old owners."""

import numpy as np

from dendritron import PlasticDendritronWeb

rng = np.random.default_rng(42)
old = rng.normal((-3, 0), 0.3, size=(250, 2))
new = rng.normal((3, 0), 0.3, size=(250, 2))

web = PlasticDendritronWeb(2, sigma=0.9, seed=42)
web.partial_fit(old, np.zeros(len(old), dtype=int))
before = np.mean(web.predict(old) == 0)

web.partial_fit(new, np.ones(len(new), dtype=int))
after = np.mean(web.predict(old) == 0)
new_accuracy = np.mean(web.predict(new) == 1)

print({"old_before": before, "old_after": after, "new": new_accuracy})
print("Structural events:", len(web.events))
