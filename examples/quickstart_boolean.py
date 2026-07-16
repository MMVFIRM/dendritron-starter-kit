"""Compile XOR locally, then compose the same primitive into 16-bit parity."""

import numpy as np

from dendritron import BooleanDendritron, ParityTissue, boolean_cube

cube = boolean_cube(2)
xor = BooleanDendritron.fit(cube, cube[:, 0] ^ cube[:, 1], name="xor")

print("XOR predictions:", xor(cube).tolist())
print("Local certificate accuracy:", xor.verify())
print("Explicit branches equal optimized lookup:", xor.verify_explicit_branch_equivalence())

parity = ParityTissue(16, xor)
rng = np.random.default_rng(9)
values = rng.integers(0, 2, size=(1_000, 16))
targets = values.sum(axis=1) % 2
print("16-bit parity accuracy:", float(np.mean(parity(values) == targets)))
print("Intact XOR Dendritrons:", parity.node_count, "depth:", parity.depth)
