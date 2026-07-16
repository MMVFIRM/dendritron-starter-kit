"""Address, verify, and execute durable functional memories."""

import numpy as np

from dendritron import MemoryRegistry, RecallMode
from dendritron.memory import make_memory_pack

rng = np.random.default_rng(5)
registry = MemoryRegistry(registration_threshold=0.8, reliable_k=2)
registry.register(
    make_memory_pack(
        "negative", lambda cue: f"negative memory handled {cue}", rng.normal(-2, 0.2, (100, 4))
    )
)
registry.register(
    make_memory_pack(
        "positive", lambda cue: f"positive memory handled {cue}", rng.normal(2, 0.2, (100, 4))
    )
)

answer, trace = registry.execute("the cue", np.full(4, 2.1), RecallMode.RELIABLE)
print(answer)
print(trace)
