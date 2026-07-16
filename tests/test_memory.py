import numpy as np

from dendritron import PPCA, LifecycleState, MemoryRegistry, RecallMode
from dendritron.memory import make_memory_pack


def test_ppca_is_stable_on_nearly_constant_data() -> None:
    rng = np.random.default_rng(2)
    values = np.ones((40, 4)) + rng.normal(0, 1e-10, (40, 4))
    model = PPCA.fit(values, rank=3, floor=1e-6)
    score = model.log_likelihood(values[:3])
    assert np.all(np.isfinite(score))


def test_memory_routing_and_lifecycle() -> None:
    rng = np.random.default_rng(12)
    registry = MemoryRegistry(reliable_k=2, critical_mass=0.99)
    left = make_memory_pack("left", lambda cue: f"L:{cue}", rng.normal(-2, 0.2, (80, 3)))
    right = make_memory_pack("right", lambda cue: f"R:{cue}", rng.normal(2, 0.2, (80, 3)))
    registry.register(left)
    registry.register(right)
    for mode in RecallMode:
        answer, trace = registry.execute("x", np.full(3, 2.0), mode)
        assert trace.selected == "right"
        assert answer == "R:x"
    signature = right.signature()
    archived = registry.uninstall("right")
    assert archived.state is LifecycleState.ARCHIVED
    registry.reinstall(archived)
    assert archived.signature() == signature
    assert archived.state is LifecycleState.REGISTERED


def test_failed_pack_is_quarantined() -> None:
    rng = np.random.default_rng(3)
    pack = make_memory_pack("bad", str, rng.normal(size=(20, 2)), validation_accuracy=0.5)
    registry = MemoryRegistry(registration_threshold=0.8)
    assert registry.register(pack) == "quarantined"
