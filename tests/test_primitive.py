import numpy as np
import pytest

from dendritron import Dendritron, LocalBranch


def test_multicompartment_prototype_inference() -> None:
    model = Dendritron.from_prototypes(
        [np.array([-2.0, 0.0]), np.array([2.0, 0.0])], [0, 1], sigma=0.5
    )
    np.testing.assert_array_equal(model.predict(np.array([[-2, 0], [2, 0]])), [0, 1])


def test_owner_mismatch_is_rejected() -> None:
    model = Dendritron(2, 1, name="owner")
    with pytest.raises(ValueError, match="owner"):
        model.add_branch(LocalBranch("someone-else", np.zeros(2), np.ones(1)))


def test_damage_is_local_and_reversible() -> None:
    model = Dendritron(1, 1, name="owner")
    model.add_branch(LocalBranch("owner", np.zeros(1), np.ones(1), branch_id="a"))
    model.add_branch(LocalBranch("owner", np.ones(1) * 10, np.ones(1), branch_id="b"))
    signature = model.signature()
    model.damage("a")
    assert model.signature() != signature
    assert model._branch("b").active
    model.repair("a")
    assert model.signature() == signature
