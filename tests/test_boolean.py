import numpy as np
import pytest

from dendritron import BooleanDendritron, ParityTissue, boolean_cube


def test_all_two_input_boolean_functions_compile_exactly() -> None:
    cube = boolean_cube(2)
    for code in range(16):
        targets = np.array([(code >> index) & 1 for index in range(4)], dtype=np.int8)
        model = BooleanDendritron.fit(cube, targets, name=f"f-{code}")
        np.testing.assert_array_equal(model(cube), targets)
        assert model.verify() == 1.0
        assert model.verify_explicit_branch_equivalence()


def test_incomplete_truth_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete"):
        BooleanDendritron.fit(boolean_cube(2)[:-1], np.array([0, 1, 1]))


@pytest.mark.parametrize("bits", [1, 2, 3, 8, 17])
def test_recursive_parity(bits: int) -> None:
    rng = np.random.default_rng(bits)
    values = rng.integers(0, 2, size=(200, bits))
    model = ParityTissue(bits)
    np.testing.assert_array_equal(model(values), values.sum(axis=1) % 2)
    assert model.node_count == max(0, bits - 1)


def test_damage_and_local_repair() -> None:
    cube = boolean_cube(2)
    model = BooleanDendritron.fit(cube, cube[:, 0] ^ cube[:, 1])
    model.damage_branch(0)
    assert model.verify() < 1.0
    model.repair_from_certificate()
    assert model.verify() == 1.0
