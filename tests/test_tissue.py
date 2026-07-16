import numpy as np

from dendritron import (
    BooleanDendritron,
    Certificate,
    DendritronTissue,
    LifecycleState,
    boolean_cube,
)


def test_registration_is_earned() -> None:
    cube = boolean_cube(2)
    xor = BooleanDendritron.fit(cube, cube[:, 0] ^ cube[:, 1], name="xor")
    tissue = DendritronTissue(registration_threshold=0.8)
    receipt = tissue.register("xor", xor, xor.certificate, immutable=True)
    assert receipt.state is LifecycleState.REGISTERED
    assert tissue.verify("xor") == 1.0


def test_copy_on_write_does_not_mutate_original() -> None:
    cube = boolean_cube(2)
    xor = BooleanDendritron.fit(cube, cube[:, 0] ^ cube[:, 1], name="xor")
    tissue = DendritronTissue()
    tissue.register("xor", xor, xor.certificate, immutable=True)
    original_signature = tissue.regions["xor"].signature
    fork = tissue.fork("xor", "or")
    fork.module.truth_table = (cube[:, 0] | cube[:, 1]).astype(np.int8)
    fork.module.branches = BooleanDendritron.fit(cube, fork.module.truth_table).branches
    certificate = Certificate(cube, fork.module.truth_table, name="or-table")
    receipt = tissue.admit_fork("or", certificate)
    assert receipt.state is LifecycleState.REGISTERED
    assert tissue.regions["xor"].signature == original_signature
    np.testing.assert_array_equal(tissue.execute("xor", cube), cube[:, 0] ^ cube[:, 1])


def test_failed_candidate_is_quarantined() -> None:
    cube = boolean_cube(2)
    wrong = BooleanDendritron.fit(cube, np.zeros(4, dtype=np.int8), name="wrong")
    desired = Certificate(cube, cube[:, 0] ^ cube[:, 1], name="desired")
    tissue = DendritronTissue()
    receipt = tissue.register("wrong", wrong, desired)
    assert receipt.state is LifecycleState.QUARANTINED
