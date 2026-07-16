"""The canonical multicompartment Dendritron primitive."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence

import numpy as np

from .branches import LocalBranch


class Dendritron:
    """A routed, locally stateful, multicompartment computational unit.

    Branches are nonlinear local owners. Integration can use a sum, maximum,
    or noisy-OR coalition. The class deliberately exposes branch state so that
    ownership, damage, and repair remain auditable.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        name: str = "dendritron",
        integration: str = "sum",
        threshold: float | None = None,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if integration not in {"sum", "max", "noisy_or"}:
            raise ValueError("integration must be sum, max, or noisy_or")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.integration = integration
        self.threshold = threshold
        self.branches: list[LocalBranch] = []

    def add_branch(self, branch: LocalBranch) -> None:
        if branch.center.shape != (self.input_dim,):
            raise ValueError("branch center has the wrong input dimension")
        if branch.output.shape != (self.output_dim,):
            raise ValueError("branch output has the wrong output dimension")
        if branch.owner != self.name:
            raise ValueError("branch owner must match the Dendritron owner")
        if any(existing.branch_id == branch.branch_id for existing in self.branches):
            raise ValueError(f"duplicate branch id: {branch.branch_id}")
        self.branches.append(branch)

    def forward(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"expected [samples, {self.input_dim}] input")
        if not self.branches:
            return np.zeros((len(values), self.output_dim), dtype=np.float64)
        responses = np.stack([branch.response(values) for branch in self.branches], axis=1)
        if self.integration == "sum":
            return responses.sum(axis=1)
        if self.integration == "max":
            return responses.max(axis=1)
        clipped = np.clip(responses, 0.0, 1.0 - 1e-9)
        return 1.0 - np.prod(1.0 - clipped, axis=1)

    __call__ = forward

    def predict(self, values: np.ndarray) -> np.ndarray:
        output = self.forward(values)
        if self.output_dim == 1:
            threshold = 0.5 if self.threshold is None else self.threshold
            return (output[:, 0] >= threshold).astype(np.int64)
        return output.argmax(axis=1)

    def damage(self, branch_id: str) -> None:
        self._branch(branch_id).active = False

    def repair(self, branch_id: str) -> None:
        self._branch(branch_id).active = True

    def _branch(self, branch_id: str) -> LocalBranch:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(branch_id)

    def clone(self, *, name: str | None = None) -> Dendritron:
        clone = copy.deepcopy(self)
        if name is not None:
            old_name = clone.name
            clone.name = name
            for branch in clone.branches:
                if branch.owner == old_name:
                    branch.owner = name
        return clone

    def signature(self) -> str:
        payload = {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "integration": self.integration,
            "threshold": self.threshold,
            "branches": [
                {
                    "id": branch.branch_id,
                    "owner": branch.owner,
                    "center": branch.center.tolist(),
                    "output": branch.output.tolist(),
                    "sigma": branch.sigma,
                    "active": branch.active,
                    "charts": [chart.name for chart in branch.charts],
                }
                for branch in self.branches
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_prototypes(
        cls,
        centers: Sequence[np.ndarray],
        labels: Sequence[int],
        *,
        sigma: float = 1.0,
        name: str = "prototype-dendritron",
    ) -> Dendritron:
        if len(centers) != len(labels) or not centers:
            raise ValueError("centers and labels must be non-empty and align")
        input_dim = len(np.asarray(centers[0]))
        output_dim = int(max(labels)) + 1
        model = cls(input_dim, output_dim, name=name, integration="max")
        for index, (center, label) in enumerate(zip(centers, labels, strict=True)):
            output = np.zeros(output_dim)
            output[int(label)] = 1.0
            model.add_branch(
                LocalBranch(
                    owner=name,
                    center=np.asarray(center),
                    output=output,
                    sigma=sigma,
                    branch_id=f"prototype-{index}",
                )
            )
        return model
