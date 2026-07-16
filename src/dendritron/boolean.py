"""Exact bounded Boolean compilation and recursive composition."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from itertools import product

import numpy as np

from .branches import MintermBranch
from .types import Certificate


def boolean_cube(dimension: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return np.array(list(product([0, 1], repeat=dimension)), dtype=np.int8)


def pattern_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int8)
    if values.ndim == 1:
        values = values[None, :]
    powers: np.ndarray = (2 ** np.arange(values.shape[1] - 1, -1, -1)).astype(np.int64)
    return values.astype(np.int64) @ powers


class BooleanDendritron:
    """Exact local Boolean Dendritron compiled from a complete certificate."""

    output_dim = 1

    def __init__(
        self,
        input_dim: int,
        branches: list[MintermBranch],
        truth_table: np.ndarray,
        certificate: Certificate,
        *,
        name: str = "boolean-dendritron",
    ) -> None:
        self.input_dim = input_dim
        self.branches = list(branches)
        self.truth_table = np.asarray(truth_table, dtype=np.int8).copy()
        self.certificate = certificate
        self.name = name
        self.enabled = True

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        targets: np.ndarray,
        *,
        name: str = "boolean-dendritron",
    ) -> BooleanDendritron:
        values = np.asarray(values, dtype=np.int8)
        targets = np.asarray(targets, dtype=np.int8).reshape(-1)
        if values.ndim != 2 or len(values) != len(targets):
            raise ValueError("values must be [samples, features] and align with targets")
        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("Boolean compilation requires binary inputs")
        expected = boolean_cube(values.shape[1])
        if not np.array_equal(values, expected):
            raise ValueError("exact compilation requires a complete lexicographic truth table")
        if not set(np.unique(targets)).issubset({0, 1}):
            raise ValueError("Boolean compilation requires binary targets")
        branches = [
            MintermBranch(pattern=row.copy())
            for row, target in zip(values, targets, strict=True)
            if target == 1
        ]
        certificate = Certificate(values, targets, name=f"{name}-truth-table")
        return cls(values.shape[1], branches, targets, certificate, name=name)

    def branch_raw(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if not self.enabled:
            return np.zeros(len(values))
        result = np.zeros(len(values))
        for branch in self.branches:
            result += branch.value * branch.activation(values)
        return result

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int8)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} Boolean inputs")
        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("inference values must be binary")
        if not self.enabled:
            return np.zeros(len(values), dtype=np.int8)
        return self.truth_table[pattern_indices(values)]

    predict = __call__

    def verify(self, certificate: Certificate | None = None) -> float:
        certificate = self.certificate if certificate is None else certificate
        return float(np.mean(self(certificate.inputs) == certificate.targets))

    def verify_explicit_branch_equivalence(self) -> bool:
        explicit = (self.branch_raw(self.certificate.inputs) >= 0.5).astype(np.int8)
        return bool(np.array_equal(explicit, self(self.certificate.inputs)))

    def damage_branch(self, branch_index: int) -> None:
        if not 0 <= branch_index < len(self.branches):
            raise IndexError("branch index out of range")
        pattern = self.branches[branch_index].pattern.copy()
        del self.branches[branch_index]
        self.truth_table[pattern_indices(pattern)[0]] = 0

    def repair_from_certificate(self) -> None:
        repaired = type(self).fit(
            self.certificate.inputs,
            self.certificate.targets,
            name=self.name,
        )
        self.branches = repaired.branches
        self.truth_table = repaired.truth_table
        self.enabled = True

    def clone(self, *, name: str | None = None) -> BooleanDendritron:
        clone = copy.deepcopy(self)
        if name is not None:
            clone.name = name
        return clone

    @property
    def stored_scalar_equivalents(self) -> int:
        return len(self.branches) * (self.input_dim + 1) + 1

    def signature(self) -> str:
        payload = {
            "name": self.name,
            "input_dim": self.input_dim,
            "enabled": self.enabled,
            "truth_table": self.truth_table.tolist(),
            "branches": [branch.pattern.tolist() for branch in self.branches],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ParityTissue:
    """Balanced recursive composition of intact two-input XOR Dendritrons."""

    def __init__(self, n_bits: int, xor_region: BooleanDendritron | None = None) -> None:
        if n_bits < 1:
            raise ValueError("n_bits must be positive")
        self.input_dim = n_bits
        if xor_region is None:
            cube = boolean_cube(2)
            xor_region = BooleanDendritron.fit(cube, cube[:, 0] ^ cube[:, 1], name="xor")
        if xor_region.input_dim != 2 or xor_region.verify() < 1.0:
            raise ValueError("xor_region must be a verified two-input Dendritron")
        self.xor_region = xor_region
        self.node_count = max(0, n_bits - 1)
        self.depth = 0 if n_bits == 1 else math.ceil(math.log2(n_bits))

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int8)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} bits")
        layer = values.copy()
        while layer.shape[1] > 1:
            next_values = []
            index = 0
            while index + 1 < layer.shape[1]:
                next_values.append(self.xor_region(layer[:, index : index + 2]))
                index += 2
            if index < layer.shape[1]:
                next_values.append(layer[:, index])
            layer = np.stack(next_values, axis=1)
        return layer[:, 0]

    predict = __call__


def bfs_connected(grid: np.ndarray) -> int:
    """Reference connectivity oracle used by the connectedness benchmark."""

    grid = np.asarray(grid, dtype=np.int8)
    occupied = np.argwhere(grid == 1)
    if len(occupied) == 0:
        return 0
    start = tuple(occupied[0])
    stack = [start]
    visited = {start}
    while stack:
        row, column = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = (row + dr, column + dc)
            if (
                0 <= candidate[0] < grid.shape[0]
                and 0 <= candidate[1] < grid.shape[1]
                and grid[candidate] == 1
                and candidate not in visited
            ):
                visited.add(candidate)
                stack.append(candidate)
    return int(len(visited) == len(occupied))
