"""
Dendritron–Minsky Benchmark v0.1

A constructive, falsifiable prototype for testing whether a Dendritron primitive
can escape the classic single-layer perceptron limitations while preserving:
  1. primitive integrity,
  2. recursive composability,
  3. non-destructive growth,
  4. local fault detection and repair.

This is not yet evidence that Dendritrons outperform optimized MLPs on general
continuous tasks. It is a proof-of-construction benchmark for XOR, parity,
connectedness, compositional closure, and local repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

SEED = 7
RNG = np.random.default_rng(SEED)


# -----------------------------------------------------------------------------
# 1. Exact certificate: a single affine threshold unit cannot represent XOR.
# -----------------------------------------------------------------------------

def xor_perceptron_contradiction() -> str:
    """Return the four inequalities and their contradiction.

    For XOR with positive points (1,0), (0,1) and negative points (0,0), (1,1):
      b < 0
      w1 + b > 0
      w2 + b > 0
      w1 + w2 + b < 0
    The middle inequalities imply w1 > -b and w2 > -b, hence
      w1 + w2 + b > -b > 0,
    contradicting the final inequality.
    """
    return (
        "b < 0; w1+b > 0; w2+b > 0; w1+w2+b < 0. "
        "But w1>-b and w2>-b imply w1+w2+b>-b>0: contradiction."
    )


# -----------------------------------------------------------------------------
# 2. Singular Dendritron: local multidimensional product branches.
# -----------------------------------------------------------------------------

@dataclass
class ProductBranch:
    """A local multidimensional conjunction branch.

    On binary inputs, activation is exactly 1 when x matches `pattern`, else 0.
    On continuous inputs in [0,1]^d, it is a differentiable multilinear term.
    """

    pattern: np.ndarray
    value: float

    def activation(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        literals = np.where(self.pattern[None, :] == 1.0, x, 1.0 - x)
        return np.prod(literals, axis=1)


class BooleanDendritron:
    """A multicompartment Boolean Dendritron.

    Each branch performs a multidimensional nonlinear computation before the
    soma-like integration step. Parameters are locally owned by this object.
    """

    output_dim = 1

    def __init__(
        self,
        input_dim: int,
        branches: Sequence[ProductBranch],
        bias: float = 0.0,
        name: str = "D",
    ) -> None:
        self.input_dim = int(input_dim)
        self.branches = list(branches)
        self.bias = float(bias)
        self.name = name
        self.enabled = True

    @classmethod
    def fit_truth_table(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        positive_only: bool = True,
        name: str = "D",
    ) -> "BooleanDendritron":
        """Allocate local branches from a Boolean truth table.

        For a positive-only Boolean function, one branch is allocated for each
        positive minterm. This is an exact constructive fit on {0,1}^d.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x must be [samples, features] and align with y")
        if not set(np.unique(x)).issubset({0.0, 1.0}):
            raise ValueError("This exact constructor expects binary inputs")

        branches: List[ProductBranch] = []
        for pattern, target in zip(x, y):
            if positive_only:
                if target > 0.5:
                    branches.append(ProductBranch(pattern.copy(), 1.0))
            else:
                branches.append(ProductBranch(pattern.copy(), float(target)))
        return cls(x.shape[1], branches, bias=0.0, name=name)

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
        if not self.enabled:
            return np.zeros(x.shape[0], dtype=np.float64)

        out = np.full(x.shape[0], self.bias, dtype=np.float64)
        for branch in self.branches:
            out += branch.value * branch.activation(x)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (self.raw(x) >= 0.5).astype(np.int8)

    def clone(self, name: str | None = None) -> "BooleanDendritron":
        clone = BooleanDendritron(
            self.input_dim,
            [ProductBranch(b.pattern.copy(), b.value) for b in self.branches],
            self.bias,
            name or self.name,
        )
        clone.enabled = self.enabled
        return clone

    def verify(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        return float(np.mean(self(x) == y))

    def signature(self) -> str:
        payload = {
            "input_dim": self.input_dim,
            "bias": self.bias,
            "enabled": self.enabled,
            "branches": [
                {"pattern": b.pattern.tolist(), "value": b.value}
                for b in self.branches
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([0, 1, 1, 0], dtype=np.int8)
XOR_TEMPLATE = BooleanDendritron.fit_truth_table(XOR_X, XOR_Y, name="XOR")


# -----------------------------------------------------------------------------
# 3. Dendritron Tissue: balanced recursive composition for parity.
# -----------------------------------------------------------------------------

class ParityTissue:
    """A balanced web of intact 2-input XOR Dendritrons.

    n input bits require n-1 Dendritrons. Existing nodes do not share mutable
    parameters. An unused node can be attached without changing behavior.
    """

    output_dim = 1

    def __init__(self, n_bits: int, xor_template: BooleanDendritron) -> None:
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self.input_dim = int(n_bits)
        self.layers: List[List[BooleanDendritron]] = []
        self.dormant: List[BooleanDendritron] = []

        width = n_bits
        depth = 0
        while width > 1:
            pairs = width // 2
            layer = [
                xor_template.clone(name=f"L{depth}N{j}") for j in range(pairs)
            ]
            self.layers.append(layer)
            width = pairs + (width % 2)
            depth += 1

    @property
    def units(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def active_branches(self) -> int:
        return sum(len(node.branches) for layer in self.layers for node in layer)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        if state.ndim == 1:
            state = state[None, :]
        if state.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input bits")

        for layer in self.layers:
            outputs = []
            for j, node in enumerate(layer):
                outputs.append(node(state[:, 2 * j : 2 * j + 2]))
            next_state = (
                np.stack(outputs, axis=1)
                if outputs
                else np.empty((state.shape[0], 0), dtype=np.int8)
            )
            if state.shape[1] % 2 == 1:
                next_state = np.concatenate(
                    [next_state, state[:, -1:].astype(np.int8)], axis=1
                )
            state = next_state
        return state[:, 0].astype(np.int8)

    def add_dormant(self, node: BooleanDendritron) -> None:
        """Attach capacity without connecting it to the active computation."""
        self.dormant.append(node)

    def signatures(self) -> List[List[str]]:
        return [[node.signature() for node in layer] for layer in self.layers]

    def verify_nodes(self) -> List[Tuple[int, int, float]]:
        return [
            (layer_i, node_i, node.verify(XOR_X, XOR_Y))
            for layer_i, layer in enumerate(self.layers)
            for node_i, node in enumerate(layer)
        ]

    def damage_node(self, layer_i: int, node_i: int) -> None:
        """Disable exactly one locally owned functional unit."""
        self.layers[layer_i][node_i].enabled = False

    def repair_node(self, layer_i: int, node_i: int) -> None:
        """Reconstruct only the damaged function from its local truth table."""
        old_name = self.layers[layer_i][node_i].name
        replacement = BooleanDendritron.fit_truth_table(
            XOR_X, XOR_Y, name=old_name
        )
        self.layers[layer_i][node_i] = replacement


class HigherOrderParityDendritron:
    """Recursive closure witness.

    Two already-functional parity regions are composed by one ordinary XOR
    Dendritron. Externally, the result again behaves like a single Dendritron:
    input vector -> output vector, with its own verification boundary.
    """

    output_dim = 1

    def __init__(
        self,
        left: ParityTissue,
        right: ParityTissue,
        integrator: BooleanDendritron,
    ) -> None:
        self.left = left
        self.right = right
        self.integrator = integrator.clone(name="higher_order_integrator")
        self.input_dim = left.input_dim + right.input_dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        left_y = self.left(x[:, : self.left.input_dim])
        right_y = self.right(x[:, self.left.input_dim :])
        return self.integrator(np.stack([left_y, right_y], axis=1))


# -----------------------------------------------------------------------------
# 4. Connectedness tissue: local recurrent propagation on a 2-D web.
# -----------------------------------------------------------------------------

class ReachabilityDendritron:
    """One local cell rule for connectedness.

    r_i(t+1) = active_i * [1 - product_j(1-r_j(t))]

    j includes the cell itself and its four von Neumann neighbors. The rule is
    exact on Boolean states and uses only local information.
    """

    input_dim = 6  # active + five reachability inputs
    output_dim = 1

    def __call__(
        self, active: np.ndarray, neighborhood_reached: np.ndarray
    ) -> np.ndarray:
        active = np.asarray(active, dtype=np.int8)
        reached = np.asarray(neighborhood_reached, dtype=np.int8)
        return (
            active * (1 - np.prod(1 - reached, axis=-1))
        ).astype(np.int8)


class ConnectednessTissue:
    """A web of locally interacting Reachability Dendritrons."""

    output_dim = 1

    def __init__(self, height: int, width: int) -> None:
        self.height = int(height)
        self.width = int(width)
        self.input_dim = self.height * self.width
        self.cell = ReachabilityDendritron()

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        grids = np.asarray(grids, dtype=np.int8)
        if grids.ndim == 2:
            grids = grids[None, :, :]
        if grids.shape[1:] != (self.height, self.width):
            raise ValueError("Grid shape mismatch")

        batch, height, width = grids.shape
        reached = np.zeros_like(grids)
        flat_active = grids.reshape(batch, -1)
        flat_reached = reached.reshape(batch, -1)

        # Deterministic seed: first active cell in row-major order.
        for b in range(batch):
            active_indices = np.flatnonzero(flat_active[b])
            if active_indices.size:
                flat_reached[b, active_indices[0]] = 1

        # A connected component in an HxW grid has graph diameter < H*W.
        for _ in range(height * width):
            up = np.zeros_like(reached)
            down = np.zeros_like(reached)
            left = np.zeros_like(reached)
            right = np.zeros_like(reached)
            up[:, 1:, :] = reached[:, :-1, :]
            down[:, :-1, :] = reached[:, 1:, :]
            left[:, :, 1:] = reached[:, :, :-1]
            right[:, :, :-1] = reached[:, :, 1:]

            neighborhood = np.stack(
                [reached, up, down, left, right], axis=-1
            )
            next_reached = self.cell(grids, neighborhood)
            if np.array_equal(next_reached, reached):
                break
            reached = next_reached

        return np.all(reached == grids, axis=(1, 2)).astype(np.int8)


def bfs_connected(grid: np.ndarray) -> int:
    """Reference implementation used only as a correctness oracle."""
    grid = np.asarray(grid, dtype=np.int8)
    points = np.argwhere(grid)
    if len(points) <= 1:
        return 1

    start = tuple(points[0])
    stack = [start]
    seen = {start}
    height, width = grid.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc]
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return int(len(seen) == len(points))


def random_connected_grid(
    rng: np.random.Generator, height: int, width: int, steps: int
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.int8)
    row, col = int(rng.integers(height)), int(rng.integers(width))
    grid[row, col] = 1
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(steps):
        dr, dc = moves[int(rng.integers(4))]
        row = int(np.clip(row + dr, 0, height - 1))
        col = int(np.clip(col + dc, 0, width - 1))
        grid[row, col] = 1
    return grid


def random_disconnected_grid(
    rng: np.random.Generator, height: int, width: int, steps: int
) -> np.ndarray:
    """Generate two components separated by a guaranteed blank column."""
    grid = np.zeros((height, width), dtype=np.int8)
    middle = width // 2
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))

    row = int(rng.integers(height))
    col = int(rng.integers(max(1, middle - 1)))
    for _ in range(steps):
        grid[row, col] = 1
        dr, dc = moves[int(rng.integers(4))]
        row = int(np.clip(row + dr, 0, height - 1))
        col = int(np.clip(col + dc, 0, max(0, middle - 2)))

    row = int(rng.integers(height))
    col = int(rng.integers(middle + 1, width))
    for _ in range(steps):
        grid[row, col] = 1
        dr, dc = moves[int(rng.integers(4))]
        row = int(np.clip(row + dr, 0, height - 1))
        col = int(np.clip(col + dc, middle + 1, width - 1))
    return grid


# -----------------------------------------------------------------------------
# 5. Benchmark runner.
# -----------------------------------------------------------------------------

def run_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)

    # A. Primitive expressivity: exact XOR.
    xor_accuracy = XOR_TEMPLATE.verify(XOR_X, XOR_Y)

    # B. Parity scaling: same primitive, recursively composed without retraining.
    parity_rows = []
    for n_bits in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        samples = 5_000
        x = rng.integers(0, 2, size=(samples, n_bits), dtype=np.int8)
        target = (x.sum(axis=1) % 2).astype(np.int8)
        tissue = ParityTissue(n_bits, XOR_TEMPLATE)
        start = time.perf_counter()
        prediction = tissue(x)
        runtime_ms = 1_000.0 * (time.perf_counter() - start)
        parity_rows.append(
            {
                "input_bits": n_bits,
                "samples": samples,
                "accuracy": float(np.mean(prediction == target)),
                "tissue_units": tissue.units,
                "tissue_branches": tissue.active_branches,
                "depth": tissue.depth,
                "direct_single_D_branches": str(2 ** (n_bits - 1)),
                "runtime_ms": runtime_ms,
            }
        )
    parity_df = pd.DataFrame(parity_rows)

    # C. Recursive closure: two complete 8-bit regions become one 16-bit region.
    left = ParityTissue(8, XOR_TEMPLATE)
    right = ParityTissue(8, XOR_TEMPLATE)
    higher_order = HigherOrderParityDendritron(left, right, XOR_TEMPLATE)
    closure_x = rng.integers(0, 2, size=(20_000, 16), dtype=np.int8)
    closure_target = (closure_x.sum(axis=1) % 2).astype(np.int8)
    closure_accuracy = float(np.mean(higher_order(closure_x) == closure_target))

    # D. Non-destructive growth invariant.
    growth_tissue = ParityTissue(64, XOR_TEMPLATE)
    growth_x = rng.integers(0, 2, size=(10_000, 64), dtype=np.int8)
    before_growth = growth_tissue(growth_x)
    growth_tissue.add_dormant(XOR_TEMPLATE.clone(name="dormant_capacity"))
    after_growth = growth_tissue(growth_x)
    growth_max_delta = int(np.max(np.abs(before_growth - after_growth)))

    # E. Local damage detection and repair.
    repair_tissue = ParityTissue(64, XOR_TEMPLATE)
    repair_x = rng.integers(0, 2, size=(20_000, 64), dtype=np.int8)
    repair_target = (repair_x.sum(axis=1) % 2).astype(np.int8)
    baseline_accuracy = float(np.mean(repair_tissue(repair_x) == repair_target))

    signatures_before = repair_tissue.signatures()
    repair_tissue.damage_node(0, 3)
    damaged_accuracy = float(np.mean(repair_tissue(repair_x) == repair_target))
    failed_nodes = [
        (layer_i, node_i, health)
        for layer_i, node_i, health in repair_tissue.verify_nodes()
        if health < 1.0
    ]
    signatures_damaged = repair_tissue.signatures()
    changed_during_damage = [
        (layer_i, node_i)
        for layer_i, layer in enumerate(signatures_before)
        for node_i, signature in enumerate(layer)
        if signature != signatures_damaged[layer_i][node_i]
    ]

    repair_tissue.repair_node(0, 3)
    repaired_accuracy = float(np.mean(repair_tissue(repair_x) == repair_target))
    signatures_repaired = repair_tissue.signatures()
    non_target_changes_after_repair = [
        (layer_i, node_i)
        for layer_i, layer in enumerate(signatures_before)
        for node_i, signature in enumerate(layer)
        if (layer_i, node_i) != (0, 3)
        and signature != signatures_repaired[layer_i][node_i]
    ]

    # F. Connectedness: local recurrent Dendritron web vs exact BFS oracle.
    connectedness_rows = []
    for side, examples in ((8, 1_000), (16, 500)):
        grids = []
        for _ in range(examples // 2):
            grids.append(
                random_connected_grid(
                    rng, side, side, steps=int(rng.integers(2, side * 4))
                )
            )
            grids.append(
                random_disconnected_grid(
                    rng, side, side, steps=int(rng.integers(2, side * 2))
                )
            )
        grid_batch = np.stack(grids)
        truth = np.array([bfs_connected(g) for g in grid_batch], dtype=np.int8)
        tissue = ConnectednessTissue(side, side)
        start = time.perf_counter()
        prediction = tissue(grid_batch)
        runtime_ms = 1_000.0 * (time.perf_counter() - start)
        connectedness_rows.append(
            {
                "grid": f"{side}x{side}",
                "examples": examples,
                "accuracy_vs_BFS": float(np.mean(prediction == truth)),
                "local_cells": side * side,
                "runtime_ms": runtime_ms,
            }
        )
    connectedness_df = pd.DataFrame(connectedness_rows)

    summary = {
        "perceptron_xor_feasible": False,
        "perceptron_certificate": xor_perceptron_contradiction(),
        "single_dendritron_xor_accuracy": xor_accuracy,
        "single_dendritron_xor_branches": len(XOR_TEMPLATE.branches),
        "recursive_closure_16bit_accuracy": closure_accuracy,
        "growth_max_output_delta": growth_max_delta,
        "repair_baseline_accuracy": baseline_accuracy,
        "repair_damaged_accuracy": damaged_accuracy,
        "failed_nodes": failed_nodes,
        "changed_nodes_during_damage": changed_during_damage,
        "repair_restored_accuracy": repaired_accuracy,
        "non_target_nodes_changed_after_repair": non_target_changes_after_repair,
    }
    return parity_df, connectedness_df, summary


def main() -> None:
    parity_df, connectedness_df, summary = run_benchmark()

    print("\nDENDRITRON–MINSKY BENCHMARK v0.1")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nPARITY SCALING")
    print(parity_df.to_string(index=False))

    print("\nCONNECTEDNESS")
    print(connectedness_df.to_string(index=False))

    parity_df.to_csv("dendritron_parity_results.csv", index=False)
    connectedness_df.to_csv("dendritron_connectedness_results.csv", index=False)
    with open("dendritron_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
