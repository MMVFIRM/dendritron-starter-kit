"""
Dendritron–Minsky Benchmark v0.2

Purpose
-------
Move from hand-constructed witnesses to learned/compiled local functions and
compare functional ownership against a shared MLP under sequential learning.

This benchmark tests:
  1. exhaustive local Boolean universality on all 65,536 four-input functions;
  2. learning XOR once, then recursively composing it to 4,096-bit parity;
  3. learning one local reachability rule, then deploying it on arbitrary grids;
  4. sequential acquisition with exact ownership and zero silent interference;
  5. shared-MLP catastrophic interference on the identical task sequence;
  6. local damage detection and certificate-based repair.

The benchmark is a constructive systems result. It is not yet a claim that this
Dendritron implementation dominates optimized MLPs on continuous real-world
benchmarks or at matched hardware efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the MLP control benchmark") from exc

SEED = 7
OUTPUT_DIR = Path(__file__).resolve().parent

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


def boolean_cube(d: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=d)), dtype=np.int8)


def pattern_indices(x: np.ndarray) -> np.ndarray:
    """Map binary rows to lexicographic truth-table indices."""
    x = np.asarray(x, dtype=np.int8)
    powers = (2 ** np.arange(x.shape[1] - 1, -1, -1)).astype(np.int64)
    return x.astype(np.int64) @ powers


@dataclass
class MintermBranch:
    """One locally owned Boolean minterm branch."""

    pattern: np.ndarray
    value: float = 1.0

    def activation(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        literals = np.where(self.pattern[None, :] == 1, x, 1.0 - x)
        return np.prod(literals, axis=1)


class CompiledBooleanDendritron:
    """A locally compiled multicompartment Boolean Dendritron.

    On the Boolean cube, each minterm branch is a Kronecker indicator. A local
    truth-table certificate can therefore be compiled into an exact function.
    The optimized inference path uses the equivalent lookup table, while the
    explicit branch path remains available and is checked for equivalence.
    """

    output_dim = 1

    def __init__(
        self,
        input_dim: int,
        branches: Sequence[MintermBranch],
        truth_table: np.ndarray,
        *,
        name: str,
        certificate_x: np.ndarray,
        certificate_y: np.ndarray,
    ) -> None:
        self.input_dim = int(input_dim)
        self.branches = list(branches)
        self.truth_table = np.asarray(truth_table, dtype=np.int8).copy()
        self.name = name
        self.certificate_x = np.asarray(certificate_x, dtype=np.int8).copy()
        self.certificate_y = np.asarray(certificate_y, dtype=np.int8).copy()
        self.enabled = True

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        name: str = "D",
    ) -> "CompiledBooleanDendritron":
        x = np.asarray(x, dtype=np.int8)
        y = np.asarray(y, dtype=np.int8).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be [samples, features] and align with y")
        if not set(np.unique(x)).issubset({0, 1}):
            raise ValueError("Compiled Boolean Dendritron expects binary inputs")

        d = x.shape[1]
        expected = boolean_cube(d)
        if len(x) != 2**d or not np.array_equal(x, expected):
            raise ValueError(
                "Exact local compilation currently requires the complete "
                "lexicographically ordered Boolean truth table"
            )

        branches = [
            MintermBranch(pattern=row.copy(), value=1.0)
            for row, target in zip(x, y)
            if target == 1
        ]
        return cls(
            d,
            branches,
            truth_table=y,
            name=name,
            certificate_x=x,
            certificate_y=y,
        )

    def branch_raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if not self.enabled:
            return np.zeros(len(x), dtype=np.float64)
        out = np.zeros(len(x), dtype=np.float64)
        for branch in self.branches:
            out += branch.value * branch.activation(x)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int8)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} inputs")
        if not self.enabled:
            return np.zeros(len(x), dtype=np.int8)
        return self.truth_table[pattern_indices(x)]

    def verify(self) -> float:
        return float(np.mean(self(self.certificate_x) == self.certificate_y))

    def verify_branch_lookup_equivalence(self) -> bool:
        branch_prediction = (self.branch_raw(self.certificate_x) >= 0.5).astype(np.int8)
        return bool(np.array_equal(branch_prediction, self(self.certificate_x)))

    @property
    def stored_scalar_equivalents(self) -> int:
        # Conservative count: one scalar per pattern bit and one branch value,
        # plus one local bias/threshold equivalent.
        return len(self.branches) * (self.input_dim + 1) + 1

    def signature(self) -> str:
        payload = {
            "name": self.name,
            "input_dim": self.input_dim,
            "enabled": self.enabled,
            "truth_table": self.truth_table.tolist(),
            "branches": [
                {"pattern": b.pattern.tolist(), "value": b.value}
                for b in self.branches
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def clone(self, name: str | None = None) -> "CompiledBooleanDendritron":
        clone = CompiledBooleanDendritron(
            self.input_dim,
            [MintermBranch(b.pattern.copy(), b.value) for b in self.branches],
            self.truth_table.copy(),
            name=name or self.name,
            certificate_x=self.certificate_x.copy(),
            certificate_y=self.certificate_y.copy(),
        )
        clone.enabled = self.enabled
        return clone

    def damage_branch(self, branch_index: int) -> None:
        """Remove one locally owned branch and recompile the local lookup."""
        if not 0 <= branch_index < len(self.branches):
            raise IndexError("branch_index out of range")
        damaged_pattern = self.branches[branch_index].pattern.copy()
        del self.branches[branch_index]
        self.truth_table[pattern_indices(damaged_pattern[None, :])[0]] = 0

    def repair_from_certificate(self) -> None:
        repaired = CompiledBooleanDendritron.fit(
            self.certificate_x,
            self.certificate_y,
            name=self.name,
        )
        self.branches = repaired.branches
        self.truth_table = repaired.truth_table
        self.enabled = True


class LearnedParityTissue:
    """Balanced recursive composition of one learned XOR Dendritron."""

    def __init__(self, n_bits: int, xor_region: CompiledBooleanDendritron) -> None:
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self.input_dim = n_bits
        self.layers: List[List[CompiledBooleanDendritron]] = []
        width = n_bits
        depth = 0
        while width > 1:
            pairs = width // 2
            self.layers.append(
                [xor_region.clone(name=f"L{depth}N{i}") for i in range(pairs)]
            )
            width = pairs + (width % 2)
            depth += 1

    @property
    def units(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def branches(self) -> int:
        return sum(len(node.branches) for layer in self.layers for node in layer)

    @property
    def depth(self) -> int:
        return len(self.layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.int8)
        if state.ndim == 1:
            state = state[None, :]
        for layer in self.layers:
            outputs = [node(state[:, 2 * i : 2 * i + 2]) for i, node in enumerate(layer)]
            next_state = (
                np.stack(outputs, axis=1)
                if outputs
                else np.empty((len(state), 0), dtype=np.int8)
            )
            if state.shape[1] % 2:
                next_state = np.concatenate([next_state, state[:, -1:]], axis=1)
            state = next_state
        return state[:, 0]


class LearnedConnectednessTissue:
    """Grid tissue using one learned six-input local reachability function."""

    def __init__(
        self,
        height: int,
        width: int,
        cell: CompiledBooleanDendritron,
    ) -> None:
        if cell.input_dim != 6:
            raise ValueError("Reachability cell must accept active + 5 reached bits")
        self.height = height
        self.width = width
        self.cell = cell

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        grids = np.asarray(grids, dtype=np.int8)
        if grids.ndim == 2:
            grids = grids[None, :, :]
        batch, height, width = grids.shape
        reached = np.zeros_like(grids)
        flat_active = grids.reshape(batch, -1)
        flat_reached = reached.reshape(batch, -1)
        for b in range(batch):
            active_indices = np.flatnonzero(flat_active[b])
            if active_indices.size:
                flat_reached[b, active_indices[0]] = 1

        for _ in range(height * width):
            up = np.zeros_like(reached)
            down = np.zeros_like(reached)
            left = np.zeros_like(reached)
            right = np.zeros_like(reached)
            up[:, 1:, :] = reached[:, :-1, :]
            down[:, :-1, :] = reached[:, 1:, :]
            left[:, :, 1:] = reached[:, :, :-1]
            right[:, :, :-1] = reached[:, :, 1:]
            local_inputs = np.stack(
                [grids, reached, up, down, left, right], axis=-1
            ).reshape(-1, 6)
            next_reached = self.cell(local_inputs).reshape(batch, height, width)
            if np.array_equal(next_reached, reached):
                break
            reached = next_reached
        return np.all(reached == grids, axis=(1, 2)).astype(np.int8)


def bfs_connected(grid: np.ndarray) -> int:
    points = np.argwhere(grid)
    if len(points) <= 1:
        return 1
    height, width = grid.shape
    start = tuple(points[0])
    stack = [start]
    seen = {start}
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


class OwnedDendritronWeb:
    """Explicitly owned function regions with an inspectable task router."""

    def __init__(self) -> None:
        self.regions: Dict[int, CompiledBooleanDendritron] = {}

    def learn_task(self, task_id: int, x: np.ndarray, y: np.ndarray) -> int:
        if task_id in self.regions:
            raise ValueError("Task already exists; use an explicit version/fork operation")
        before = {key: region.signature() for key, region in self.regions.items()}
        self.regions[task_id] = CompiledBooleanDendritron.fit(
            x, y, name=f"task_{task_id}"
        )
        after = {key: region.signature() for key, region in self.regions.items()}
        changed_old_regions = sum(before[key] != after[key] for key in before)
        return changed_old_regions

    def predict(self, task_id: int, x: np.ndarray) -> np.ndarray:
        return self.regions[task_id](x)

    def total_scalar_equivalents(self) -> int:
        # Include one routing-table entry per region.
        return sum(r.stored_scalar_equivalents for r in self.regions.values()) + len(self.regions)

    def verify(self) -> Dict[int, float]:
        return {task_id: region.verify() for task_id, region in self.regions.items()}


class SharedContextMLP(nn.Module):
    def __init__(self, input_dim: int, task_count: int, hidden: int = 32) -> None:
        super().__init__()
        self.task_count = task_count
        self.net = nn.Sequential(
            nn.Linear(input_dim + task_count, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        context = torch.zeros((len(x), self.task_count), dtype=x.dtype)
        context[:, task_id] = 1.0
        return self.net(torch.cat([x, context], dim=1))


def binary_accuracy(logits: torch.Tensor, y: np.ndarray) -> float:
    prediction = (logits.detach().cpu().numpy().reshape(-1) > 0).astype(np.int8)
    return float(np.mean(prediction == y))


def train_shared_mlp_sequential(
    x: np.ndarray,
    task_labels: np.ndarray,
    *,
    hidden: int = 32,
    max_epochs: int = 3_000,
    learning_rate: float = 0.01,
) -> Tuple[pd.DataFrame, SharedContextMLP]:
    torch.manual_seed(SEED)
    model = SharedContextMLP(x.shape[1], len(task_labels), hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tensor = torch.tensor(x, dtype=torch.float32)

    rows = []
    for learned_task in range(len(task_labels)):
        y = task_labels[learned_task]
        y_tensor = torch.tensor(y[:, None], dtype=torch.float32)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            logits = model(x_tensor, learned_task)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0 and binary_accuracy(model(x_tensor, learned_task), y) == 1.0:
                break

        for evaluated_task in range(len(task_labels)):
            accuracy = binary_accuracy(model(x_tensor, evaluated_task), task_labels[evaluated_task])
            rows.append(
                {
                    "model": "shared_mlp",
                    "after_learning_task": learned_task,
                    "evaluated_task": evaluated_task,
                    "accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows), model


def train_independent_mlp_bank(
    x: np.ndarray,
    task_labels: np.ndarray,
    *,
    hidden: int = 16,
    max_epochs: int = 5_000,
    learning_rate: float = 0.03,
) -> Tuple[pd.DataFrame, int]:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    accuracies = []
    total_parameters = 0
    for task_id, y in enumerate(task_labels):
        torch.manual_seed(100 + task_id)
        model = nn.Sequential(
            nn.Linear(x.shape[1], hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        total_parameters += sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        y_tensor = torch.tensor(y[:, None], dtype=torch.float32)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0 and binary_accuracy(model(x_tensor), y) == 1.0:
                break
        accuracies.append(binary_accuracy(model(x_tensor), y))
    return pd.DataFrame(
        {
            "task": np.arange(len(task_labels)),
            "accuracy": accuracies,
            "model": "independent_mlp_bank",
        }
    ), total_parameters


def exhaustive_boolean_universality() -> dict:
    """Exhaustively check every four-input Boolean function.

    The 16 minterm branches form the identity basis on the 16 Boolean inputs.
    Every one of the 2^16 possible label vectors is therefore reconstructed.
    """
    x = boolean_cube(4)
    branch_matrix = np.zeros((len(x), len(x)), dtype=np.int8)
    for column, pattern in enumerate(x):
        branch = MintermBranch(pattern)
        branch_matrix[:, column] = (branch.activation(x) >= 0.5).astype(np.int8)

    function_ids = np.arange(2**16, dtype=np.uint32)
    labels = ((function_ids[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.int8)
    reconstructed = labels @ branch_matrix.T
    exact = bool(np.array_equal(reconstructed, labels))
    return {
        "input_dim": 4,
        "boolean_functions_tested": int(2**16),
        "branch_basis_is_identity": bool(np.array_equal(branch_matrix, np.eye(16, dtype=np.int8))),
        "all_functions_reconstructed_exactly": exact,
    }


def make_balanced_random_tasks(
    rng: np.random.Generator,
    task_count: int,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = boolean_cube(input_dim)
    labels = []
    half = len(x) // 2
    for _ in range(task_count):
        y = np.array([0] * half + [1] * half, dtype=np.int8)
        rng.shuffle(y)
        labels.append(y)
    return x, np.stack(labels)


def accuracy_matrix_summary(matrix_df: pd.DataFrame, task_count: int) -> dict:
    matrix = matrix_df.pivot(
        index="after_learning_task", columns="evaluated_task", values="accuracy"
    ).to_numpy()
    acquisition = np.array([matrix[t, t] for t in range(task_count)])
    final = matrix[-1]
    forgetting = np.array(
        [np.max(matrix[t:, t]) - final[t] for t in range(task_count - 1)]
    )
    return {
        "mean_acquisition_accuracy": float(np.mean(acquisition)),
        "min_acquisition_accuracy": float(np.min(acquisition)),
        "mean_final_accuracy": float(np.mean(final)),
        "min_final_accuracy": float(np.min(final)),
        "mean_forgetting": float(np.mean(forgetting)),
        "max_forgetting": float(np.max(forgetting)),
    }


def run_benchmark() -> Tuple[dict, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(SEED)

    # A. Exhaustive local function class for d=4.
    universality = exhaustive_boolean_universality()

    # B. Learn XOR locally once, then scale through recursive composition.
    xor_x = boolean_cube(2)
    xor_y = (xor_x.sum(axis=1) % 2).astype(np.int8)
    learned_xor = CompiledBooleanDendritron.fit(xor_x, xor_y, name="learned_XOR")
    xor_accuracy = learned_xor.verify()
    xor_branch_equivalence = learned_xor.verify_branch_lookup_equivalence()

    parity_rows = []
    for n_bits, samples in (
        (2, 5_000),
        (4, 5_000),
        (8, 5_000),
        (16, 5_000),
        (32, 5_000),
        (64, 5_000),
        (128, 5_000),
        (256, 5_000),
        (512, 3_000),
        (1_024, 2_000),
        (2_048, 1_000),
        (4_096, 500),
    ):
        x = rng.integers(0, 2, size=(samples, n_bits), dtype=np.int8)
        target = (x.sum(axis=1) % 2).astype(np.int8)
        tissue = LearnedParityTissue(n_bits, learned_xor)
        start = time.perf_counter()
        prediction = tissue(x)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        parity_rows.append(
            {
                "input_bits": n_bits,
                "samples": samples,
                "accuracy": float(np.mean(prediction == target)),
                "dendritrons": tissue.units,
                "branches": tissue.branches,
                "depth": tissue.depth,
                "single_D_direct_branch_log10": (n_bits - 1) * math.log10(2),
                "runtime_ms": elapsed_ms,
            }
        )
    parity_df = pd.DataFrame(parity_rows)

    # C. Learn one local reachability rule and deploy it at multiple grid sizes.
    reach_x = boolean_cube(6)
    reach_y = (
        (reach_x[:, 0] == 1) & (np.sum(reach_x[:, 1:], axis=1) >= 1)
    ).astype(np.int8)
    learned_reach = CompiledBooleanDendritron.fit(
        reach_x, reach_y, name="learned_reachability"
    )
    connectedness_rows = []
    for side, examples in ((8, 1_000), (16, 500), (32, 200)):
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
        batch = np.stack(grids)
        truth = np.array([bfs_connected(g) for g in batch], dtype=np.int8)
        tissue = LearnedConnectednessTissue(side, side, learned_reach)
        start = time.perf_counter()
        prediction = tissue(batch)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        connectedness_rows.append(
            {
                "grid": f"{side}x{side}",
                "examples": examples,
                "accuracy_vs_BFS": float(np.mean(prediction == truth)),
                "local_dendritrons": side * side,
                "learned_rule_branches": len(learned_reach.branches),
                "runtime_ms": elapsed_ms,
            }
        )
    connectedness_df = pd.DataFrame(connectedness_rows)

    # D. Sequential functions: owned Dendritron regions vs shared MLP.
    task_count = 12
    task_x, task_labels = make_balanced_random_tasks(rng, task_count, input_dim=4)

    web = OwnedDendritronWeb()
    web_rows = []
    old_region_changes = 0
    signatures_after_each_task: List[Dict[int, str]] = []
    for learned_task in range(task_count):
        old_region_changes += web.learn_task(
            learned_task, task_x, task_labels[learned_task]
        )
        signatures_after_each_task.append(
            {key: region.signature() for key, region in web.regions.items()}
        )
        for evaluated_task in range(task_count):
            accuracy = (
                float(np.mean(web.predict(evaluated_task, task_x) == task_labels[evaluated_task]))
                if evaluated_task in web.regions
                else np.nan
            )
            web_rows.append(
                {
                    "model": "owned_dendritron_web",
                    "after_learning_task": learned_task,
                    "evaluated_task": evaluated_task,
                    "accuracy": accuracy,
                }
            )
    web_df = pd.DataFrame(web_rows)

    shared_mlp_df, shared_mlp = train_shared_mlp_sequential(task_x, task_labels)
    independent_mlp_df, independent_mlp_parameters = train_independent_mlp_bank(
        task_x, task_labels
    )

    web_complete = web_df.dropna()
    web_matrix = web_complete.pivot(
        index="after_learning_task", columns="evaluated_task", values="accuracy"
    )
    web_final_accuracy = float(
        np.mean(
            [
                np.mean(web.predict(task, task_x) == task_labels[task])
                for task in range(task_count)
            ]
        )
    )
    shared_summary = accuracy_matrix_summary(shared_mlp_df, task_count)

    shared_mlp_parameters = sum(p.numel() for p in shared_mlp.parameters())

    # E. Local damage and certificate-based repair.
    target_task = 5
    signatures_before_damage = {
        key: region.signature() for key, region in web.regions.items()
    }
    task_accuracy_before_damage = web.regions[target_task].verify()
    web.regions[target_task].damage_branch(0)
    task_accuracy_after_damage = web.regions[target_task].verify()
    damaged_health = web.verify()
    failed_regions = [key for key, health in damaged_health.items() if health < 1.0]
    signatures_after_damage = {
        key: region.signature() for key, region in web.regions.items()
    }
    changed_during_damage = [
        key
        for key in signatures_before_damage
        if signatures_before_damage[key] != signatures_after_damage[key]
    ]
    web.regions[target_task].repair_from_certificate()
    task_accuracy_after_repair = web.regions[target_task].verify()
    signatures_after_repair = {
        key: region.signature() for key, region in web.regions.items()
    }
    non_target_changes_after_repair = [
        key
        for key in signatures_before_damage
        if key != target_task
        and signatures_before_damage[key] != signatures_after_repair[key]
    ]

    summary = {
        "benchmark": "Dendritron–Minsky Benchmark v0.2",
        "seed": SEED,
        "universality": universality,
        "learned_xor_accuracy": xor_accuracy,
        "xor_explicit_branches_equal_compiled_lookup": xor_branch_equivalence,
        "parity_max_bits": int(parity_df.input_bits.max()),
        "parity_min_accuracy": float(parity_df.accuracy.min()),
        "reachability_rule_accuracy": learned_reach.verify(),
        "connectedness_min_accuracy": float(connectedness_df.accuracy_vs_BFS.min()),
        "sequential_task_count": task_count,
        "owned_web_mean_final_accuracy": web_final_accuracy,
        "owned_web_old_regions_changed_during_growth": old_region_changes,
        "owned_web_scalar_equivalents": web.total_scalar_equivalents(),
        "shared_mlp_parameters": int(shared_mlp_parameters),
        "shared_mlp": shared_summary,
        "independent_mlp_bank_mean_accuracy": float(independent_mlp_df.accuracy.mean()),
        "independent_mlp_bank_parameters": int(independent_mlp_parameters),
        "damage_target_task": target_task,
        "damage_accuracy_before": task_accuracy_before_damage,
        "damage_accuracy_after": task_accuracy_after_damage,
        "failed_regions_detected": failed_regions,
        "regions_changed_during_damage": changed_during_damage,
        "repair_accuracy": task_accuracy_after_repair,
        "non_target_regions_changed_after_repair": non_target_changes_after_repair,
        "known_limitations": [
            "The exact compiler currently requires a complete local Boolean truth table.",
            "Worst-case branch count is exponential in one Dendritron's local input dimension.",
            "The sequential benchmark uses an explicit task router; learned routing is not yet tested.",
            "The MLP control intentionally uses no replay, regularization, or task-specific frozen modules.",
            "Continuous real-world learning and matched-hardware energy efficiency remain untested.",
        ],
    }

    frames = {
        "parity": parity_df,
        "connectedness": connectedness_df,
        "owned_web_accuracy": web_df,
        "shared_mlp_accuracy": shared_mlp_df,
        "independent_mlp": independent_mlp_df,
    }
    return summary, frames


def save_results(summary: dict, frames: Dict[str, pd.DataFrame]) -> None:
    with open(OUTPUT_DIR / "dendritron_v0_2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / f"dendritron_v0_2_{name}.csv", index=False)


def main() -> None:
    summary, frames = run_benchmark()
    save_results(summary, frames)

    print("\nDENDRITRON–MINSKY BENCHMARK v0.2")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print("\nPARITY")
    print(frames["parity"].to_string(index=False))
    print("\nCONNECTEDNESS")
    print(frames["connectedness"].to_string(index=False))


if __name__ == "__main__":
    main()
