"""Functional memory packs and reliability-aware autonomous recall."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import InferenceTrace, LifecycleState, RecallMode


class PPCA:
    """Low-rank generative likelihood model over a shared coordinate."""

    def __init__(
        self,
        mean: np.ndarray,
        basis: np.ndarray,
        retained_variance: np.ndarray,
        residual_variance: float,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        self.retained_variance = np.asarray(retained_variance, dtype=np.float64)
        self.residual_variance = float(residual_variance)
        if self.residual_variance <= 0 or np.any(self.retained_variance <= 0):
            raise ValueError("PPCA variances must be positive")
        self.dimension = len(self.mean)
        self.rank = self.basis.shape[1]
        self.log_determinant = float(
            np.log(self.retained_variance).sum()
            + (self.dimension - self.rank) * math.log(self.residual_variance)
        )

    @classmethod
    def fit(cls, values: np.ndarray, rank: int = 6, floor: float = 1e-5) -> PPCA:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("PPCA needs at least two coordinate rows")
        if not np.all(np.isfinite(values)):
            raise ValueError("PPCA values contain NaN or infinity")
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        retained_rank = min(max(int(rank), 0), max(0, covariance.shape[0] - 1))
        tail = eigenvalues[retained_rank:]
        residual = max(float(tail.mean()) if len(tail) else floor, floor)
        retained = np.maximum(eigenvalues[:retained_rank], residual)
        return cls(mean, eigenvectors[:, :retained_rank], retained, residual)

    def log_likelihood(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != self.dimension:
            raise ValueError(f"expected coordinate dimension {self.dimension}")
        difference = values - self.mean
        total_energy = np.sum(difference * difference, axis=1)
        if self.rank:
            projection = difference @ self.basis
            projected_energy = projection * projection
            retained_energy = projected_energy.sum(axis=1)
            mahalanobis = (
                np.sum(projected_energy / self.retained_variance, axis=1)
                + (total_energy - retained_energy) / self.residual_variance
            )
        else:
            mahalanobis = total_energy / self.residual_variance
        return -0.5 * (
            mahalanobis + self.log_determinant + self.dimension * math.log(2.0 * math.pi)
        )


class BinaryPPCAResolver:
    """Generative verifier for a binary memory's two output classes."""

    def __init__(self, label_models: tuple[PPCA, PPCA]) -> None:
        self.label_models = label_models

    @classmethod
    def fit(
        cls, coordinates: np.ndarray, labels: np.ndarray, rank: int = 6, floor: float = 1e-5
    ) -> BinaryPPCAResolver:
        coordinates = np.asarray(coordinates)
        labels = np.asarray(labels).reshape(-1)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("binary verifier requires both labels 0 and 1")
        return cls(
            (
                PPCA.fit(coordinates[labels == 0], rank, floor),
                PPCA.fit(coordinates[labels == 1], rank, floor),
            )
        )

    def score(self, coordinates: np.ndarray) -> np.ndarray:
        likelihood = np.stack(
            [model.log_likelihood(coordinates) for model in self.label_models], axis=1
        )
        maximum = likelihood.max(axis=1, keepdims=True)
        return maximum[:, 0] + np.log(np.exp(likelihood - maximum).sum(axis=1)) - math.log(2)


@dataclass
class MemoryPack:
    """A durable functional module plus address, verifier, and manifest."""

    name: str
    function: Callable[[Any], Any]
    address_model: PPCA
    verifier: Callable[[np.ndarray], np.ndarray] | BinaryPPCAResolver | None = None
    validation_accuracy: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    state: LifecycleState = LifecycleState.CANDIDATE

    def verifier_score(self, coordinate: np.ndarray) -> float:
        if self.verifier is None:
            return float(self.address_model.log_likelihood(coordinate)[0])
        if hasattr(self.verifier, "score"):
            return float(np.asarray(self.verifier.score(coordinate)).reshape(-1)[0])
        return float(np.asarray(self.verifier(coordinate)).reshape(-1)[0])

    def signature(self) -> str:
        payload = {
            "name": self.name,
            "validation_accuracy": self.validation_accuracy,
            "metadata": self.metadata,
            "address_mean": self.address_model.mean.tolist(),
            "address_basis": self.address_model.basis.tolist(),
            "state": self.state.value,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class MemoryRegistry:
    """Input cue → address → candidates → verification → selected memory."""

    def __init__(
        self,
        *,
        registration_threshold: float = 0.8,
        efficient_threshold: float = 0.85,
        reliable_k: int = 2,
        critical_mass: float = 0.95,
        abstain_threshold: float = 0.0,
    ) -> None:
        self.registration_threshold = registration_threshold
        self.efficient_threshold = efficient_threshold
        self.reliable_k = reliable_k
        self.critical_mass = critical_mass
        self.abstain_threshold = abstain_threshold
        self.packs: dict[str, MemoryPack] = {}

    def register(self, pack: MemoryPack) -> str:
        if pack.name in self.packs:
            raise ValueError(f"memory already registered: {pack.name}")
        if pack.validation_accuracy < self.registration_threshold:
            pack.state = LifecycleState.QUARANTINED
        else:
            pack.state = LifecycleState.REGISTERED
        self.packs[pack.name] = pack
        return pack.state.value

    def uninstall(self, name: str) -> MemoryPack:
        pack = self.packs.pop(name)
        pack.state = LifecycleState.ARCHIVED
        return pack

    def reinstall(self, pack: MemoryPack) -> None:
        if pack.name in self.packs:
            raise ValueError(f"memory already registered: {pack.name}")
        if pack.validation_accuracy < self.registration_threshold:
            raise ValueError("archived pack no longer satisfies the registration gate")
        pack.state = LifecycleState.REGISTERED
        self.packs[pack.name] = pack

    def _registered(self) -> list[MemoryPack]:
        packs = [pack for pack in self.packs.values() if pack.state is LifecycleState.REGISTERED]
        if not packs:
            raise RuntimeError("the registry contains no registered memories")
        return packs

    def address_probabilities(self, coordinate: np.ndarray) -> tuple[list[MemoryPack], np.ndarray]:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        if coordinate.ndim == 1:
            coordinate = coordinate[None, :]
        if len(coordinate) != 1:
            raise ValueError("recall currently accepts one coordinate at a time")
        packs = self._registered()
        scores = np.array([pack.address_model.log_likelihood(coordinate)[0] for pack in packs])
        shifted = scores - scores.max()
        probability = np.exp(shifted)
        probability /= probability.sum()
        return packs, probability

    def candidates(
        self, coordinate: np.ndarray, mode: RecallMode | str
    ) -> tuple[list[MemoryPack], np.ndarray]:
        mode = RecallMode(mode)
        packs, probability = self.address_probabilities(coordinate)
        ranking = np.argsort(-probability)
        if mode is RecallMode.FAST:
            selected = ranking[:1]
        elif mode is RecallMode.EFFICIENT:
            count = 1 if probability[ranking[0]] >= self.efficient_threshold else min(2, len(packs))
            selected = ranking[:count]
        elif mode is RecallMode.RELIABLE:
            selected = ranking[: min(self.reliable_k, len(packs))]
        else:
            cumulative = 0.0
            selected_values: list[int] = []
            for index in ranking:
                selected_values.append(int(index))
                cumulative += float(probability[index])
                if cumulative >= self.critical_mass:
                    break
            selected = np.asarray(selected_values)
        return [packs[int(index)] for index in selected], probability

    def recall(
        self, coordinate: np.ndarray, mode: RecallMode | str = RecallMode.RELIABLE
    ) -> tuple[MemoryPack | None, InferenceTrace]:
        mode = RecallMode(mode)
        packs, all_probability = self.address_probabilities(coordinate)
        candidates, _ = self.candidates(coordinate, mode)
        verifier_scores = {
            pack.name: pack.verifier_score(np.atleast_2d(coordinate)) for pack in candidates
        }
        selected = max(candidates, key=lambda pack: verifier_scores[pack.name])
        selected_score = verifier_scores[selected.name]
        abstained = selected_score < self.abstain_threshold
        trace = InferenceTrace(
            candidates=tuple(pack.name for pack in candidates),
            selected=None if abstained else selected.name,
            mode=mode,
            address_probabilities={
                pack.name: float(value) for pack, value in zip(packs, all_probability, strict=True)
            },
            verifier_scores=verifier_scores,
            abstained=abstained,
        )
        return (None if abstained else selected), trace

    def execute(
        self,
        cue: Any,
        coordinate: np.ndarray,
        mode: RecallMode | str = RecallMode.RELIABLE,
    ) -> tuple[Any, InferenceTrace]:
        selected, trace = self.recall(coordinate, mode)
        if selected is None:
            return None, trace
        return selected.function(cue), trace


def make_memory_pack(
    name: str,
    function: Callable[[Any], Any],
    address_coordinates: np.ndarray,
    *,
    verifier: Callable[[np.ndarray], np.ndarray] | BinaryPPCAResolver | None = None,
    validation_accuracy: float = 1.0,
    rank: int = 6,
    metadata: dict[str, Any] | None = None,
) -> MemoryPack:
    return MemoryPack(
        name=name,
        function=function,
        address_model=PPCA.fit(address_coordinates, rank=rank),
        verifier=verifier,
        validation_accuracy=validation_accuracy,
        metadata={} if metadata is None else metadata,
    )
