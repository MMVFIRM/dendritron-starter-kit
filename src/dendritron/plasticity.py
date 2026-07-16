"""Replay-free structural plasticity with bounded ownership certificates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Chart, Geometry, chart_distance


@dataclass
class PlasticBranch:
    branch_id: int
    owner: int
    center: np.ndarray
    sigma: float
    chart: Chart
    count: int = 1
    active: bool = True
    damaged: bool = False
    buffer: list[np.ndarray] = field(default_factory=list)

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if not self.active or self.damaged:
            return np.zeros(len(values))
        distance = chart_distance(values, self.center, self.chart)
        return np.exp(-0.5 * (distance / max(self.sigma, 1e-8)) ** 2)


class PlasticDendritronWeb:
    """Small, dependency-light realization of local structural plasticity."""

    def __init__(
        self,
        input_dim: int,
        *,
        sigma: float = 1.0,
        novelty_threshold: float = 2.25,
        certificate_size: int = 64,
        quarantine_margin: float = 0.02,
        charts: dict[int, Chart] | None = None,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.sigma = sigma
        self.novelty_threshold = novelty_threshold
        self.certificate_size = certificate_size
        self.quarantine_margin = quarantine_margin
        self.charts = charts or {}
        self.rng = np.random.default_rng(seed)
        self.regions: dict[int, list[PlasticBranch]] = {}
        self.certificates: dict[int, list[np.ndarray]] = {}
        self.events: list[dict[str, object]] = []
        self.branch_counter = 0

    def _chart(self, owner: int) -> Chart:
        return self.charts.get(owner, Chart(f"owner-{owner}-euclidean", Geometry.EUCLIDEAN))

    def scores(self, values: np.ndarray, owner: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        branches = [
            branch for branch in self.regions.get(owner, []) if branch.active and not branch.damaged
        ]
        if not branches:
            return np.zeros(len(values))
        branch_scores = np.stack([branch.score(values) for branch in branches], axis=1)
        return 1.0 - np.prod(1.0 - np.clip(branch_scores, 0.0, 1.0 - 1e-9), axis=1)

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        owners = sorted(self.regions)
        if not owners:
            return np.full(len(values), -1, dtype=np.int64)
        scores = np.stack([self.scores(values, owner) for owner in owners], axis=1)
        return np.asarray(owners)[scores.argmax(axis=1)]

    def _remember(self, value: np.ndarray, owner: int) -> None:
        certificate = self.certificates.setdefault(owner, [])
        if len(certificate) < self.certificate_size:
            certificate.append(value.copy())
        elif self.rng.random() < 0.05:
            certificate[int(self.rng.integers(len(certificate)))] = value.copy()

    def _safe(self, candidate: PlasticBranch) -> bool:
        for owner, points in self.certificates.items():
            if owner == candidate.owner or not points:
                continue
            values = np.stack(points)
            if np.any(
                candidate.score(values) > self.scores(values, owner) + self.quarantine_margin
            ):
                return False
        return True

    def _grow(self, value: np.ndarray, owner: int, reason: str) -> PlasticBranch | None:
        sigma = self.sigma
        for _ in range(8):
            self.branch_counter += 1
            candidate = PlasticBranch(
                branch_id=self.branch_counter,
                owner=owner,
                center=value.copy(),
                sigma=sigma,
                chart=self._chart(owner),
            )
            if self._safe(candidate):
                self.regions.setdefault(owner, []).append(candidate)
                self.events.append(
                    {
                        "event": "grow",
                        "owner": owner,
                        "branch": candidate.branch_id,
                        "reason": reason,
                    }
                )
                return candidate
            self.events.append(
                {
                    "event": "quarantine",
                    "owner": owner,
                    "branch": candidate.branch_id,
                    "reason": reason,
                }
            )
            sigma *= 0.72
        return None

    def learn_one(self, value: np.ndarray, owner: int) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.input_dim,):
            raise ValueError(f"expected a {self.input_dim}-dimensional value")
        self._remember(value, owner)
        active = [
            branch for branch in self.regions.get(owner, []) if branch.active and not branch.damaged
        ]
        if not active:
            self._grow(value, owner, "new_or_empty_region")
            return
        distances = np.array(
            [
                chart_distance(value[None, :], branch.center, branch.chart)[0]
                / max(branch.sigma, 1e-8)
                for branch in active
            ]
        )
        winner = active[int(distances.argmin())]
        if float(distances.min()) > self.novelty_threshold:
            candidate = self._grow(value, owner, "novelty")
            if candidate is not None:
                winner = candidate
        winner.count += 1
        rate = min(0.08, 1.0 / math.sqrt(winner.count))
        winner.center = (1.0 - rate) * winner.center + rate * value
        winner.buffer.append(value.copy())
        if len(winner.buffer) > 64:
            winner.buffer.pop(0)

    def partial_fit(self, values: np.ndarray, owners: np.ndarray) -> PlasticDendritronWeb:
        for value, owner in zip(np.asarray(values), np.asarray(owners), strict=True):
            self.learn_one(value, int(owner))
        return self

    def damage(self, owner: int, fraction: float = 0.25) -> tuple[int, ...]:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        branches = [branch for branch in self.regions.get(owner, []) if branch.active]
        count = max(1, math.ceil(len(branches) * fraction)) if branches else 0
        targets = sorted(branches, key=lambda branch: branch.count, reverse=True)[:count]
        for branch in targets:
            branch.damaged = True
            self.events.append({"event": "damage", "owner": owner, "branch": branch.branch_id})
        return tuple(branch.branch_id for branch in targets)

    def repair(self, owner: int) -> None:
        for branch in self.regions.get(owner, []):
            if branch.damaged:
                branch.damaged = False
                self.events.append({"event": "repair", "owner": owner, "branch": branch.branch_id})

    def certificate_accuracy(self, owner: int) -> float:
        points = self.certificates.get(owner, [])
        if not points:
            return float("nan")
        return float(np.mean(self.predict(np.stack(points)) == owner))
