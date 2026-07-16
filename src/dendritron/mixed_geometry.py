"""Compartmentalized Euclidean/hyperbolic chart banks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Chart, chart_distance


@dataclass
class MixedGeometryBranch:
    branch_id: int
    owner: int
    center: np.ndarray
    sigma: float
    charts: tuple[Chart, ...]
    chart_weights: np.ndarray
    count: int = 1
    active: bool = True
    damaged: bool = False
    buffer: list[np.ndarray] = field(default_factory=list)
    chart_history: list[int] = field(default_factory=list)

    def distances(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        return np.stack(
            [chart_distance(values, self.center, chart) for chart in self.charts], axis=1
        )

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if not self.active or self.damaged:
            return np.zeros(len(values))
        support = np.exp(-0.5 * (self.distances(values) / max(self.sigma, 1e-8)) ** 2)
        return support @ self.chart_weights

    @property
    def selected_chart(self) -> Chart:
        return self.charts[int(np.argmax(self.chart_weights))]


class MixedGeometryWeb:
    """A plastic web whose branches keep one owner and a local chart bank.

    Geometry is a routing property. Ownership never changes when a branch
    switches between Euclidean and hyperbolic charts.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        charts: tuple[Chart, ...] | list[Chart],
        chart_temperature: float = 0.35,
        sigma: float = 1.0,
        novelty_threshold: float = 2.25,
        certificate_size: int = 64,
        quarantine_margin: float = 0.02,
        seed: int = 0,
    ) -> None:
        if not charts:
            raise ValueError("at least one chart is required")
        self.input_dim = input_dim
        self.sigma = sigma
        self.novelty_threshold = novelty_threshold
        self.certificate_size = certificate_size
        self.quarantine_margin = quarantine_margin
        self.rng = np.random.default_rng(seed)
        self.regions: dict[int, list[MixedGeometryBranch]] = {}
        self.certificates: dict[int, list[np.ndarray]] = {}
        self.events: list[dict[str, object]] = []
        self.branch_counter = 0
        self.chart_bank = tuple(charts)
        self.chart_temperature = chart_temperature

    def _remember(self, value: np.ndarray, owner: int) -> None:
        certificate = self.certificates.setdefault(owner, [])
        if len(certificate) < self.certificate_size:
            certificate.append(value.copy())
        elif self.rng.random() < 0.05:
            certificate[int(self.rng.integers(len(certificate)))] = value.copy()

    def scores(self, values: np.ndarray, owner: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        branches = [
            branch for branch in self.regions.get(owner, []) if branch.active and not branch.damaged
        ]
        if not branches:
            return np.zeros(len(values))
        support = np.stack([branch.score(values) for branch in branches], axis=1)
        return 1.0 - np.prod(1.0 - np.clip(support, 0.0, 1.0 - 1e-9), axis=1)

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        owners = sorted(self.regions)
        if not owners:
            return np.full(len(values), -1, dtype=np.int64)
        scores = np.stack([self.scores(values, owner) for owner in owners], axis=1)
        return np.asarray(owners)[scores.argmax(axis=1)]

    def _chart_violation(self, value: np.ndarray, owner: int, chart: Chart, sigma: float) -> float:
        worst = 0.0
        for protected_owner, points in self.certificates.items():
            if protected_owner == owner or not points:
                continue
            values = np.stack(points)
            incumbent = self.scores(values, protected_owner)
            distance = chart_distance(values, value, chart)
            candidate = np.exp(-0.5 * (distance / max(sigma, 1e-8)) ** 2)
            worst = max(worst, float(np.max(candidate - incumbent - self.quarantine_margin)))
        return max(0.0, worst)

    def _grow(self, value: np.ndarray, owner: int, reason: str) -> MixedGeometryBranch | None:
        sigma = self.sigma
        for _ in range(8):
            violations = np.array(
                [self._chart_violation(value, owner, chart, sigma) for chart in self.chart_bank]
            )
            best = int(violations.argmin())
            self.branch_counter += 1
            weights = np.zeros(len(self.chart_bank))
            weights[best] = 1.0
            candidate = MixedGeometryBranch(
                branch_id=self.branch_counter,
                owner=owner,
                center=value.copy(),
                sigma=sigma,
                charts=self.chart_bank,
                chart_weights=weights,
                chart_history=[best],
            )
            if violations[best] <= 0:
                self.regions.setdefault(owner, []).append(candidate)
                self.events.append(
                    {
                        "event": "grow",
                        "owner": owner,
                        "branch": candidate.branch_id,
                        "reason": reason,
                        "chart": candidate.selected_chart.name,
                    }
                )
                return candidate
            self.events.append(
                {
                    "event": "quarantine",
                    "owner": owner,
                    "branch": candidate.branch_id,
                    "reason": reason,
                    "minimum_violation": float(violations[best]),
                }
            )
            sigma *= 0.72
        return None

    def _calibrate(self, branch: MixedGeometryBranch) -> None:
        if len(branch.buffer) < 4:
            return
        local = np.stack(branch.buffer)
        losses = []
        for chart in branch.charts:
            local_spread = float(np.mean(chart_distance(local, branch.center, chart)))
            violation = self._chart_violation(branch.center, branch.owner, chart, branch.sigma)
            losses.append(local_spread + 10.0 * violation)
        losses_array = np.asarray(losses)
        logits = -(losses_array - losses_array.min()) / max(self.chart_temperature, 1e-6)
        weights = np.exp(logits)
        branch.chart_weights = weights / weights.sum()
        selected = int(branch.chart_weights.argmax())
        if not branch.chart_history or selected != branch.chart_history[-1]:
            branch.chart_history.append(selected)
            self.events.append(
                {
                    "event": "chart_switch",
                    "owner": branch.owner,
                    "branch": branch.branch_id,
                    "chart": branch.charts[selected].name,
                }
            )

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
        normalized = np.array(
            [float(np.min(branch.distances(value))) / max(branch.sigma, 1e-8) for branch in active]
        )
        winner = active[int(normalized.argmin())]
        if float(normalized.min()) > self.novelty_threshold:
            candidate = self._grow(value, owner, "novelty")
            if candidate is not None:
                winner = candidate
        winner.count += 1
        rate = min(0.08, 1.0 / math.sqrt(winner.count))
        winner.center = (1.0 - rate) * winner.center + rate * value
        winner.buffer.append(value.copy())
        if len(winner.buffer) > 64:
            winner.buffer.pop(0)
        if winner.count % 16 == 0:
            self._calibrate(winner)

    def chart_usage(self) -> dict[str, float]:
        weights = [
            branch.chart_weights
            for branches in self.regions.values()
            for branch in branches
            if branch.active and not branch.damaged
        ]
        mean = np.mean(np.stack(weights), axis=0) if weights else np.zeros(len(self.chart_bank))
        return {
            chart.name: float(value) for chart, value in zip(self.chart_bank, mean, strict=True)
        }

    def partial_fit(self, values: np.ndarray, owners: np.ndarray) -> MixedGeometryWeb:
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
