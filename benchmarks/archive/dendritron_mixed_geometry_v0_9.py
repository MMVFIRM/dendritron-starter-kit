from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DENDRITRON_OUTPUT_DIR") or _HERE.parent / "results" / "local")
OUT.mkdir(parents=True, exist_ok=True)
V07_PATH = _HERE / 'dendritron_plasticity_benchmark_v0_7.py'
spec = importlib.util.spec_from_file_location('dendritron_v07', V07_PATH)
v07 = importlib.util.module_from_spec(spec)
sys.modules['dendritron_v07'] = v07
assert spec.loader is not None
spec.loader.exec_module(v07)

EPS = 1e-9


@dataclass(frozen=True)
class Chart:
    name: str
    geometry: str  # euclidean | hyperbolic
    dims: Tuple[int, ...]
    curvature: float = 1.0
    tangent_scale: float = 0.18
    input_is_ball: bool = False


def _clip_ball(x: np.ndarray, c: float) -> np.ndarray:
    max_norm = (1.0 - 1e-6) / math.sqrt(c)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(n, EPS))
    return x * scale


def expmap0(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sqrt_c = math.sqrt(c)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    coef = np.tanh(sqrt_c * n) / np.maximum(sqrt_c * n, EPS)
    out = coef * v
    return _clip_ball(out, c)


def poincare_distance(A: np.ndarray, B: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Broadcasted Poincare-ball distance over the last dimension."""
    A = _clip_ball(np.asarray(A, dtype=float), c)
    B = _clip_ball(np.asarray(B, dtype=float), c)
    diff2 = np.sum((A - B) ** 2, axis=-1)
    an2 = np.sum(A * A, axis=-1)
    bn2 = np.sum(B * B, axis=-1)
    denom = np.maximum((1.0 - c * an2) * (1.0 - c * bn2), 1e-12)
    arg = 1.0 + 2.0 * c * diff2 / denom
    return np.arccosh(np.maximum(arg, 1.0 + 1e-12)) / math.sqrt(c)


def chart_project(X: np.ndarray, chart: Chart) -> np.ndarray:
    Z = np.asarray(X, dtype=float)[..., list(chart.dims)]
    if chart.geometry == 'euclidean':
        return Z
    if chart.input_is_ball:
        return _clip_ball(Z, chart.curvature)
    return expmap0(chart.tangent_scale * Z, chart.curvature)


def chart_distance(X: np.ndarray, center: np.ndarray, chart: Chart) -> np.ndarray:
    XP = chart_project(X, chart)
    CP = chart_project(np.asarray(center)[None, :], chart)[0]
    if chart.geometry == 'euclidean':
        return np.linalg.norm(XP - CP, axis=-1)
    return poincare_distance(XP, CP, chart.curvature)


class MixedGeometryDendritronWeb(v07.PlasticDendritronWeb):
    """Dendritron tissue with a local bank of Euclidean/hyperbolic charts.

    A branch retains one functional owner while its routing support is a learned
    convex mixture over local charts. Chart weights and radii are calibrated
    only from the branch's own buffer and bounded ownership certificates.
    """

    def __init__(self, *args, charts: Sequence[Chart], chart_temperature: float = 0.45,
                 calibration_interval: int = 24, **kwargs):
        super().__init__(*args, **kwargs)
        if not charts:
            raise ValueError('At least one chart is required')
        self.charts = list(charts)
        self.chart_temperature = chart_temperature
        self.calibration_interval = calibration_interval
        self.chart_switch_events = 0

    def _ensure_meta(self, b) -> None:
        k = len(self.charts)
        if not hasattr(b, 'chart_weights'):
            b.chart_weights = np.ones(k, dtype=float) / k
            b.chart_sigmas = np.array([self._initial_chart_sigma(b.sigma, ch, b.center) for ch in self.charts])
            b.chart_best = int(np.argmax(b.chart_weights))
            b.chart_history = [b.chart_best]

    def _initial_chart_sigma(self, sigma: float, chart: Chart, center: np.ndarray) -> float:
        if chart.geometry == 'euclidean':
            return max(0.05, sigma)
        # Estimate local hyperbolic scale by moving one Euclidean sigma along one chart dimension.
        delta = np.zeros_like(center, dtype=float)
        step = sigma if not chart.input_is_ball else min(0.04, max(0.012, 0.04 * sigma))
        delta[chart.dims[0]] = step
        d = float(chart_distance((center + delta)[None, :], center, chart)[0])
        return max(0.05, d)

    def _branch_chart_distances(self, X: np.ndarray, b) -> np.ndarray:
        self._ensure_meta(b)
        X = np.asarray(X, dtype=float)
        return np.stack([chart_distance(X, b.center, ch) for ch in self.charts], axis=1)

    def _branch_score_batch(self, X: np.ndarray, b) -> np.ndarray:
        if not b.active or b.damage_disabled:
            return np.zeros(len(X), dtype=float)
        self._ensure_meta(b)
        D = self._branch_chart_distances(X, b)
        Z = D / np.maximum(b.chart_sigmas[None, :], 1e-6)
        S = np.exp(-0.5 * Z * Z)
        # Mixture permits a functional region to keep identity while evidence moves charts.
        return np.clip(S @ b.chart_weights, 0.0, 0.999999)

    def _branch_score_one(self, x: np.ndarray, b) -> float:
        return float(self._branch_score_batch(np.asarray(x)[None, :], b)[0])

    def _branch_normalized_distance(self, x: np.ndarray, b) -> float:
        self._ensure_meta(b)
        D = self._branch_chart_distances(np.asarray(x)[None, :], b)[0]
        Z = D / np.maximum(b.chart_sigmas, 1e-6)
        # Soft minimum: a branch may recognize evidence through either owned chart.
        return float(-self.chart_temperature * np.log(np.sum(b.chart_weights * np.exp(-Z / self.chart_temperature)) + EPS))

    def class_score(self, x: np.ndarray, class_id: int) -> float:
        branches = [b for b in self.regions.get(class_id, []) if b.active and not b.damage_disabled]
        if not branches:
            return 0.0
        scores = [self._branch_score_one(x, b) for b in branches]
        return 1.0 - float(np.prod([1.0 - min(max(s, 0.0), 0.999999) for s in scores]))

    def class_score_batch(self, X: np.ndarray, class_id: int) -> np.ndarray:
        branches = [b for b in self.regions.get(class_id, []) if b.active and not b.damage_disabled]
        if not branches:
            return np.zeros(len(X), dtype=float)
        scores = np.stack([self._branch_score_batch(X, b) for b in branches], axis=1)
        return 1.0 - np.prod(1.0 - np.clip(scores, 0.0, 0.999999), axis=1)

    def _calibrate_branch(self, b, force: bool = False) -> None:
        self._ensure_meta(b)
        own = list(b.buffer)
        if len(own) < 10 and not force:
            return
        if len(own) < 4:
            return
        Xown = np.stack(own[-self.buffer_size:])
        others = [np.stack(v) for c, v in self.certificates.items() if c != b.class_id and v]
        Xother = np.vstack(others) if others else None

        margins, sigmas = [], []
        for ch in self.charts:
            do = chart_distance(Xown, b.center, ch)
            sig = max(0.035, float(np.quantile(do, 0.80)))
            sigmas.append(sig)
            own_q = float(np.quantile(do / sig, 0.90))
            if Xother is None or len(Xother) == 0:
                margin = -own_q
            else:
                # Bounded sample avoids calibration cost growing with the lifetime stream.
                if len(Xother) > 256:
                    idx = self.rng.choice(len(Xother), 256, replace=False)
                    Xo = Xother[idx]
                else:
                    Xo = Xother
                dn = chart_distance(Xo, b.center, ch) / sig
                margin = float(np.quantile(dn, 0.15) - own_q)
            margins.append(margin)

        logits = np.asarray(margins) / max(self.chart_temperature, 1e-6)
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        old_best = int(np.argmax(b.chart_weights))
        new_best = int(np.argmax(weights))
        b.chart_weights = 0.70 * b.chart_weights + 0.30 * weights
        b.chart_weights /= b.chart_weights.sum()
        b.chart_sigmas = 0.70 * b.chart_sigmas + 0.30 * np.asarray(sigmas)
        b.chart_best = int(np.argmax(b.chart_weights))
        if b.chart_best != old_best:
            self.chart_switch_events += 1
            b.chart_history.append(b.chart_best)
            self.events.append({
                'step': self.step, 'event': 'chart_switch', 'class_id': b.class_id,
                'branch_id': b.branch_id, 'from_chart': self.charts[old_best].name,
                'to_chart': self.charts[b.chart_best].name,
            })

    def _admission_chart_selection(self, candidate) -> None:
        """Select a non-interfering chart before a new branch enters the tissue.

        Equal chart weights are unsafe: an irrelevant chart can steal old certificates
        even when another compartment cleanly separates the new evidence.
        """
        self._ensure_meta(candidate)
        losses = np.zeros(len(self.charts), dtype=float)
        for j, ch in enumerate(self.charts):
            worst = -1e9
            for true_c, pts in self.certificates.items():
                if true_c == candidate.class_id or not pts:
                    continue
                X = np.stack(pts)
                incumbent = self.class_score_batch(X, true_c)
                d = chart_distance(X, candidate.center, ch)
                cand = np.exp(-0.5 * (d / max(candidate.chart_sigmas[j], 1e-6)) ** 2)
                worst = max(worst, float(np.max(cand - incumbent - self.quarantine_margin)))
            losses[j] = max(0.0, worst if worst > -1e8 else 0.0)
        best = int(np.argmin(losses))
        # Hard admission ownership. Plastic calibration may later spread or switch.
        candidate.chart_weights = np.zeros(len(self.charts), dtype=float)
        candidate.chart_weights[best] = 1.0
        candidate.chart_best = best
        candidate.chart_history = [best]

    def _quarantine_safe(self, candidate) -> bool:
        self._ensure_meta(candidate)
        if len(candidate.buffer) >= 4:
            self._calibrate_branch(candidate, force=True)
        else:
            self._admission_chart_selection(candidate)
        for true_c, pts in self.certificates.items():
            if true_c == candidate.class_id or not pts:
                continue
            X = np.stack(pts)
            incumbent = self.class_score_batch(X, true_c)
            cand = self._branch_score_batch(X, candidate)
            if np.any(cand > incumbent + self.quarantine_margin):
                return False
        return True

    def _update_branch(self, b, x: np.ndarray, prediction_was_correct: bool) -> None:
        # Keep the base center/plasticity update, then locally recalibrate geometry.
        super()._update_branch(b, x, prediction_was_correct)
        self._ensure_meta(b)
        if b.count % self.calibration_interval == 0:
            self._calibrate_branch(b)

    def _propose_branch(self, x: np.ndarray, class_id: int, reason: str):
        archived = self.archives.get(class_id, [])
        if archived:
            norms = [self._branch_normalized_distance(x, b) for b in archived]
            idx = int(np.argmin(norms))
            if norms[idx] <= 2.0:
                b = archived.pop(idx)
                b.active = True; b.archived = False; b.damage_disabled = False; b.last_seen = self.step
                self.regions.setdefault(class_id, []).append(b)
                self.events.append({'step': self.step, 'event': 'reactivate', 'class_id': class_id,
                                    'branch_id': b.branch_id, 'reason': reason,
                                    'normalized_distance': float(norms[idx])})
                return b
        return super()._propose_branch(x, class_id, reason)

    def learn_one(self, x: np.ndarray, y: int) -> None:
        self.step += 1
        pred = self.predict_one(x)
        self._add_certificate(x, y)
        if y not in self.regions or len(self.regions[y]) == 0:
            self._propose_branch(x, y, 'new_region')
        else:
            active = [b for b in self.regions[y] if b.active and not b.damage_disabled]
            if not active:
                self._propose_branch(x, y, 'repair_empty_region')
                active = [b for b in self.regions[y] if b.active and not b.damage_disabled]
            if active:
                true_score = self.class_score(x, y)
                dormant = self.archives.get(y, [])
                winner = None
                if dormant:
                    norms = np.array([self._branch_normalized_distance(x, b) for b in dormant])
                    idx = int(np.argmin(norms)); b0 = dormant[idx]
                    dormant_support = self._branch_score_one(x, b0)
                    if norms[idx] <= 2.4 and dormant_support > true_score + 0.08:
                        dormant.pop(idx)
                        b0.active = True; b0.archived = False; b0.damage_disabled = False; b0.last_seen = self.step
                        self.regions[y].append(b0)
                        self.events.append({'step': self.step, 'event': 'reactivate', 'class_id': y,
                                            'branch_id': b0.branch_id, 'reason': 'recurrence_match',
                                            'normalized_distance': float(norms[idx])})
                        winner = b0
                        active.append(b0)
                if winner is None:
                    dnorm = np.array([self._branch_normalized_distance(x, b) for b in active])
                    winner = active[int(np.argmin(dnorm))]
                    if winner.count > 180 and float(np.min(dnorm)) > self.grow_z:
                        candidate = self._propose_branch(x, y, 'novelty_or_error')
                        if candidate is not None:
                            winner = candidate
                self._update_branch(winner, x, pred == y)
        if self.step % self.split_interval == 0:
            self._split_candidates()
        if self.step % self.merge_interval == 0:
            self._merge_candidates()
        if self.step % self.retire_interval == 0:
            self._retire_stale()

    def chart_usage(self) -> Dict[str, float]:
        weights = []
        for bs in self.regions.values():
            for b in bs:
                if b.active and not b.damage_disabled:
                    self._ensure_meta(b)
                    weights.append(b.chart_weights)
        if not weights:
            return {ch.name: 0.0 for ch in self.charts}
        mean = np.mean(np.stack(weights), axis=0)
        return {ch.name: float(v) for ch, v in zip(self.charts, mean)}


class HierarchicalStreamingWorld(v07.StreamingWorld):
    """Same developmental protocol, but class ancestry is embedded in a Poincare compartment.

    Inputs have 12 dimensions. Dimensions 0:2 are Poincare-ball coordinates carrying
    hierarchy; dimensions 2:12 are noisy nuisance/weak Euclidean coordinates. Class
    modes are descendants of class-level cones near the Poincare boundary.
    """
    def __init__(self, input_dim: int = 12, n_classes: int = 8, seed: int = 42):
        super().__init__(input_dim=input_dim, n_classes=n_classes, seed=seed)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        # Binary-tree-like angles: sibling pairs are near, higher-level families are farther.
        angles = np.array([-2.55, -2.25, -1.05, -0.75, 0.60, 0.90, 2.10, 2.40])
        radii = np.array([0.89, 0.90, 0.88, 0.91, 0.89, 0.90, 0.88, 0.91])
        ball = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        self.class_base = np.zeros((n_classes, input_dim), dtype=float)
        self.class_base[:, :2] = ball
        # Weak Euclidean identity plus nuisance dimensions.
        self.class_base[:, 2:4] = 0.30 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.mode_offsets = {}
        for c in range(n_classes):
            tangent = np.array([-math.sin(angles[c]), math.cos(angles[c])])
            radial = np.array([math.cos(angles[c]), math.sin(angles[c])])
            off_a = np.zeros(input_dim); off_b = np.zeros(input_dim)
            off_a[:2] = 0.025 * tangent - 0.025 * radial
            off_b[:2] = -0.025 * tangent + 0.020 * radial
            # Euclidean nuisance modes deliberately overlap.
            off_a[4:] = self.rng.normal(scale=0.18, size=input_dim-4)
            off_b[4:] = self.rng.normal(scale=0.18, size=input_dim-4)
            self.mode_offsets[(c, 'A')] = off_a
            self.mode_offsets[(c, 'B')] = off_b
        for c, name, sign in [(2, 'C', 1.0), (3, 'D', -1.0)]:
            off = np.zeros(input_dim)
            tangent = np.array([-math.sin(angles[c]), math.cos(angles[c])])
            off[:2] = sign * 0.055 * tangent - 0.035 * np.array([math.cos(angles[c]), math.sin(angles[c])])
            off[4:] = self.rng.normal(scale=0.22, size=input_dim-4)
            self.mode_offsets[(c, name)] = off
        self.noise = 0.035
        self.nuisance_noise = 0.42

    def sample(self, class_modes: Dict[int, List[str]], n: int, class_probs: Optional[Dict[int, float]] = None):
        classes = sorted(class_modes)
        probs = np.ones(len(classes)) / len(classes) if class_probs is None else np.array([class_probs.get(c, 0.0) for c in classes], float)
        probs = probs / probs.sum()
        ys = self.rng.choice(classes, size=n, p=probs)
        X, modes = [], []
        for y in ys:
            mode = str(self.rng.choice(class_modes[int(y)]))
            mu = self.class_base[int(y)] + self.mode_offsets[(int(y), mode)]
            x = mu.copy()
            x[:2] += self.rng.normal(scale=self.noise, size=2)
            x[:2] = _clip_ball(x[:2], 1.0)
            x[2:] += self.rng.normal(scale=self.nuisance_noise, size=self.input_dim-2)
            X.append(x); modes.append(f'{int(y)}:{mode}')
        return np.stack(X), ys.astype(int), modes

    def eval_set(self, class_modes: Dict[int, List[str]], per_mode: int = 250):
        X, y, mode = [], [], []
        for c in sorted(class_modes):
            for m in class_modes[c]:
                mu = self.class_base[c] + self.mode_offsets[(c, m)]
                pts = np.repeat(mu[None, :], per_mode, axis=0)
                pts[:, :2] += self.rng.normal(scale=self.noise, size=(per_mode, 2))
                pts[:, :2] = _clip_ball(pts[:, :2], 1.0)
                pts[:, 2:] += self.rng.normal(scale=self.nuisance_noise, size=(per_mode, self.input_dim-2))
                X.append(pts); y.extend([c]*per_mode); mode.extend([f'{c}:{m}']*per_mode)
        return np.vstack(X), np.array(y, int), np.array(mode)


class HybridStreamingWorld(v07.StreamingWorld):
    """A world where one tissue must operate across Euclidean and hyperbolic compartments."""
    def __init__(self, input_dim: int = 12, n_classes: int = 8, seed: int = 42):
        super().__init__(input_dim=input_dim, n_classes=n_classes, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.class_base = np.zeros((n_classes, input_dim), dtype=float)
        self.mode_offsets = {}
        # Flat Euclidean owners 0..3 in dimensions 0:6.
        A = self.rng.normal(size=(6, 6)); Q, _ = np.linalg.qr(A)
        self.class_base[:4, :6] = 3.0 * Q[:, :4].T
        for c in range(4):
            v = self.rng.normal(size=6); v /= np.linalg.norm(v)
            a = np.zeros(input_dim); b = np.zeros(input_dim)
            a[:6] = 0.8*v; b[:6] = -0.8*v
            self.mode_offsets[(c,'A')] = a; self.mode_offsets[(c,'B')] = b
        # Hierarchical owners 4..7 in Poincare dimensions 6:8.
        angles = np.array([-1.15, -0.85, 0.85, 1.15])
        for j, c in enumerate(range(4,8)):
            r = 0.90
            self.class_base[c,6:8] = [r*np.cos(angles[j]), r*np.sin(angles[j])]
            tangent = np.array([-np.sin(angles[j]), np.cos(angles[j])])
            a = np.zeros(input_dim); b = np.zeros(input_dim)
            a[6:8] = 0.028*tangent; b[6:8] = -0.028*tangent
            self.mode_offsets[(c,'A')] = a; self.mode_offsets[(c,'B')] = b
        # Cross-chart class 2 receives novel mode C in the hyperbolic compartment.
        c = 2; angle = 2.55; r = 0.91
        off = np.zeros(input_dim)
        off[:6] = -self.class_base[c,:6]  # neutralize Euclidean identity for this mode
        off[6:8] = [r*np.cos(angle), r*np.sin(angle)]
        self.mode_offsets[(2,'C')] = off
        # Class 3 gets a further Euclidean specialization.
        v = self.rng.normal(size=6); v /= np.linalg.norm(v)
        offd = np.zeros(input_dim); offd[:6] = 1.8*v
        self.mode_offsets[(3,'D')] = offd
        self.noise_e = 0.42; self.noise_h = 0.032; self.nuisance = 0.35

    def _draw(self, c: int, m: str) -> np.ndarray:
        mu = self.class_base[c] + self.mode_offsets[(c,m)]
        x = mu.copy()
        x[:6] += self.rng.normal(scale=self.noise_e, size=6)
        x[6:8] += self.rng.normal(scale=self.noise_h, size=2)
        x[6:8] = _clip_ball(x[6:8], 1.0)
        x[8:] += self.rng.normal(scale=self.nuisance, size=4)
        return x

    def sample(self, class_modes, n, class_probs=None):
        classes=sorted(class_modes)
        probs=np.ones(len(classes))/len(classes) if class_probs is None else np.array([class_probs.get(c,0.0) for c in classes],float)
        probs/=probs.sum(); ys=self.rng.choice(classes,size=n,p=probs)
        X=[]; modes=[]
        for y in ys:
            m=str(self.rng.choice(class_modes[int(y)])); X.append(self._draw(int(y),m)); modes.append(f'{int(y)}:{m}')
        return np.stack(X), ys.astype(int), modes

    def eval_set(self,class_modes,per_mode=250):
        X=[]; y=[]; modes=[]
        for c in sorted(class_modes):
            for m in class_modes[c]:
                X.extend([self._draw(c,m) for _ in range(per_mode)]); y.extend([c]*per_mode); modes.extend([f'{c}:{m}']*per_mode)
        return np.stack(X),np.array(y,int),np.array(modes)


def make_charts(world_kind: str, input_dim: int, mode: str) -> List[Chart]:
    if world_kind == 'flat':
        if mode == 'euclidean':
            return [Chart('E-full','euclidean',tuple(range(input_dim)))]
        if mode == 'hyperbolic':
            return [Chart('H-full','hyperbolic',tuple(range(input_dim)),tangent_scale=0.14)]
        return [Chart('E-full','euclidean',tuple(range(input_dim))),
                Chart('H-full','hyperbolic',tuple(range(input_dim)),tangent_scale=0.14)]
    if world_kind == 'hierarchical':
        if mode == 'euclidean':
            return [Chart('E-full','euclidean',tuple(range(input_dim)))]
        if mode == 'compartment_euclidean':
            return [Chart('E-full','euclidean',tuple(range(input_dim))), Chart('E-tree','euclidean',(0,1))]
        if mode == 'hyperbolic':
            return [Chart('H-tree','hyperbolic',(0,1),input_is_ball=True)]
        return [Chart('E-full','euclidean',tuple(range(input_dim))),
                Chart('E-tree','euclidean',(0,1)),
                Chart('H-tree','hyperbolic',(0,1),input_is_ball=True)]
    # hybrid
    if mode == 'euclidean':
        return [Chart('E-full','euclidean',tuple(range(input_dim)))]
    if mode == 'compartment_euclidean':
        return [Chart('E-flat','euclidean',tuple(range(0,6))), Chart('E-tree','euclidean',(6,7)), Chart('E-full','euclidean',tuple(range(input_dim)))]
    if mode == 'hyperbolic':
        return [Chart('H-full','hyperbolic',tuple(range(input_dim)),tangent_scale=0.14)]
    return [Chart('E-flat','euclidean',tuple(range(0,6))),
            Chart('E-tree','euclidean',(6,7)),
            Chart('H-tree','hyperbolic',(6,7),input_is_ball=True),
            Chart('E-full','euclidean',tuple(range(input_dim)))]


def world_for(kind: str, seed: int):
    if kind == 'flat': return v07.StreamingWorld(seed=seed)
    if kind == 'hierarchical': return HierarchicalStreamingWorld(seed=seed)
    if kind == 'hybrid': return HybridStreamingWorld(seed=seed)
    raise ValueError(kind)


def run_protocol(world_kind: str, router: str, seed: int, exposure_scale: float = 0.55, flat_center_scale: float = 4.1, flat_noise: float = 0.48, hierarchical_noise: float = 0.035, hierarchical_nuisance: float = 0.42, hybrid_noise_scale: float = 1.0) -> dict:
    rng=np.random.default_rng(seed+900)
    world=world_for(world_kind,seed)
    if world_kind == 'flat':
        world.class_base *= flat_center_scale / 4.1
        world.noise = flat_noise
        base_sigma=max(0.75,3.05*flat_noise); min_sigma=max(0.20,0.75*flat_noise); max_sigma=max(1.5,5.8*flat_noise)
    elif world_kind == 'hierarchical':
        world.noise = hierarchical_noise
        world.nuisance_noise = hierarchical_nuisance
        base_sigma=0.75; min_sigma=0.08; max_sigma=2.0
    else:
        world.noise_e *= hybrid_noise_scale; world.noise_h *= hybrid_noise_scale; world.nuisance *= hybrid_noise_scale
        base_sigma=1.10; min_sigma=0.12; max_sigma=2.5
    web=MixedGeometryDendritronWeb(
        input_dim=world.input_dim, charts=make_charts(world_kind,world.input_dim,router),
        base_sigma=base_sigma,min_sigma=min_sigma,max_sigma=max_sigma,grow_z=2.05,
        certificate_size=48,retirement_steps=max(500,int(1000*exposure_scale)),
        split_interval=max(160,int(340*exposure_scale)),merge_interval=max(220,int(480*exposure_scale)),
        retire_interval=max(120,int(240*exposure_scale)),rng=rng)
    base_counts=[1800,1100,1200,1100,1200,700,850]
    counts=[max(280,int(n*exposure_scale)) for n in base_counts]
    phases=[
        ('warmup',counts[0],{c:['A','B'] for c in range(6)},None),
        ('novel_mode',counts[1],{**{c:['A','B'] for c in range(6)},2:['A','B','C']},None),
        ('new_classes',counts[2],{**{c:['A','B'] for c in range(6)},2:['A','B','C'],6:['A','B'],7:['A','B']},None),
        ('specialization',counts[3],{**{c:['A','B'] for c in range(8)},2:['A','B','C'],3:['A','B','D']},None),
        ('inactivity',counts[4],{**{c:['A','B'] for c in range(8) if c!=4},2:['A','B','C'],3:['A','B','D']},None),
        ('recurrence',counts[5],{**{c:['A','B'] for c in range(8)},2:['A','B','C'],3:['A','B','D'],4:['B']},{4:.42,**{c:.58/7 for c in range(8) if c!=4}}),
        ('repair',counts[6],{**{c:['A','B'] for c in range(8)},2:['A','B','C'],3:['A','B','D']},{1:.38,**{c:.62/7 for c in range(8) if c!=1}}),
    ]
    eval_modes={c:['A','B'] for c in range(8)}; eval_modes[2]=['A','B','C']; eval_modes[3]=['A','B','D']
    Xev,yev,gev=world.eval_set(eval_modes,per_mode=90)
    phase_rows=[]; recurrence_start=None; repair_start=None; c4_before=None; c1_damage=None; detected=False
    for phase,n,modes,probs in phases:
        X,y,_=world.sample(modes,n,probs)
        if phase=='repair':
            web.damage_region(1,1.0); c1_damage=float(np.mean(web.predict(Xev[yev==1])==1)); detected=1 in web.repair_scan(threshold=.72); repair_start=web.step
        if phase=='recurrence':
            recurrence_start=web.step; c4_before=float(np.mean(web.predict(Xev[yev==4])==4))
        web.learn_batch(X,y)
        if phase=='warmup':
            web.inject_redundant_branch(0); web._merge_candidates()
        if phase=='inactivity':
            web.induce_dormancy(4,world.class_base[4]+world.mode_offsets[(4,'A')])
        phase_rows.append({'phase':phase,'accuracy':float(np.mean(web.predict(Xev)==yev))})
    pred=web.predict(Xev)
    final=float(np.mean(pred==yev)); c1=float(np.mean(pred[yev==1]==1)); c4=float(np.mean(pred[yev==4]==4))
    old_mask=np.array([g not in {'2:C','3:D','6:A','6:B','7:A','7:B'} for g in gev]); new_mask=~old_mask
    events=pd.DataFrame(web.events); ops=events.event.value_counts().to_dict() if len(events) else {}
    react=events[(events.event=='reactivate')&(events.class_id==4)&(events.step>=recurrence_start)] if len(events) else pd.DataFrame()
    repairs=events[(events.event.isin(['reactivate','grow']))&(events.class_id==1)&(events.step>=repair_start)] if len(events) else pd.DataFrame()
    usage=web.chart_usage()
    class_usage = {}
    branch_rows = []
    for cid, branches in sorted(web.regions.items()):
        ws = []
        for b in branches:
            if b.active and not b.damage_disabled:
                web._ensure_meta(b)
                ws.append(b.chart_weights)
                branch_rows.append({
                    'class_id': int(cid), 'branch_id': int(b.branch_id),
                    **{ch.name: float(w) for ch, w in zip(web.charts, b.chart_weights)},
                    'best_chart': web.charts[int(np.argmax(b.chart_weights))].name,
                    'chart_switches': int(max(0, len(getattr(b, 'chart_history', [])) - 1)),
                })
        if ws:
            mw = np.mean(np.stack(ws), axis=0)
            class_usage[str(cid)] = {ch.name: float(w) for ch, w in zip(web.charts, mw)}
    # Functional target is lower for the deliberately mixed world but still strict.
    threshold={'flat':.90,'hierarchical':.86,'hybrid':.84}[world_kind]
    return {
        'world':world_kind,'router':router,'seed':seed,'flat_center_scale':flat_center_scale,'flat_noise':flat_noise,'hierarchical_noise':hierarchical_noise,'hierarchical_nuisance':hierarchical_nuisance,'hybrid_noise_scale':hybrid_noise_scale,'final_accuracy':final,
        'old_accuracy':float(np.mean(pred[old_mask]==yev[old_mask])),
        'new_accuracy':float(np.mean(pred[new_mask]==yev[new_mask])),
        'class4_before':c4_before,'class4_after':c4,
        'reactivation_samples':None if len(react)==0 else int(react.iloc[0].step-recurrence_start),
        'class1_after_damage':c1_damage,'class1_after_repair':c1,
        'repair_samples':None if len(repairs)==0 else int(repairs.iloc[0].step-repair_start),
        'damage_detected':bool(detected),'active_branches':web.structural_counts()['active_branches'],
        'archived_branches':web.structural_counts()['archived_branches'],
        'grow_events':int(ops.get('grow',0)),'split_events':int(ops.get('split',0)),
        'merge_events':int(ops.get('merge',0)),'retire_events':int(ops.get('retire',0)),
        'reactivate_events':int(ops.get('reactivate',0)),'chart_switch_events':int(web.chart_switch_events),
        **{f'chart_{k}':v for k,v in usage.items()},
        'functional_pass':bool(final>=threshold and c1>=threshold-0.08 and c4>=threshold-0.08 and detected),
        'phase_accuracy_json':json.dumps(phase_rows),'class_chart_usage_json':json.dumps(class_usage,sort_keys=True),'branch_chart_rows_json':json.dumps(branch_rows,sort_keys=True),
    }


def run_all() -> dict:
    started=time.time(); rows=[]
    for world in ['flat','hierarchical','hybrid']:
        for router in ['euclidean','hyperbolic','mixed']:
            for seed in [11,29,47]:
                r=run_protocol(world,router,seed)
                rows.append(r)
                print(world,router,seed,f"acc={r['final_accuracy']:.4f}",f"pass={r['functional_pass']}",flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'dendritron_v0_9_runs.csv',index=False)
    metrics=df.groupby(['world','router'],as_index=False).agg(
        runs=('seed','count'),pass_rate=('functional_pass','mean'),final_accuracy=('final_accuracy','mean'),
        final_accuracy_std=('final_accuracy','std'),old_accuracy=('old_accuracy','mean'),new_accuracy=('new_accuracy','mean'),
        reactivation_samples=('reactivation_samples','mean'),repair_samples=('repair_samples','mean'),
        active_branches=('active_branches','mean'),chart_switch_events=('chart_switch_events','mean'))
    metrics.to_csv(OUT/'dendritron_v0_9_geometry_comparison.csv',index=False)
    # Chart usage for mixed router only.
    chart_cols=[c for c in df.columns if c.startswith('chart_')]
    usage=df[df.router=='mixed'].groupby('world',as_index=False)[chart_cols].mean(numeric_only=True)
    usage.to_csv(OUT/'dendritron_v0_9_chart_usage.csv',index=False)
    summary={'runtime_seconds':time.time()-started,'runs':len(df),'comparison':metrics.to_dict(orient='records'),
             'mixed_chart_usage':usage.to_dict(orient='records')}
    with open(OUT/'dendritron_v0_9_summary.json','w') as f: json.dump(summary,f,indent=2)
    print('\nAGGREGATE\n',metrics.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nMIXED CHART USAGE\n',usage.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return summary


if __name__=='__main__':
    run_all()
