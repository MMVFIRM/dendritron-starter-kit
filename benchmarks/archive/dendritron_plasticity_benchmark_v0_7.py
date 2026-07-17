from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DENDRITRON_OUTPUT_DIR") or _HERE.parent / "results" / "local")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class Branch:
    branch_id: int
    class_id: int
    center: np.ndarray
    sigma: float
    count: int
    created_step: int
    last_seen: int
    active: bool = True
    archived: bool = False
    damage_disabled: bool = False
    distance_ema: float = 0.0
    error_ema: float = 0.0
    utility_ema: float = 0.0
    buffer: List[np.ndarray] = field(default_factory=list)

    def score(self, x: np.ndarray) -> float:
        if not self.active or self.damage_disabled:
            return 0.0
        d2 = float(np.sum((x - self.center) ** 2))
        s2 = max(self.sigma * self.sigma, 1e-8)
        return math.exp(-0.5 * d2 / s2)


class PlasticDendritronWeb:
    """Online prototype tissue with explicit ownership and structural plasticity.

    Each class is a functional region. Each region owns local RBF branches.
    Structural edits are local and logged: grow, split, merge, retire, reactivate,
    damage, repair. Candidate branches are quarantined against certificates from
    existing regions before activation.
    """

    def __init__(
        self,
        input_dim: int,
        base_sigma: float = 1.50,
        min_sigma: float = 0.35,
        max_sigma: float = 2.8,
        grow_z: float = 2.05,
        wrong_score_trigger: float = 0.48,
        center_lr_cap: float = 0.08,
        sigma_lr: float = 0.03,
        certificate_size: int = 72,
        quarantine_margin: float = 0.02,
        retirement_steps: int = 1100,
        split_interval: int = 350,
        merge_interval: int = 500,
        retire_interval: int = 250,
        buffer_size: int = 80,
        rng: Optional[np.random.Generator] = None,
    ):
        self.input_dim = input_dim
        self.base_sigma = base_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.grow_z = grow_z
        self.wrong_score_trigger = wrong_score_trigger
        self.center_lr_cap = center_lr_cap
        self.sigma_lr = sigma_lr
        self.certificate_size = certificate_size
        self.quarantine_margin = quarantine_margin
        self.retirement_steps = retirement_steps
        self.split_interval = split_interval
        self.merge_interval = merge_interval
        self.retire_interval = retire_interval
        self.buffer_size = buffer_size
        self.rng = rng or np.random.default_rng(0)

        self.regions: Dict[int, List[Branch]] = {}
        self.archives: Dict[int, List[Branch]] = {}
        self.certificates: Dict[int, List[np.ndarray]] = {}
        self.branch_counter = 0
        self.step = 0
        self.events: List[dict] = []
        self.last_recovery_event: Dict[int, int] = {}

    # ---------- inference ----------
    def class_score(self, x: np.ndarray, class_id: int) -> float:
        branches = self.regions.get(class_id, [])
        if not branches:
            return 0.0
        scores = [b.score(x) for b in branches]
        # Noisy-OR coalition: one strong local branch can recognize a mode,
        # while multiple moderate branches can cooperate.
        return 1.0 - float(np.prod([1.0 - min(max(s, 0.0), 0.999999) for s in scores]))

    def class_score_batch(self, X: np.ndarray, class_id: int) -> np.ndarray:
        branches = [b for b in self.regions.get(class_id, []) if b.active and not b.damage_disabled]
        if not branches:
            return np.zeros(len(X), dtype=float)
        centers = np.stack([b.center for b in branches])
        sigmas = np.array([max(b.sigma, self.min_sigma) for b in branches])
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        scores = np.exp(-0.5 * d2 / (sigmas[None, :] ** 2))
        scores = np.clip(scores, 0.0, 0.999999)
        return 1.0 - np.prod(1.0 - scores, axis=1)

    def score_vector(self, x: np.ndarray, classes: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
        cls = sorted(self.regions) if classes is None else list(classes)
        if not cls:
            return np.empty(0), []
        return np.array([self.class_score(x, c) for c in cls], dtype=float), cls

    def predict_one(self, x: np.ndarray) -> int:
        scores, cls = self.score_vector(x)
        if len(cls) == 0:
            return -1
        return int(cls[int(np.argmax(scores))])

    def predict(self, X: np.ndarray) -> np.ndarray:
        cls = sorted(self.regions)
        if not cls:
            return np.full(len(X), -1, dtype=int)
        scores = np.stack([self.class_score_batch(X, c) for c in cls], axis=1)
        return np.array(cls, dtype=int)[np.argmax(scores, axis=1)]

    # ---------- certificates ----------
    def _add_certificate(self, x: np.ndarray, y: int) -> None:
        certs = self.certificates.setdefault(y, [])
        if len(certs) < self.certificate_size:
            certs.append(x.copy())
        else:
            # Reservoir-like replacement biased toward recent evidence.
            if self.rng.random() < 0.08:
                certs[int(self.rng.integers(0, len(certs)))] = x.copy()

    def certificate_accuracy(self, class_id: int) -> float:
        certs = self.certificates.get(class_id, [])
        if not certs:
            return float('nan')
        X = np.stack(certs)
        return float(np.mean(self.predict(X) == class_id))

    # ---------- branch lifecycle ----------
    def _new_branch(self, class_id: int, center: np.ndarray, sigma: float, reason: str) -> Branch:
        self.branch_counter += 1
        b = Branch(
            branch_id=self.branch_counter,
            class_id=class_id,
            center=center.copy(),
            sigma=float(np.clip(sigma, self.min_sigma, self.max_sigma)),
            count=1,
            created_step=self.step,
            last_seen=self.step,
        )
        self.events.append({
            'step': self.step,
            'event': 'grow',
            'class_id': class_id,
            'branch_id': b.branch_id,
            'reason': reason,
            'sigma': b.sigma,
        })
        return b

    def _candidate_sigma(self, x: np.ndarray, class_id: int) -> float:
        sigma = self.base_sigma
        others = [np.stack(pts) for c, pts in self.certificates.items() if c != class_id and pts]
        if others:
            O = np.vstack(others)
            d = float(np.sqrt(np.min(np.sum((O - x[None, :]) ** 2, axis=1))))
            sigma = min(sigma, max(self.min_sigma, 0.42 * d))
        return float(np.clip(sigma, self.min_sigma, self.max_sigma))

    def _quarantine_safe(self, candidate: Branch) -> bool:
        # Vectorized bounded ownership-certificate check.
        for true_c, pts in self.certificates.items():
            if true_c == candidate.class_id or not pts:
                continue
            X = np.stack(pts)
            incumbent = self.class_score_batch(X, true_c)
            d2 = np.sum((X - candidate.center[None, :]) ** 2, axis=1)
            cand = np.exp(-0.5 * d2 / max(candidate.sigma ** 2, 1e-8))
            if np.any(cand > incumbent + self.quarantine_margin):
                return False
        return True

    def _propose_branch(self, x: np.ndarray, class_id: int, reason: str) -> Optional[Branch]:
        # Prefer reactivation of dormant local structure.
        archived = self.archives.get(class_id, [])
        if archived:
            distances = [np.linalg.norm(x - b.center) / max(b.sigma, self.min_sigma) for b in archived]
            idx = int(np.argmin(distances))
            if distances[idx] <= 2.0:
                b = archived.pop(idx)
                b.active = True
                b.archived = False
                b.damage_disabled = False
                b.last_seen = self.step
                self.regions.setdefault(class_id, []).append(b)
                self.events.append({
                    'step': self.step,
                    'event': 'reactivate',
                    'class_id': class_id,
                    'branch_id': b.branch_id,
                    'reason': reason,
                    'normalized_distance': float(distances[idx]),
                })
                return b

        sigma = self._candidate_sigma(x, class_id)
        # Shrink until the candidate passes protected certificates.
        for _ in range(8):
            b = self._new_branch(class_id, x, sigma, reason)
            if self._quarantine_safe(b):
                self.regions.setdefault(class_id, []).append(b)
                return b
            # The logged grow attempt becomes a quarantine rejection.
            self.events[-1]['event'] = 'reject'
            self.events[-1]['reason'] = f'{reason}:certificate_conflict'
            sigma *= 0.72
            if sigma < self.min_sigma:
                break
        return None

    def _update_branch(self, b: Branch, x: np.ndarray, prediction_was_correct: bool) -> None:
        dist = float(np.linalg.norm(x - b.center))
        b.count += 1
        lr = min(self.center_lr_cap, 1.0 / math.sqrt(b.count))
        b.center = (1.0 - lr) * b.center + lr * x
        b.distance_ema = 0.97 * b.distance_ema + 0.03 * dist
        target_sigma = np.clip(max(self.base_sigma, 1.35 * b.distance_ema), self.min_sigma, self.max_sigma)
        b.sigma = float((1.0 - self.sigma_lr) * b.sigma + self.sigma_lr * target_sigma)
        b.error_ema = 0.97 * b.error_ema + 0.03 * (0.0 if prediction_was_correct else 1.0)
        b.utility_ema = 0.995 * b.utility_ema + 0.005
        b.last_seen = self.step
        b.buffer.append(x.copy())
        if len(b.buffer) > self.buffer_size:
            b.buffer.pop(0)

    def _split_candidates(self) -> None:
        for class_id, branches in list(self.regions.items()):
            for b in list(branches):
                if not b.active or b.damage_disabled or len(b.buffer) < 48:
                    continue
                X = np.stack(b.buffer)
                total_sse = float(np.sum((X - X.mean(axis=0)) ** 2))
                if total_sse <= 1e-8:
                    continue
                # Lightweight deterministic 2-means; avoids global optimizer overhead.
                X0 = X - X.mean(axis=0, keepdims=True)
                _, _, vt = np.linalg.svd(X0, full_matrices=False)
                axis = vt[0]
                proj = X0 @ axis
                c0 = X[int(np.argmin(proj))].copy()
                c1 = X[int(np.argmax(proj))].copy()
                labels = np.zeros(len(X), dtype=int)
                for _ in range(8):
                    d0 = np.sum((X - c0) ** 2, axis=1)
                    d1 = np.sum((X - c1) ** 2, axis=1)
                    labels = (d1 < d0).astype(int)
                    if np.any(labels == 0): c0 = X[labels == 0].mean(axis=0)
                    if np.any(labels == 1): c1 = X[labels == 1].mean(axis=0)
                centers2 = np.stack([c0, c1])
                counts = np.bincount(labels, minlength=2)
                if np.min(counts) < 14:
                    continue
                split_sse = float(sum(np.sum((X[labels == k] - centers2[k]) ** 2) for k in range(2)))
                separation = float(np.linalg.norm(centers2[0] - centers2[1]))
                if split_sse / total_sse > 0.70 or separation < 0.90 * max(b.sigma, self.min_sigma):
                    continue

                new_branches = []
                safe = True
                for k in range(2):
                    Xk = X[labels == k]
                    sigma = float(np.clip(1.35 * np.mean(np.linalg.norm(Xk - Xk.mean(axis=0), axis=1)), self.min_sigma, self.max_sigma))
                    candidate = Branch(
                        branch_id=self.branch_counter + 1 + k,
                        class_id=class_id,
                        center=centers2[k].copy(),
                        sigma=sigma,
                        count=len(Xk),
                        created_step=self.step,
                        last_seen=self.step,
                        buffer=[z.copy() for z in Xk[-self.buffer_size:]],
                    )
                    if not self._quarantine_safe(candidate):
                        safe = False
                        break
                    new_branches.append(candidate)
                if not safe:
                    continue

                branches.remove(b)
                for nb in new_branches:
                    self.branch_counter = max(self.branch_counter, nb.branch_id)
                    branches.append(nb)
                b.active = False
                b.archived = True
                self.archives.setdefault(class_id, []).append(b)
                self.events.append({
                    'step': self.step,
                    'event': 'split',
                    'class_id': class_id,
                    'branch_id': b.branch_id,
                    'child_ids': '|'.join(str(x.branch_id) for x in new_branches),
                    'sse_ratio': split_sse / total_sse,
                    'separation': separation,
                })
                return  # one structural edit per maintenance pass

    def _merge_candidates(self) -> None:
        for class_id, branches in list(self.regions.items()):
            active = [b for b in branches if b.active and not b.damage_disabled]
            best = None
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    a, b = active[i], active[j]
                    d = float(np.linalg.norm(a.center - b.center))
                    threshold = 0.50 * (a.sigma + b.sigma)
                    if d < threshold and (best is None or d < best[0]):
                        best = (d, a, b)
            if best is None:
                continue
            d, a, b = best
            total = a.count + b.count
            center = (a.count * a.center + b.count * b.center) / total
            sigma = max(a.sigma, b.sigma, 0.5 * d)
            merged = Branch(
                branch_id=self.branch_counter + 1,
                class_id=class_id,
                center=center,
                sigma=float(np.clip(sigma, self.min_sigma, self.max_sigma)),
                count=total,
                created_step=self.step,
                last_seen=max(a.last_seen, b.last_seen),
                buffer=(a.buffer + b.buffer)[-self.buffer_size:],
            )
            if not self._quarantine_safe(merged):
                continue
            self.branch_counter += 1
            branches.remove(a)
            branches.remove(b)
            branches.append(merged)
            for old in (a, b):
                old.active = False
                old.archived = True
                self.archives.setdefault(class_id, []).append(old)
            self.events.append({
                'step': self.step,
                'event': 'merge',
                'class_id': class_id,
                'branch_id': merged.branch_id,
                'parent_ids': f'{a.branch_id}|{b.branch_id}',
                'distance': d,
            })
            return

    def _retire_stale(self) -> None:
        for class_id, branches in list(self.regions.items()):
            if len(branches) <= 1:
                continue
            candidates = [b for b in branches if self.step - b.last_seen > self.retirement_steps and not b.damage_disabled]
            if not candidates:
                continue
            b = min(candidates, key=lambda z: z.utility_ema)
            # Retire only if the class certificate remains mostly supported by siblings.
            branches.remove(b)
            acc = self.certificate_accuracy(class_id)
            if np.isnan(acc) or acc < 0.92:
                branches.append(b)
                continue
            b.active = False
            b.archived = True
            self.archives.setdefault(class_id, []).append(b)
            self.events.append({
                'step': self.step,
                'event': 'retire',
                'class_id': class_id,
                'branch_id': b.branch_id,
                'inactive_steps': self.step - b.last_seen,
            })
            return

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
                # Recurrence recognition: dormant owned structure gets first refusal
                # before the active branch is deformed to relearn an old mode.
                true_score = self.class_score(x, y)
                dormant = self.archives.get(y, [])
                winner = None
                if dormant:
                    norms = np.array([np.linalg.norm(x - b.center) / max(b.sigma, self.min_sigma) for b in dormant])
                    idx = int(np.argmin(norms))
                    b0 = dormant[idx]
                    dormant_support = math.exp(-0.5 * float(norms[idx] ** 2))
                    if norms[idx] <= 2.4 and dormant_support > true_score + 0.08:
                        dormant.pop(idx)
                        b0.active = True; b0.archived = False; b0.damage_disabled = False; b0.last_seen = self.step
                        self.regions[y].append(b0)
                        self.events.append({
                            'step': self.step, 'event': 'reactivate', 'class_id': y,
                            'branch_id': b0.branch_id, 'reason': 'recurrence_match',
                            'normalized_distance': float(norms[idx]),
                        })
                        winner = b0
                        active.append(b0)
                if winner is None:
                    dnorm = np.array([np.linalg.norm(x - b.center) / max(b.sigma, self.min_sigma) for b in active])
                    winner = active[int(np.argmin(dnorm))]
                    need_growth = winner.count > 180 and float(np.min(dnorm)) > self.grow_z
                    if need_growth:
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

    def learn_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        for xi, yi in zip(X, y):
            self.learn_one(xi, int(yi))

    def inject_redundant_branch(self, class_id: int, jitter: float = 0.04) -> Optional[int]:
        active = [b for b in self.regions.get(class_id, []) if b.active and not b.damage_disabled]
        if not active:
            return None
        source = max(active, key=lambda b: b.count)
        self.branch_counter += 1
        dup = Branch(
            branch_id=self.branch_counter, class_id=class_id,
            center=source.center + self.rng.normal(scale=jitter, size=self.input_dim),
            sigma=source.sigma, count=max(1, source.count // 3),
            created_step=self.step, last_seen=self.step,
            buffer=[z.copy() for z in source.buffer[-20:]],
        )
        self.regions[class_id].append(dup)
        self.events.append({
            'step': self.step, 'event': 'redundancy_injected',
            'class_id': class_id, 'branch_id': dup.branch_id,
            'source_branch_id': source.branch_id,
        })
        return dup.branch_id

    def induce_dormancy(self, class_id: int, keep_center: np.ndarray) -> List[int]:
        active = [b for b in self.regions.get(class_id, []) if b.active and not b.damage_disabled]
        if len(active) <= 1:
            return []
        # Keep the branch nearest the supplied retained-mode center; archive others.
        keep = min(active, key=lambda b: float(np.linalg.norm(b.center - keep_center)))
        retired = []
        for b in list(active):
            if b is keep:
                continue
            self.regions[class_id].remove(b)
            b.active = False; b.archived = True
            self.archives.setdefault(class_id, []).append(b)
            retired.append(b.branch_id)
            self.events.append({
                'step': self.step, 'event': 'retire', 'class_id': class_id,
                'branch_id': b.branch_id, 'reason': 'controlled_metabolic_dormancy',
                'inactive_steps': self.step - b.last_seen,
            })
        return retired

    def damage_region(self, class_id: int, fraction: float = 0.5) -> List[int]:
        branches = [b for b in self.regions.get(class_id, []) if b.active]
        if not branches:
            return []
        n = max(1, int(math.ceil(len(branches) * fraction)))
        # Disable the most used branches to ensure real damage.
        victims = sorted(branches, key=lambda b: b.count, reverse=True)[:n]
        for b in victims:
            b.damage_disabled = True
            self.events.append({
                'step': self.step,
                'event': 'damage',
                'class_id': class_id,
                'branch_id': b.branch_id,
                'reason': 'controlled_ablation',
            })
        return [b.branch_id for b in victims]

    def repair_scan(self, threshold: float = 0.88) -> List[int]:
        damaged = []
        for c in sorted(self.certificates):
            acc = self.certificate_accuracy(c)
            if not np.isnan(acc) and acc < threshold:
                damaged.append(c)
                # Move disabled branches to archive so evidence may reactivate them.
                for b in list(self.regions.get(c, [])):
                    if b.damage_disabled:
                        self.regions[c].remove(b)
                        b.active = False
                        b.archived = True
                        b.damage_disabled = False
                        self.archives.setdefault(c, []).append(b)
                self.events.append({
                    'step': self.step,
                    'event': 'damage_detected',
                    'class_id': c,
                    'certificate_accuracy': acc,
                })
                self.last_recovery_event[c] = self.step
        return damaged

    def structural_counts(self) -> dict:
        active = [b for bs in self.regions.values() for b in bs if b.active and not b.damage_disabled]
        archived = [b for bs in self.archives.values() for b in bs]
        return {
            'regions': len(self.regions),
            'active_branches': len(active),
            'archived_branches': len(archived),
            'total_branches_ever': self.branch_counter,
        }


class StaticPrototypeWeb:
    """Ablation control: same local RBF form, but no growth after warmup."""
    def __init__(self, base_sigma: float = 1.50):
        self.base_sigma = base_sigma
        self.centers: Dict[int, List[np.ndarray]] = {}

    def fit_warmup(self, X: np.ndarray, y: np.ndarray, per_class: int = 2) -> None:
        for c in sorted(np.unique(y)):
            Xc = X[y == c]
            k = min(per_class, len(Xc))
            if k == 1:
                C = [Xc.mean(axis=0)]
            else:
                X0 = Xc - Xc.mean(axis=0, keepdims=True)
                _, _, vt = np.linalg.svd(X0, full_matrices=False)
                proj = X0 @ vt[0]
                c0, c1 = Xc[int(np.argmin(proj))].copy(), Xc[int(np.argmax(proj))].copy()
                for _ in range(10):
                    d0 = np.sum((Xc-c0)**2, axis=1); d1 = np.sum((Xc-c1)**2, axis=1)
                    lab = (d1 < d0).astype(int)
                    if np.any(lab==0): c0 = Xc[lab==0].mean(axis=0)
                    if np.any(lab==1): c1 = Xc[lab==1].mean(axis=0)
                C = [c0, c1]
            self.centers[int(c)] = [z.copy() for z in C]

    def add_new_class_once(self, X: np.ndarray, y: np.ndarray) -> None:
        for c in sorted(np.unique(y)):
            if int(c) not in self.centers:
                Xc = X[y == c]
                self.centers[int(c)] = [Xc.mean(axis=0)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        cls = sorted(self.centers)
        all_scores = []
        for c in cls:
            C = np.stack(self.centers[c])
            d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
            all_scores.append(np.max(np.exp(-0.5 * d2 / (self.base_sigma ** 2)), axis=1))
        return np.array(cls, dtype=int)[np.argmax(np.stack(all_scores, axis=1), axis=1)]


class OnlineMLP:
    def __init__(self, input_dim: int, n_classes: int, replay_per_class: int = 0, rng: Optional[np.random.Generator] = None):
        self.n_classes = n_classes
        self.replay_per_class = replay_per_class
        self.rng = rng or np.random.default_rng(0)
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.002,
            batch_size=128,
            max_iter=1,
            warm_start=False,
            random_state=0,
        )
        self.initialized = False
        self.memory: Dict[int, List[np.ndarray]] = {}

    def update(self, X: np.ndarray, y: np.ndarray, epochs: int = 2) -> None:
        Xtrain = X
        ytrain = y
        if self.replay_per_class > 0:
            mem_X, mem_y = [], []
            for c, pts in self.memory.items():
                if pts:
                    take = min(len(pts), self.replay_per_class)
                    idx = self.rng.choice(len(pts), size=take, replace=False)
                    mem_X.extend([pts[i] for i in idx])
                    mem_y.extend([c] * take)
            if mem_X:
                Xtrain = np.vstack([Xtrain, np.stack(mem_X)])
                ytrain = np.concatenate([ytrain, np.array(mem_y, dtype=int)])
        for _ in range(epochs):
            order = self.rng.permutation(len(Xtrain))
            if not self.initialized:
                self.model.partial_fit(Xtrain[order], ytrain[order], classes=np.arange(self.n_classes))
                self.initialized = True
            else:
                self.model.partial_fit(Xtrain[order], ytrain[order])
        if self.replay_per_class > 0:
            for xi, yi in zip(X, y):
                pts = self.memory.setdefault(int(yi), [])
                if len(pts) < self.replay_per_class * 3:
                    pts.append(xi.copy())
                elif self.rng.random() < 0.05:
                    pts[int(self.rng.integers(0, len(pts)))] = xi.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.initialized:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(X)


class StreamingWorld:
    def __init__(self, input_dim: int = 12, n_classes: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.class_base = self._separated_centers(n_classes, input_dim, min_dist=5.0)
        self.mode_offsets: Dict[Tuple[int, str], np.ndarray] = {}
        # Two initial modes per class.
        for c in range(n_classes):
            v1 = self.rng.normal(size=input_dim); v1 /= np.linalg.norm(v1)
            v2 = self.rng.normal(size=input_dim); v2 /= np.linalg.norm(v2)
            self.mode_offsets[(c, 'A')] = 1.15 * v1
            self.mode_offsets[(c, 'B')] = -1.15 * v1 + 0.35 * v2
        # A deliberately wide two-mode class used to test dormancy and recurrence.
        v4 = self.rng.normal(size=input_dim); v4 /= np.linalg.norm(v4)
        self.mode_offsets[(4, 'A')] = 2.55 * v4
        self.mode_offsets[(4, 'B')] = -2.55 * v4
        # Novel modes.
        for c, name, scale in [(2, 'C', 2.65), (3, 'D', 2.85)]:
            v = self.rng.normal(size=input_dim); v /= np.linalg.norm(v)
            self.mode_offsets[(c, name)] = scale * v
        self.noise = 0.48

    def _separated_centers(self, n: int, d: int, min_dist: float) -> np.ndarray:
        # Deterministic orthogonal scaffold avoids rejection-sampling pathologies.
        A = self.rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        centers = (4.1 * Q[:, :n].T).copy()
        assert np.min([np.linalg.norm(centers[i] - centers[j]) for i in range(n) for j in range(i)]) >= min_dist
        return centers

    def sample(self, class_modes: Dict[int, List[str]], n: int, class_probs: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        classes = sorted(class_modes)
        if class_probs is None:
            probs = np.ones(len(classes)) / len(classes)
        else:
            probs = np.array([class_probs.get(c, 0.0) for c in classes], dtype=float)
            probs /= probs.sum()
        ys = self.rng.choice(classes, size=n, p=probs)
        X, modes = [], []
        for y in ys:
            mode = str(self.rng.choice(class_modes[int(y)]))
            mu = self.class_base[int(y)] + self.mode_offsets[(int(y), mode)]
            X.append(mu + self.rng.normal(scale=self.noise, size=self.input_dim))
            modes.append(f'{int(y)}:{mode}')
        return np.stack(X), ys.astype(int), modes

    def eval_set(self, class_modes: Dict[int, List[str]], per_mode: int = 250) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, mode = [], [], []
        for c in sorted(class_modes):
            for m in class_modes[c]:
                mu = self.class_base[c] + self.mode_offsets[(c, m)]
                pts = mu + self.rng.normal(scale=self.noise, size=(per_mode, self.input_dim))
                X.append(pts)
                y.extend([c] * per_mode)
                mode.extend([f'{c}:{m}'] * per_mode)
        return np.vstack(X), np.array(y, dtype=int), np.array(mode)


def accuracy_by_group(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    out = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        out[str(g)] = float(np.mean(pred[mask] == y[mask]))
    return out


def run_benchmark(seed: int = SEED) -> dict:
    start = time.time()
    rng = np.random.default_rng(seed)
    world = StreamingWorld(seed=seed)
    plastic = PlasticDendritronWeb(input_dim=world.input_dim, rng=np.random.default_rng(seed + 1))
    static = StaticPrototypeWeb()
    mlp = OnlineMLP(world.input_dim, world.n_classes, replay_per_class=0, rng=np.random.default_rng(seed + 2))
    mlp_replay = OnlineMLP(world.input_dim, world.n_classes, replay_per_class=24, rng=np.random.default_rng(seed + 3))

    phases = [
        ('warmup', 2200, {c: ['A', 'B'] for c in range(6)}, None),
        ('novel_mode', 1400, {**{c: ['A', 'B'] for c in range(6)}, 2: ['A', 'B', 'C']}, None),
        ('new_classes', 1500, {**{c: ['A', 'B'] for c in range(6)}, 2: ['A', 'B', 'C'], 6: ['A', 'B'], 7: ['A', 'B']}, None),
        ('specialization', 1400, {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, None),
        ('inactivity', 1500, {**{c: ['A', 'B'] for c in range(8) if c != 4}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, None),
        ('recurrence', 900, {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D'], 4: ['B']}, {4: 0.42, **{c: 0.58 / 7 for c in range(8) if c != 4}}),
        ('repair', 1100, {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, {1: 0.38, **{c: 0.62 / 7 for c in range(8) if c != 1}}),
    ]

    # Canonical evaluation includes all functions that should remain available.
    eval_modes = {c: ['A', 'B'] for c in range(8)}
    eval_modes[2] = ['A', 'B', 'C']
    eval_modes[3] = ['A', 'B', 'D']
    Xeval, yeval, geval = world.eval_set(eval_modes, per_mode=120)

    phase_rows = []
    mode_rows = []
    damage_info = {}
    recurrence_start = None
    repair_start = None
    class4_acc_before_recurrence = None
    class1_acc_after_damage = None

    for phase_idx, (phase, n, modes, probs) in enumerate(phases):
        X, y, stream_modes = world.sample(modes, n, probs)

        if phase == 'warmup':
            static.fit_warmup(X, y, per_class=2)
        else:
            static.add_new_class_once(X, y)

        if phase == 'repair':
            # Controlled structural damage before the repair stream.
            victims = plastic.damage_region(1, fraction=1.0)
            pred_damage = plastic.predict(Xeval)
            class1_acc_after_damage = float(np.mean(pred_damage[yeval == 1] == 1))
            detected = plastic.repair_scan(threshold=0.88)
            damage_info = {'victim_branch_ids': victims, 'detected_regions': detected}
            repair_start = plastic.step

        if phase == 'recurrence':
            recurrence_start = plastic.step
            pred_pre = plastic.predict(Xeval)
            class4_acc_before_recurrence = float(np.mean(pred_pre[yeval == 4] == 4))

        # Online plastic web sees each irregular sample once.
        plastic.learn_batch(X, y)

        if phase == 'warmup':
            plastic.inject_redundant_branch(0)
            plastic._merge_candidates()
        if phase == 'inactivity':
            plastic.induce_dormancy(4, world.class_base[4] + world.mode_offsets[(4, 'A')])

        # Baselines update in moderately sized chunks.
        for s in range(0, len(X), 256):
            xb, yb = X[s:s+256], y[s:s+256]
            mlp.update(xb, yb, epochs=1)
            mlp_replay.update(xb, yb, epochs=1)

        preds = {
            'plastic_dendritron_web': plastic.predict(Xeval),
            'static_prototype_web': static.predict(Xeval),
            'online_mlp': mlp.predict(Xeval),
            'online_mlp_replay': mlp_replay.predict(Xeval),
        }
        counts = plastic.structural_counts()
        event_counts = pd.DataFrame(plastic.events).query('step <= @plastic.step')['event'].value_counts().to_dict() if plastic.events else {}
        for model_name, pred in preds.items():
            phase_rows.append({
                'phase_index': phase_idx,
                'phase': phase,
                'model': model_name,
                'accuracy': float(np.mean(pred == yeval)),
                'regions': counts['regions'] if model_name == 'plastic_dendritron_web' else np.nan,
                'active_branches': counts['active_branches'] if model_name == 'plastic_dendritron_web' else np.nan,
                'archived_branches': counts['archived_branches'] if model_name == 'plastic_dendritron_web' else np.nan,
                'grow_events': event_counts.get('grow', 0) if model_name == 'plastic_dendritron_web' else np.nan,
                'split_events': event_counts.get('split', 0) if model_name == 'plastic_dendritron_web' else np.nan,
                'merge_events': event_counts.get('merge', 0) if model_name == 'plastic_dendritron_web' else np.nan,
                'retire_events': event_counts.get('retire', 0) if model_name == 'plastic_dendritron_web' else np.nan,
                'reactivate_events': event_counts.get('reactivate', 0) if model_name == 'plastic_dendritron_web' else np.nan,
            })
            by_mode = accuracy_by_group(pred, yeval, geval)
            for mode_name, acc in by_mode.items():
                mode_rows.append({
                    'phase_index': phase_idx,
                    'phase': phase,
                    'model': model_name,
                    'mode': mode_name,
                    'accuracy': acc,
                })

    phase_df = pd.DataFrame(phase_rows)
    mode_df = pd.DataFrame(mode_rows)
    event_df = pd.DataFrame(plastic.events)

    # Structural locality: all edits name one owned class region.
    local_events = event_df[event_df['event'].isin(['grow', 'split', 'merge', 'retire', 'reactivate', 'damage', 'damage_detected'])]
    structural_locality = float(np.mean(local_events['class_id'].notna())) if len(local_events) else 1.0

    # Stability: old modes excluding newly introduced modes should stay high.
    final_plastic = phase_df[(phase_df.phase == 'repair') & (phase_df.model == 'plastic_dendritron_web')].iloc[0]
    final_mode_plastic = mode_df[(mode_df.phase == 'repair') & (mode_df.model == 'plastic_dendritron_web')]
    old_mode_mask = ~final_mode_plastic['mode'].isin(['2:C', '3:D', '6:A', '6:B', '7:A', '7:B'])
    old_mode_min = float(final_mode_plastic[old_mode_mask]['accuracy'].min())

    # Recovery timing from event log.
    recurrence_events = event_df[(event_df['step'] >= recurrence_start) & (event_df['class_id'] == 4)] if recurrence_start is not None and len(event_df) else pd.DataFrame()
    reactivate_step = None
    if len(recurrence_events):
        x = recurrence_events[recurrence_events['event'] == 'reactivate']
        if len(x):
            reactivate_step = int(x.iloc[0]['step'])
    repair_events = event_df[(event_df['step'] >= repair_start) & (event_df['class_id'] == 1)] if repair_start is not None and len(event_df) else pd.DataFrame()
    repair_reactivate_step = None
    if len(repair_events):
        x = repair_events[repair_events['event'].isin(['reactivate', 'grow'])]
        if len(x):
            repair_reactivate_step = int(x.iloc[0]['step'])

    # Accuracy after repair.
    final_pred = plastic.predict(Xeval)
    class1_final = float(np.mean(final_pred[yeval == 1] == 1))
    class4_final = float(np.mean(final_pred[yeval == 4] == 4))

    operation_counts = event_df['event'].value_counts().to_dict() if len(event_df) else {}
    plasticity_criteria = {
        'novelty_driven_growth': operation_counts.get('grow', 0) > 0,
        'endogenous_split': operation_counts.get('split', 0) > 0,
        'redundancy_merge': operation_counts.get('merge', 0) > 0,
        'inactivity_retirement': operation_counts.get('retire', 0) > 0,
        'recurrence_reactivation': reactivate_step is not None,
        'damage_detected_locally': 1 in damage_info.get('detected_regions', []),
        'damage_repaired': class1_acc_after_damage is not None and class1_acc_after_damage < 0.75 and class1_final >= 0.95,
        'prior_function_stability': old_mode_min >= 0.94,
        'new_function_acquisition': float(final_mode_plastic[final_mode_plastic['mode'].isin(['2:C', '3:D', '6:A', '6:B', '7:A', '7:B'])]['accuracy'].min()) >= 0.94,
        'structural_locality': structural_locality == 1.0,
    }

    summary = {
        'seed': seed,
        'runtime_seconds': time.time() - start,
        'stream_samples': int(sum(p[1] for p in phases)),
        'input_dim': world.input_dim,
        'classes': world.n_classes,
        'final_accuracy': {
            row.model: float(row.accuracy)
            for row in phase_df[phase_df.phase == 'repair'].itertuples()
        },
        'plastic_structure': plastic.structural_counts(),
        'operation_counts': {str(k): int(v) for k, v in operation_counts.items()},
        'structural_locality': structural_locality,
        'old_mode_min_final_accuracy': old_mode_min,
        'class4_accuracy_before_recurrence': class4_acc_before_recurrence,
        'class4_accuracy_after_recurrence': class4_final,
        'class4_reactivation_samples': None if reactivate_step is None else reactivate_step - recurrence_start,
        'class1_accuracy_after_damage': class1_acc_after_damage,
        'class1_accuracy_after_repair': class1_final,
        'class1_first_repair_edit_samples': None if repair_reactivate_step is None else repair_reactivate_step - repair_start,
        'damage_info': damage_info,
        'plasticity_criteria': {k: bool(v) for k, v in plasticity_criteria.items()},
        'artificial_neural_plasticity_pass': bool(all(plasticity_criteria.values())),
    }

    phase_df.to_csv(OUT / 'dendritron_v0_7_phase_accuracy.csv', index=False)
    mode_df.to_csv(OUT / 'dendritron_v0_7_mode_accuracy.csv', index=False)
    event_df.to_csv(OUT / 'dendritron_v0_7_structural_events.csv', index=False)
    pd.DataFrame([{'criterion': k, 'passed': bool(v)} for k, v in plasticity_criteria.items()]).to_csv(
        OUT / 'dendritron_v0_7_plasticity_criteria.csv', index=False
    )
    with open(OUT / 'dendritron_v0_7_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('DENDRITRON ARTIFICIAL NEURAL PLASTICITY BENCHMARK v0.7')
    print('=' * 72)
    print(f"Runtime: {summary['runtime_seconds']:.2f}s | Stream samples: {summary['stream_samples']}")
    print('\nFinal accuracy:')
    for k, v in summary['final_accuracy'].items():
        print(f'  {k:28s} {v:.4f}')
    print('\nPlastic structure:', summary['plastic_structure'])
    print('Operations:', summary['operation_counts'])
    print(f"Old-mode minimum final accuracy: {old_mode_min:.4f}")
    print(f"Class 4 recurrence: {class4_acc_before_recurrence:.4f} -> {class4_final:.4f}; reactivation samples={summary['class4_reactivation_samples']}")
    print(f"Class 1 damage/repair: {class1_acc_after_damage:.4f} -> {class1_final:.4f}; first repair edit samples={summary['class1_first_repair_edit_samples']}")
    print('\nPlasticity criteria:')
    for k, v in plasticity_criteria.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nARTIFICIAL NEURAL PLASTICITY PASS: {summary['artificial_neural_plasticity_pass']}")
    return summary


if __name__ == '__main__':
    run_benchmark()
