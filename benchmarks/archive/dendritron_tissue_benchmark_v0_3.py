"""
Dendritron Tissue Benchmark v0.3

Adds continuous inputs, learned local RBF branches, semantic routing without a
task ID, quarantine activation, conflict detection, and local repair.

This is a synthetic architecture test, not a real-world superiority claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
from torch import nn

SEED = 7
OUTPUT_DIR = Path(__file__).resolve().parent
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


@dataclass
class RBFBranch:
    center: np.ndarray


class ContinuousDendritronRegion:
    """One class-owned Dendritron region with locally learned RBF branches."""

    def __init__(
        self,
        label: int,
        branches: List[RBFBranch],
        scale: float,
        certificate_x: np.ndarray,
        *,
        name: str | None = None,
    ) -> None:
        self.label = int(label)
        self.branches = branches
        self.scale = float(max(scale, 1e-8))
        self.certificate_x = np.asarray(certificate_x, dtype=np.float64).copy()
        self.name = name or f"region_{label}"
        self.enabled = True

    @classmethod
    def fit(
        cls,
        label: int,
        x: np.ndarray,
        *,
        n_branches: int = 2,
        certificate_size: int = 64,
        seed: int = SEED,
    ) -> "ContinuousDendritronRegion":
        x = np.asarray(x, dtype=np.float64)
        if len(x) < n_branches:
            raise ValueError("Need at least as many samples as branches")
        kmeans = KMeans(n_clusters=n_branches, n_init=10, random_state=seed + label)
        kmeans.fit(x)
        centers = kmeans.cluster_centers_
        distances = np.sqrt(
            np.min(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1), axis=1)
        )
        scale = float(np.quantile(distances, 0.95) + 1e-6)
        cert = x[: min(certificate_size, len(x))]
        return cls(
            label,
            [RBFBranch(center=c.copy()) for c in centers],
            scale,
            cert,
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self.enabled:
            return np.full(len(x), -np.inf)
        centers = np.stack([branch.center for branch in self.branches])
        distance = np.sqrt(
            np.min(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1), axis=1)
        )
        # A normalized local-support score. Larger is better.
        return -distance / self.scale

    @property
    def inference_scalar_equivalents(self) -> int:
        if not self.branches:
            return 1
        return len(self.branches) * len(self.branches[0].center) + 1

    @property
    def certificate_scalar_equivalents(self) -> int:
        return int(np.prod(self.certificate_x.shape))

    def signature(self) -> str:
        payload = {
            "label": self.label,
            "scale": self.scale,
            "enabled": self.enabled,
            "centers": [branch.center.tolist() for branch in self.branches],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def clone(self) -> "ContinuousDendritronRegion":
        clone = ContinuousDendritronRegion(
            self.label,
            [RBFBranch(b.center.copy()) for b in self.branches],
            self.scale,
            self.certificate_x.copy(),
            name=self.name,
        )
        clone.enabled = self.enabled
        return clone

    def repair_from_certificate(self, n_branches: int = 2) -> None:
        repaired = ContinuousDendritronRegion.fit(
            self.label,
            self.certificate_x,
            n_branches=n_branches,
            certificate_size=len(self.certificate_x),
        )
        self.branches = repaired.branches
        self.scale = repaired.scale
        self.enabled = True


class QuarantinedSemanticRouter:
    """Semantic router with certificate-protected activation.

    New regions are trained in quarantine. They may enter the active web only if
    adding them changes zero predictions on the protected certificates of all
    previously active regions.
    """

    def __init__(self) -> None:
        self.regions: Dict[int, ContinuousDendritronRegion] = {}
        self.rejected: Dict[int, str] = {}

    def score_matrix(
        self,
        x: np.ndarray,
        extra: ContinuousDendritronRegion | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = sorted(self.regions)
        regions = [self.regions[label] for label in labels]
        if extra is not None:
            labels = labels + [extra.label]
            regions = regions + [extra]
        if not regions:
            return np.empty((len(x), 0)), np.array([], dtype=np.int64)
        scores = np.stack([region.score(x) for region in regions], axis=1)
        return scores, np.asarray(labels, dtype=np.int64)

    def predict(
        self,
        x: np.ndarray,
        extra: ContinuousDendritronRegion | None = None,
    ) -> np.ndarray:
        scores, labels = self.score_matrix(x, extra=extra)
        if scores.shape[1] == 0:
            return np.full(len(x), -1, dtype=np.int64)
        return labels[np.argmax(scores, axis=1)]

    def protected_certificate_set(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.regions:
            return np.empty((0, 0)), np.empty(0, dtype=np.int64)
        x = np.concatenate([r.certificate_x for r in self.regions.values()], axis=0)
        y = np.concatenate(
            [np.full(len(r.certificate_x), label, dtype=np.int64) for label, r in self.regions.items()]
        )
        return x, y

    def propose(self, candidate: ContinuousDendritronRegion) -> dict:
        if candidate.label in self.regions:
            raise ValueError("Label already active")
        if not self.regions:
            self.regions[candidate.label] = candidate
            return {
                "label": candidate.label,
                "accepted": True,
                "protected_predictions_changed": 0,
                "protected_accuracy_before": 1.0,
                "protected_accuracy_after": 1.0,
            }

        cert_x, cert_y = self.protected_certificate_set()
        before = self.predict(cert_x)
        after = self.predict(cert_x, extra=candidate)
        changed = int(np.sum(before != after))
        before_acc = float(np.mean(before == cert_y))
        after_acc = float(np.mean(after == cert_y))
        accepted = changed == 0
        if accepted:
            self.regions[candidate.label] = candidate
        else:
            self.rejected[candidate.label] = (
                f"Rejected: candidate changed {changed} protected old predictions"
            )
        return {
            "label": candidate.label,
            "accepted": accepted,
            "protected_predictions_changed": changed,
            "protected_accuracy_before": before_acc,
            "protected_accuracy_after": after_acc,
        }

    def inference_scalar_equivalents(self) -> int:
        return sum(r.inference_scalar_equivalents for r in self.regions.values())

    def certificate_scalar_equivalents(self) -> int:
        return sum(r.certificate_scalar_equivalents for r in self.regions.values())


class ClassIncrementalMLP(nn.Module):
    def __init__(self, input_dim: int, class_count: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_continuous_mixture_data(
    rng: np.random.Generator,
    *,
    class_count: int = 10,
    input_dim: int = 8,
    train_per_class: int = 150,
    test_per_class: int = 200,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    raw = rng.normal(size=(input_dim, input_dim))
    basis, _ = np.linalg.qr(raw)
    centers = []
    for label in range(class_count):
        if label < input_dim:
            center = basis[:, label] * 6.0
        else:
            center = (basis[:, label % input_dim] + basis[:, (label + 3) % input_dim]) * 4.2
        centers.append(center)
    centers = np.asarray(centers)

    mode_offsets = rng.normal(size=(class_count, input_dim))
    mode_offsets /= np.linalg.norm(mode_offsets, axis=1, keepdims=True)
    mode_offsets *= 1.2

    train, test = [], []
    for label in range(class_count):
        signs = rng.choice([-1.0, 1.0], size=train_per_class)
        train_x = (
            centers[label]
            + signs[:, None] * mode_offsets[label]
            + rng.normal(scale=0.45, size=(train_per_class, input_dim))
        )
        signs = rng.choice([-1.0, 1.0], size=test_per_class)
        test_x = (
            centers[label]
            + signs[:, None] * mode_offsets[label]
            + rng.normal(scale=0.45, size=(test_per_class, input_dim))
        )
        train.append(train_x.astype(np.float64))
        test.append(test_x.astype(np.float64))
    return train, test, centers, mode_offsets


def evaluate_router(router: QuarantinedSemanticRouter, test: List[np.ndarray]) -> dict:
    labels = sorted(router.regions)
    x = np.concatenate([test[label] for label in labels], axis=0)
    y = np.concatenate([np.full(len(test[label]), label) for label in labels])
    prediction = router.predict(x)
    per_class = {
        label: float(np.mean(router.predict(test[label]) == label)) for label in labels
    }
    return {
        "mean_accuracy": float(np.mean(prediction == y)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class": per_class,
    }


def train_mlp_sequence(
    train: List[np.ndarray],
    test: List[np.ndarray],
    *,
    replay_per_class: int = 0,
) -> Tuple[pd.DataFrame, int, int]:
    torch.manual_seed(SEED + replay_per_class)
    input_dim = train[0].shape[1]
    class_count = len(train)
    model = ClassIncrementalMLP(input_dim, class_count, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    memory_x: List[np.ndarray] = []
    memory_y: List[np.ndarray] = []
    rows = []

    for learned_class in range(class_count):
        x_parts = [train[learned_class]]
        y_parts = [np.full(len(train[learned_class]), learned_class)]
        if replay_per_class > 0 and memory_x:
            x_parts.extend(memory_x)
            y_parts.extend(memory_y)
        x_np = np.concatenate(x_parts)
        y_np = np.concatenate(y_parts).astype(np.int64)
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        for epoch in range(160):
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()
            current_prediction = logits.detach().argmax(dim=1).cpu().numpy()
            if epoch % 10 == 0 and np.mean(current_prediction == y_np) == 1.0:
                break

        if replay_per_class > 0:
            memory_x.append(train[learned_class][:replay_per_class])
            memory_y.append(np.full(replay_per_class, learned_class))

        with torch.no_grad():
            for evaluated_class in range(class_count):
                x_eval = torch.tensor(test[evaluated_class], dtype=torch.float32)
                prediction = model(x_eval).argmax(dim=1).cpu().numpy()
                rows.append(
                    {
                        "model": f"mlp_replay_{replay_per_class}",
                        "after_learning_class": learned_class,
                        "evaluated_class": evaluated_class,
                        "accuracy": float(np.mean(prediction == evaluated_class)),
                    }
                )

    parameters = sum(p.numel() for p in model.parameters())
    memory_scalars = replay_per_class * class_count * (input_dim + 1)
    return pd.DataFrame(rows), int(parameters), int(memory_scalars)


def sequential_summary(df: pd.DataFrame, class_count: int) -> dict:
    matrix = df.pivot(
        index="after_learning_class", columns="evaluated_class", values="accuracy"
    ).to_numpy()
    acquisition = np.array([matrix[i, i] for i in range(class_count)])
    final = matrix[-1]
    forgetting = np.array(
        [np.max(matrix[i:, i]) - final[i] for i in range(class_count - 1)]
    )
    return {
        "mean_acquisition_accuracy": float(np.mean(acquisition)),
        "mean_final_accuracy": float(np.mean(final)),
        "min_final_accuracy": float(np.min(final)),
        "mean_forgetting": float(np.mean(forgetting)),
        "max_forgetting": float(np.max(forgetting)),
    }


def run_benchmark() -> Tuple[dict, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(SEED)
    class_count = 10
    train, test, centers, mode_offsets = make_continuous_mixture_data(
        rng, class_count=class_count
    )

    router = QuarantinedSemanticRouter()
    activation_rows = []
    sequential_rows = []
    for label in range(class_count):
        candidate = ContinuousDendritronRegion.fit(label, train[label], n_branches=2)
        result = router.propose(candidate)
        activation_rows.append(result)
        evaluation = evaluate_router(router, test)
        sequential_rows.append(
            {
                "after_learning_class": label,
                "active_regions": len(router.regions),
                "mean_accuracy": evaluation["mean_accuracy"],
                "min_class_accuracy": evaluation["min_class_accuracy"],
                "protected_predictions_changed": result["protected_predictions_changed"],
            }
        )

    final_router_eval = evaluate_router(router, test)

    # Conflict test: same evidence as class 0, relabeled as a new class. Increase
    # support scale so it would steal old certificates if activated.
    impossible_candidate = router.regions[0].clone()
    impossible_candidate.label = class_count
    impossible_candidate.name = "impossible_relabel"
    impossible_candidate.scale *= 2.0
    conflict_result = router.propose(impossible_candidate)

    # Resolved candidate: similar first seven dimensions to class 0 but with a
    # strong distinguishing eighth dimension. This represents new evidence that
    # makes ownership separable.
    resolved_label = class_count + 1
    resolved_train = train[0].copy()
    resolved_test = test[0].copy()
    resolved_train[:, -1] += 8.0
    resolved_test[:, -1] += 8.0
    resolved_candidate = ContinuousDendritronRegion.fit(
        resolved_label, resolved_train, n_branches=2
    )
    resolved_result = router.propose(resolved_candidate)
    resolved_accuracy = float(np.mean(router.predict(resolved_test) == resolved_label))
    base_x_after_resolved = np.concatenate(test, axis=0)
    base_y_after_resolved = np.concatenate([
        np.full(len(test[label]), label) for label in range(class_count)
    ])
    old_accuracy_after_resolved = float(
        np.mean(router.predict(base_x_after_resolved) == base_y_after_resolved)
    )

    # Local damage/repair.
    damage_label = 4
    signatures_before = {label: region.signature() for label, region in router.regions.items()}
    before_damage = float(np.mean(router.predict(test[damage_label]) == damage_label))
    router.regions[damage_label].enabled = False
    after_damage = float(np.mean(router.predict(test[damage_label]) == damage_label))
    failed_regions = [
        label
        for label, region in router.regions.items()
        if float(np.mean(router.predict(region.certificate_x) == label)) < 1.0
    ]
    signatures_damaged = {label: region.signature() for label, region in router.regions.items()}
    changed_damage = [
        label for label in signatures_before if signatures_before[label] != signatures_damaged[label]
    ]
    router.regions[damage_label].repair_from_certificate(n_branches=2)
    after_repair = float(np.mean(router.predict(test[damage_label]) == damage_label))
    signatures_repaired = {label: region.signature() for label, region in router.regions.items()}
    non_target_changes = [
        label
        for label in signatures_before
        if label != damage_label and signatures_before[label] != signatures_repaired[label]
    ]

    # MLP controls.
    no_replay_df, no_replay_params, no_replay_memory = train_mlp_sequence(
        train, test, replay_per_class=0
    )
    replay_df, replay_params, replay_memory = train_mlp_sequence(
        train, test, replay_per_class=20
    )

    summary = {
        "benchmark": "Dendritron Tissue Benchmark v0.3",
        "continuous_input_dim": train[0].shape[1],
        "base_classes": class_count,
        "branches_per_region": 2,
        "semantic_router_uses_task_id": False,
        "all_base_regions_accepted": bool(all(row["accepted"] for row in activation_rows)),
        "base_growth_protected_prediction_changes": int(
            sum(row["protected_predictions_changed"] for row in activation_rows)
        ),
        "dendritron_final_mean_accuracy": final_router_eval["mean_accuracy"],
        "dendritron_final_min_class_accuracy": final_router_eval["min_class_accuracy"],
        "dendritron_inference_scalar_equivalents": router.inference_scalar_equivalents(),
        "dendritron_certificate_scalar_equivalents": router.certificate_scalar_equivalents(),
        "conflicting_candidate_accepted": conflict_result["accepted"],
        "conflicting_candidate_protected_predictions_changed": conflict_result[
            "protected_predictions_changed"
        ],
        "resolved_candidate_accepted": resolved_result["accepted"],
        "resolved_candidate_accuracy": resolved_accuracy,
        "old_base_accuracy_after_resolved_candidate": old_accuracy_after_resolved,
        "damage_label": damage_label,
        "damage_accuracy_before": before_damage,
        "damage_accuracy_after": after_damage,
        "failed_regions_detected": failed_regions,
        "regions_changed_during_damage": changed_damage,
        "repair_accuracy": after_repair,
        "non_target_regions_changed_after_repair": non_target_changes,
        "mlp_no_replay_parameters": no_replay_params,
        "mlp_no_replay_memory_scalars": no_replay_memory,
        "mlp_no_replay": sequential_summary(no_replay_df, class_count),
        "mlp_replay_parameters": replay_params,
        "mlp_replay_memory_scalars": replay_memory,
        "mlp_replay": sequential_summary(replay_df, class_count),
        "limitations": [
            "The continuous data are synthetic and deliberately clusterable.",
            "Certificate protection guarantees only the retained certificate set, not all possible old inputs.",
            "The semantic router is nearest normalized local support, not a fully learned graph router.",
            "The RBF region is one Dendritron variation, not the canonical definition of the primitive.",
            "Replay can match retention by storing and retraining on old examples; the distinction is where ownership and adaptation live.",
        ],
    }

    frames = {
        "activation": pd.DataFrame(activation_rows),
        "dendritron_sequence": pd.DataFrame(sequential_rows),
        "mlp_no_replay": no_replay_df,
        "mlp_replay": replay_df,
    }
    return summary, frames


def save(summary: dict, frames: Dict[str, pd.DataFrame]) -> None:
    with open(OUTPUT_DIR / "dendritron_v0_3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / f"dendritron_v0_3_{name}.csv", index=False)


def main() -> None:
    summary, frames = run_benchmark()
    save(summary, frames)
    print("\nDENDRITRON TISSUE BENCHMARK v0.3")
    print("=" * 76)
    print(json.dumps(summary, indent=2))
    print("\nDENDRITRON SEQUENCE")
    print(frames["dendritron_sequence"].to_string(index=False))


if __name__ == "__main__":
    main()
