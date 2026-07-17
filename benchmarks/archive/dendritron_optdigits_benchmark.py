import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
try:
    from caas_jupyter_tools import display_dataframe_to_user
except ImportError:
    def display_dataframe_to_user(name, dataframe):
        print(f"\n{name}:\n{dataframe.to_string(index=False)}")


# ============================================================
# Configuration
# ============================================================

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DENDRITRON_OUTPUT_DIR") or _HERE.parent / "results" / "local")
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
TEST_SIZE = 0.30
EPOCHS_PER_TASK = 25
PROTOTYPES_PER_CLASS = 16
TOTAL_PROTOTYPES = PROTOTYPES_PER_CLASS * 10
MLP_HIDDEN = 136
REPLAY_PER_CLASS = 40
REPAIR_PER_CLASS = 40

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
TASK_NAMES = ["0/1", "2/3", "4/5", "6/7", "8/9"]

# Learned scalar counts:
# Dendritron / fixed prototype:
#   160 branch centers * 64 pixels = 10,240
# Backprop MLP:
#   64*136 + 136 + 136*10 + 10 = 10,210
DENDRITRON_PARAMETERS = TOTAL_PROTOTYPES * 64
MLP_PARAMETERS = 64 * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN * 10 + 10


# ============================================================
# Dataset
# ============================================================

digits = load_digits()
ALL_X = digits.data.astype(np.float32) / 16.0
ALL_Y = digits.target.astype(np.int64)


def make_split(seed):
    x_train, x_test, y_train, y_test = train_test_split(
        ALL_X,
        ALL_Y,
        test_size=TEST_SIZE,
        stratify=ALL_Y,
        random_state=seed,
    )

    train_tasks = []
    for task in TASKS:
        mask = np.isin(y_train, task)
        train_tasks.append((x_train[mask], y_train[mask]))

    return x_train, x_test, y_train, y_test, train_tasks


def kmeans_plus_plus(x, k, rng):
    """Label-local initialization; no gradients."""
    first = int(rng.integers(len(x)))
    chosen = [first]
    nearest_distance = np.sum((x - x[first]) ** 2, axis=1)

    while len(chosen) < k:
        total = float(nearest_distance.sum())
        if total <= 1e-12:
            candidate = int(rng.integers(len(x)))
        else:
            candidate = int(
                rng.choice(len(x), p=nearest_distance / total)
            )
        chosen.append(candidate)
        candidate_distance = np.sum(
            (x - x[candidate]) ** 2,
            axis=1,
        )
        nearest_distance = np.minimum(
            nearest_distance,
            candidate_distance,
        )

    return x[chosen].copy()


def accuracy(model, x, y, seen_classes):
    return float(
        np.mean(model.predict(x, seen_classes) == y)
    )


# ============================================================
# Locally adaptive Dendritron
# ============================================================

class DendritronDigits:
    """
    Class-local dendritic branches.

    Each branch contains a prototype center. For an observed class:
      1. only branches assigned to that class compete;
      2. the winning branch moves toward the input;
      3. no derivative is sent through another branch or layer.

    At inference the soma performs winner-take-all arbitration over all
    active branches belonging to seen classes.
    """

    def __init__(
        self,
        prototypes_per_class=16,
        learning_rate=0.18,
        seed=0,
    ):
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.centers = np.zeros(
            (10, prototypes_per_class, 64),
            dtype=np.float32,
        )
        self.initialized = np.zeros(10, dtype=bool)
        self.active = np.ones(
            (10, prototypes_per_class),
            dtype=bool,
        )
        self.usage = np.zeros(
            (10, prototypes_per_class),
            dtype=np.int64,
        )

    @property
    def parameter_count(self):
        return self.centers.size

    def _initialize_class(self, x_class, class_id):
        if self.initialized[class_id]:
            return

        self.centers[class_id] = kmeans_plus_plus(
            x_class,
            self.prototypes_per_class,
            self.rng,
        )
        self.initialized[class_id] = True

    def train_epoch(self, x, y):
        for class_id in np.unique(y):
            x_class = x[y == class_id]
            self._initialize_class(x_class, int(class_id))

        for index in self.rng.permutation(len(x)):
            class_id = int(y[index])
            class_centers = self.centers[class_id]

            distances = np.sum(
                (class_centers - x[index]) ** 2,
                axis=1,
            )
            distances[~self.active[class_id]] = np.inf
            winner = int(np.argmin(distances))

            self.usage[class_id, winner] += 1
            local_rate = self.learning_rate / (
                1.0 + 0.002 * self.usage[class_id, winner]
            )

            # Branch-local winner update.
            self.centers[class_id, winner] += (
                local_rate
                * (x[index] - self.centers[class_id, winner])
            )

    def predict(self, x, seen_classes):
        seen_classes = np.asarray(seen_classes, dtype=int)
        class_scores = []

        for class_id in seen_classes:
            distances = np.sum(
                (
                    x[:, None, :]
                    - self.centers[class_id][None, :, :]
                ) ** 2,
                axis=-1,
            )
            distances[:, ~self.active[class_id]] = np.inf
            class_scores.append(np.min(distances, axis=1))

        score_matrix = np.stack(class_scores, axis=1)
        return seen_classes[np.argmin(score_matrix, axis=1)]

    def damage(self, fraction=0.25):
        number = int(round(self.centers.shape[0]
                           * self.centers.shape[1]
                           * fraction))
        flat_usage = self.usage.reshape(-1)
        damaged_flat = np.argsort(flat_usage)[-number:]

        damaged = np.column_stack(
            np.unravel_index(
                damaged_flat,
                self.usage.shape,
            )
        )

        for class_id, branch_id in damaged:
            self.active[class_id, branch_id] = False

        return damaged

    def repair(self, x, y, damaged, epochs=8):
        """
        Rebuild only damaged branches. All surviving branch centers remain
        frozen.
        """
        for class_id in np.unique(damaged[:, 0]):
            class_id = int(class_id)
            damaged_ids = damaged[
                damaged[:, 0] == class_id, 1
            ]
            x_class = x[y == class_id]

            if len(x_class) == 0:
                continue

            surviving_ids = np.where(
                self.active[class_id]
            )[0]
            replacement_centers = []

            for branch_id in damaged_ids:
                reference_sets = []

                if len(surviving_ids):
                    reference_sets.append(
                        self.centers[class_id, surviving_ids]
                    )

                if replacement_centers:
                    reference_sets.append(
                        np.asarray(replacement_centers)
                    )

                if reference_sets:
                    references = np.vstack(reference_sets)
                    distances = np.sum(
                        (
                            x_class[:, None, :]
                            - references[None, :, :]
                        ) ** 2,
                        axis=-1,
                    )
                    sample_index = int(
                        np.argmax(np.min(distances, axis=1))
                    )
                else:
                    sample_index = int(
                        self.rng.integers(len(x_class))
                    )

                new_center = x_class[sample_index].copy()
                self.centers[class_id, branch_id] = new_center
                self.usage[class_id, branch_id] = 1
                self.active[class_id, branch_id] = True
                replacement_centers.append(new_center)

            # Local competitive refinement among replacement branches only.
            for _ in range(epochs):
                for index in self.rng.permutation(len(x_class)):
                    distances = np.sum(
                        (
                            self.centers[class_id, damaged_ids]
                            - x_class[index]
                        ) ** 2,
                        axis=1,
                    )
                    local_winner = int(np.argmin(distances))
                    branch_id = int(
                        damaged_ids[local_winner]
                    )
                    self.usage[class_id, branch_id] += 1
                    local_rate = self.learning_rate / (
                        1.0
                        + 0.002
                        * self.usage[class_id, branch_id]
                    )
                    self.centers[class_id, branch_id] += (
                        local_rate
                        * (
                            x_class[index]
                            - self.centers[class_id, branch_id]
                        )
                    )


# ============================================================
# Fixed local-prototype baseline
# ============================================================

class FixedPrototypeDigits:
    """Same branch budget, but branch centers never adapt."""

    def __init__(self, prototypes_per_class=16, seed=0):
        self.prototypes_per_class = prototypes_per_class
        self.rng = np.random.default_rng(seed)

        self.centers = np.zeros(
            (10, prototypes_per_class, 64),
            dtype=np.float32,
        )
        self.initialized = np.zeros(10, dtype=bool)
        self.active = np.ones(
            (10, prototypes_per_class),
            dtype=bool,
        )
        self.usage = np.zeros(
            (10, prototypes_per_class),
            dtype=np.int64,
        )

    @property
    def parameter_count(self):
        return self.centers.size

    def train_epoch(self, x, y):
        for class_id in np.unique(y):
            class_id = int(class_id)
            if not self.initialized[class_id]:
                x_class = x[y == class_id]
                self.centers[class_id] = kmeans_plus_plus(
                    x_class,
                    self.prototypes_per_class,
                    self.rng,
                )
                self.initialized[class_id] = True

    def predict(self, x, seen_classes):
        seen_classes = np.asarray(seen_classes, dtype=int)
        class_scores = []

        for class_id in seen_classes:
            distances = np.sum(
                (
                    x[:, None, :]
                    - self.centers[class_id][None, :, :]
                ) ** 2,
                axis=-1,
            )
            distances[:, ~self.active[class_id]] = np.inf
            class_scores.append(np.min(distances, axis=1))

        score_matrix = np.stack(class_scores, axis=1)
        return seen_classes[np.argmin(score_matrix, axis=1)]

    def calculate_usage(self, x, y):
        self.usage[:] = 0

        for class_id in range(10):
            x_class = x[y == class_id]
            if len(x_class) == 0:
                continue

            distances = np.sum(
                (
                    x_class[:, None, :]
                    - self.centers[class_id][None, :, :]
                ) ** 2,
                axis=-1,
            )
            winners = np.argmin(distances, axis=1)

            for winner in winners:
                self.usage[class_id, winner] += 1

    def damage(self, fraction=0.25):
        number = int(round(self.centers.shape[0]
                           * self.centers.shape[1]
                           * fraction))
        flat_usage = self.usage.reshape(-1)
        damaged_flat = np.argsort(flat_usage)[-number:]

        damaged = np.column_stack(
            np.unravel_index(
                damaged_flat,
                self.usage.shape,
            )
        )

        for class_id, branch_id in damaged:
            self.active[class_id, branch_id] = False

        return damaged

    def repair(self, x, y, damaged):
        """Reseed damaged supports; no center adaptation."""
        for class_id in np.unique(damaged[:, 0]):
            class_id = int(class_id)
            damaged_ids = damaged[
                damaged[:, 0] == class_id, 1
            ]
            x_class = x[y == class_id]

            if len(x_class) == 0:
                continue

            surviving_ids = np.where(
                self.active[class_id]
            )[0]
            replacement_centers = []

            for branch_id in damaged_ids:
                reference_sets = []

                if len(surviving_ids):
                    reference_sets.append(
                        self.centers[class_id, surviving_ids]
                    )

                if replacement_centers:
                    reference_sets.append(
                        np.asarray(replacement_centers)
                    )

                if reference_sets:
                    references = np.vstack(reference_sets)
                    distances = np.sum(
                        (
                            x_class[:, None, :]
                            - references[None, :, :]
                        ) ** 2,
                        axis=-1,
                    )
                    sample_index = int(
                        np.argmax(np.min(distances, axis=1))
                    )
                else:
                    sample_index = int(
                        self.rng.integers(len(x_class))
                    )

                new_center = x_class[sample_index].copy()
                self.centers[class_id, branch_id] = new_center
                self.active[class_id, branch_id] = True
                replacement_centers.append(new_center)


# ============================================================
# Backpropagation MLP
# ============================================================

class BackpropMLP:
    def __init__(
        self,
        input_dim=64,
        hidden=136,
        learning_rate=0.03,
        seed=0,
    ):
        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(
            0.0,
            1.0 / np.sqrt(input_dim),
            size=(input_dim, hidden),
        ).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)

        self.w2 = rng.normal(
            0.0,
            1.0 / np.sqrt(hidden),
            size=(hidden, 10),
        ).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed + 1)
        self.active = np.ones(hidden, dtype=bool)

    @property
    def parameter_count(self):
        return (
            self.w1.size
            + self.b1.size
            + self.w2.size
            + self.b2.size
        )

    def _forward(self, x):
        hidden = np.tanh(x @ self.w1 + self.b1)
        hidden *= self.active[None, :]
        logits = hidden @ self.w2 + self.b2
        return hidden, logits

    def train_epoch(
        self,
        x,
        y,
        batch_size=64,
        trainable_units=None,
    ):
        order = self.rng.permutation(len(x))

        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            xb = x[indices]
            yb = y[indices]

            hidden, logits = self._forward(xb)
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(
                probabilities,
                axis=1,
                keepdims=True,
            )

            probabilities[
                np.arange(len(indices)),
                yb,
            ] -= 1.0
            probabilities /= len(indices)

            grad_w2 = hidden.T @ probabilities
            grad_b2 = probabilities.sum(axis=0)

            grad_hidden = (
                probabilities @ self.w2.T
            ) * (1.0 - hidden ** 2)
            grad_hidden *= self.active[None, :]

            grad_w1 = xb.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            if trainable_units is not None:
                mask = np.zeros(
                    len(self.active),
                    dtype=np.float32,
                )
                mask[trainable_units] = 1.0

                grad_w1 *= mask[None, :]
                grad_b1 *= mask
                grad_w2 *= mask[:, None]

            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1
            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2

    def predict(self, x, seen_classes):
        _, logits = self._forward(x)
        masked = np.full_like(logits, -1e9)
        masked[:, seen_classes] = logits[:, seen_classes]
        return np.argmax(masked, axis=1)

    def damage(self, fraction=0.25):
        number = int(round(len(self.active) * fraction))
        importance = (
            np.linalg.norm(self.w1, axis=0)
            * np.linalg.norm(self.w2, axis=1)
        )
        damaged = np.argsort(importance)[-number:]
        self.active[damaged] = False
        return damaged

    def repair(
        self,
        x,
        y,
        damaged,
        epochs=30,
    ):
        self.w1[:, damaged] = self.rng.normal(
            0.0,
            1.0 / np.sqrt(self.w1.shape[0]),
            size=(self.w1.shape[0], len(damaged)),
        )
        self.b1[damaged] = 0.0
        self.w2[damaged] = self.rng.normal(
            0.0,
            1.0 / np.sqrt(self.w2.shape[0]),
            size=(len(damaged), self.w2.shape[1]),
        )
        self.active[damaged] = True

        for _ in range(epochs):
            self.train_epoch(
                x,
                y,
                trainable_units=damaged,
            )


# ============================================================
# Linear softmax baseline
# ============================================================

class LinearSoftmax:
    def __init__(
        self,
        input_dim=64,
        learning_rate=0.08,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            0.0,
            0.01,
            size=(input_dim, 10),
        ).astype(np.float32)
        self.bias = np.zeros(10, dtype=np.float32)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed + 1)

    @property
    def parameter_count(self):
        return self.weights.size + self.bias.size

    def train_epoch(self, x, y, batch_size=64):
        order = self.rng.permutation(len(x))

        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            xb = x[indices]
            yb = y[indices]

            logits = xb @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)

            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(
                axis=1,
                keepdims=True,
            )
            probabilities[
                np.arange(len(indices)),
                yb,
            ] -= 1.0
            probabilities /= len(indices)

            self.weights -= (
                self.learning_rate
                * xb.T
                @ probabilities
            )
            self.bias -= (
                self.learning_rate
                * probabilities.sum(axis=0)
            )

    def predict(self, x, seen_classes):
        logits = x @ self.weights + self.bias
        masked = np.full_like(logits, -1e9)
        masked[:, seen_classes] = logits[:, seen_classes]
        return np.argmax(masked, axis=1)


# ============================================================
# Experiment
# ============================================================

retention_rows = []
adaptation_rows = []
damage_rows = []
parameter_rows = []
final_prediction_rows = []

run_start = time.perf_counter()

for seed in range(N_SEEDS):
    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_tasks,
    ) = make_split(seed)

    models = {
        "Linear softmax": LinearSoftmax(seed=seed),
        "Backprop MLP": BackpropMLP(seed=seed),
        "Backprop MLP + replay": BackpropMLP(
            seed=seed + 100_000
        ),
        "Fixed branches": FixedPrototypeDigits(
            prototypes_per_class=PROTOTYPES_PER_CLASS,
            seed=seed,
        ),
        "Dendritron": DendritronDigits(
            prototypes_per_class=PROTOTYPES_PER_CLASS,
            seed=seed,
        ),
    }

    replay_x = []
    replay_y = []
    replay_rng = np.random.default_rng(seed + 900_000)

    for model_name, model in models.items():
        parameter_rows.append({
            "Seed": seed,
            "Model": model_name,
            "Learned scalar parameters": model.parameter_count,
        })

    for task_index, (task_x, task_y) in enumerate(train_tasks):
        seen_classes = list(range(2 * task_index + 2))

        for epoch in range(1, EPOCHS_PER_TASK + 1):
            models["Linear softmax"].train_epoch(
                task_x,
                task_y,
            )
            models["Backprop MLP"].train_epoch(
                task_x,
                task_y,
            )

            if replay_x:
                replay_train_x = np.vstack(
                    [task_x] + replay_x
                )
                replay_train_y = np.concatenate(
                    [task_y] + replay_y
                )
            else:
                replay_train_x = task_x
                replay_train_y = task_y

            models["Backprop MLP + replay"].train_epoch(
                replay_train_x,
                replay_train_y,
            )

            models["Fixed branches"].train_epoch(
                task_x,
                task_y,
            )
            models["Dendritron"].train_epoch(
                task_x,
                task_y,
            )

            seen_test_mask = np.isin(
                y_test,
                seen_classes,
            )

            for model_name, model in models.items():
                current_accuracy = accuracy(
                    model,
                    x_test[seen_test_mask],
                    y_test[seen_test_mask],
                    seen_classes,
                )

                adaptation_rows.append({
                    "Seed": seed,
                    "Model": model_name,
                    "Task index": task_index,
                    "Task": TASK_NAMES[task_index],
                    "Epoch": epoch,
                    "Seen-class accuracy": current_accuracy,
                })

        # Add an exemplar memory only after the current task is learned.
        for class_id in TASKS[task_index]:
            class_examples = task_x[task_y == class_id]
            selected = replay_rng.choice(
                len(class_examples),
                size=min(
                    REPLAY_PER_CLASS,
                    len(class_examples),
                ),
                replace=False,
            )
            replay_x.append(class_examples[selected])
            replay_y.append(
                np.full(
                    len(selected),
                    class_id,
                    dtype=np.int64,
                )
            )

        # Evaluate every learned task separately after this stage.
        for evaluation_task in range(task_index + 1):
            pair = TASKS[evaluation_task]
            pair_mask = np.isin(y_test, pair)

            for model_name, model in models.items():
                pair_accuracy = accuracy(
                    model,
                    x_test[pair_mask],
                    y_test[pair_mask],
                    seen_classes,
                )

                retention_rows.append({
                    "Seed": seed,
                    "Model": model_name,
                    "After task index": task_index,
                    "After task": TASK_NAMES[task_index],
                    "Evaluated task index": evaluation_task,
                    "Evaluated task": TASK_NAMES[evaluation_task],
                    "Accuracy": pair_accuracy,
                })

    all_classes = list(range(10))

    for model_name, model in models.items():
        predictions = model.predict(
            x_test,
            all_classes,
        )

        if seed == 0:
            for true_label, prediction in zip(
                y_test,
                predictions,
            ):
                final_prediction_rows.append({
                    "Model": model_name,
                    "True label": int(true_label),
                    "Prediction": int(prediction),
                })

    # Build a common repair set with 40 examples per class.
    repair_rng = np.random.default_rng(seed + 700_000)
    repair_indices = []

    for class_id in range(10):
        class_indices = np.where(y_train == class_id)[0]
        chosen = repair_rng.choice(
            class_indices,
            size=min(
                REPAIR_PER_CLASS,
                len(class_indices),
            ),
            replace=False,
        )
        repair_indices.extend(chosen.tolist())

    repair_indices = np.asarray(repair_indices)
    repair_x_data = x_train[repair_indices]
    repair_y_data = y_train[repair_indices]

    # Fixed branch usage is measured from the training stream.
    models["Fixed branches"].calculate_usage(
        x_train,
        y_train,
    )

    for model_name in [
        "Backprop MLP",
        "Backprop MLP + replay",
        "Fixed branches",
        "Dendritron",
    ]:
        model = models[model_name]

        before = accuracy(
            model,
            x_test,
            y_test,
            all_classes,
        )
        damaged = model.damage(fraction=0.25)
        after_damage = accuracy(
            model,
            x_test,
            y_test,
            all_classes,
        )

        if model_name.startswith("Backprop"):
            model.repair(
                repair_x_data,
                repair_y_data,
                damaged,
                epochs=30,
            )
        elif model_name == "Fixed branches":
            model.repair(
                repair_x_data,
                repair_y_data,
                damaged,
            )
        else:
            model.repair(
                repair_x_data,
                repair_y_data,
                damaged,
                epochs=8,
            )

        after_repair = accuracy(
            model,
            x_test,
            y_test,
            all_classes,
        )

        damage_rows.extend([
            {
                "Seed": seed,
                "Model": model_name,
                "Stage": "Before damage",
                "Accuracy": before,
            },
            {
                "Seed": seed,
                "Model": model_name,
                "Stage": "After 25% damage",
                "Accuracy": after_damage,
            },
            {
                "Seed": seed,
                "Model": model_name,
                "Stage": "After constrained repair",
                "Accuracy": after_repair,
            },
        ])


# ============================================================
# Metrics
# ============================================================

retention_raw = pd.DataFrame(retention_rows)
adaptation_raw = pd.DataFrame(adaptation_rows)
damage_raw = pd.DataFrame(damage_rows)
parameter_raw = pd.DataFrame(parameter_rows)
prediction_raw = pd.DataFrame(final_prediction_rows)

metric_rows = []

for (seed, model_name), group in retention_raw.groupby(
    ["Seed", "Model"]
):
    final_group = group[
        group["After task index"] == 4
    ]
    final_accuracy = float(
        final_group["Accuracy"].mean()
    )

    forgetting_values = []

    for task_index in range(4):
        task_history = group[
            group["Evaluated task index"] == task_index
        ].sort_values("After task index")

        forgetting_values.append(
            float(
                task_history["Accuracy"].max()
                - task_history.iloc[-1]["Accuracy"]
            )
        )

    adaptation_group = adaptation_raw[
        (adaptation_raw["Seed"] == seed)
        & (adaptation_raw["Model"] == model_name)
    ]

    epochs_to_90 = []
    early_auc = []

    for task_index in range(5):
        curve = adaptation_group[
            adaptation_group["Task index"] == task_index
        ].sort_values("Epoch")

        reached = curve[
            curve["Seen-class accuracy"] >= 0.90
        ]

        epochs_to_90.append(
            int(reached.iloc[0]["Epoch"])
            if len(reached)
            else EPOCHS_PER_TASK + 1
        )
        early_auc.append(
            float(
                curve.head(5)[
                    "Seen-class accuracy"
                ].mean()
            )
        )

    metric_rows.append({
        "Seed": seed,
        "Model": model_name,
        "Final average task accuracy": final_accuracy,
        "Mean forgetting": float(
            np.mean(forgetting_values)
        ),
        "Mean epochs to 90%": float(
            np.mean(epochs_to_90)
        ),
        "First-five-epoch AUC": float(
            np.mean(early_auc)
        ),
    })

metrics_raw = pd.DataFrame(metric_rows)

summary = (
    metrics_raw.groupby("Model", as_index=False)
    .agg(
        Final_accuracy=(
            "Final average task accuracy",
            "mean",
        ),
        Final_accuracy_std=(
            "Final average task accuracy",
            "std",
        ),
        Mean_forgetting=(
            "Mean forgetting",
            "mean",
        ),
        Forgetting_std=(
            "Mean forgetting",
            "std",
        ),
        Epochs_to_90=(
            "Mean epochs to 90%",
            "mean",
        ),
        Early_AUC=(
            "First-five-epoch AUC",
            "mean",
        ),
    )
)

for column in summary.columns[1:]:
    summary[column] = summary[column].round(4)

parameter_summary = (
    parameter_raw.groupby("Model", as_index=False)
    .agg(
        Learned_scalar_parameters=(
            "Learned scalar parameters",
            "first",
        )
    )
)

damage_summary = (
    damage_raw.groupby(
        ["Model", "Stage"],
        as_index=False,
    )
    .agg(
        Mean_accuracy=("Accuracy", "mean"),
        Std_accuracy=("Accuracy", "std"),
    )
)
damage_summary["Mean_accuracy"] = (
    damage_summary["Mean_accuracy"].round(4)
)
damage_summary["Std_accuracy"] = (
    damage_summary["Std_accuracy"].round(4)
)

final_task_summary = (
    retention_raw[
        retention_raw["After task index"] == 4
    ]
    .groupby(
        ["Model", "Evaluated task"],
        as_index=False,
    )
    .agg(
        Mean_accuracy=("Accuracy", "mean"),
        Std_accuracy=("Accuracy", "std"),
    )
)
final_task_summary["Mean_accuracy"] = (
    final_task_summary["Mean_accuracy"].round(4)
)
final_task_summary["Std_accuracy"] = (
    final_task_summary["Std_accuracy"].round(4)
)


# ============================================================
# Save files
# ============================================================

retention_raw.to_csv(
    OUT / "dendritron_optdigits_retention_raw.csv",
    index=False,
)
adaptation_raw.to_csv(
    OUT / "dendritron_optdigits_adaptation_raw.csv",
    index=False,
)
damage_raw.to_csv(
    OUT / "dendritron_optdigits_damage_raw.csv",
    index=False,
)
metrics_raw.to_csv(
    OUT / "dendritron_optdigits_metrics_raw.csv",
    index=False,
)
summary.to_csv(
    OUT / "dendritron_optdigits_summary.csv",
    index=False,
)
parameter_summary.to_csv(
    OUT / "dendritron_optdigits_parameter_budget.csv",
    index=False,
)
damage_summary.to_csv(
    OUT / "dendritron_optdigits_damage_summary.csv",
    index=False,
)
final_task_summary.to_csv(
    OUT / "dendritron_optdigits_final_tasks.csv",
    index=False,
)
prediction_raw.to_csv(
    OUT / "dendritron_optdigits_seed0_predictions.csv",
    index=False,
)


# ============================================================
# Charts
# ============================================================

plot_order = [
    "Linear softmax",
    "Backprop MLP",
    "Backprop MLP + replay",
    "Fixed branches",
    "Dendritron",
]

plot_summary = summary.set_index("Model").reindex(
    plot_order
)

ax = plot_summary["Final_accuracy"].plot(
    kind="bar",
    yerr=plot_summary["Final_accuracy_std"],
    capsize=4,
    figsize=(10, 5.5),
    rot=20,
)
ax.set_ylim(0.0, 1.02)
ax.set_ylabel("Final 10-class accuracy")
ax.set_xlabel("")
ax.set_title(
    "Split Optical Digits: final class-incremental accuracy"
)
plt.tight_layout()
plt.savefig(
    OUT / "dendritron_optdigits_final_accuracy.png",
    dpi=180,
)
plt.show()

ax = plot_summary["Mean_forgetting"].plot(
    kind="bar",
    yerr=plot_summary["Forgetting_std"],
    capsize=4,
    figsize=(10, 5.5),
    rot=20,
)
ax.set_ylim(
    0.0,
    max(
        0.6,
        float(
            plot_summary["Mean_forgetting"].max()
            + 0.08
        ),
    ),
)
ax.set_ylabel("Mean forgetting; lower is better")
ax.set_xlabel("")
ax.set_title(
    "Split Optical Digits: catastrophic interference"
)
plt.tight_layout()
plt.savefig(
    OUT / "dendritron_optdigits_forgetting.png",
    dpi=180,
)
plt.show()

ax = plot_summary["Epochs_to_90"].plot(
    kind="bar",
    figsize=(10, 5.5),
    rot=20,
)
ax.set_ylim(0, EPOCHS_PER_TASK + 2)
ax.set_ylabel(
    "Mean epochs to 90% seen-class accuracy"
)
ax.set_xlabel("")
ax.set_title(
    "Split Optical Digits: adaptation speed"
)
plt.tight_layout()
plt.savefig(
    OUT / "dendritron_optdigits_adaptation_speed.png",
    dpi=180,
)
plt.show()

damage_order = [
    "Before damage",
    "After 25% damage",
    "After constrained repair",
]
damage_plot = damage_summary.pivot(
    index="Stage",
    columns="Model",
    values="Mean_accuracy",
).reindex(damage_order)

ax = damage_plot.plot(
    kind="bar",
    figsize=(10, 5.5),
    rot=0,
)
ax.set_ylim(0.0, 1.02)
ax.set_ylabel("10-class test accuracy")
ax.set_xlabel("")
ax.set_title(
    "Split Optical Digits: localized damage and repair"
)
plt.tight_layout()
plt.savefig(
    OUT / "dendritron_optdigits_damage_repair.png",
    dpi=180,
)
plt.show()

# Dendritron confusion matrix for seed zero.
dend_seed0 = prediction_raw[
    prediction_raw["Model"] == "Dendritron"
]
matrix = confusion_matrix(
    dend_seed0["True label"],
    dend_seed0["Prediction"],
    labels=list(range(10)),
)

fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(matrix)
ax.set_title("Dendritron confusion matrix — seed 0")
ax.set_xlabel("Predicted digit")
ax.set_ylabel("True digit")
ax.set_xticks(range(10))
ax.set_yticks(range(10))

for row in range(10):
    for column in range(10):
        ax.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
        )

fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.savefig(
    OUT / "dendritron_optdigits_confusion_matrix.png",
    dpi=180,
)
plt.show()


# ============================================================
# README
# ============================================================

readme = f"""DENDRITRON — SPLIT OPTICAL DIGITS BENCHMARK

Dataset:
- scikit-learn Optical Recognition of Handwritten Digits
- {len(ALL_X)} real handwritten digit images
- 64 pixel features per 8x8 image
- stratified 70/30 train-test split per seed

Protocol:
- five sequential tasks: {TASK_NAMES}
- class-incremental inference: no task identity is supplied
- final prediction is over all ten digit classes
- {N_SEEDS} random train-test splits
- {EPOCHS_PER_TASK} training epochs per task
- no replay in the main Dendritron / fixed branch / standard MLP comparison

Capacity:
- Dendritron: {DENDRITRON_PARAMETERS} learned scalar center values
- Fixed branches: {DENDRITRON_PARAMETERS} learned scalar center values
- Backprop MLP: {MLP_PARAMETERS} learned scalar parameters
- Difference between Dendritron and MLP: {
    abs(DENDRITRON_PARAMETERS - MLP_PARAMETERS)
} scalars

Dendritron learning:
- 16 class-local branches per digit
- label-local branch allocation
- winner-take-all branch competition
- only the winning branch center updates
- no autodiff, backpropagation, or global gradient
- no task identity during inference

Replay control:
- Backprop MLP + replay stores {REPLAY_PER_CLASS} examples per learned class
- maximum replay memory after all tasks: {REPLAY_PER_CLASS * 10} images

Damage test:
- disables the 25% most-used Dendritron / fixed branches
- disables the 25% most important MLP hidden units
- repair uses {REPAIR_PER_CLASS} examples per class
- only damaged/replacement structures are updated during repair

Interpretive limitation:
This is a prototype / LVQ-style realization of a Dendritron, not yet the full
dendritic architecture with emergent branch growth, eligibility traces,
multimodal modulators, recurrent coalitions, or autonomous context discovery.
"""
(OUT / "dendritron_optdigits_README.txt").write_text(
    readme,
    encoding="utf-8",
)


display_dataframe_to_user(
    "Split Optical Digits benchmark",
    summary,
)
display_dataframe_to_user(
    "Parameter budget",
    parameter_summary,
)
display_dataframe_to_user(
    "Damage and repair",
    damage_summary,
)
display_dataframe_to_user(
    "Final task retention",
    final_task_summary,
)

elapsed = time.perf_counter() - run_start

print(f"Benchmark completed in {elapsed:.2f} seconds.")
print("\nSummary:")
print(summary.to_string(index=False))
print("\nDamage and repair:")
print(damage_summary.to_string(index=False))