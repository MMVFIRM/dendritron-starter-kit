
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from dendritron.gating import evaluate_gate, evaluate_logit_equivalence


MEMORY_IDS = (
    "sum_threshold",
    "vowel_majority",
    "balanced_brackets",
    "alternating_sequence",
    "endpoint_match",
)

OBJECTIVE_VERSION = "restricted_binary_next_token_v2_cosine"
RUNTIME_VERSION = "dendritron-smollm2-v0.4.2"


@dataclass
class Config:
    model_id: str = "HuggingFaceTB/SmolLM2-360M"
    output_dir: str = "/content/dendritron_smollm2_360m_v4_2"
    seed: int = 7
    quick_mode: bool = False
    bootstrap_from_dir: Optional[str] = None

    train_examples_per_memory: int = 640
    validation_examples_per_memory: int = 160
    test_examples_per_memory: int = 240

    max_length: int = 96
    train_epochs: int = 8
    minimum_train_epochs: int = 3
    early_stopping_patience: int = 3
    registration_min_accuracy: float = 0.80
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.08
    minimum_learning_rate_fraction: float = 0.10
    max_grad_norm: float = 1.0

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    tap_fraction: float = 0.35
    pca_dim: int = 64
    verifier_rank: int = 8
    covariance_floor: float = 1e-5
    address_scale: float = 3.0
    temperature_steps: int = 120
    temperature_learning_rate: float = 0.05

    efficient_singleton_accuracy: float = 0.97
    reliable_candidate_coverage: float = 0.97
    critical_nominal_coverage: float = 0.99
    raps_penalty: float = 0.02
    raps_regularization_rank: int = 1

    known_acceptance_target: float = 0.98
    coordinate_batch_size: int = 32
    inference_batch_size: int = 48
    reload_logit_tolerance: float = 2e-3
    reload_relative_logit_tolerance: float = 0.15
    noise_floor_factor: float = 4.0

    def finalized(self) -> "Config":
        if self.quick_mode:
            self.train_examples_per_memory = 320
            self.validation_examples_per_memory = 100
            self.test_examples_per_memory = 120
            self.train_epochs = 8
            self.minimum_train_epochs = 3
            self.early_stopping_patience = 3
            self.batch_size = 16
            self.gradient_accumulation = 1
            self.learning_rate = 4e-4
            self.coordinate_batch_size = 24
            self.inference_batch_size = 24
            self.pca_dim = 32
            self.verifier_rank = 6
        return self


GATE_VERSION = "v2-noise-floor"

# Scalar criteria pass at or above the threshold, except the keys in
# UPPER_BOUND_GATE_KEYS, which pass at or below it. Raw-logit equality is no
# longer a direct criterion: reload equivalence is evaluated by
# dendritron.gating.evaluate_logit_equivalence against an absolute tolerance,
# a measured run-to-run noise floor, and a scale-relative tolerance, because
# the benchmark computes in bfloat16, where a float32-grade absolute
# tolerance (2e-3) is unattainable even when the reloaded function is
# identical.
GATE_THRESHOLDS = {
    "minimum_pack_validation_accuracy": 0.80,
    "oracle_accuracy": 0.90,
    "reliable_accuracy": 0.84,
    "critical_accuracy": 0.87,
    "critical_oracle_retention": 0.94,
    "critical_average_candidates": 3.20,
    "address_top2_coverage": 0.90,
    "old_memory_prediction_retention": 1.0,
    "backbone_hash_retention": 1.0,
    "checkpoint_prediction_equivalence": 1.0,
    "checkpoint_candidate_equivalence": 1.0,
    "checkpoint_logit_equivalence": 1.0,
    "uninstall_selection_exclusion": 1.0,
    "reinstall_prediction_equivalence": 1.0,
    "reinstall_logit_equivalence": 1.0,
    "adapter_hash_equivalence": 1.0,
}

UPPER_BOUND_GATE_KEYS = frozenset({"critical_average_candidates"})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def append_progress(output_dir: Path, stage: str, **kwargs: Any) -> None:
    path = output_dir / "progress.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"events": []}
    payload["events"].append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            **kwargs,
        }
    )
    payload["last_stage"] = stage
    atomic_json_dump(payload, path)


def select_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a Colab GPU. Select Runtime > Change runtime type > GPU."
        )
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def validate_torchao_environment() -> Dict[str, Any]:
    """
    Standard LoRA does not require torchao.

    Some Colab images include an old torchao build. New PEFT releases inspect
    that optional package while dispatching LoRA modules and reject versions
    <=0.16.0. Prefer no torchao at all for this non-quantized benchmark.
    """
    try:
        version_text = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return {
            "torchao_installed": False,
            "torchao_version": None,
            "status": "PASS: optional torchao is absent",
        }

    numeric_parts = []
    for part in version_text.split("."):
        match = re.match(r"(\d+)", part)
        if match is None:
            break
        numeric_parts.append(int(match.group(1)))
    while len(numeric_parts) < 3:
        numeric_parts.append(0)
    version_tuple = tuple(numeric_parts[:3])

    if version_tuple <= (0, 16, 0):
        raise RuntimeError(
            f"Incompatible optional torchao version {version_text} is installed. "
            "This benchmark uses ordinary BF16/FP16 LoRA and does not need "
            "torchao. Run `pip uninstall -y torchao`, restart the Colab session, "
            "and run the notebook again."
        )

    return {
        "torchao_installed": True,
        "torchao_version": version_text,
        "status": "PASS: torchao version is compatible",
    }


def gpu_summary() -> Dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "total_memory_gb": round(properties.total_memory / 1024**3, 2),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "torch_version": torch.__version__,
    }


def random_word(rng: np.random.Generator, minimum: int = 4, maximum: int = 9) -> str:
    alphabet = np.asarray(list("abcdefghijklmnopqrstuvwxyz"))
    length = int(rng.integers(minimum, maximum + 1))
    return "".join(rng.choice(alphabet, size=length).tolist())


def balanced_parentheses(rng: np.random.Generator, pairs: int) -> str:
    result: List[str] = []
    opens = 0
    closes = 0
    while closes < pairs:
        if opens < pairs and (opens == closes or rng.random() < 0.58):
            result.append("(")
            opens += 1
        else:
            result.append(")")
            closes += 1
    return "".join(result)


def make_example(memory_index: int, label: int, rng: np.random.Generator) -> str:
    """Generate five functions inside one common prompt envelope."""
    rule: str
    payload: str

    if memory_index == 0:
        while True:
            values = rng.integers(0, 100, size=5).astype(int)
            outcome = int(values.sum() >= 250)
            if outcome == label:
                break
        rule = "The sum of the five integers is at least 250."
        payload = f"Numbers: {', '.join(str(value) for value in values)}"

    elif memory_index == 1:
        vowels = np.asarray(list("aeiou"))
        consonants = np.asarray(list("bcdfghjklmnpqrstvwxyz"))
        length = 12
        vowel_count = int(rng.integers(7, 11)) if label == 1 else int(rng.integers(1, 6))
        characters = np.concatenate([
            rng.choice(vowels, size=vowel_count),
            rng.choice(consonants, size=length - vowel_count),
        ])
        rng.shuffle(characters)
        text = "".join(characters.tolist())
        rule = "Vowels are strictly more than half of the lowercase string."
        payload = f"String: {text}"

    elif memory_index == 2:
        pairs = int(rng.integers(3, 8))
        text = balanced_parentheses(rng, pairs)
        if label == 0:
            if rng.random() < 0.5:
                text = ")" + text[1:]
            else:
                text = text[:-1] + "("
        rule = "Every prefix is valid and the complete parenthesis string is balanced."
        payload = f"Sequence: {text}"

    elif memory_index == 3:
        first, second = rng.choice(np.arange(1, 20), size=2, replace=False).astype(int)
        values = np.asarray([
            first if position % 2 == 0 else second
            for position in range(10)
        ], dtype=int)
        if label == 0:
            position = int(rng.integers(1, 9))
            replacement_pool = [
                value for value in range(1, 20)
                if value not in (first, second)
            ]
            values[position] = int(rng.choice(replacement_pool))
        rule = "The sequence alternates exactly between two distinct integers."
        payload = f"Sequence: {' '.join(str(value) for value in values)}"

    elif memory_index == 4:
        # Token-level endpoint relation. The signal remains explicit after
        # subword tokenization.
        symbols = np.asarray(
            [
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "orange",
                "purple",
            ]
        )
        first_symbol = str(rng.choice(symbols))
        middle = [
            str(value)
            for value in rng.choice(
                symbols,
                size=6,
                replace=True,
            )
        ]
        if label == 1:
            final_symbol = first_symbol
        else:
            final_symbol = str(
                rng.choice(
                    symbols[
                        symbols != first_symbol
                    ]
                )
            )
        sequence = [
            first_symbol,
            *middle,
            final_symbol,
        ]
        rule = (
            "The first and final color tokens "
            "are identical."
        )
        payload = (
            "Color sequence: "
            + " | ".join(sequence)
        )

    else:
        raise ValueError(memory_index)

    return (
        "Evaluate one rule against one input. Return yes when the rule is true "
        "and no when it is false.\n"
        f"Rule: {rule}\n"
        f"Input:\n{payload}\n"
        "Answer:"
    )


def generate_records(
    split: str,
    examples_per_memory: int,
    seed: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}[split]
    for memory_index, memory_id in enumerate(MEMORY_IDS):
        rng = np.random.default_rng(seed + split_offset + memory_index * 100_000)
        labels = np.asarray(
            [0, 1] * ((examples_per_memory + 1) // 2),
            dtype=np.int64,
        )[:examples_per_memory]
        rng.shuffle(labels)
        for example_index, label in enumerate(labels):
            records.append(
                {
                    "split": split,
                    "memory_index": memory_index,
                    "memory_id": memory_id,
                    "example_index": example_index,
                    "prompt": make_example(memory_index, int(label), rng),
                    "label": int(label),
                }
            )
    random.Random(seed + split_offset).shuffle(records)
    return records


def save_records(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def choose_label_tokens(tokenizer: Any) -> Tuple[Tuple[str, str], Tuple[int, int]]:
    candidates = (
        (" no", " yes"),
        (" false", " true"),
        (" B", " A"),
        (" 0", " 1"),
    )
    for texts in candidates:
        token_ids = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in texts
        ]
        if all(len(ids) == 1 for ids in token_ids) and token_ids[0][0] != token_ids[1][0]:
            return texts, (int(token_ids[0][0]), int(token_ids[1][0]))
    raise RuntimeError("Could not identify a pair of distinct single-token labels.")


class CausalLabelDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        label_texts: Tuple[str, str],
        max_length: int,
    ):
        self.items: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token_id
        for record in records:
            prompt_ids = tokenizer.encode(
                record["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 2,
            )
            answer_ids = tokenizer.encode(
                label_texts[int(record["label"])],
                add_special_tokens=False,
            )
            if len(answer_ids) != 1:
                raise RuntimeError("Label text ceased to be one token.")
            input_ids = prompt_ids + answer_ids
            if eos is not None:
                input_ids += [eos]
            labels = [-100] * len(prompt_ids) + answer_ids
            if eos is not None:
                labels += [-100]
            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.items[index]


class CausalLabelCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        maximum = max(len(item["input_ids"]) for item in items)
        input_ids = []
        attention_masks = []
        labels = []
        for item in items:
            padding = maximum - len(item["input_ids"])
            input_ids.append(
                F.pad(item["input_ids"], (0, padding), value=self.pad_token_id)
            )
            attention_masks.append(
                F.pad(item["attention_mask"], (0, padding), value=0)
            )
            labels.append(F.pad(item["labels"], (0, padding), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }


class BinaryPromptDataset(Dataset):
    """Prompt-only examples for restricted two-token next-token training."""

    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.items: List[Dict[str, torch.Tensor]] = []
        for record in records:
            encoding = tokenizer(
                record["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            self.items.append({
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                "binary_label": torch.tensor(int(record["label"]), dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.items[index]


class BinaryPromptCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        maximum = max(len(item["input_ids"]) for item in items)
        return {
            "input_ids": torch.stack([
                F.pad(item["input_ids"], (0, maximum - len(item["input_ids"])), value=self.pad_token_id)
                for item in items
            ]),
            "attention_mask": torch.stack([
                F.pad(item["attention_mask"], (0, maximum - len(item["attention_mask"])), value=0)
                for item in items
            ]),
            "binary_label": torch.stack([item["binary_label"] for item in items]),
        }


def restricted_binary_logits(
    output_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    label_token_ids: Tuple[int, int],
) -> torch.Tensor:
    """Return [no, yes] logits at each prompt's final unmasked token."""
    last_indices = attention_mask.sum(dim=1) - 1
    return output_logits[
        torch.arange(len(output_logits), device=output_logits.device),
        last_indices,
    ][:, list(label_token_ids)]


@torch.no_grad()
def evaluate_binary_adapter(
    model: Any,
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
) -> Dict[str, float]:
    dataset = BinaryPromptDataset(records, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.inference_batch_size,
        shuffle=False,
        collate_fn=BinaryPromptCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    model.eval()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    margins: List[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        binary_logits = restricted_binary_logits(
            output.logits,
            batch["attention_mask"],
            label_token_ids,
        ).float()
        loss = F.cross_entropy(binary_logits, batch["binary_label"])
        losses.append(float(loss.item()) * len(binary_logits))
        predictions.append(binary_logits.argmax(dim=1).cpu().numpy())
        labels.append(batch["binary_label"].cpu().numpy())
        margins.append(torch.abs(binary_logits[:, 1] - binary_logits[:, 0]).cpu().numpy())
    prediction_array = np.concatenate(predictions)
    label_array = np.concatenate(labels)
    margin_array = np.concatenate(margins)
    correct_count = int(
        np.sum(prediction_array == label_array)
    )
    example_count = int(len(label_array))
    return {
        "loss": float(sum(losses) / max(example_count, 1)),
        "accuracy": float(correct_count / max(example_count, 1)),
        "correct": correct_count,
        "examples": example_count,
        "mean_absolute_logit_margin": float(margin_array.mean()),
        "minimum_absolute_logit_margin": float(margin_array.min()),
    }


def canonical_base_sha256(model: Any) -> str:
    """Hash every non-LoRA parameter by canonical name and raw bytes."""
    hasher = hashlib.sha256()
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" not in name and "modules_to_save" not in name
    ]
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        hasher.update(name.encode("utf-8"))
        tensor = parameter.detach().to("cpu").contiguous()
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        del tensor
    return hasher.hexdigest()


def directory_sha256(directory: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        hasher.update(str(path.relative_to(directory)).encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def assert_only_lora_trainable(model: Any) -> None:
    invalid = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if invalid:
        raise RuntimeError("Non-LoRA parameters became trainable: " + ", ".join(invalid[:10]))


class PromptDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.records = list(records)
        self.encodings = [
            tokenizer(
                record["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        encoding = self.encodings[index]
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "record_index": index,
        }


class PromptCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        maximum = max(len(item["input_ids"]) for item in items)
        return {
            "input_ids": torch.stack(
                [
                    F.pad(item["input_ids"], (0, maximum - len(item["input_ids"])), value=self.pad_token_id)
                    for item in items
                ]
            ),
            "attention_mask": torch.stack(
                [
                    F.pad(item["attention_mask"], (0, maximum - len(item["attention_mask"])), value=0)
                    for item in items
                ]
            ),
            "record_index": torch.tensor([item["record_index"] for item in items]),
        }


def load_base_model(config: Config, dtype: torch.dtype) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.config.use_cache = False
    return model


def assert_cuda_context_healthy() -> None:
    """Fail early with a useful message when a prior CUDA assert poisoned Colab."""
    try:
        probe = torch.tensor([1.0, 2.0], device="cuda")
        result = (probe * 2.0).sum()
        torch.cuda.synchronize()
        if float(result.cpu().item()) != 6.0:
            raise RuntimeError("Unexpected CUDA health-check result.")
    except Exception as error:
        raise RuntimeError(
            "The CUDA context is unhealthy. A device-side assert permanently "
            "poisons the current Colab process. Choose Runtime > Restart session, "
            "then run the notebook again from the first cell."
        ) from error


def _safe_parameter_sentinel(
    parameter: torch.Tensor,
    values_per_tensor: int,
) -> torch.Tensor:
    """
    Copy a few bounded contiguous slices to CPU.

    This deliberately avoids CUDA advanced indexing. The original prototype
    created a GPU `linspace` index tensor and indexed a flattened parameter,
    which can trigger a device-side bounds assert on some CUDA/PyTorch builds.
    """
    flattened = parameter.detach().reshape(-1)
    available = int(flattened.numel())
    requested = min(int(values_per_tensor), available)
    if requested <= 0:
        return torch.empty(0, dtype=parameter.dtype)

    segment_count = min(8, requested)
    segment_width = max(1, math.ceil(requested / segment_count))
    maximum_start = max(available - segment_width, 0)
    starts = np.linspace(
        0,
        maximum_start,
        num=segment_count,
        dtype=np.int64,
    )

    pieces: List[torch.Tensor] = []
    for start_value in starts:
        start = int(start_value)
        length = min(segment_width, available - start)
        if length <= 0:
            continue
        pieces.append(
            flattened
            .narrow(0, start, int(length))
            .to(device="cpu", non_blocking=False)
            .clone()
        )

    if not pieces:
        return torch.empty(0, dtype=parameter.dtype)
    return torch.cat(pieces, dim=0)[:requested].contiguous()


def non_lora_sentinels(
    model: Any,
    sample_count: int = 6,
    values_per_tensor: int = 64,
) -> Dict[str, torch.Tensor]:
    assert_cuda_context_healthy()
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" not in name
        and parameter.numel() >= values_per_tensor
    ]
    if not candidates:
        raise RuntimeError("No eligible frozen parameters were found for sentinels.")

    actual_sample_count = min(int(sample_count), len(candidates))
    selected_indices = np.linspace(
        0,
        len(candidates) - 1,
        num=actual_sample_count,
        dtype=np.int64,
    )

    sentinels: Dict[str, torch.Tensor] = {}
    for selected_index in selected_indices:
        name, parameter = candidates[int(selected_index)]
        sentinels[name] = _safe_parameter_sentinel(
            parameter,
            values_per_tensor,
        )
    return sentinels


def sentinel_retention(
    reference: Dict[str, torch.Tensor],
    model: Any,
) -> float:
    assert_cuda_context_healthy()
    current = dict(model.named_parameters())
    comparisons: List[bool] = []
    for name, reference_value in reference.items():
        if name not in current:
            comparisons.append(False)
            continue
        current_value = _safe_parameter_sentinel(
            current[name],
            len(reference_value),
        )
        comparisons.append(
            current_value.shape == reference_value.shape
            and torch.equal(current_value, reference_value)
        )
    return float(bool(comparisons) and all(comparisons))


def cosine_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(
        max((step - warmup_steps) / decay_steps, 0.0),
        1.0,
    )
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return float(
        minimum_fraction
        + (1.0 - minimum_fraction) * cosine
    )


def train_adapter(
    base_model: Any,
    memory_id: str,
    records: Sequence[Dict[str, Any]],
    validation_records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
    adapter_dir: Path,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weight_candidates = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    metadata_path = adapter_dir / "dendritron_training.json"
    if adapter_config_path.exists() and any(path.exists() for path in adapter_weight_candidates) and metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("objective_version") == OBJECTIVE_VERSION
            and bool(existing_metadata.get("registered", False))
            and int(
                existing_metadata.get(
                    "best_validation_correct",
                    -1,
                )
            )
            >= int(
                existing_metadata.get(
                    "registration_required_correct",
                    10**9,
                )
            )
        ):
            existing_metadata["status"] = "resumed_registered_adapter"
            existing_metadata["adapter_bytes"] = sum(
                path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
            )
            return base_model, existing_metadata
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    seed_everything(seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    assert_only_lora_trainable(model)
    reference_sentinels = non_lora_sentinels(model)

    dataset = BinaryPromptDataset(records, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=BinaryPromptCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_updates = math.ceil(len(loader) / config.gradient_accumulation) * config.train_epochs
    warmup_steps = int(total_updates * config.warmup_fraction)
    scaler_enabled = select_dtype() == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    history: List[Dict[str, float]] = []
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_validation_correct = -1
    validation_example_count = len(validation_records)
    registration_required_correct = int(
        math.ceil(
            config.registration_min_accuracy
            * validation_example_count
            - 1e-12
        )
    )
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.train_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        examples = 0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=select_dtype(), enabled=True):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                binary_logits = restricted_binary_logits(
                    output.logits,
                    batch["attention_mask"],
                    label_token_ids,
                ).float()
                full_loss = F.cross_entropy(binary_logits, batch["binary_label"])
                loss = full_loss / config.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(full_loss.item()) * len(binary_logits)
            correct += int((binary_logits.argmax(dim=1) == batch["binary_label"]).sum().item())
            examples += len(binary_logits)
            should_update = (
                (batch_index + 1) % config.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                learning_rate_scale = cosine_warmup(
                    global_update,
                    total_updates,
                    warmup_steps,
                    config.minimum_learning_rate_fraction,
                )
                for group in optimizer.param_groups:
                    group["lr"] = (
                        config.learning_rate
                        * learning_rate_scale
                    )
                global_update += 1

        validation_metrics = evaluate_binary_adapter(
            model,
            validation_records,
            tokenizer,
            label_token_ids,
            config,
        )
        epoch_row = {
            "epoch": epoch,
            "training_loss": running_loss / max(examples, 1),
            "training_accuracy": correct / max(examples, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_correct": validation_metrics["correct"],
            "validation_examples": validation_metrics["examples"],
            "registration_required_correct": registration_required_correct,
            "validation_mean_margin": validation_metrics["mean_absolute_logit_margin"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(
            f"{memory_id} epoch {epoch}: "
            f"train_acc={epoch_row['training_accuracy']:.4f} "
            f"val_acc={epoch_row['validation_accuracy']:.4f} "
            f"({epoch_row['validation_correct']}/"
            f"{epoch_row['validation_examples']}; "
            f"need {registration_required_correct}) "
            f"val_loss={epoch_row['validation_loss']:.4f} "
            f"margin={epoch_row['validation_mean_margin']:.4f}"
        )
        improved = (
            validation_metrics["accuracy"] > best_validation_accuracy + 1e-12
            or (
                abs(validation_metrics["accuracy"] - best_validation_accuracy) <= 1e-12
                and validation_metrics["loss"] < best_validation_loss
            )
        )
        if improved:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_validation_loss = validation_metrics["loss"]
            best_validation_correct = validation_metrics["correct"]
            best_epoch = epoch
            epochs_without_improvement = 0
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir, safe_serialization=True)
        else:
            epochs_without_improvement += 1
        if (
            epoch >= config.minimum_train_epochs
            and best_validation_correct >= registration_required_correct
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"{memory_id}: early stopping after epoch {epoch}; "
                f"best epoch={best_epoch}, best val={best_validation_accuracy:.4f}"
            )
            break

    retention = sentinel_retention(reference_sentinels, model)
    if retention != 1.0:
        raise RuntimeError(f"Frozen backbone changed while training {memory_id}.")
    registered = (
        best_validation_correct
        >= registration_required_correct
    )
    parameter_counts = {
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "total_wrapped_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    metadata = {
        "memory_id": memory_id,
        "status": "trained_and_registered" if registered else "rejected",
        "objective_version": OBJECTIVE_VERSION,
        "registered": registered,
        "registration_threshold": config.registration_min_accuracy,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_correct": best_validation_correct,
        "validation_examples": validation_example_count,
        "registration_required_correct": registration_required_correct,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "formation_seconds": time.perf_counter() - started,
        "backbone_sentinel_retention": retention,
        **parameter_counts,
        "adapter_bytes": sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()) if adapter_dir.exists() else 0,
    }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = adapter_dir / "dendritron_training.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(
        adapter_dir / "learning_curve.csv",
        index=False,
    )

    base_model = model.unload()
    base_model.to("cuda")
    base_model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    if not registered:
        print(
            f"{memory_id}: QUARANTINED — best validation "
            f"{best_validation_correct}/"
            f"{validation_example_count}; "
            f"registration requires "
            f"{registration_required_correct}."
        )
    return base_model, metadata


@torch.no_grad()
def extract_pooled_hidden(
    model: Any,
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    config: Config,
    tap_index: int,
) -> np.ndarray:
    model.eval()
    dataset = PromptDataset(records, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.coordinate_batch_size,
        shuffle=False,
        collate_fn=PromptCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    output_values: Optional[np.ndarray] = None
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[tap_index].float()
        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        last_indices = attention_mask.sum(dim=1) - 1
        last_pool = hidden[
            torch.arange(len(hidden), device=hidden.device),
            last_indices,
        ]
        pooled = torch.cat([mean_pool, last_pool], dim=1).cpu().numpy().astype(np.float32)
        record_indices = batch["record_index"].numpy()
        if output_values is None:
            output_values = np.empty((len(records), pooled.shape[1]), dtype=np.float32)
        output_values[record_indices] = pooled
    assert output_values is not None
    return output_values


class PPCA:
    def __init__(
        self,
        mean: np.ndarray,
        basis: np.ndarray,
        retained_variance: np.ndarray,
        residual_variance: float,
    ):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        self.retained_variance = np.asarray(retained_variance, dtype=np.float64)
        self.residual_variance = float(residual_variance)
        self.dimension = int(len(self.mean))
        self.rank = int(self.basis.shape[1])
        self.log_determinant = float(
            np.log(self.retained_variance).sum()
            + (self.dimension - self.rank) * math.log(self.residual_variance)
        )

    @classmethod
    def fit(cls, values: np.ndarray, rank: int, floor: float) -> "PPCA":
        values = values.astype(np.float64)
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        retained_rank = min(rank, covariance.shape[0] - 1)
        residual_variance = max(float(eigenvalues[retained_rank:].mean()), floor)
        retained_variance = np.maximum(eigenvalues[:retained_rank], residual_variance)
        return cls(
            mean,
            eigenvectors[:, :retained_rank],
            retained_variance,
            residual_variance,
        )

    def log_likelihood(self, values: np.ndarray) -> np.ndarray:
        difference = values.astype(np.float64) - self.mean
        total_energy = np.sum(difference * difference, axis=1)
        projection = difference @ self.basis
        projected_energy = projection * projection
        retained_energy = projected_energy.sum(axis=1)
        mahalanobis = (
            np.sum(projected_energy / self.retained_variance, axis=1)
            + (total_energy - retained_energy) / self.residual_variance
        )
        return -0.5 * (
            mahalanobis
            + self.log_determinant
            + self.dimension * math.log(2.0 * math.pi)
        )


class MemoryVerifier:
    def __init__(self, label_models: Tuple[PPCA, PPCA]):
        self.label_models = label_models

    @classmethod
    def fit(
        cls,
        coordinates: np.ndarray,
        labels: np.ndarray,
        rank: int,
        floor: float,
    ) -> "MemoryVerifier":
        return cls(
            (
                PPCA.fit(coordinates[labels == 0], rank, floor),
                PPCA.fit(coordinates[labels == 1], rank, floor),
            )
        )

    def score(self, coordinates: np.ndarray) -> np.ndarray:
        scores = np.stack(
            [model.log_likelihood(coordinates) for model in self.label_models],
            axis=1,
        )
        maximum = scores.max(axis=1, keepdims=True)
        return maximum[:, 0] + np.log(np.exp(scores - maximum).sum(axis=1)) - math.log(2.0)


def fit_shared_temperature(scores: np.ndarray, domains: np.ndarray, config: Config) -> float:
    score_tensor = torch.from_numpy(scores.astype(np.float32))
    domain_tensor = torch.from_numpy(domains.astype(np.int64))
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=config.temperature_learning_rate)
    for _ in range(config.temperature_steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = F.softplus(log_temperature) + 1e-3
        loss = F.cross_entropy(score_tensor / temperature, domain_tensor)
        loss.backward()
        optimizer.step()
    return float((F.softplus(log_temperature) + 1e-3).detach().item())


def bounded_address_scores(
    heads: Sequence[LogisticRegression],
    coordinates: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.stack(
        [
            np.tanh(head.decision_function(coordinates) / scale)
            for head in heads
        ],
        axis=1,
    ).astype(np.float32)


def smallest_fixed_k(probability: np.ndarray, domains: np.ndarray, target: float) -> int:
    ranking = np.argsort(-probability, axis=1)
    for k in range(1, probability.shape[1] + 1):
        covered = np.any(ranking[:, :k] == domains[:, None], axis=1)
        if float(covered.mean()) >= target:
            return k
    return probability.shape[1]


def finite_sample_quantile(values: Sequence[float], target: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    count = len(array)
    level = min(1.0, math.ceil((count + 1) * target) / count)
    index = min(max(int(math.ceil(level * count)) - 1, 0), count - 1)
    return float(array[index])


def fit_policies(
    scores: np.ndarray,
    domains: np.ndarray,
    config: Config,
) -> Dict[str, Any]:
    temperature = fit_shared_temperature(scores, domains, config)
    probability = F.softmax(
        torch.from_numpy(scores) / temperature,
        dim=1,
    ).numpy()
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == domains
    best = None
    for threshold in np.unique(confidence):
        mask = confidence >= threshold
        if not np.any(mask):
            continue
        accuracy = float(correct[mask].mean())
        coverage = float(mask.mean())
        if accuracy >= config.efficient_singleton_accuracy:
            if best is None or coverage > best[0]:
                best = (coverage, float(threshold))
    efficient_threshold = float(confidence.max() + 1e-6) if best is None else best[1]
    reliable_k = smallest_fixed_k(
        probability,
        domains,
        config.reliable_candidate_coverage,
    )

    conformity = []
    for row, true_domain in zip(probability, domains):
        order = np.argsort(-row)
        true_rank = int(np.where(order == true_domain)[0][0])
        cumulative = np.cumsum(row[order])
        conformity.append(
            float(
                cumulative[true_rank]
                + config.raps_penalty
                * max(true_rank + 1 - config.raps_regularization_rank, 0)
            )
        )
    critical_quantile = finite_sample_quantile(
        conformity,
        config.critical_nominal_coverage,
    )
    return {
        "temperature": temperature,
        "efficient_threshold": efficient_threshold,
        "reliable_k": int(reliable_k),
        "critical_quantile": critical_quantile,
    }


def candidate_sets(
    probability: np.ndarray,
    policy: Dict[str, Any],
    mode: str,
    config: Config,
) -> List[List[int]]:
    ranking = np.argsort(-probability, axis=1)
    if mode == "fast":
        return [[int(row[0])] for row in ranking]
    if mode == "fixed_top2":
        k = min(2, probability.shape[1])
        return [[int(value) for value in row[:k]] for row in ranking]
    if mode == "efficient":
        result = []
        for index, row in enumerate(ranking):
            k = 1 if probability[index, row[0]] >= policy["efficient_threshold"] else min(2, len(row))
            result.append([int(value) for value in row[:k]])
        return result
    if mode == "reliable":
        k = min(int(policy["reliable_k"]), probability.shape[1])
        return [[int(value) for value in row[:k]] for row in ranking]
    if mode == "critical":
        result = []
        for row in probability:
            order = np.argsort(-row)
            cumulative = 0.0
            selected = []
            for rank, memory_index in enumerate(order):
                selected.append(int(memory_index))
                cumulative += float(row[memory_index])
                adjusted = (
                    cumulative
                    + config.raps_penalty
                    * max(rank + 1 - config.raps_regularization_rank, 0)
                )
                if adjusted >= policy["critical_quantile"]:
                    break
            result.append(selected)
        return result
    raise ValueError(mode)


def verifier_matrix(
    verifiers: Sequence[MemoryVerifier],
    coordinates: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [verifier.score(coordinates) for verifier in verifiers],
        axis=1,
    )


def bind_candidates(
    verifier_scores: np.ndarray,
    candidates: Sequence[Sequence[int]],
) -> np.ndarray:
    return np.asarray(
        [
            max(
                candidate,
                key=lambda memory_index: verifier_scores[example_index, memory_index],
            )
            for example_index, candidate in enumerate(candidates)
        ],
        dtype=np.int64,
    )


def tokenize_prompts(
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    dataset = PromptDataset(records, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PromptCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    for batch in loader:
        yield (
            batch["input_ids"].to("cuda", non_blocking=True),
            batch["attention_mask"].to("cuda", non_blocking=True),
            batch["record_index"].numpy(),
        )


@torch.no_grad()
def restricted_label_logits(
    model: Any,
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
) -> np.ndarray:
    logits = np.empty((len(records), 2), dtype=np.float32)
    model.eval()
    for input_ids, attention_mask, record_indices in tokenize_prompts(
        records,
        tokenizer,
        config.max_length,
        config.inference_batch_size,
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        binary_logits = restricted_binary_logits(
            outputs.logits,
            attention_mask,
            label_token_ids,
        )
        logits[record_indices] = binary_logits.float().cpu().numpy()
    return logits


def restricted_label_predictions(
    model: Any,
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
) -> np.ndarray:
    return restricted_label_logits(
        model,
        records,
        tokenizer,
        label_token_ids,
        config,
    ).argmax(axis=1)


@torch.no_grad()
def activate_adapter_for_inference(model: Any, adapter_name: str) -> None:
    """Activate one adapter without accidentally making it trainable."""
    try:
        model.set_adapter(
            adapter_name,
            inference_mode=True,
        )
    except TypeError:
        model.set_adapter(adapter_name)
        if hasattr(model, "set_requires_grad"):
            model.set_requires_grad(
                adapter_name,
                False,
            )
        else:
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(False)


def grouped_adapter_logits(
    model: Any,
    records: Sequence[Dict[str, Any]],
    selected_memories: np.ndarray,
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
) -> np.ndarray:
    logits = np.full((len(records), 2), np.nan, dtype=np.float32)
    for memory_index, memory_id in enumerate(MEMORY_IDS):
        indices = np.where(selected_memories == memory_index)[0]
        if len(indices) == 0:
            continue
        activate_adapter_for_inference(model, memory_id)
        subset = [records[int(index)] for index in indices]
        subset_logits = restricted_label_logits(
            model,
            subset,
            tokenizer,
            label_token_ids,
            config,
        )
        logits[indices] = subset_logits
    if np.isnan(logits).any():
        raise RuntimeError("At least one example was not assigned to an installed memory.")
    return logits


def grouped_adapter_predictions(
    model: Any,
    records: Sequence[Dict[str, Any]],
    selected_memories: np.ndarray,
    tokenizer: Any,
    label_token_ids: Tuple[int, int],
    config: Config,
) -> np.ndarray:
    return grouped_adapter_logits(
        model,
        records,
        selected_memories,
        tokenizer,
        label_token_ids,
        config,
    ).argmax(axis=1)


def load_all_adapters(base_model: Any, adapters_dir: Path) -> Any:
    """Load all independent packs into one frozen inference model."""
    model: Any = PeftModel.from_pretrained(
        base_model,
        adapters_dir / MEMORY_IDS[0],
        adapter_name=MEMORY_IDS[0],
        is_trainable=False,
    )
    for memory_id in MEMORY_IDS[1:]:
        model.load_adapter(
            adapters_dir / memory_id,
            adapter_name=memory_id,
            is_trainable=False,
        )
    activate_adapter_for_inference(
        model,
        MEMORY_IDS[0],
    )
    model.eval()
    return model


def package_memory(
    output_dir: Path,
    memory_index: int,
    address_head: LogisticRegression,
    verifier: MemoryVerifier,
    pca: PCA,
    policy: Dict[str, Any],
    config: Config,
) -> Path:
    memory_id = MEMORY_IDS[memory_index]
    pack_dir = output_dir / "memory_pack_build" / memory_id
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapters" / memory_id
    shutil.copytree(adapter_dir, pack_dir / "adapter")

    portable_state: Dict[str, np.ndarray] = {
        "address_coef": address_head.coef_.astype(np.float32),
        "address_intercept": address_head.intercept_.astype(np.float32),
        "address_classes": address_head.classes_.astype(np.int64),
    }
    for label_index, model in enumerate(verifier.label_models):
        portable_state[f"label_{label_index}_mean"] = model.mean.astype(np.float32)
        portable_state[f"label_{label_index}_basis"] = model.basis.astype(np.float32)
        portable_state[f"label_{label_index}_retained_variance"] = (
            model.retained_variance.astype(np.float32)
        )
        portable_state[f"label_{label_index}_residual_variance"] = np.asarray(
            [model.residual_variance],
            dtype=np.float32,
        )
    np.savez_compressed(pack_dir / "memory_state.npz", **portable_state)
    manifest = {
        "format": "dendritron-smollm2-memory-v0.4.2",
        "memory_id": memory_id,
        "memory_index": memory_index,
        "base_model": config.model_id,
        "lora_rank": config.lora_rank,
        "verifier_rank": config.verifier_rank,
        "shared_coordinate_required": True,
        "global_pca_components": int(pca.n_components_),
        "policy_snapshot": policy,
        "objective_version": OBJECTIVE_VERSION,
        "registration_min_accuracy": config.registration_min_accuracy,
        "raw_examples_stored": 0,
    }
    atomic_json_dump(manifest, pack_dir / "manifest.json")
    zip_path = output_dir / "memory_packs" / f"{memory_id}.dmemory.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in pack_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(pack_dir))
    return zip_path


def build_audit_rows(
    records: Sequence[Dict[str, Any]],
    address_scores: np.ndarray,
    probability: np.ndarray,
    candidates: Sequence[Sequence[int]],
    verifier_scores: np.ndarray,
    selected: np.ndarray,
    predictions: np.ndarray,
    count: int = 20,
) -> List[Dict[str, Any]]:
    rows = []
    for index in range(min(count, len(records))):
        rows.append(
            {
                "example_index": index,
                "memory_id_true": records[index]["memory_id"],
                "label_true": int(records[index]["label"]),
                "prompt": records[index]["prompt"],
                "address_scores": {
                    MEMORY_IDS[memory_index]: float(address_scores[index, memory_index])
                    for memory_index in range(len(MEMORY_IDS))
                },
                "address_probabilities": {
                    MEMORY_IDS[memory_index]: float(probability[index, memory_index])
                    for memory_index in range(len(MEMORY_IDS))
                },
                "candidate_memories": [
                    MEMORY_IDS[memory_index] for memory_index in candidates[index]
                ],
                "verifier_scores": {
                    MEMORY_IDS[memory_index]: float(verifier_scores[index, memory_index])
                    for memory_index in candidates[index]
                },
                "selected_memory": MEMORY_IDS[int(selected[index])],
                "prediction": int(predictions[index]),
            }
        )
    return rows


def bootstrap_registered_adapters(
    bootstrap_from_dir: Optional[str],
    adapters_dir: Path,
) -> List[Dict[str, Any]]:
    """Copy and hash-verify compatible registered packs from a prior run."""
    events: List[Dict[str, Any]] = []
    if bootstrap_from_dir is None:
        return events

    source_adapters = Path(bootstrap_from_dir) / "adapters"
    if not source_adapters.exists():
        events.append(
            {
                "status": "bootstrap_source_missing",
                "source": str(source_adapters),
            }
        )
        return events

    for memory_id in MEMORY_IDS:
        source_memory = source_adapters / memory_id
        target_memory = adapters_dir / memory_id
        metadata_path = source_memory / "dendritron_training.json"

        if (
            not source_memory.exists()
            or not metadata_path.exists()
            or target_memory.exists()
        ):
            continue

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        compatible = (
            metadata.get("objective_version") == OBJECTIVE_VERSION
            and bool(metadata.get("registered", False))
            and int(metadata.get("best_validation_correct", -1))
            >= int(metadata.get("registration_required_correct", 10**9))
        )
        if not compatible:
            events.append(
                {
                    "memory_id": memory_id,
                    "status": "bootstrap_incompatible",
                    "source": str(source_memory),
                }
            )
            continue

        shutil.copytree(source_memory, target_memory)
        source_hash = directory_sha256(source_memory)
        target_hash = directory_sha256(target_memory)
        if source_hash != target_hash:
            shutil.rmtree(target_memory, ignore_errors=True)
            raise RuntimeError(
                f"Bootstrap hash mismatch for {memory_id}."
            )

        events.append(
            {
                "memory_id": memory_id,
                "status": "bootstrapped_registered_pack",
                "source": str(source_memory),
                "target": str(target_memory),
                "sha256": target_hash,
                "best_validation_correct": int(
                    metadata["best_validation_correct"]
                ),
                "validation_examples": int(
                    metadata["validation_examples"]
                ),
            }
        )

    return events


def main(config: Config) -> Dict[str, Any]:
    config = config.finalized()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bootstrap_events = bootstrap_registered_adapters(
        config.bootstrap_from_dir,
        adapters_dir,
    )
    atomic_json_dump(
        {
            "bootstrap_from_dir": config.bootstrap_from_dir,
            "events": bootstrap_events,
        },
        output_dir / "bootstrap_report.json",
    )

    seed_everything(config.seed)
    dtype = select_dtype()
    assert_cuda_context_healthy()
    optional_dependency_status = validate_torchao_environment()
    runtime_environment = gpu_summary()
    runtime_environment["optional_dependencies"] = optional_dependency_status
    atomic_json_dump(
        {
            "config": asdict(config),
            "environment": runtime_environment,
        },
        output_dir / "run_config.json",
    )
    append_progress(
        output_dir,
        "start",
        environment=runtime_environment,
        bootstrap_events=bootstrap_events,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    label_texts, label_token_ids = choose_label_tokens(tokenizer)
    atomic_json_dump(
        {
            "label_texts": label_texts,
            "label_token_ids": label_token_ids,
        },
        output_dir / "label_tokens.json",
    )

    train_records = generate_records(
        "train",
        config.train_examples_per_memory,
        config.seed,
    )
    validation_records = generate_records(
        "validation",
        config.validation_examples_per_memory,
        config.seed,
    )
    test_records = generate_records(
        "test",
        config.test_examples_per_memory,
        config.seed,
    )
    save_records(train_records, output_dir / "data" / "train.jsonl")
    save_records(validation_records, output_dir / "data" / "validation.jsonl")
    save_records(test_records, output_dir / "data" / "test.jsonl")
    append_progress(output_dir, "data_ready")

    base_model = load_base_model(config, dtype)
    append_progress(output_dir, "hashing_initial_backbone")
    initial_backbone_hash = canonical_base_sha256(base_model)
    initial_sentinels = non_lora_sentinels(base_model)
    formation_rows = []

    for memory_index, memory_id in enumerate(MEMORY_IDS):
        memory_records = [
            record for record in train_records
            if record["memory_index"] == memory_index
        ]
        memory_validation_records = [
            record for record in validation_records
            if record["memory_index"] == memory_index
        ]
        base_model, metadata = train_adapter(
            base_model,
            memory_id,
            memory_records,
            memory_validation_records,
            tokenizer,
            label_token_ids,
            config,
            adapters_dir / memory_id,
            config.seed + memory_index * 10_000,
        )
        formation_rows.append(metadata)
        append_progress(
            output_dir,
            (
                "adapter_ready"
                if metadata["registered"]
                else "adapter_quarantined"
            ),
            memory_id=memory_id,
            registered=bool(metadata["registered"]),
            best_validation_correct=int(
                metadata["best_validation_correct"]
            ),
            validation_examples=int(
                metadata["validation_examples"]
            ),
            registration_required_correct=int(
                metadata[
                    "registration_required_correct"
                ]
            ),
            adapter_bytes=metadata["adapter_bytes"],
        )

    formation_df_early = pd.DataFrame(
        formation_rows
    )
    formation_df_early.to_csv(
        output_dir
        / "dendritron_smollm2_formation.csv",
        index=False,
    )

    backbone_sentinel_retention = sentinel_retention(initial_sentinels, base_model)
    append_progress(output_dir, "hashing_final_backbone")
    final_backbone_hash = canonical_base_sha256(base_model)
    backbone_hash_retention = float(final_backbone_hash == initial_backbone_hash)
    if (
        backbone_sentinel_retention != 1.0
        or backbone_hash_retention != 1.0
    ):
        raise RuntimeError(
            "Frozen backbone integrity check failed "
            "after memory formation."
        )

    quarantined = [
        row
        for row in formation_rows
        if not bool(row["registered"])
    ]
    if quarantined:
        diagnostic_payload = {
            "runtime_version": RUNTIME_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "registration_threshold": (
                config.registration_min_accuracy
            ),
            "quarantined_memories": [
                {
                    "memory_id": row["memory_id"],
                    "best_validation_correct": int(
                        row["best_validation_correct"]
                    ),
                    "validation_examples": int(
                        row["validation_examples"]
                    ),
                    "registration_required_correct": int(
                        row[
                            "registration_required_correct"
                        ]
                    ),
                    "best_validation_accuracy": float(
                        row["best_validation_accuracy"]
                    ),
                    "best_epoch": int(row["best_epoch"]),
                }
                for row in quarantined
            ],
            "backbone_hash_retention": (
                backbone_hash_retention
            ),
        }
        atomic_json_dump(
            diagnostic_payload,
            output_dir
            / "registration_diagnostics.json",
        )
        diagnostic_zip = (
            output_dir
            / "Dendritron_v4_1_"
            "Registration_Diagnostics.zip"
        )
        with zipfile.ZipFile(
            diagnostic_zip,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(
                output_dir
                / "dendritron_smollm2_formation.csv",
                arcname=(
                    "dendritron_smollm2_"
                    "formation.csv"
                ),
            )
            archive.write(
                output_dir
                / "registration_diagnostics.json",
                arcname=(
                    "registration_diagnostics.json"
                ),
            )
            for row in quarantined:
                memory_dir = (
                    adapters_dir
                    / row["memory_id"]
                )
                for path in memory_dir.rglob("*"):
                    if path.is_file():
                        archive.write(
                            path,
                            arcname=(
                                f"quarantined/"
                                f"{row['memory_id']}/"
                                f"{path.relative_to(memory_dir)}"
                            ),
                        )
        failed_text = ", ".join(
            (
                f"{row['memory_id']} "
                f"({int(row['best_validation_correct'])}/"
                f"{int(row['validation_examples'])}; "
                f"need "
                f"{int(row['registration_required_correct'])})"
            )
            for row in quarantined
        )
        raise RuntimeError(
            "Registry assembly stopped after "
            "training every pack. Quarantined: "
            f"{failed_text}. Diagnostics: "
            f"{diagnostic_zip}"
        )

    model_layers = int(
        getattr(
            base_model.config,
            "num_hidden_layers",
        )
    )
    tap_index = max(1, min(model_layers, int(round(model_layers * config.tap_fraction))))
    coordinates_path = output_dir / "coordinates.npz"

    if coordinates_path.exists():
        coordinate_payload = np.load(coordinates_path)
        train_pooled = coordinate_payload["train_pooled"]
        validation_pooled = coordinate_payload["validation_pooled"]
        test_pooled = coordinate_payload["test_pooled"]
    else:
        base_model.eval()
        train_pooled = extract_pooled_hidden(
            base_model,
            train_records,
            tokenizer,
            config,
            tap_index,
        )
        validation_pooled = extract_pooled_hidden(
            base_model,
            validation_records,
            tokenizer,
            config,
            tap_index,
        )
        test_pooled = extract_pooled_hidden(
            base_model,
            test_records,
            tokenizer,
            config,
            tap_index,
        )
        np.savez_compressed(
            coordinates_path,
            train_pooled=train_pooled,
            validation_pooled=validation_pooled,
            test_pooled=test_pooled,
        )
    append_progress(output_dir, "frozen_hidden_states_ready", tap_index=tap_index)

    pca_path = output_dir / "shared_coordinate_pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
    else:
        pca_components = min(
            config.pca_dim,
            train_pooled.shape[0] - 1,
            train_pooled.shape[1],
        )
        pca = PCA(
            n_components=pca_components,
            whiten=True,
            svd_solver="randomized",
            random_state=config.seed,
        )
        pca.fit(train_pooled)
        joblib.dump(pca, pca_path)

    train_coordinate = pca.transform(train_pooled).astype(np.float32)
    validation_coordinate = pca.transform(validation_pooled).astype(np.float32)
    test_coordinate = pca.transform(test_pooled).astype(np.float32)

    train_domains = np.asarray([record["memory_index"] for record in train_records], dtype=np.int64)
    validation_domains = np.asarray(
        [record["memory_index"] for record in validation_records],
        dtype=np.int64,
    )
    test_domains = np.asarray([record["memory_index"] for record in test_records], dtype=np.int64)
    train_labels = np.asarray([record["label"] for record in train_records], dtype=np.int64)
    validation_labels = np.asarray(
        [record["label"] for record in validation_records],
        dtype=np.int64,
    )
    test_labels = np.asarray([record["label"] for record in test_records], dtype=np.int64)

    router_path = output_dir / "router_and_verifiers.joblib"
    if router_path.exists():
        routing_payload = joblib.load(router_path)
        address_heads = routing_payload["address_heads"]
        verifiers = routing_payload["verifiers"]
        policy = routing_payload["policy"]
    else:
        address_heads = []
        for memory_index in range(len(MEMORY_IDS)):
            address_head = LogisticRegression(
                C=2.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=config.seed + memory_index,
            )
            address_head.fit(
                train_coordinate,
                (train_domains == memory_index).astype(np.int64),
            )
            address_heads.append(address_head)

        verifiers = [
            MemoryVerifier.fit(
                train_coordinate[train_domains == memory_index],
                train_labels[train_domains == memory_index],
                config.verifier_rank,
                config.covariance_floor,
            )
            for memory_index in range(len(MEMORY_IDS))
        ]
        validation_address_scores = bounded_address_scores(
            address_heads,
            validation_coordinate,
            config.address_scale,
        )
        policy = fit_policies(
            validation_address_scores,
            validation_domains,
            config,
        )
        joblib.dump(
            {
                "address_heads": address_heads,
                "verifiers": verifiers,
                "policy": policy,
            },
            router_path,
        )
    append_progress(output_dir, "router_and_verifiers_ready", policy=policy)

    for memory_index in range(len(MEMORY_IDS)):
        package_memory(
            output_dir,
            memory_index,
            address_heads[memory_index],
            verifiers[memory_index],
            pca,
            policy,
            config,
        )

    test_address_scores = bounded_address_scores(
        address_heads,
        test_coordinate,
        config.address_scale,
    )
    test_probability = F.softmax(
        torch.from_numpy(test_address_scores) / float(policy["temperature"]),
        dim=1,
    ).numpy()
    test_verifier_scores = verifier_matrix(verifiers, test_coordinate)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    evaluation_model = load_base_model(config, dtype)
    evaluation_model.eval()
    base_predictions = restricted_label_predictions(
        evaluation_model,
        test_records,
        tokenizer,
        label_token_ids,
        config,
    )
    base_accuracy = float((base_predictions == test_labels).mean())

    # Install packs sequentially and require prior pack outputs to remain exact.
    old_memory_prediction_retention = 1.0
    prediction_snapshots: Dict[str, np.ndarray] = {}
    logit_snapshots: Dict[str, np.ndarray] = {}
    loaded_memory_ids: List[str] = []
    for memory_index, memory_id in enumerate(MEMORY_IDS):
        evaluation_model.load_adapter(
            adapters_dir / memory_id,
            adapter_name=memory_id,
            is_trainable=False,
        )
        loaded_memory_ids.append(memory_id)
        for prior_index, prior_id in enumerate(loaded_memory_ids):
            indices = np.where(test_domains == prior_index)[0][:80]
            activate_adapter_for_inference(evaluation_model, prior_id)
            subset_records = [test_records[int(index)] for index in indices]
            current_logits = restricted_label_logits(
                evaluation_model,
                subset_records,
                tokenizer,
                label_token_ids,
                config,
            )
            current = current_logits.argmax(axis=1)
            if prior_id in prediction_snapshots:
                old_memory_prediction_retention = min(
                    old_memory_prediction_retention,
                    float(np.mean(current == prediction_snapshots[prior_id])),
                )
            else:
                prediction_snapshots[prior_id] = current
                logit_snapshots[prior_id] = current_logits
    evaluation_model.eval()

    oracle_predictions = grouped_adapter_predictions(
        evaluation_model,
        test_records,
        test_domains,
        tokenizer,
        label_token_ids,
        config,
    )
    oracle_accuracy = float((oracle_predictions == test_labels).mean())

    mode_rows = []
    mode_payloads: Dict[str, Dict[str, Any]] = {}
    for mode in ("fast", "efficient", "reliable", "critical"):
        candidates = candidate_sets(
            test_probability,
            policy,
            mode,
            config,
        )
        selected = bind_candidates(test_verifier_scores, candidates)
        selected_logits = grouped_adapter_logits(
            evaluation_model,
            test_records,
            selected,
            tokenizer,
            label_token_ids,
            config,
        )
        predictions = selected_logits.argmax(axis=1)
        candidate_coverage = float(
            np.mean(
                [
                    int(test_domains[index]) in candidates[index]
                    for index in range(len(test_records))
                ]
            )
        )
        accuracy = float((predictions == test_labels).mean())
        mode_rows.append(
            {
                "mode": mode,
                "accuracy": accuracy,
                "oracle_accuracy": oracle_accuracy,
                "oracle_retention": accuracy / max(oracle_accuracy, 1e-12),
                "memory_selection_accuracy": float((selected == test_domains).mean()),
                "candidate_coverage": candidate_coverage,
                "average_candidates": float(np.mean([len(item) for item in candidates])),
            }
        )
        mode_payloads[mode] = {
            "candidates": candidates,
            "selected": selected,
            "logits": selected_logits,
            "predictions": predictions,
        }

    top_two = np.argsort(-test_address_scores, axis=1)[:, :2]
    address_top2_coverage = float(
        np.mean(
            [
                int(test_domains[index]) in top_two[index]
                for index in range(len(test_domains))
            ]
        )
    )

    checkpoint_dir = output_dir / "runtime_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    shutil.copy2(pca_path, checkpoint_dir / pca_path.name)
    shutil.copy2(router_path, checkpoint_dir / router_path.name)
    atomic_json_dump(
        {
            "model_id": config.model_id,
            "memory_ids": MEMORY_IDS,
            "tap_index": tap_index,
            "policy": policy,
            "label_texts": label_texts,
            "label_token_ids": label_token_ids,
        },
        checkpoint_dir / "runtime_manifest.json",
    )

    reference_mode = mode_payloads["reliable"]
    reference_logits = reference_mode["logits"]
    reference_predictions = reference_mode["predictions"]
    reference_candidates = reference_mode["candidates"]

    # Same-instance rerun: a lower bound on run-to-run logit noise. The reload
    # gate compares against this measured floor so that reduced-precision
    # (bfloat16) accumulation noise is not misread as an integrity failure.
    rerun_logits = grouped_adapter_logits(
        evaluation_model,
        test_records,
        reference_mode["selected"],
        tokenizer,
        label_token_ids,
        config,
    )
    rerun_max_logit_delta = float(np.max(np.abs(rerun_logits - reference_logits)))

    # Reload the base and every independent adapter from their installable directories.
    del evaluation_model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base_model = load_base_model(config, dtype)
    reloaded_test_pooled = extract_pooled_hidden(
        reloaded_base_model,
        test_records,
        tokenizer,
        config,
        tap_index,
    )
    reloaded_test_coordinate = pca.transform(
        reloaded_test_pooled
    ).astype(np.float32)
    reloaded_address_scores = bounded_address_scores(
        address_heads,
        reloaded_test_coordinate,
        config.address_scale,
    )
    reloaded_probability = F.softmax(
        torch.from_numpy(reloaded_address_scores)
        / float(policy["temperature"]),
        dim=1,
    ).numpy()
    reloaded_candidates = candidate_sets(
        reloaded_probability,
        policy,
        "reliable",
        config,
    )
    reloaded_verifier_scores = verifier_matrix(
        verifiers,
        reloaded_test_coordinate,
    )
    reloaded_selected = bind_candidates(
        reloaded_verifier_scores,
        reloaded_candidates,
    )
    reloaded_model = load_all_adapters(
        reloaded_base_model,
        adapters_dir,
    )
    reloaded_model.eval()
    reloaded_logits = grouped_adapter_logits(
        reloaded_model,
        test_records,
        reloaded_selected,
        tokenizer,
        label_token_ids,
        config,
    )
    reloaded_predictions = reloaded_logits.argmax(axis=1)
    checkpoint_prediction_equivalence = float(
        np.mean(reloaded_predictions == reference_predictions)
    )
    checkpoint_max_logit_delta = float(
        np.max(np.abs(reloaded_logits - reference_logits))
    )
    checkpoint_equivalence = evaluate_logit_equivalence(
        "checkpoint",
        reference_logits,
        reloaded_logits,
        absolute_tolerance=config.reload_logit_tolerance,
        relative_tolerance=config.reload_relative_logit_tolerance,
        noise_floor=rerun_max_logit_delta,
        noise_factor=config.noise_floor_factor,
    )
    checkpoint_candidate_equivalence = float(
        np.mean(
            [
                reloaded_candidates[index]
                == reference_candidates[index]
                for index in range(
                    len(reference_candidates)
                )
            ]
        )
    )

    # Physical adapter deletion and reload.
    removed_memory = MEMORY_IDS[2]
    removed_adapter_dir = adapters_dir / removed_memory
    adapter_hash_before = directory_sha256(removed_adapter_dir)
    reloaded_model.delete_adapter(removed_memory)
    uninstall_selection_exclusion = float(
        removed_memory not in set(reloaded_model.peft_config.keys())
    )
    reloaded_model.load_adapter(
        removed_adapter_dir,
        adapter_name=removed_memory,
        is_trainable=False,
    )
    adapter_hash_after = directory_sha256(removed_adapter_dir)
    adapter_hash_equivalence = float(adapter_hash_before == adapter_hash_after)
    reinstall_indices = np.where(test_domains == 2)[0][:80]
    reinstall_records = [test_records[int(index)] for index in reinstall_indices]
    activate_adapter_for_inference(reloaded_model, removed_memory)
    after_reinstall_logits = restricted_label_logits(
        reloaded_model,
        reinstall_records,
        tokenizer,
        label_token_ids,
        config,
    )
    after_reinstall = after_reinstall_logits.argmax(axis=1)
    reinstall_prediction_equivalence = float(
        np.mean(after_reinstall == prediction_snapshots[removed_memory])
    )
    reinstall_max_logit_delta = float(
        np.max(np.abs(after_reinstall_logits - logit_snapshots[removed_memory]))
    )
    reinstall_equivalence = evaluate_logit_equivalence(
        "reinstall",
        logit_snapshots[removed_memory],
        after_reinstall_logits,
        absolute_tolerance=config.reload_logit_tolerance,
        relative_tolerance=config.reload_relative_logit_tolerance,
        noise_floor=rerun_max_logit_delta,
        noise_factor=config.noise_floor_factor,
    )

    critical_payload = mode_payloads["critical"]
    audit_rows = build_audit_rows(
        test_records,
        test_address_scores,
        test_probability,
        critical_payload["candidates"],
        test_verifier_scores,
        critical_payload["selected"],
        critical_payload["predictions"],
    )
    atomic_json_dump({"traces": audit_rows}, output_dir / "audit_trace.json")

    formation_df = pd.DataFrame(formation_rows)
    mode_df = pd.DataFrame(mode_rows)
    adapter_parameters = int(formation_df["trainable_parameters"].mean())
    adapter_bytes = float(formation_df["adapter_bytes"].mean())

    reliable_row = mode_df.set_index("mode").loc["reliable"]
    critical_row = mode_df.set_index("mode").loc["critical"]
    minimum_pack_validation_accuracy = float(
        formation_df["best_validation_accuracy"].min()
    )
    results = {
        "model": "Dendritron Runtime v0.4.2 on SmolLM2-360M",
        "runtime_version": RUNTIME_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "base_model": config.model_id,
        "quick_mode": config.quick_mode,
        "bootstrap_from_dir": config.bootstrap_from_dir,
        "bootstrapped_memories": int(
            sum(
                event.get("status")
                == "bootstrapped_registered_pack"
                for event in bootstrap_events
            )
        ),
        "memories": len(MEMORY_IDS),
        "base_accuracy": base_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "reliable_accuracy": float(reliable_row["accuracy"]),
        "critical_accuracy": float(critical_row["accuracy"]),
        "critical_oracle_retention": float(critical_row["oracle_retention"]),
        "critical_candidate_coverage": float(critical_row["candidate_coverage"]),
        "critical_average_candidates": float(critical_row["average_candidates"]),
        "minimum_pack_validation_accuracy": minimum_pack_validation_accuracy,
        "address_top2_coverage": address_top2_coverage,
        "old_memory_prediction_retention": old_memory_prediction_retention,
        "backbone_sentinel_retention": backbone_sentinel_retention,
        "backbone_hash_retention": backbone_hash_retention,
        "initial_backbone_sha256": initial_backbone_hash,
        "final_backbone_sha256": final_backbone_hash,
        "checkpoint_prediction_equivalence": checkpoint_prediction_equivalence,
        "checkpoint_candidate_equivalence": checkpoint_candidate_equivalence,
        "checkpoint_max_logit_delta": checkpoint_max_logit_delta,
        "rerun_max_logit_delta": rerun_max_logit_delta,
        "checkpoint_scaled_logit_delta": checkpoint_equivalence.scaled_delta,
        "checkpoint_logit_scale": checkpoint_equivalence.scale,
        "checkpoint_logit_equivalence": float(checkpoint_equivalence.passed),
        "checkpoint_logit_equivalence_basis": checkpoint_equivalence.basis,
        "uninstall_selection_exclusion": uninstall_selection_exclusion,
        "reinstall_prediction_equivalence": reinstall_prediction_equivalence,
        "reinstall_max_logit_delta": reinstall_max_logit_delta,
        "reinstall_scaled_logit_delta": reinstall_equivalence.scaled_delta,
        "reinstall_logit_scale": reinstall_equivalence.scale,
        "reinstall_logit_equivalence": float(reinstall_equivalence.passed),
        "reinstall_logit_equivalence_basis": reinstall_equivalence.basis,
        "adapter_hash_equivalence": adapter_hash_equivalence,
        "adapter_parameters_per_memory": adapter_parameters,
        "adapter_bytes_per_memory": adapter_bytes,
        "shared_coordinate_dimension": int(pca.n_components_),
        "tap_index": tap_index,
        "raw_examples_stored_in_memory_pack": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    gate_pass, gate_failures = evaluate_gate(
        results,
        GATE_THRESHOLDS,
        upper_bound_keys=UPPER_BOUND_GATE_KEYS,
    )
    results["gate_pass"] = gate_pass
    results["gate_failures"] = gate_failures
    results["gate_version"] = GATE_VERSION

    pd.DataFrame([results]).to_csv(output_dir / "dendritron_smollm2_results.csv", index=False)
    mode_df.to_csv(output_dir / "dendritron_smollm2_modes.csv", index=False)
    formation_df.to_csv(output_dir / "dendritron_smollm2_formation.csv", index=False)
    atomic_json_dump(
        {
            "config": asdict(config),
            "thresholds": GATE_THRESHOLDS,
            "environment": runtime_environment,
            "policy": policy,
            "results": results,
        },
        output_dir / "dendritron_smollm2_summary.json",
    )
    append_progress(output_dir, "complete", gate_pass=gate_pass)

    final_zip = output_dir / "Dendritron_SmolLM2_360M_Results_and_Memory_Packs.zip"
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if not path.is_file() or path == final_zip:
                continue
            # Do not duplicate the large combined checkpoint model in the compact handoff zip.
            if "runtime_checkpoint/model_with_adapters" in str(path):
                continue
            archive.write(path, arcname=path.relative_to(output_dir))

    print("\nFINAL RESULTS")
    print(pd.DataFrame([results]).to_string(index=False))
    print("\nMODE SUMMARY")
    print(mode_df.to_string(index=False))
    print(f"\nArtifacts: {output_dir}")
    print(f"Compact package: {final_zip}")
    return {
        "results": results,
        "modes": mode_df,
        "formation": formation_df,
        "output_dir": str(output_dir),
        "zip": str(final_zip),
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/content/dendritron_smollm2_360m_v4_2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick-mode", action="store_true")
    arguments = parser.parse_args()
    return Config(
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        quick_mode=arguments.quick_mode,
    )


if __name__ == "__main__":
    import traceback

    parsed_config = parse_args()
    try:
        main(parsed_config)
    except Exception as error:
        failure_dir = Path(parsed_config.output_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_payload = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json_dump(
            failure_payload,
            failure_dir / "failure.json",
        )
        print("\nDENDRITRON RUN FAILED\n")
        traceback.print_exc()
        print(
            "\nFailure details saved to:",
            failure_dir / "failure.json",
        )
        raise
