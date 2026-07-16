"""Functional ownership, registration, sharing, quarantine, and repair."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import Certificate, LifecycleState, RegistrationReceipt


@runtime_checkable
class FunctionalModule(Protocol):
    name: str

    def __call__(self, values: np.ndarray) -> np.ndarray: ...

    def clone(self, *, name: str | None = None) -> FunctionalModule: ...


@dataclass
class OwnedRegion:
    name: str
    module: FunctionalModule
    certificate: Certificate
    state: LifecycleState
    signature: str
    immutable: bool = False
    consumers: tuple[str, ...] = ()


class DendritronTissue:
    """An explicit graph of independently verified functional owners."""

    def __init__(self, *, registration_threshold: float = 0.8) -> None:
        if not 0.0 <= registration_threshold <= 1.0:
            raise ValueError("registration_threshold must be in [0, 1]")
        self.registration_threshold = registration_threshold
        self.regions: dict[str, OwnedRegion] = {}
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _predict(module: FunctionalModule, certificate: Certificate) -> np.ndarray:
        return np.asarray(module(certificate.inputs))

    @staticmethod
    def _signature(module: FunctionalModule) -> str:
        if hasattr(module, "signature"):
            return str(module.signature())
        payload = {"type": type(module).__name__, "state": repr(vars(module))}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def register(
        self,
        name: str,
        module: FunctionalModule,
        certificate: Certificate,
        *,
        immutable: bool = False,
        consumers: tuple[str, ...] = (),
    ) -> RegistrationReceipt:
        if name in self.regions:
            raise ValueError(f"region already exists: {name}")
        prediction = self._predict(module, certificate)
        accuracy = float(np.mean(prediction == certificate.targets))
        required = max(self.registration_threshold, certificate.minimum_accuracy)
        state = LifecycleState.REGISTERED if accuracy >= required else LifecycleState.QUARANTINED
        signature = self._signature(module)
        self.regions[name] = OwnedRegion(
            name=name,
            module=module,
            certificate=certificate,
            state=state,
            signature=signature,
            immutable=immutable,
            consumers=consumers,
        )
        self.events.append(
            {
                "event": "register" if state is LifecycleState.REGISTERED else "quarantine",
                "owner": name,
                "accuracy": accuracy,
                "signature": signature,
            }
        )
        return RegistrationReceipt(name, state, accuracy, signature)

    def execute(self, owner: str, values: np.ndarray) -> np.ndarray:
        region = self.regions[owner]
        if region.state is not LifecycleState.REGISTERED:
            raise RuntimeError(f"region {owner} is not registered")
        return np.asarray(region.module(values))

    def verify(self, owner: str) -> float:
        region = self.regions[owner]
        prediction = self._predict(region.module, region.certificate)
        return float(np.mean(prediction == region.certificate.targets))

    def verify_all(self) -> dict[str, float]:
        return {name: self.verify(name) for name in self.regions}

    def disable(self, owner: str) -> None:
        region = self.regions[owner]
        region.state = LifecycleState.DISABLED
        self.events.append({"event": "disable", "owner": owner})

    def repair(self, owner: str) -> RegistrationReceipt:
        region = self.regions[owner]
        if hasattr(region.module, "repair_from_certificate"):
            region.module.repair_from_certificate()
        accuracy = self.verify(owner)
        required = max(self.registration_threshold, region.certificate.minimum_accuracy)
        region.state = (
            LifecycleState.REGISTERED if accuracy >= required else LifecycleState.QUARANTINED
        )
        region.signature = self._signature(region.module)
        self.events.append({"event": "repair", "owner": owner, "accuracy": accuracy})
        return RegistrationReceipt(owner, region.state, accuracy, region.signature)

    def fork(self, owner: str, new_owner: str) -> OwnedRegion:
        """Copy-on-write specialization; the original owner remains unchanged."""

        if new_owner in self.regions:
            raise ValueError(f"region already exists: {new_owner}")
        original = self.regions[owner]
        original_signature = self._signature(original.module)
        if hasattr(original.module, "clone"):
            module = original.module.clone(name=new_owner)
        else:
            module = copy.deepcopy(original.module)
            if hasattr(module, "name"):
                module.name = new_owner
        fork = OwnedRegion(
            name=new_owner,
            module=module,
            certificate=original.certificate,
            state=LifecycleState.CANDIDATE,
            signature=self._signature(module),
            immutable=False,
            consumers=(),
        )
        self.regions[new_owner] = fork
        if self._signature(original.module) != original_signature:
            raise RuntimeError("copy-on-write mutated the original region")
        self.events.append({"event": "fork", "owner": owner, "new_owner": new_owner})
        return fork

    def admit_fork(self, owner: str, certificate: Certificate) -> RegistrationReceipt:
        region = self.regions[owner]
        if region.state is not LifecycleState.CANDIDATE:
            raise ValueError("only candidate forks can be admitted")
        prediction = self._predict(region.module, certificate)
        accuracy = float(np.mean(prediction == certificate.targets))
        required = max(self.registration_threshold, certificate.minimum_accuracy)
        region.certificate = certificate
        region.state = (
            LifecycleState.REGISTERED if accuracy >= required else LifecycleState.QUARANTINED
        )
        region.signature = self._signature(region.module)
        self.events.append({"event": "admit_fork", "owner": owner, "accuracy": accuracy})
        return RegistrationReceipt(owner, region.state, accuracy, region.signature)
