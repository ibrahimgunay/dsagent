"""The Blackboard — how agents communicate.

Agents do not call each other directly. Each reads the upstream artifacts it
needs and writes its outputs back, keyed by task id. This decouples agents
(any agent can be swapped/retried), gives us a single lineage graph for free,
and makes concurrent execution within a batch safe via an internal lock.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    key: str                       # usually the producing task id
    kind: str                      # profiling / sql / econometrics / ...
    producer: str                  # agent/tool name
    value: Any                     # the payload (json-serializable)
    inputs: list[str] = field(default_factory=list)   # upstream artifact keys
    created_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            blob = json.dumps(self.value, sort_keys=True, default=str).encode()
            self.content_hash = hashlib.sha256(blob).hexdigest()[:12]


class Blackboard:
    def __init__(self) -> None:
        self._store: dict[str, Artifact] = {}
        self._lock = threading.RLock()
        self.log: list[str] = []

    def put(self, art: Artifact) -> Artifact:
        with self._lock:
            self._store[art.key] = art
            self.log.append(f"WRITE {art.key} <- {art.producer} "
                            f"({art.kind}, #{art.content_hash}, inputs={art.inputs})")
        return art

    def get(self, key: str) -> Artifact | None:
        with self._lock:
            return self._store.get(key)

    def value(self, key: str, default: Any = None) -> Any:
        art = self.get(key)
        return art.value if art else default

    def require(self, *keys: str) -> dict[str, Any]:
        """Fetch required upstream values or raise — agents call this first."""
        out = {}
        with self._lock:
            for k in keys:
                if k not in self._store:
                    raise KeyError(f"Blackboard missing required artifact: {k}")
                out[k] = self._store[k].value
        return out

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store)

    def lineage(self, key: str) -> dict:
        """Recursive provenance of an artifact."""
        with self._lock:
            art = self._store.get(key)
            if not art:
                return {}
            return {"key": key, "producer": art.producer, "hash": art.content_hash,
                    "inputs": [self.lineage(i) for i in art.inputs if i in self._store]}
