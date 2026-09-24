import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    model: str
    usage: dict[str, Any]
    provider: str | None = None
    request_id: str | None = None


class ProviderFatalError(RuntimeError):
    pass


class LayaProvider:
    name = "laya"

    def __init__(
        self,
        model: str,
        revision: str | None,
        compute_units: str | None,
        offline: bool,
    ) -> None:
        try:
            import laya_coreml as laya
        except ImportError as error:
            raise RuntimeError(
                "Laya support is not installed. Run: uv sync --extra laya"
            ) from error
        started = time.perf_counter()
        self.agent = laya.load(
            model,
            revision=revision,
            compute_units=compute_units,
            local_files_only=offline,
        )
        self.load_ms = (time.perf_counter() - started) * 1000
        self.model = model
        self.revision = revision
        self.compute_units = getattr(self.agent, "compute_units", compute_units)
        self.max_options = int(self.agent.shape["max_options"])
        self.manifest = {
            key: self.agent.manifest.get(key)
            for key in (
                "format",
                "format_version",
                "source",
                "revision",
                "source_weights_sha256",
                "package_sha256",
                "precision",
                "shape",
            )
        }

    def decide(self, state: str, question: dict[str, Any]) -> Decision:
        response = self.agent.predict(state, {"intent": question})
        answer = response["answers"]["intent"]
        return Decision(
            choice=answer["choice"],
            probabilities={key: float(value) for key, value in answer["probabilities"].items()},
            confidence=float(answer.get("confidence", answer["probabilities"][answer["choice"]])),
            model=str(response.get("model", self.model)),
            usage=dict(response.get("usage", {})),
            provider="local",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "revision": self.revision,
            "compute_units": self.compute_units,
            "max_options": self.max_options,
            "load_ms": round(self.load_ms, 3),
            "laya_coreml_version": version("laya-coreml"),
            "model_manifest": self.manifest,
        }


class JevProvider:
    name = "jev"

    def __init__(
        self,
        model: str,
        api_key_env: str,
        endpoint: str,
        backend: str,
        timeout: float,
        delay_ms: float,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self.model = model
        self.backend = backend
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.delay_ms = delay_ms
        self.last_request_finished = 0.0
        self.max_options = 255

    def pace(self) -> None:
        remaining = self.delay_ms / 1000 - (time.perf_counter() - self.last_request_finished)
        if remaining > 0:
            time.sleep(remaining)

    def decide(self, state: str, question: dict[str, Any]) -> Decision:
        body = json.dumps(
            {"model": self.model, "state": state, "questions": {"intent": question}},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "laya-jev-bench/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            details = error.read().decode(errors="replace")[:1000]
            message = f"Jev HTTP {error.code}: {details}"
            if error.code in {401, 403, 429}:
                raise ProviderFatalError(message) from error
            raise RuntimeError(message) from error
        finally:
            self.last_request_finished = time.perf_counter()
        answer = payload["answers"]["intent"]
        return Decision(
            choice=answer["choice"],
            probabilities={key: float(value) for key, value in answer["probabilities"].items()},
            confidence=float(answer.get("confidence", answer["probabilities"][answer["choice"]])),
            model=str(payload.get("model", self.model)),
            usage=dict(payload.get("usage", {})),
            provider=str(payload["provider"]) if payload.get("provider") is not None else None,
            request_id=str(payload["id"]) if payload.get("id") is not None else None,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "backend": self.backend,
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout,
            "delay_ms": self.delay_ms,
            "api_key_env": self.api_key_env,
        }
