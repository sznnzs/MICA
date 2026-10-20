from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .call_identity import logical_call_id, logical_call_offset
from .schemas import ModelCall, ModelCallPosition, ModelUsage


def _redact_error(value: object, *, secret: str | None = None) -> str:
    text = str(value)
    if secret:
        text = text.replace(secret, "[redacted]")
    for name, key in os.environ.items():
        if (name == "API_KEY" or name.endswith(("_API_KEY", "_TOKEN", "_SECRET"))) and key:
            text = text.replace(key, "[redacted]")
    text = re.sub(r"(?i)(authorization|api[-_ ]?key)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    return text[:2000]


def _base_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("base URL must not contain credentials, query or fragment")
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def resolve_credentials(config: Mapping[str, Any]) -> tuple[str, str]:
    # Read runtime credentials from the configured environment variable.
    if config.get("credential_module") or any(
        name in config for name in ("api_key", "access_token", "authorization")
    ):
        raise ValueError("credentials must use environment variable names only")
    if str(config.get("auth", "required")) == "none":
        endpoint = str(config.get("base_url") or "").strip()
        if not endpoint:
            raise RuntimeError("local no-auth model requires base_url")
        return "local-no-auth", _base_url(endpoint)
    key = os.environ.get(str(config.get("api_key_env", ""))) if config.get("api_key_env") else None
    endpoint = str(config.get("base_url") or "").strip() or None
    if config.get("base_url_env"):
        endpoint = endpoint or os.environ.get(str(config["base_url_env"]))
    if not key:
        raise RuntimeError("missing API credential; configure api_key_env")
    if not endpoint:
        raise RuntimeError("missing OpenAI-compatible base URL")
    return key, _base_url(endpoint)


def _http_proxy() -> str | None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip().lower().startswith(("http://", "https://")):
            return value.strip()
    return None


class ModelInvocationError(RuntimeError):
    def __init__(self, message: str, call: ModelCall, *, prompt: Any = None) -> None:
        super().__init__(message)
        self.call = call
        self.prompt = prompt


class OpenAICompatibleModel:
    def __init__(self, model_id: str, config: Mapping[str, Any]) -> None:
        self.model_id = model_id
        self.config = dict(config)
        self.model_name = str(config["model"])
        self.role = str(config.get("role", "policy"))
        if self.role != "policy":
            raise ValueError(f"model role must be policy: {model_id}")
        key, endpoint = resolve_credentials(self.config)
        self.base_url = endpoint
        try:
            from openai import OpenAI
            import httpx
        except ImportError as exc:
            raise RuntimeError("openai and httpx are required by the controller environment") from exc
        proxy_mode = str(config.get("proxy_mode", "direct" if config.get("auth") == "none" else "http_env_only"))
        proxy = _http_proxy() if proxy_mode == "http_env_only" else None
        timeout_raw = config.get("timeout_seconds", 120.0)
        timeout = None if timeout_raw is None else float(timeout_raw)
        http_client = httpx.Client(
            proxy=proxy,
            timeout=timeout,
            trust_env=False,
        )
        self.client = OpenAI(
            api_key=key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def generate(
        self,
        *,
        call_id: str,
        messages: tuple[Mapping[str, str], ...],
        purpose: str,
        generation: Mapping[str, Any],
        seed: int,
    ) -> ModelCall:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [dict(item) for item in messages],
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "max_tokens": int(
                generation.get("request_max_new_tokens", generation["max_new_tokens"])
            ),
        }
        if bool(self.config.get("send_seed", True)):
            request["seed"] = int(seed)
        extra_body = dict(self.config.get("extra_body") or {})
        if extra_body:
            request["extra_body"] = extra_body
        if self.config.get("response_format"):
            request["response_format"] = dict(self.config["response_format"])
        max_retries = int(self.config.get("max_retries", 2))
        backoff = float(self.config.get("retry_backoff_seconds", 1.0))
        started = time.perf_counter()
        retry_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(**request)
                choice = response.choices[0]
                content = getattr(choice.message, "content", None)
                if isinstance(content, list):
                    content = "".join(
                        str(getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else ""))
                        for item in content
                    )
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning is None:
                    reasoning = dict(getattr(choice.message, "model_extra", None) or {}).get("reasoning_content")
                usage = getattr(response, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                return ModelCall(
                    call_id=call_id,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    role=self.role,
                    purpose=purpose,
                    messages=messages,
                    generation={**dict(generation), "seed": int(seed)},
                    raw_output=str(content or ""),
                    reasoning_output=(str(reasoning) if reasoning is not None else None),
                    usage=ModelUsage(
                        input_tokens=(int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None),
                        output_tokens=(int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None),
                        reasoning_tokens=(int(details.reasoning_tokens) if details and getattr(details, "reasoning_tokens", None) is not None else None),
                    ),
                    latency_seconds=time.perf_counter() - started,
                    retry_count=retry_count,
                    request_status="response_received",
                    provider_request_id=str(getattr(response, "id", "") or "") or None,
                    finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
                    response_model=str(getattr(response, "model", "") or "") or None,
                    system_fingerprint=(
                        str(getattr(response, "system_fingerprint", "") or "") or None
                    ),
                )
            except Exception as exc:
                if retry_count < max_retries:
                    delay = backoff * (2**retry_count)
                    retry_count += 1
                    if delay > 0:
                        time.sleep(min(delay, 30.0))
                    continue
                error = _redact_error(exc, secret=os.environ.get(str(self.config.get("api_key_env", ""))))
                call = ModelCall(
                    call_id=call_id,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    role=self.role,
                    purpose=purpose,
                    messages=messages,
                    generation={**dict(generation), "seed": int(seed)},
                    raw_output=None,
                    reasoning_output=None,
                    usage=ModelUsage(),
                    latency_seconds=time.perf_counter() - started,
                    retry_count=retry_count,
                    request_status="request_failed",
                    error=f"{type(exc).__name__}: {error}",
                )
                raise ModelInvocationError(call.error or "model invocation failed", call) from None

    def close(self) -> None:
        self.client.close()


class ModelPool:
    """Named model registry used unchanged by single- and multi-model policies."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        run_id: str,
        generation_seed: int,
        max_steps: int = 1,
        max_calls_per_step: int = 8,
    ) -> None:
        self.config = dict(config)
        self.run_id = run_id
        self.generation_seed = int(generation_seed)
        self.max_steps = int(max_steps)
        self.max_calls_per_step = int(max_calls_per_step)
        if self.max_steps <= 0 or self.max_calls_per_step <= 0:
            raise ValueError("model call position limits must be positive")
        self.models: dict[str, OpenAICompatibleModel] = {}
        self.call_index = 0
        self.lock = threading.Lock()

    def _model(self, model_id: str) -> OpenAICompatibleModel:
        if model_id not in self.models:
            registry = dict(self.config.get("models") or {})
            if model_id not in registry:
                raise KeyError(f"unknown model_id: {model_id}")
            model_config = dict(registry[model_id])
            provider = str(model_config.get("provider", "openai_compatible"))
            if provider != "openai_compatible":
                raise ValueError(f"unsupported model provider: {provider}")
            self.models[model_id] = OpenAICompatibleModel(model_id, model_config)
        return self.models[model_id]

    def generate(
        self,
        model_id: str,
        messages: tuple[Mapping[str, str], ...],
        purpose: str,
        generation: Mapping[str, Any],
        *,
        position: ModelCallPosition | None = None,
    ) -> ModelCall:
        if position is None:
            with self.lock:
                self.call_index += 1
                index = self.call_index
            call_id = f"{self.run_id}:c{index:06d}"
            seed = self.generation_seed + index - 1
        else:
            offset = logical_call_offset(
                position,
                max_steps=self.max_steps,
                max_calls_per_step=self.max_calls_per_step,
            )
            call_id = logical_call_id(self.run_id, position)
            seed = self.generation_seed + offset
        try:
            model = self._model(model_id)
        except Exception as exc:
            raw = dict(dict(self.config.get("models") or {}).get(model_id) or {})
            error = f"{type(exc).__name__}: {_redact_error(exc)}"
            call = ModelCall(
                call_id=call_id,
                model_id=model_id,
                model_name=str(raw.get("model") or model_id),
                role=str(raw.get("role") or "unknown"),
                purpose=purpose,
                messages=messages,
                generation={**dict(generation), "seed": seed},
                raw_output=None,
                reasoning_output=None,
                usage=ModelUsage(),
                latency_seconds=0.0,
                retry_count=0,
                request_status="request_failed",
                error=error,
            )
            raise ModelInvocationError(error, call) from exc
        return model.generate(
            call_id=call_id,
            messages=messages,
            purpose=purpose,
            generation=generation,
            seed=seed,
        )

    def generation_for(
        self, model_id: str, base_generation: Mapping[str, Any]
    ) -> dict[str, Any]:
        registry = dict(self.config.get("models") or {})
        if model_id not in registry:
            raise KeyError(f"unknown model_id: {model_id}")
        result = dict(base_generation)
        result.update(dict(dict(registry[model_id]).get("generation") or {}))
        max_model_len = int(result["max_model_len"])
        max_input = int(result["max_input_tokens"])
        max_new = int(result["max_new_tokens"])
        if max_input + max_new > max_model_len:
            raise ValueError(
                f"model {model_id} input plus output limits exceed max_model_len"
            )
        if result.get("output_budget_policy") != "fixed_output":
            raise ValueError(
                f"model {model_id} output_budget_policy must be fixed_output"
            )
        return result

    def public_registry(
        self,
        base_generation: Mapping[str, Any] | None = None,
        *,
        model_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for model_id, raw in dict(self.config.get("models") or {}).items():
            if model_ids is not None and model_id not in model_ids:
                continue
            config = dict(raw)
            endpoint = str(config.get("base_url") or "")
            resolution_error = None
            if not endpoint:
                try:
                    _credential, endpoint = resolve_credentials(config)
                except Exception as exc:
                    resolution_error = f"{type(exc).__name__}: {_redact_error(exc)}"
            result[model_id] = {
                "model": config.get("model"),
                "server_id": config.get("server_id"),
                "role": config.get("role"),
                "reasoning_mode": config.get("reasoning_mode"),
                "provider": config.get("provider", "openai_compatible"),
                "endpoint_host": urlsplit(endpoint).netloc if endpoint else None,
                "credential_source": (
                    "environment" if config.get("api_key_env") else
                    "python_module" if config.get("credential_module") else
                    "none"
                ),
                "credential_module": config.get("credential_module"),
                "resolution_error": resolution_error,
                "request_options": {
                    "send_seed": bool(config.get("send_seed", True)),
                    "response_format": dict(config.get("response_format") or {}),
                    "extra_body": dict(config.get("extra_body") or {}),
                    "timeout_seconds": (
                        None
                        if config.get("timeout_seconds", 120.0) is None
                        else float(config.get("timeout_seconds", 120.0))
                    ),
                    "max_retries": int(config.get("max_retries", 2)),
                    "retry_backoff_seconds": float(
                        config.get("retry_backoff_seconds", 1.0)
                    ),
                    "proxy_mode": str(
                        config.get(
                            "proxy_mode",
                            "direct" if config.get("auth") == "none" else "http_env_only",
                        )
                    ),
                },
                "effective_generation": (
                    self.generation_for(model_id, base_generation)
                    if base_generation is not None
                    else dict(config.get("generation") or {})
                ),
            }
        if model_ids is not None and set(result) != set(model_ids):
            missing = sorted(set(model_ids) - set(result))
            raise KeyError("unknown model IDs requested for registry: " + ", ".join(missing))
        return result

    def close(self) -> None:
        for model in self.models.values():
            model.close()
        self.models.clear()
