"""Portable configuration loading for the benchmark runner.

Scientific settings use YAML mappings. Runtime placeholders resolve deployment
paths and service settings.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sysconfig
from pathlib import Path
from typing import Any, Mapping, Sequence


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_FORBIDDEN_FIELDS = {"api_key", "access_token", "authorization", "credential_module"}


def repository_root() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "configs/base.yaml").is_file():
        return checkout
    installed = Path(sysconfig.get_path("data")) / "share/mica"
    if (installed / "configs/base.yaml").is_file():
        return installed
    raise FileNotFoundError("MICA configuration assets are missing from the installation")


def configuration_path(path: str | Path) -> Path:
    """Resolve local configuration files or the installed configuration assets."""
    source = Path(path).expanduser()
    if source.is_file() or source.is_absolute():
        return source
    if source.parts and source.parts[0] == "configs" and ".." not in source.parts:
        return repository_root() / source
    return source


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_configuration(path: str | Path) -> dict[str, Any]:
    source = configuration_path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
    else:
        import yaml

        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration must be a mapping: {source.name}")
    _reject_secrets(value)
    return dict(value)


def _reject_secrets(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _FORBIDDEN_FIELDS:
                raise ValueError(f"{location}: use environment-variable names, not credentials")
            _reject_secrets(child, location)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_secrets(child, prefix)


def unresolved_environment(value: Any) -> tuple[str, ...]:
    names: set[str] = set()
    if isinstance(value, str):
        names.update(match.group(1) for match in _ENV.finditer(value) if match.group(2) is None and not os.environ.get(match.group(1)))
    elif isinstance(value, Mapping):
        for item in value.values():
            names.update(unresolved_environment(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.update(unresolved_environment(item))
    return tuple(sorted(names))


def expand_environment(value: Any, *, strict: bool = True) -> Any:
    """Expand deployment placeholders without reading credential variables."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name == "API_KEY" or name.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
                raise ValueError("credential values must not be interpolated into configuration")
            found = os.environ.get(name)
            if found:
                return found
            if default is not None:
                return default
            if strict:
                raise ValueError(f"required environment variable is unset: {name}")
            return match.group(0)

        return _ENV.sub(replace, value)
    if isinstance(value, Mapping):
        return {str(key): expand_environment(child, strict=strict) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_environment(child, strict=strict) for child in value]
    return value


def load_configuration(
    paths: Sequence[str | Path], *, overrides: Mapping[str, Any] | None = None,
    strict_environment: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        result = deep_merge(result, read_configuration(path))
    if overrides:
        _reject_secrets(overrides)
        result = deep_merge(result, overrides)
    result = expand_environment(result, strict=strict_environment)
    validate_configuration(result)
    return result


def validate_configuration(config: Mapping[str, Any]) -> None:
    _reject_secrets(config)
    benchmark = dict(config.get("benchmark") or {})
    if benchmark.get("name") not in {"alfworld", "webshop", "scienceworld"}:
        raise ValueError("benchmark.name must be alfworld, webshop, or scienceworld")
    max_steps = int(benchmark.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("benchmark.max_steps must be positive")
    interface = dict(config.get("interface") or {})
    if interface.get("type", "mcp") not in {"native", "mcp", "cli"}:
        raise ValueError("interface.type must be native, mcp, or cli")
    if int(interface.get("max_retries", 1)) != 1:
        raise ValueError("interface.max_retries must be 1 for the controlled protocol")
    execution = dict(config.get("execution") or {})
    seeds = dict(config.get("seeds") or {})
    for name in ("environment", "generation", "sampling"):
        if type(seeds.get(name)) is not int:
            raise ValueError(f"seeds.{name} must be an explicit integer")
    if int(execution.get("max_decisions", max_steps)) < max_steps:
        raise ValueError("execution.max_decisions must cover benchmark.max_steps")
    if int(execution.get("max_model_calls", max_steps * 2)) <= 0:
        raise ValueError("execution.max_model_calls must be positive")
    generation = dict(config.get("generation") or {})
    model = dict(config.get("model") or {})
    generation.update(dict(model.get("generation") or {}))
    for name in ("max_input_tokens", "max_new_tokens", "max_model_len"):
        if int(generation.get(name, 0)) <= 0:
            raise ValueError(f"generation.{name} must be positive")
    if int(generation["max_input_tokens"]) + int(generation["max_new_tokens"]) > int(generation["max_model_len"]):
        raise ValueError("input and output allowances exceed generation.max_model_len")
    if generation.get("output_budget_policy") != "fixed_output":
        raise ValueError("generation.output_budget_policy must be fixed_output")
    if not model.get("model_id") or not model.get("model"):
        raise ValueError("model.model_id and model.model are required")
    if model.get("provider", "openai_compatible") != "openai_compatible":
        raise ValueError("only the openai_compatible provider is supported")
    if model.get("auth", "required") not in {"none", "required"}:
        raise ValueError("model.auth must be none or required")
    if model.get("auth", "required") != "none" and not model.get("api_key_env"):
        raise ValueError("authenticated models require model.api_key_env")
    if not model.get("base_url") and not model.get("base_url_env"):
        raise ValueError("model.base_url_env or model.base_url is required")
    prompt_id = str(dict(config.get("prompt") or {}).get("prompt_id") or "")
    if not prompt_id or Path(prompt_id).name != prompt_id:
        raise ValueError("prompt.prompt_id must be a directory name under prompts")
    if config.get("dart"):
        from .controllers import DARTSettings

        DARTSettings.resolve(config["dart"], benchmark["name"], interface.get("type", "mcp"))


def resolve_runtime_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    result = expand_environment(copy.deepcopy(dict(config)), strict=True)
    model = dict(result["model"])
    endpoint_variable = str(model.get("base_url_env") or "")
    if not model.get("base_url") and endpoint_variable:
        endpoint = os.environ.get(endpoint_variable)
        if not endpoint:
            raise ValueError(f"required endpoint variable is unset: {endpoint_variable}")
        model["base_url"] = endpoint
    result["model"] = model
    validate_configuration(result)
    return result


def load_task_records(path: str | Path, *, benchmark: str, resolve_environment: bool = False) -> list[dict[str, Any]]:
    value = json.loads(configuration_path(path).read_text(encoding="utf-8"))
    rows = value.get("tasks") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("task file must contain a nonempty list of TaskSpec records")
    tasks: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("task files must contain full TaskSpec records, not bare IDs")
        task = dict(row)
        identifier = str(task.get("task_id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError("task identifiers must be nonempty and unique")
        if task.get("benchmark") != benchmark:
            raise ValueError(f"task {identifier} does not belong to the configured benchmark")
        if not task.get("family") or not task.get("split"):
            raise ValueError(f"task {identifier} must declare family and split")
        identifiers.add(identifier)
        tasks.append(expand_environment(task, strict=resolve_environment))
    return tasks
