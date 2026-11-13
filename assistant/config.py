from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any
from urllib.parse import urlsplit

from assistant.context import POLICIES


_FIELDS = {
    "model": {
        "name",
        "host",
        "num_ctx",
        "num_predict",
        "temperature",
        "seed",
        "think",
        "timeout_seconds",
    },
    "assistant": {"system_prompt"},
    "context": {"policy", "recent_turns"},
    "trace": {"directory"},
}
_THINKING_MODES = {"off", "answers", "maintenance", "all"}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    host: str
    num_ctx: int
    num_predict: int
    temperature: float
    seed: int
    think: str
    timeout_seconds: int


@dataclass(frozen=True)
class AssistantConfig:
    system_prompt: str


@dataclass(frozen=True)
class ContextConfig:
    policy: str
    recent_turns: int


@dataclass(frozen=True)
class TraceConfig:
    directory: Path


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    assistant: AssistantConfig
    context: ContextConfig
    trace: TraceConfig


def _invalid(field: str, value: Any, requirement: str) -> ValueError:
    return ValueError(
        f"Invalid configuration field {field}={value!r}: {requirement}"
    )


def _validate_shape(raw: Any) -> None:
    if not isinstance(raw, Mapping):
        raise _invalid("configuration", raw, "must be a TOML table")

    unknown_sections = set(raw) - set(_FIELDS)
    if unknown_sections:
        section = sorted(unknown_sections)[0]
        raise _invalid(section, raw[section], "unknown top-level field")

    for section, expected_fields in _FIELDS.items():
        if section not in raw:
            raise ValueError(f"Missing required configuration section [{section}]")
        values = raw[section]
        if not isinstance(values, Mapping):
            raise _invalid(section, values, "must be a TOML table")
        unknown_fields = set(values) - expected_fields
        if unknown_fields:
            field = sorted(unknown_fields)[0]
            raise _invalid(
                f"{section}.{field}", values[field], "unknown field"
            )
        missing_fields = expected_fields - set(values)
        if missing_fields:
            field = sorted(missing_fields)[0]
            raise ValueError(
                f"Missing required configuration field {section}.{field}"
            )


def _nonempty_string(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(field, value, "must be a non-empty string")
    return value


def _integer(field: str, value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise _invalid(field, value, "must be an integer")
    if minimum is not None and value < minimum:
        comparison = "positive" if minimum == 1 else f"at least {minimum}"
        raise _invalid(field, value, f"must be a {comparison} integer")
    return value


def _validate(raw: Mapping[str, Any]) -> Config:
    model = raw["model"]
    assistant = raw["assistant"]
    context = raw["context"]
    trace = raw["trace"]

    name = _nonempty_string("model.name", model["name"])
    host = _nonempty_string("model.host", model["host"])
    try:
        parsed_host = urlsplit(host)
        valid_host = parsed_host.scheme in {"http", "https"} and bool(
            parsed_host.hostname
        )
    except ValueError:
        valid_host = False
    if not valid_host:
        raise _invalid(
            "model.host", host, "must be an HTTP(S) URL with a host"
        )

    num_ctx = _integer("model.num_ctx", model["num_ctx"], minimum=1)
    num_predict = _integer(
        "model.num_predict", model["num_predict"], minimum=1
    )
    timeout_seconds = _integer(
        "model.timeout_seconds", model["timeout_seconds"], minimum=1
    )

    temperature = model["temperature"]
    if (
        type(temperature) not in {int, float}
        or not math.isfinite(temperature)
        or not 0.0 <= temperature <= 2.0
    ):
        raise _invalid(
            "model.temperature",
            temperature,
            "must be a finite number from 0.0 to 2.0",
        )

    seed = _integer("model.seed", model["seed"])
    think = model["think"]
    if not isinstance(think, str) or think not in _THINKING_MODES:
        choices = ", ".join(sorted(_THINKING_MODES))
        raise _invalid("model.think", think, f"must be one of: {choices}")

    system_prompt = _nonempty_string(
        "assistant.system_prompt", assistant["system_prompt"]
    )
    policy = context["policy"]
    if not isinstance(policy, str) or policy not in POLICIES:
        choices = ", ".join(sorted(POLICIES))
        raise _invalid("context.policy", policy, f"must be one of: {choices}")
    recent_turns = _integer(
        "context.recent_turns", context["recent_turns"], minimum=0
    )

    directory_value = _nonempty_string("trace.directory", trace["directory"])
    if "\0" in directory_value:
        raise _invalid("trace.directory", directory_value, "must be a valid path")
    directory = Path(directory_value)
    if directory.exists() and not directory.is_dir():
        raise _invalid(
            "trace.directory", directory_value, "must refer to a directory"
        )

    return Config(
        model=ModelConfig(
            name=name,
            host=host,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=float(temperature),
            seed=seed,
            think=think,
            timeout_seconds=timeout_seconds,
        ),
        assistant=AssistantConfig(system_prompt=system_prompt),
        context=ContextConfig(policy=policy, recent_turns=recent_turns),
        trace=TraceConfig(directory=directory),
    )


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    _validate_shape(raw)
    return _validate(raw)
