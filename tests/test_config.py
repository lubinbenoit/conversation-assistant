import json
from pathlib import Path

import pytest

from assistant.config import load_config


VALID_CONFIG = '''[model]
name = "test"
host = "http://localhost:11434"
num_ctx = 4096
num_predict = 500
temperature = 0.2
seed = 42
think = "off"
timeout_seconds = 300

[assistant]
system_prompt = "test"

[context]
policy = "stateless"
recent_turns = 3

[trace]
directory = "logs"
'''


def write_config(
    path: Path,
    *,
    replace: tuple[str, str] | None = None,
    remove: str | None = None,
    suffix: str = "",
) -> None:
    content = VALID_CONFIG
    if replace is not None:
        content = content.replace(*replace)
    if remove is not None:
        content = content.replace(f"{remove}\n", "")
    path.write_text(content + suffix, encoding="utf-8")


def test_valid_configuration_loads_every_field(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path)

    config = load_config(path)

    assert config.model.name == "test"
    assert config.model.host == "http://localhost:11434"
    assert config.model.num_ctx == 4096
    assert config.model.num_predict == 500
    assert config.model.temperature == 0.2
    assert config.model.seed == 42
    assert config.model.think == "off"
    assert config.model.timeout_seconds == 300
    assert config.assistant.system_prompt == "test"
    assert config.context.policy == "stateless"
    assert config.context.recent_turns == 3
    assert config.trace.directory == Path("logs")


@pytest.mark.parametrize("mode", ["off", "answers", "maintenance", "all"])
def test_thinking_modes_are_accepted(tmp_path, mode):
    path = tmp_path / "config.toml"
    write_config(path, replace=('think = "off"', f'think = "{mode}"'))

    assert load_config(path).model.think == mode


@pytest.mark.parametrize(
    ("original", "replacement", "field"),
    [
        ('name = "test"', 'name = "  "', "model.name"),
        (
            'host = "http://localhost:11434"',
            'host = "ftp://localhost:11434"',
            "model.host",
        ),
        ("num_ctx = 4096", "num_ctx = 0", "model.num_ctx"),
        ("num_ctx = 4096", "num_ctx = true", "model.num_ctx"),
        ("num_predict = 500", "num_predict = -1", "model.num_predict"),
        ("temperature = 0.2", "temperature = -0.1", "model.temperature"),
        ("temperature = 0.2", "temperature = 2.1", "model.temperature"),
        ("temperature = 0.2", 'temperature = "cold"', "model.temperature"),
        ("seed = 42", "seed = true", "model.seed"),
        ('think = "off"', 'think = "on"', "model.think"),
        ("timeout_seconds = 300", "timeout_seconds = 0", "model.timeout_seconds"),
        ('system_prompt = "test"', 'system_prompt = ""', "assistant.system_prompt"),
        ('policy = "stateless"', 'policy = "unknown"', "context.policy"),
        ("recent_turns = 3", "recent_turns = -1", "context.recent_turns"),
        ("recent_turns = 3", "recent_turns = true", "context.recent_turns"),
        ('directory = "logs"', 'directory = ""', "trace.directory"),
    ],
)
def test_invalid_values_identify_the_full_field(
    tmp_path, original, replacement, field
):
    path = tmp_path / "config.toml"
    write_config(path, replace=(original, replacement))

    with pytest.raises(ValueError, match=rf"{field}="):
        load_config(path)


def test_trace_directory_cannot_be_an_existing_file(tmp_path):
    trace_file = tmp_path / "not-a-directory"
    trace_file.write_text("occupied", encoding="utf-8")
    path = tmp_path / "config.toml"
    write_config(
        path,
        replace=('directory = "logs"', f"directory = {json.dumps(str(trace_file))}"),
    )

    with pytest.raises(ValueError, match="trace.directory=.*must refer to a directory"):
        load_config(path)


def test_missing_field_has_an_understandable_error(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, remove="seed = 42")

    with pytest.raises(
        ValueError, match="Missing required configuration field model.seed"
    ):
        load_config(path)


def test_missing_section_has_an_understandable_error(tmp_path):
    path = tmp_path / "config.toml"
    content = VALID_CONFIG.split("[trace]", 1)[0]
    path.write_text(content, encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Missing required configuration section \[trace\]"
    ):
        load_config(path)


def test_unknown_field_reports_its_name_and_value(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, suffix="unexpected = true\n")

    with pytest.raises(
        ValueError, match="trace.unexpected=True: unknown field"
    ):
        load_config(path)


def test_unknown_section_reports_its_name(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, suffix="\n[extra]\nenabled = true\n")

    with pytest.raises(ValueError, match="extra=.*unknown top-level field"):
        load_config(path)
