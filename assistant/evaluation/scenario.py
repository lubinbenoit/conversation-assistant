from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    turns: list[str]


def load_scenario(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    turns = data.get("turns", [])
    if not turns or not all(isinstance(x, str) for x in turns):
        raise ValueError("Scenario must contain a non-empty list of string turns.")
    return Scenario(data["name"], data.get("description", ""), turns)
