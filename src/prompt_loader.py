from importlib.metadata import version
from pathlib import Path


import yaml 

# pyrefly: ignore [missing-import]
from src.models import PromptConfig


def load_prompt(path: str) -> PromptConfig:
    prompt_path = Path(path)


    with prompt_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)


    return PromptConfig(
    version=data["version"],
    system_prompt=data["system_prompt"],
    examples=data.get("examples", []),
   )    