import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / ".cache"
PROJECTS_FILE = CACHE_DIR / "projects.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def default_projects_manifest() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "environment": {},
        "learning": {},
    }


def load_projects() -> dict[str, dict[str, dict[str, Any]]]:
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return default_projects_manifest()


def save_projects(projects: dict[str, dict[str, dict[str, Any]]]) -> None:
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2))
