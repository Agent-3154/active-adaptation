import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from .discovery import discover_projects
from .manifest import PROJECTS_FILE, load_projects


def import_module_from_project(project_info: dict[str, Any]) -> None:
    project_parent = str(Path(project_info["path"]).parent)
    sys.path.insert(0, project_parent)
    try:
        importlib.import_module(project_info["value"])
    finally:
        if sys.path and sys.path[0] == project_parent:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(project_parent)
            except ValueError:
                pass


def import_environment_projects(
    projects: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if projects is None:
        projects = load_projects() if PROJECTS_FILE.exists() else discover_projects(enabled=False)

    for project_name, project_info in projects["environment"].items():
        if project_info["enabled"]:
            print(f"Importing project: {project_name} from {project_info['path']}")
            import_module_from_project(project_info)

    return projects


def _normalize_wandb_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value == "...":
        return None
    return value


def resolve_wandb_defaults(
    projects: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve optional (WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY)."""
    if projects is None:
        if not PROJECTS_FILE.exists():
            return None, None, None
        try:
            projects = load_projects()
        except Exception as exc:
            logging.warning("Failed to load projects manifest for WandB defaults: %s", exc)
            return None, None, None

    top_level_wandb = projects.get("wandb", {})
    top_api_key = _normalize_wandb_value(top_level_wandb.get("WANDB_API_KEY"))
    top_project = _normalize_wandb_value(top_level_wandb.get("WANDB_PROJECT"))
    top_entity = _normalize_wandb_value(top_level_wandb.get("WANDB_ENTITY"))

    enabled_env_projects = [
        (name, info)
        for name, info in projects.get("environment", {}).items()
        if info.get("enabled", False)
    ]

    def _pick_enabled_project_value(key: str) -> Optional[str]:
        values = []
        for name, info in enabled_env_projects:
            value = _normalize_wandb_value(info.get(key))
            if value is not None:
                values.append((name, value))

        if not values:
            return None

        unique_values = sorted({value for _, value in values})
        if len(unique_values) > 1:
            owners = ", ".join(f"{name}={value}" for name, value in values)
            logging.warning(
                "Ignoring %s from projects.json due to conflicting values across enabled projects: %s",
                key,
                owners,
            )
            return None
        return unique_values[0]

    api_key = (
        top_api_key
        if top_api_key is not None
        else _pick_enabled_project_value("WANDB_API_KEY")
    )
    project = (
        top_project
        if top_project is not None
        else _pick_enabled_project_value("WANDB_PROJECT")
    )
    entity = (
        top_entity
        if top_entity is not None
        else _pick_enabled_project_value("WANDB_ENTITY")
    )
    return api_key, project, entity


def import_learning_modules(
    projects: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for project_name, project_info in projects["learning"].items():
        if not project_info["enabled"]:
            continue
        import_module_from_project(project_info)
        print(f"Importing learning module: {project_name} from {project_info['path']}")
