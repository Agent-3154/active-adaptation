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


def _normalize_logger_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value == "...":
        return None
    return value


def resolve_logger_defaults(
    projects: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve optional (COMET_API_KEY, COMET_PROJECT, COMET_WORKSPACE).

    Also accepts legacy ``wandb`` / ``WANDB_*`` keys in ``projects.json`` as aliases.
    """
    if projects is None:
        if not PROJECTS_FILE.exists():
            return None, None, None
        try:
            projects = load_projects()
        except Exception as exc:
            logging.warning("Failed to load projects manifest for logger defaults: %s", exc)
            return None, None, None

    top = projects.get("logger") or projects.get("comet") or projects.get("wandb") or {}
    top_api_key = _normalize_logger_value(
        top.get("COMET_API_KEY") or top.get("WANDB_API_KEY")
    )
    top_project = _normalize_logger_value(
        top.get("COMET_PROJECT") or top.get("WANDB_PROJECT")
    )
    top_workspace = _normalize_logger_value(
        top.get("COMET_WORKSPACE") or top.get("WANDB_ENTITY")
    )

    enabled_env_projects = [
        (name, info)
        for name, info in projects.get("environment", {}).items()
        if info.get("enabled", False)
    ]

    def _pick_enabled_project_value(*keys: str) -> Optional[str]:
        values = []
        for name, info in enabled_env_projects:
            for key in keys:
                value = _normalize_logger_value(info.get(key))
                if value is not None:
                    values.append((name, value))
                    break

        if not values:
            return None

        unique_values = sorted({value for _, value in values})
        if len(unique_values) > 1:
            owners = ", ".join(f"{name}={value}" for name, value in values)
            logging.warning(
                "Ignoring logger defaults from projects.json due to conflicting values: %s",
                owners,
            )
            return None
        return unique_values[0]

    api_key = top_api_key or _pick_enabled_project_value(
        "COMET_API_KEY", "WANDB_API_KEY"
    )
    project = top_project or _pick_enabled_project_value(
        "COMET_PROJECT", "WANDB_PROJECT"
    )
    workspace = top_workspace or _pick_enabled_project_value(
        "COMET_WORKSPACE", "WANDB_ENTITY"
    )
    return api_key, project, workspace


def import_learning_modules(
    projects: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for project_name, project_info in projects["learning"].items():
        if not project_info["enabled"]:
            continue
        import_module_from_project(project_info)
        print(f"Importing learning module: {project_name} from {project_info['path']}")
