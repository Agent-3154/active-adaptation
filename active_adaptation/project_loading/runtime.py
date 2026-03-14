import importlib
import sys
from pathlib import Path
from typing import Any

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


def import_learning_modules(
    projects: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for project_name, project_info in projects["learning"].items():
        if not project_info["enabled"]:
            continue
        import_module_from_project(project_info)
        print(f"Importing learning module: {project_name} from {project_info['path']}")
