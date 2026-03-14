import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

from .manifest import load_projects, save_projects


def _task_dir_for_path(project_path: Path) -> Path | None:
    """Return cfg/task directory for a project path, or None if not found."""
    for candidate in (project_path, project_path.parent, project_path.parent.parent):
        task_dir = candidate / "cfg" / "task"
        if task_dir.is_dir():
            return task_dir
    return None


def _discover_group(
    projects: dict[str, dict[str, dict[str, Any]]],
    *,
    group: str,
    target_key: str,
    project_type: str,
    enabled: bool,
    label: str,
    find_task_dir: bool = False,
) -> None:
    for entry_point in importlib.metadata.entry_points(group=group):
        spec = importlib.util.find_spec(entry_point.value)
        if entry_point.name in projects[target_key]:
            continue
        if spec is None or spec.origin is None:
            raise ImportError(
                f"Could not resolve entry point {entry_point.name} -> {entry_point.value}"
            )
        pkg_path = Path(spec.origin).parent.absolute()
        projects[target_key][entry_point.name] = {
            "value": entry_point.value,
            "path": str(pkg_path),
            "type": project_type,
            "enabled": enabled,
        }
        if find_task_dir:
            task_dir = _task_dir_for_path(pkg_path)
            projects[target_key][entry_point.name]["task_dir"] = str(task_dir) if task_dir is not None else None
        print(f"Discovered {label}: {entry_point.name} at {pkg_path}")


def discover_projects(enabled: bool = False) -> dict[str, dict[str, dict[str, Any]]]:
    projects = load_projects()
    _discover_group(
        projects,
        group="active_adaptation.projects",
        target_key="environment",
        project_type="environment",
        enabled=enabled,
        label="project",
        find_task_dir=True,
    )
    _discover_group(
        projects,
        group="active_adaptation.learning",
        target_key="learning",
        project_type="learning",
        enabled=enabled,
        label="learning module",
        find_task_dir=False, # no need to find task dir for learning modules
    )
    save_projects(projects)
    return projects
