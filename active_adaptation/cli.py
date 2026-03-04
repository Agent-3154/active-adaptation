from active_adaptation import CACHE_DIR
from pathlib import Path
import json
import subprocess
import warnings
import argparse
import importlib
import importlib.util
import importlib.metadata


def aa_pull():
    """
    Runs `git pull` for active-adaptation and all projects discovered and listed in `projects.json`.
    
    If `--all` is passed, it will pull all projects, including inactive ones.
    
    Returns:
        bool: True if all pulls succeeded, False otherwise.
    """
    parser = argparse.ArgumentParser(description="Update all projects")
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Pull all projects, including inactive ones"
    )
    args = parser.parse_args()

    if args.all:
        print("Pulling all projects, including inactive ones")
    else:
        print("Pulling active projects only")

    projects_file = CACHE_DIR / "projects.json"
    if projects_file.exists():
        projects = json.loads(projects_file.read_text())
    else:
        projects = {
            "environment": {},
            "learning": {},
        }
    project_paths = set()
    project_paths.add(Path(__file__).parent) # active-adaptation itself

    for category in ["environment", "learning"]:
        for project_name, project_info in projects[category].items():
            if args.all or project_info["enabled"]:
                project_paths.add(Path(project_info["path"]))

    for i, project_path in enumerate(project_paths):
        print(f"[{i+1}/{len(project_paths)}] Pulling {project_path}")
        subprocess.run(["git", "branch"], cwd=project_path)
        result = subprocess.run(
            ["git", "pull"], cwd=project_path
        )
        if result.returncode != 0:
            warnings.warn(f"Failed to pull {project_path} with result: {result.returncode}")
            print(result.stderr)
        

def aa_discover_projects(enabled: bool = False):
    projects_file = CACHE_DIR / "projects.json"
    if projects_file.exists():
        projects = json.loads(projects_file.read_text())
    else:
        projects = {
            "environment": {},
            "learning": {},
        }
    for entry_point in importlib.metadata.entry_points(group="active_adaptation.projects"):
        # get the module path
        spec = importlib.util.find_spec(entry_point.value)
        try:
            # note that `value` may differ from `name`
            pkg_path = Path(spec.origin).parent.absolute()
        except Exception as e:
            raise ValueError(f"Entrypoint {str(entry_point)} is invalid.") from e

        env_projects = projects.setdefault("environment", {})
        project_info = env_projects.setdefault(
            entry_point.name,
            {
                "value": entry_point.value,
                "path": str(pkg_path),
                "type": "environment",
                "enabled": enabled,
            },
        )
        # Ensure path/value stay in sync with the entry point
        project_info.setdefault("value", entry_point.value)
        project_info.setdefault("path", str(pkg_path))

        task_dir = _task_dir_for_path(Path(project_info["path"]))
        project_info["task_dir"] = str(task_dir) if task_dir is not None else None
        print(f"Discovered project: {entry_point.name} at {project_info['path']}")

    for entry_point in importlib.metadata.entry_points(group="active_adaptation.learning"):
        # get the module path
        spec = importlib.util.find_spec(entry_point.value)
        try:
            # note that `value` may differ from `name`
            pkg_path = Path(spec.origin).parent.absolute()
        except Exception as e:
            raise ValueError(f"Entrypoint {str(entry_point)} is invalid.") from e

        learning_projects = projects.setdefault("learning", {})
        project_info = learning_projects.setdefault(
            entry_point.name,
            {
                "value": entry_point.value,
                "path": str(pkg_path),
                "type": "learning",
                "enabled": enabled,
            },
        )
        # Ensure path/value stay in sync with the entry point
        project_info.setdefault("value", entry_point.value)
        project_info.setdefault("path", str(pkg_path))
        print(f"Discovered learning module: {entry_point.name} at {project_info['path']}")
    projects_file.write_text(json.dumps(projects, indent=2))
    print(f"Modify {projects_file} to enable/disable projects.")


def _task_dir_for_path(project_path: Path) -> Path | None:
    """Return cfg/task directory for a project path, or None if not found."""
    for candidate in (project_path, project_path.parent, project_path.parent.parent):
        task_dir = candidate / "cfg" / "task"
        if task_dir.is_dir():
            return task_dir
    return None


def aa_list_tasks():
    """
    List task names from YAML files under cfg/task in active-adaptation and in
    all projects from projects.json. Task names preserve the directory prefix
    (e.g. "G1/G1LocoFlat" instead of "G1LocoFlat").
    """
    # active-adaptation's own cfg/task
    repo_root = Path(__file__).parent.parent
    task_dirs: list[tuple[str, Path]] = []
    main_task_dir = repo_root / "cfg" / "task"
    if main_task_dir.is_dir():
        task_dirs.append(("active-adaptation", main_task_dir))

    # cfg/task from each project in projects.json
    projects_file = CACHE_DIR / "projects.json"
    if projects_file.exists():
        projects = json.loads(projects_file.read_text())
        for project_name, project_info in projects.get("environment", {}).items():
            task_dir_str = project_info.get("task_dir")
            if task_dir_str:
                task_dir = Path(task_dir_str)
            if task_dir is not None and task_dir.is_dir() and not any(
                d == task_dir for _, d in task_dirs
            ):
                task_dirs.append((project_name, task_dir))

    for source_name, task_dir in task_dirs:
        for yaml_path in sorted(task_dir.rglob("*.yaml")):
            rel = yaml_path.relative_to(task_dir)
            task_id = str(rel.with_suffix("")).replace("\\", "/")
            print(f"  {task_id}  (from {source_name})")

