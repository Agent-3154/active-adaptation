import argparse
import subprocess
import warnings
import argparse
import importlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path

from .project_loading.manifest import CACHE_DIR, load_projects


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
        help="Pull all projects, including inactive ones",
    )
    args = parser.parse_args()

    if args.all:
        print("Pulling all projects, including inactive ones")
    else:
        print("Pulling active projects only")

    projects = load_projects()
    project_paths = set()
    project_paths.add(Path(__file__).resolve().parents[2])

    for category in ["environment", "learning"]:
        for project_info in projects[category].values():
            if args.all or project_info["enabled"]:
                project_paths.add(Path(project_info["path"]))

    for i, project_path in enumerate(project_paths):
        print(f"[{i + 1}/{len(project_paths)}] Pulling {project_path}")
        subprocess.run(["git", "branch"], cwd=project_path)
        result = subprocess.run(["git", "pull"], cwd=project_path)
        if result.returncode != 0:
            warnings.warn(
                f"Failed to pull {project_path} with result: {result.returncode}"
            )
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


def aa_recent_commands(n: int = 5):
    """
    List the n most recent commands (training runs / entry-point invocations)
    from the stored command history. Optionally filter by script name.
    """
    parser = argparse.ArgumentParser(description="List recent commands")
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=5,
        help="Number of recent commands to show (default: 5)",
    )
    parser.add_argument(
        "-s", "--script",
        action="append",
        default=[],
        metavar="NAME",
        help="Filter by script name (e.g. train_ppo, eval_run). Can be given multiple times (OR).",
    )
    args = parser.parse_args()
    n = max(1, args.num)
    script_filters = args.script

    history_file = CACHE_DIR / "command_history.json"
    if not history_file.exists():
        print("No command history found.")
        return

    history = json.loads(history_file.read_text())

    # Apply script filter: keep entries whose command line matches any of the script names
    if script_filters:
        filtered = []
        for entry in history:
            cmd = entry.get("args", [])
            line = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if any(s in line for s in script_filters):
                filtered.append(entry)
        history = filtered

    recent = history[-n:]
    if not recent:
        print("No command history found." if not script_filters else "No matching commands found.")
        return

    title = f"Last {len(recent)} command(s)"
    if script_filters:
        title += f" (script: {' | '.join(script_filters)})"
    print(f"\n{title}:\n")
    for i, entry in enumerate(reversed(recent), start=1):
        ts = entry.get("timestamp", "?")
        cmd = entry.get("args", [])
        line = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        print(f"  {i}. [{ts}]")
        print(f"     {line}")
        print()


def aa_create_project():
    """
    Create a new active-adaptation project scaffold: packages {name}/ and {name}_learning/,
    pyproject.toml with entry points, cfg/task and cfg/exp, README, and .gitignore.
    Usage: aa-create-project -n myproject [-d /path/to/parent]
    """
    parser = argparse.ArgumentParser(
        description="Create a new active-adaptation project scaffold",
    )
    parser.add_argument(
        "-n", "--name",
        required=True,
        metavar="NAME",
        help="Project/package name (e.g. myproject → myproject/ and myproject_learning/)",
    )
    parser.add_argument(
        "-d", "--dir",
        default=".",
        metavar="PATH",
        help="Parent directory in which to create the project folder (default: current directory)",
    )
    args = parser.parse_args()

    name = args.name.strip()
    if not name.replace("_", "").isalnum():
        raise SystemExit("Project name must be alphanumeric (and underscores only).")
    if name != name.lower():
        raise SystemExit("Project name must be lowercase.")

    parent = Path(args.dir).resolve()
    root: Path = parent / name
    if root.exists():
        raise SystemExit(f"Directory already exists: {root}")

    root.mkdir(parents=True, exist_ok=True)

    # pyproject.toml
    pyproject = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "active_adaptation",
]

[project.entry-points."active_adaptation.projects"]
{name} = "{name}"

[project.entry-points."active_adaptation.learning"]
{name} = "{name}_learning"

[tool.setuptools.package-data]
"*" = ["**/*"]
'''
    (root / "pyproject.toml").write_text(pyproject)

    # Main package
    pkg_dir: Path = root / "src" / name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        f"# {name} environment package. Register tasks and assets here.\n"
    )

    # Learning package
    learning_dir = root / "src" / f"{name}_learning"
    learning_dir.mkdir(parents=True)
    (learning_dir / "__init__.py").write_text(
        "# Learning scripts and entry points.\n"
    )

    # cfg/task and cfg/exp
    (root / "cfg" / "task").mkdir(parents=True)
    (root / "cfg" / "task" / ".gitkeep").write_text("")
    (root / "cfg" / "exp").mkdir(parents=True)
    (root / "cfg" / "exp" / ".gitkeep").write_text("")

    # README and .gitignore: do not overwrite if present (e.g. existing git repo)
    readme_path = root / "README.md"
    gitignore_path = root / ".gitignore"
    if not readme_path.exists():
        readme_path.write_text(
            f"# {name}\n\nActive-adaptation project. Add tasks under `cfg/task/`, experiments under `cfg/exp/`.\n"
        )
    else:
        print(f"  (kept existing README.md)")
    if not gitignore_path.exists():
        gitignore_path.write_text("""# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
*.egg

# robot assets
*.usd
*.xml
*.urdf

""")
    else:
        print(f"  (kept existing .gitignore)")

    print(f"Created project at: {root}")
    print(f"  - {name}/")
    print(f"  - {name}_learning/")
    print(f"  - pyproject.toml")
    print(f"  - cfg/task/, cfg/exp/")
    print(f"  - README.md, .gitignore")
