"""CLI for managing active-adaptation extension projects."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer

from .project_loading.discovery import _task_dir_for_path, discover_projects
from .project_loading.manifest import PROJECTS_FILE, load_projects, save_projects

AA_REPO_ROOT = Path(__file__).resolve().parents[1]

# GitHub Python.gitignore (https://github.com/github/gitignore), CC0-1.0.
_PYTHON_GITIGNORE = """\
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

# C extensions
*.so

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
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
*.lcov
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
# For a library or package, you might want to ignore these files since the code is
# intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# However, in case of collaboration, if having platform-specific dependencies or dependencies
# having no cross-platform support, pipenv may install dependencies that don't work, or not
# install all needed dependencies.
# Pipfile.lock

# UV
# Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
# This is especially recommended for binary packages to ensure reproducibility, and is more
# commonly ignored for libraries.
# uv.lock

# poetry
# Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
# This is especially recommended for binary packages to ensure reproducibility, and is more
# commonly ignored for libraries.
# https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
# poetry.lock
# poetry.toml

# pdm
# Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
# pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
# https://pdm-project.org/en/latest/usage/project/#working-with-version-control
# pdm.lock
# pdm.toml
.pdm-python
.pdm-build/

# pixi
# Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
# pixi.lock
# Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
# in the .venv directory. It is recommended not to include this directory in version control.
.pixi/*
!.pixi/config.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule*
celerybeat.pid

# Redis
*.rdb
*.aof
*.pid

# RabbitMQ
mnesia/
rabbitmq/
rabbitmq-data/

# ActiveMQ
activemq-data/

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
# JetBrains specific template is maintained in a separate JetBrains.gitignore that can
# be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
# and can be added to the global gitignore or merged into this file. For a more nuclear
# option (not recommended) you can uncomment the following to ignore the entire idea folder.
# .idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
# Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore that
# can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
# and can be added to the global gitignore or merged into this file. However, if you prefer, you
# could uncomment the following to ignore the entire vscode folder
# .vscode/
# Temporary file for partial code execution
tempCodeRunnerFile.py

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/

# Streamlit
.streamlit/secrets.toml
"""

app = typer.Typer(
    no_args_is_help=True,
    help="Manage active-adaptation extension projects.",
)


def _fail(message: str, code: int = 1) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _set_enabled(name: str | None, enabled: bool) -> None:
    projects = load_projects()
    env = projects.setdefault("environment", {})
    learning = projects.setdefault("learning", {})

    if name is None:
        names = sorted(set(env) | set(learning))
        if not names:
            _fail("No projects in the manifest. Run `aa-project discover` first.")
    else:
        name = name.strip()
        if not name:
            _fail("Project name must be non-empty.")
        if name not in env and name not in learning:
            _fail(
                f"Unknown project {name!r}: not in environment or learning manifest. "
                "Run `aa-project discover` first."
            )
        names = [name]

    state = "enabled" if enabled else "disabled"
    for project_name in names:
        updated: list[str] = []
        if project_name in env:
            env[project_name]["enabled"] = enabled
            updated.append("environment")
        if project_name in learning:
            learning[project_name]["enabled"] = enabled
            updated.append("learning")
        print(f"Project {project_name!r} {state} ({', '.join(updated)}).")

    save_projects(projects)


def _git_toplevel(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return path


def _iter_manifest_project_paths(*, all_projects: bool, name: str | None) -> list[Path]:
    projects = load_projects()
    paths: dict[str, Path] = {}

    for category in ("environment", "learning"):
        for project_name, project_info in projects.get(category, {}).items():
            if name is not None and project_name != name:
                continue
            if name is None and not all_projects and not project_info.get("enabled", False):
                continue
            paths[project_name] = _git_toplevel(Path(project_info["path"]))

    if name is not None and name not in paths:
        _fail(
            f"Unknown project {name!r}: not in environment or learning manifest. "
            "Run `aa-project discover` first."
        )
    return list(paths.values())


def _repo_name_from_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    if "://" in cleaned:
        path = urlparse(cleaned).path
        name = Path(path).name
    elif ":" in cleaned and not cleaned.startswith("/"):
        # git@host:org/repo
        name = cleaned.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    else:
        name = Path(cleaned).name
    if not name:
        _fail(f"Could not infer repository name from {url!r}.")
    return name


def _create_project_scaffold(name: str, parent: Path) -> Path:
    if not name.replace("_", "").isalnum():
        _fail("Project name must be alphanumeric (and underscores only).")
    if name != name.lower():
        _fail("Project name must be lowercase.")

    root = parent.resolve() / name
    if root.exists():
        _fail(f"Directory already exists: {root}")

    root.mkdir(parents=True, exist_ok=True)

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

[tool.setuptools.packages.find]
where = ["src"]
include = ["{name}", "{name}_learning"]

[tool.setuptools.package-data]
"*" = ["**/*"]
'''
    (root / "pyproject.toml").write_text(pyproject)

    pkg_dir = root / "src" / name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        f"# {name} environment package. Register tasks and assets here.\n"
    )

    learning_dir = root / "src" / f"{name}_learning"
    learning_dir.mkdir(parents=True)
    (learning_dir / "__init__.py").write_text("# Learning scripts and entry points.\n")

    (root / "cfg" / "task").mkdir(parents=True)
    (root / "cfg" / "task" / ".gitkeep").write_text("")
    (root / "cfg" / "exp").mkdir(parents=True)
    (root / "cfg" / "exp" / ".gitkeep").write_text("")

    readme_path = root / "README.md"
    gitignore_path = root / ".gitignore"
    if not readme_path.exists():
        readme_path.write_text(
            f"# {name}\n\nActive-adaptation project. Add tasks under `cfg/task/`, experiments under `cfg/exp/`.\n"
        )
    else:
        print("  (kept existing README.md)")
    if not gitignore_path.exists():
        gitignore_path.write_text(_PYTHON_GITIGNORE)
    else:
        print("  (kept existing .gitignore)")

    _init_git_repo(root)

    print(f"Created project at: {root}")
    print(f"  - src/{name}/")
    print(f"  - src/{name}_learning/")
    print("  - pyproject.toml")
    print("  - cfg/task/, cfg/exp/")
    print("  - README.md, .gitignore")
    print("  - git repository (branch main)")
    return root


def _init_git_repo(root: Path) -> None:
    """Initialize ``root`` as a git repository whose default branch is ``main``."""
    result = subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        _fail(
            f"git init failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )


def _editable_install(project_root: Path, *, no_deps: bool) -> None:
    if not (project_root / "pyproject.toml").is_file():
        _fail(f"No pyproject.toml in {project_root}; cannot install.")
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(project_root)]
    if no_deps:
        cmd.append("--no-deps")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _fail(f"pip install failed with exit code {result.returncode}")


@app.command("create")
def create_cmd(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Project/package name (lowercase, alphanumeric + underscores)."),
    ],
    dir: Annotated[
        Path,
        typer.Option(
            "--dir",
            "-d",
            help="Parent directory for the new project folder (default: sibling aa-projects/).",
        ),
    ] = AA_REPO_ROOT.parent / "aa-projects",
    no_deps: Annotated[
        bool,
        typer.Option(
            "--no-deps/--deps",
            help="Editable-install without resolving dependencies (default: --no-deps; safer when active_adaptation is already installed).",
        ),
    ] = True,
    skip_discover: Annotated[
        bool,
        typer.Option("--skip-discover", help="Do not run discover after create/install."),
    ] = False,
) -> None:
    """Scaffold a project, editable-install it, and discover it (default: aa-projects/<name>)."""
    root = _create_project_scaffold(name.strip(), dir)
    _editable_install(root, no_deps=no_deps)
    if not skip_discover:
        discover_cmd()


@app.command("install")
def install_cmd(
    url: Annotated[str, typer.Argument(help="GitHub URL (HTTPS or SSH) of the project repository.")],
    dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Parent directory to clone into."),
    ] = Path("."),
    no_deps: Annotated[
        bool,
        typer.Option("--no-deps", help="Editable-install without resolving dependencies (safer for locked backend envs)."),
    ] = False,
    skip_discover: Annotated[
        bool,
        typer.Option("--skip-discover", help="Do not run discover after install."),
    ] = False,
) -> None:
    """Clone a project repo and editable-install it into the current Python environment."""
    parent = dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    repo_name = _repo_name_from_url(url)
    target = parent / repo_name
    if target.exists():
        _fail(f"Destination already exists: {target}")

    print(f"Cloning {url} -> {target}")
    clone = subprocess.run(["git", "clone", url, str(target)])
    if clone.returncode != 0:
        _fail(f"git clone failed with exit code {clone.returncode}")

    _editable_install(target, no_deps=no_deps)

    if not skip_discover:
        discover_cmd()


@app.command("discover")
def discover_cmd(
    enabled: Annotated[
        bool,
        typer.Option("--enabled", help="Mark newly discovered projects as enabled."),
    ] = False,
) -> None:
    """Scan installed entry points and update `.cache/projects.json`."""
    projects = discover_projects(enabled=enabled)

    for project_info in projects.get("environment", {}).values():
        task_dir = _task_dir_for_path(Path(project_info["path"]))
        project_info["task_dir"] = str(task_dir) if task_dir is not None else None

    save_projects(projects)
    print(f"Updated {PROJECTS_FILE}. Use `aa-project enable|disable` to toggle projects.")


@app.command("enable")
def enable_cmd(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Entry-point name in projects.json. Omit to enable all discovered projects."),
    ] = None,
) -> None:
    """Enable a project (or all projects) for import / Hydra cfg loading."""
    _set_enabled(name, True)


@app.command("disable")
def disable_cmd(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Entry-point name in projects.json. Omit to disable all discovered projects."),
    ] = None,
) -> None:
    """Disable a project (or all projects)."""
    _set_enabled(name, False)


@app.command("pull")
def pull_cmd(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Entry-point name to pull. Omit to pull active-adaptation and enabled projects."),
    ] = None,
    all: Annotated[
        bool,
        typer.Option("--all", help="When NAME is omitted, also pull disabled projects."),
    ] = False,
) -> None:
    """Run `git pull` for active-adaptation and/or extension projects."""
    if name is not None and all:
        _fail("Use either NAME or --all, not both.")

    if name is not None:
        project_paths = _iter_manifest_project_paths(all_projects=True, name=name.strip())
        print(f"Pulling project {name.strip()!r}")
    else:
        if all:
            print("Pulling active-adaptation and all discovered projects")
        else:
            print("Pulling active-adaptation and enabled projects")
        project_paths = [AA_REPO_ROOT, *_iter_manifest_project_paths(all_projects=all, name=None)]

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in project_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    for i, project_path in enumerate(unique_paths):
        print(f"[{i + 1}/{len(unique_paths)}] Pulling {project_path}")
        subprocess.run(["git", "branch"], cwd=project_path)
        result = subprocess.run(["git", "pull"], cwd=project_path)
        if result.returncode != 0:
            warnings.warn(
                f"Failed to pull {project_path} with result: {result.returncode}"
            )


def aa_list_tasks() -> None:
    """List task IDs from cfg/task in active-adaptation and discovered projects."""
    task_dirs: list[tuple[str, Path]] = []
    main_task_dir = AA_REPO_ROOT / "cfg" / "task"
    if main_task_dir.is_dir():
        task_dirs.append(("active-adaptation", main_task_dir))

    projects = load_projects()
    for project_name, project_info in projects.get("environment", {}).items():
        task_dir_str = project_info.get("task_dir")
        task_dir = (
            Path(task_dir_str)
            if task_dir_str
            else _task_dir_for_path(Path(project_info["path"]))
        )
        if task_dir is None or not task_dir.is_dir():
            continue
        if any(existing_task_dir == task_dir for _, existing_task_dir in task_dirs):
            continue
        task_dirs.append((project_name, task_dir))

    for source_name, task_dir in task_dirs:
        for yaml_path in sorted(task_dir.rglob("*.yaml")):
            rel = yaml_path.relative_to(task_dir)
            task_id = str(rel.with_suffix("")).replace("\\", "/")
            print(f"  {task_id}  (from {source_name})")


def aa_discover_projects(enabled: bool = False):
    """Compatibility entry point for existing project setup scripts."""
    return discover_cmd(enabled=enabled)


def aa_project() -> None:
    """Compatibility entry point for the pre-v0.8 executable name."""
    main()


def main() -> None:
    app()
