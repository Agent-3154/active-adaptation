import argparse
import subprocess
import warnings
from pathlib import Path

from .manifest import CACHE_DIR, load_projects


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
