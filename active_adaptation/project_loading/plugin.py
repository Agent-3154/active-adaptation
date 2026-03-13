from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from pathlib import Path
from typing import Any

from .manifest import PROJECTS_FILE, load_projects
from .runtime import import_learning_modules


def collect_environment_cfg_paths(
    projects: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    config_search_paths: list[str] = []
    for project_info in projects["environment"].values():
        if not project_info["enabled"]:
            continue
        pkg_path = Path(project_info["path"])
        if pkg_path.parent.name == "src":
            path = pkg_path.parent.parent / "cfg"
        else:
            path = pkg_path.parent / "cfg"
        config_search_paths.append(str(path))
    return config_search_paths


class ActiveAdaptationSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if not PROJECTS_FILE.exists():
            return

        projects = load_projects()
        config_search_paths = collect_environment_cfg_paths(projects)
        import_learning_modules(projects)

        for path in config_search_paths:
            search_path.append(provider="aa_searchpath_plugin", path=f"file://{path}")
