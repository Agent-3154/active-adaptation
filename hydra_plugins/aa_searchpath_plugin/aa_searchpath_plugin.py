from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from pathlib import Path
import json

class ActiveAdaptationSearchPathPlugin(SearchPathPlugin):
    
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        CONFIG_SEARCH_PATHS = []
        projects_file = Path(__file__).parent.parent.parent / ".cache" / "projects.json"
        
        if projects_file.exists():
            projects = json.loads(projects_file.read_text())
            for project_name, project_info in projects.items():
                if project_info["enabled"]:
                    pkg_path = Path(project_info["path"])
                    if pkg_path.parent.name == "src":
                        path = pkg_path.parent.parent / "cfg"
                    else:
                        path = pkg_path.parent / "cfg"
                    CONFIG_SEARCH_PATHS.append(str(path))
        for path in CONFIG_SEARCH_PATHS:
            search_path.append(
                provider="aa_searchpath_plugin", path=f"file://{path}"
            )

