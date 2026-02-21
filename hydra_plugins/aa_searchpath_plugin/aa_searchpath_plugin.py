from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins

class ActiveAdaptationSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        from active_adaptation import _CONFIG_SEARCH_PATHS
        for path in _CONFIG_SEARCH_PATHS:
            search_path.append(
                provider="aa_searchpath_plugin", path=f"file://{path}"
            )

Plugins.instance().register(ActiveAdaptationSearchPathPlugin)

