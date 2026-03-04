import inspect
import warnings
import active_adaptation as aa
from collections import defaultdict


class Registry:
    """
    A singleton class implementing a global registry for configurations.
    Ensures unique keys and provides methods for managing configurations.
    """

    _instance = None
    _configs = defaultdict(dict)  # Stores configurations with unique names as keys
    _call_locations = defaultdict(dict)  # Stores where each config was registered
    verbose: bool = True

    def __new__(cls):
        """Ensure only one instance of GlobalRegistry exists (singleton)"""
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls):
        """Get the singleton instance of the AssetRegistry"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def groups(self) -> list:
        """Get the list of all registered groups"""
        return list(self._configs.keys())

    def register(self, group_name: str, name: str, config) -> bool:
        """
        Register a new configuration with a unique name.

        Args:
            group_name: The group name of the configuration
            name: Unique identifier for the configuration
            config: The configuration to store (can be any type)

        Returns:
            bool: True if registered successfully, False if name already exists
        """
        if name in self._configs[group_name]:
            raise ValueError(
                f"Configuration {name} already registered in group {group_name}"
            )

        # Record where this registration was called from
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_filename = caller_frame.f_code.co_filename
        caller_lineno = caller_frame.f_lineno

        self._configs[group_name][name] = config
        self._call_locations[group_name][name] = {
            "file": caller_filename,
            "line": caller_lineno,
            "function": caller_frame.f_code.co_name,
        }
        if self.verbose:
            print(
                f"Registered '{name}' in '{group_name}' from {caller_filename}:{caller_lineno}."
            )
        return True

    def get(self, group_name: str, name: str):
        """
        Retrieve a configuration by name.

        Args:
            name: The unique identifier of the configuration

        Returns:
            The stored configuration or None if not found
        """
        result = self._configs[group_name].get(name)
        if result is None:
            raise ValueError(
                f"Configuration {name} not found in group {group_name}."
                f"Available configurations: {list(self._configs[group_name].keys())}"
            )
        return result

    def update(self, group_name: str, name: str, config) -> bool:
        """
        Update an existing configuration.

        Args:
            name: The unique identifier of the configuration
            config: The new configuration value

        Returns:
            bool: True if updated successfully, False if name doesn't exist
        """
        if name not in self._configs[group_name]:
            return False
        self._configs[group_name][name] = config
        return True

    def unregister(self, group_name: str, name: str) -> bool:
        """
        Remove a configuration from the registry.

        Args:
            name: The unique identifier of the configuration

        Returns:
            bool: True if removed successfully, False if name doesn't exist
        """
        if name in self._configs[group_name]:
            del self._configs[group_name][name]
            return True
        return False

    def list_all(self, group_name: str) -> list:
        """
        Get a list of all registered configuration names.

        Returns:
            list: Names of all registered configurations
        """
        return list(self._configs[group_name].keys())

    def clear(self) -> None:
        """Remove all configurations from the registry"""
        self._configs.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a configuration exists in the registry"""
        flag = False
        for group_name in self._configs.keys():
            if name in self._configs[group_name]:
                flag = True
                break
        return flag

    def __len__(self) -> int:
        """Return the number of registered configurations"""
        return sum(len(group) for group in self._configs.values())


class RegistryMixin:

    supported_backends: tuple[str, ...] = ("isaac", "mujoco", "mjlab")
    """List of supported backends. Subclasses should override this to declare which backends they support."""

    namespace: str | None = None
    """Optional namespace for registration. Subclasses can override this as a class attribute."""

    def __init_subclass__(cls, **kwargs) -> None:
        """Put the subclass in the global registry"""
        if not hasattr(cls, "registry"):
            cls.registry = {}

        # Backwards compatibility: allow deprecated 'namespace' kwarg
        namespace = kwargs.pop("namespace", None)
        if namespace is not None:
            warnings.warn(
                "Passing 'namespace' as a keyword argument to a registered class "
                "is deprecated. Set 'namespace' as a class attribute instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # New style: read namespace from class attribute if not provided via kwarg
        if namespace is None:
            namespace = getattr(cls, "namespace", None)

        if namespace is None:
            cls_name = cls.__name__
        else:
            cls_name = f"{namespace}.{cls.__name__}"

        if cls_name not in cls.registry:
            cls._file = inspect.getfile(cls)
            cls._line = inspect.getsourcelines(cls)[1]
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")


    @classmethod
    def make(cls, class_name, *args,**kwargs):
        if class_name not in cls.registry:
            raise ValueError(f"Class '{class_name}' not found in {cls.__name__}.registry")
        instance_cls = cls.registry[class_name]
        if aa.get_backend() not in instance_cls.supported_backends:
            warnings.warn(f"Class '{class_name}' does not support backend '{aa.get_backend()}'. "
                          f"Supported backends: {instance_cls.supported_backends}")
            return None
        return instance_cls(*args, **kwargs)

