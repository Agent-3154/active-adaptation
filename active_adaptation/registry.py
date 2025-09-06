from collections import defaultdict

class Registry:
    """
    A singleton class implementing a global registry for configurations.
    Ensures unique keys and provides methods for managing configurations.
    """
    _instance = None
    _configs = defaultdict(dict)  # Stores configurations with unique names as keys

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
            raise ValueError(f"Configuration {name} already registered in group {group_name}")
        self._configs[group_name][name] = config
        return True

    def get(self, group_name: str, name: str):
        """
        Retrieve a configuration by name.
        
        Args:
            name: The unique identifier of the configuration
            
        Returns:
            The stored configuration or None if not found
        """
        return self._configs[group_name].get(name)

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

