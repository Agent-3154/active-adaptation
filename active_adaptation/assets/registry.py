class AssetRegistry:
    """
    A singleton class implementing a global registry for configurations.
    Ensures unique keys and provides methods for managing configurations.
    """
    _instance = None
    _configs = {}  # Stores configurations with unique names as keys

    def __new__(cls):
        """Ensure only one instance of GlobalRegistry exists (singleton)"""
        if cls._instance is None:
            cls._instance = super(AssetRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls):
        """Get the singleton instance of the AssetRegistry"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, config) -> bool:
        """
        Register a new configuration with a unique name.
        
        Args:
            name: Unique identifier for the configuration
            config: The configuration to store (can be any type)
            
        Returns:
            bool: True if registered successfully, False if name already exists
        """
        if name in self._configs:
            return False
        self._configs[name] = config
        return True

    def get(self, name: str):
        """
        Retrieve a configuration by name.
        
        Args:
            name: The unique identifier of the configuration
            
        Returns:
            The stored configuration or None if not found
        """
        return self._configs.get(name)

    def update(self, name: str, config) -> bool:
        """
        Update an existing configuration.
        
        Args:
            name: The unique identifier of the configuration
            config: The new configuration value
            
        Returns:
            bool: True if updated successfully, False if name doesn't exist
        """
        if name not in self._configs:
            return False
        self._configs[name] = config
        return True

    def unregister(self, name: str) -> bool:
        """
        Remove a configuration from the registry.
        
        Args:
            name: The unique identifier of the configuration
            
        Returns:
            bool: True if removed successfully, False if name doesn't exist
        """
        if name in self._configs:
            del self._configs[name]
            return True
        return False

    def list_all(self) -> list:
        """
        Get a list of all registered configuration names.
        
        Returns:
            list: Names of all registered configurations
        """
        return list(self._configs.keys())

    def clear(self) -> None:
        """Remove all configurations from the registry"""
        self._configs.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a configuration exists in the registry"""
        return name in self._configs

    def __len__(self) -> int:
        """Return the number of registered configurations"""
        return len(self._configs)