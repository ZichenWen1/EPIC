from typing import Dict, Type, Any
from abc import ABC, abstractmethod

class PruningMethodRegistry:
    """Token pruning method registry"""

    def __init__(self):
        self._methods: Dict[str, Dict[str, Any]] = {}

    def register(self, method_name: str, model_class: Type, trainer_class: Type, 
                config_class: Type, config_key: str = None):
        """Register a new pruning method"""
        if config_key is None:
            config_key = f"{method_name}_config"

        self._methods[method_name] = {
            'model_class': model_class,
            'trainer_class': trainer_class, 
            'config_class': config_class,
            'config_key': config_key
        }

    def get_method(self, method_name: str) -> Dict[str, Any]:
        """Get classes for a specific pruning method"""
        if method_name not in self._methods:
            raise ValueError(f"Unknown pruning method: {method_name}. "
                           f"Available methods: {list(self._methods.keys())}")
        return self._methods[method_name]

    def list_methods(self) -> list:
        """List all available pruning methods"""
        return list(self._methods.keys())

# Global registry instance
registry = PruningMethodRegistry()

def register_pruning_method(method_name: str, config_key: str = None):
    """Decorator for registering pruning methods"""
    def decorator(cls):
        registry.register(method_name, cls, None, None, config_key)
        return cls
    return decorator
