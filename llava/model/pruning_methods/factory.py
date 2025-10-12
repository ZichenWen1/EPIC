from typing import Type, Dict, Any
from transformers import LlamaConfig
from .config import PruningMethod, create_pruning_config
from . import registry

# Import all model classes
try:
    from ..language_model.dart.llava_llama_dart import LlavaLlamaForCausalLM_DART
    from ..language_model.fastv.llava_llama_fastv import LlavaLlamaForCausalLM_fastv  
    from ..language_model.random.llava_llama_random import LlavaLlamaForCausalLM_random
except ImportError as e:
    print(f"Warning: Could not import some model classes: {e}")

class ModelFactory:
    """Model factory class"""

    # Mapping from pruning method to model class
    MODEL_CLASSES = {
        PruningMethod.DART: LlavaLlamaForCausalLM_DART,
        PruningMethod.FASTV: LlavaLlamaForCausalLM_fastv,
        PruningMethod.RANDOM: LlavaLlamaForCausalLM_random,
    }
    
    @classmethod
    def create_model(cls, method: str, config: LlamaConfig, model_name_or_path: str = None, **kwargs):
        """Create corresponding model by method name"""
        method_enum = PruningMethod(method.lower())
        
        if method_enum not in cls.MODEL_CLASSES:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        model_class = cls.MODEL_CLASSES[method_enum]
        
        # If model_name_or_path is provided, use from_pretrained
        if model_name_or_path:
            return model_class.from_pretrained(
                model_name_or_path,
                config=config,
                **kwargs
            )
        else:
            # Fallback to direct instantiation (for backward compatibility)
            return model_class(config)
    
    @classmethod
    def configure_model(cls, model, pruning_config):
        """Configure model's pruning parameters"""
        if pruning_config.sparse:
            config_dict = pruning_config.to_dict()
            # Set different config keys according to method
            if pruning_config.method == PruningMethod.DART:
                # Map pruned_layer to K and second_pruned_layer to K2 for DART
                dart_config = config_dict.copy()
                dart_config['K'] = dart_config.pop('pruned_layer')
                dart_config['K2'] = dart_config.pop('second_pruned_layer')
                model.config.DART_config = dart_config
            else:
                model.config.pruning_config = config_dict
        else:
            # Disable pruning
            if hasattr(model.config, 'DART_config'):
                model.config.DART_config = None
            if hasattr(model.config, 'pruning_config'):
                model.config.pruning_config = None
        
        return model

def get_model_class(method: str) -> Type:
    """Get the model class for a specified method"""
    method_enum = PruningMethod(method.lower())
    return ModelFactory.MODEL_CLASSES[method_enum]
