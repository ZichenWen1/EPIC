from typing import Dict, Type, Any, Optional
from .config import PruningMethod

# Lazy import to avoid circular dependencies
_model_classes: Optional[Dict[PruningMethod, Type]] = None

def get_model_classes() -> Dict[PruningMethod, Type]:
    """Get the mapping of all model classes."""
    global _model_classes
    if _model_classes is None:
        _model_classes = {}
        try:
            from ..language_model.dart.llava_llama_dart import LlavaLlamaForCausalLM_DART
            _model_classes[PruningMethod.DART] = LlavaLlamaForCausalLM_DART
        except ImportError:
            pass

        try:
            from ..language_model.fastv.llava_llama_fastv import LlavaLlamaForCausalLM_fastv
            _model_classes[PruningMethod.FASTV] = LlavaLlamaForCausalLM_fastv
        except ImportError:
            pass

        try:
            from ..language_model.random.llava_llama_random import LlavaLlamaForCausalLM_random
            _model_classes[PruningMethod.RANDOM] = LlavaLlamaForCausalLM_random
        except ImportError:
            pass

    return _model_classes

def get_model_class(method: str) -> Type:
    """Get model class by method name."""
    method_enum = PruningMethod(method.lower())
    model_classes = get_model_classes()

    if method_enum not in model_classes:
        raise ValueError(f"Model class not found for method: {method}")

    return model_classes[method_enum]

def list_available_methods() -> list:
    """List all available methods."""
    model_classes = get_model_classes()
    return [method.value for method in model_classes.keys()]
