from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

class PruningMethod(Enum):
    """Enumeration of supported pruning methods."""
    DART = "dart"
    FASTV = "fastv" 
    RANDOM = "random"

@dataclass
class BasePruningConfig:
    """Base pruning configuration."""
    method: PruningMethod = field(default=PruningMethod.DART)
    sparse: bool = field(default=False)
    pruned_layer: int = field(default=2)
    second_pruned_layer: int = field(default=20)
    do_second_pruned: bool = field(default=False)
    only_do_second_pruned: bool = field(default=False)
    image_token_start_index: int = field(default=35)
    image_token_length: int = field(default=576)
    max_num_trunction: int = field(default=0)
    reduction_ratio: float = field(default=0.9)
    retain_token_num_for_llava_next: int = field(default=0)
    pivot_image_token: int = field(default=4)
    pivot_text_token: int = field(default=4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "method": self.method.value,
            "sparse": self.sparse,
            "pruned_layer": self.pruned_layer,
            "second_pruned_layer": self.second_pruned_layer,
            "do_second_pruned": self.do_second_pruned,
            "only_do_second_pruned": self.only_do_second_pruned,
            "image_token_start_index": self.image_token_start_index,
            "image_token_length": self.image_token_length,
            "max_num_trunction": self.max_num_trunction,
            "reduction_ratio": self.reduction_ratio,
            "retain_token_num_for_llava_next": self.retain_token_num_for_llava_next,
            "pivot_image_token": self.pivot_image_token,
            "pivot_text_token": self.pivot_text_token,
        }

@dataclass 
class DARTConfig(BasePruningConfig):
    """DART-specific pruning configuration."""
    method: PruningMethod = field(default=PruningMethod.DART)
    # DART-specific parameters can be added here

@dataclass
class FastVConfig(BasePruningConfig):
    """FastV-specific pruning configuration."""
    method: PruningMethod = field(default=PruningMethod.FASTV)
    # FastV-specific parameters can be added here

@dataclass
class RandomConfig(BasePruningConfig):
    """Random-specific pruning configuration."""
    method: PruningMethod = field(default=PruningMethod.RANDOM)
    # Random-specific parameters can be added here

def create_pruning_config(method: str, **kwargs) -> BasePruningConfig:
    """Create the corresponding configuration according to the method name."""
    method_enum = PruningMethod(method.lower())
    
    if method_enum == PruningMethod.DART:
        return DARTConfig(**kwargs)
    elif method_enum == PruningMethod.FASTV:
        return FastVConfig(**kwargs)
    elif method_enum == PruningMethod.RANDOM:
        return RandomConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported pruning method: {method}")
