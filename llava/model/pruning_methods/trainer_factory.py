from typing import Type, Dict, Any
from .config import PruningMethod
from . import registry

# Import all trainer classes
try:
    from ...train.llava_trainer_KD_from_pretrain_dart import LLaVATrainer_KD as DARTTrainer
    from ...train.llava_trainer_KD_from_pretrain_fastv import LLaVATrainer_KD as FastVTrainer
    from ...train.llava_trainer_KD_from_pretrain_random import LLaVATrainer_KD as RandomTrainer
except ImportError as e:
    print(f"Warning: Could not import some trainer classes: {e}")

class TrainerFactory:
    """Trainer factory class"""

    # Mapping from pruning method to trainer class
    TRAINER_CLASSES = {
        PruningMethod.DART: DARTTrainer,
        PruningMethod.FASTV: FastVTrainer,
        PruningMethod.RANDOM: RandomTrainer,
    }

    @classmethod
    def create_trainer(cls, method: str, *args, **kwargs):
        """Create corresponding trainer by method name"""
        method_enum = PruningMethod(method.lower())

        if method_enum not in cls.TRAINER_CLASSES:
            raise ValueError(f"Unsupported pruning method: {method}")

        trainer_class = cls.TRAINER_CLASSES[method_enum]
        return trainer_class(*args, **kwargs)

    @classmethod
    def get_trainer_class(cls, method: str) -> Type:
        """Get trainer class for the specified method"""
        method_enum = PruningMethod(method.lower())
        return cls.TRAINER_CLASSES[method_enum]

def get_trainer_class(method: str) -> Type:
    """Get trainer class for the specified method"""
    return TrainerFactory.get_trainer_class(method)
