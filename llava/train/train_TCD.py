import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
import torch.distributed as dist
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def colored_print(*args, color="green", **kwargs):
    color_codes = {
        "grey": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    prefix = color_codes.get(color, color_codes["green"])
    suffix = color_codes["reset"]
    print(prefix + " ".join(str(a) for a in args) + suffix, **kwargs)

# Import factory and pruning configuration
from llava.model.pruning_methods.factory import ModelFactory
from llava.model.pruning_methods.trainer_factory import TrainerFactory
from llava.model.pruning_methods.config import create_pruning_config, PruningMethod

import warnings
warnings.filterwarnings("ignore")

local_rank = None

def rank0_print(*args, color="green", **kwargs):
    if local_rank == 0:
        colored_print(*args, color=color, **kwargs)

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    teacher_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class PruningArguments:
    # Pruning method selection
    pruning_method: str = field(default="dart", metadata={"help": "Pruning method: dart, fastv, random"})
    
    # General pruning parameters
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

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    distill_all_tokens: bool = field(default=False)
    alpha: float = field(default=0.5)
    distill_weight: float = field(default=0.5)
    sft_weight: float = field(default=0.5)
    disable_dropout: bool = field(default=False)

# Reuse the original data processing functions
from llava.train.train import (
    maybe_zero_3, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3, find_all_linear_names, safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize, _tokenize_fn, _mask_targets, _add_speaker_and_signal,
    preprocess_multimodal, preprocess_llama_2, preprocess_v1, preprocess_mpt, preprocess_plain,
    preprocess, LazySupervisedDataset, DataCollatorForSupervisedDataset, make_supervised_data_module
)

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PruningArguments))
    model_args, data_args, training_args, pruning_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Validate pruning method
    try:
        PruningMethod(pruning_args.pruning_method.lower())
    except ValueError:
        colored_print(f"Unsupported pruning method: {pruning_args.pruning_method}. "
                      f"Supported methods: {[m.value for m in PruningMethod]}", color="red")
        raise ValueError(f"Unsupported pruning method: {pruning_args.pruning_method}. "
                        f"Supported methods: {[m.value for m in PruningMethod]}")

    compute_dtype = (torch.float16 if training_args.fp16 else 
                    (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    # Create model
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            # MPT models currently do not support pruning methods
            from llava.model import LlavaMptForCausalLM
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # Use the model factory to create the model
            model = ModelFactory.create_model(
                pruning_args.pruning_method,
                transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
                model_name_or_path=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else 
                                 (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...", color="cyan")
        model = get_peft_model(model, lora_config)

    # Tokenizer setup
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # Vision tower setup
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # Handle mm_tunable_parts parameter
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            if model_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False
        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}", color="yellow")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name:
                        param.requires_grad_(True)
            
            # Print parameter statistics
            total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
            trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
            rank0_print(f"Total parameters: ~{total_params/1e6:.2f}M", color="magenta")
            rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f}M", color="magenta")
            rank0_print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%", color="magenta")
            
            # Set vision tower learning rate
            model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Pruning configuration
    pruning_config = create_pruning_config(
        pruning_args.pruning_method,
        sparse=pruning_args.sparse,
        pruned_layer=pruning_args.pruned_layer,
        second_pruned_layer=pruning_args.second_pruned_layer,
        do_second_pruned=pruning_args.do_second_pruned,
        only_do_second_pruned=pruning_args.only_do_second_pruned,
        image_token_start_index=pruning_args.image_token_start_index,
        image_token_length=pruning_args.image_token_length,
        max_num_trunction=pruning_args.max_num_trunction,
        reduction_ratio=pruning_args.reduction_ratio,
        retain_token_num_for_llava_next=pruning_args.retain_token_num_for_llava_next,
        pivot_image_token=pruning_args.pivot_image_token,
        pivot_text_token=pruning_args.pivot_text_token,
    )
    
    model = ModelFactory.configure_model(model, pruning_config)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Use the trainer factory to create the trainer
    trainer = TrainerFactory.create_trainer(
        pruning_args.pruning_method,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        pruning_args=pruning_args,
        teacher_model=None,                   
        temperature=1.0,             
        **data_module                
    )
    
    # Resume from checkpoint if exists, else start new training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from existing checkpoint...", color="blue")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting new training...", color="blue")
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # Save model state based on lora_enable
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            colored_print("Saving model (LoRA mode)...", color="cyan")
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        colored_print("Saving model...", color="cyan")
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
