import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler
import math
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from .llava_trainer import LLaVATrainer
import torch.nn.functional as F
import pdb
import random
import math
import wandb

import torch.distributed as dist
import ipdb
from llava.model.utils import disable_dropout_in_model

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    

class LLaVATrainer_KD(LLaVATrainer):
    def __init__(self, *args, pruning_args, teacher_model=None, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if teacher_model is not None:
            self.teacher_model.eval()  
        # self.alpha = alpha
        self.temperature = temperature
        self.pruning_args = pruning_args

        # follow trl
        if self.args.disable_dropout: # default: false
            disable_dropout_in_model(self.model)
            if self.teacher_model is not None:
                disable_dropout_in_model(self.teacher_model)

    # TODO: TCD token consistency distillation
    def compute_loss(self, model, inputs, return_outputs=False):
        initial_max_ratio = 0.1   
        final_max_ratio = 0.9   
        initial_min_ratio = 0.1         
        initial_teacher_gap = 0.1       
        teacher_min_ratio = 0.0 
        current_step = self.state.global_step
        max_steps = self.state.max_steps
        progress = current_step / max_steps if max_steps > 0 else 0.0
        progress = min(progress, 1.0)  

        current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.8 # TODO: accelerate
        current_max_ratio = min(current_max_ratio, 1.0)

        current_min_ratio = initial_min_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.2 # TODO: accelerate

        teacher_gap = initial_teacher_gap + progress * 0.5
        teacher_gap = min(teacher_gap, 0.3)

        # input_len = inputs['input_ids'].shape[1]
        # labels = inputs["labels"]

        # HACK: student model forward
        # sample reduction_ratio
        student_reduction_ratio = random.uniform(current_min_ratio, current_max_ratio)

        model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
        model.model.config.DART_config['do_second_pruned'] = False
        # model.model.config.DART_config['K'] = 2    # Fix: K2 -> second pruned layer, K -> pruned layer

        student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        student_logits = student_outputs.logits
        student_retained_indices = student_outputs.retained_indices

        # HACK: teacher model forward
        # teacher_reduction_ratio = student_reduction_ratio - 0.05

        teacher_reduction_ratio = max(
            student_reduction_ratio - teacher_gap,
            teacher_min_ratio
        )  

        # HACK: teacher model forward
        model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
        model.model.config.DART_config['do_second_pruned'] = False
        # model.model.config.DART_config['K'] = current_second_pruned_layer    # Fix: K2 -> second pruned layer, K -> pruned layer
        with torch.no_grad():
            teacher_outputs = model(**inputs, return_dict=True)
        teacher_logits = teacher_outputs.logits.detach() 
        teacher_retained_indices = teacher_outputs.retained_indices

        if teacher_logits.size(1) != student_logits.size(1):
            filled_teacher_logits = self.restore_sparse_tokens(teacher_retained_indices, teacher_logits)
            expanded_indices = student_retained_indices.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
            teacher_logits_aligned = torch.gather(filled_teacher_logits, dim=1, index=expanded_indices)
        else:
            teacher_logits_aligned = teacher_logits

        probs, _ = self.get_p(teacher_logits_aligned, teacher_outputs)
        logprobs, _, student_labels = self.get_logp(student_logits, student_outputs)
        
        # distillation_loss = self.compute_distillation_loss(logprobs, probs, student_labels)
        distillation_loss = self.compute_distillation_loss_standard(logprobs, probs, student_labels)
        # loss = self.args.alpha * distillation_loss + (1 - self.args.alpha) * student_loss_sft
        loss = self.args.alpha * distillation_loss + (1 - self.args.alpha) * student_loss_sft

        if model.training and self.is_world_process_zero():
            if self.state.global_step % self.args.logging_steps == 0:
                # Log custom metrics to tensorboard/wandb
                custom_metrics = {
                    "student_reduction_ratio": student_reduction_ratio,
                    "teacher_reduction_ratio": teacher_reduction_ratio,
                    "teacher_gap": teacher_gap,
                    # "current_second_pruned_layer": current_second_pruned_layer,
                    "distillation_loss": distillation_loss.item(),
                    "student_loss_sft": student_loss_sft.item(),
                    "total_loss": loss.item(),
                    "learning_rate": self._get_learning_rate(),
                    "epoch": self.state.epoch,
                }
                
                # Use self.log() for tensorboard/wandb compatibility
                self.log(custom_metrics)
                
                # Also log to wandb if initialized (for backward compatibility)
                if wandb.run is not None:
                    wandb.log(custom_metrics)

        return (loss, student_outputs) if return_outputs else loss

    
    
    def get_pruning_num(self, input_len, reduction_ratio, image_token_length, pivot_image_token, pivot_text_token):

        retained_token_num =  int(
            (1 - reduction_ratio) * image_token_length 
            / (pivot_image_token + pivot_text_token)
        ) * ((pivot_image_token + pivot_text_token)) + pivot_image_token
        pruning_token_num = image_token_length - retained_token_num

        return pruning_token_num


    def restore_sparse_tokens(
        self,
        teacher_retained_indices: torch.Tensor,  # [batch_size, sparse_length]
        sparse_tokens: torch.Tensor,             # [batch_size, sparse_length, embedding_dim]
        ) -> torch.Tensor:

        batch_size, sparse_length, embedding_dim = sparse_tokens.shape

        original_length = torch.max(teacher_retained_indices) + 1  

        restored = torch.zeros(
            batch_size, original_length, embedding_dim,
            device=sparse_tokens.device,
            dtype=sparse_tokens.dtype
        )

        teacher_retained_indices = teacher_retained_indices.long()

        restored.scatter_(
            dim=1,  
            index=teacher_retained_indices.unsqueeze(-1).expand(-1, -1, embedding_dim),  
            src=sparse_tokens  
        )

        return restored


    # TODO: distil_loss v2
    def get_distil_loss(self, teacher_logits, labels, logits):
        # Step 1: 
        min_len = min(teacher_logits.shape[1], logits.shape[1])
        teacher_logits = teacher_logits[:, :min_len, :]
        logits = logits[:, :min_len, :]
        labels = labels[:, :min_len]  

        # Step 2
        mask = (labels != -100).int()  
        inf_mask = torch.isinf(logits) 
        # print(f"labels: {labels}")
        # print(f"mask: {mask}")
        # print(f"inf_mask: {inf_mask}")

        # Step 3
        teacher_logits = teacher_logits * mask.unsqueeze(-1)  
        logits = logits * mask.unsqueeze(-1)  
        logits = torch.masked_fill(logits, inf_mask, 0)  

        # Step 4
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = teacher_probs * logprobs
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss
    
    def get_p(self, logits, outputs):   # teacher model  # HACK: copy from llava-Mod

        # outputs = model(**inputs, return_dict=True)

        # logits = outputs.logits
        labels = outputs.labels

        sft_loss = outputs.loss


        # if logits.shape[:-1] != labels.shape:
        #     raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        logits_aligned = logits[:, :, :151936]

        probs = F.softmax(logits_aligned / self.temperature, dim=-1, dtype=torch.float32)  # probs


        return probs, sft_loss
    
    def get_logp(self, logits, outputs):  # student model # HACK: copy from llava-Mod

        # outputs = model(**inputs, return_dict=True)

        # logits = outputs.logits
        labels = outputs.labels

        sft_loss = outputs.loss


        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        logits_aligned = logits[:, :, :151936]  # NOTE: FIXED ME

        logprobs = F.log_softmax(logits_aligned / self.temperature, dim=-1, dtype=torch.float32)  # student logp


        return logprobs, sft_loss, labels
    
    def compute_distillation_loss(self, policy_logprobs, reference_probs, labels):  # HACK: copy from llava-Mod

        # print("#### policy_logprobs: ", policy_logprobs.size())
        # print("#### reference_probs: ", reference_probs.size())
        # print("#### labels: ", labels.size())

        inf_mask = torch.isinf(policy_logprobs)
        prod_probs = torch.masked_fill(reference_probs * policy_logprobs, inf_mask, 0)

        # print("#### prod_probs: ", prod_probs.size())

        x = torch.sum(prod_probs, dim=-1).view(-1)

        # print("#### x: ", x.size())
        if self.args.distill_all_tokens:
            # If we're using all tokens, the loss mask will be all ones
            print("####### compute multimodal instruction + response tokens")
            loss_mask = torch.ones_like(labels).int()
        else:
            loss_mask = (labels != -100).int()  # modify it to distill text instruction + vision tokens


        distillation_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)

        return distillation_loss

    def compute_distillation_loss_standard(self, student_logprobs, teacher_probs, labels):
        

        kl_div = F.kl_div(student_logprobs, teacher_probs, reduction='none', log_target=False)
        

        loss_mask = torch.ones_like(labels) if self.args.distill_all_tokens else (labels != -100).int()
        kl_div_masked = kl_div * loss_mask.unsqueeze(-1).expand_as(kl_div)
        

        distillation_loss = kl_div_masked.sum() / loss_mask.sum()
        
        return distillation_loss * (self.temperature ** 2)  


    


# class LLaVATrainer(Trainer):

#     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
#         if self.train_dataset is None or not has_length(self.train_dataset):
#             return None

#         if self.args.group_by_modality_length:
#             lengths = self.train_dataset.modality_lengths
#             return LengthGroupedSampler(
#                 self.args.train_batch_size,
#                 world_size=self.args.world_size * self.args.gradient_accumulation_steps,
#                 lengths=lengths,
#                 group_by_modality=True,
#             )
#         else:
#             return super()._get_train_sampler()

#     def create_optimizer(self):
#         """
#         Setup the optimizer.

#         We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
#         Trainer's init through `optimizers`, or subclass and override this method in a subclass.
#         """
#         if is_sagemaker_mp_enabled():
#             return super().create_optimizer()

#         opt_model = self.model

#         if self.optimizer is None:
#             decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
#             decay_parameters = [name for name in decay_parameters if "bias" not in name]
#             if self.args.mm_projector_lr is not None:
#                 projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
#                 optimizer_grouped_parameters = [
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                     },
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                     },
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                         "lr": self.args.mm_projector_lr,
#                     },
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                         "lr": self.args.mm_projector_lr,
#                     },
#                 ]
#             else:
#                 optimizer_grouped_parameters = [
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                     },
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                     },
#                 ]

#             optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

#             self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
#             if optimizer_cls.__name__ == "Adam8bit":
#                 import bitsandbytes

#                 manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

#                 skipped = 0
#                 for module in opt_model.modules():
#                     if isinstance(module, nn.Embedding):
#                         skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
#                         logger.info(f"skipped {module}: {skipped/2**20}M params")
#                         manager.register_module_override(module, "weight", {"optim_bits": 32})
#                         logger.debug(f"bitsandbytes: will optimize {module} in fp32")
#                 logger.info(f"skipped: {skipped/2**20}M params")

#         return self.optimizer

#     def _save_checkpoint(self, model, trial, metrics=None):
#         if getattr(self.args, 'tune_mm_mlp_adapter', False):
#             from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#             checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

#             run_dir = self._get_output_dir(trial=trial)
#             output_dir = os.path.join(run_dir, checkpoint_folder)

#             # Only save Adapter
#             keys_to_match = ['mm_projector', 'vision_resampler']
#             if getattr(self.args, "use_im_start_end", False):
#                 keys_to_match.extend(['embed_tokens', 'embed_in'])

#             weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

#             if self.args.local_rank == 0 or self.args.local_rank == -1:
#                 self.model.config.save_pretrained(output_dir)
#                 torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
#         else:
#             super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         if getattr(self.args, 'tune_mm_mlp_adapter', False):
#             pass
#         else:
#             super(LLaVATrainer, self)._save(output_dir, state_dict)
