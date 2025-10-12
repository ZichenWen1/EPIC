import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler
from copy import deepcopy
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
from accelerate.utils import is_deepspeed_available
if is_deepspeed_available():
    import deepspeed
import torch.distributed as dist
import ipdb


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
    def __init__(self, *args, pruning_args, teacher_model, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        # if teacher_model is not None:
        #     self.teacher_model.eval()  
        self.alpha = alpha
        self.temperature = temperature
        self.pruning_args = pruning_args

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(self.teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        self.teacher_model.module.eval()
        # self.teacher_model.eval()

        print("##### teacher_model exists ", self.teacher_model is not None)
        print("##### teacher_model is in training mode:", self.teacher_model.module.training)

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     loss_sft, outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     logits = outputs.logits
    #     labels = inputs["labels"]

    #     if self.teacher_model is not None:
    #         with torch.no_grad():
    #             teacher_logits = self.teacher_model(**inputs).logits
    #         distillation_loss = F.kl_div(
    #             F.log_softmax(logits / self.temperature, dim=-1),
    #             F.softmax(teacher_logits / self.temperature, dim=-1),
    #             reduction="batchmean",
    #         ) * (self.temperature ** 2)
    #         loss = self.alpha * distillation_loss + (1 - self.alpha) * loss_sft
    #     else:
    #         loss = loss_sft 

    #     return (loss, outputs) if return_outputs else loss

    # # TODO: HACK: loss v1
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     # print(model.config)
    #     # print(model.model.config)
    #     # print(inputs['input_ids'].shape)
    #     initial_max_ratio = 0.3   # 初始最大采样值
    #     final_max_ratio = 1.0    # 最终最大采样值
    #     min_ratio = 0.05         # 固定最小采样值
    #     teacher_gap = 0.05       # 师生模型压缩率差值
    #     teacher_min_ratio = 0.0 # 教师模型最小压缩率
    #     current_step = self.state.global_step
    #     max_steps = self.state.max_steps
    #     progress = current_step / max_steps if max_steps > 0 else 0.0
    #     progress = min(progress, 1.0)  
    #     current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress


    #     input_len = inputs['input_ids'].shape[1]

    #     # HACK: student model forward
    #     # sample reduction_ratio
    #     # student_reduction_ratio = random.uniform(0.05, 0.3)
    #     student_reduction_ratio = random.uniform(min_ratio, current_max_ratio)

    #     student_input_len = input_len - self.get_pruning_num(
    #         input_len,
    #         student_reduction_ratio,
    #         self.pruning_args.image_token_length,
    #         self.pruning_args.pivot_image_token,
    #         self.pruning_args.pivot_text_token
    #     )
    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)

    #     # print(f"model config: {model.config}")
    #     student_logits = student_outputs.logits
    #     student_gen_logits = student_logits[:, student_input_len:, :]  # shape: (batch, gen_len, vocab_size)

    #     # HACK: teacher model forward
    #     # teacher_reduction_ratio = student_reduction_ratio - 0.05
    #     teacher_reduction_ratio = max(
    #         student_reduction_ratio - teacher_gap,
    #         teacher_min_ratio
    #     )  

    #     teacher_input_len = input_len - self.get_pruning_num(
    #         input_len,
    #         teacher_reduction_ratio,
    #         self.pruning_args.image_token_length,
    #         self.pruning_args.pivot_image_token,
    #         self.pruning_args.pivot_text_token
    #     )
    #     model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
    #     with torch.no_grad():  
    #         teacher_outputs = model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach()  
    #     teacher_gen_logits = teacher_logits[:, teacher_input_len:, :]  # shape: (batch, gen_len, vocab_size)

    #     labels = inputs["labels"]

    #     # align strictly start
    #     min_gen_len = min(
    #         student_gen_logits.shape[1], 
    #         teacher_gen_logits.shape[1]
    #     )
    #     student_gen_logits = student_gen_logits[:, :min_gen_len, :]
    #     teacher_gen_logits = teacher_gen_logits[:, :min_gen_len, :]
    #     # align strictly end

    #     distillation_loss = F.kl_div(
    #         F.log_softmax(student_gen_logits / self.temperature, dim=-1),
    #         F.softmax(teacher_gen_logits / self.temperature, dim=-1),
    #         reduction="batchmean",
    #     ) * (self.temperature ** 2)  # ! debug 怀疑有问题

    #     # V = student_logits.size(-1)  
    #     # distillation_loss = F.kl_div(
    #     #     F.log_softmax(student_gen_logits.reshape(-1, V) / self.temperature, dim=-1), 
    #     #     F.softmax(teacher_gen_logits.reshape(-1, V) / self.temperature, dim=-1),    
    #     #     reduction="mean"
    #     # ) * (self.temperature ** 2)

    #     loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "teacher_reduction_ratio": teacher_reduction_ratio,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss
    
    
    # # TODO: HACK: loss v2
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     # 动态调整参数设置
    #     initial_max_ratio = 0.3   # 初始最大采样值
    #     final_max_ratio = 1.0    # 最终最大采样值
    #     min_ratio = 0.05         # 固定最小采样值
    #     teacher_gap = 0.05       # 师生模型压缩率差值
    #     teacher_min_ratio = 0.0 # 教师模型最小压缩率

    #     current_step = self.state.global_step
    #     max_steps = self.state.max_steps
    #     progress = current_step / max_steps if max_steps > 0 else 0.0
    #     progress = min(progress, 1.0)  

    #     current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress

    #     # HACK: student model forward
    #     student_reduction_ratio = random.uniform(min_ratio, current_max_ratio)
    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     student_logits = student_outputs.logits

    #     # HACK: teacher model forward
    #     teacher_reduction_ratio = max(
    #         student_reduction_ratio - teacher_gap,
    #         teacher_min_ratio
    #     )  
    #     model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio

    #     with torch.no_grad():  
    #         teacher_outputs = model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach()  

    #     labels = inputs["labels"]

    #     # align strictly start
    #     min_len = min(
    #         student_logits.shape[1], 
    #         teacher_logits.shape[1]
    #     )
    #     student_logits = student_logits[:, :min_len, :]
    #     teacher_logits = teacher_logits[:, :min_len, :]
    #     # align strictly end

    #     distillation_loss = F.kl_div(
    #         F.log_softmax(student_logits / self.temperature, dim=-1),
    #         F.softmax(teacher_logits / self.temperature, dim=-1),
    #         reduction="batchmean",
    #     ) * (self.temperature ** 2)
    #     loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "teacher_reduction_ratio": teacher_reduction_ratio,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss


    # # TODO: HACK: loss v3
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     initial_max_ratio = 0.3   # 初始最大采样值
    #     final_max_ratio = 0.9    # 最终最大采样值
    #     initial_min_ratio = 0.05         # 固定最小采样值
    #     initial_teacher_gap = 0.05       # 师生模型压缩率差值
    #     teacher_min_ratio = 0.0 # 教师模型最小压缩率
    #     current_step = self.state.global_step
    #     max_steps = self.state.max_steps
    #     progress = current_step / max_steps if max_steps > 0 else 0.0
    #     progress = min(progress, 1.0)  

    #     current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress * 1.5 # TODO: accelerate
    #     current_max_ratio = min(current_max_ratio, 0.95)

    #     current_min_ratio = initial_min_ratio + (final_max_ratio - initial_max_ratio) * progress * 1.0 # TODO: accelerate

    #     teacher_gap = initial_teacher_gap + progress
    #     teacher_gap = min(teacher_gap, 0.25)


    #     input_len = inputs['input_ids'].shape[1]

    #     # HACK: student model forward
    #     # sample reduction_ratio
    #     student_reduction_ratio = random.uniform(current_min_ratio, current_max_ratio)

    #     student_input_len = input_len - self.get_pruning_num(
    #         input_len,
    #         student_reduction_ratio,
    #         self.pruning_args.image_token_length,
    #         self.pruning_args.pivot_image_token,
    #         self.pruning_args.pivot_text_token
    #     )
    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)

    #     # print(f"model config: {model.config}")
    #     student_logits = student_outputs.logits
    #     student_gen_logits = student_logits[:, student_input_len:, :]  # shape: (batch, gen_len, vocab_size)

    #     # HACK: teacher model forward
    #     # teacher_reduction_ratio = student_reduction_ratio - 0.05
    #     teacher_reduction_ratio = max(
    #         student_reduction_ratio - teacher_gap,
    #         teacher_min_ratio
    #     )  

    #     teacher_input_len = input_len - self.get_pruning_num(
    #         input_len,
    #         teacher_reduction_ratio,
    #         self.pruning_args.image_token_length,
    #         self.pruning_args.pivot_image_token,
    #         self.pruning_args.pivot_text_token
    #     )
    #     model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
    #     with torch.no_grad():  
    #         teacher_outputs = model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach()  
    #     teacher_gen_logits = teacher_logits[:, teacher_input_len:, :]  # shape: (batch, gen_len, vocab_size)

    #     labels = inputs["labels"]

    #     # align strictly start
    #     min_gen_len = min(
    #         student_gen_logits.shape[1], 
    #         teacher_gen_logits.shape[1]
    #     )
    #     student_gen_logits = student_gen_logits[:, :min_gen_len, :]
    #     teacher_gen_logits = teacher_gen_logits[:, :min_gen_len, :]
    #     # align strictly end

    #     distillation_loss = F.kl_div(
    #         F.log_softmax(student_gen_logits / self.temperature, dim=-1),
    #         F.softmax(teacher_gen_logits / self.temperature, dim=-1),
    #         reduction="batchmean",
    #     ) * (self.temperature ** 2)  # ! debug 怀疑有问题

    #     # V = student_logits.size(-1)  
    #     # distillation_loss = F.kl_div(
    #     #     F.log_softmax(student_gen_logits.reshape(-1, V) / self.temperature, dim=-1), 
    #     #     F.softmax(teacher_gen_logits.reshape(-1, V) / self.temperature, dim=-1),    
    #     #     reduction="mean"
    #     # ) * (self.temperature ** 2)

    #     loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "teacher_reduction_ratio": teacher_reduction_ratio,
    #                 "teacher_gap": teacher_gap,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss


    # # TODO: HACK: loss v4
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     initial_max_ratio = 0.3   # 初始最大采样值
    #     final_max_ratio = 0.9    # 最终最大采样值
    #     initial_min_ratio = 0.05         # 固定最小采样值
    #     initial_teacher_gap = 0.05       # 师生模型压缩率差值
    #     teacher_min_ratio = 0.0 # 教师模型最小压缩率
    #     current_step = self.state.global_step
    #     max_steps = self.state.max_steps
    #     progress = current_step / max_steps if max_steps > 0 else 0.0
    #     progress = min(progress, 1.0)  

    #     current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.8 # TODO: accelerate
    #     current_max_ratio = min(current_max_ratio, 0.95)

    #     current_min_ratio = initial_min_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.3 # TODO: accelerate

    #     teacher_gap = initial_teacher_gap

    #     input_len = inputs['input_ids'].shape[1]

    #     # HACK: student model forward
    #     # sample reduction_ratio
    #     student_reduction_ratio = random.uniform(current_min_ratio, current_max_ratio)

    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     student_logits = student_outputs.logits
    #     student_retained_indices = student_outputs.retained_indices

    #     # HACK: teacher model forward
    #     # teacher_reduction_ratio = student_reduction_ratio - 0.05
    #     teacher_reduction_ratio = max(
    #         student_reduction_ratio - teacher_gap,
    #         teacher_min_ratio
    #     )  

    #     model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
    #     with torch.no_grad():
    #         teacher_outputs = model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach() 
    #     teacher_retained_indices = teacher_outputs.retained_indices

    #     labels = inputs["labels"]

    #     # rank = dist.get_rank()
    #     # local_rank = int(os.environ['LOCAL_RANK'])
    #     # if rank == 0:  
    #     #     ipdb.set_trace()

    #     # dist.barrier()

    #     filled_teacher_logits = self.restore_sparse_tokens(teacher_retained_indices, teacher_logits)

    #     expanded_indices = student_retained_indices.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
    #     teacher_logits_aligned = torch.gather(filled_teacher_logits, dim=1, index=expanded_indices)
    #     probs, sft_loss = self.get_p(teacher_logits_aligned, teacher_outputs)
    #     logprobs, sft_loss, student_labels = self.get_logp(student_logits, student_outputs)
        
    #     distillation_loss = self.compute_distillation_loss(logprobs, probs, student_labels)
    #     loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "teacher_reduction_ratio": teacher_reduction_ratio,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss


    # # TODO: HACK: loss v5
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     input_len = inputs['input_ids'].shape[1]
    #     labels = inputs["labels"]
    #     teacher_min_ratio = 0.0

    #     # HACK: student model forward
    #     # sample reduction_ratio
    #     student_reduction_ratio = random.uniform(0.05, 1.0)

    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     student_logits = student_outputs.logits
    #     student_retained_indices = student_outputs.retained_indices

    #     # HACK: teacher model forward
    #     # teacher_reduction_ratio = student_reduction_ratio - 0.05
    #     teacher_reduction_ratio = max(
    #         student_reduction_ratio - 0.1,
    #         teacher_min_ratio
    #     )  

    #     model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
    #     with torch.no_grad():
    #         teacher_outputs = model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach() 
    #     teacher_retained_indices = teacher_outputs.retained_indices

    #     filled_teacher_logits = self.restore_sparse_tokens(teacher_retained_indices, teacher_logits)

    #     expanded_indices = student_retained_indices.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
    #     teacher_logits_aligned = torch.gather(filled_teacher_logits, dim=1, index=expanded_indices)
    #     probs, sft_loss = self.get_p(teacher_logits_aligned, teacher_outputs)
    #     logprobs, sft_loss, student_labels = self.get_logp(student_logits, student_outputs)
        
    #     # distillation_loss = self.compute_distillation_loss(logprobs, probs, student_labels)
    #     distillation_loss = self.compute_distillation_loss_standard(logprobs, probs, student_labels)
    #     loss = self.args.alpha * distillation_loss + (1 - self.args.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "teacher_reduction_ratio": teacher_reduction_ratio,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss

    # # TODO: HACK: loss v6 -> the pruning ratio of teacher model is set 0
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     initial_max_ratio = 0.1   
    #     final_max_ratio = 0.9   
    #     initial_min_ratio = 0.1         
    #     current_step = self.state.global_step
    #     max_steps = self.state.max_steps
    #     progress = current_step / max_steps if max_steps > 0 else 0.0
    #     progress = min(progress, 1.0)  

    #     current_max_ratio = initial_max_ratio + (final_max_ratio - initial_max_ratio) * progress * 1.2 # TODO: accelerate
    #     current_max_ratio = min(current_max_ratio, 1.0)

    #     current_min_ratio = initial_min_ratio + (final_max_ratio - initial_max_ratio) * progress * 0.3 # TODO: accelerate
    #     current_min_ratio = max(current_min_ratio, 0.5)
    #     # HACK: student model forward
    #     # sample reduction_ratio
    #     student_reduction_ratio = random.uniform(current_min_ratio, current_max_ratio)

    #     # self.configure_DART(model.model, self.pruning_args)
    #     model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     student_logits = student_outputs.logits
    #     student_retained_indices = student_outputs.retained_indices

    #     # HACK: teacher model forward
    #     teacher_reduction_ratio = 0
    #     # rank = dist.get_rank()
    #     # local_rank = int(os.environ['LOCAL_RANK'])
    #     # if rank == 0:  
    #     #     ipdb.set_trace()
    #     # dist.barrier()
    #     # model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
    #     self.teacher_model.model.config.DART_config = None
    #     with torch.no_grad():
    #         teacher_outputs = self.teacher_model(**inputs, return_dict=True)
    #     teacher_logits = teacher_outputs.logits.detach() 
    #     teacher_retained_indices = teacher_outputs.retained_indices

    #     filled_teacher_logits = self.restore_sparse_tokens(teacher_retained_indices, teacher_logits)

    #     expanded_indices = student_retained_indices.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
    #     teacher_logits_aligned = torch.gather(filled_teacher_logits, dim=1, index=expanded_indices)
    #     probs, sft_loss = self.get_p(teacher_logits_aligned, teacher_outputs)
    #     logprobs, sft_loss, student_labels = self.get_logp(student_logits, student_outputs)
        
    #     # distillation_loss = self.compute_distillation_loss(logprobs, probs, student_labels)
    #     distillation_loss = self.compute_distillation_loss_standard(logprobs, probs, student_labels)
    #     loss = self.args.alpha * distillation_loss + (1 - self.args.alpha) * student_loss_sft

    #     if model.training and self.is_world_process_zero():
    #         if self.state.global_step % self.args.logging_steps == 0:
    #             wandb.log({
    #                 "student_reduction_ratio": student_reduction_ratio,
    #                 "distillation_loss": distillation_loss.item(),
    #                 "student_loss_sft": student_loss_sft.item(),
    #                 "total_loss": loss.item(),
    #                 "learning_rate": self._get_learning_rate(),
    #                 "epoch": self.state.epoch,
    #             })

    #     return (loss, student_outputs) if return_outputs else loss


    # TODO: ablation: fix a pruning ratio 0.889 for student model and apply a vanilla no-prune teacher model for distill
    # TODO: ablation: apply sft loss and distill loss

    def compute_loss(self, model, inputs, return_outputs=False):

        # HACK: student model forward
        # sample reduction_ratio
        student_reduction_ratio = 0.889

        # self.configure_DART(model.model, self.pruning_args)
        model.model.config.DART_config['reduction_ratio'] = student_reduction_ratio
        student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        student_logits = student_outputs.logits
        student_retained_indices = student_outputs.retained_indices

        # HACK: teacher model forward
        # teacher_reduction_ratio = 0
        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()
        # dist.barrier()
        # model.model.config.DART_config['reduction_ratio'] = teacher_reduction_ratio
        self.teacher_model.model.config.DART_config = None
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, return_dict=True)
        teacher_logits = teacher_outputs.logits.detach() 
        teacher_retained_indices = teacher_outputs.retained_indices

        filled_teacher_logits = self.restore_sparse_tokens(teacher_retained_indices, teacher_logits)

        expanded_indices = student_retained_indices.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
        teacher_logits_aligned = torch.gather(filled_teacher_logits, dim=1, index=expanded_indices)
        probs, sft_loss = self.get_p(teacher_logits_aligned, teacher_outputs)
        logprobs, sft_loss, student_labels = self.get_logp(student_logits, student_outputs)
        
        # distillation_loss = self.compute_distillation_loss(logprobs, probs, student_labels)
        distillation_loss = self.compute_distillation_loss_standard(logprobs, probs, student_labels)
        loss = self.args.alpha * distillation_loss + (1 - self.args.alpha) * student_loss_sft

        if model.training and self.is_world_process_zero():
            if self.state.global_step % self.args.logging_steps == 0:
                wandb.log({
                    "student_reduction_ratio": student_reduction_ratio,
                    "distillation_loss": distillation_loss.item(),
                    "student_loss_sft": student_loss_sft.item(),
                    "total_loss": loss.item(),
                    "learning_rate": self._get_learning_rate(),
                    "epoch": self.state.epoch,
                })

        return (loss, student_outputs) if return_outputs else loss
    
    # # TODO: HACK: loss v6 -> only sft llava again and no distillation loss, this is a baseline
    # def compute_loss(self, model, inputs, return_outputs=False):


    #     student_loss_sft, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     loss = student_loss_sft


    #     return (loss, student_outputs) if return_outputs else loss
    
    
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
        if teacher_retained_indices is None:  # HACK: when teacher model is no pruned
            return sparse_tokens
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

    # copy from MiniLLM
    # TODO: distil_loss v1
    # def get_distil_loss(self, teacher_logits, labels, logits):

    #     teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    #     inf_mask = torch.isinf(logits)
    #     logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    #     prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    #     x = torch.sum(prod_probs, dim=-1).view(-1)
    #     mask = (labels != -100).int()
    #     distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        
    #     return distil_loss

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

        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()

        # dist.barrier()

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

        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()

        # dist.barrier()

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

        # print("#### loss_mask: ", loss_mask.size())

        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()

        # dist.barrier()

        distillation_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)

        return distillation_loss

    def compute_distillation_loss_standard(self, student_logprobs, teacher_probs, labels):

        # teacher_probs = F.softmax(reference_probs / temperature, dim=-1)
        # student_logprobs = F.log_softmax(policy_logprobs / temperature, dim=-1)
        

        kl_div = F.kl_div(student_logprobs, teacher_probs, reduction='none', log_target=False)
        

        loss_mask = torch.ones_like(labels) if self.args.distill_all_tokens else (labels != -100).int()
        kl_div_masked = kl_div * loss_mask.unsqueeze(-1).expand_as(kl_div)
        

        distillation_loss = kl_div_masked.sum() / loss_mask.sum()
        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()

        # dist.barrier()
        
        return distillation_loss * (self.temperature ** 2)  

    def configure_DART(self, model, args):

        if args.sparse:
            DART_config = {
                "K": args.pruned_layer,
                "image_token_start_index": args.image_token_start_index, 
                "image_token_length": args.image_token_length,
                "max_num_trunction": args.max_num_trunction,
                "reduction_ratio": args.reduction_ratio,
                "retain_token_num_for_llava_next": args.retain_token_num_for_llava_next,
                "pivot_image_token": args.pivot_image_token,
                "pivot_text_token": args.pivot_text_token,
            }
            model.config.DART_config = DART_config

        else:
            model.config.DART_config = None

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # 设置 ZeRO Stage 2 优化
        config_kwargs["zero_optimization"]["stage"] = 3
        # print("#### deepspeed config for ref_model")
        # print(config_kwargs)

        # 通过 DeepSpeed 初始化模型
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        dummty_optimizer = DeepSpeedCPUAdam(model.parameters(), lr=0.0001)
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs, optimizer=dummty_optimizer,
                                         model_parameters=None)
        model.eval()  # FIXME: is it valid?
        # model.module.eval()

        return model


    



