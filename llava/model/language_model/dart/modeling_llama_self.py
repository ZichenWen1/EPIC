import torch
from typing import Tuple, Callable
from transformers import AutoConfig
# from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaModel, _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, Cache, DynamicCache
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, Cache, DynamicCache
# from transformers.models.llama import LlamaConfig
from .configuration_llama import LlamaConfig
# from transformers.modeling_outputs import BaseModelOutputWithPast
from ...utils import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb
import json
import time
from einops import rearrange
import math
import torch.nn.functional as F
import torch.nn as nn
import math
import warnings
import os

# from transformers.modeling_outputs import CausalLMOutputWithPast
from ...utils import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv, LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, apply_rotary_pos_emb, LlamaRMSNorm,LlamaPreTrainedModel, _get_unpad_data

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import torch.distributed as dist
import ipdb
# rank = dist.get_rank()
# local_rank = int(os.environ['LOCAL_RANK'])


logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FastV(LlamaModel):

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        self.retained_indices = None
        self.reverse_position = None
        self.retained_image_num = 576 # fix
        super().__init__(config)
        self.config = config


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # pdb.set_trace()
        # print(f"using DART!!!")
        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                # rank = dist.get_rank()
                # local_rank = int(os.environ['LOCAL_RANK'])
                # if rank == 0:  
                #     ipdb.set_trace()
                # dist.barrier()
                
                # NEW start
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    K2 = DART_config['K2'] # second pruned layer, such as 15th, 20th layer

                    assert K2 >= K, "K2 should be greater than K"

                    do_second_pruned = DART_config['do_second_pruned']
                    only_do_second_pruned = DART_config['only_do_second_pruned']

                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    post_last_image_token_index = image_token_start_index + image_token_length
                    # self.reverse_position = seq_length - post_last_image_token_index
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1 and only_do_second_pruned == False:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]

                        # image_attention_score = self.last_attention.mean(dim=1)[0][-1][image_start:image_end]



                        # rank = dist.get_rank()
                        # local_rank = int(os.environ['LOCAL_RANK'])
                        # if rank == 0:  
                        #     ipdb.set_trace()
                        # dist.barrier()

                        # keep index
                        # retained_image_tokens_index = self.get_retained_image_token(self.config, last_layer_state, any_states)

                        # batch -> debug 发现 get_retained_image_token_batch有问题
                        # retained_image_tokens_index_batch = self.get_retained_image_token_batch(self.config, last_layer_state, any_states)
                        retained_image_tokens_index_batch = self.process_batch(self.config, last_layer_state, self.last_attention)
                        self.retained_image_num = retained_image_tokens_index_batch.size(1)

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        post_image_indices = torch.arange(image_token_start_index + image_token_length, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)


                        # keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            retained_image_tokens_index_batch,  # [batch_size, num_retained_tokens]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )
                        # position_ids = keep_indexs.unsqueeze(0)
                        # position_ids = keep_indexs.unsqueeze(1)
                        position_ids = keep_indexs
                        self.retained_indices = position_ids.clone()


                    if do_second_pruned and decoder_layer.self_attn.layer_idx == K2 and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # rank = dist.get_rank()
                        # local_rank = int(os.environ['LOCAL_RANK'])
                        # if rank == 0:  
                        #     ipdb.set_trace()
                        # dist.barrier()

                        # retained_image_tokens_index_batch = self.process_batch(self.config, last_layer_state, any_states)
                        current_seq_length = hidden_states.shape[1]

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        
                        post_image_start = image_token_start_index + self.retained_image_num
                        post_image_indices = torch.arange(post_image_start, current_seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)  # drop all image tokens

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )
                        if position_ids.size(0) != batch_size:
                            position_ids = position_ids.expand(batch_size, -1)

                        position_ids = torch.gather(position_ids, dim=1, index=keep_indexs)
                        self.retained_indices = position_ids.clone()
                else:
                    self.retained_indices = None
                # NEW end

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            
            else:
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # keep index
                        retained_image_tokens_index = self.get_retained_image_token(self.config, last_layer_state, any_states)

                        keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values

                        hidden_states = hidden_states[:,keep_indexs,:]
                        # print(f"hidden_states: {hidden_states.shape}")
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)


                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retained_indices=self.retained_indices,
        )#, self.retained_indices  # ! debug

    def process_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, attention_scores: torch.Tensor) -> torch.Tensor:

        retained_indices_batch = []

        for i in range(last_layer_state.shape[0]):  
            sample_last_layer_state = last_layer_state[i]  
            sample_last_layer_state = sample_last_layer_state.unsqueeze(0)

            sample_attention_score = attention_scores[i]  
            sample_attention_score = attention_scores.unsqueeze(0)

            retained_indices = self.get_retained_image_token(config, sample_last_layer_state, sample_attention_score)
            retained_indices_batch.append(retained_indices)

        retained_image_tokens_index_batch = torch.stack(retained_indices_batch, dim=0)

        return retained_image_tokens_index_batch

    
    def get_retained_image_token(self, config: LlamaConfig, last_layer_state: torch.Tensor, attention_score: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']  # pruned layer
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        MAX_NUM_TRUNCTION = DART_config['max_num_trunction']

        reduction_ratio = DART_config['reduction_ratio']

        device = last_layer_state.device

        image_attention_score = attention_score.mean(dim=1)[0][-1][image_token_start_index:image_token_start_index+image_token_length]

        retained_count = max(int(image_token_length * (1 - reduction_ratio)), 0)

        retained_image_tokens_index = image_attention_score.topk(int(retained_count)).indices + image_token_start_index

        # retained_image_tokens_index = torch.tensor(top_attention_rank_index, device=device)

        return retained_image_tokens_index


    # TODO： still have some bugs
    def get_retained_image_token_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        batch_size = any_states.size(0)
        device = last_layer_state.device

        # 调整维度 [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.size(0), any_states.size(2), -1)

        batch_retained_indices = []

        for batch_idx in range(batch_size):

            current_any_states = any_states[batch_idx]  # [seq_len, features]
            current_last_layer = last_layer_state[batch_idx]  # [seq_len, hidden_dim]

            image_tokens = current_any_states[image_token_start_index:image_token_start_index + image_token_length]
            query_tokens = current_any_states[image_token_start_index + image_token_length:]


            image_norms = torch.norm(image_tokens, p=1, dim=-1)
            query_norms = torch.norm(query_tokens, p=1, dim=-1)

            img_topk = image_norms.topk(pivot_image_token).indices + image_token_start_index
            q_topk = query_norms.topk(pivot_text_token).indices + (image_token_start_index + image_token_length)

            indices_set = set(img_topk.tolist() + q_topk.tolist())
            valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(img_topk.tolist())
            valid_list = list(valid_indices)

            for item in list(indices_set):
                if not valid_list:
                    break

                anchor_vec = current_last_layer[item].unsqueeze(0)
                candidate_vecs = current_last_layer[valid_list]

                similarities = -torch.nn.functional.cosine_similarity(
                    anchor_vec, candidate_vecs, dim=-1
                )

                curr_topk = min(TOKEN_TOPK, len(similarities))
                if curr_topk <= 0:
                    continue

                _, least_similar_indices = similarities.topk(curr_topk)
                selected_indices = [valid_list[i] for i in least_similar_indices]

                indices_set.update(selected_indices)
                valid_indices -= set(selected_indices)
                valid_list = list(valid_indices)


            indices_set -= set(q_topk.tolist())


            sorted_indices = torch.tensor(sorted(indices_set), 
                                    device=device, 
                                    dtype=torch.long)
            batch_retained_indices.append(sorted_indices)

            retained_image_tokens_index_batch = torch.stack(batch_retained_indices, dim=0)


        return retained_image_tokens_index_batch

    # TODO： still have some bugs
    def get_retained_image_token_batch_accelerated(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        device = last_layer_state.device
        batch_size = any_states.size(0)
        seq_len = any_states.size(2)

        # [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        image_region = any_states[:, image_token_start_index:image_token_start_index+image_token_length]
        image_norms = torch.norm(image_region, p=1, dim=-1)
        img_topk = image_norms.topk(pivot_image_token, dim=1).indices + image_token_start_index

        query_start = image_token_start_index + image_token_length
        query_region = any_states[:, query_start:]
        query_norms = torch.norm(query_region, p=1, dim=-1)
        q_topk = query_norms.topk(pivot_text_token, dim=1).indices + query_start

        all_indices = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        batch_retained_indices = []
        for bidx in range(batch_size):
            # initial mask
            indices_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

            indices_mask[img_topk[bidx]] = True
            indices_mask[q_topk[bidx]] = True

            valid_mask[image_token_start_index:image_token_start_index+image_token_length] = True
            valid_mask[img_topk[bidx]] = False  

            current_selected = torch.where(indices_mask.clone())[0].tolist()
            for item in current_selected:
                if not valid_mask.any():
                    break

                candidate_idx = torch.where(valid_mask)[0]
                anchor = last_layer_state[bidx, item]
                candidates = last_layer_state[bidx, candidate_idx]
                
                sim = -torch.cosine_similarity(anchor.unsqueeze(0), candidates, dim=-1)
                k = min(TOKEN_TOPK, len(candidate_idx))
                if k <= 0:
                    continue

                _, least_idx = torch.topk(sim, k, largest=False)
                selected = candidate_idx[least_idx]

                indices_mask[selected] = True
                valid_mask[selected] = False

            indices_mask[q_topk[bidx]] = False


            retained = all_indices[bidx][indices_mask].sort().values
            batch_retained_indices.append(retained)


        max_len = max(len(x) for x in batch_retained_indices)
        padded = [torch.nn.functional.pad(x, (0, max_len-len(x)), value=-1) for x in batch_retained_indices]
        return torch.stack(padded, dim=0).to(torch.long)




def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class Random(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        self.retained_indices = None
        self.reverse_position = None
        self.retained_image_num = 576 # fix
        super().__init__(config)
        self.config = config


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # pdb.set_trace()
        # print(f"using DART!!!")
        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                # rank = dist.get_rank()
                # local_rank = int(os.environ['LOCAL_RANK'])
                # if rank == 0:  
                #     ipdb.set_trace()
                # dist.barrier()
                
                # NEW start
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    K2 = DART_config['K2'] # second pruned layer, such as 15th, 20th layer

                    assert K2 >= K, "K2 should be greater than K"

                    do_second_pruned = DART_config['do_second_pruned']
                    only_do_second_pruned = DART_config['only_do_second_pruned']

                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    post_last_image_token_index = image_token_start_index + image_token_length
                    # self.reverse_position = seq_length - post_last_image_token_index
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1 and only_do_second_pruned == False:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # rank = dist.get_rank()
                        # local_rank = int(os.environ['LOCAL_RANK'])
                        # if rank == 0:  
                        #     ipdb.set_trace()
                        # dist.barrier()

                        # keep index
                        # retained_image_tokens_index = self.get_retained_image_token(self.config, last_layer_state, any_states)

                        # batch -> debug 发现 get_retained_image_token_batch有问题
                        # retained_image_tokens_index_batch = self.get_retained_image_token_batch(self.config, last_layer_state, any_states)
                        retained_image_tokens_index_batch = self.process_batch(self.config, last_layer_state, any_states)
                        self.retained_image_num = retained_image_tokens_index_batch.size(1)

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        post_image_indices = torch.arange(image_token_start_index + image_token_length, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)


                        # keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            retained_image_tokens_index_batch,  # [batch_size, num_retained_tokens]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )
                        # position_ids = keep_indexs.unsqueeze(0)
                        # position_ids = keep_indexs.unsqueeze(1)
                        position_ids = keep_indexs
                        self.retained_indices = position_ids.clone()


                    if do_second_pruned and decoder_layer.self_attn.layer_idx == K2 and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # rank = dist.get_rank()
                        # local_rank = int(os.environ['LOCAL_RANK'])
                        # if rank == 0:  
                        #     ipdb.set_trace()
                        # dist.barrier()

                        # retained_image_tokens_index_batch = self.process_batch(self.config, last_layer_state, any_states)
                        current_seq_length = hidden_states.shape[1]

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        
                        post_image_start = image_token_start_index + self.retained_image_num
                        post_image_indices = torch.arange(post_image_start, current_seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)  # drop all image tokens

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )
                        if position_ids.size(0) != batch_size:
                            position_ids = position_ids.expand(batch_size, -1)

                        position_ids = torch.gather(position_ids, dim=1, index=keep_indexs)
                        self.retained_indices = position_ids.clone()
                else:
                    self.retained_indices = None
                # NEW end

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            
            else:
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # keep index
                        retained_image_tokens_index = self.get_retained_image_token(self.config, last_layer_state, any_states)

                        keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values

                        hidden_states = hidden_states[:,keep_indexs,:]
                        # print(f"hidden_states: {hidden_states.shape}")
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)


                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retained_indices=self.retained_indices,
        )#, self.retained_indices  # ! debug

    def process_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:

        retained_indices_batch = []

        for i in range(any_states.shape[0]):  
            sample_last_layer_state = last_layer_state[i]  
            sample_last_layer_state = sample_last_layer_state.unsqueeze(0)
            sample_any_states = any_states[i]
            sample_any_states = sample_any_states.unsqueeze(0)

            retained_indices = self.get_retained_image_token(config, sample_last_layer_state, sample_any_states)
            retained_indices_batch.append(retained_indices)

        retained_image_tokens_index_batch = torch.stack(retained_indices_batch, dim=0)

        return retained_image_tokens_index_batch

    
    def get_retained_image_token(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']  # pruned layer
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        MAX_NUM_TRUNCTION = DART_config['max_num_trunction']

        reduction_ratio = DART_config['reduction_ratio']

        device = last_layer_state.device

        retained_count = max(int((1 - reduction_ratio) * image_token_length), 0)

        image_indexs = random.sample(range(image_token_start_index, image_token_start_index + image_token_length), int(retained_count))

        retained_image_tokens_index = torch.tensor(image_indexs, device=device)

        return retained_image_tokens_index


    # TODO： still have some bugs
    def get_retained_image_token_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        batch_size = any_states.size(0)
        device = last_layer_state.device

        # 调整维度 [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.size(0), any_states.size(2), -1)

        batch_retained_indices = []

        for batch_idx in range(batch_size):

            current_any_states = any_states[batch_idx]  # [seq_len, features]
            current_last_layer = last_layer_state[batch_idx]  # [seq_len, hidden_dim]

            image_tokens = current_any_states[image_token_start_index:image_token_start_index + image_token_length]
            query_tokens = current_any_states[image_token_start_index + image_token_length:]


            image_norms = torch.norm(image_tokens, p=1, dim=-1)
            query_norms = torch.norm(query_tokens, p=1, dim=-1)

            img_topk = image_norms.topk(pivot_image_token).indices + image_token_start_index
            q_topk = query_norms.topk(pivot_text_token).indices + (image_token_start_index + image_token_length)

            indices_set = set(img_topk.tolist() + q_topk.tolist())
            valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(img_topk.tolist())
            valid_list = list(valid_indices)

            for item in list(indices_set):
                if not valid_list:
                    break

                anchor_vec = current_last_layer[item].unsqueeze(0)
                candidate_vecs = current_last_layer[valid_list]

                similarities = -torch.nn.functional.cosine_similarity(
                    anchor_vec, candidate_vecs, dim=-1
                )

                curr_topk = min(TOKEN_TOPK, len(similarities))
                if curr_topk <= 0:
                    continue

                _, least_similar_indices = similarities.topk(curr_topk)
                selected_indices = [valid_list[i] for i in least_similar_indices]

                indices_set.update(selected_indices)
                valid_indices -= set(selected_indices)
                valid_list = list(valid_indices)


            indices_set -= set(q_topk.tolist())


            sorted_indices = torch.tensor(sorted(indices_set), 
                                    device=device, 
                                    dtype=torch.long)
            batch_retained_indices.append(sorted_indices)

            retained_image_tokens_index_batch = torch.stack(batch_retained_indices, dim=0)


        return retained_image_tokens_index_batch

    # TODO： still have some bugs
    def get_retained_image_token_batch_accelerated(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        device = last_layer_state.device
        batch_size = any_states.size(0)
        seq_len = any_states.size(2)

        # [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        image_region = any_states[:, image_token_start_index:image_token_start_index+image_token_length]
        image_norms = torch.norm(image_region, p=1, dim=-1)
        img_topk = image_norms.topk(pivot_image_token, dim=1).indices + image_token_start_index

        query_start = image_token_start_index + image_token_length
        query_region = any_states[:, query_start:]
        query_norms = torch.norm(query_region, p=1, dim=-1)
        q_topk = query_norms.topk(pivot_text_token, dim=1).indices + query_start

        all_indices = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        batch_retained_indices = []
        for bidx in range(batch_size):
            # initial mask
            indices_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

            indices_mask[img_topk[bidx]] = True
            indices_mask[q_topk[bidx]] = True

            valid_mask[image_token_start_index:image_token_start_index+image_token_length] = True
            valid_mask[img_topk[bidx]] = False  

            current_selected = torch.where(indices_mask.clone())[0].tolist()
            for item in current_selected:
                if not valid_mask.any():
                    break

                candidate_idx = torch.where(valid_mask)[0]
                anchor = last_layer_state[bidx, item]
                candidates = last_layer_state[bidx, candidate_idx]
                
                sim = -torch.cosine_similarity(anchor.unsqueeze(0), candidates, dim=-1)
                k = min(TOKEN_TOPK, len(candidate_idx))
                if k <= 0:
                    continue

                _, least_idx = torch.topk(sim, k, largest=False)
                selected = candidate_idx[least_idx]

                indices_mask[selected] = True
                valid_mask[selected] = False

            indices_mask[q_topk[bidx]] = False


            retained = all_indices[bidx][indices_mask].sort().values
            batch_retained_indices.append(retained)


        max_len = max(len(x) for x in batch_retained_indices)
        padded = [torch.nn.functional.pad(x, (0, max_len-len(x)), value=-1) for x in batch_retained_indices]
        return torch.stack(padded, dim=0).to(torch.long)



class DART(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        self.retained_indices = None
        self.reverse_position = None
        self.retained_image_num = 576 # fix
        super().__init__(config)
        self.config = config


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                # rank = dist.get_rank()
                # local_rank = int(os.environ['LOCAL_RANK'])
                # if rank == 0:  
                #     ipdb.set_trace()
                # dist.barrier()
                
                # HACK: NEW start
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    K2 = DART_config['K2'] # second pruned layer, such as 15th, 20th layer
                    assert K2 >= K, "K2 should be greater than K"

                    do_second_pruned = DART_config['do_second_pruned']
                    only_do_second_pruned = DART_config['only_do_second_pruned']

                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    post_last_image_token_index = image_token_start_index + image_token_length
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1 and only_do_second_pruned == False:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # keep index
                        retained_image_tokens_index_batch = self.process_batch(self.config,   last_layer_state, any_states) # TODO: batch process needs to be optimized
                        self.retained_image_num = retained_image_tokens_index_batch.size(1)

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        post_image_indices = torch.arange(image_token_start_index + image_token_length, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)


                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            retained_image_tokens_index_batch,  # [batch_size, num_retained_tokens]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )

                        position_ids = keep_indexs
                        self.retained_indices = position_ids.clone()


                    if do_second_pruned and decoder_layer.self_attn.layer_idx == K2 and seq_length > 1: # HACK: ultimately not adopted
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        current_seq_length = hidden_states.shape[1]

                        pre_image_indices = torch.arange(0, image_token_start_index, device=device).unsqueeze(0).expand(batch_size, -1)
                        
                        post_image_start = image_token_start_index + self.retained_image_num
                        post_image_indices = torch.arange(post_image_start, current_seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

                        keep_indexs = torch.cat((
                            pre_image_indices,  # [batch_size, pre_image_len]
                            post_image_indices  # [batch_size, post_image_len]
                        ), dim=1)  # drop all image tokens

                        # sort index
                        keep_indexs, _ = keep_indexs.sort(dim=1)

                        # hidden_states = hidden_states[:,keep_indexs,:]
                        hidden_states = torch.gather(
                            hidden_states, 
                            dim=1, 
                            index=keep_indexs.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                        )

                        if attention_mask is not None:
                            attention_mask = torch.gather(
                                attention_mask, 
                                dim=1, 
                                index=keep_indexs
                            )
                        if position_ids.size(0) != batch_size:
                            position_ids = position_ids.expand(batch_size, -1)

                        position_ids = torch.gather(position_ids, dim=1, index=keep_indexs)
                        self.retained_indices = position_ids.clone()
                else:
                    self.retained_indices = None
                # HACK: NEW end

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            
            else:
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        any_states = layer_outputs[-2]  # HACK: get any_states

                        # keep index
                        retained_image_tokens_index = self.get_retained_image_token(self.config, last_layer_state, any_states)

                        keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values

                        hidden_states = hidden_states[:,keep_indexs,:]
                        # print(f"hidden_states: {hidden_states.shape}")
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)


                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retained_indices=self.retained_indices,
        )#, self.retained_indices  # ! debug

    def process_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:

        retained_indices_batch = []

        for i in range(any_states.shape[0]):  
            sample_last_layer_state = last_layer_state[i]  
            sample_last_layer_state = sample_last_layer_state.unsqueeze(0)
            sample_any_states = any_states[i]
            sample_any_states = sample_any_states.unsqueeze(0)

            retained_indices = self.get_retained_image_token(config, sample_last_layer_state, sample_any_states)
            retained_indices_batch.append(retained_indices)

        retained_image_tokens_index_batch = torch.stack(retained_indices_batch, dim=0)

        return retained_image_tokens_index_batch

    
    def get_retained_image_token(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']  # pruned layer
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        MAX_NUM_TRUNCTION = DART_config['max_num_trunction']

        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']


        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio)*image_token_length / (pivot_image_token + pivot_text_token))

        device = last_layer_state.device

        # if config.text_length is not None:
        #     text_length = config.text_length
        #     image_token_length = any_states.shape[2] - text_length
        #     retain_token_num_for_llava_next = min(DART_config['retain_token_num_for_llava_next'], image_token_length - pivot_image_token)
        #     TOKEN_TOPK = int(retain_token_num_for_llava_next // (pivot_image_token + pivot_text_token))

        any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.shape[0], any_states.shape[2], -1)

        k_states_image_token = any_states[0][image_token_start_index:image_token_start_index + image_token_length, :]
        k_states_query_token = any_states[0][image_token_start_index + image_token_length:, :]

        k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
        k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

        image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist() 
        query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist()
        indices_set = set(image_indices + query_indices)

        valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)

        valid_indices_list = list(valid_indices)  
        for item in list(indices_set):
            valid_vectors = last_layer_state[0][valid_indices_list, :]
            cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[0][item, :], valid_vectors, dim=-1)

            k = min(TOKEN_TOPK, len(cos_sim))
            if k == 0 or k < 0:
                continue
            # TOKEN_TOPK = min(TOKEN_TOPK, cos_sim.shape[0])
            top_k_indices = cos_sim.topk(k).indices

            top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
            indices_set.update(top_k_real_indices)
            
            valid_indices.difference_update(top_k_real_indices)
            valid_indices_list = list(valid_indices)  

        indices_set.difference_update(query_indices)

        retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

        # print(f"retained_image_tokens_index_length: {len(retained_image_tokens_index)}")

        return retained_image_tokens_index


    # TODO： still have some bugs
    def get_retained_image_token_batch(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        K = DART_config['K']
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        batch_size = any_states.size(0)
        device = last_layer_state.device

        # 调整维度 [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.size(0), any_states.size(2), -1)

        batch_retained_indices = []

        for batch_idx in range(batch_size):

            current_any_states = any_states[batch_idx]  # [seq_len, features]
            current_last_layer = last_layer_state[batch_idx]  # [seq_len, hidden_dim]

            image_tokens = current_any_states[image_token_start_index:image_token_start_index + image_token_length]
            query_tokens = current_any_states[image_token_start_index + image_token_length:]


            image_norms = torch.norm(image_tokens, p=1, dim=-1)
            query_norms = torch.norm(query_tokens, p=1, dim=-1)

            img_topk = image_norms.topk(pivot_image_token).indices + image_token_start_index
            q_topk = query_norms.topk(pivot_text_token).indices + (image_token_start_index + image_token_length)

            indices_set = set(img_topk.tolist() + q_topk.tolist())
            valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(img_topk.tolist())
            valid_list = list(valid_indices)

            for item in list(indices_set):
                if not valid_list:
                    break

                anchor_vec = current_last_layer[item].unsqueeze(0)
                candidate_vecs = current_last_layer[valid_list]

                similarities = -torch.nn.functional.cosine_similarity(
                    anchor_vec, candidate_vecs, dim=-1
                )

                curr_topk = min(TOKEN_TOPK, len(similarities))
                if curr_topk <= 0:
                    continue

                _, least_similar_indices = similarities.topk(curr_topk)
                selected_indices = [valid_list[i] for i in least_similar_indices]

                indices_set.update(selected_indices)
                valid_indices -= set(selected_indices)
                valid_list = list(valid_indices)


            indices_set -= set(q_topk.tolist())


            sorted_indices = torch.tensor(sorted(indices_set), 
                                    device=device, 
                                    dtype=torch.long)
            batch_retained_indices.append(sorted_indices)

            retained_image_tokens_index_batch = torch.stack(batch_retained_indices, dim=0)


        return retained_image_tokens_index_batch

    # TODO： still have some bugs
    def get_retained_image_token_batch_accelerated(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        image_token_start_index = DART_config['image_token_start_index']
        image_token_length = DART_config['image_token_length']
        pivot_image_token = DART_config['pivot_image_token']
        pivot_text_token = DART_config['pivot_text_token']
        reduction_ratio = DART_config['reduction_ratio']
        TOKEN_TOPK = int((1 - reduction_ratio) * image_token_length / (pivot_image_token + pivot_text_token))

        device = last_layer_state.device
        batch_size = any_states.size(0)
        seq_len = any_states.size(2)

        # [batch, seq_len, layers*hidden_dim]
        any_states = any_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        image_region = any_states[:, image_token_start_index:image_token_start_index+image_token_length]
        image_norms = torch.norm(image_region, p=1, dim=-1)
        img_topk = image_norms.topk(pivot_image_token, dim=1).indices + image_token_start_index

        query_start = image_token_start_index + image_token_length
        query_region = any_states[:, query_start:]
        query_norms = torch.norm(query_region, p=1, dim=-1)
        q_topk = query_norms.topk(pivot_text_token, dim=1).indices + query_start

        all_indices = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        batch_retained_indices = []
        for bidx in range(batch_size):
            # initial mask
            indices_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

            indices_mask[img_topk[bidx]] = True
            indices_mask[q_topk[bidx]] = True

            valid_mask[image_token_start_index:image_token_start_index+image_token_length] = True
            valid_mask[img_topk[bidx]] = False  

            current_selected = torch.where(indices_mask.clone())[0].tolist()
            for item in current_selected:
                if not valid_mask.any():
                    break

                candidate_idx = torch.where(valid_mask)[0]
                anchor = last_layer_state[bidx, item]
                candidates = last_layer_state[bidx, candidate_idx]
                
                sim = -torch.cosine_similarity(anchor.unsqueeze(0), candidates, dim=-1)
                k = min(TOKEN_TOPK, len(candidate_idx))
                if k <= 0:
                    continue

                _, least_idx = torch.topk(sim, k, largest=False)
                selected = candidate_idx[least_idx]

                indices_mask[selected] = True
                valid_mask[selected] = False

            indices_mask[q_topk[bidx]] = False


            retained = all_indices[bidx][indices_mask].sort().values
            batch_retained_indices.append(retained)


        max_len = max(len(x) for x in batch_retained_indices)
        padded = [torch.nn.functional.pad(x, (0, max_len-len(x)), value=-1) for x in batch_retained_indices]
        return torch.stack(padded, dim=0).to(torch.long)

    

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1) # ! debug
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, query_states, key_states, value_states # ! debug


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1) # ! debug
        # key_states_without_rope = key_states
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, query_states.permute(0, 2, 1, 3), key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3) # ! debug
        # return attn_output, attn_weights, past_key_value, query_states.permute(0, 2, 1, 3), key_states_without_rope, value_states.permute(0, 2, 1, 3) # ! debug

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)  # ! debug
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, query_states, key_states, value_states # ! debug


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        x_states = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, query_states, key_states, value_states = self.self_attn( # ! debug
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        outputs += (x_states, query_states, key_states, value_states) # ! debug

        return outputs



class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(   # debug
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            retained_indices = outputs.retained_indices
            if retained_indices is not None:
                # rank = dist.get_rank()
                # local_rank = int(os.environ['LOCAL_RANK'])
                # if rank == 0:  
                #     ipdb.set_trace()

                # dist.barrier()  # add by zichen
                # labels = labels[..., retained_indices]  # ! debug
                labels = torch.gather(labels, dim=1, index=retained_indices)  # ! debug
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # rank = dist.get_rank()
        # local_rank = int(os.environ['LOCAL_RANK'])
        # if rank == 0:  
        #     ipdb.set_trace()

        # dist.barrier()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            labels=labels,   # ! debug add by zichen
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            retained_indices=retained_indices,
        )#, retained_indices # ! debug

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


