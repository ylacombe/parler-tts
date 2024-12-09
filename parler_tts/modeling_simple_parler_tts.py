# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch PrefixLMParlerTTS model."""
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForTextEncoding
from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation import GenerationMixin 
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
    replace_return_docstrings,
    is_torchdynamo_compiling,
)
from transformers.utils.import_utils import is_flash_attn_2_available, is_torch_greater_or_equal

from .configuration_simple_parler_tts import PrefixLMParlerTTSConfig, PrefixLMParlerTTSDecoderConfig
from .dac_wrapper import DACConfig, DACModel

from importlib.metadata import version
from packaging.version import Version

is_dac_integrated_to_transformers = Version(version("transformers")) > Version("4.44.2dev")
if not is_dac_integrated_to_transformers:
    AutoConfig.register("dac", DACConfig)
else:
    AutoConfig.register("dac_on_the_hub", DACConfig)

AutoModel.register(DACConfig, DACModel)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

if is_torch_greater_or_equal("2.5"):
    from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask
    flex_attention = torch.compile(flex_attention) #, dynamic=False)
    # create_block_mask = torch.compile(create_block_mask, dynamic=False)



_CONFIG_FOR_DOC = "PrefixLMParlerTTSConfig"


NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache, "sliding_window": SlidingWindowCache}

def eager_attention_forward(config, query, key, value, mask, **_kwargs):
    key_states = repeat_kv(key, config.num_key_value_groups)
    value_states = repeat_kv(value, config.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * config.scaling

    if mask is not None:  # no matter the length, we just slice it
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=config.attention_dropout, training=config.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def flash_attention_forward(config, query, key, value, mask, target_dtype=torch.float16, **_kwargs):
    if mask is not None:
        seq_len = mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor rotary embedding
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    dropout_rate = config.attention_dropout if config.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        mask,
        seq_len,
        dropout=dropout_rate,
        softmax_scale=config.scaling,
        is_causal=config.is_causal,
        sliding_window=None,#config.sliding_window,
        use_top_left_mask=config._flash_attn_uses_top_left_mask,
    )

    return attn_output, None


def mod_causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def flex_attention_forward(config, query, key, value, mask, target_dtype=torch.float16, output_attentions=False, **_kwargs):
    block_mask = _kwargs.get("block_mask")

    input_dtype = query.dtype
    if input_dtype == torch.float32:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    attn_output = flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=_kwargs["enable_gqa"],
        return_lse=output_attentions,
    )

    if not output_attentions:
        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    else:
        # Reshape outputs
        return attn_output[0].transpose(1, 2).contiguous(), attn_output[1]


def sdpa_attention_forward(config, query, key, value, mask, **_kwargs):
    # key = repeat_kv(key, config.num_key_value_groups)
    # value = repeat_kv(value, config.num_key_value_groups)

    causal_mask = mask
    if mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and query.shape[1] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=config.attention_dropout if config.training else 0.0,
        is_causal=is_causal,
        scale=config.scaling,
        enable_gqa=_kwargs["enable_gqa"],
    )
    
    # Reshape outputs
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


SIMPLE_PARLER_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


@dataclass
class PrefixLMParlerTTSSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_codebook_losses: Optional[List[torch.FloatTensor]] = None

@dataclass
class PrefixLMParlerTTSCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_codebook_losses: Optional[List[torch.FloatTensor]] = None

def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
    """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
    the mask is set to -1, and otherwise setting to the value detailed in the mask."""
    seq_len = input_ids.shape[-1]
    decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
    input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
    return input_ids


def give_delay(codebook, strategy):
    if strategy == "group":
        return math.floor((codebook-1) / 2 + 1)
    elif strategy == "delay_first_8":
        return min(codebook, 7)
    elif strategy == "group_first_6":
        return math.floor((codebook-1) / 2 + 1) if codebook < 7 else 3
    else:
        return codebook
    
def build_special_tokens_delay_pattern(strategy, num_codebooks, max_length):
    if strategy == "group":
        triangular_num_codebooks = math.floor(num_codebooks / 2) + 1
    elif strategy == "delay_first_8":
        triangular_num_codebooks = min(num_codebooks, 8)
    elif strategy == "group_first_6":
        triangular_num_codebooks = min(num_codebooks, 7)
        triangular_num_codebooks = math.floor(triangular_num_codebooks / 2) + 1
    else:
        triangular_num_codebooks = num_codebooks
    
    # construct a pattern mask that indicates the positions of padding tokens for each codebook
    # first fill the upper triangular part (the EOS padding)
    eos_delay_pattern = torch.triu(
        torch.ones((triangular_num_codebooks, max_length), dtype=torch.bool), diagonal=max_length - triangular_num_codebooks + 1
    )
    # then fill the lower triangular part (the BOS padding)
    bos_delay_pattern = torch.tril(torch.ones((triangular_num_codebooks, max_length), dtype=torch.bool))
    
    if strategy == "group":
        tmp_bos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)
        tmp_eos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)

        # the first codebook delay stays the same
        tmp_eos_delay_pattern[0] = eos_delay_pattern[0]
        tmp_bos_delay_pattern[0] = bos_delay_pattern[0]
        
        # duplicate the other codebooks delays
        eos_delay_pattern = torch.repeat_interleave(eos_delay_pattern[1:], 2, dim=0)
        bos_delay_pattern = torch.repeat_interleave(bos_delay_pattern[1:], 2, dim=0)
        
        # drop the last duplicated codebook if it's not even
        tmp_eos_delay_pattern[1:] = eos_delay_pattern[:-1] if (num_codebooks - 1) % 2 else eos_delay_pattern
        tmp_bos_delay_pattern[1:] = bos_delay_pattern[:-1] if (num_codebooks - 1) % 2 else bos_delay_pattern
        bos_delay_pattern = tmp_bos_delay_pattern
        eos_delay_pattern = tmp_eos_delay_pattern
    elif strategy == "delay_first_8":
        # add handmade block pattern for the last num_codebooks-8 codebooks
        tmp_bos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)
        tmp_eos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)
        
        tmp_bos_delay_pattern[:triangular_num_codebooks] =  bos_delay_pattern
        tmp_eos_delay_pattern[:triangular_num_codebooks] =  eos_delay_pattern
        
        tmp_bos_delay_pattern[triangular_num_codebooks:, :8] = True
        bos_delay_pattern = tmp_bos_delay_pattern

        # eos stays the same
        eos_delay_pattern = tmp_eos_delay_pattern
    elif strategy == "group_first_6":
        tmp_bos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)
        tmp_eos_delay_pattern = torch.zeros((num_codebooks, max_length), dtype=torch.bool)

        # the first codebook delay stays the same
        tmp_eos_delay_pattern[0] = eos_delay_pattern[0]
        tmp_bos_delay_pattern[0] = bos_delay_pattern[0]
        
        # duplicate the other codebooks delays
        eos_delay_pattern = torch.repeat_interleave(eos_delay_pattern[1:], 2, dim=0)
        bos_delay_pattern = torch.repeat_interleave(bos_delay_pattern[1:], 2, dim=0)
        
        # drop the last duplicated codebook if it's not even
        tmp_eos_delay_pattern[1:7] = eos_delay_pattern
        tmp_bos_delay_pattern[1:7] = bos_delay_pattern
        
        tmp_bos_delay_pattern[7:, :4] = True
        bos_delay_pattern = tmp_bos_delay_pattern
        eos_delay_pattern = tmp_eos_delay_pattern

    return bos_delay_pattern, eos_delay_pattern

def build_delay_pattern_mask(
    input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int, max_length: int, num_codebooks: int, strategy: str = None
):
    """Build a delayed pattern mask to the input_ids. 
    
    
    If `strategy` is not specified or `delay`:
    
    Each codebook is offset by the previous codebook by one, giving a delayed pattern mask at the start of sequence and 
    end of sequence. Take the example where there are 4 codebooks and a max sequence length of 8, we have the delayed 
    pattern mask of shape `(codebooks, seq_len)`:
    - [B, -1, -1, -1, -1, P, P, P]
    - [B, B, -1, -1, -1, -1, P, P]
    - [B, B, B, -1, -1, -1, -1, P]
    - [B, B, B, B, -1, -1, -1, -1]
    where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
    a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
    mask is set to the value in the prompt:
    - [B, a, b, -1, -1, P, P, P]
    - [B, B, c, d, -1, -1, P, P]
    - [B, B, B, e, f, -1, -1, P]
    - [B, B, B, B, g, h, -1, -1]
    where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
    tokens in our prediction.
    
    If `strategy` is `group`:
    Except the first codebook, delays are applied to group of codebooks. To illustrate with 6 codebooks:
    - [B, a, b, -1, -1,  P,  P,  P]
    - [B, B, c,  d, -1, -1,  P,  P]
    - [B, B, e,  f, -1, -1,  P,  P]
    - [B, B, B,  g,  h, -1, -1,  P]
    - [B, B, B,  i,  j, -1, -1,  P]
    - [B, B, B,  B,  k,  l, -1, -1]

    If `strategy` is `delay_first_8`:
    1-delays are apply for the 8 first codebooks. The rest of the codebooks applies a delay of 8. 
    
    if `strategy` is `group_first_6`:
    Groups are applied from the second codebook to the 7th included. The first codebook doesn't have delay. The rest of the codebooks applies the same as the 7th codebook.
    """
    # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape

    input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1

    # we only apply the mask if we have a large enough seq len - otherwise we return as is
    if max_length < 2 * num_codebooks - 1:
        return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

    # fill the shifted ids with the prompt entries, offset by the codebook idx
    for codebook in range(num_codebooks):
        # mono channel - loop over the codebooks one-by-one
        codebook_delay = give_delay(codebook, strategy)
        input_ids_shifted[:, codebook, codebook_delay : seq_len + codebook_delay] = input_ids[:, codebook]

    # construct a pattern mask that indicates the positions of padding tokens for each codebook
    bos_delay_pattern, eos_delay_pattern = build_special_tokens_delay_pattern(strategy, num_codebooks, max_length)

    bos_mask = ~(bos_delay_pattern).to(input_ids.device)
    eos_mask = ~(eos_delay_pattern).to(input_ids.device)
    mask = ~(bos_delay_pattern + eos_delay_pattern).to(input_ids.device)
    input_ids = mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id

    # find the first position to start generating - this is the first place we have the -1 token
    # and will always be in the first codebook (since it has no codebook offset)
    first_codebook_ids = input_ids[:, 0, :]
    start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
    if len(start_ids) > 0:
        first_start_id = min(start_ids)
    else:
        # we have no tokens that need to be filled - return entire matrix of input ids
        first_start_id = seq_len

    # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
    input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    return input_ids, pattern_mask


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

    
# Copied from transformers.models.gemma.modeling_gemma.GemmaRMSNorm with Gemma->PrefixLMParlerTTSDecoder
class PrefixLMParlerTTSRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Ignore copy

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # Ignore copy
    def forward(self, x):
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


ALL_LAYERNORM_LAYERS.append(PrefixLMParlerTTSRMSNorm)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->PrefixLMParlerTTSDecoder
class PrefixLMParlerTTSRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.compile #jit.script
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PrefixLMParlerTTSDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PrefixLMParlerTTSDecoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.scaling = 1 / math.sqrt(self.head_dim)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    # Copied from transformers.models.gemma.modeling_gemma.GemmaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # Ignore copy

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = (
                {"sin": sin, "cos": cos, "cache_position": cache_position}
            )  # Ignore copy
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if output_attentions and self.config._attn_implementation in ["sdpa", "flash_attention_2"]:
            logger.warning_once("Setting `attention_type` to `flex_attention` because `output_attentions=True`")
            attention_type = "flex_attention"
        else:
            attention_type = self.config._attn_implementation
            
        kwargs["enable_gqa"] = self.num_key_value_groups > 1
        kwargs["target_dtype"] = value_states.dtype

        attn_output, attn_weights = SIMPLE_PARLER_ATTENTION_FUNCTION[attention_type](
            self, query_states, key_states, value_states, attention_mask, output_attentions=output_attentions, **kwargs
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PrefixLMParlerTTSGatingMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.activation_fn = ACT2FN[config.activation_function]
        ffn_dim = config.ffn_dim
        hidden_size = config.hidden_size
        self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim // 2, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int = None) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) if layer_idx is None else self.fc1(hidden_states, layer_idx)

        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, sequence_length, 2, -1)
        hidden_states = self.activation_fn(hidden_states[..., 0, :]) * hidden_states[..., 1, :]
        hidden_states = self.fc2(hidden_states) if layer_idx is None else self.fc2(hidden_states, layer_idx)
        return hidden_states
    

class PrefixLMParlerTTSDecoderLayer(nn.Module):
    def __init__(self, config: PrefixLMParlerTTSDecoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PrefixLMParlerTTSDecoderAttention(config=config, layer_idx=layer_idx)

        self.mlp = PrefixLMParlerTTSGatingMLP(config)
        self.input_layernorm = PrefixLMParlerTTSRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PrefixLMParlerTTSRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
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
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            cos=cos,
            sin=sin,
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

        return outputs


class PrefixLMParlerTTSPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PrefixLMParlerTTSDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PrefixLMParlerTTSDecoderDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    main_input_name = "input_ids"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PrefixLMParlerTTSDecoder(PrefixLMParlerTTSPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PrefixLMParlerTTSDecoderLayer`]
    """

    def __init__(self, config: PrefixLMParlerTTSDecoderConfig):
        super().__init__(config)
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks)]
        )

        self.rotary_emb = PrefixLMParlerTTSRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [PrefixLMParlerTTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PrefixLMParlerTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn_implementation = config._attn_implementation
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
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # pre-pended to inputs
        encoder_attention_mask: Optional[torch.LongTensor] = None, # pre-pended to inputs
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        block_mask=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
            

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # (bsz * codebooks, seq_len) -> (bsz, codebooks, seq_len)
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = sum([self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks)])
            
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True  # noqa: F841
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        prepended_sequence_length = 0
        # if encoder__hidden_states, fuse to inputs_embeds and update input shape
        if encoder_hidden_states is not None:
            prepended_sequence_length = encoder_hidden_states.shape[-2]
            inputs_embeds = torch.cat([encoder_hidden_states, inputs_embeds], dim=1)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0].item()
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()


        if encoder_attention_mask is not None and attention_mask is not None:
            attention_mask = torch.cat([encoder_attention_mask, attention_mask], dim=1)
        elif encoder_attention_mask is not None:
            if past_key_values_length == 0:
                attention_mask = torch.cat(
                    [
                        encoder_attention_mask,
                        torch.ones(input_shape, device=self.device, dtype=encoder_attention_mask.dtype),
                    ],
                    dim=1,
                )
            else:
                # In the generation case of `prompt_cross_attention=True`, we need to recreate an attention mask from scratch
                # to be able to prepend the prompt attention mask.
                # Since we generate token per token, we can recompute the generated length from the information we have.
                generated_length = past_key_values_length - encoder_attention_mask.shape[1] + 1
                attention_mask = torch.cat(
                    [
                        encoder_attention_mask,
                        torch.ones(
                            (input_shape[0], generated_length), device=self.device, dtype=encoder_attention_mask.dtype
                        ),
                    ],
                    dim=1,
                )
                
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1] + prepended_sequence_length, device=inputs_embeds.device
            )

        if position_ids is None: 
            if attention_mask is not None:
                position_ids = torch.clamp(attention_mask.cumsum(1) - 1, min=0)[:, past_key_values_length:] # cache_position.unsqueeze(0)
            else:
                position_ids = cache_position.unsqueeze(0)
                
        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            
        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)  # Ignore copy

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
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    block_mask,
                    cos,
                    sin,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    block_mask=block_mask,
                    cos=cos,
                    sin=sin,
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

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=None,#self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Moshi
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: PrefixLMParlerTTSDecoderConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`PrefixLMParlerTTSDecoderConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            # if config.sliding_window is not None:
            #     # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
            #     # the check is needed to verify is current checkpoint was trained with sliding window or not
            #     if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
            #         sliding_attend_mask = torch.arange(target_length, device=device) <= (
            #             cache_position.reshape(-1, 1) - config.sliding_window
            #         )
            #         diagonal_attend_mask |= sliding_attend_mask
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask



# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenModel with Musicgen->PrefixLMParlerTTS
class PrefixLMParlerTTSModel(PrefixLMParlerTTSPreTrainedModel):
    def __init__(self, config: PrefixLMParlerTTSDecoderConfig):
        super().__init__(config)
        self.decoder = PrefixLMParlerTTSDecoder(config)
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_mask=None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            block_mask=block_mask,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class PrefixLMParlerTTSForCausalLM(PrefixLMParlerTTSPreTrainedModel):
    def __init__(self, config: PrefixLMParlerTTSDecoderConfig):
        super().__init__(config)

        self.model = PrefixLMParlerTTSModel(config)

        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.num_codebooks = config.num_codebooks
        
        self.use_fused_lm_heads = config.use_fused_lm_heads
        if self.use_fused_lm_heads:
            self.lm_heads = nn.Linear(config.hidden_size, config.vocab_size * config.num_codebooks, bias=False)
        else:
            self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=PrefixLMParlerTTSCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_reduction: str = "mean",
        block_mask=None,
        ) -> Union[Tuple, PrefixLMParlerTTSCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            block_mask=block_mask,
        )

        hidden_states = outputs[0]

        if self.use_fused_lm_heads:
            lm_logits = self.lm_heads(hidden_states).view(hidden_states.shape[0], -1, self.num_codebooks, self.vocab_size).transpose(1,2)
        else:
            lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        loss = None
        per_codebook_losses = None
        if labels is not None:
            codebook_weights = self.config.codebook_weights
            # since encoder hidden states have concatenated to hidden states, take the last hidden states corresponding to labels
            logits = lm_logits[:, :, -labels.shape[1] :]

            loss_fct = CrossEntropyLoss(reduction=loss_reduction)
            loss = torch.zeros([], device=self.device)
            
            per_codebook_losses = []

            # (bsz, vocab_size, seq_len, num_codebooks), (bsz, seq_len, num_codebooks)
            labels = labels.masked_fill(labels == self.config.bos_token_id, -100)

            # we use every codebooks token AND one single EOS at the end of each codebooks
            mask = (input_ids.transpose(1, 2) != self.config.eos_token_id) & ((labels != -100))

            # per codebook cross-entropy
            for codebook in range(self.config.num_codebooks):
                codebook_logits = logits[:, codebook].contiguous().view(-1, logits.shape[-1])
                codebook_mask = mask[..., codebook].contiguous().view(-1)
                codebook_labels = labels[..., codebook].contiguous().view(-1)

                codebook_loss = loss_fct(codebook_logits[codebook_mask], codebook_labels[codebook_mask])
                per_codebook_losses.append(codebook_loss)

                if codebook_weights is not None:
                    codebook_loss = codebook_loss*codebook_weights[codebook]
                    
                loss += codebook_loss

            if codebook_weights is not None:
                loss = loss / sum(codebook_weights)
            else:
                loss = loss / self.config.num_codebooks

        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output + (per_codebook_losses, )) if loss is not None else output

        return PrefixLMParlerTTSCausalLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            per_codebook_losses=per_codebook_losses,
        )

    # Ignore copy
    def build_delay_pattern_mask(
        self, input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int, max_length: int = None, strategy: str = None
    ):
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [B, -1, -1, -1, -1, P, P, P]
        - [B, B, -1, -1, -1, -1, P, P]
        - [B, B, B, -1, -1, -1, -1, P]
        - [B, B, B, B, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [B, a, b, -1, -1, P, P, P]
        - [B, B, c, d, -1, -1, P, P]
        - [B, B, B, e, f, -1, -1, P]
        - [B, B, B, B, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        max_length = max_length if max_length is not None else self.generation_config.max_length
        strategy = strategy if strategy is not None else self.config.delay_strategy
        return build_delay_pattern_mask(input_ids, bos_token_id, pad_token_id, max_length, self.num_codebooks, strategy=strategy)

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        return apply_delay_pattern_mask(input_ids, decoder_pad_token_mask)


class PrefixLMParlerTTSForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = PrefixLMParlerTTSConfig
    base_model_prefix = "decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(
        self,
        config: Optional[PrefixLMParlerTTSConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PrefixLMParlerTTSForCausalLM] = None,
    ):
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError(
                "Either a configuration has to be provided, or all three of text encoder, audio encoder and Parler-TTS decoder."
            )
        if config is None:
            config = PrefixLMParlerTTSConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")


        # initialize with config
        super().__init__(config)

        if text_encoder is None:
            from transformers.models.auto.modeling_auto import AutoModelForTextEncoding

            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder_config)

        if audio_encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            audio_encoder = AutoModel.from_config(config.audio_encoder_config)

        if decoder is None:
            decoder = PrefixLMParlerTTSForCausalLM._from_config(config.decoder_config)

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

        if self.text_encoder.config.to_dict() != self.config.text_encoder_config.to_dict():
            logger.warning(
                f"Config of the text_encoder: {self.text_encoder.__class__} is overwritten by shared text_encoder config:"
                f" {self.config.text_encoder_config}"
            )
        if self.audio_encoder.config.to_dict() != self.config.audio_encoder_config.to_dict():
            logger.warning(
                f"Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config:"
                f" {self.config.audio_encoder_config}"
            )
        if self.decoder.config.to_dict() != self.config.decoder_config.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder_config}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.config.text_encoder_config._attn_implementation = self.text_encoder.config._attn_implementation
        self.config.audio_encoder_config._attn_implementation = self.audio_encoder.config._attn_implementation
        self.config.decoder_config._attn_implementation = self.config._attn_implementation
        self.text_encoder.config = self.config.text_encoder_config
        self.audio_encoder.config = self.config.audio_encoder_config
        self.decoder.config = self.config.decoder_config

        # text encoder outputs might need to be projected to different dimension for decoder
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
        ):
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)

        # prompt embeddings
        self.embed_prompts = nn.Embedding(config.vocab_size, self.decoder.config.hidden_size)

        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head"
            )

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        audio_encoder_signature = set(inspect.signature(self.audio_encoder.decode).parameters.keys())
        self.use_audio_scales = "audio_scales" in audio_encoder_signature

        self.use_4dim_audio_codes = False
        audio_type = audio_encoder.config.model_type
        if audio_type in {"encodec", "dac_on_the_hub"} or (audio_type == "dac" and not is_dac_integrated_to_transformers):
            self.use_4dim_audio_codes = True 
 
        # Initialize projection and embedding layers and tie text encoder and decoder weights if set accordingly
        self.post_init()

    def _init_weights(self, module):
        std = self.decoder.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def tie_weights(self):
        # tie text encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie text encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_audio_encoder(self):
        return self.audio_encoder

    def get_text_encoder(self):
        return self.text_encoder

    def get_encoder(self):
        # get the text encoder to compute the encoder hidden-states for generation
        return self.get_text_encoder()

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

# TODO: check if necessary
#    @classmethod
#    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
#        r"""
#        Example:
#
#        ```python
#        >>> from simple_parler_tts import PrefixLMParlerTTSForConditionalGeneration
#
#        >>> model = PrefixLMParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
#        ```"""
#
#        # At the moment fast initialization is not supported for composite models
#        if kwargs.get("_fast_init", False):
#            logger.warning(
#                "Fast initialization is currently not supported for PrefixLMParlerTTSForConditionalGeneration. "
#                "Falling back to slow initialization..."
#            )
#        kwargs["_fast_init"] = False
#
#        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @replace_return_docstrings(output_type=PrefixLMParlerTTSSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        prompt_input_ids: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        prompt_hidden_states: Optional[torch.FloatTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_reduction: str = "mean",
        block_mask = None,
        **kwargs,
    ) -> Union[Tuple, PrefixLMParlerTTSSeq2SeqLMOutput]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, PrefixLMParlerTTSForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("parler-tts/parler-tts-mini-v1")
        >>> model = PrefixLMParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")

        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )

        >>> pad_token_id = model.generation_config.pad_token_id
        >>> decoder_input_ids = (
        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
        ...     * pad_token_id
        ... )

        >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
        >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
        torch.Size([8, 1, 2048])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_text_encoder = {
            argument[len("text_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_encoder_")
        }

        kwargs_audio_encoder = {
            argument[len("audio_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("audio_encoder_")
        }


        if prompt_hidden_states is None:
            if prompt_input_ids is not None:
                prompt_hidden_states = self.embed_prompts(prompt_input_ids)

        if encoder_outputs is None and (input_ids is not None or inputs_embeds is not None):
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_text_encoder,
            )
            encoder_hidden_states = encoder_outputs[0]

            # optionally project encoder_hidden_states
            if (
                self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            ):
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            if attention_mask is not None:
                encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]

            encoder_outputs["last_hidden_state"] = encoder_hidden_states

        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = None if encoder_outputs is None else encoder_outputs.last_hidden_state
        
        if prompt_hidden_states is not None and encoder_hidden_states is not None:
            encoder_hidden_states = torch.cat([prompt_hidden_states, encoder_hidden_states], dim=1)
        elif prompt_hidden_states is not None and encoder_hidden_states is None:
            encoder_hidden_states = prompt_hidden_states
            
        prompt_lengths, description_lengths = kwargs.get("prompt_lengths", None), kwargs.get("description_lengths", None)
        max_prompt_length, max_description_length = kwargs.get("max_prompt_length", 0), kwargs.get("max_description_length", 0)
        if prompt_attention_mask is not None and attention_mask is not None:
            prompt_lengths, description_lengths = prompt_attention_mask.sum(dim=1), attention_mask.sum(dim=1)
            max_prompt_length = prompt_attention_mask.shape[1]
            max_description_length = attention_mask.shape[1]
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        elif prompt_attention_mask is not None and attention_mask is None:
            prompt_lengths, description_lengths = prompt_attention_mask.sum(dim=1), None
            max_prompt_length = prompt_attention_mask.shape[1]
            attention_mask = prompt_attention_mask

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            ).transpose(1, 2)

        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_encoder_outputs = self.audio_encoder(
                input_values=input_values,
                padding_mask=padding_mask,
                **kwargs_audio_encoder,
            )
            audio_codes = audio_encoder_outputs.audio_codes
            # TODO: probably need to adapt depending on the audio encoder model
            frames, bsz, codebooks, seq_len = audio_codes.shape
            if frames != 1:
                raise ValueError(
                    f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                    "disabled by setting `chunk_length=None` in the audio encoder."
                )

            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)


        if self.config._attn_implementation == "flex_attention" and block_mask is None:      
            # suppose that padding side is left for both tokenizers  
            if prompt_lengths is None and description_lengths is not None:
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_description = torch.logical_and(kv_idx>=max_prompt_length,kv_idx < max_prompt_length+max_description_length-description_lengths[b])
                    return torch.logical_not(not_attend_description)
            elif prompt_lengths is not None and description_lengths is None:
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_prompt = kv_idx < max_prompt_length - prompt_lengths[b]
                    return torch.logical_not(not_attend_prompt)
            else:                 
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_prompt = kv_idx < max_prompt_length - prompt_lengths[b]
                    not_attend_description = torch.logical_and(kv_idx>=max_prompt_length,kv_idx < max_prompt_length+max_description_length-description_lengths[b])
                    
                    return torch.logical_not(torch.logical_or(not_attend_description, not_attend_prompt))

            mask_mod = and_masks(sparse_mask, mod_causal_mask) if description_lengths is not None or prompt_lengths is not None else mod_causal_mask
            length = max_prompt_length + max_description_length + decoder_input_ids.shape[-1]
            block_mask = create_block_mask(mask_mod, B=encoder_hidden_states.shape[0], H=None, Q_LEN=length, KV_LEN=length, device=decoder_input_ids.device, _compile=False)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            cache_position=cache_position,
            loss_reduction=loss_reduction,
            block_mask=block_mask,
        )

        if not return_dict:
            return decoder_outputs + (encoder_hidden_states,)

        return PrefixLMParlerTTSSeq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=None if encoder_outputs is None else encoder_outputs.last_hidden_state,
            encoder_hidden_states=None if encoder_outputs is None else encoder_outputs.hidden_states,
            encoder_attentions=None if encoder_outputs is None else encoder_outputs.attentions,
            per_codebook_losses=decoder_outputs.per_codebook_losses,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_delay_pattern_mask=None,
        cache_position=None,
        block_mask=None,
        **kwargs,
    ):
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                bos_token_id=self.generation_config.bos_token_id,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # apply the delay pattern mask
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                if past_key_values.get_seq_length() > 0:
                    # we only want to use prompt signal in the 1st generation step
                    encoder_outputs = None
            else:
                past_length = past_key_values[0][0].shape[2]
                # we only want to use prompt signal in the 1st generation step
                encoder_outputs = None

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cur_len = decoder_input_ids.shape[1]
            if encoder_outputs is not None:
                # meaning we are in 1st generation step and inputs_embeds will be prepended
                cur_len += encoder_outputs.last_hidden_state.shape[1]

            cache_position = cache_position[-cur_len:]

        adapted_block_mask = None
        if block_mask is not None and len(cache_position) == 1:
            block_index = cache_position // block_mask.BLOCK_SIZE[0]
            adapted_block_mask = block_mask[:, :, block_index]
            def get_mask_mod(mask_mod, offset: int):
                def _mask_mod(b, h, q, kv):
                    return mask_mod(b, h, q + offset, kv)

                return _mask_mod
            adapted_block_mask.mask_mod = get_mask_mod(block_mask.mask_mod, cache_position[0])
        
        if decoder_attention_mask is None and attention_mask is not None:
            input = decoder_input_ids.reshape(-1, self.decoder.num_codebooks, decoder_input_ids.shape[-1])
            bsz, _, seq_len = input.shape
            input_shape = (bsz, seq_len)

            past_key_values_length = 0
            if cache_position is not None:
                past_key_values_length = cache_position[0]
            elif past_key_values is not None:
                past_key_values_length = past_key_values.get_seq_length()


            if past_key_values is None or (
                isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0
            ):
                decoder_attention_mask = torch.ones(input_shape, device=self.device, dtype=decoder_input_ids.dtype)
            elif attention_mask is not None:
                len_ = attention_mask.shape[1]
                dtype = attention_mask.dtype
                # In the generation case of `prompt_cross_attention=True`, we need to recreate an attention mask from scratch
                # to be able to prepend the prompt attention mask.
                # Since we generate token per token, we can recompute the generated length from the information we have.
                generated_length = past_key_values_length - len_ + 1
                decoder_attention_mask = torch.ones(
                    (input_shape[0], generated_length), device=self.device, dtype=dtype
                )

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids.contiguous(),
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "block_mask": adapted_block_mask,
            "prompt_lengths":kwargs.get("prompt_lengths"),
            "description_lengths":kwargs.get("description_lengths"),
            "max_prompt_length":kwargs.get("max_prompt_length"),
            "max_description_length":kwargs.get("max_description_length"),
        }

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = (
            torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        num_codebooks = self.decoder.num_codebooks
        input = decoder_input_ids.reshape(-1, num_codebooks, decoder_input_ids.shape[-1])
        inputs_embeds = sum(
            [
                self.decoder.model.decoder.embed_tokens[codebook](input[:, codebook])
                for codebook in range(num_codebooks)
            ]
        )
        
        encoder_hidden_states = model_kwargs.get("encoder_outputs", None)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.last_hidden_state
        prompt_hidden_states = model_kwargs.get("prompt_hidden_states", None)

        if prompt_hidden_states is not None and encoder_hidden_states is not None:
            encoder_hidden_states = torch.cat([prompt_hidden_states, encoder_hidden_states], dim=1)
        elif prompt_hidden_states is not None and encoder_hidden_states is None:
            encoder_hidden_states = prompt_hidden_states
        
        if encoder_hidden_states is not None:
            inputs_embeds = torch.cat([encoder_hidden_states, inputs_embeds], dim=1)

        # get totally rid of prompt related stuff sice it's already in encoder_hidden_states now
        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        model_kwargs["prompt_hidden_states"] = None
        
        prompt_attention_mask = model_kwargs.get("prompt_attention_mask", None)
        attention_mask = model_kwargs.get("attention_mask", None)

        prompt_lengths = None
        if prompt_attention_mask is not None:
            prompt_lengths  = prompt_attention_mask.sum(dim=1)
            model_kwargs["prompt_lengths"] = prompt_lengths
            model_kwargs["max_prompt_length"] = prompt_attention_mask.shape[1]
            
        description_lengths = None
        if attention_mask is not None:
            description_lengths = attention_mask.sum(dim=1)
            model_kwargs["description_lengths"] = description_lengths
            model_kwargs["max_description_length"] = attention_mask.shape[1]

        if prompt_attention_mask is not None and attention_mask is not None:
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        elif prompt_attention_mask is not None and attention_mask is None:
            attention_mask = prompt_attention_mask

        model_kwargs["prompt_attention_mask"] = None
        model_kwargs["attention_mask"] = attention_mask

        return decoder_input_ids, model_kwargs

    def _prepare_text_encoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get text encoder
        encoder = self.get_text_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "prompt_", "use_cache", "labels"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state

        # we optionnally project last_hidden_state to avoid recomputing every time
        encoder_hidden_states = last_hidden_state
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if model_kwargs.get("attention_mask", None) is not None:
            encoder_hidden_states = encoder_hidden_states * model_kwargs["attention_mask"][..., None]

        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        return model_kwargs

    def _prepare_prompt_kwargs_for_generation(self, prompt_input_ids, model_kwargs):
        prompt_hidden_states = self.embed_prompts(prompt_input_ids)

        model_kwargs["prompt_hidden_states"] = prompt_hidden_states
        # we're keeping the prompt attention mask because it has to be prepended to the decoder attention mask on the fly
        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get audio encoder
        encoder = self.get_audio_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        
        if "num_quantizers" in encoder_signature:
            encoder_kwargs["num_quantizers"] = self.config.decoder_config.num_codebooks
        elif "num_codebooks" in encoder_signature:
            encoder_kwargs["num_codebooks"] = self.config.decoder_config.num_codebooks
        elif "n_quantizers" in encoder_signature:
            encoder_kwargs["n_quantizers"] = self.config.decoder_config.num_codebooks

        encoder_kwargs[model_input_name] = input_values
        audio_encoder_outputs = encoder.encode(**encoder_kwargs)
        audio_codes = audio_encoder_outputs.audio_codes
        audio_scales = audio_encoder_outputs.get("audio_scales")

        if audio_codes.ndim == 3:
            bsz, codebooks, seq_len = audio_codes.shape
            decoder_input_ids = audio_codes.reshape(bsz * self.decoder.num_codebooks, seq_len)
        else:
            frames, bsz, codebooks, seq_len = audio_codes.shape

            if frames != 1:
                raise ValueError(
                    f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                    "disabled by setting `chunk_length=None` in the audio encoder."
                )

            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        model_kwargs["decoder_input_ids"] = decoder_input_ids
        if audio_scales is not None:
            model_kwargs["audio_scales"] = audio_scales

        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.decoder_config.pad_token_id, self.config.decoder_config.bos_token_id
        ).transpose(1, 2)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: Union[int, List[int]] = None, bos_token_id: int = None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def _get_cache(self, cache_implementation: str, max_batch_size: int, max_cache_len: int, model_kwargs) -> Cache:
        """
        Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache.

        Returns the resulting cache object.
        """
        cache_cls: Cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_implementation]

        if hasattr(self, "_cache"):
            cache_to_check = self._cache

        if cache_implementation == "sliding_window":
            max_cache_len = min(self.config.sliding_window, max_cache_len)

        need_new_cache = (
            not hasattr(self, "_cache")
            or (not isinstance(cache_to_check, cache_cls))
            or cache_to_check.max_batch_size != max_batch_size
            or cache_to_check.max_cache_len < max_cache_len
        )


        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.dtype
            cache_kwargs = {
                "config": self.config.decoder_config,
                "max_batch_size": max_batch_size,
                "max_cache_len": max_cache_len,
                "device": self.device,
                "dtype": cache_dtype,
            }
            self._cache = cache_cls(**cache_kwargs)
        else:
            self._cache.reset()
        return self._cache

    def freeze_encoders(self, freeze_text_encoder=True):
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder._requires_grad = False

        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder._requires_grad = False

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) == tuple:
            # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        requires_attention_mask = False # "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if "encoder_outputs" not in model_kwargs:
            # encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if "prompt_hidden_states" not in model_kwargs and "prompt_input_ids" in model_kwargs:
            # `prompt_hidden_states` are created and added to `model_kwargs`
            model_kwargs = self._prepare_prompt_kwargs_for_generation(
                model_kwargs["prompt_input_ids"],
                model_kwargs,
            )

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            device=inputs_tensor.device,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                # when we prepend prompt_hidden_state to inputs_embeds, max_cache_len needs to be actualised
                input_embeds_seq_length = model_kwargs["encoder_outputs"].last_hidden_state.shape[1]
                max_cache_len = generation_config.max_length + input_embeds_seq_length

                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    max_cache_len,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                raise ValueError(
                    "This model does not support the quantized cache. If you want your model to support quantized "
                    "cache, please open an issue on the Parler-TTS repository https://github.com/huggingface/parler-tts"
                )
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get("past_key_values", None)
            if past is None:
                model_kwargs["past_key_values"] = DynamicCache()
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = DynamicCache.from_legacy_cache(past)

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Parler-TTS)
        delayed_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config._bos_token_tensor,
            pad_token_id=generation_config._pad_token_tensor,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        if self.config._attn_implementation == "flex_attention":
            # suppose that padding side is left for both tokenizers  
            prompt_lengths, description_lengths = model_kwargs.get("prompt_lengths"), model_kwargs.get("description_lengths")
            max_prompt_length, max_description_length = model_kwargs.get("max_prompt_length"), model_kwargs.get("max_description_length")
            sparse_mask = None
            if prompt_lengths is None and description_lengths is not None:
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_description = torch.logical_and(kv_idx>=max_prompt_length,kv_idx < max_prompt_length+max_description_length-description_lengths[b])
                    return torch.logical_not(not_attend_description)
            elif prompt_lengths is not None and description_lengths is None:
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_prompt = kv_idx < max_prompt_length - prompt_lengths[b]
                    return torch.logical_not(not_attend_prompt)
            else:                 
                def sparse_mask(b, h, q_idx, kv_idx):
                    not_attend_prompt = kv_idx < max_prompt_length - prompt_lengths[b]
                    not_attend_description = torch.logical_and(kv_idx>=max_prompt_length,kv_idx < max_prompt_length+max_description_length-description_lengths[b])
                    
                    return torch.logical_not(torch.logical_or(not_attend_description, not_attend_prompt))

            mask_mod = and_masks(sparse_mask, mod_causal_mask) if description_lengths is not None or prompt_lengths is not None else mod_causal_mask
            
            max_length = generation_config.max_length+max_prompt_length+max_description_length
            block_mask = create_block_mask(mask_mod, B=batch_size, H=None, Q_LEN=max_length, KV_LEN=max_length, device=delayed_input_ids.device, _compile=False)
            model_kwargs["block_mask"] = block_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(delayed_input_ids.cpu())

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=delayed_input_ids.device,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # expand input_ids with `num_return_sequences` additional sequences per batch
            delayed_input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=delayed_input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 10. run sample
            outputs = self._sample(
                delayed_input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # Apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # Revert the pattern delay mask by filtering the eos and bos token ids from the delay pattern mask
        _, mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=output_ids.shape[1],
        )

        mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
        output_ids = output_ids[mask].reshape(batch_size, self.decoder.num_codebooks, -1)

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_decode_kwargs = {}
        if self.use_audio_scales:
            audio_scales = model_kwargs.get("audio_scales")
            if audio_scales is None:
                audio_scales = [None] * batch_size
            audio_decode_kwargs["audio_scales"] = audio_scales

        
        if not self.use_4dim_audio_codes:
            # remove chunk dim
            output_ids = output_ids.squeeze(0)
            
            
        decode_sequentially = (output_ids >= self.audio_encoder.config.codebook_size).any()
        if not decode_sequentially:
            output_values = self.audio_encoder.decode(
                audio_codes=output_ids,
                **audio_decode_kwargs,
            ).audio_values.squeeze(1)
            output_lengths = [audio.shape[0] for audio in output_values]
        else:
            output_values = []
            for sample_id in range(batch_size):
                sample = output_ids[:, sample_id] if self.use_4dim_audio_codes else output_ids[sample_id]
                sample_mask = (sample >= self.audio_encoder.config.codebook_size)
                sample_mask = (sample_mask.sum(dim=(0, 1)) == 0) if self.use_4dim_audio_codes else (sample_mask.sum(dim=0) == 0)
                single_audio_decode_kwargs = {}
                if self.use_audio_scales:
                    single_audio_decode_kwargs["audio_scales"] = [audio_decode_kwargs["audio_scales"][sample_id]]
                if sample_mask.sum() > 0:
                    sample = sample[:, :, sample_mask] if self.use_4dim_audio_codes else sample[:, sample_mask]
                    sample = self.audio_encoder.decode(audio_codes=sample[None, ...], **single_audio_decode_kwargs).audio_values
                    sample = sample if sample.ndim == 3 else sample.unsqueeze(0)
                    output_values.append(sample.transpose(0, 2))
                else:
                    output_values.append(torch.zeros((1, 1, 1)).to(self.device))
            output_lengths = [audio.shape[0] for audio in output_values]
            output_values = (
                torch.nn.utils.rnn.pad_sequence(output_values, batch_first=True, padding_value=0)
                .squeeze(-1)
                .squeeze(-1)
            )
        if generation_config.return_dict_in_generate:
            outputs["audios_length"] = output_lengths
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            initial_length = model_kwargs["inputs_embeds"].shape[1]
            device = model_kwargs["inputs_embeds"].device
        else:
            initial_length = input_ids.shape[1]
            device = input_ids.device

        if model_kwargs.get("encoder_outputs", None) is not None:
            initial_length += model_kwargs["encoder_outputs"].last_hidden_state.shape[1]

        cache_position = torch.ones((initial_length,), dtype=torch.int64, device=device).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    @classmethod
    def from_sub_models_pretrained(
        cls,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate a text encoder, an audio encoder, and a Parler-TTS decoder from one, two or three base classes of the
        library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `t5-base`, or namespaced under a user or
                      organization name, like `google/flan-t5-base.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `facebook/encodec_24khz`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `gpt2`, or namespaced under a user or
                      organization name, like `parler-tts/parler-tts-mini-v1`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                  parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                  parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from parler_tts import ParlerTTSForConditionalGeneration

        >>> # initialize a parler_tts model from a t5 text encoder, encodec audio encoder, and parler_tts decoder
        >>> model = ParlerTTSForConditionalGeneration.from_sub_models_pretrained(
        ...     text_encoder_pretrained_model_name_or_path="t5-base",
        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
        ...     decoder_pretrained_model_name_or_path="parler-tts/parler-tts-mini-v1",
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./parler_tts-ft")
        >>> # load fine-tuned model
        >>> model = ParlerTTSForConditionalGeneration.from_pretrained("./parler_tts-ft")
        ```"""

        kwargs_text_encoder = {
            argument[len("text_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_encoder_")
        }

        kwargs_audio_encoder = {
            argument[len("audio_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("audio_encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove text encoder, audio encoder and decoder kwargs from kwargs
        for key in kwargs_text_encoder.keys():
            del kwargs["text_encoder_" + key]
        for key in kwargs_audio_encoder.keys():
            del kwargs["audio_encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        text_encoder = kwargs_text_encoder.pop("model", None)
        if text_encoder is None:
            if text_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_text_encoder:
                encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(
                    text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_text_encoder["config"] = encoder_config

            text_encoder = AutoModelForTextEncoding.from_pretrained(
                text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder
            )

        audio_encoder = kwargs_audio_encoder.pop("model", None)
        if audio_encoder is None:
            if audio_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_audio_encoder:
                encoder_config, kwargs_audio_encoder = AutoConfig.from_pretrained(
                    audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_audio_encoder["config"] = encoder_config

            audio_encoder = AutoModel.from_pretrained(
                audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = PrefixLMParlerTTSDecoderConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if isinstance(decoder_config, PrefixLMParlerTTSConfig):
                    decoder_config = decoder_config.decoder

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_sub_models_pretrained(...)`"
                )

            decoder = PrefixLMParlerTTSForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = PrefixLMParlerTTSConfig.from_sub_models_config(
            text_encoder.config, audio_encoder.config, decoder.config, **kwargs
        )
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)
