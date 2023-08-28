"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import torch.distributed as dist
import torch.multiprocessing as mp


MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.k_cache = torch.nn.Parameter(torch.zeros(cache_shape, device=device, dtype=dtype))
        self.v_cache = torch.nn.Parameter(torch.zeros(cache_shape, device=device, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val

        return self.k_cache, self.v_cache

class KVCacheAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_caches = nn.ModuleList([])

    def initialize(self,layers, max_batch_size, max_seq_length, n_heads, head_size, device='cuda', dtype=torch.bfloat16):
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.kv_caches = nn.ModuleList([KVCache(max_batch_size, max_seq_length, n_heads, head_size) for _ in range(layers)])

    def __getitem__(self, idx):
        return self.kv_caches[idx]

    def clear(self):
        self.kv_caches = nn.ParameterList([])

class LLaMA(nn.Module):
    """
    LLaMA Language Model
    └── Input (Token IDs)
        │
        ├── Positional Embeddings (RoPE)
        │
        ├── Transformer Blocks (x32, from llama_config n_layer)
        │     │
        │     ├── Causal Self-Attention
        │     │     │
        │     │     ├── Key, Query, Value Projections
        │     │     │
        │     │     ├── Rotary Positional Embeddings
        │     │     │
        │     │     └── Attention Calculation
        │     │
        │     ├── Feedforward Neural Network (MLP)
        │     │
        │     └── Residual Connection and Layer Normalization
        │
        ├── Linear Projection to Vocabulary Size
        │
        └── Output (Logits)
    """
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches = KVCacheAggregator()
        self.max_batch_size = None
        self.max_seq_length = None

    def setup_caches(self, max_batch_size, max_seq_length, device='cuda', dtype=torch.bfloat16):
        head_size = self.config.n_embd // self.config.n_head

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.kv_caches.initialize(layers=self.config.n_layer, max_batch_size=max_batch_size, max_seq_length=max_seq_length, n_heads=self.config.n_head, head_size=head_size)

        self.rope_cache = build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=dtype,
            device=device,
        )
        ones = torch.ones((self.config.block_size, self.config.block_size), device=device, dtype=torch.bool)
        self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        max_seq_length = self.max_seq_length
        if max_seq_length is None:
            max_seq_length = block_size

        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        rope = self.rope_cache.index_select(0, input_pos)
        mask = self.mask_cache.index_select(2, input_pos)
        mask = mask[:, :, :, :max_seq_length]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # Transformer
        for i, block in enumerate(self.transformer.h):
            x, new_kv_cache = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        # (Residual Connection?) and Layer Normalization
        x = self.transformer.ln_f(x)

        # Linear Projection to Vocabulary Size
        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))

    def reset_cache(self) -> None:
        self.kv_caches.clear()


class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP_TP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # Self-attention
        h, new_kv_cache = self.attn(self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + h
        # MLP
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # TODO: distribute this
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            k, v = kv_cache.update(input_pos, k, v)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # y = F.scaled_dot_product_attention(q, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        # TODO: distribute this
        y = self.c_proj(y)

        return y, kv_cache


def mlp_forward_tp(rank, x, results_queue):
    n_gpus = 8
    dist.init_process_group(backend="nccl", rank=rank, world_size=n_gpus)

    x = x.cuda(rank) # (1, 6, 4096)

    # for llama-7b
    # keep feature dim 4096
    # split hidden dim 11008 8-way: 1376
    c_fc1 = nn.Linear(4096, 1376, bias=False, device=rank, dtype=x.dtype)
    c_fc2 = nn.Linear(4096, 1376, bias=False, device=rank, dtype=x.dtype)

    kq = c_fc1(x) # (1, 6, 1376)
    v = c_fc2(x) # (1, 6, 1376)
    x = F.silu(kq) * v # (1, 6, 1376)

    outputs = [torch.zeros(x.shape, device=rank, dtype=x.dtype) for _ in range(n_gpus)]
    dist.all_gather(outputs, x)
    output = torch.cat(outputs, dim=2) # (1, 6, 11008)

    torch.cuda.synchronize()

    if rank == 0:
        print(f"rank={rank}, done compute")
        # compute on all ranks for now
        c_proj = nn.Linear(11008, 4096, bias=False, device=rank, dtype=x.dtype)
        print(f"rank={rank}, applying classifier")
        output = c_proj(output) # (1, 6, 4096)
        print(f"rank={rank}, inserting into results_queue")
        torch.cuda.synchronize()
        results_queue.put(1)
        print(f"rank={rank}, inserted into results_queue")


class MLP_TP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        # hidden_dim = 4 * config.n_embd
        # n_hidden = int(2 * hidden_dim / 3)
        # n_hidden = find_multiple(n_hidden, 256)

        # # 4096 -> 11008
        # self.c_fc1 = Linear_TP(config.n_embd, n_hidden, bias=False)
        # self.c_fc2 = Linear_TP(config.n_embd, n_hidden, bias=False)
        # # 11008 -> 4096
        # self.c_proj = Linear_TP(n_hidden, config.n_embd, bias=False)
        self.setup_tp()

    def setup_tp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape is (1, 6, 4096) (b, seq, feature)

        results_queue = mp.Queue()
        mp.spawn(mlp_forward_tp, args=(x, results_queue,), nprocs=8, join=True)

        if results_queue.empty():
            raise RuntimeError("No results from MLP_TP")

        print("retrieving results queue item")
        output = results_queue.get()
        results_queue.task_done()
        # kq = self.c_fc1(x)
        # v = self.c_fc2(x)
        # x = F.silu(kq) * v
        # x = self.c_proj(x)

        # output shape is (1, 6, 4096) (b, seq, feature)
        print(f"main process received output.shape={output.shape}")
        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
