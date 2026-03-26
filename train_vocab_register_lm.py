#!/usr/bin/env python3
"""
VocabRegisterLM: Each register IS a word.

- Input: one-hot over vocabulary → R["cat"] = 1.0, everything else 0.0
- State: always a distribution over words — transparent by definition
- Computation: shared attention + Fourier register ops in vocabulary space
- Output: register state IS the prediction logits — no output projection
- Every intermediate step readable as "which words are active and how strongly"

Interpretability by construction.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model: dim = vocab_size (registers ARE words)
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    num_recurrent_steps: int = int(os.environ.get("NUM_RECURRENT_STEPS", 24))
    n_fourier_basis: int = int(os.environ.get("N_FOURIER_BASIS", 16))
    n_channels: int = int(os.environ.get("N_CHANNELS", 8))
    activation: str = os.environ.get("ACTIVATION", "gelu")
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def model_dim(self) -> int:
        return self.vocab_size  # registers ARE words

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scales", "op_scales", "q_gain", "read_coeffs", "write_coeffs",
    "mix", "bias", "out_scale", "fourier_basis", "logit_scale",
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def make_fourier_basis(dim: int, n_basis: int) -> mx.array:
    positions = np.arange(dim, dtype=np.float32) / dim
    basis = np.zeros((dim, 2 * n_basis), dtype=np.float32)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = np.cos(2 * np.pi * freq * positions)
        basis[:, 2 * k + 1] = np.sin(2 * np.pi * freq * positions)
    return mx.array(basis)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


def apply_activation(x: mx.array, activation: str) -> mx.array:
    if activation == "gelu":
        return nn.gelu(x)
    elif activation == "relu2":
        r = nn.relu(x)
        return r * r
    elif activation == "swish":
        return nn.silu(x)
    return nn.gelu(x)


class FourierRegisterOp(nn.Module):
    """Register operation in vocabulary space. Each dimension IS a word.

    Fourier basis over vocabulary indices: low frequencies group words broadly,
    high frequencies distinguish specific words.
    """
    def __init__(self, n_basis: int, n_channels: int):
        super().__init__()
        scale = 0.02
        self.read_coeffs = mx.random.normal((n_channels, 2 * n_basis)) * scale
        self.write_coeffs = mx.random.normal((n_channels, 2 * n_basis)) * scale
        self.mix = mx.random.normal((n_channels, n_channels)) * scale
        self.bias = mx.zeros((n_channels,))
        self.out_scale = mx.array(0.01, dtype=mx.float32)

    def __call__(self, x: mx.array, basis: mx.array, activation: str = "gelu") -> mx.array:
        dtype = x.dtype
        read_patterns = basis @ self.read_coeffs.astype(mx.float32).T
        read_patterns = mx.softmax(read_patterns, axis=0).astype(dtype)
        values = x @ read_patterns
        values = values @ self.mix.astype(dtype) + self.bias.astype(dtype)
        values = apply_activation(values, activation)
        write_patterns = (basis @ self.write_coeffs.astype(mx.float32).T).astype(dtype)
        return values @ write_patterns.T * self.out_scale.astype(dtype)


# ==============================================================================
# VOCAB REGISTER LM
# ==============================================================================

class VocabRegisterLM(nn.Module):
    """Language model where registers ARE words.

    - Input: one-hot(token) → 1024-dim register state (no embedding matrix)
    - Computation: shared attention + Fourier ops in vocabulary space
    - Output: register state = logits (no output projection)
    - Every intermediate state readable as word activations
    """
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_steps: int,
        n_fourier_basis: int,
        n_channels: int,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        activation: str = "gelu",
    ):
        super().__init__()
        dim = vocab_size  # registers ARE words
        self.dim = dim
        self.vocab_size = vocab_size
        self.logit_softcap = logit_softcap
        self.num_steps = num_steps
        self.activation = activation

        # NO embedding matrix. Input is one-hot.
        # NO output projection. Register state IS the logits.

        # Shared causal self-attention in vocabulary space
        self.attn_norm = RMSNormNoWeight()
        self.shared_attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.shared_attn.proj.weight = mx.zeros_like(self.shared_attn.proj.weight)

        # Per-step Fourier register ops (vocabulary-space operations)
        self.register_ops = [FourierRegisterOp(n_fourier_basis, n_channels) for _ in range(num_steps)]
        self.op_norms = [RMSNormNoWeight() for _ in range(num_steps)]

        # Per-step residual scales
        self.attn_scales = mx.ones((num_steps, dim), dtype=mx.float32)
        self.op_scales = mx.ones((num_steps, dim), dtype=mx.float32)

        self.final_norm = RMSNormNoWeight()

        # Learned output scale (register magnitudes → logit range)
        self.logit_scale = mx.array(1.0, dtype=mx.float32)

        # Fourier basis over vocabulary indices (frozen)
        self.fourier_basis = make_fourier_basis(dim, n_fourier_basis)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def one_hot_input(self, input_ids: mx.array) -> mx.array:
        """Token → one-hot register state. R["cat"] = 1.0, all else 0.0."""
        return mx.one_hot(input_ids, self.vocab_size).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids: mx.array) -> mx.array:
        # Input: one-hot over vocabulary — the register bank starts with one word active
        x = self.one_hot_input(input_ids)  # (B, T, vocab_size)
        x = rms_norm(x)
        basis = self.fourier_basis.astype(mx.float32)

        for step in range(self.num_steps):
            # Cross-position communication in vocabulary space
            attn_out = self.shared_attn(self.attn_norm(x))
            x = x + self.attn_scales[step].astype(x.dtype)[None, None, :] * attn_out

            # Within-position register manipulation
            op_out = self.register_ops[step](self.op_norms[step](x), basis, self.activation)
            x = x + self.op_scales[step].astype(x.dtype)[None, None, :] * op_out

        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Register state IS the logits — no projection needed
        x = self(input_ids)  # (B, T, vocab_size)
        logits = x * self.logit_scale.astype(x.dtype)
        logits = self.softcap(logits)
        logits = logits.reshape(-1, self.vocab_size)
        y = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# OPTIMIZERS
# ==============================================================================

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    """No embedding LR group needed — no embedding matrix exists."""
    def __init__(self, model: VocabRegisterLM, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.trainable_parameters()))

        # Muon: shared attention weight matrices (large 2D)
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("shared_attn.") and k.endswith(".weight") and p.ndim == 2
        ]

        # Adam: everything else (register op params, scales, logit_scale, q_gain)
        self.scalar_keys = [
            k for k in params.keys()
            if k not in self.matrix_keys
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: VocabRegisterLM, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.trainable_parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in params}
        updated.update(self.adam.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# QUANTIZATION (INT8 + ZLIB) — identical to baseline
# ==============================================================================

MX_DTYPE_FROM_NAME = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr):
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj):
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[quant_obj["dtypes"][name]])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig = passthrough_orig_dtypes.get(name)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig]) if isinstance(orig, str) else mx.array(out_arr)
    return out


# ==============================================================================
# VALIDATION
# ==============================================================================

def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token_lut[tid] = False
        if sp.is_byte(tid):
            base_bytes_lut[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_lut[tid] = True
            piece = piece[1:]
        base_bytes_lut[tid] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path, tokenizer_path):
    dataset_dir = Path(data_path).resolve()
    actual = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if entry is None:
        return dataset_dir.name, actual, None
    tn = entry.get("tokenizer_name")
    te = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tn), None) if tn else None
    en = Path((te or {}).get("model_path") or (te or {}).get("path") or "").name
    if en and Path(tokenizer_path).name != en:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {en}, got {Path(tokenizer_path).name}")
    expected = (entry.get("stats") or {}).get("files_train")
    if expected is not None:
        expected = int(expected)
        if actual > expected:
            raise ValueError(f"Too many train shards: {actual} > {expected}")
    return dataset_dir.name, actual, expected


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files for {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=None):
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = total_tokens = total_bytes = 0.0
    for batch_idx, start in enumerate(range(0, total_seqs, val_batch_seqs), 1):
        end = min(start + val_batch_seqs, total_seqs)
        chunk = val_tokens[start * args.train_seq_len : end * args.train_seq_len + 1]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x, y = mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        ct = float(y.size)
        total_loss_sum += float(batch_loss.item()) * ct
        prev_ids, tgt_ids = x_np.reshape(-1), y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16)
        total_tokens += ct
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    return val_loss, val_loss / math.log(2.0) * (total_tokens / total_bytes)


def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(float(np.sum(np.square(_np_float32(g)), dtype=np.float64)) for g in flat.values())
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    s = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * s) for k, g in flat.items()])


# ==============================================================================
# TRAINING
# ==============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg, console=True):
        if console: print(msg)
        with logfile.open("a", encoding="utf-8") as f: print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Need .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer={int(sp.vocab_size())}")

    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = VocabRegisterLM(
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_steps=args.num_recurrent_steps,
        n_fourier_basis=args.n_fourier_basis,
        n_channels=args.n_channels,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        activation=args.activation,
    )
    model.freeze(keys=["fourier_basis"])

    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    n_trainable = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.trainable_parameters()))

    log(f"run_id:{args.run_id}")
    log(f"architecture:VocabRegisterLM (registers ARE words)")
    log(f"model_params:{n_params} trainable:{n_trainable} vocab=dim={args.vocab_size} "
        f"heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"steps:{args.num_recurrent_steps} fourier:{args.n_fourier_basis} "
        f"channels:{args.n_channels} activation:{args.activation}")
    log(f"NO embedding matrix. NO output projection. Registers = vocabulary.")
    log(f"iterations:{args.iterations} batch:{args.train_batch_tokens} "
        f"grad_accum:{args.grad_accum_steps} seq_len:{args.train_seq_len}")
    log(f"optimizer: muon({len(opt.matrix_keys)} matrices) + adam({len(opt.scalar_keys)} scalars)")

    # Warmup
    if args.warmup_steps > 0:
        for ws in range(args.warmup_steps):
            accum = None
            wl = mx.array(0.0, dtype=mx.float32)
            gs = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                wl, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, gs)
            mx.eval(wl, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log(f"warmup_step:{ws + 1}/{args.warmup_steps}")

        vbt = args.val_batch_size // args.grad_accum_steps
        if vbt >= args.train_seq_len:
            wvs = min(vbt // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
            wc = val_tokens[: wvs * args.train_seq_len + 1]
            xv = mx.array(wc[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
            yv = mx.array(wc[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
            mx.eval(compiled_loss(xv, yv))
            mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            if step % 25 == 0 or last_step:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms")
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        gs = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, gs)
            train_loss = train_loss + loss.astype(mx.float32) * gs
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_ms >= max_wallclock_ms:
            stop_after_step = step

    # Serialize
    out_path = out_dir / f"{args.run_id}_vocab_register_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_path = out_dir / f"{args.run_id}_vocab_register_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    log(f"serialized_int8_zlib:{quant_path.stat().st_size} bytes (ratio:{quant_stats['baseline_tensor_bytes'] / max(quant_stats['int8_payload_bytes'], 1):.2f}x)")

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    model.update(tree_unflatten(list(dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk))).items())))
    qt0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - qt0):.0f}ms")


if __name__ == "__main__":
    main()
