"""
VocabRegisterLM — PyTorch/CUDA version for GPU training.
Each register IS a word. No embedding, no output projection.
Compatible with torchrun for multi-GPU training.
"""
from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model: dim = vocab_size (registers ARE words)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_recurrent_steps = int(os.environ.get("NUM_RECURRENT_STEPS", 24))
    n_fourier_basis = int(os.environ.get("N_FOURIER_BASIS", 16))
    n_channels = int(os.environ.get("N_CHANNELS", 8))
    activation = os.environ.get("ACTIVATION", "gelu")
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# Patterns for scalar/control tensors (Adam, not Muon)
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scales", "op_scales", "q_gain", "read_coeffs", "write_coeffs",
    "mix_weight", "bias", "out_scale", "logit_scale",
)

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# -----------------------------
# VALIDATION + QUANTIZATION (from baseline)
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    ts = max(sv, vocab_size)
    bb = np.zeros((ts,), dtype=np.int16)
    hs = np.zeros((ts,), dtype=np.bool_)
    ib = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool, device=device),
            torch.tensor(ib, dtype=torch.bool, device=device))


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad header: {file}")
    n = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=n, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(left, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = chunk[start:start + span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl):
    lbt = args.val_batch_size // (world_size * grad_accum_steps)
    if lbt < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    lbs = lbt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    ss, se = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss, se, lbs):
            be = min(bs + lbs, se)
            local = val_tokens[bs * args.train_seq_len:be * args.train_seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            ct = float(y.numel())
            loss_sum += bl.to(torch.float64) * ct
            tok_count += ct
            tb = bbl[y.reshape(-1)].to(torch.float64)
            tb += (hsl[y.reshape(-1)] & ~ibl[x.reshape(-1)]).to(torch.float64)
            byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_count).item()
    model.train()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())


INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536

def quantize_state_dict_int8(sd):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        stats["num_float_tensors"] += 1
        t32 = t.float()
        if t32.ndim == 2:
            ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            clipped = torch.clamp(t32, -ca[:, None], ca[:, None])
            s = (ca / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127).to(torch.int8)
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            scales[name] = s.to(torch.float16).contiguous()
        else:
            ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
            s = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / s), -127, 127).to(torch.int8)
            scales[name] = s
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + (scales[name].numel() * scales[name].element_size())
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dt = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(torch.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dt)
        else:
            out[name] = (q.float() * s.item()).to(dt)
    for name, t in obj["passthrough"].items():
        od = pod.get(name)
        out[name] = t.to(getattr(torch, od)) if isinstance(od, str) else t.detach().cpu()
    return out


# -----------------------------
# MODEL
# -----------------------------

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._cache = (0, None, None)

    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len or self._cache[1] is None or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


def apply_activation(x, activation):
    if activation == "gelu":
        return F.gelu(x)
    elif activation == "relu2":
        return F.relu(x).square()
    elif activation == "swish":
        return F.silu(x)
    return F.gelu(x)


class FourierRegisterOp(nn.Module):
    """Vocabulary-space register operation. Each dim IS a word."""
    def __init__(self, n_basis, n_channels, activation="gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.mix_weight = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, basis):
        # basis: (vocab_size, 2*n_basis) — Fourier over vocabulary indices
        read_p = torch.softmax(basis @ self.read_coeffs.T, dim=0).to(x.dtype)  # (V, C)
        values = x @ read_p  # (B, T, C)
        values = values @ self.mix_weight.to(x.dtype) + self.bias.to(x.dtype)
        values = apply_activation(values, self.activation)
        write_p = (basis @ self.write_coeffs.T).to(x.dtype)  # (V, C)
        return values @ write_p.T * self.out_scale.to(x.dtype)


class VocabRegisterLM(nn.Module):
    """Each register IS a word. No embedding. No output projection."""
    def __init__(self, vocab_size, num_heads, num_kv_heads, num_steps, n_fourier_basis,
                 n_channels, logit_softcap, rope_base, qk_gain_init, activation="gelu"):
        super().__init__()
        dim = vocab_size
        self.dim = dim
        self.vocab_size = vocab_size
        self.logit_softcap = logit_softcap
        self.num_steps = num_steps
        self.activation = activation

        self.attn_norm = nn.Identity()  # rms_norm applied inline
        self.shared_attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.register_ops = nn.ModuleList([FourierRegisterOp(n_fourier_basis, n_channels, activation) for _ in range(num_steps)])
        self.op_norms = nn.ModuleList([nn.Identity() for _ in range(num_steps)])

        self.attn_scales = nn.Parameter(torch.ones(num_steps, dim))
        self.op_scales = nn.Parameter(torch.ones(num_steps, dim))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Fourier basis over vocabulary (frozen)
        positions = torch.arange(dim, dtype=torch.float32) / dim
        basis = torch.zeros(dim, 2 * n_fourier_basis)
        for k in range(n_fourier_basis):
            basis[:, 2 * k] = torch.cos(2 * math.pi * (k + 1) * positions)
            basis[:, 2 * k + 1] = torch.sin(2 * math.pi * (k + 1) * positions)
        self.register_buffer("fourier_basis", basis, persistent=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids, target_ids):
        # One-hot: R["cat"] = 1.0
        x = F.one_hot(input_ids, self.vocab_size).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (x.size(-1),))
        basis = self.fourier_basis

        for step in range(self.num_steps):
            attn_out = self.shared_attn(F.rms_norm(x, (x.size(-1),)))
            x = x + self.attn_scales[step].to(x.dtype) * attn_out
            op_out = self.register_ops[step](F.rms_norm(x, (x.size(-1),)), basis)
            x = x + self.op_scales[step].to(x.dtype) * op_out

        x = F.rms_norm(x, (x.size(-1),))
        # Register state IS logits — no projection needed
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float().reshape(-1, self.vocab_size), target_ids.reshape(-1), reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    log0(code, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bbl, hsl, ibl = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = VocabRegisterLM(
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
    ).to(device).bfloat16()

    # Keep linear weights and small params in fp32
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
        if isinstance(m, Rotary):
            m.inv_freq.data = m.inv_freq.data.float()
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer: Muon for attention matrices, Adam for everything else
    attn_matrix_params = [p for n, p in base_model.shared_attn.named_parameters() if p.ndim == 2 and "weight" in n]
    scalar_params = [p for n, p in base_model.named_parameters()
                     if not (n.startswith("shared_attn.") and p.ndim == 2 and "weight" in n)]

    optimizer_muon = Muon(attn_matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log0(f"run_id:{args.run_id}")
    log0(f"architecture:VocabRegisterLM (registers ARE words, PyTorch/CUDA)")
    log0(f"model_params:{n_params} trainable:{n_trainable} vocab=dim={args.vocab_size}")
    log0(f"heads:{args.num_heads} kv_heads:{args.num_kv_heads} steps:{args.num_recurrent_steps}")
    log0(f"fourier:{args.n_fourier_basis} channels:{args.n_channels} activation:{args.activation}")
    log0(f"NO embedding. NO output projection. Registers = vocabulary.")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} batch:{args.train_batch_tokens}")
    log0(f"muon_params:{len(attn_matrix_params)} scalar_params:{len(scalar_params)}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wc_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    # Warmup
    for ws in range(args.warmup_steps):
        zero_grad_all()
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
        for o in optimizers:
            o.step()
        torch.cuda.synchronize()
        if master and (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
            log0(f"warmup:{ws + 1}/{args.warmup_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training
    train_ms = 0.0
    stop_after = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last = step == args.iterations or (stop_after is not None and step >= stop_after)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} train_time:{train_ms:.0f}ms")
            t0 = time.perf_counter()
        if last:
            if stop_after is not None and step < args.iterations:
                log0(f"stopping_early: step:{step}/{args.iterations}")
            break

        lm = lr_mul(step, train_ms + 1000.0 * (time.perf_counter() - t0))
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * lm

        step_t0 = time.perf_counter()
        zero_grad_all()
        train_loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            train_loss_accum += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        torch.cuda.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if master and args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss_accum:.4f} time:{approx_ms:.0f}ms avg:{approx_ms / step:.1f}ms tok/s:{args.train_batch_tokens / (step_ms / 1000):.0f}")
        if max_wc_ms and stop_after is None and approx_ms >= max_wc_ms:
            stop_after = step

    # Serialize
    if master:
        sd = {k: v for k, v in base_model.state_dict().items()}
        out = f"logs/{args.run_id}_model.pt"
        torch.save(sd, out)
        log0(f"saved:{out} bytes:{Path(out).stat().st_size}")

        qobj, qstats = quantize_state_dict_int8(sd)
        buf = io.BytesIO()
        torch.save(qobj, buf)
        compressed = zlib.compress(buf.getvalue(), 9)
        qpath = f"logs/{args.run_id}_model.int8.ptz"
        Path(qpath).write_bytes(compressed)
        log0(f"quantized:{qpath} bytes:{len(compressed)} ratio:{qstats['baseline_tensor_bytes'] / max(qstats['int8_payload_bytes'], 1):.2f}x")

        # Roundtrip eval
        dq = dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(Path(qpath).read_bytes())), weights_only=False))
        base_model.load_state_dict(dq, strict=False)
        qvl, qvbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl)
        log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvbpb:.4f}")


if __name__ == "__main__":
    main()
