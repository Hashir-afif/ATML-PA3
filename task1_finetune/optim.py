"""Optimizer and penalty utilities for Task 1.

``OffloadedAdamW``: AdamW whose FP32 moment buffers live in CPU RAM and are streamed to the GPU one tensor at
a time. Full FT of Qwen3-0.6B with FP32 master weights, FP32 gradients and FP32 AdamW states needs ~9.5 GB,
which does not fit on the 8 GB GPU used here. The update is the single-tensor ``torch.optim.AdamW`` rule
(decoupled weight decay, bias-corrected moments, eps added after the bias-corrected square root), so the
optimization is mathematically unchanged; only where the moments are stored differs. ``verify_against_torch``
checks equality with ``torch.optim.AdamW`` numerically.

``L2SPReference``: exact copy of the starting parameters theta_0 kept on the CPU (in BF16 when every value is
exactly representable, which holds for the BF16 Qwen3 release; FP32 otherwise) and streamed to the GPU to
evaluate Omega = 0.5 * sum_j (theta_j - theta0_j)^2 and to add lambda * (theta - theta0) to the gradients.
"""
from __future__ import annotations

import math
import time

import torch

CHUNK = 1 << 24  # elements per streamed chunk (64 MB in FP32)


class OffloadedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 pin_memory: bool = True):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.transfer_seconds = 0.0

    def _cpu_zeros_like(self, p: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
        if self.pin_memory:
            try:
                t = t.pin_memory()
            except RuntimeError:
                self.pin_memory = False
        return t

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = self._cpu_zeros_like(p)
                    state["exp_avg_sq"] = self._cpu_zeros_like(p)
                state["step"] += 1
                step = state["step"]
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                t0 = time.perf_counter()
                # Element-wise update, streamed in fixed-size chunks so the largest tensor (the 155M-element
                # tied embedding) never needs full-size moment copies and temporaries on the GPU.
                p_flat, g_flat = p.data.view(-1), p.grad.view(-1)
                m_cpu, v_cpu = state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)
                for s in range(0, p_flat.numel(), CHUNK):
                    e = min(s + CHUNK, p_flat.numel())
                    pc, g = p_flat[s:e], g_flat[s:e].float()
                    m = m_cpu[s:e].to(p.device, non_blocking=True)
                    v = v_cpu[s:e].to(p.device, non_blocking=True)
                    pc.mul_(1 - lr * wd)
                    m.lerp_(g, 1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    pc.addcdiv_(m, denom, value=-lr / bc1)
                    m_cpu[s:e].copy_(m, non_blocking=True)
                    v_cpu[s:e].copy_(v, non_blocking=True)
                self.transfer_seconds += time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return None

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore moments onto the CPU (torch's default would cast them to the parameters' GPU device)."""
        id_map = {}
        for group, saved in zip(self.param_groups, state_dict["param_groups"]):
            group.update({k: v for k, v in saved.items() if k != "params"})
            id_map.update(zip(saved["params"], group["params"]))
        self.state.clear()
        for pid, s in state_dict["state"].items():
            p = id_map[pid]
            st = {"step": int(s["step"])}
            for k in ("exp_avg", "exp_avg_sq"):
                t = self._cpu_zeros_like(p)
                t.copy_(s[k].float().cpu())
                st[k] = t
            self.state[p] = st

    def state_bytes(self) -> int:
        return sum(s["exp_avg"].numel() * 4 * 2 for s in self.state.values() if s)


def param_groups(named_params, weight_decay: float) -> list[dict]:
    """Weight decay on tensors with >= 2 dims; none on biases and 1-D parameters (norms)."""
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class L2SPReference:
    def __init__(self, model: torch.nn.Module):
        self.ref: dict[str, torch.Tensor] = {}
        exact_bf16 = True
        with torch.no_grad():
            for name, p in model.named_parameters():  # named_parameters de-duplicates tied weights
                cpu = p.detach().float().cpu()
                as_bf16 = cpu.to(torch.bfloat16)
                if exact_bf16 and torch.equal(as_bf16.float(), cpu):
                    self.ref[name] = as_bf16
                else:
                    exact_bf16 = False
                    self.ref[name] = cpu
        if not exact_bf16:  # keep a single dtype if any tensor is not exactly representable
            self.ref = {k: v.float() for k, v in self.ref.items()}
        self.storage_dtype = "bfloat16" if exact_bf16 else "float32"
        self.storage_bytes = sum(v.numel() * v.element_size() for v in self.ref.values())
        self.n_params = sum(v.numel() for v in self.ref.values())
        self.transfer_bytes_total = 0
        self.seconds = 0.0

    @torch.no_grad()
    def penalty_and_grad(self, model: torch.nn.Module, lam: float, add_grad: bool = True) -> float:
        """Return Omega(theta, theta0) and, if requested, add lam * (theta - theta0) to every gradient."""
        t0 = time.perf_counter()
        omega = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if add_grad and lam > 0 and p.grad is None:
                p.grad = torch.zeros_like(p)
            p_flat, r_flat = p.detach().view(-1), self.ref[name].view(-1)
            for s in range(0, p_flat.numel(), CHUNK):
                e = min(s + CHUNK, p_flat.numel())
                diff = p_flat[s:e].float() - r_flat[s:e].to(p.device, non_blocking=True).float()
                omega += 0.5 * diff.double().pow(2).sum()
                if add_grad and lam > 0:
                    p.grad.view(-1)[s:e].add_(diff.to(p.grad.dtype), alpha=lam)
            self.transfer_bytes_total += self.ref[name].numel() * self.ref[name].element_size()
        val = float(omega.item())
        self.seconds += time.perf_counter() - t0
        return val

    def displacement(self, model: torch.nn.Module) -> float:
        """||theta - theta0||_2 over all unique parameters."""
        return math.sqrt(2.0 * self.penalty_and_grad(model, 0.0, add_grad=False))


def verify_against_torch(n_steps: int = 5, seed: int = 0) -> float:
    """Max abs difference between OffloadedAdamW and torch.optim.AdamW on random problems (should be ~0)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 32), (32,), (16, 16)]
    a = [torch.randn(s, generator=g).to(dev).requires_grad_() for s in shapes]
    b = [x.detach().clone().requires_grad_() for x in a]
    opt_a = OffloadedAdamW([{"params": a[:1] + a[2:], "weight_decay": 0.01}, {"params": a[1:2], "weight_decay": 0.0}],
                           lr=1e-2)
    opt_b = torch.optim.AdamW([{"params": b[:1] + b[2:], "weight_decay": 0.01}, {"params": b[1:2], "weight_decay": 0.0}],
                              lr=1e-2, foreach=False)
    for _ in range(n_steps):
        grads = [torch.randn(s, generator=g).to(dev) for s in shapes]
        for x, y, gr in zip(a, b, grads):
            x.grad, y.grad = gr.clone(), gr.clone()
        opt_a.step()
        opt_b.step()
    return max(float((x - y).abs().max()) for x, y in zip(a, b))
