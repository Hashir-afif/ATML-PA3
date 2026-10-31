"""LoRA and OPLoRA adapters for ``nn.Linear`` layers (shared by Tasks 1 and 2).

LoRA (Hu et al., 2021):      W = W0 + (alpha / r) B A,  A in R^{r x d_in} (random), B in R^{d_out x r} (zero).
OPLoRA (Xiong and Xie):      with W0 = U S V^T and the leading k singular bases U_k, V_k frozen,
                             B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T,  dW = (alpha / r) B~ A~.

OPLoRA never forms the dense projectors (I - U_k U_k^T); the projected ("effective") factors are computed
from the low-rank factors on every forward pass, so the protected subspaces are excluded exactly up to
floating-point error. The initial A is drawn from a dedicated CPU generator in a fixed module order, so a
LoRA run and an OPLoRA run with the same seed, rank and targets start from identical unprojected factors.

Implemented from the formulas in the manual and papers rather than through PEFT, so that the OPLoRA
projection, the FP32-adapter / FP16-backbone split and the merge path are all explicit.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_A: torch.Tensor):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(init_A.to(device=dev, dtype=torch.float32).clone())
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lora_A, self.lora_B

    _cached: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        A, B = self._cached if self._cached is not None else self.effective_factors()
        h = self.dropout(x)
        if not torch.is_autocast_enabled():
            A, B = A.to(h.dtype), B.to(h.dtype)
        return out + F.linear(F.linear(h, A), B) * self.scaling

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """Dense FP32 update dW = scaling * B_eff A_eff (used for merging and diagnostics)."""
        A, B = self.effective_factors()
        return self.scaling * (B.float() @ A.float())


class OPLoRALinear(LoRALinear):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_A: torch.Tensor,
                 U_k: torch.Tensor, V_k: torch.Tensor):
        super().__init__(base, r, alpha, dropout, init_A)
        dev = base.weight.device
        # Frozen FP32 singular bases of the starting weight: U_k (d_out x k), V_k (d_in x k).
        self.register_buffer("U_k", U_k.to(device=dev, dtype=torch.float32).contiguous(), persistent=False)
        self.register_buffer("V_k", V_k.to(device=dev, dtype=torch.float32).contiguous(), persistent=False)

    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.lora_A, self.lora_B
        with torch.autocast(device_type=A.device.type, enabled=False):  # projections always in FP32
            B_t = B - self.U_k @ (self.U_k.t() @ B)
            A_t = A - (A @ self.V_k) @ self.V_k.t()
        return A_t, B_t


def target_linears(model: nn.Module, targets: list[str]) -> list[tuple[str, nn.Linear]]:
    """Linear layers whose attribute name is one of ``targets``, in deterministic module order."""
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            found.append((name, module))
    if not found:
        raise ValueError(f"No nn.Linear modules matched targets {targets}")
    return found


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


@torch.no_grad()
def singular_bases(weight: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading-k left/right singular vectors of a weight, computed in FP32 on the CPU.

    CPU (LAPACK) FP32 SVD returns bases orthonormal to ~1e-6; the CUDA FP32 SVD measured only ~7e-4 on
    Qwen3-0.6B weights, which visibly weakens the OPLoRA projection, so the CPU path is used. Returned on
    the input's device.
    """
    U, _, Vh = torch.linalg.svd(weight.detach().float().cpu(), full_matrices=False)
    return U[:, :k].contiguous().to(weight.device), Vh[:k, :].t().contiguous().to(weight.device)


def inject_adapters(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float, seed: int,
                    method: str = "lora", protect_k: int | None = None,
                    reference_weights: dict[str, torch.Tensor] | None = None) -> dict:
    """Replace targeted linears with LoRA/OPLoRA wrappers. Returns bookkeeping (counts, timings, storage).

    ``reference_weights`` optionally maps module names to FP32 starting weights for the OPLoRA SVD; by
    default the (possibly FP16) live base weight is upcast, which callers should avoid when an exact FP32
    copy of the original checkpoint is available.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    info = {"method": method, "r": r, "alpha": alpha, "dropout": dropout, "targets": targets,
            "n_modules": 0, "adapter_params": 0, "svd_seconds": 0.0, "basis_bytes": 0}
    for name, lin in target_linears(model, targets):
        init_A = torch.empty(r, lin.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(init_A, a=math.sqrt(5), generator=gen)  # PEFT/LoRA default init for A
        if method == "lora":
            new = LoRALinear(lin, r, alpha, dropout, init_A)
        elif method == "oplora":
            W0 = reference_weights[name] if reference_weights is not None else lin.weight
            t = time.perf_counter()
            U_k, V_k = singular_bases(W0.cpu(), protect_k)
            info["svd_seconds"] += time.perf_counter() - t
            info["basis_bytes"] += (U_k.numel() + V_k.numel()) * 4
            new = OPLoRALinear(lin, r, alpha, dropout, init_A, U_k, V_k)
        else:
            raise ValueError(method)
        _set_submodule(model, name, new)
        info["n_modules"] += 1
        info["adapter_params"] += new.lora_A.numel() + new.lora_B.numel()
    return info


def adapter_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def cache_effective_factors(model: nn.Module, enable: bool) -> None:
    """Compute each adapter's effective factors once (with autograd) and reuse them for every forward call,
    including gradient-checkpoint recomputation, until disabled. Gradients still reach the raw factors through
    the single cached projection graph. Call with ``enable=True`` before each microbatch forward and
    ``enable=False`` after its backward pass; the result is numerically identical to recomputing per call.
    """
    for m in adapter_modules(model).values():
        m._cached = m.effective_factors() if enable else None


def adapter_state_dict(model: nn.Module, effective: bool = False) -> dict[str, torch.Tensor]:
    """Raw trainable factors (for resuming) or effective/projected factors (for inference export)."""
    sd = {}
    for name, m in adapter_modules(model).items():
        if effective:
            with torch.no_grad():
                A, B = m.effective_factors()
            sd[f"{name}.A"] = A.detach().float().cpu().contiguous()
            sd[f"{name}.B"] = B.detach().float().cpu().contiguous()
        else:
            sd[f"{name}.lora_A"] = m.lora_A.detach().cpu().contiguous()
            sd[f"{name}.lora_B"] = m.lora_B.detach().cpu().contiguous()
    return sd


def load_adapter_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    for name, m in adapter_modules(model).items():
        m.lora_A.data.copy_(sd[f"{name}.lora_A"].to(m.lora_A.device))
        m.lora_B.data.copy_(sd[f"{name}.lora_B"].to(m.lora_B.device))


@torch.no_grad()
def merge_effective_factors(model: nn.Module, factors: dict[str, torch.Tensor], scaling: float,
                            reference_weights: dict[str, torch.Tensor] | None = None) -> int:
    """Merge exported effective factors into a fresh (unwrapped) backbone: W = W0 + scaling * B A.

    The merge is computed in FP32 from ``reference_weights`` when given (exact original weights), then cast
    to the live parameter dtype. Returns the number of merged modules.
    """
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    for name in names:
        lin = model.get_submodule(name)
        A = factors[f"{name}.A"].to(lin.weight.device, torch.float32)
        B = factors[f"{name}.B"].to(lin.weight.device, torch.float32)
        W0 = (reference_weights[name] if reference_weights is not None else lin.weight).to(lin.weight.device,
                                                                                         torch.float32)
        lin.weight.copy_((W0 + scaling * (B @ A)).to(lin.weight.dtype))
    return len(names)
