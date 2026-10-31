"""SigLIP-Qwen VLM: connector, prompt construction, example-weighted loss, reference NLL and greedy decoding.

Input layout (user turn of Qwen3's chat template, thinking disabled):
    <|im_start|>user\\n <|vision_start|> [N visual tokens] <|vision_end|> \\n {question} <|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n {target} <|im_end|>
The N placeholder positions (``<|image_pad|>``) are replaced by connector outputs Z(I) = g_phi(F(I)). For the
no-image condition the vision block (start token, placeholders, end token and its newline) is removed and
masks/positions are rebuilt from the shorter sequence; the text instruction is unchanged.

Loss (manual): causal CE on target tokens and the end-of-turn token only, averaged within each example, then
averaged over the effective batch (each caption or short answer has equal weight). Answer NLL for evaluation
excludes the end-of-turn token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SENTINEL = "\u0001SENTINEL\u0001"


class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024; optional 2 x 2 average pooling (196 -> 49 tokens)."""

    def __init__(self, in_dim: int = 768, hidden: int = 1024, out_dim: int = 1024, pool2x2: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.pool2x2 = pool2x2

    def forward(self, f: torch.Tensor) -> torch.Tensor:  # f: (B, 196, 768), row-major 14 x 14
        if self.pool2x2:
            B, N, D = f.shape
            g = int(N ** 0.5)
            f = f.view(B, g, g, D).permute(0, 3, 1, 2)
            f = F.avg_pool2d(f, 2).permute(0, 2, 3, 1).reshape(B, (g // 2) ** 2, D)
        return self.fc2(self.act(self.fc1(f)))


class PromptBuilder:
    def __init__(self, tok, cfg: dict, n_visual: int):
        self.tok = tok
        text = tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pre, post = text.split(SENTINEL)
        self.prefix = tok(pre, add_special_tokens=False).input_ids
        self.suffix = tok(post, add_special_tokens=False).input_ids
        ids = tok.convert_tokens_to_ids
        self.image_pad = ids("<|image_pad|>")
        self.eot = ids("<|im_end|>")
        self.vision = [ids("<|vision_start|>")] + [self.image_pad] * n_visual + [ids("<|vision_end|>")] + \
            tok("\n", add_special_tokens=False).input_ids
        self.caps = cfg["caps"]
        self.caption_instruction = tok(cfg["prompts"]["caption_instruction"], add_special_tokens=False).input_ids
        self.answer_suffix = tok("\n" + cfg["prompts"]["answer_suffix"], add_special_tokens=False).input_ids

    def build(self, ex: dict, with_image: bool, target: str | None, include_eot: bool = True) -> dict:
        """Token ids + labels for one example. ``target=None`` builds the generation prompt only."""
        if ex["kind"] == "caption":
            q = self.caption_instruction
            q_trunc = False
            cap = self.caps["caption_target_tokens"]
        else:
            q_full = self.tok(ex["question"], add_special_tokens=False).input_ids
            q_trunc = len(q_full) > self.caps["question_tokens"]
            q = q_full[:self.caps["question_tokens"]] + self.answer_suffix
            cap = self.caps["answer_target_tokens"]
        prompt = self.prefix + (self.vision if with_image else []) + q + self.suffix
        out = {"prompt_len": len(prompt), "question_truncated": q_trunc, "target_truncated": False}
        if target is None:
            out.update(ids=prompt, labels=[-100] * len(prompt))
            return out
        t_full = self.tok(target, add_special_tokens=False).input_ids
        t = t_full[:cap]
        out["target_truncated"] = len(t_full) > cap
        tail = t + [self.eot]
        lab = t + ([self.eot] if include_eot else [-100])
        out.update(ids=prompt + tail, labels=[-100] * len(prompt) + lab)
        return out


def collate(built: list[dict], pad_id: int, left_pad: bool = False) -> dict:
    T = max(len(b["ids"]) for b in built)
    B = len(built)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for i, b in enumerate(built):
        n = len(b["ids"])
        sl = slice(T - n, T) if left_pad else slice(0, n)
        ids[i, sl] = torch.tensor(b["ids"])
        lab[i, sl] = torch.tensor(b["labels"])
        att[i, sl] = 1
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    return {"input_ids": ids, "labels": lab, "attention_mask": att, "position_ids": pos}


class VLM(nn.Module):
    def __init__(self, qwen, connector: Connector, builder: PromptBuilder):
        super().__init__()
        self.qwen = qwen
        self.connector = connector
        self.builder = builder

    def embed(self, input_ids: torch.Tensor, feats: torch.Tensor | None) -> torch.Tensor:
        emb = self.qwen.get_input_embeddings()(input_ids)
        mask = input_ids == self.builder.image_pad
        if feats is not None and feats.shape[0] > 0:
            vis = self.connector(feats.to(next(self.connector.parameters()).device))
            assert int(mask.sum()) == vis.shape[0] * vis.shape[1], "visual-token count mismatch"
            emb = emb.masked_scatter(mask.unsqueeze(-1), vis.reshape(-1, vis.shape[-1]).to(emb.dtype))
        else:
            assert not mask.any()
        return emb

    def per_example_nll(self, batch: dict, feats: torch.Tensor | None) -> torch.Tensor:
        """Mean token NLL of supervised positions for each example (B,)."""
        emb = self.embed(batch["input_ids"], feats)
        h = self.qwen.model(inputs_embeds=emb, attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"]).last_hidden_state
        y = batch["labels"][:, 1:]
        m = y != -100
        logits = self.qwen.lm_head(h[:, :-1][m]).float()
        tok_nll = F.cross_entropy(logits, y[m], reduction="none")
        ex_idx = torch.arange(y.shape[0], device=y.device).unsqueeze(1).expand_as(y)[m]
        sums = torch.zeros(y.shape[0], device=y.device).index_add_(0, ex_idx, tok_nll)
        counts = m.sum(1).clamp_min(1)
        return sums / counts

    @torch.no_grad()
    def generate(self, batch: dict, feats: torch.Tensor | None, max_new_tokens: int) -> list[list[int]]:
        """KV-cached greedy decoding; stops per sequence at <|im_end|> (or EOS)."""
        stop = {self.builder.eot, self.builder.tok.eos_token_id}
        att, pos = batch["attention_mask"], batch["position_ids"]
        emb = self.embed(batch["input_ids"], feats)
        out = self.qwen(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        B = nxt.shape[0]
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=nxt.device)
        for step in range(max_new_tokens):
            for i in range(B):
                if not done[i]:
                    t = int(nxt[i])
                    if t in stop:
                        done[i] = True
                    else:
                        gen[i].append(t)
            if bool(done.all()) or step == max_new_tokens - 1:
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=att.device)], 1)
            pos = pos[:, -1:] + 1
            out = self.qwen(input_ids=nxt.unsqueeze(1), attention_mask=att, position_ids=pos, past_key_values=past,
                            use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        return gen


def variant_settings(cfg: dict, variant: str) -> dict:
    """``default`` = 196 visual tokens; ``pool49`` = optional 2 x 2-pooled 49-token run (separate dirs)."""
    from common.utils import rel

    n = cfg["visual_tokens"] if variant == "default" else 49
    return {"variant": variant, "n_visual": n, "pool2x2": variant == "pool49",
            "ckpt": rel(cfg["paths"]["checkpoints"]) / variant, "results": rel(cfg["paths"]["results"]) / variant}


def build_vlm(cfg: dict, variant: str, with_lora: bool, device: str = "cuda") -> tuple["VLM", dict]:
    """Frozen FP16 Qwen3 (+ optional LoRA on q/k/v/o, FP32) and an FP32 connector, fresh initialization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.lora import inject_adapters

    vs = variant_settings(cfg, variant)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    qwen = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float16,
                                                attn_implementation="sdpa").to(device)
    for p in qwen.parameters():
        p.requires_grad_(False)
    qwen.config.use_cache = False
    if cfg["train"]["gradient_checkpointing"]:
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(cfg["seed"])
    connector = Connector(pool2x2=vs["pool2x2"]).to(device=device, dtype=torch.float32)
    info = {"variant": vs, "connector_params": sum(p.numel() for p in connector.parameters())}
    if with_lora:
        lc = cfg["lora"]
        info["lora"] = inject_adapters(qwen, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], seed=cfg["seed"])
    return VLM(qwen, connector, PromptBuilder(tok, cfg, vs["n_visual"])), info


def calibrate_connector(connector: Connector, qwen, feats: torch.Tensor, text_token_ids: torch.Tensor) -> dict:
    """Scale fc2 so the mean L2 norm of visual tokens matches the mean norm of training-text token embeddings.

    Rule: s = mean_t ||E[t]||_2 over all token occurrences in training captions/questions divided by the mean
    ||g_phi(F)||_2 over visual tokens of a fixed subset of training images at initialization; fc2's weight and
    bias are multiplied by s once, before Stage 1. Uses training data only.
    """
    with torch.no_grad():
        E = qwen.get_input_embeddings().weight.float()
        text_norm = float(E[text_token_ids.to(E.device)].norm(dim=-1).mean())
        dev = next(connector.parameters()).device
        vis = torch.cat([connector(feats[i:i + 64].to(dev).float()) for i in range(0, len(feats), 64)])
        vis_norm = float(vis.norm(dim=-1).mean())
        s = text_norm / vis_norm
        connector.fc2.weight.mul_(s)
        connector.fc2.bias.mul_(s)
        after = float(torch.cat([connector(feats[i:i + 64].to(dev).float())
                                 for i in range(0, len(feats), 64)]).norm(dim=-1).mean())
    return {"rule": "scale fc2 by mean text-embedding norm / mean initial visual-token norm (training data only)",
            "text_embedding_mean_norm": text_norm, "visual_token_mean_norm_before": vis_norm, "scale": s,
            "visual_token_mean_norm_after": after, "n_images": int(len(feats)), "n_text_tokens": int(len(text_token_ids))}
