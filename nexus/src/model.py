"""Transformer causal: RMSNorm + RoPE + SwiGLU + atencion con GQA y atado de embeddings."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = torch.ones(d); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(T, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    frq = torch.outer(pos, inv)
    return torch.cos(frq), torch.sin(frq)

def rope_apply(x, cos, sin):
    # x: [B, H, T, hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]][None, None], sin[: x.shape[-2]][None, None]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, h, hkv, drop=0.0):
        super().__init__()
        self.h, self.hkv, self.hd = h, hkv, d // h
        self.q = nn.Linear(d, h * self.hd, bias=False)
        self.k = nn.Linear(d, hkv * self.hd, bias=False)
        self.v = nn.Linear(d, hkv * self.hd, bias=False)
        self.o = nn.Linear(h * self.hd, d, bias=False)
        self.drop = drop
    def forward(self, x, cos, sin, mask=None):
        B, T, d = x.shape
        g = self.h // self.hkv
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.hkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.hkv, self.hd).transpose(1, 2)
        q, k = rope_apply(q, cos, sin), rope_apply(k, cos, sin)
        if g > 1:
            k = k.repeat_interleave(g, dim=1); v = v.repeat_interleave(g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(mask is None), dropout_p=self.drop)
        return self.o(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, h, hkv, ff, drop=0.0):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attn(d, h, hkv, drop)
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, ff, bias=False); self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class NexusLM(nn.Module):
    def __init__(self, vocab, block=256, d=320, n_layers=4, n_heads=8, n_kv=4, ff=960, drop=0.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, block=block, d=d, n_layers=n_layers, n_heads=n_heads,
                        n_kv=n_kv, ff=ff, drop=drop)
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, n_kv, ff, drop) for _ in range(n_layers)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.drop = nn.Dropout(drop)
        self.apply(self._init)
        for pb in self.blocks:
            for lin in (pb.attn.o, pb.down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def _cache_rope(self, T, device):
        if getattr(self, "_rc", None) is None or self._rc[0] < T or self._rc[1] != device:
            hd = self.cfg["d"] // self.cfg["n_heads"]
            c, s = rope_cache(max(T, 2048), hd, device)
            self._rc = (T, device, c, s)
        return self._rc[2], self._rc[3]
    def forward(self, idx, targets=None, y=None):
        B, T = idx.shape
        if y is None: y = torch.zeros((B, T, self.cfg["d"]), device=idx.device)
        x = self.drop(self.tok(idx)) + y
        cos, sin = self._cache_rope(T, idx.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg["vocab"]), targets.view(-1),
                                   ignore_index=-1, label_smoothing=0.0)
        return logits, loss
    @torch.no_grad()
    def generate(self, ids, max_new, temperature=1.0, top_k=None, top_p=None, eos=None, y=None, stream=None):
        self.eval()
        for _ in range(max_new):
            ctx = ids[:, -self.cfg["block"]:]
            logits, _ = self.forward(ctx, None, y)
            nxt = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, i = torch.topk(nxt, min(top_k, nxt.size(-1)))
                m = torch.full_like(nxt, float("-inf")); nxt = m.scatter(-1, i, v)
            if top_p:
                srt, idxs = torch.sort(nxt, descending=True)
                cm = torch.cumsum(F.softmax(srt, -1), -1)
                rm = (cm - F.softmax(srt, -1)) > top_p
                srt[rm] = float("-inf")
                nxt = torch.full_like(nxt, float("-inf")).scatter(-1, idxs, srt)
            pr = torch.softmax(nxt, -1)[0]
            nxt = torch.multinomial(pr, 1).view(1, 1)
            ids = torch.cat((ids, nxt), 1)
            if stream: stream(nxt.item())
            if eos is not None and nxt.item() == eos: break
        return ids
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def load_model(ckpt, vocab, device="cpu"):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    m = NexusLM(**d["cfg"]).to(device).eval()
    m.load_state_dict(d["model"]); return m, d
