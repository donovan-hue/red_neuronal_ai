"""Motor de inferencia. Contrato unico para local y API frontier."""
from __future__ import annotations
import json, os, re, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

def render_prompt(messages, add_generation=True):
    """Plantilla de chat. Los tokens <|...|> son ids reales del vocabulario BPE."""
    s = []
    for m in messages:
        r = m.get("role", "user")
        tag = {"system": "system", "user": "user", "assistant": "assistant"}.get(r, "user")
        s.append(f"<|{tag}|>\n{m.get('content','')}<|end|>\n")
    if add_generation:
        s.append("<|assistant|>\n")
    return "".join(s)

def stop_at(text):
    for t in ("<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "Observacion:"):
        i = text.find(t)
        if i >= 0:
            text = text[:i]
    return text

class LocalEngine:
    kind = "local"
    def __init__(self, cfg, device="cpu"):
        import torch
        from tokenizer_ext import Tok
        from model import load_model
        self.torch = torch
        self.cfg = cfg
        tk_path = os.path.join(ROOT, cfg["model"]["params_dir"], "tokenizer.json")
        self.tok = Tok(tk_path)
        ck = os.path.join(ROOT, cfg["train"]["ckpt"])
        if not os.path.exists(ck):
            raise SystemExit(f"falta checkpoint {ck}; ejecuta: python3 src/train.py")
        self.model, meta = load_model(ck, self.tok.vocab, device)
        self.step = meta.get("step")
        self.val = meta.get("losses", {}).get("val")
        self.block = self.model.cfg["block"]
        self.lock = threading.Lock()
    def _ids(self, text):
        return self.tok.encode(text)
    def complete(self, prompt, max_new=320, temperature=0.6, top_k=200, top_p=0.92, stop=None):
        torch = self.torch
        ids = self._ids(prompt)
        ids = torch.tensor([ids[-self.block * 3:]], dtype=torch.long)
        with self.lock, torch.no_grad():
            out = self.model.generate(ids, max_new, temperature=temperature, top_k=top_k,
                                      top_p=top_p, eos=self.tok.eos_id)
        gen = out[0][ids.shape[1]:].tolist()
        txt = stop_at(self.tok.decode(gen))
        if stop:
            for s in stop:
                i = txt.find(s)
                if i >= 0: txt = txt[:i]
        return txt.strip()
    @torch.no_grad() if False else (lambda f: f)
    def choice(self, prompt, options):
        """Enruta por verosimilitud real del modelo: que prefiere a continuacion.

        No es un if heuristico: comparamos la probabilidad del primer token de cada
        continuation posible. Devuelve [(opcion, logprob), ...].
        """
        import torch
        ids = self._ids(prompt)[-self.block * 2:]
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = self.model(x)
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        out = []
        for o in options:
            tid = self._ids(o)[0] if self._ids(o) else 0
            out.append((o, float(lp[tid].detach())))
        return sorted(out, key=lambda t: -t[1])

    def stream(self, prompt, **kw):
        # el modelo local es pequeno: generamos y vertemos en trozos (mismo contrato que la API)
        full = self.complete(prompt, **kw)
        for i in range(0, len(full), 12):
            yield full[i:i + 12]
            time.sleep(0.01)
    def info(self):
        return {"kind": "local", "step": self.step, "val_loss": self.val,
                "params_M": round(self.model.n_params / 1e6, 2)}

class APIEngine:
    kind = "api"
    def __init__(self, cfg, name=None):
        import httpx
        self.httpx = httpx
        self.cfg = cfg
        prov = cfg["providers"]
        self.name = name or next((k for k, v in prov.items() if os.environ.get(v["key_env"])), None)
        if not self.name:
            raise RuntimeError("no hay clave de API en el entorno")
        self.p = prov[self.name]
        self.key = os.environ.get(self.p["key_env"])
    def _payload(self, messages, max_new, temperature):
        k = self.p["kind"]
        if k == "openai":
            return f"{self.p['base_url']}/chat/completions", {"model": self.p["model"], "messages": messages,
                    "max_tokens": max_new, "temperature": temperature}, {"Authorization": f"Bearer {self.key}"}
        if k == "anthropic":
            sysmsg = " ".join(m["content"] for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            return f"{self.p['base_url']}/messages", {"model": self.p["model"], "system": sysmsg or None,
                    "messages": rest, "max_tokens": max_new, "temperature": temperature}, \
                   {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        return (f"{self.p['base_url']}/models/{self.p['model']}:generateContent?key={self.key}",
                {"contents": [{"role": "user" if m["role"] != "assistant" else "model",
                               "parts": [{"text": m["content"]}]} for m in messages],
                 "generationConfig": {"maxOutputTokens": max_new, "temperature": temperature}}, {})
    def complete(self, prompt, max_new=600, temperature=0.4, top_k=None, top_p=None, stop=None, messages=None):
        msgs = messages or [{"role": "user", "content": prompt}]
        url, body, hdr = self._payload(msgs, max_new, temperature)
        r = self.httpx.post(url, json={k: v for k, v in body.items() if v is not None},
                            headers=hdr, timeout=120)
        r.raise_for_status()
        d = r.json()
        k = self.p["kind"]
        if k == "openai": return d["choices"][0]["message"]["content"]
        if k == "anthropic": return "".join(b.get("text", "") for b in d.get("content", []))
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    def choice(self, prompt, options):
        return [(o, 0.0) for o in options]      # la API decide sola; no la enrutamos

    def stream(self, prompt, **kw):
        txt = self.complete(prompt, **kw)
        for i in range(0, len(txt), 24):
            yield txt[i:i + 24]
    def info(self):
        return {"kind": "api", "provider": self.name, "model": self.p["model"]}

def make_engine(cfg, force=None):
    """auto: API si hay clave, si no motor local entrenado aqui."""
    which = force or cfg.get("provider", "auto")
    if which != "local":
        try:
            return APIEngine(cfg, None if which == "auto" else which)
        except Exception as e:
            if which != "auto":
                raise
    return LocalEngine(cfg)
