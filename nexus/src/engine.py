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
    def complete(self, prompt, max_new=320, temperature=0.6, top_k=200, top_p=0.92, stop=None, messages=None):
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
    def choice(self, prompt, options):
        """Compara log-probabilidad de la secuencia completa de cada continuación."""
        import torch
        context = self._ids(prompt)[-self.block + 1:]
        if not context:
            return []
        out = []
        with self.lock, torch.no_grad():
            for option in options:
                continuation = self._ids(option)
                if not continuation:
                    out.append((option, float("-inf")))
                    continue
                combined = (context + continuation)[:self.block]
                if len(combined) <= len(context):
                    out.append((option, float("-inf")))
                    continue
                x = torch.tensor([combined], dtype=torch.long)
                logits, _ = self.model(x)
                log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                score = 0.0
                for i, token_id in enumerate(continuation[:len(combined) - len(context)]):
                    score += float(log_probs[len(context) - 1 + i, token_id])
                out.append((option, score))
        return sorted(out, key=lambda item: -item[1])

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
    def _payload(self, messages, max_new, temperature, stop=None):
        k = self.p["kind"]
        if k == "openai":
            body = {"model": self.p["model"], "messages": messages,
                    "max_tokens": max_new, "temperature": temperature}
            if stop:
                body["stop"] = stop
            return f"{self.p['base_url']}/chat/completions", body, {"Authorization": f"Bearer {self.key}"}
        if k == "anthropic":
            sysmsg = " ".join(m["content"] for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            body = {"model": self.p["model"], "system": sysmsg or None,
                    "messages": rest, "max_tokens": max_new, "temperature": temperature}
            if stop:
                body["stop_sequences"] = stop
            return f"{self.p['base_url']}/messages", body, {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        sysmsg = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "user" if m["role"] != "assistant" else "model",
                     "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        generation = {"maxOutputTokens": max_new, "temperature": temperature}
        if stop:
            generation["stopSequences"] = stop
        body = {"contents": contents, "generationConfig": generation}
        if sysmsg:
            body["systemInstruction"] = {"parts": [{"text": sysmsg}]}
        return (f"{self.p['base_url']}/models/{self.p['model']}:generateContent", body,
                {"x-goog-api-key": self.key})

    @staticmethod
    def _messages_from_prompt(prompt):
        pattern = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)<\|end\|>\n?", re.S)
        messages = [{"role": role, "content": content}
                    for role, content in pattern.findall(prompt or "")]
        if not messages:
            return [{"role": "user", "content": prompt or ""}]
        tail = pattern.sub("", prompt or "").strip()
        if tail and tail not in ("<|assistant|>", "<|assistant|>\\n"):
            messages.append({"role": "assistant", "content": tail.removeprefix("<|assistant|>").strip()})
        return messages

    def complete(self, prompt, max_new=600, temperature=0.4, top_k=None, top_p=None,
                 stop=None, messages=None):
        msgs = messages or self._messages_from_prompt(prompt)
        url, body, hdr = self._payload(msgs, max_new, temperature, stop=stop)
        r = self.httpx.post(url, json={k: v for k, v in body.items() if v is not None},
                            headers=hdr, timeout=120)
        r.raise_for_status()
        d = r.json()
        k = self.p["kind"]
        if k == "openai":
            text = d["choices"][0]["message"].get("content") or ""
        elif k == "anthropic":
            text = "".join(b.get("text", "") for b in d.get("content", []))
        else:
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        for marker in stop or []:
            if marker in text:
                text = text.split(marker, 1)[0]
        return text

    def choice(self, prompt, options):
        # Estos endpoints no exponen log-probabilidades comparables de continuaciones.
        return []

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
