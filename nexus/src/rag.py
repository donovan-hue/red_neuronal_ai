"""BM25 y caché persistente comprimida del corpus indexado."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import threading

import numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORD = re.compile(r"[a-záéíóúüñ0-9']+", re.I)
_CACHE = {}
_CACHE_LOCK = threading.RLock()


def tokenize(text):
    return [word.lower() for word in WORD.findall(text)]


def _source_fingerprint(cfg):
    data_dir = os.path.join(ROOT, cfg["data"]["dir"])
    paths = []
    if os.path.isdir(data_dir):
        paths.extend(os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".txt"))
    extra = os.path.join(ROOT, "docs", "knowledge.jsonl")
    if os.path.isfile(extra):
        paths.append(extra)
    sig = []
    for path in sorted(paths):
        st = os.stat(path)
        sig.append((os.path.relpath(path, ROOT), st.st_size, st.st_mtime_ns))
    payload = {"files": sig, "chunk_chars": cfg["rag"]["chunk_chars"],
               "overlap": cfg["rag"].get("overlap", 0)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _docs_from_books(cfg, max_docs=9000):
    ddir = os.path.join(ROOT, cfg["data"]["dir"])
    out = []
    if not os.path.isdir(ddir):
        return out
    chunk = int(cfg["rag"]["chunk_chars"])
    step = max(1, chunk - int(cfg["rag"].get("overlap", 0)))
    for fn in sorted(os.listdir(ddir)):
        if not fn.endswith(".txt"):
            continue
        try:
            with open(os.path.join(ddir, fn), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        text = re.sub(r"\r\n?", "\n", text)
        start = re.search(r"Start of (the )?Project Gutenberg.*?[*]{3,}\s*", text, re.S)
        if start:
            text = text[start.end():]
        end = re.search(r"[*]{3,}\s*END (OF|of) THE PROJECT GUTENBERG.*", text, re.S)
        if end:
            text = text[:end.start()]
        for i in range(0, min(len(text), 1_400_000), step):
            content = re.sub(r"\s+", " ", text[i:i + chunk]).strip()
            if len(content) > 200:
                out.append({"src": fn, "text": content})
                if len(out) >= max_docs:
                    return out
    return out


def _docs_from_extra():
    path = os.path.join(ROOT, "docs", "knowledge.jsonl")
    if not os.path.isfile(path):
        return []
    docs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"knowledge.jsonl:{line_no}: se esperaba un objeto con campo text")
            docs.append({"src": str(item.get("src", "knowledge.jsonl")), "text": item["text"]})
    return docs


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.df = Counter()
        self.tokens = []
        for doc in docs:
            counts = Counter(tokenize(doc["text"]))
            self.tokens.append(counts)
            self.df.update(counts.keys())
        self.N = max(1, len(docs))
        self.lens = np.array([sum(counts.values()) for counts in self.tokens], dtype=np.float64)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 1.0
        self.idf = {word: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for word, n in self.df.items()}

    def to_payload(self):
        return {"docs": self.docs, "k1": self.k1, "b": self.b,
                "tokens": [dict(c) for c in self.tokens], "df": dict(self.df),
                "N": self.N, "lens": self.lens.tolist(), "avgdl": self.avgdl,
                "idf": self.idf}

    @classmethod
    def from_payload(cls, payload):
        obj = cls.__new__(cls)
        obj.docs = payload["docs"]
        obj.k1, obj.b = payload["k1"], payload["b"]
        obj.tokens = [Counter(c) for c in payload["tokens"]]
        obj.df = Counter(payload["df"])
        obj.N = int(payload["N"])
        obj.lens = np.asarray(payload["lens"], dtype=np.float64)
        obj.avgdl = float(payload["avgdl"])
        obj.idf = {str(k): float(v) for k, v in payload["idf"].items()}
        return obj

    def search(self, query, k=4):
        words = set(tokenize(query))
        if not words or not self.docs:
            return []
        scores = np.zeros(self.N, dtype=np.float64)
        for word in words:
            idf = self.idf.get(word)
            if idf is None:
                continue
            for i, counts in enumerate(self.tokens):
                freq = counts.get(word, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                    scores[i] += idf * freq * (self.k1 + 1) / denom
        top = np.argsort(-scores)[:max(0, int(k))]
        return [{"score": float(scores[i]), "src": self.docs[i].get("src", "?"),
                 "text": self.docs[i]["text"]} for i in top if scores[i] > 0]


class Index:
    def __init__(self, cfg, rebuild=False):
        self.cfg = cfg
        self.path = os.path.join(ROOT, cfg["rag"]["index"])
        self.fingerprint = _source_fingerprint(cfg)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not rebuild:
            try:
                with gzip.open(self.path, "rt", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("fingerprint") == self.fingerprint:
                    self.bm = BM25.from_payload(cached["index"])
                    self.docs = self.bm.docs
                    return
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        docs = _docs_from_books(cfg) + _docs_from_extra()
        self.bm = BM25(docs)
        self.docs = docs
        payload = {"fingerprint": self.fingerprint, "index": self.bm.to_payload()}
        fd, tmp = tempfile.mkstemp(prefix="rag-", suffix=".tmp", dir=os.path.dirname(self.path))
        try:
            with os.fdopen(fd, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                    gz.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def search(self, query, k=None):
        return self.bm.search(query, k if k is not None else self.cfg["rag"]["top_k"])

    def __len__(self):
        return len(self.docs)


def get_index(cfg):
    key = (cfg["rag"]["index"], _source_fingerprint(cfg))
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Index(cfg)
        return _CACHE[key]
