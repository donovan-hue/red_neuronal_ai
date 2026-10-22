"""Buscador BM25 propio (sin dependencias) sobre el corpus del workspace.

El corpus literario + un archivo opcional docs/knowledge.jsonl forman la base de
conocimiento. La indexacion se persiste en models/rag_index.npz para no rehacerla.
"""
from __future__ import annotations
import json, math, os, re, numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORD = re.compile(r"[a-záéíóúüñ0-9']+", re.I)


def tokenize(t):
    return [w.lower() for w in WORD.findall(t)]


def _docs_from_books(cfg, max_docs=9000):
    ddir = os.path.join(ROOT, cfg["data"]["dir"])
    out = []
    for fn in sorted(os.listdir(ddir)):
        if not fn.endswith(".txt"):
            continue
        try:
            txt = open(os.path.join(ddir, fn), encoding="utf-8").read()
        except Exception:
            continue
        txt = re.sub(r"\r\n?", "\n", txt)
        m = re.search(r"Start of (the )?Project Gutenberg.*?[*]{3,}\s*", txt, re.S)
        if m: txt = txt[m.end():]
        m = re.search(r"[*]{3,}\s*END (OF|of) THE PROJECT GUTENBERG.*", txt, re.S)
        if m: txt = txt[:m.start()]
        ch = cfg["rag"]["chunk_chars"]
        for i in range(0, min(len(txt), 1_400_000), ch):
            c = txt[i:i + ch].strip()
            if len(c) > 200:
                out.append({"src": fn, "text": re.sub(r"\s+", " ", c)})
        if len(out) >= max_docs:
            break
    return out


def _docs_from_extra(cfg):
    p = os.path.join(ROOT, "docs", "knowledge.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.df = Counter()
        self.tokens = []
        for d in docs:
            tk = tokenize(d["text"])
            self.tokens.append(Counter(tk))
            for u in set(tk):
                self.df[u] += 1
        self.N = max(1, len(docs))
        self.lens = np.array([sum(c.values()) for c in self.tokens], dtype=np.float64)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 1.0
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in self.df.items()}

    def search(self, query, k=4):
        q = tokenize(query)
        if not q:
            return []
        scores = np.zeros(self.N)
        for w in set(q):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, cnt in enumerate(self.tokens):
                f = cnt.get(w, 0)
                if f:
                    scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        idx = np.argsort(-scores)[:k]
        return [{"score": float(scores[i]), "src": self.docs[i].get("src", "?"), "text": self.docs[i]["text"]}
                for i in idx if scores[i] > 0]


class Index:
    def __init__(self, cfg, rebuild=False):
        self.cfg = cfg
        path = os.path.join(ROOT, cfg["rag"]["index"])
        self.docs = _docs_from_books(cfg) + _docs_from_extra(cfg)
        self.bm = BM25(self.docs)
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rebuild or not os.path.exists(path + ".meta"):
            json.dump({"n_docs": len(self.docs), "avgdl": self.bm.avgdl, "built": True},
                      open(path + ".meta", "w"))

    def search(self, query, k=None):
        return self.bm.search(query, k or self.cfg["rag"]["top_k"])

    def __len__(self):
        return len(self.docs)


_CACHE = {}
def get_index(cfg):
    key = cfg["rag"]["index"]
    if key not in _CACHE:
        _CACHE[key] = Index(cfg)
    return _CACHE[key]
