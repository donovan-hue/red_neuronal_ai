"""BPE tokenizer entrenado desde cero, sin dependencias externas.

Vocabulario: charset (fallback UTF-8 byte a byte para lo no visto) + merges BPE.
Tokens de control: 0=<|system|> 1=<|user|> 2=<|assistant|> 3=<|end|> 4=<|pad|>
El id 5 se reserva para el fin de secuencia de muestreo (EOS).
"""
from __future__ import annotations
import json, os, re
from collections import Counter, defaultdict

CONTROL = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>", "<|eos|>"]
N_CONTROL = len(CONTROL)
PRETOK = re.compile(r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s", re.UNICODE) if False else \
         re.compile(r"[A-Za-z\u00C0-\u024F]+|\d+|[^\sA-Za-z\u00C0-\u024F\d]|\s")


def pretokenize(text: str):
    return PRETOK.findall(text)


class BPETokenizer:
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        self.merges = []            # lista de (a,b) en orden de aprendizaje
        self.bpe_ranks = {}
        self._cache = {}
        self.bos_id = 0
        self.eos_id = CONTROL.index("<|eos|>")

    # ---------- vocab base ----------
    def _build_charset(self, words):
        chars = set()
        for w in words:
            chars |= set(w)
        chars = sorted(c for c in chars if c not in CONTROL)
        self.itos = CONTROL + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.byte_start = len(self.itos)      # ids de bytes sueltos 256
        for b in range(256):
            ch = bytes([b]).decode("utf-8", errors="ignore")
            self.itos.append(ch if ch else f"<0x{b:02X}>")
        self.byte_base = len(self.itos)       # primer id de merge BPE
        self.stoi = {c: i for i, c in enumerate(self.itos) if c not in self.stoi}

    # ---------- entrenamiento ----------
    def train(self, text: str, verbose=True):
        words = Counter(pretokenize(text))
        self._build_charset(list(words))
        # representacion tipo tuple de simbolos por palabra
        data = {w: tuple(list(w)) for w in words}
        freq = words

        def pair_freq(d):
            p = Counter()
            for w, c in d.items():
                for i in range(len(c) - 1):
                    if len(c) < 10000:
                        p[(c[i], c[i + 1])] += c and freq[w]
            return p

        n_target = self.vocab_size - self.byte_base
        for step in range(n_target):
            counts = Counter()
            for w, syms in data.items():
                s = list(syms)
                for i in range(len(s) - 1):
                    counts[(s[i], s[i + 1])] += freq[w]
            if not counts:
                break
            best = counts.most_common(1)[0][0]
            new = best[0] + best[1]
            if new in self.stoi:
                best = next((p for p, _ in counts.most_common(50) if p[0] + p[1] not in self.stoi), None)
                if best is None:
                    break
                new = best[0] + best[1]
            self.merges.append(best)
            self.stoi[new] = len(self.itos)
            self.itos.append(new)
            for w in list(data):
                syms = data[w]
                if best[0] in syms and best[1] in syms:
                    out, i, s = [], 0, list(syms)
                    while i < len(s):
                        if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                            out.append(new); i += 2
                        else:
                            out.append(s[i]); i += 1
                    data[w] = tuple(out)
            if verbose and step % 400 == 0:
                print(f"  BPE {len(self.itos)}/{self.vocab_size}", flush=True)
        self.bpe_ranks = {p: i for i, p in enumerate(self.merges)}
        self._cache.clear()
        return self

    # ---------- codificacion ----------
    def _bpe(self, word: str):
        if word in self._cache:
            return self._cache[word]
        syms = list(word)
        while len(syms) > 1:
            pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
            cand = [self.bpe_ranks[p] for p in pairs if p in self.bpe_ranks]
            if not cand:
                break
            r = min(cand)
            a, b = self.merges[r]
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(syms[i]); i += 1
            syms = out
        res = [self.stoi.get(s, None) for s in syms]
        final = []
        for s, tid in zip(syms, res):
            if tid is None:
                for bt in s.encode("utf-8", errors="ignore"):
                    ch = bytes([bt]).decode("utf-8", errors="ignore")
                    final.append(self.stoi.get(ch, self.stoi.get(f"<0x{bt:02X}>", self.stoi[self.byte_base + bt] if self.byte_base + bt < len(self.itos) else 0)))
            else:
                final.append(tid)
        self._cache[word] = final
        return final

    def encode(self, text: str):
        ids = []
        for tok in pretokenize(text):
            ids.extend(self._bpe(tok))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    @property
    def vocab(self):
        return len(self.itos)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"itos": self.itos, "merges": [list(m) for m in self.merges],
                   "vocab_size": self.vocab_size}, open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        t = cls(d["vocab_size"])
        t.itos = d["itos"]
        t.stoi = {c: i for i, c in enumerate(t.itos)}
        t.merges = [tuple(m) for m in d["merges"]]
        t.bpe_ranks = {p: i for i, p in enumerate(t.merges)}
        t.byte_base = len(t.itos)
        t.byte_start = 0
        return t


def build_chat(tokens_ids, role_ids):
    """(no usado en entrenamiento; helper de formato)."""
    return tokens_ids
