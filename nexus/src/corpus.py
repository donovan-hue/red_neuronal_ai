"""Construccion del dataset: corpus literario bilingue + sintetica de razonamiento.

Salidas: models/tokenizer.json, data/tokens/{train.bin,val.bin,manifest.json}
"""
from __future__ import annotations
import json, os, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import synth
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>", "<|eos|>"]


def clean_book(text):
    # recorta cabecera/legal y pie de licencia de Project Gutenberg
    m = re.search(r"Start of (the )?Project Gutenberg.*?[*]{3,}\s*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"[*]{3,}\s*END (OF|of) THE PROJECT GUTENBERG.*", text, re.S)
    if m:
        text = text[:m.start()]
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return text.strip()

def paragraphs(text, min_chars, target=1600):
    """Agrupa parrafos en documentos de tamano ~target para diversidad de longitud."""
    out, buf = [], []
    n = 0
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if len(p) < 40:
            continue
        buf.append(p); n += len(p)
        if n >= target:
            out.append("\n\n".join(buf)); buf, n = [], 0
    if buf:
        out.append("\n\n".join(buf))
    return [d for d in out if len(d) >= min_chars]

def load_docs(cfg):
    ddir = os.path.join(ROOT, cfg["data"]["dir"])
    docs = []
    for fn in sorted(os.listdir(ddir)):
        if not fn.endswith(".txt"):
            continue
        try:
            raw = open(os.path.join(ddir, fn), encoding="utf-8").read()
        except Exception:
            continue
        docs.extend(paragraphs(clean_book(raw), cfg["data"]["min_doc_chars"]))
    return docs

def synth_blocks(cfg, n, seed):
    rng = random_seed(seed)
    blocks, tools = [], []
    for i in range(n):
        it = synth.make_item(rng)
        blocks.append(synth.render_train(it["question"], it["answer"], True,
                                         it.get("why"), i, it.get("expr")))
        if i % 4 == 0:  # ejemplos del formato de herramientas que usara el agente
            a, b = rng.randint(11, 999), rng.randint(3, 99)
            expr = f"{a} * {b}"
            tools.append(synth.render_tool_example(f"Calcula {expr} con precision exacta.", expr, a * b))
    return blocks, tools

def random_seed(seed):
    import random
    return random.Random(seed)

def train_tokenizer(texts, vocab, path):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, show_progress=False, special_tokens=SPECIALS,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)
    tok.train_from_iterator(texts, tr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok

def build(cfg, force=False):
    outdir = os.path.join(ROOT, cfg["data"]["out"])
    manifest = os.path.join(outdir, "manifest.json")
    if os.path.exists(manifest) and not force:
        print("dataset ya construido:", json.load(open(manifest))); return
    os.makedirs(outdir, exist_ok=True)
    docs = load_docs(cfg)
    print(f"documentos literarios: {len(docs)}  ({sum(len(d) for d in docs)/1e6:.1f} MB)")

    tk_path = os.path.join(ROOT, cfg["model"]["params_dir"], "tokenizer.json")
    sample = docs[: max(1, len(docs) // 4)]
    tok = train_tokenizer(sample, cfg["model"]["vocab_size"], tk_path)
    V = tok.get_vocab_size()
    print(f"tokenizer: vocab={V}  ->  {tk_path}")
    eos = tok.token_to_id("<|eos|>")

    blocks, tools = synth_blocks(cfg, cfg["data"]["synth_samples"], cfg["data"]["seed"])
    share = cfg["data"]["synth_share"]
    cap = int(len(docs) * share / (1 - share) * 14)   # los bloques sinteticos son cortos: compensar
    n_synth = min(len(blocks), max(2000, cap))
    synth_use = blocks[:n_synth] + tools[: max(3000, n_synth)]
    corpus = docs + synth_use
    rng = random_seed(cfg["data"]["seed"] + 1); rng.shuffle(corpus)
    print(f"documentos totales tras mezcla: {len(corpus)} (sinteticos {len(synth_use)})")

    enc = [tok.encode(d).ids + [eos] for d in corpus]
    lengths = [len(e) for e in enc]
    print(f"tokens: {sum(lengths)/1e6:.2f}M  media/doc {int(np.mean(lengths))}")
    val_n = max(1, int(len(enc) * 0.01))
    val, train = enc[:val_n], enc[val_n:]
    def dump(arr, path, V):
        flat = np.concatenate(arr) if arr else np.array([eos], dtype=np.uint16)
        dtype = np.uint16 if V < 65536 else np.uint32
        np.array(flat, dtype=dtype).tofile(path)
        return flat.size
    nt = dump(train, os.path.join(outdir, "train.bin"), V)
    nv = dump(val, os.path.join(outdir, "val.bin"), V)
    json.dump({"train_tokens": int(nt), "val_tokens": int(nv), "vocab": V,
               "docs": len(docs), "synth_blocks": len(synth_use),
               "docs_total": len(corpus), "char_len_avg": int(np.mean(lengths))},
              open(manifest, "w"), indent=2)
    json.dump({"vocab": V}, open(os.path.join(outdir, "vocab.json"), "w"))
    print("listo:", nt, "tokens train /", nv, "val")

if __name__ == "__main__":
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    build(cfg, force="--force" in sys.argv)
