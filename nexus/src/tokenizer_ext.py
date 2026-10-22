"""Envoltura del BPE entrenado, con ids de control y truncado por documento."""
import json, os

SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>", "<|eos|>"]

class Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)
        self.tk.no_truncation()
        self.vocab = self.tk.get_vocab_size()
        self.eos_id = self.tk.token_to_id("<|eos|>")
        self.end_id = self.tk.token_to_id("<|end|>")
        self.role_ids = {r: self.tk.token_to_id(f"<|{r}|>") for r in ("system", "user", "assistant")}
    def encode(self, text):
        return self.tk.encode(text).ids
    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=False)
