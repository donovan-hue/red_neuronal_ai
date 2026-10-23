"""Entrenamiento desde cero: AdamW, warmup+coseno, clip, checkpoints reanudables.

Pensado para sesiones cortas: si se interrumpe, --resume continua del ultimo paso.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NexusLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_batch(data, T, B, dev, gen):
    ix = torch.randint(0, data.shape[0] - T - 1, (B,), device='cpu', generator=gen)
    x = torch.stack([data[i:i + T] for i in ix]).to(dev)
    y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(dev)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, T, B, n, dev, gen):
    model.eval(); outs = {}
    for split in ("train", "val"):
        src = data[split]
        ls = []
        for _ in range(n):
            x, y = get_batch(src, T, B, dev, gen)
            _, loss = model(x, y)
            ls.append(float(loss))
        outs[split] = sum(ls) / len(ls)
    model.train()
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--steps", type=int, default=None, help="paso final absoluto")
    ap.add_argument("--additional-steps", type=int, default=None, help="pasos a añadir al checkpoint actual")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    tc, mc = cfg["train"], cfg["model"]
    torch.set_num_threads(int(tc.get("threads", 2)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = os.path.join(ROOT, cfg["data"]["out"])
    vocab = json.load(open(os.path.join(dd, "vocab.json")))["vocab"]
    tr = np.fromfile(os.path.join(dd, "train.bin"), dtype=np.uint16 if vocab < 65536 else np.uint32)
    va = np.fromfile(os.path.join(dd, "val.bin"), dtype=np.uint16 if vocab < 65536 else np.uint32)
    print(f"tokens train={len(tr)/1e6:.2f}M val={len(va)/1e6:.2f}M vocab={vocab} dev={dev}", flush=True)

    data = {"train": torch.from_numpy(tr.astype(np.int64)), "val": torch.from_numpy(va.astype(np.int64))}
    if len(tr) < 8 or len(va) < 8:
        raise SystemExit("dataset train/val insuficiente; ejecuta ./run.sh data y reconstruye el corpus")
    T = min(mc["block_size"], int(tr.shape[0] / tc["batch_size"] / 4))
    if T < 2 or len(va) <= T + 1:
        raise SystemExit("secuencia o validación demasiado pequeñas para entrenar")
    model = NexusLM(vocab=vocab, block=T, d=mc["d_model"], n_layers=mc["n_layers"],
                    n_heads=mc["n_heads"], n_kv=mc.get("n_kv", 4),
                    ff=int(tc.get("ff", mc["ff_mult"]) * mc["d_model"]) // 32 * 32,
                    drop=mc["dropout"])
    print(f"modelo: {model.n_params/1e6:.2f}M params, T={T}, ff={model.cfg['ff']}", flush=True)
    model.to(dev)

    # grupos de optimizacion: decaimiento solo a matrices 2D
    dec = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nod = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": tc["weight_decay"]},
                             {"params": nod, "weight_decay": 0.0}], lr=tc["lr"],
                            betas=(tc["beta1"], tc["beta2"]))
    ckpt_path = os.path.join(ROOT, tc["ckpt"])
    start, best = 0, float("inf")
    if a.resume and not a.no_resume and os.path.exists(ckpt_path):
        d = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if d.get("vocab") == vocab and all(d.get("cfg", {}).get(key) == value
                                                for key, value in model.cfg.items()):
            incompatible = model.load_state_dict(d["model"], strict=False)
            allowed_missing = {key for key in incompatible.missing_keys if key.endswith(".w")}
            if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
                raise SystemExit("checkpoint tiene pesos incompatibles")
            try:
                opt.load_state_dict(d["opt"])
            except (ValueError, KeyError, RuntimeError):
                print("checkpoint de optimizador anterior; reinicio el estado de AdamW", flush=True)
            start = int(d["step"]) + 1
            best = d.get("best", best)
            print(f"reanudando desde step {start} (val {d.get('losses',{}).get('val',float('nan')):.3f})", flush=True)
        else:
            print("checkpoint incompatible: empiezo de cero", flush=True)

    if a.steps is not None and a.additional_steps is not None:
        raise SystemExit("usa solo uno entre --steps y --additional-steps")
    if a.additional_steps is not None and a.additional_steps < 1:
        raise SystemExit("--additional-steps debe ser positivo")
    total = start + a.additional_steps if a.additional_steps is not None else (a.steps or tc["steps"])
    steps = total - start
    if steps <= 0:
        print("ya entrenado hasta", total); return
    gen = torch.Generator().manual_seed(1337)
    log = open(os.path.join(ROOT, tc["log_file"]), "a")
    warm, lrm, lrd, Tt = tc["warmup"], tc["lr"], tc["min_lr"], total
    def lr_at(s):
        if s < warm: return lrm * s / max(1, warm)
        r = (s - warm) / max(1, Tt - warm)
        return lrd + (lrm - lrd) * 0.5 * (1 + math.cos(math.pi * min(1.0, r)))
    B = tc["batch_size"]
    t0 = time.time(); tstep = t0; toks = 0; done = 0
    model.train()
    for s in range(start, total):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, y = get_batch(data["train"], T, B, dev, gen)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
        toks += x.numel(); done += 1
        if (s + 1) % tc["log_every"] == 0:
            dt = time.time() - tstep; tstep = time.time()
            sps = tc["log_every"] / max(dt, 1e-6)
            print(f"step {s+1}/{total}  loss {float(loss.detach()):.3f}  {dt:.1f}s/{tc['log_every']}  "
                  f"{toks/max(time.time()-t0,1)/1e3:.0f}k tok/s  lr {lr_at(s):.2e}", flush=True)
            toks = 0; t0 = time.time()
        if (s + 1) % tc["eval_every"] == 0 or s + 1 == total:
            ev = estimate_loss(model, data, T, B, tc["eval_batches"], dev, gen)
            print(f"  eval val {ev['val']:.4f}  ({math.exp(ev['val']):.1f} ppl)  train {ev['train']:.4f}", flush=True)
            log.write(json.dumps({"step": s + 1, **ev}) + "\n"); log.flush()
            if ev["val"] < best: best = ev["val"]
            model.train()
        if (s + 1) % 150 == 0 or s + 1 == total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s,
                        "cfg": model.cfg, "vocab": vocab, "losses": ev if (s + 1) % tc["eval_every"] == 0 else {},
                        "best": best}, ckpt_path)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": total - 1,
                "cfg": model.cfg, "vocab": vocab, "losses": {}, "best": best}, ckpt_path)
    print(f"entrenamiento terminado. {total} pasos en {(time.time()-t0)/60:.1f} min. ckpt -> {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
