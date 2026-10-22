"""nexus-bench-v1: evaluacion automatica con respuestas verificables por programa.

Compara brazos (ablations) para que la mejora sea medible y no rhetorica:
  lm        modelo base, sin nada
  lm+cot    cadena de pensamiento
  lm+tools  agente con calculadora/python/busqueda
  lm+tools+verify  + autoconsistencia y verificador
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import synth
from agent import norm_ans, extract_answer

def items(n_per_task, seed):
    import synth as S
    rng = random.Random(seed)
    out = []
    for name in S.GENS:
        for _ in range(n_per_task):
            q, a = S.GENS[name](rng)
            out.append({"task": name, "question": q, "answer": a})
    rng.shuffle(out)
    return out

def grade(pred, gold):
    return norm_ans(extract_answer(pred)) == norm_ans(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--tasks", default="arith,mixed_expr,linear_eq,percent,rate,sequence,gcdlcm,modulo,digits,convert,compare,prime,logic,extract")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=990001, help="semilla distinta a la de entrenamiento")
    ap.add_argument("--arms", default="lm,lm_cot,agent")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    import engine as E
    eng = E.make_engine(cfg)
    print("motor:", eng.info())

    rng = random.Random(a.seed)
    keep = set(a.tasks.split(","))
    pool = [it for it in items(a.n, a.seed) if it["task"] in keep]
    print(f"items: {len(pool)}  brazos: {a.arms}")

    def lm_answer(q, cot=False):
        prompt = f"<|user|>\n{q}<|end|>\n<|assistant|>\n" if not cot else \
                 f"<|user|>\n{q}\nRazona paso a paso breve y cierra con 'Respuesta: x'.<|end|>\n<|assistant|>\n"
        out = eng.complete(prompt, max_new=160, temperature=0.2)
        return out or ""

    results = {}
    for arm in a.arms.split(","):
        t0 = time.time(); ok = 0; per = {}
        for it in pool:
            try:
                if arm == "rawlm":
                    ans = extract_answer(lm_answer(it["question"], False))
                elif arm == "lm_cot":
                    ans = extract_answer(lm_answer(it["question"], True))
                else:
                    import agent as AG, rag
                    idx = rag.get_index(cfg)
                    mem = AG.T.Memory(os.path.join(ROOT, cfg["agent"]["memory_db"]))
                    mem = AG.T.Memory(os.path.join(ROOT, cfg["agent"]["memory_db"]))
                    ag = AG.Agent(cfg, eng, index=idx, memory=mem)
                    if arm.startswith("rawlm"):
                        ans = ag.solve(it["question"], tools=False)["answer"]
                    else:
                        ans = ag.solve(it["question"], tools=True,
                                       router=False if arm == "tools" else (True if arm == "tools+router" else None),
                                       k=cfg["agent"].get("self_consistency", 1))["answer"]
                good = norm_ans(ans) == norm_ans(it["answer"])
            except Exception as e:
                good = False; ans = f"[error {type(e).__name__}: {e}]"
            ok += good
            per.setdefault(it["task"], [0, 0]); per[it["task"]][0] += good; per[it["task"]][1] += 1
        acc = ok / max(1, len(pool))
        results[arm] = {"acc": acc, "ok": ok, "n": len(pool), "secs": round(time.time() - t0, 1),
                        "by_task": {t: f"{v[0]}/{v[1]}" for t, v in sorted(per.items())}}
        print(f"[{arm:16s}] acc {acc*100:5.1f}%  ({ok}/{len(pool)})  {results[arm]['secs']}s")
        print("      ", "  ".join(f"{t}:{v}" for t, v in results[arm]["by_task"].items()))
    out = os.path.join(ROOT, cfg["eval"]["report"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"engine": eng.info(), "suite": cfg["eval"]["suite"], "seed": a.seed,
               "n": len(pool), "results": results}, open(out, "w"), indent=2, ensure_ascii=False)
    print("reporte ->", out)

if __name__ == "__main__":
    main()
