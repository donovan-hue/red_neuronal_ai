"""Benchmark de ablaciones con brazos explícitos y memoria aislada por ejecución."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import synth
from agent import extract_answer, norm_ans

ALIASES = {"lm": "rawlm", "cot": "lm_cot", "agent": "tools"}
VALID_ARMS = {"rawlm", "lm_cot", "tools", "tools+router"}


def items(n_per_task, seed, tasks):
    rng = random.Random(seed)
    out = []
    for name in tasks:
        if name not in synth.GENS:
            raise ValueError(f"tarea desconocida: {name}")
        for _ in range(n_per_task):
            q, answer = synth.GENS[name](rng)
            out.append({"task": name, "question": q, "answer": answer})
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    parser.add_argument("--tasks", default="arith,mixed_expr,linear_eq,percent,rate,sequence,gcdlcm,modulo,digits,convert,compare,prime,logic,extract")
    parser.add_argument("--n", type=int, default=None, help="ítems por familia (por defecto: eval.items_per_task en config.json)")
    parser.add_argument("--seed", type=int, default=990001)
    parser.add_argument("--arms", default="rawlm,lm_cot,tools,tools+router")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    n_per_task = args.n if args.n is not None else int(cfg["eval"].get("items_per_task", 8))
    if n_per_task < 1 or n_per_task > 1000:
        parser.error("--n debe estar entre 1 y 1000")
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    pool = items(n_per_task, args.seed, tasks)
    arms = [ALIASES.get(name.strip(), name.strip()) for name in args.arms.split(",") if name.strip()]
    invalid = sorted(set(arms) - VALID_ARMS)
    if invalid:
        parser.error("brazos no reconocidos: " + ", ".join(invalid))
    if not arms:
        parser.error("debe especificarse al menos un brazo")

    import engine as E
    eng = E.make_engine(cfg)
    print("motor:", eng.info())
    results = {}
    for arm in arms:
        started = time.time()
        correct = 0
        per_task = {}
        agent = None
        if arm in {"tools", "tools+router"}:
            import agent as AG
            import rag
            isolated_cfg = copy.deepcopy(cfg)
            isolated_cfg["agent"]["memory_db"] = ""
            isolated_cfg["agent"]["trajectory_exemplars"] = 0
            memory = AG.T.Memory(":memory:")
            agent = AG.Agent(isolated_cfg, eng, index=rag.get_index(cfg), memory=memory)

        for item in pool:
            question = item["question"]
            try:
                if arm in {"rawlm", "lm_cot"}:
                    content = question
                    if arm == "lm_cot":
                        content += "\nRazona paso a paso brevemente y termina con 'Respuesta final: <dato>'."
                    messages = [{"role": "user", "content": content},
                                {"role": "assistant", "content": ""}]
                    prompt = E.render_prompt(messages[:-1])
                    output = eng.complete(prompt, messages=messages[:-1], max_new=160, temperature=0.2)
                    answer = extract_answer(output)
                else:
                    router = arm == "tools+router"
                    answer = agent.solve(question, tools=True, router=router, k=1)["answer"]
                good = norm_ans(answer) == norm_ans(item["answer"])
            except Exception as exc:
                good = False
                answer = f"[error {type(exc).__name__}: {exc}]"
            correct += int(good)
            counts = per_task.setdefault(item["task"], [0, 0])
            counts[0] += int(good)
            counts[1] += 1
        acc = correct / len(pool)
        results[arm] = {"acc": acc, "ok": correct, "n": len(pool),
                        "secs": round(time.time() - started, 1),
                        "by_task": {task: f"{value[0]}/{value[1]}"
                                    for task, value in sorted(per_task.items())}}
        print(f"[{arm:16s}] acc {acc*100:5.1f}% ({correct}/{len(pool)}) {results[arm]['secs']}s")

    output_path = os.path.join(ROOT, cfg["eval"]["report"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"engine": eng.info(), "suite": cfg["eval"]["suite"], "seed": args.seed,
                   "n": len(pool), "tasks": tasks, "results": results}, f, indent=2, ensure_ascii=False)
    print("reporte ->", output_path)


if __name__ == "__main__":
    main()
