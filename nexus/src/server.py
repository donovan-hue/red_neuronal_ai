"""API HTTP + streaming SSE para el producto. Solo libreria estandar (sin frameworks)."""
from __future__ import annotations
import json, os, queue, re, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
WEB = os.path.join(ROOT, "web")

CFG_PATH = os.path.join(ROOT, "config.json")
_STATE = {"cfg": None, "engine": None, "agent": None, "index": None, "err": None}
_LOCK = threading.Lock()

def load_cfg():
    return json.load(open(CFG_PATH))

def boot(force=False):
    with _LOCK:
        if _STATE["agent"] and not force:
            return _STATE["agent"]
        import agent as AG, rag, engine as E
        cfg = load_cfg(); _STATE["cfg"] = cfg
        _STATE["index"] = rag.get_index(cfg)
        try:
            eng = E.make_engine(cfg)
            _STATE["err"] = None
        except Exception as e:
            eng = None
            _STATE["err"] = f"{type(e).__name__}: {e}"
        mem = AG.T.Memory(os.path.join(ROOT, cfg["agent"]["memory_db"]))
        _STATE["engine"] = eng
        _STATE["agent"] = AG.Agent(cfg, eng, index=_STATE["index"], memory=mem) if eng else None
        return _STATE["agent"]

def status():
    cfg = _STATE["cfg"] or load_cfg()
    eng = _STATE["engine"]
    tr = []
    lp = os.path.join(ROOT, cfg["train"]["log_file"])
    if os.path.exists(lp):
        for line in open(lp):
            try: tr.append(json.loads(line))
            except Exception: pass
    run = {"steps_done": len(tr), "last": tr[-1] if tr else None,
           "curve": tr[:: max(1, len(tr) // 60)]}
    if os.path.exists(lp):
        try: run["tail"] = open(lp).read().strip().splitlines()[-1][:160]
        except Exception: pass
    out = {"provider": cfg.get("provider"), "engine": eng.info() if eng else None,
           "error": _STATE["err"], "docs_indexed": len(_STATE["index"]) if _STATE["index"] else 0,
           "config": {"model": cfg["model"], "agent": cfg["agent"]}, "train": run}
    rep = os.path.join(ROOT, cfg["eval"]["report"])
    out["eval"] = json.load(open(rep)) if os.path.exists(rep) else None
    return out

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a): pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/status":
            try: self._send(200, status())
            except Exception as e: self._send(500, {"error": str(e)})
        elif p == "/api/config":
            self._send(200, load_cfg())
        elif p in ("/", "/index.html"):
            self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), "text/html")
        elif p.startswith("/api/download/"):
            f = os.path.basename(p)
            fp = os.path.join(ROOT, "runs", f)
            if os.path.exists(fp):
                self._send(200, open(fp, "rb").read(), "application/octet-stream",
                           extra={"Content-Disposition": f'attachment; filename="{f}"'})
            else:
                self._send(404, {"e": "no existe"})
        elif p.startswith("/api/run/"):
            f = os.path.basename(p)
            fp = os.path.join(ROOT, "runs", f)
            self._send(200, open(fp, "rb").read(), "text/plain") if os.path.exists(fp) else self._send(404, {"e": "no existe"})
        else:
            fp = os.path.join(WEB, p.lstrip("/"))
            if os.path.isfile(fp):
                ct = "text/css" if fp.endswith(".css") else "application/javascript" if fp.endswith(".js") else "text/html"
                self._send(200, open(fp, "rb").read(), ct)
            else:
                self._send(404, {"error": "404"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        p = self.path.split("?")[0]
        if p == "/api/chat":
            self.chat(req)
        elif p == "/api/config":
            cfg = load_cfg()
            for k, v in (req or {}).items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
                else: cfg[k] = v
            json.dump(cfg, open(CFG_PATH, "w"), indent=2, ensure_ascii=False)
            boot(force=True)
            self._send(200, {"ok": True, "config": cfg})
        elif p == "/api/eval":
            self._send(200, {"started": True})
            threading.Thread(target=_run_eval, daemon=True).start()
        elif p == "/api/train":
            self._send(200, {"started": True})
            threading.Thread(target=_run_train, daemon=True).start()
        else:
            self._send(404, {"error": "404"})

    def chat(self, req):
        q = (req.get("message") or "").strip()
        hist = req.get("history") or []
        if not q:
            return self._send(400, {"error": "mensaje vacio"})
        ag = boot()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def ev(obj):
            try:
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                raise
        try:
            if _STATE["agent"] is None:
                ev({"type": "error", "message": "Sin motor disponible: " + str(_STATE["err"])}); return
            ev({"type": "start", "engine": _STATE["engine"].info()})
            res = _STATE["agent"].solve(q, history=hist, k=req.get("k"),
                                       emit=lambda e: ev({"type": "tool" if e["type"] == "action" else e["type"],
                                                          **{k: v for k, v in e.items() if k != "type"}}))
            ans = res["answer"]
            for i in range(0, len(ans), 28):
                ev({"type": "token", "text": ans[i:i + 28]}); time.sleep(0.015)
            ev({"type": "done", "answer": ans, "trace": res["trace"], "sources": res.get("sources", []),
                "guarded": res.get("guarded", False), "verify_ok": res["verify_ok"],
                "verify_note": res["verify_note"], "votes": res.get("votes"), "k": res.get("k"),
                "candidates": res.get("candidates", [])})
        except Exception as e:
            ev({"type": "error", "message": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]})

_T = {}
def _run_eval():
    import subprocess
    if _T.get("eval"): return
    _T["eval"] = True
    try:
        with open(os.path.join(ROOT, "runs", "eval.log"), "w") as f:
            subprocess.run([sys.executable, os.path.join(ROOT, "src", "eval.py")],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, timeout=1800)
    except Exception as e:
        open(os.path.join(ROOT, "runs", "eval.log"), "a").write(f"error {e}")
    finally:
        _T["eval"] = False

def _run_train():
    import subprocess
    if _T.get("train"): return
    _T["train"] = True
    try:
        with open(os.path.join(ROOT, "runs", "train.log"), "a") as f:
            subprocess.run([sys.executable, "-u", os.path.join(ROOT, "src", "train.py")],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    finally:
        _T["train"] = False

def main():
    cfg = load_cfg()
    port = int(os.environ.get("PORT", cfg["server"]["port"]))
    threading.Thread(target=boot, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"NEXUS en http://0.0.0.0:{port}", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
