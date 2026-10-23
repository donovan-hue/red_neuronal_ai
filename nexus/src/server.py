"""API HTTP de NEXUS. Las operaciones administrativas requieren NEXUS_ADMIN_TOKEN."""
from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WEB = (ROOT / "web").resolve()
CFG_PATH = ROOT / "config.json"
MAX_BODY = 1_000_000
_STATE = {"cfg": None, "engine": None, "agent": None, "index": None, "err": None}
_BOOT_LOCK = threading.RLock()
_TASK_LOCK = threading.Lock()
_TASKS = {"eval": False, "train": False}
_RATE_LOCK = threading.Lock()
_RATE = {}


def load_cfg():
    with CFG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def boot(force=False):
    with _BOOT_LOCK:
        if _STATE["agent"] and not force:
            return _STATE["agent"]
        import agent as AG
        import engine as E
        import rag
        cfg = load_cfg()
        index = rag.get_index(cfg)
        try:
            eng = E.make_engine(cfg)
            error = None
        except Exception as exc:
            eng = None
            error = f"{type(exc).__name__}: {exc}"
        memory_path = ROOT / cfg["agent"]["memory_db"]
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        mem = AG.T.Memory(str(memory_path))
        _STATE.update(cfg=cfg, engine=eng, agent=AG.Agent(cfg, eng, index=index, memory=mem) if eng else None,
                      index=index, err=error)
        return _STATE["agent"]


def status():
    cfg = _STATE["cfg"] or load_cfg()
    eng = _STATE["engine"]
    log_path = ROOT / cfg["train"]["log_file"]
    records = []
    if log_path.is_file():
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                except json.JSONDecodeError:
                    continue
    steps = max((int(row.get("step", 0)) for row in records), default=0)
    train = {"steps_done": steps, "last": records[-1] if records else None,
             "curve": records[::max(1, len(records) // 60)]}
    report_path = ROOT / cfg["eval"]["report"]
    report = None
    if report_path.is_file():
        try:
            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            report = None
    return {"provider": cfg.get("provider"), "engine": eng.info() if eng else None,
            "error": _STATE["err"], "docs_indexed": len(_STATE["index"]) if _STATE["index"] else 0,
            "config": {"model": cfg["model"], "agent": cfg["agent"]}, "train": train,
            "eval": report, "admin_enabled": bool(os.environ.get("NEXUS_ADMIN_TOKEN")),
            "api_token_configured": bool(os.environ.get("NEXUS_API_TOKEN"))}


def _authorized(handler):
    expected = os.environ.get("NEXUS_ADMIN_TOKEN", "")
    supplied = handler.headers.get("X-Nexus-Admin-Token", "")
    return bool(expected) and hmac.compare_digest(supplied, expected)


def _rate_allowed(address, limit):
    now = time.monotonic()
    with _RATE_LOCK:
        calls = [stamp for stamp in _RATE.get(address, []) if now - stamp < 60]
        if len(calls) >= limit:
            _RATE[address] = calls
            return False
        calls.append(now)
        _RATE[address] = calls
        return True


def _json_body(handler):
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError:
        raise ValueError("Content-Length inválido") from None
    if length < 0 or length > MAX_BODY:
        raise ValueError("solicitud demasiado grande")
    try:
        value = json.loads(handler.rfile.read(length) or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("JSON inválido") from None
    if not isinstance(value, dict):
        raise ValueError("se esperaba un objeto JSON")
    return value


class Handler(BaseHTTPRequestHandler):
    server_version = "NEXUS/1.0"

    def log_message(self, *_args):
        return

    def _send(self, code, body, content_type="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _safe_file(self, base, relative):
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            try:
                self._send(200, status())
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        if path == "/api/config":
            cfg = load_cfg()
            self._send(200, cfg)
            return
        if path in ("/", "/index.html"):
            target = WEB / "index.html"
        elif path.startswith("/api/download/") or path.startswith("/api/run/"):
            prefix = "/api/download/" if path.startswith("/api/download/") else "/api/run/"
            name = path[len(prefix):]
            if not name or Path(name).name != name:
                self._send(400, {"error": "ruta inválida"})
                return
            target = self._safe_file(ROOT / "runs", name)
            if target is None:
                self._send(400, {"error": "ruta inválida"})
                return
            if not target.is_file():
                self._send(404, {"error": "no existe"})
                return
            content_type = "application/octet-stream" if prefix.endswith("download/") else "text/plain"
            extra = {"Content-Disposition": f'attachment; filename="{name}"'} if prefix.endswith("download/") else None
            self._send(200, target.read_bytes(), content_type, extra)
            return
        else:
            relative = path.lstrip("/")
            target = self._safe_file(WEB, relative)
            if target is None:
                self._send(404, {"error": "404"})
                return
        if target.is_file():
            suffix = target.suffix.lower()
            content_type = {".css": "text/css", ".js": "application/javascript",
                            ".html": "text/html"}.get(suffix, "application/octet-stream")
            self._send(200, target.read_bytes(), content_type)
        else:
            self._send(404, {"error": "404"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            req = _json_body(self)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        if path == "/api/chat":
            self.chat(req)
        elif path in ("/api/train", "/api/eval"):
            if not _authorized(self):
                self._send(403, {"error": "operación administrativa deshabilitada o token inválido"})
                return
            if path == "/api/train":
                additional = req.get("additional_steps", 300)
                if isinstance(additional, bool) or not isinstance(additional, int) or not 1 <= additional <= 100_000:
                    self._send(400, {"error": "additional_steps debe estar entre 1 y 100000"})
                    return
                started = _start_task("train", additional)
            else:
                started = _start_task("eval", None)
            self._send(202 if started else 409, {"started": started})
        elif path == "/api/config":
            self._send(405, {"error": "configuración de solo lectura; edite config.json fuera del servidor"},
                       extra={"Allow": "GET"})
        else:
            self._send(404, {"error": "404"})

    def chat(self, req):
        q = req.get("message")
        history = req.get("history", [])
        if not isinstance(q, str) or not q.strip() or len(q) > 8000:
            self._send(400, {"error": "message debe ser texto de 1 a 8000 caracteres"})
            return
        if not isinstance(history, list) or len(history) > 12:
            self._send(400, {"error": "history debe ser una lista de hasta 12 mensajes"})
            return
        requested_k = req.get("k")
        if requested_k is not None and (isinstance(requested_k, bool) or not isinstance(requested_k, int)
                                        or not 1 <= requested_k <= 10):
            self._send(400, {"error": "k debe ser un entero entre 1 y 10"})
            return
        clean_history = []
        for item in history:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant") or not isinstance(item.get("content"), str):
                self._send(400, {"error": "mensaje de historial inválido"})
                return
            clean_history.append({"role": item["role"], "content": item["content"][:4000]})
        try:
            agent = boot()
        except Exception as exc:
            self._send(503, {"error": f"no se pudo iniciar NEXUS: {type(exc).__name__}: {exc}"})
            return
        if _STATE["engine"] and _STATE["engine"].kind == "api":
            expected = os.environ.get("NEXUS_API_TOKEN", "")
            if not expected:
                self._send(503, {"error": "para exponer un proveedor de pago configura NEXUS_API_TOKEN"})
                return
            supplied = self.headers.get("X-Nexus-Api-Token", "")
            if not hmac.compare_digest(supplied, expected):
                self._send(401, {"error": "token de API inválido"})
                return
        rate_limit = 10 if _STATE["engine"] and _STATE["engine"].kind == "api" else 30
        if not _rate_allowed(self.client_address[0], rate_limit):
            self._send(429, {"error": "límite temporal de solicitudes excedido"},
                       extra={"Retry-After": "60"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        def emit(event):
            data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()

        try:
            if agent is None:
                emit({"type": "error", "message": "Sin motor disponible: " + str(_STATE["err"])})
                return
            emit({"type": "start", "engine": _STATE["engine"].info()})
            result = agent.solve(q.strip(), history=clean_history,
                                 k=req.get("k"), emit=lambda event: emit({
                                     "type": "tool" if event["type"] == "action" else event["type"],
                                     **{key: val for key, val in event.items() if key != "type"}}))
            answer = result["answer"]
            # No es generación token a token, pero conserva el contrato SSE de la interfaz.
            for offset in range(0, len(answer), 28):
                emit({"type": "token", "text": answer[offset:offset + 28]})
            emit({"type": "done", "answer": answer, "trace": result["trace"],
                  "sources": result.get("sources", []), "guarded": result.get("guarded", False),
                  "verify_ok": result["verify_ok"], "verify_note": result["verify_note"],
                  "votes": result.get("votes"), "k": result.get("k"),
                  "candidates": result.get("candidates", [])})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                      "tb": traceback.format_exc()[-800:]})
            except OSError:
                pass


def _start_task(name, argument):
    with _TASK_LOCK:
        if any(_TASKS.values()):
            return False
        _TASKS[name] = True
    threading.Thread(target=_run_task, args=(name, argument), daemon=True).start()
    return True


def _run_task(name, argument):
    try:
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "src" / ("eval.py" if name == "eval" else "train.py"))]
        if name == "train":
            command.extend(["--additional-steps", str(argument)])
        log_path = ROOT / "runs" / f"{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                           timeout=1800 if name == "eval" else 86_400, check=False)
    except Exception as exc:
        with (ROOT / "runs" / f"{name}.log").open("a", encoding="utf-8") as log:
            log.write(f"\nerror {type(exc).__name__}: {exc}\n")
    finally:
        with _TASK_LOCK:
            _TASKS[name] = False


def main():
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    port = int(os.environ.get("PORT", cfg["server"]["port"]))
    threading.Thread(target=boot, daemon=True).start()
    server = ThreadingHTTPServer((cfg["server"].get("host", "0.0.0.0"), port), Handler)
    server.daemon_threads = True
    print(f"NEXUS escuchando en puerto {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
