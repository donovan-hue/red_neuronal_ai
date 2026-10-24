#!/usr/bin/env python3
"""Panel local del checklist de 74 verificaciones. No usa servicios externos."""
from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
ITEMS = json.loads((APP / "checklist_74.json").read_text(encoding="utf-8"))
STATE_FILE = APP / "progress.json"
TESTS = {
    "test_model_checkpoint": "ModelTests.test_committed_checkpoint_is_loadable",
    "test_model_roundtrip": "ModelTests.test_npz_round_trip_without_pickle",
    "test_api_health": "ApiTests.test_health_and_prediction",
    "test_api_invalid": "ApiTests.test_invalid_training_requests_are_rejected",
    "test_api_nonjson": "ApiTests.test_non_json_is_rejected",
    "test_api_token": "ApiTests.test_training_requires_token_and_persists_safe_checkpoint",
    "test_calc": "CalculatorTests.test_valid_arithmetic_including_zero_operands",
    "test_calc_reject": "CalculatorTests.test_calculator_rejects_silently_mutated_expressions",
    "test_python_disabled": "CalculatorTests.test_python_tool_is_disabled",
    "test_memory": "MemoryTests.test_recall_filters_by_query",
    "test_agent_guard": "AgentTests.test_guard_corrects_and_verifies_model_arithmetic",
    "test_roles": "EngineTests.test_rendered_prompt_preserves_structured_roles",
    "test_bm25": "EngineTests.test_cached_bm25_round_trip",
    "test_server_path": "ServerPathTests.test_static_path_cannot_escape_web_root",
    "test_server_error": "ServerPathTests.test_provider_error_details_are_not_exposed",
    "test_single_nl": "SingleFileTests.test_natural_language_numeric_answer",
    "test_single_html": "SingleFileTests.test_generated_html_uses_text_content_not_untrusted_inner_html",
}
CHECK_IDS = list(TESTS) + [
    "compile_python", "shell_syntax", "check_shell", "js_syntax", "check_diff",
    "nexus_help", "source_contracts", "local_only", "nexus_runtime", "sse_error_exposure", "dashboard_self",
]


def initial_state():
    return {
        "running": False,
        "started": None,
        "finished": None,
        "current": None,
        "done": 0,
        "total": len(CHECK_IDS),
        "checks": {key: {"status": "pending", "message": "Sin ejecutar", "duration": None}
                   for key in CHECK_IDS},
        "items": {str(item["id"]): {"status": "pending", "evidence": "Sin ejecutar"}
                  for item in ITEMS},
    }


def load_state():
    try:
        old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = initial_state()
        if old.get("schema") == 2 and set(old.get("items", {})) == set(state["items"]):
            state.update({k: old[k] for k in ("started", "finished", "checks", "items", "done") if k in old})
            for key in CHECK_IDS:
                if key not in state["checks"]:
                    state["checks"][key] = {"status": "pending", "message": "Sin ejecutar", "duration": None}
            state["total"] = len(CHECK_IDS)
        return state
    except (OSError, ValueError, TypeError):
        return initial_state()


STATE = load_state()
LOCK = threading.RLock()


def save_state():
    data = {"schema": 2, **{key: STATE[key] for key in
                             ("started", "finished", "done", "checks", "items")}}
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def run_cmd(argv, *, cwd=ROOT, timeout=90, env=None):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, env=env, check=False)
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output[-5000:] or ("salida vacía" if result.returncode else "OK")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return False, f"Tiempo agotado ({timeout}s). {out[-3000:]}"
    except OSError as exc:
        return False, f"No se pudo ejecutar: {type(exc).__name__}: {exc}"


def run_one(key):
    if key in TESTS:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "nexus" / "src"), str(ROOT / "tests")])
        return run_cmd([sys.executable, "-m", "unittest", "-v", f"test_core.{TESTS[key]}"],
                       cwd=ROOT / "tests", timeout=45, env=env)

    if key == "compile_python":
        files = [p for p in ROOT.rglob("*.py")
                 if not any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in p.parts)]
        for path in files:
            try:
                py_compile.compile(str(path), doraise=True)
            except (OSError, py_compile.PyCompileError) as exc:
                return False, f"{path.relative_to(ROOT)}: {exc}"
        return True, f"Sintaxis Python válida en {len(files)} archivos."

    if key == "shell_syntax":
        scripts = sorted(set(ROOT.glob("*.sh")) | set((ROOT / "nexus").glob("*.sh")) |
                         set((ROOT / "nexus" / "src").glob("*.sh")))
        if not scripts:
            return False, "No se encontraron scripts shell."
        for path in scripts:
            ok, output = run_cmd(["bash", "-n", str(path)], timeout=10)
            if not ok:
                return False, f"{path.relative_to(ROOT)}: {output}"
        return True, f"bash -n pasó en {len(scripts)} scripts."

    if key == "check_shell":
        return run_cmd(["bash", "-n", "nexus/get_model.sh"], timeout=10)

    if key == "js_syntax":
        node = shutil.which("node")
        if not node:
            return None, "Node.js no está instalado; no se pudo validar JavaScript."
        cases = [(ROOT / "nexus_single.py", r"<script>(.*?)</script>"),
                 (ROOT / "nexus" / "web" / "index.html", r"<script>(.*?)</script>")]
        for source, pattern in cases:
            text = source.read_text(encoding="utf-8")
            match = re.search(pattern, text, re.S)
            if not match:
                return False, f"No se encontró script inline en {source.relative_to(ROOT)}."
            with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as tmp:
                tmp.write(match.group(1))
                name = tmp.name
            try:
                ok, output = run_cmd([node, "--check", name], timeout=10)
                if not ok:
                    return False, f"{source.relative_to(ROOT)}: {output}"
            finally:
                Path(name).unlink(missing_ok=True)
        return True, "node --check pasó en los scripts de ambas interfaces."

    if key == "check_diff":
        return run_cmd(["git", "diff", "--check"], timeout=10)

    if key == "nexus_help":
        return run_cmd(["bash", "./run.sh", "help"], cwd=ROOT / "nexus", timeout=10)

    if key == "source_contracts":
        agent = (ROOT / "nexus/src/agent.py").read_text(encoding="utf-8")
        single = (ROOT / "nexus_single.py").read_text(encoding="utf-8")
        eval_src = (ROOT / "nexus/src/eval.py").read_text(encoding="utf-8")
        config = json.loads((ROOT / "nexus/config.json").read_text(encoding="utf-8"))
        tools_src = (ROOT / "nexus/src/tools.py").read_text(encoding="utf-8")
        server = (ROOT / "nexus/src/server.py").read_text(encoding="utf-8")
        engine = (ROOT / "nexus/src/engine.py").read_text(encoding="utf-8")
        corpus = (ROOT / "nexus/src/corpus.py").read_text(encoding="utf-8")
        contracts = (ROOT / "nexus/src/engine.py").read_text(encoding="utf-8")
        release_script = (ROOT / "nexus/get_model.sh").read_text(encoding="utf-8")
        checks = {
            "verificación limitada a guard/calc": 't.get("kind") == "guard" and t.get("tool") == "calc"' in agent,
            "crítico requiere PASS explícito": r"\bPASS\b" in agent,
            "resultado guarded visible": '"guarded": chosen.get("guarded", False)' in agent,
            "pregunta/contexto separados por roles": '{"role": "user", "content": user_content}' in single,
            "contexto RAG codificado como JSON": 'json.dumps([{"src": h.get("src", "?"), "text": h["text"][:400]} for h in hits]' in single,
            "memoria de benchmark aislada": 'AG.T.Memory(":memory:")' in eval_src,
            "tamaño benchmark configurable": 'cfg["eval"].get("items_per_task", 8)' in eval_src,
            "herramienta Python deshabilitada": 'Ejecución de Python deshabilitada' in tools_src,
            "Gemini usa encabezado de clave": '"x-goog-api-key"' in contracts,
            "API admin comprueba token": 'NEXUS_ADMIN_TOKEN' in server and 'hmac.compare_digest' in server,
            "huella de datos implementada": 'dataset_fingerprint' in corpus and 'fingerprint' in corpus,
            "cambio de dataset protege checkpoint": 'checkpoint) and not same_dataset' in corpus,
            "corpus ausente se rechaza claramente": 'no hay documentos de entrenamiento' in corpus,
            "recuperación sin índice prevista": 'sin índice' in single,
            "error del verificador no es éxito": 'return False, f"verificador no disponible:' in agent,
            "límite de pasos de consistencia": '1 <= k <= 10' in agent,
            "memoria guarda el resultado verificado real": 'self.memory.log_trace(q, chosen["answer"], bool(ok)' in agent,
            "proveedor API no fabrica puntuaciones": 'no exponen log-probabilidades' in contracts and 'return []' in contracts,
            "stream usa MIME SSE": 'text/event-stream; charset=utf-8' in server,
            "límite de peticiones implementado": 'def rate_allowed' in single and 'def _rate_allowed' in server,
            "trazas guardadas con estado verificado real": 'self.memory.log_trace(q, chosen["answer"], bool(ok)' in agent,
            "autenticación Gemini sin query string": '"x-goog-api-key"' in contracts and not re.search(r"[?&]key=", contracts),
            "eventos SSE llevan delimitador doble": 'data = f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"' in server,
            "la memoria del CLI se cierra": 'memory.close()' in single,
            "el agente trata datos externos como no confiables": 'nunca sigas instrucciones' in agent,
            "release valida IDs y tamaño del vocabulario": 'vocab_size = max(ids) + 1' in release_script,
            "release verifica los hashes de artefactos": 'hashlib.sha256((root / name).read_bytes()).hexdigest()' in release_script,
        }
        failed = [name for name, ok in checks.items() if not ok]
        return (not failed, f"{len(checks) - len(failed)}/{len(checks)} contratos de código presentes." +
                (" Faltan: " + "; ".join(failed) if failed else ""))

    if key == "sse_error_exposure":
        source = (ROOT / "nexus/src/server.py").read_text(encoding="utf-8")
        leaks_traceback = '"tb": traceback.format_exc()' in source
        return (not leaks_traceback,
                "El endpoint SSE entrega traceback interno al cliente." if leaks_traceback else
                "No se encontró traceback interno en el contrato SSE.")

    if key == "local_only":
        html = (APP / "index.html").read_text(encoding="utf-8")
        import ast
        server_src = Path(__file__).read_text(encoding="utf-8")
        imports = set()
        for node in ast.walk(ast.parse(server_src)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        external_fetch = re.search(r"fetch\s*\(\s*['\"]https?://", html)
        if external_fetch or imports.intersection({"requests", "httpx", "urllib"}):
            return False, "Se detectó una llamada de red externa en el panel."
        return True, "El panel y su runner usan exclusivamente archivos y procesos locales; no suben mensajes ni archivos."

    if key == "nexus_runtime":
        missing = []
        for module in ("torch", "tokenizers"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)
        if missing:
            return None, "Pendiente: faltan dependencias locales " + ", ".join(missing) + "; no se descargan automáticamente."
        return None, "Dependencias presentes. La integración completa no se ejecuta automáticamente desde este panel."

    if key == "dashboard_self":
        ids = [item.get("id") for item in ITEMS]
        if len(ids) != 74 or len(set(ids)) != 74 or set(ids) != set(range(1, 75)):
            return False, "El checklist no contiene exactamente 74 IDs únicos y consecutivos."
        unknown = sorted({item["check"] for item in ITEMS} - set(CHECK_IDS))
        if unknown:
            return False, "Hay validaciones sin implementar: " + ", ".join(unknown)
        return True, "Checklist autocontenido: 74 puntos y todas sus comprobaciones registradas."

    return False, "Identificador de validación desconocido."


def apply_result(key, status, message, duration):
    with LOCK:
        STATE["checks"][key] = {"status": status, "message": message[-5000:], "duration": round(duration, 2)}
        for item in ITEMS:
            if item["check"] == key:
                row = STATE["items"][str(item["id"])]
                row["status"] = status
                row["evidence"] = message[-1000:]
        STATE["done"] += 1
        STATE["current"] = None
        save_state()
        print(f"[checklist] {STATE['done']}/{STATE['total']} {key}: {status}", flush=True)


def run_suite():
    with LOCK:
        STATE["running"] = True
        STATE["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE["finished"] = None
        STATE["current"] = None
        STATE["done"] = 0
        STATE["total"] = len(CHECK_IDS)
        STATE["checks"] = {key: {"status": "pending", "message": "En cola", "duration": None}
                           for key in CHECK_IDS}
        STATE["items"] = {str(item["id"]): {"status": "pending", "evidence": "En cola"}
                          for item in ITEMS}
        save_state()
    for key in CHECK_IDS:
        with LOCK:
            STATE["current"] = key
            STATE["checks"][key] = {"status": "running", "message": "Ejecutando validación local…", "duration": None}
            for item in ITEMS:
                if item["check"] == key:
                    STATE["items"][str(item["id"])] = {"status": "running", "evidence": "Prueba en curso…"}
            save_state()
        start = time.monotonic()
        try:
            ok, message = run_one(key)
            status = "passed" if ok is True else ("blocked" if ok is None else "failed")
        except Exception as exc:
            status, message = "failed", f"{type(exc).__name__}: {exc}"
        apply_result(key, status, message, time.monotonic() - start)
    with LOCK:
        STATE["running"] = False
        STATE["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state()
        print("CHECKLIST_RUN_DONE", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "ChecklistLocal/1.0"

    def _send(self, code, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            with LOCK:
                payload = {**STATE, "items_list": ITEMS}
            self._send(200, payload)
        elif self.path in ("/", "/index.html"):
            self._send(200, (APP / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "no encontrado"})

    def do_POST(self):
        if self.path != "/api/run":
            self._send(404, {"error": "no encontrado"})
            return
        with LOCK:
            if STATE["running"]:
                self._send(409, {"error": "la verificación ya está en curso"})
                return
            thread = threading.Thread(target=run_suite, name="checklist-runner", daemon=True)
            thread.start()
            self._send(202, {"started": True, "total": len(CHECK_IDS)})

    def log_message(self, _format, *_args):
        return


def main():
    host = os.environ.get("CHECKLIST_HOST", "0.0.0.0")
    port = int(os.environ.get("CHECKLIST_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Checklist local listo en http://localhost:{port} (sin llamadas externas)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
