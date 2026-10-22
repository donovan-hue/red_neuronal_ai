"""Cliente CLI del servidor (util para pruebas y para el usuario)."""
import json, sys, urllib.request
msg = " ".join(sys.argv[1:-1]) if sys.argv[-1] == "--k1" else " ".join(sys.argv[1:])
k = 1 if "--k1" in sys.argv else 3
req = urllib.request.Request("http://127.0.0.1:8080/api/chat",
        data=json.dumps({"message": msg, "k": k}).encode(),
        headers={"Content-Type": "application/json"})
ev = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"): continue
        try: e = json.loads(line[5:])
        except Exception: continue
        t = e.get("type")
        if t == "route": print(f"  [ruta] el modelo prefiere '{e['prefers']}' (margen {e['margin']})")
        elif t in ("action","tool"): print(f"  [herramienta] {e['tool']} {json.dumps(e.get('args',{}),ensure_ascii=False)[:90]}\n     -> {str(e.get('obs',''))[:220]}")
        elif t == "done":
            print(f"  ANSWER: {e['answer']!r}\n  verify_ok={e.get('verify_ok')} votes={e.get('votes')}/{e.get('k')} nota={str(e.get('verify_note'))[:120]}")
        elif t == "error": print("  ERROR:", e.get("message"))
