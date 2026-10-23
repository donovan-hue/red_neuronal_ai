"""Herramientas del agente. Cada una es pura, acotada y devuelve texto plano."""
from __future__ import annotations
import ast, datetime as dt, json, math, os, re, sqlite3, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_BIN = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd}

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        l, r = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(r) > 64 or abs(l) > 1e6):
            raise ValueError("exponente demasiado grande")
        if isinstance(node.op, ast.Div) and r == 0: raise ZeroDivisionError
        if isinstance(node.op, ast.FloorDiv) and r == 0: raise ZeroDivisionError
        if isinstance(node.op, ast.Mod) and r == 0: raise ZeroDivisionError
        value = {"Add": lambda: l + r, "Sub": lambda: l - r, "Mult": lambda: l * r,
                 "Div": lambda: l / r, "FloorDiv": lambda: l // r, "Mod": lambda: l % r,
                 "Pow": lambda: l ** r}[type(node.op).__name__]()
        if isinstance(value, int) and value.bit_length() > 256:
            raise ValueError("resultado fuera de rango")
        if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 1e100):
            raise ValueError("resultado fuera de rango")
        return value
    if isinstance(node, ast.UnaryOp) and node.op.__class__ in _BIN:
        v = _safe_eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("expresion no permitida")

def calc(expr, _cfg=None):
    """Evalua aritmetica exacta. Soporta ** // % y parentesis."""
    e = re.sub(r"[xX×]", "*", str(expr).strip()).replace("^", "**")
    if not e or len(e) > 256 or re.search(r"[^\d+\-*/%(). ]", e):
        raise ValueError("expresion vacia, demasiado larga o con caracteres no permitidos")
    tree = ast.parse(e, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("expresion demasiado compleja")
    if any(isinstance(n, ast.Constant) and isinstance(n.value, int) and n.value.bit_length() > 256
           for n in ast.walk(tree)):
        raise ValueError("literal fuera de rango")
    val = _safe_eval(tree)
    if isinstance(val, float) and abs(val - round(val)) < 1e-12:
        val = int(round(val))
    return f"{val}"

NL_RULES = [
    (r"el m[áa]ximo com[úu]n divisor de (\d+) y (\d+)", lambda a, b: f"__gcd({a},{b})"),
    (r"el m[íi]nimo com[úu]n multiplo de (\d+) y (\d+)", lambda a, b: f"({a}*{b})//__gcd({a},{b})"),
    (r"\bmcm\b[^\d]*(\d+)[^\d]*(\d+)", lambda a, b: f"({a}*{b})//__gcd({a},{b})"),
    (r"\bmcd\b[^\d]*(\d+)[^\d]*(\d+)", lambda a, b: f"__gcd({a},{b})"),
    (r"residuo de dividir (\d+) entre (\d+)", lambda a, b: f"{a} % {b}"),
    (r"(\d+(?:\.\d+)?)\s*% de (\d+)", lambda a, b: f"({a}*{b})/100"),
    (r"sumar? las cifras del numero (\d+)", lambda a: " + ".join(str(a))),
    (r"a velocidad de (\d+)[^\d]*(\d+)\s*(h|horas)", lambda a, b: f"{a}*{b}"),
    (r"mantiene (\d+) km/h durante (\d+)", lambda a, b: f"{a}*{b}"),
    (r"convierte (\d+) km a m\b", lambda a: f"{a}*1000"),
    (r"convierte (\d+) h a min", lambda a: f"{a}*60"),
]

def _gcd(a, b):
    while b: a, b = b, a % b
    return a

def calc_nl(text, _cfg=None):
    """Resuelve aritmetica redactada en lenguaje natural a expresion exacta.

    Es un resolvedor determinista del harness (ver docs/ARCHITECTURE.md): no sustituye
    al modelo, sirve para verificar sus cuentas y para darle la operacion correcta.
    """
    t = re.sub(r"\s+", " ", str(text or ""))
    for pat, fn in NL_RULES:
        m = re.search(pat, t, re.I)
        if m:
            e = fn(*m.groups())
            try:
                val = eval(e.replace("__gcd", "_gcd"), {"_gcd": _gcd, "__builtins__": {}}, {})
                if isinstance(val, float) and abs(val - round(val)) < 1e-9:
                    val = int(round(val))
                return f"{val}"
            except Exception:
                continue
    pick = _best_expr(t) or (lambda m: (m.group(1).strip(), m.start()) if m else (None, 0))(NUM_RE.search(t))
    if pick and pick[0]:
        e = pick[0]
        try:
            val = calc(e)
            # composicion de dos pasos: "... y dime el residuo entre d"
            tm = re.search(r"(?:residuo|modulo|resto)[^\d]{0,18}(\d+)", t[pick[1] + len(e):], re.I)
            if tm:
                val = calc(f"({e}) % {tm.group(1)}")
            return f"{val}"
        except Exception:
            pass
    return "[sin expresion resoluble]"

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?\s*[-+*/%]\s*-?\d+(?:\s*[-+*/%]\s*-?\d+)*)")
SPAN_CHARS = set("0123456789.+-*/%() ")

def _arith_spans(t):
    """Maximos tramos de la cadena que solo contienen digitos y operadores."""
    spans, cur, start = [], "", 0
    for i, ch in enumerate(t):
        if ch in SPAN_CHARS:
            if not cur: start = i
            cur += ch
        elif cur:
            spans.append((cur, start)); cur = ""
    if cur: spans.append((cur, start))
    return spans

def _best_expr(t):
    """La expresion aritmetica mas larga y evaluable del texto (admite parentesis)."""
    best = None
    for span, start in _arith_spans(t):
        e = span.strip()
        for _ in range(14):                      # recorta basura de los extremos
            if not e: break
            try:
                calc(e)
                n_ops = sum(e.count(o) for o in "+-*/%")
                if n_ops >= 1:
                    cand = (n_ops, len(e), e, start)
                    if best is None or cand[:2] > best[:2]: best = cand
                break
            except Exception:
                e = e[:-1].strip() if e[-1] in " +-*/%" or not e[-1].isdigit() else e[:-1].strip()
                if e and e[-1] in "(": e = e[:-1].strip()
        for cut in range(1, 8):                   # y prueba prefijos mas cortos
            if len(e) < 4: break
            frag = e[: -cut] if cut % 2 else e[cut:]
            try:
                calc(frag)
                if sum(frag.count(o) for o in "+-*/%") >= 1:
                    cand = (sum(frag.count(o) for o in "+-*/%"), len(frag), frag, start)
                    if best is None or cand[:2] > best[:2]: best = cand
                    break
            except Exception:
                continue
    return (best[2], best[3]) if best else None

def python_exec(code, _cfg=None, timeout=10):
    """Ejecución de Python deshabilitada: -I no constituye un sandbox de seguridad."""
    return "[herramienta deshabilitada: ejecutar Python requiere un sandbox externo]"


def now(_args=None, _cfg=None):
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def kb_search(query, index=None, k=None, _cfg=None):
    from rag import get_index
    idx = index or get_index(_cfg or {})
    hits = idx.search(query, k)
    if not hits:
        return "sin coincidencias"
    return "\n\n".join(f"[{h['score']:.2f}] ({h['src']}) {h['text'][:700]}" for h in hits)

class Memory:
    """Memoria persistente: hechos declarados + trayectorias resueltas (ejemplares)."""
    def __init__(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.c.executescript("""
          CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, k TEXT, v TEXT, t REAL);
          CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY, q TEXT, a TEXT, ok INTEGER,
                                            steps TEXT, t REAL);
        """)
        self.c.commit()
    def remember(self, k, v, _cfg=None):
        with self.lock:
            self.c.execute("INSERT INTO facts(k,v,t) VALUES(?,?,?)", (str(k), str(v), time.time()))
            self.c.commit()
        return "hecho guardado"
    def recall(self, q, n=5, _cfg=None):
        # La tabla de hechos no tiene embeddings; filtra lexicalmente para no mezclar hechos ajenos.
        terms = {x.lower() for x in re.findall(r"[\wáéíóúñ]+", str(q or "")) if len(x) > 2}
        with self.lock:
            rows = self.c.execute("SELECT k,v FROM facts ORDER BY id DESC LIMIT 500").fetchall()
        if terms:
            rows = [(k, v) for k, v in rows if terms & {x.lower() for x in re.findall(r"[\wáéíóúñ]+", f"{k} {v}")}]
        return "\n".join(f"- {k}: {v}" for k, v in rows[:n]) or "(sin hechos coincidentes)"
    def log_trace(self, q, a, ok, steps):
        with self.lock:
            self.c.execute("INSERT INTO traces(q,a,ok,steps,t) VALUES(?,?,?,?,?)",
                           (q, a, int(bool(ok)), json.dumps(steps, ensure_ascii=False)[:4000], time.time()))
            self.c.commit()
    def good_traces(self, q, n=2):
        """Recupera trayectorias correctas pareadas lexicalmente (autoaprendizaje)."""
        from rag import tokenize
        qw = set(tokenize(q))
        with self.lock:
            rows = self.c.execute("SELECT q,a,steps FROM traces WHERE ok=1 ORDER BY id DESC LIMIT 400").fetchall()
        sc = []
        for rq, ra, rs in rows:
            rw = set(tokenize(rq))
            j = len(qw & rw) / max(1, len(qw | rw))
            if j > 0: sc.append((j, rq, ra, rs))
        sc.sort(reverse=True)
        return sc[:n]

def build_registry(cfg, index=None, memory=None):
    reg = {
        "calc": {"fn": lambda expr=None, **kw: calc(expr, cfg),
                 "args": {"expr": "expresion aritmetica exacta, ej: (12+7)*3"},
                 "desc": "Calculadora exacta. Usarla SIEMPRE que haya numeros: nunca calcules de memoria."},
        "python": {"fn": lambda code=None, **kw: python_exec(code, cfg),
                   "args": {"code": "codigo python valido que imprime el resultado"},
                   "desc": "Interprete Python aislado para calculos complejos, secuencias o verificacion."},
        "search": {"fn": lambda query=None, k=None, **kw: kb_search(query, index, k, cfg),
                   "args": {"query": "palabras clave de busqueda", "k": "cuantos fragmentos (opcional)"},
                   "desc": "Busqueda lexical sobre los libros y documentos indexados en el workspace."},
        "remember": {"fn": lambda key=None, value=None, **kw: memory.remember(key, value) if memory else "sin memoria",
                     "args": {"key": "etiqueta corta", "value": "dato a recordar"},
                     "desc": "Guarda un hecho para usarlo en conversaciones futuras."},
        "recall": {"fn": lambda query=None, **kw: memory.recall(query or "") if memory else "sin memoria",
                   "args": {"query": "que recordar"},
                   "desc": "Recupera hechos guardados por el usuario o el sistema."},
        "now": {"fn": lambda **kw: now(), "args": {}, "desc": "Fecha y hora actuales del sistema."},
    }
    return reg

def call_tool(registry, name, args):
    if name not in registry:
        return f"[herramienta desconocida: {name}]"
    try:
        return str(registry[name]["fn"](**(args or {})))[:3000]
    except Exception as e:
        return f"[error en {name}: {type(e).__name__}: {e}]"

def spec_text(registry):
    return "\n".join(f"- {n}: {d['desc']}  args={json.dumps(d['args'], ensure_ascii=False)}"
                     for n, d in registry.items())
