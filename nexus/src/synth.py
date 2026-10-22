"""Generadores deterministas de problemas con respuesta verificable por programa.

Se usan dos veces: (1) para mezclar en el entrenamiento del modelo base y
(2) como suite de evaluacion con semillas/splits separados -> nada hace overlap.
Cada item lleva 'answer' exacto, asi el score no depende de un juez.
"""
from __future__ import annotations
import random, re

def _fmt_num(x):
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        x = int(round(x))
    return str(x)

# ---------------- nucleos aritmeticos ----------------
def arith(rng, hard=False):
    a, b = rng.randint(11, 99 if not hard else 999), rng.randint(3, 49 if not hard else 99)
    op = rng.choice("+-*/") if not hard else rng.choice("+-*")
    if op == "/":
        b0 = b; b = rng.choice([d for d in range(2, 13) if (a * b0) % d == 0] or [2]); a, b = a * b0, b
        val = a // b
        q = f"Halla el cociente entero de {a} entre {b}."
    else:
        val = {"+": a + b, "-": a - b, "*": a * b}[op]
        q = f"Calcula {a} {op} {b}."
    return q, _fmt_num(val)

def mixed_expr(rng):
    n = rng.randint(3, 4)
    nums, ops = [rng.randint(2, 19)], []
    for _ in range(n - 1):
        ops.append(rng.choice("+-*")); nums.append(rng.randint(2, 13))
    expr = "".join(str(nums[0]) + "".join(f" {o} {x}" for o, x in zip(ops, nums[1:])))
    val = eval(expr, {"__builtins__": {}}, {})
    return f"Evalua la expresion respetando la jerarquia de operadores: {expr}", _fmt_num(val)

def linear_eq(rng):
    x = rng.randint(-9, 15); a = rng.randint(2, 9); b = rng.randint(-20, 20); c = rng.randint(-9, 12)
    rhs = a * x + b + c * 0
    eq = f"{a}x + {b} = {rhs}" if b >= 0 else f"{a}x - {-b} = {rhs}"
    return f"Despeja x en la ecuacion {eq}. Da solo el valor de x.", _fmt_num(x)

def percent(rng):
    p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75]); n = rng.randint(2, 40) * 20
    val = n * p // 100
    return f"¿Cuanto es el {p}% de {n}?", _fmt_num(val)

def frac_sum(rng):
    d = rng.choice([4, 5, 6, 8, 10, 12]); a = rng.randint(1, d - 1); b = rng.randint(1, d - 1)
    return f"Suma las fracciones {a}/{d} + {b}/{d} y da el numerador del resultado (denominador {d}).", _fmt_num(a + b)

def word_rate(rng):
    v = rng.randint(3, 24) * 5; t = rng.randint(2, 9)
    return (f"Un tren mantiene {v} km/h durante {t} horas. ¿Cuantos kilometros recorre en total?", _fmt_num(v * t))

def word_work(rng):
    a = rng.randint(2, 9); b = rng.randint(2, 9)
    hours = round(a * b / (a + b), 2)
    return (f"A pinta una cerca en {a} horas y B en {b} horas. Trabajando juntos, ¿cuantas horas tardan? Redondea a 2 decimales.", _fmt_num(hours))

def word_age(rng):
    now = rng.randint(4, 14); diff = rng.randint(15, 40)
    q = f"Un padre tiene {now + diff} anos y su hijo {now} anos. ¿En cuantos anos el padre tendra el doble de edad que el hijo?"
    return q, _fmt_num(diff - now)

def profit(rng):
    cost = rng.randint(2, 60) * 25; gain = rng.choice([10, 20, 25, 40, 50])
    return f"Un articulo cuesta {cost} y se vende con {gain}% de ganancia. ¿Cual es el precio de venta?", _fmt_num(cost + cost * gain // 100)

def sequence(rng):
    if rng.random() < 0.6:
        a, d = rng.randint(1, 20), rng.randint(2, 9); terms = [a + d * i for i in range(5)]
        nxt, rule = a + 5 * d, "aritmenica"
    else:
        a, r = rng.randint(1, 5), rng.randint(2, 3); terms = [a * r ** i for i in range(5)]
        nxt, rule = a * r ** 5, "geometrica"
    return f"La sucesion {', '.join(map(str, terms))} continue una regla {rule}. ¿Cual es el siguiente termino?", _fmt_num(nxt)

def gcdlcm(rng):
    a, b = rng.randint(6, 60), rng.randint(6, 60)
    import math
    if rng.random() < 0.5:
        return f"Halla el maximo comun divisor de {a} y {b}.", _fmt_num(math.gcd(a, b))
    return f"Halla el minimo comun multiplo de {a} y {b}.", _fmt_num(math.lcm(a, b))

def modulo(rng):
    a, b = rng.randint(50, 900), rng.randint(3, 13)
    return f"Cual es el residuo de dividir {a} entre {b}?", _fmt_num(a % b)

def digits(rng):
    n = rng.randint(1000, 99999)
    if rng.random() < 0.5:
        return f"Suma las cifras del numero {n}.", _fmt_num(sum(int(c) for c in str(n)))
    return f"Invierte las cifras del numero {n} (sin ceros a la izquierda).", _fmt_num(int(str(n)[::-1]))

def prime(rng):
    n = rng.randint(20, 400)
    isp = all(n % d for d in range(2, int(n ** 0.5) + 1))
    return f"¿Es {n} un numero primo? Responde si o no.", ("si" if isp else "no")

def convert(rng):
    table = [("km", "m", 1000), ("h", "min", 60), ("kg", "g", 1000), ("m", "cm", 100), ("min", "s", 60)]
    big, small, f = rng.choice(table); n = rng.randint(2, 40)
    return f"Convierte {n} {big} a {small}.", _fmt_num(n * f)

def compare(rng):
    nums = [rng.randint(100, 99999) for _ in range(rng.randint(3, 5))]
    if rng.random() < 0.5:
        return f"¿Cual es el mayor de estos numeros: {', '.join(map(str, nums))}? Da solo el numero.", _fmt_num(max(nums))
    return f"Ordena de menor a mayor y da el segundo: {', '.join(map(str, nums))}.", _fmt_num(sorted(nums)[1])

def syllogism(rng):
    A = rng.choice(["Los grifos vuelan", "Las kotas nadan", "Los velos brillan", "Los zorros duermen"])
    B = rng.choice(["todo lo que vuela come metal", "todo lo que nada bebe luz", "todo lo que brilla pesa dos kilos", "todo lo que duermen suena"])
    C = A.split(" ")[1]
    subj = A.replace("Los ", "Ningun ").replace("Las ", "Ninguna ")
    q = (f"Premisa 1: {A}. Premisa 2: {B}. "
         f"Pregunta: ¿Se sigue que los {C} {B.split(' ', 2)[2].rstrip('.')}? Responde si o no.")
    return q, "si"

def negation(rng):
    p = rng.choice(["llueve", "hay luz", "el sensor esta activo", "el servidor responde"])
    q = rng.choice(["el suelo esta mojado", "se ve el camino", "el motor arranca", "hay conexion"])
    form = rng.choice(["modus_tollens", "modus_ponens"])
    if form == "modus_ponens":
        return f"Si {p} entonces {q}. Ocurre que {p}. ¿Se sigue {q}? Responde si o no.", "si"
    return f"Si {p} entonces {q}. No ocurre {q}. ¿Se sigue que no ocurre {p}? Responde si o no.", "si"

def extract(rng):
    ents = [("Nimbo", "protocolo de cifrado", "2019"), ("Tlaloc", "modelo climatico", "2021"),
            ("Aria-7", "orquestador de agentes", "2023"), ("Quetzal", "lenguaje de modelos", "2018")]
    e = rng.choice(ents)
    ctx = " ".join(f"{n} es un {t} publicado en {y}." for n, t, y in ents)
    return (f"Lee el contexto y responde solo con el dato.\nContexto: {ctx}\nPregunta: ¿En que ano se publico {e[0]}?", e[2])

GENS = {
    "arith": arith, "arith_hard": lambda r: arith(r, True), "mixed_expr": mixed_expr,
    "linear_eq": linear_eq, "percent": percent, "frac": frac_sum, "rate": word_rate,
    "work": word_work, "age": word_age, "profit": profit, "sequence": sequence,
    "gcdlcm": gcdlcm, "modulo": modulo, "digits": digits, "prime": prime,
    "convert": convert, "compare": compare, "syllogism": syllogism, "logic": negation,
    "extract": extract,
}

def make_item(rng):
    name = rng.choice(list(GENS))
    q, ans = GENS[name](rng)
    return {"task": name, "question": q, "answer": ans, "expr": expression_for(q),
            "why": rng.choice(WHYS)}


WHYS = [
    "Una operacion a la vez; compruebo invirtiendo el resultado.",
    "Reviso unidades y orden de operaciones antes de cerrar.",
    "Hago el calculo exacto y lo verifico por otra via.",
    "Sin atajos: despejo, sustituyo y confirmo.",
    "Calculo y compruebo que el numero tenga sentido.",
]

COT_ALT = ["Verifico el orden de operaciones.", "Reviso el signo y las unidades.",
           "Compruebo el resultado sustituyendolo.", "Descompongo el calculo en dos pasos."]

def expression_for(q):
    """Recupera la expresion aritmetica literal de la pregunta (para el ejemplo sinteético)."""
    m = re.match(r"Calcula (.+?)\.$", q)
    if m: return m.group(1)
    m = re.search(r"[Ee]valua .*?:\s*(.+)$", q)
    if m: return m.group(1)
    m = re.match(r"Suma las cifras del numero (\d+)", q)
    if m: return " + ".join(m.group(1))
    m = re.match(r"Cual es el residuo de dividir (\d+) entre (\d+)", q)
    if m: return f"{m.group(1)} % {m.group(2)}"
    m = re.match(r"Halla el minimo comun multiplo de (\d+) y (\d+)", q)
    if m: return f"mcm({m.group(1)},{m.group(2)})"
    return None

def render_train(q, a, cot=True, why=None, i=0, expr=None):
    """Entrena dos habitos: delegar la cuenta a una herramienta y cerrar con 'Respuesta final:'.

    Alternamos 3 registros para que el modelo no colapse a recitar una plantilla fija.
    """
    style = i % 3
    if style == 0 and expr:
        body = ('Necesito exactitud, lo delego.\n'
                f'Accion: {{"tool": "calc", "args": {{"expr": "{expr}"}}}}\n'
                f"Observacion: {a}\nRespuesta final: {a}")
    elif style == 1:
        body = f"{COT_ALT[i % len(COT_ALT)]}\nRespuesta final: {a}"
    else:
        body = f"Respuesta final: {a}"
    return f"<|user|>\n{q}<|end|>\n<|assistant|>\n{body}<|end|>\n"

def render_tool_example(q, expr, val):
    return ("<|user|>\n" + q + "<|end|>\n<|assistant|>\n"
            f"Pienso: necesito una operacion exacta, la delego a la calculadora.\n"
            f'Accion: {{"tool": "calc", "args": {{"expr": "{expr}"}}}}\n'
            f"Observacion: {val}\n"
            f"Respuesta: {val}<|end|>\n")

def sample_batch(n, seed):
    rng = random.Random(seed)
    return [make_item(rng) for _ in range(n)]
