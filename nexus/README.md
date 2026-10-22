# NEXUS

Sistema de IA completo y ejecutable: un transformer **entrenado desde cero en esta máquina**,
un agente con herramientas y verificación, recuperación, memoria, API con streaming y un
banco de pruebas que mide de verdad lo que aporta cada pieza.

No es un envoltorio de una API. `src/model.py`, `src/train.py`, `src/corpus.py`,
`src/rag.py` y `src/eval.py` son implementacion propia (PyTorch + stdlib). Si añades una clave de API,
el mismo agente la usa como motor; si no, usa el modelo local. Un campo decide cuál.

## Arranque rápido

```bash
cd nexus
./run.sh all            # corpus + entrenamiento en segundo plano + UI en :8080
./run.sh eval --n 6     # benchmark de 4 brazos
./run.sh ask "Calcula 27*13 + 401."
```

Dependencias: `torch` (cpu), `tokenizers`, `numpy`. El script las instala si faltan, porque
este entorno no conserva lo instalado entre sesiones.

El corpus (18 MB de Gutenberg) **no esta en el repo**: `./run.sh data` lo descarga con
`src/fetch_corpus.sh` y reconstruye tokenizador y dataset. El repo pesa lo que importa.

## Qué hay dentro

| Ruta | Qué es |
|---|---|
| `config.json` | **todo lo configurable**: proveedor, tamaño del modelo, pasos, temperatura, k de autoconsistencia |
| `src/model.py` | transformer: RMSNorm · RoPE · SwiGLU · GQA · embeddings atados (7.3M params) |
| `src/train.py` | AdamW, warmup+coseno, clip, checkpoints reanudables, registro JSONL de la curva de pérdida |
| `src/corpus.py` | 18 MB de literatura bilingüe real + datos sintéticos con respuesta verificable → 8.2M tokens |
| `src/synth.py` | 20 generadores de problemas (aritmética, ecuaciones, lógica, conversión, extracción) |
| `src/agent.py` | plan → acción → observación → verificación → autocrítica → memoria |
| `src/tools.py` | calculadora exacta (AST), Python aislado, búsqueda BM25, memoria SQLite, fecha |
| `src/engine.py` | `LocalEngine` (aquí) y `APIEngine` (OpenAI/Anthropic/Gemini) con el mismo contrato |
| `src/eval.py` | `nexus-bench-v1`: exactitud por brazo, puntuable por programa, sin fuga entre train/eval |
| `src/server.py` | HTTP + SSE, `/api/chat`, `/api/status`, `/api/config`, `/api/eval` |
| `web/index.html` | UI con trazas de herramientas, veredicto del verificador y curva de pérdida |
| `docs/ARCHITECTURE.md` | decisiones técnicas y límites honestos del sistema |
| `docs/RESULTS.md` | **números medidos**: 26 % → 64 % solo con andamiaje, y dónde fallamos |
| `docs/ROADMAP.md` | qué cuesta llegar a un modelo frontier, en FLOPs, GPU y euros |

## Usar un modelo frontier (1 minuto)

```bash
export OPENAI_API_KEY=sk-...      # o ANTHROPIC_API_KEY / GEMINI_API_KEY
cd nexus && ./run.sh serve        # provider:"auto" ya lo detecta
```
o fija el proveedor en `config.json` (`"provider": "anthropic"`) y pon el modelo que quieras.
Sin clave, sigue funcionando con el motor local: no hay dependencia oculta de red.

## Medir en vez de opinar

```bash
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
cat runs/eval_report.json
```

Se espera que la curva suba del modelo desnudo al agente. Lo importante es la **pendiente**:
cuánto rinde el andamiaje sobre un modelo de 7M parámetros. La misma suite corre contra un
modelo frontier para comparar arquitecturas en igualdad de condiciones.

## Estado y advertencias

- El modelo local es un demostrador de arquitectura: texto gramatical, aritmética poco
  fiable a propósito (por eso manda la verificación por programa).
- `tools.python` ejecuta código del modelo en un subproceso acotado. Antes de exponer esto
  a usuarios: contenedor sin red, allow-list de herramientas y límite de gasto por sesión.
- Conocimiento = corpus literario descargado, no un índice factual. El RAG responde de
  Quijote, Holmes o Frankenstein, no de la actualidad.
