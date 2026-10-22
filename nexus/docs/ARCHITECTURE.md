# NEXUS — arquitectura del sistema

Fecha: 2026-10-21 · Estado: prototipo funcional ejecutándose en CPU (2 núcleos, 2 GB RAM, sin GPU).

## 0. Postura de diseño (leer primero)

La inteligencia de un producto de IA hoy no vive solo en los pesos. Vive en tres capas:

1. **El modelo** (capacidad bruta: lenguaje, conocimiento, razonamiento implícito).
2. **El andamiaje** (herramientas, recuperación, verificación, presupuesto de cómputo en inferencia).
3. **El producto** (memoria, latencia, coste, seguridad, eval continuo).

Un modelo pequeño con andamiaje fuerte le gana a un modelo grande sin él en tareas
verificables. Un modelo frontier con andamiaje le gana a ambos. Este sistema está
diseñado para lo segundo y funcionando con lo primero: **el mismo contrato sirve para el
transformer entrenado aquí y para una API frontier**, cambiando un campo de `config.json`.

Nada de este README afirma superioridad sobre los modelos frontier. Lo que afirma es que
la *arquitectura* es la correcta y que se puede medir.

## 1. Capas

```
┌─────────────────────────────────────────────────────────────┐
│ web/index.html          UI (SSE, trazas de herramientas)     │
├─────────────────────────────────────────────────────────────┤
│ src/server.py           HTTP + streaming, /api/{chat,eval…}  │
├─────────────────────────────────────────────────────────────┤
│ src/agent.py            plan → actuar → verificar → reflex.  │
│   · router por logprob (¿herramienta o responder?)           │
│   · autoconsistencia (muestreos k + voto)                    │
│   · verificador: un crítico + verificación programática      │
│   · memoria de trayectorias correctas (few-shot dinámico)    │
├──────────────┬──────────────┬───────────────┬───────────────┤
│ src/tools.py │ src/rag.py   │ src/engine.py │ src/eval.py   │
│ calc exacta  │ BM25 propio  │ LocalEngine   │ bench con     │
│ python aislado│ 9.8k frags  │ + APIEngine    │ ground truth  │
│ memoria SQL  │ sin deps     │ (OpenAI/      │ por programa  │
│              │              │  Anthropic/   │               │
│              │              │  Gemini)      │               │
├──────────────┴──────────────┴───────────────┴───────────────┤
│ src/model.py   transformer: RMSNorm, RoPE, SwiGLU, GQA,      │
│                atado de embeddings, 7.3M params, fp32        │
│ src/corpus.py  8.2M tokens: literatura bilingüe + sintética  │
│ src/train.py   AdamW, warmup+coseno, clip, resume por ckpt   │
└─────────────────────────────────────────────────────────────┘
```

## 2. Modelo base (lo que se entrenó de verdad aquí)

| Aspecto | Decisión | Por qué |
|---|---|---|
| Vocabulario | BPE 8192 (ByteLevel, entrenado en el propio corpus) | Coste/latencia de CPU; 8.2M tokens no dan para 50k vocab |
| Contexto | 256 tokens | Memoria: 2 GB. El agente no necesita ventana larga porque recupera |
| Arquitectura | pre-norm RMSNorm · RoPE · SwiGLU · GQA 8q/4kv · atado emb/cabeza | Estado del arte en eficiencia; menos FLOPs por token |
| Parámetros | 7.29M | Es un *motor de formato y lenguaje*, no un repositorio de conocimiento |
| Precisión | fp32 | Sin GPU; bf16 no aporta aquí |
| Datos | 8.24M tokens (79% literatura bilingüe real, 21% problemas sintéticos con respuesta verificable) | Mezcla deliberada: lenguaje natural + protocolo de herramientas |
| Optimización | AdamW β=(0.9,0.95), wd=0.1, lr 6e-4 → 6e-5 coseno, warmup 150, clip 1.0 | Receta estándar de modelos pequeños |
| Checkpoints | cada 150 pasos, reanudable | El sandbox reinicia: el progreso sobrevive en `models/` |

**Qué sabe hacer bien**: seguir el protocolo `<|user|>…<|assistant|>`, emitir
`Accion: {json}`, cerrar con `Respuesta final:`, y copiar/ajustar números del contexto.
**Qué no sabe**: aritmética fiable de memoria (ningún transformer pequeño la sabe; los
grandes la approximan). Por eso existe la capa 2.

### Por qué el modelo *debe* fallar la cuenta y el sistema no
Es la decisión central: el lenguaje lo pone el modelo, la verdad la pone la máquina.
`tools.calc` evalúa con AST restringida (solo `+ - * / // % **` y números; un intento de
`__import__("os")` se rechaza). `tools.python` corre en subproceso con `-I`, sin red y
con límite de 10 s y 4 KB de salida.

## 3. Agente

Bucle (`agent.Agent.roll`) con presupuesto `max_steps`:

1. **Plan** — clasificación barata: `calcular | buscar | deducir | responder`.
2. **Acción obligada si `buscar`** — se recupera con BM25 y se inyecta la Observación antes
   de permitir responder, con la orden "responde solo con lo que diga la Observación".
   Evita la alucinación típica del modelo pequeño con contexto delante.
3. **Enrutado por logprob** — `engine.choice(prompt, ["Accion:", "Respuesta final:"])`
   pregunta al modelo qué prefiere *por probabilidad del primer token*, no por un `if`.
   Si prefiere herramienta y la pregunta es aritmética, el harness prepara la llamada.
4. **Parseo tolerante** — JSON válido, y si viene roto se reparan `tool`/`expr` con regex;
   si nada funciona, se reintenta con el formato exacto (los modelos pequeños rompen JSON).
5. **Verificación** — dos jueces: (a) un crítico que emite PASS/REVISA; (b) un chequeo
   **programático**: si existe un valor calculado por máquina y la respuesta del modelo no
   coincide, manda la máquina y se marca `guarded`.
6. **Autoconsistencia** — `k` muestras a temperatura variable + voto por respuesta
   normalizada. El número de votos se expone en la UI como confianza.
7. **Memoria** — `models/memory.sqlite`: hechos (`remember/recall`) y trayectorias
   correctas que se reutilizan como ejemplos pocos-disparos en preguntas parecidas
   (similitud Jaccard léxica). Es auto-mejora sin reentrenar.

### `tools.calc_nl` — el honesto guardrail determinista
Resolvedor de aritmética en lenguaje natural (mcm/mcd, residuo, % de N, cifras,
km/h × h, conversiones). **No es "inteligencia"**: es un parser determinista que el
sistema usa para (i) verificar y (ii) dar la expresión correcta. El benchmark lo reporta
como brazo aparte (`tools+router`) para no inflar la medida del modelo. Devuelve
`[sin expresion resoluble]` cuando no aplica — verificado sin falsos positivos en
preguntas conversacionales.

## 4. Recuperación (RAG)

BM25 escrito a mano (`src/rag.py`): k1=1.4, b=0.75, idf estándar, 9 842 fragmentos de
los libros del workspace + `docs/knowledge.jsonl` si existe. Sin dependencias: numpy sobre
contadores dispersos. Suficiente y más rápido que un denso en 2 núcleos.

En un producto real esto se cambia por: incrustaciones + HNSW/pgvector, reranker cruzado,
troceo con detección de estructura, y citación obligatoria (el modelo ya recibe `[score]`).

## 5. Evaluación

`src/eval.py` — suite `nexus-bench-v1`, 20 familias de generadores deterministas
(`src/synth.py`) donde la respuesta se calcula por programa, no por un juez humano ni por
otro LLM. Semillas de evaluación **distintas** a las de entrenamiento (nada de fuga de
datos). Brazos:

`rawlm` (el modelo desnudo) → `lm_cot` (cadena de pensamiento) → `tools` (agente) →
`tools+router` (+ enrutado y verificación programática).

El punto es la **pendiente entre brazos**, no el número absoluto: mide cuánto aporta el
andamiaje. Con `provider: openai` en `config.json`, la misma suite corre contra el modelo
frontier y compara arquitecturas, no marcas.

## 6. API y superficie de producto

- `POST /api/chat` (SSE: `start · route · tool · candidate · token · done`)
- `GET /api/status` (motor, paso de entrenamiento, curva de pérdida, tamaño del índice, último reporte)
- `GET|POST /api/config` (editar config en caliente; reinicia el motor)
- `POST /api/eval` · `POST /api/train` (dispara brazos desde la UI)
- `GET /api/run/<archivo>` (ver `train.log` / `eval.log` en crudo)

Sin frameworks: `http.server` + threading. Menos superficie de ataque y cero dependencias
para desplegar.

## 7. Lo que hay que saber antes de fiarse

- 7.3M parámetros no es un modelo de lenguaje útil en abierto: es un demostrador de la
  arquitectura. Su texto es gramaticalmente plausible y frecuentemente erróneo.
- La latencia está limitada por 2 CPUs; el mismo código en una GPU con batch continuo
  cambia un orden de magnitud.
- No hay alineación (SFT/RL) ni guardas de seguridad reales: `tools.python` ejecuta código
  del modelo en un subproceso. Para producción: contenedor sin red, sin secretos montados,
  `allow-list` de herramientas por inquilino y revisión humana en acciones destructivas.
- El "conocimiento" viene del corpus literario (18 MB de novelas): no es un índice factual.
