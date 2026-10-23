# De prototipo a IA de frontera — hoja de ruta real

Cuánto cuesta de verdad construir "la IA más avanzada", en números, y qué se puede
defender con presupuesto pequeño. Todo lo de abajo es ingeniería conocida, no especulación.

## 1. La ley física del asunto

Entrenar un transformer cuesta `C ≈ 6 · N · D` FLOPs (N parámetros, D tokens), y el óptimo
de Chinchilla pide ~20 tokens por parámetro. De ahí sale todo lo demás:

| Escala | Parámetros | Tokens | FLOPs de entrenamiento | Cómputo (~H100 bf16 a 55 % MFU) | Coste de nube orientativo |
|---|---|---|---|---|---|
| **Este prototipo** | 7.3 M | 8 M | 3.6e14 | segundos | 0 € (CPU local) |
| Modelo serio de nicho | 3 B | 100 B | 1.8e21 | ~1 día · 512 GPU | 1–3 k€ |
| Rival de Llama/Gemma | 70 B | 15 T | 6.3e24 | ~3 semanas · 8 192 GPU | 2–6 M€ |
| Frontera actual | ~1e3–1e4 B activos (MoE) | 15–35 T | 1e26 | ~2–3 meses · ~100 k GPU | 100–500 M$ |
| Frontera + RL/post-entreno | idem ×3–10 | idem | 1e26–1e27 | idem + meses | varios cientos de M$ |

MFU realista en clusters grandes: 35–55 %. Un 30 B denso en 1e26 FLOPs tarda ~4 meses en
100 k H100: la frontera es hoy un problema de *parque de GPUs y fiabilidad*, no de ideas.

**Consecuencia práctica**: nadie compite en pre-entrenamiento con presupuesto pequeño. Se
compite en (a) post-entrenamiento, (b) andamiaje, (c) datos y tarea vertical, (d) latencia
y coste de inferencia. Eso es lo que hace rentable un producto de IA en 2026.

## 2. Ruta en cuatro fases

### Fase 0 — ya hecha aquí (días, 0 €)
- Tokenizador BPE entrenado en el corpus propio, transformer RMSNorm/RoPE/SwiGLU/GQA.
- Agente con herramientas, verificación programática, memoria y benchmark propio.
- Sirve para: validar la arquitectura, el formato de protocolo y el producto.

### Fase 1 — modelo propio defendible (1–2 meses, 3–30 k€)
1. **Datos**: 200–500 B tokens limpios. Common Crawl + filtrado por clasificador de
   calidad, deduplicación MinHash, código (The Stack filtrado), matemáticas, synthetic
   destilado de un teacher frontier (esto es lo que sube el razonamiento barato).
2. **Arquitectura**: 1.5–4 B denso o MoE 8×1 B activo; ventana ≥32 k con YaRN/RoPE base
   alta; atado de embeddings; QK-norm.
3. **Pre-entrenamiento**: 512–1024 H100 en `torchrun` + FSDP/BF16, muP para escalar lr sin
   reintentarlo, checkpoints cada 2 h, recuperación automática de fallos de nodo.
4. **Post-entrenamiento** (aquí está la jugada): SFT con 50–200 k trazas destiladas y
   filtradas por verificador; después **RLVR** (premios verificables por programa en mate,
   código y herramientas) — es lo que hace que un modelo pequeño rinda como uno grande en
   tareas con respuesta exacta. DPO/ORPO si no hay verificadores.
5. **Evaluación continua**: la suite propia + MMLU-Pro, GPQA, AIME, BFCL, SWE-bench-lite,
   tau-bench (agentes), y latencia/coste por tarea resuelta.

### Fase 2 — producto (paralelo a la 1)
- **Inferencia**: vLLM/SGLang, batch continuo, cache de prefijos, decodificación
  especulativa (un borrador de 0.5 B acelera 1.8–2.5×), cuantización FP8/AWQ.
- **Orquestación**: enrutador por coste (barato primero, escalar si el verificador duda),
  paralelización de presupuesto en test-time (muestreos k + votación verificada).
- **Datos propietarios**: trazas de usuario + correcciones → el foso real. Un vertical con
  100 k tareas resueltas y verificadas vale más que 1 B de tokens genéricos.
- **Coste objetivo**: < 0.30 €/1 000 consultas y p50 < 1.5 s. Se logra destilando a 3–8 B
  y sirviendo en 1–2 GPUs, o llamando a un API barato con caché de respuestas.

### Fase 3 — frontera (años, >100 M$)
Solo si Fase 1–2 demuestran retención. Requiere: cluster propio o reservado a 18 meses,
equipo de sistemas (paralelismo 4D, tolerancia a fallos, kernel tuning), datos
sintéticos auto-verificados a escala, y RL en entorno agéntico con verificadores ricos.

## 3. Dónde está la ventaja competitiva pequeña

| Palanca | Ganancia típica | Coste |
|---|---|---|
| Verificación programática + reintentos | +15–40 pts en mate/código | bajo |
| Autoconsistencia con k=5 y votación | +5–15 pts | cómputo ×5 |
| RAG con reranker en dominio | +20–35 pts en QA factual | medio |
| RLVR en el vertical | +10–30 pts en la tarea paga | medio |
| Destilar teacher → alumno 3 B | −70 % coste, −8 pts calidad | medio |
| Caché semántica de respuestas | −30 % llamadas | bajo |

Ninguna necesita 100 M$. Todas están ya esbozadas en este repo (verificación,
autoconsistencia, RAG, enrutado por coste en `engine.make_engine`).

## 4. Seguridad y cumplimiento (no opcional en producto)

- Herramientas con *allow-list* por usuario; Python permanece deshabilitado hasta que exista un contenedor real sin red ni secretos. El modo `-I` no es un sandbox.
- Presunción de inyección por recuperación: el texto del corpus nunca es instrucciones.
- Registro completo de trazas (aquí: `traces` en SQLite) para auditoría post-hoc.
- Evaluación de jailbreak y sesgo antes de cada release; límite de gasto por sesión.

## 5. Criterio de salida de cada fase

- **F0→F1**: el benchmark muestra que el andamiaje sube la exactitud ≥2× sobre el modelo
  desnudo. Si no, el problema es el andamiaje, no el tamaño.
- **F1→F2**: modelo propio que empata o gana al API barato usado como teacher en tu tarea.
- **F2→F3**: coste por tarea resuelta por debajo del precio de comprarlo, y datos
  propietarios que el teacher no tiene.
