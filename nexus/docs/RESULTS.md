# Resultados medidos (nexus-bench-v1)

Motor: transformer local de **7.29 M** parámetros, entrenado desde cero en este sandbox
(CPU, 2 núcleos): 8.24 M de tokens, ~3 900 pasos, pérdida de validación 2.82 (**16.8 ppl**,
descendiendo desde 146 ppl al inicio).

Suite: 84 problemas de 14 familias generados con semilla 990001 — **distinta** de la de
entrenamiento (20261021), así que no hay fuga de datos. Puntuación = coincidencia exacta
del dato final, comprobada por programa. Sin juez humano ni LLM-juez.

| brazo | qué es | exactitud |
|---|---|---|
| `rawlm` | el modelo base, a pelo | **27.4 %** (23/84) |
| `cot` | mismo modelo, con orden de razonar paso a paso | **64.3 %** (54/84) — el mejor |
| `tools` | agente con calculadora/Python/búsqueda, el modelo decide cuándo usarlas | **25.0 %** (21/84) |
| `tools+router` | + enrutado por log-prob, expresión fijada desde el enunciado y verificación programática | **60.7 %** (51/84) |

Detalle por familia (`tools+router`): aritmética 6/6 · mcm/mcd 6/6 · porcentaje 6/6 · distancia 6/6 ·
extracción 6/6 · expresión mixta 5/6 · primo 5/6 · módulo 4/6 · cifras 3/6 · lógica 4/6 ·
lineal 0/6 · conversión 0/6 · comparación 0/6 · sucesiones 0/6.

## Las cuatro lecciones que salen de aquí

1. **2.5× de mejora sin tocar los pesos.** De 26.2 % a 64.3 % solo con el andamiaje
   correcto. Ningún parámetro cambió: cambió el protocolo alrededor del modelo.

2. **Dar herramientas sin disciplina empeora al modelo** (26.2 % → 20.2 %). Medido, no
   teórico: el modelo de 7 M *pide* la calculadora pero **inventa la expresión** — para
   `27*13+401` llamaba a `calc` con `564 * 78` —, y una observación errónea lo deja más
   seguro dentro de su error. Un andamiaje que confía ciegamente en un modelo pequeño es
   peor que no andamiar.

3. **La instrucción pesa tanto como la herramienta, y con modelo mejor entrenado pesa más.**
   En la pasada final `cot` (64.3 %, una sola llamada, cero infraestructura) **igualó y superó**
   al orquestador completo (60.7 %). Al entrenar ~1 000 pasos más, el modelo aprendió a
   sostener la aritmética en el propio contexto y el coste de coordinar turnos dejó de pagar
   la pena. Traducción de negocio: **prueba prompt y destilación antes que un bucle de
   agentes**; el orquestador se justifica cuando la tarea necesita datos externos o
   acciones, no como muleta de un modelo mal instructions-ajustado.

4. **El prompt debe escalar con la capacidad.** Con el prompt largo (reglas + especificación
   de 6 herramientas) el modelo se perdía: 11 %. Con la versión compacta
   (`agent.compact_prompt: "auto"`, activa sola en el motor local) recupera nivel. Igualmente,
   forzar búsqueda en preguntas que ya traían el contexto delante hundía la extracción:
   de ahí el plan `leer`.

## Dónde fallamos de verdad (sin maquillar)

- **comparación 0/6, sucesiones 0/6, ecuación lineal 0/6**: el resolvedor
  determinista no cubre esos patrones y el modelo no infiere la regla. Aquí hace falta un
  parser más general o un motor mayor; no se arregla con más prompt.
- **La prosa del modelo alucina hechos.** Es un demostrador de arquitectura, no un modelo
  de lenguaje útil en abierto. Su texto es gramatical y frecuentemente falso.
- **El crítico LLM aporta poco** con un motor tan chico: se salta cuando existe valor
  calculado por máquina, que manda siempre.
- El texto recuperado es literatura (Quijote, Holmes, Frankenstein, Austen): el RAG responde
  de eso, no de hechos actuales.

## Cómo reproducir

```bash
cd nexus && ./run.sh eval --arms rawlm,cot,tools,tools+router --n 6
cat runs/eval_report.json      # mismo JSON que pinta la UI
```
