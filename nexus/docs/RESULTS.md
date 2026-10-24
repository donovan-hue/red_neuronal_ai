# Resultados históricos

Las cifras de este archivo proceden de una revisión anterior del prototipo. No se han vuelto a ejecutar después de los cambios del agente, la verificación, el benchmark y la memoria; por ello no representan resultados validados del estado actual del código.

La revisión anterior reportó, con semilla 990001 y 84 preguntas, estas cifras:

| brazo histórico | exactitud histórica |
|---|---:|
| `rawlm` | 27.4 % (23/84) |
| `cot` | 64.3 % (54/84) |
| `tools` | 25.0 % (21/84) |
| `tools+router` | 60.7 % (51/84) |

Estos números son referencias no reproducidas. El código actual acepta `rawlm`, `lm_cot`, `tools` y `tools+router`; el alias `cot` apunta a `lm_cot`.

Para generar un informe actual:

```bash
cd nexus
./run.sh setup
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
cat runs/eval_report.json
```

El benchmark mide exactitud exacta sobre tareas sintéticas generadas por programa. No mide seguridad, utilidad general, factualidad abierta ni calidad humana.
