# NEXUS — arquitectura y límites actuales

Estado: prototipo experimental. Los artefactos entrenados no están en Git; descarga el release o prepara el corpus y entrena localmente.

## Componentes

- `src/model.py`: transformer causal PyTorch con RMSNorm, RoPE, SwiGLU, GQA y embeddings atados.
- `src/corpus.py` / `src/train.py`: tokenizer Hugging Face `tokenizers`, dataset y entrenamiento PyTorch.
- `src/engine.py`: contrato común para motor local y proveedores HTTP API.
- `src/agent.py`: bucle de herramientas, recuperación, verificación, autoconsistencia y memoria.
- `src/tools.py`: calculadora AST restringida, RAG, SQLite y reloj. Ejecución arbitraria de Python deshabilitada.
- `src/rag.py`: BM25 con caché de índice comprimida y validada por huella de archivos fuente.
- `src/server.py`: API basada en `http.server`; respuesta SSE con eventos y fragmentos de respuesta. La respuesta del modelo se resuelve antes de emitir los fragmentos finales; no es generación token a token.
- `web/index.html`: cliente web.

## Verificación y herramientas

La calculadora acepta únicamente operaciones aritméticas y aplica límites de longitud, complejidad y magnitud. Un guard determinista puede corregir operaciones que puede extraer del enunciado. No es un parser general de lenguaje natural y no verifica cualquier clase de pregunta numérica.

El registro de herramientas incluye una entrada de Python deshabilitada. El modo aislado de Python (`-I`) no restringe archivos, red, procesos ni secretos y no se considera aislamiento seguro.

El texto recuperado se presenta como observación no confiable y las respuestas reciben comprobaciones según la tarea. La verificación del modelo no equivale a una garantía formal de corrección.

## Memoria

La memoria SQLite distingue hechos y trayectorias. Los hechos se recuperan por coincidencia lexical simple; no hay búsqueda semántica ni embeddings. Las trayectorias se usan como ejemplos solo cuando están marcadas verificadas. La memoria de benchmark es temporal y separada de la memoria de la aplicación.

## Recuperación

BM25 indexa los archivos `.txt` de `data/` y `docs/knowledge.jsonl` si existe. El índice y sus estadísticas se serializan en un archivo gzip JSON bajo `models/`; una huella de tamaños, fechas de modificación y configuración invalida la caché cuando cambian las fuentes. La primera indexación puede tardar.

## Entrenamiento y evaluación

El checkpoint contiene el estado del modelo y del optimizador. La reanudación valida las dimensiones de arquitectura y vocabulario. `--additional-steps` permite continuar por un número explícito de pasos.

El benchmark usa generadores deterministas de `src/synth.py`, semillas configurables y memoria aislada. Los brazos son `rawlm`, `lm_cot`, `tools` y `tools+router`; las puntuaciones son coincidencias exactas con los generadores, no una evaluación de calidad general. Los números del documento de resultados son históricos y no se garantizan para el estado actual.

## API y controles

- `POST /api/chat`: conversación SSE.
- `GET /api/status`: estado de proveedor, entrenamiento, corpus y evaluación.
- `GET /api/config`: configuración de solo lectura.
- `POST /api/eval` y `POST /api/train`: requieren `NEXUS_ADMIN_TOKEN`.
- Si se usa un proveedor API, `/api/chat` requiere `NEXUS_API_TOKEN`.

Los tokens se envían en headers, no se almacenan en la configuración. El servidor limita tamaño de solicitudes e historial, valida `k`, protege rutas estáticas y serializa tareas administrativas. Para exposición pública aún hacen falta HTTPS, límites de tasa y gasto, y supervisión de infraestructura.

## Reproducibilidad

```bash
./run.sh setup
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
```
