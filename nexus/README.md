# NEXUS

Prototipo de agente con un transformer local opcional, proveedores API, herramientas aritméticas, RAG BM25, memoria SQLite y benchmark reproducible. No es un modelo frontier ni un servicio listo para exposición pública.

## Inicio

```bash
cd nexus
./run.sh setup
./run.sh train
./run.sh serve
```

`setup` instala dependencias y descarga el corpus de Project Gutenberg cuando hace falta. El entrenamiento guarda checkpoints bajo `models/` y los datos bajo `data/`; ambos están excluidos de Git.

Para iniciar el servidor sin entrenar, descarga el checkpoint publicado:

```bash
./get_model.sh
./run.sh serve
```

El descargador verifica SHA-256 completo y crea el vocabulario y una partición de validación no solapada a partir del artefacto de entrenamiento publicado.

## Benchmark

```bash
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
```

Brazos aceptados: `rawlm`, `lm_cot`, `tools`, `tools+router` (también se aceptan los alias `lm`, `cot` y `agent`). Los resultados se escriben en `runs/eval_report.json`. El benchmark crea memoria aislada y no reutiliza trayectorias entre ítems.

Los resultados históricos de `docs/RESULTS.md` corresponden a otra revisión y **no deben interpretarse como resultado reproducido por el código actual**.

## Proveedor API

Define la clave correspondiente (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` o `GEMINI_API_KEY`). Para habilitar llamadas desde el servidor configura también `NEXUS_API_TOKEN`; el cliente web solicitará ese token. Las llamadas a APIs pueden tener coste.

## Seguridad y límites

- Las operaciones de entrenamiento y evaluación requieren `NEXUS_ADMIN_TOKEN`.
- `/api/config` es de solo lectura; los cambios se realizan fuera del servidor.
- El servicio limita tamaño y estructura de solicitudes.
- La herramienta de ejecución de Python está **deshabilitada**. `python -I` no constituye un sandbox.
- NEXUS no debe exponerse públicamente sin HTTPS, límites de tasa/gasto y controles de despliegue adicionales.
- El corpus por defecto es literatura, no una base factual actualizada.

La descripción de arquitectura está en `docs/ARCHITECTURE.md`; la hoja de ruta no representa capacidades ya implementadas.
