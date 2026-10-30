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

- Las operaciones de entrenamiento, evaluación y descarga de archivos de `runs/` requieren `NEXUS_ADMIN_TOKEN` en el encabezado `X-Nexus-Admin-Token`.
- `/api/config` es de solo lectura; los cambios se realizan fuera del servidor.
- El servicio limita tamaño y estructura de solicitudes.
- La herramienta de ejecución de Python está **deshabilitada**. `python -I` no constituye un sandbox.
- NEXUS no debe exponerse públicamente sin HTTPS, límites de tasa/gasto y controles de despliegue adicionales.
- El corpus por defecto es literatura, no una base factual actualizada.

La descripción de arquitectura está en `docs/ARCHITECTURE.md`; la hoja de ruta no representa capacidades ya implementadas.

Las trazas de errores del chat se registran en el servidor y no se envían al cliente. Protege el acceso a los logs, que pueden contener datos sensibles. El límite de solicitudes en memoria es por proceso y no sustituye un limitador compartido en producción.

El servidor limita por proceso el chat a 10 solicitudes por IP y minuto y a cuatro
conversaciones simultáneas; las operaciones administrativas comparten un límite de
10 solicitudes por IP y minuto. Los rechazos y acciones administrativas se registran
sin tokens ni contenido de solicitudes. Estos controles no son globales entre procesos,
no limitan el gasto y no reemplazan autenticación por usuario, un proxy HTTPS o un
limitador compartido. Si se usa un proxy, configurar allí los límites por IP real;
el servidor solo observa la dirección del cliente conectado directamente.

### Memoria en el servidor web

El chat web usa memoria temporal aislada por solicitud y no guarda hechos ni
trayectorias entre peticiones. El historial enviado por el navegador mantiene el
contexto de la conversación. Esta medida evita compartir datos entre usuarios
mientras no exista autenticación individual. Las herramientas y programas de
línea de comandos conservan su configuración de memoria independiente.

El endpoint administrativo `GET /api/tasks` acepta `X-Nexus-Admin-Token` y
muestra el estado de la última tarea de entrenamiento y evaluación en el proceso
actual (`idle`, `running`, `succeeded`, `failed` o `timed_out`). Este estado es
volátil: se pierde al reiniciar y no se comparte entre procesos.
