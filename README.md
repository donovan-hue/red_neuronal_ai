# red_neuronal_ai

Este repositorio contiene **dos demostradores separados**:

- **Red neuronal clásica** en la raíz (`model.py`, `main.py`, `api.py`). `Procfile` despliega este servicio Flask legado.
- **NEXUS** en `nexus/`, un prototipo de agente con motor local/API, RAG, memoria y benchmark.

No comparten checkpoints ni rutas de API.

La lista única de desarrollo pendiente está en [PENDIENTES.md](PENDIENTES.md).

## Demostrador clásico

```bash
python -m pip install -r requirements.txt
python main.py                 # entrena XOR y escribe modelo.npz
python predecir.py
python interactuar.py
```

El checkpoint NPZ se carga con `allow_pickle=False`. No se admiten checkpoints pickle.

API: `POST /predecir` (requiere `x1` y `x2`), `POST /entrenar` y `GET /health`. Los cuerpos JSON se validan y el entrenamiento tiene límites de tamaño. `POST /entrenar` requiere `API_TRAIN_TOKEN` y el header `X-API-Train-Token`.

## NEXUS

```bash
cd nexus
./run.sh setup                # dependencias, corpus y dataset
./run.sh train                # entrenar
./run.sh serve                # interfaz y API en :8080
./run.sh eval --arms rawlm,lm_cot,tools,tools+router --n 6
```

Los artefactos se guardan bajo `nexus/models`, `nexus/data` y `nexus/runs`, excluidos de Git. Para descargar el checkpoint publicado:

```bash
cd nexus
./get_model.sh
```

NEXUS deshabilita la ejecución de Python porque un subproceso con `-I` no es un sandbox. Para habilitar un proveedor API de pago, configura `NEXUS_API_TOKEN`; para acciones administrativas configura además `NEXUS_ADMIN_TOKEN`. No expongas un servidor sin esos controles en un entorno público.

Las limitaciones y resultados experimentales se documentan en `nexus/README.md` y `nexus/docs/`.

### Simulación local con avatar

Desde la raíz del repositorio:

```bash
python3 nexus_single.py serve 8080
```

Abre `http://127.0.0.1:8080`. Esta UI es una conversación guionizada de demostración, con avatar SVG local; no envía los mensajes a una API. El servidor se limita a `127.0.0.1` por defecto.

El entrenamiento de la API clásica prepara una copia y solo reemplaza el modelo
activo después de guardar el checkpoint correctamente. Los bloqueos son por
proceso: usa un único worker si habilitas entrenamiento; varios workers no
sincronizan automáticamente sus modelos.

### Entrenamiento y predicción por lotes

`POST /predecir/lote` recibe `{"X": [[0, 1], [1, 0]]}` y devuelve
`predicciones` y `redondeado` para hasta 500 filas. `POST /entrenar` devuelve
`metricas` con la pérdida cuadrática media, precisión binaria y épocas ejecutadas.
Opcionalmente acepta `validation: {"X": [...], "y": [...]}` y `patience` para
detención temprana; cuando se proporcionan datos de validación se restaura el
modelo de la época con menor pérdida de validación. Los conjuntos deben ser
independientes; el servidor no los divide automáticamente.

Los gradientes se normalizan por el tamaño del lote, por lo que la tasa de
aprendizaje tiene una escala diferente a la de versiones anteriores. El ejemplo
XOR usa `lr=0.4` para conservar la convergencia con 10 000 épocas.
