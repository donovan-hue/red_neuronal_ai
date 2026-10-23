# red_neuronal_ai

Este repositorio contiene **dos demostradores separados**:

- **Red neuronal clásica** en la raíz (`model.py`, `main.py`, `api.py`). `Procfile` despliega este servicio Flask legado.
- **NEXUS** en `nexus/`, un prototipo de agente con motor local/API, RAG, memoria y benchmark.

No comparten checkpoints ni rutas de API.

## Demostrador clásico

```bash
python -m pip install -r requirements.txt
python main.py                 # entrena XOR y escribe modelo.npz
python predecir.py
python interactuar.py
```

El checkpoint NPZ se carga con `allow_pickle=False`. No se admiten checkpoints pickle.

API: `POST /predecir`, `POST /entrenar` y `GET /health`. Los cuerpos JSON se validan y el entrenamiento tiene límites de tamaño. `POST /entrenar` requiere `API_TRAIN_TOKEN` y el header `X-API-Train-Token`.

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
