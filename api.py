"""API Flask para la misma red XOR que usan los clientes de consola."""
import hmac
import os
from pathlib import Path
from threading import RLock

import numpy as np
from flask import Flask, jsonify, request

from model import RedNeuronal

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000
MODEL_PATH = Path(__file__).resolve().with_name("modelo.npz")
_state_lock = RLock()
_training_lock = RLock()
model = RedNeuronal.cargar(str(MODEL_PATH)) if MODEL_PATH.is_file() else RedNeuronal(2, 3, 1)


def _json_object():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _finite_array(value, name, ndim):
    try:
        raw = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
            raise ValueError
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' debe contener números") from None
    if arr.ndim != ndim or arr.size == 0 or not np.isfinite(arr).all() or np.any(np.abs(arr) > 1e6):
        raise ValueError(f"'{name}' tiene dimensiones o valores fuera de rango")
    return arr


@app.post('/predecir')
def predecir():
    data = _json_object()
    if data is None:
        return jsonify({'error': 'Se esperaba un objeto JSON'}), 400
    try:
        x = _finite_array([data.get('x1', 0), data.get('x2', 0)], 'entrada', 1)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    with _state_lock:
        pred = float(model.forward(x.reshape(1, 2))[0, 0])
    return jsonify({'entrada': x.tolist(), 'prediccion': pred, 'redondeado': int(pred >= 0.5)})


@app.post('/entrenar')
def entrenar():
    global model
    data = _json_object()
    if data is None:
        return jsonify({'error': 'Se esperaba un objeto JSON'}), 400
    try:
        X = _finite_array(data.get('X'), 'X', 2)
        y = _finite_array(data.get('y'), 'y', 1)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if X.shape[1] != 2 or X.shape[0] != y.shape[0] or X.shape[0] > 500 or not np.isin(y, [0.0, 1.0]).all():
        return jsonify({'error': 'X debe tener dos columnas, hasta 500 filas y etiquetas 0/1 por fila'}), 400
    epochs = data.get('epochs', 1000)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10_000:
        return jsonify({'error': "'epochs' debe ser un entero entre 1 y 10000"}), 400
    lr = data.get('lr', 0.1)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not np.isfinite(lr) or not 0 < lr <= 1:
        return jsonify({'error': "'lr' debe estar entre 0 y 1"}), 400
    expected = os.environ.get("API_TRAIN_TOKEN", "")
    if not expected:
        return jsonify({'error': 'entrenamiento deshabilitado: configura API_TRAIN_TOKEN'}), 503
    if not hmac.compare_digest(request.headers.get("X-API-Train-Token", ""), expected):
        return jsonify({'error': 'token de entrenamiento inválido'}), 401
    if not _training_lock.acquire(blocking=False):
        return jsonify({'error': 'Ya hay un entrenamiento en curso'}), 409
    try:
        with _state_lock:
            model.train(X, y.reshape(-1, 1), epochs=epochs, lr=float(lr))
            model.guardar(str(MODEL_PATH))
            result = {'mensaje': 'Entrenamiento exitoso', 'modelo': str(MODEL_PATH.name),
                      'predicciones': model.forward(X).reshape(-1).tolist()}
        return jsonify(result)
    except (ValueError, OSError) as exc:
        return jsonify({'error': str(exc)}), 400
    finally:
        _training_lock.release()


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'xor-neural-network'})
