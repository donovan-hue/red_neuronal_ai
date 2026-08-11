import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

weights = np.array([0.5, 0.5])
bias = 0.0

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

@app.route('/predecir', methods=['POST'])
def predecir():
  data = request.get_json()
  x = np.array([data.get('x1', 0), data.get('x2', 0)])
  pred = sigmoid(np.dot(x, weights) + bias)
  return jsonify({
      'entrada': x.tolist(),
      'prediccion': float(pred),
      'redondeado': int(round(pred)),
  })

@app.route('/entrenar', methods=['POST'])
def entrenar():
  global weights, bias
  data = request.get_json()
  X = np.array(data.get('X', []))
  y = np.array(data.get('y', []))
  epochs = data.get('epochs', 1000)
  lr = 0.1

  if len(X) == 0 or len(y) == 0:
    return jsonify({'error': 'Faltan datos de entrenamiento'}), 400

  for _ in range(epochs):
    preds = sigmoid(np.dot(X, weights) + bias)
    error = preds - y
    d_weights = np.dot(X.T, error * preds * (1 - preds)) / len(X)
    d_bias = np.mean(error * preds * (1 - preds))
    weights -= lr * d_weights
    bias -= lr * d_bias

  return jsonify({
      'mensaje': 'Entrenamiento exitoso',
      'pesos_actualizados': weights.tolist(),
      'sesgo_actualizado': float(bias),
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=10000)
