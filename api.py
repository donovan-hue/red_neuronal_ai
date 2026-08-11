from flask import Flask, request, jsonify
import numpy as np
from model import RedNeuronal

app = Flask(__name__)
nn = RedNeuronal.cargar("modelo.pkl")

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    x1 = float(data.get('x1', 0))
    x2 = float(data.get('x2', 0))
    
    X = np.array([[x1, x2]])
    pred = nn.forward(X)
    valor_pred = float(pred[0][0])
    
    return jsonify({
        "entrada": [x1, x2],
        "prediccion": valor_pred,
        "redondeado": int(round(valor_pred))
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
