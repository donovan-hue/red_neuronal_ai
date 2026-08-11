import numpy as np
from model import RedNeuronal

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = RedNeuronal(input_size=2, hidden_size=3, output_size=1)
nn.train(X, y, epochs=10000, lr=0.1)

nn.guardar()
print("Modelo guardado exitosamente.")
print("Predicciones finales:")
print(nn.forward(X).round())
