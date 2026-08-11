import numpy as np
from model import RedNeuronal

# Cargar el modelo guardado
nn = RedNeuronal.cargar("modelo.pkl")

# Probar con los datos
X_nuevo = np.array([[0,0], [0,1], [1,0], [1,1]])
print("Predicciones usando el modelo cargado:")
print(nn.forward(X_nuevo).round())
