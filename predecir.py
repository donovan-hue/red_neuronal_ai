"""Predicciones del demostrador neuronal clásico."""
import numpy as np

from model import RedNeuronal


def main():
    nn = RedNeuronal.cargar("modelo.npz")
    X_nuevo = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    print("Predicciones usando el modelo cargado:")
    print(nn.forward(X_nuevo).round())


if __name__ == "__main__":
    main()
