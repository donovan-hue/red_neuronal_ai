"""Entrena la red neuronal clásica sobre XOR y guarda el checkpoint NPZ."""
import numpy as np

from model import RedNeuronal


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    nn = RedNeuronal(input_size=2, hidden_size=3, output_size=1)
    nn.train(X, y, epochs=10_000, lr=0.1)
    nn.guardar()
    print("Modelo guardado exitosamente en modelo.npz.")
    print("Predicciones finales:")
    print(nn.forward(X).round())


if __name__ == "__main__":
    main()
