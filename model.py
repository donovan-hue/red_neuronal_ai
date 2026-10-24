"""Red neuronal pequeña del demostrador clásico; serialización segura en NPZ."""
import os
import tempfile
from pathlib import Path

import numpy as np


class RedNeuronal:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    @staticmethod
    def sigmoid_derivada(x):
        return x * (1.0 - x)

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError(f"X debe tener forma (n, {self.W1.shape[0]})")
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def train(self, X, y, epochs, lr):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError(f"X debe tener forma (n, {self.W1.shape[0]})")
        if y.shape != (X.shape[0], self.W2.shape[1]):
            raise ValueError(f"y debe tener forma ({X.shape[0]}, {self.W2.shape[1]})")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X e y deben contener valores finitos")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs debe ser un entero positivo")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr debe ser positivo y finito")
        for _ in range(epochs):
            output = self.forward(X)
            d_output = (y - output) * self.sigmoid_derivada(output)
            d_hidden = (d_output @ self.W2.T) * self.sigmoid_derivada(self.A1)
            self.W2 += self.A1.T @ d_output * lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
            self.W1 += X.T @ d_hidden * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    def guardar(self, filename="modelo.npz"):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def cargar(filename="modelo.npz"):
        path = Path(filename)
        if path.suffix.lower() != ".npz":
            raise ValueError("Solo se admiten checkpoints .npz seguros; convierte el checkpoint pickle antiguo")
        with np.load(path, allow_pickle=False) as data:
            W1, b1, W2, b2 = (np.array(data[k], dtype=np.float64) for k in ("W1", "b1", "W2", "b2"))
        if W1.ndim != 2 or W2.ndim != 2 or b1.shape != (1, W1.shape[1]) or b2.shape != (1, W2.shape[1]) or W2.shape[0] != W1.shape[1]:
            raise ValueError("Checkpoint neuronal con dimensiones inválidas")
        model = RedNeuronal(W1.shape[0], W1.shape[1], W2.shape[1])
        model.W1, model.b1, model.W2, model.b2 = W1, b1, W2, b2
        return model
