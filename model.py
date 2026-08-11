import numpy as np

class RedNeuronal:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivada(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            d_output = error * self.sigmoid_derivada(output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivada(self.A1)
            self.W2 += self.A1.T.dot(d_output) * lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
            self.W1 += X.T.dot(d_hidden) * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
