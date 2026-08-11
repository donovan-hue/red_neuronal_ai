import numpy as np
from model import RedNeuronal

nn = RedNeuronal.cargar("modelo.pkl")
print("--- Probador Interactivo de la IA ---")
print("Ingresa dos números (0 o 1) separados por espacio, por ejemplo: 1 0")
print("Escribe 'salir' para terminar.\n")

while True:
    entrada = input("Valores (ej. 1 0): ")
    if entrada.lower() == 'salir':
        break
    try:
        vals = [float(x) for x in entrada.split()]
        if len(vals) != 2:
            print("Deben ser exactamente dos números.")
            continue
        X_test = np.array([vals])
        pred = nn.forward(X_test)
        print(f"Predicción: {pred[0][0]:.4f} (Redondeado: {int(round(pred[0][0]))})\n")
    except Exception as e:
        print("Entrada inválida. Intenta de nuevo.")
