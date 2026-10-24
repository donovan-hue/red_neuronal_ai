"""Probador interactivo del demostrador neuronal clásico."""
import numpy as np

from model import RedNeuronal


def main():
    nn = RedNeuronal.cargar("modelo.npz")
    print("--- Probador Interactivo de la IA ---")
    print("Ingresa dos números separados por espacio, por ejemplo: 1 0")
    print("Escribe 'salir' para terminar.\n")
    while True:
        try:
            entrada = input("Valores (ej. 1 0): ").strip()
        except EOFError:
            print()
            break
        if entrada.lower() == "salir":
            break
        try:
            vals = [float(x) for x in entrada.split()]
            if len(vals) != 2 or not np.isfinite(vals).all():
                print("Deben ser exactamente dos números finitos.")
                continue
            pred = float(nn.forward(np.array([vals], dtype=np.float64))[0, 0])
            print(f"Predicción: {pred:.4f} (Redondeado: {int(pred >= 0.5)})\n")
        except (ValueError, TypeError):
            print("Entrada inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
