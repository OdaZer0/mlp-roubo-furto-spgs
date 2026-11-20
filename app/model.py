"""Inferência da MLP só com NumPy (pesos exportados do Keras no notebook)."""
import json, os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
META = json.load(open(os.path.join(BASE, "meta.json"), encoding="utf-8"))
_W = np.load(os.path.join(BASE, "mlp_weights.npz"))
FEATURES = META["features"]
CATEGORIAS = META["categorias"]
_N = len(META["ativacoes"])


def codificar(entrada: dict) -> np.ndarray:
    """One-hot na mesma ordem do OneHotEncoder do treino; valor desconhecido -> zeros."""
    partes = []
    for f in FEATURES:
        cats = CATEGORIAS[f]
        v = np.zeros(len(cats), dtype=np.float32)
        if entrada.get(f) in cats:
            v[cats.index(entrada[f])] = 1.0
        partes.append(v)
    return np.concatenate(partes)


def prever(entrada: dict) -> dict:
    x = codificar(entrada)
    for i, act in enumerate(META["ativacoes"]):
        x = x @ _W[f"W{i}"] + _W[f"b{i}"]
        x = np.maximum(x, 0) if act == "relu" else np.tanh(x) if act == "tanh" else 1 / (1 + np.exp(-x))
    p = float(x[0])
    return {"roubo": p, "furto": 1 - p, "classe": "roubo" if p >= 0.5 else "furto"}
