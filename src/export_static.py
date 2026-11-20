"""Gera docs/index.html: versão estática do app (MLP em JavaScript, sem servidor) para o GitHub Pages.
Uso: python src/export_static.py   (lê models/meta.json e models/mlp_weights.npz)"""
import json, numpy as np, os
meta = json.load(open("models/meta.json", encoding="utf-8"))
w = np.load("models/mlp_weights.npz")
n = len(meta["ativacoes"])
pesos = {"W": [np.round(w[f"W{i}"], 6).tolist() for i in range(n)], "b": [np.round(w[f"b{i}"], 6).tolist() for i in range(n)]}
dados = {"features": meta["features"], "categorias": meta["categorias"], "ativacoes": meta["ativacoes"], "pesos": pesos,
         "modelo": meta["modelo"], "acc": meta["metricas_teste"]["acc"], "f1": meta["metricas_teste"]["f1"], "n": meta["n_registros"]}
html = open("app/static_template.html", encoding="utf-8").read().replace("/*DADOS*/null", json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
os.makedirs("docs", exist_ok=True)
open("docs/index.html", "w", encoding="utf-8").write(html)
print("docs/index.html", len(html) // 1024, "KB")
