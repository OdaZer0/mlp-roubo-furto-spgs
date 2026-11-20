from flask import Flask, render_template, request, jsonify
from model import prever, FEATURES, CATEGORIAS, META

app = Flask(__name__)

ROTULOS = {"bairro": "Bairro", "periodo": "Período do dia", "dia_semana": "Dia da semana",
           "tipo_local": "Tipo de local", "subtipo_local": "Subtipo de local", "marca": "Marca do celular"}
PADRAO = {"bairro": "OUTROS", "periodo": "noite", "dia_semana": "sexta", "tipo_local": "Via Pública",
          "subtipo_local": "Via Pública", "marca": "Apple"}


def _opcoes(f):
    cats = CATEGORIAS[f]
    if f == "dia_semana":
        ordem = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
        return [c for c in ordem if c in cats]
    if f == "periodo":
        ordem = ["madrugada", "manhã", "tarde", "noite", "não informado"]
        return [c for c in ordem if c in cats]
    return sorted(cats)


@app.route("/", methods=["GET", "POST"])
def index():
    valores = dict(PADRAO)
    resultado = None
    if request.method == "POST":
        valores = {f: request.form.get(f, PADRAO[f]) for f in FEATURES}
        resultado = prever(valores)
    return render_template("index.html", features=FEATURES, rotulos=ROTULOS, opcoes={f: _opcoes(f) for f in FEATURES},
                           valores=valores, resultado=resultado, meta=META)


@app.route("/api/predict", methods=["POST"])
def api():
    dados = request.get_json(force=True, silent=True) or {}
    faltando = [f for f in FEATURES if f not in dados]
    if faltando:
        return jsonify(erro=f"campos ausentes: {faltando}"), 400
    return jsonify(prever(dados))


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
