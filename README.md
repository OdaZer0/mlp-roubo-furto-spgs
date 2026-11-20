# Roubo ou furto? Classificação de ocorrências em São Paulo com MLP

Global Solution – Aprendizado Profundo com Redes Neurais Convolucionais

Integrantes: Enzo Hitoshi Oda (RM567313), Ana Carolina Pereira Lopez (RM568401) e Henrique Fessel Trench (RM567513)

Uma rede neural **MLP** classifica uma ocorrência de **celular subtraído na capital paulista** como **roubo** (art. 157) ou **furto** (art. 155), a partir de bairro, período do dia, dia da semana, tipo/subtipo de local e marca do aparelho

Vale deixar claro desde já: o modelo **não prevê** onde nem quando um crime vai acontecer e **não calcula risco**. Ele só estima o *tipo* de uma ocorrência que já foi descrita

**Aplicação web:** https://odazer0.github.io/mlp-roubo-furto-spgs/

## Resultados (conjunto de teste, usado uma única vez)

| Modelo escolhido | Acurácia | Precision (roubo) | Recall (roubo) | F1 (roubo) |
|---|---:|---:|---:|---:|
| D – MLP 64-32, tanh, L2, lr 5e-4 | 71,9% | 64,0% | 45,7% | 53,3% |

O baseline "sempre furto" acerta 64,9%. As 4 arquiteturas que testamos ficaram bem próximas (acurácia de validação entre 69,6% e 70,0%), então a limitação tá nos atributos mesmo, não na capacidade da rede. Detalhes e discussão no notebook

## Fonte dos dados

- **SSP-SP – Portal de Estatísticas → Consultas → Bases de Dados → Celulares Subtraídos**: <https://www.ssp.sp.gov.br/estatistica/consultas>
- Arquivo usado: `CelularesSubtraidos_2026.xlsx` (aba `CELULAR_2026`, fatos de jan–set/2026)
- **Recorte:** `CIDADE == "S.PAULO"` (local dos fatos) → `RUBRICA` ∈ {Roubo, Furto} → fatos de 2026 → remoção de 26 BOs com rótulo ambíguo → 1 linha por boletim (110.594) → amostra estratificada de 20.000
- A base original (~50 MB) **não está no repositório**. A amostra já preparada tá em `data/amostra_20k.csv`, sem dados pessoais

## Estrutura

```
├── notebook_mlp_roubo_furto.ipynb   # pipeline completo e documentado (4 MLPs, métricas, curvas, matriz de confusão)
├── src/data_prep.py                 # xlsx da SSP -> data/amostra_20k.csv
├── src/export_static.py             # gera docs/index.html a partir dos pesos em models/
├── data/amostra_20k.csv             # amostra usada no treino
├── models/                          # modelo treinado (.keras), pesos (.npz), encoder, meta.json
├── docs/index.html                  # aplicação web publicada no GitHub Pages (MLP em JavaScript)
├── app/                             # versão Flask da aplicação (inferência NumPy)
└── requirements.txt / app/requirements.txt
```

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. (Opcional) refazer a amostra a partir da SSP.** Baixa o `.xlsx` no link acima, salva em `data/` e roda

```bash
python src/data_prep.py data/CelularesSubtraidos_2026.xlsx
```

**2. Reproduzir o treino.** Abre e executa o notebook. As sementes são fixas, mas o resultado pode variar um pouco de uma máquina pra outra (versão do TensorFlow, por exemplo). Ele regrava a pasta `models/`

```bash
jupyter nbconvert --to notebook --execute --inplace notebook_mlp_roubo_furto.ipynb
```

**3. Rodar o app Flask localmente**

```bash
cd app && python app.py        # http://localhost:5000
```

O app não precisa de TensorFlow: faz a inferência só com NumPy, usando `models/mlp_weights.npz`

Tem também uma API opcional: `POST /api/predict` com JSON `{"bairro","periodo","dia_semana","tipo_local","subtipo_local","marca"}` que devolve `{"classe","roubo","furto"}`

## Aplicação web

A versão publicada é uma página estática (`docs/index.html`) servida pelo GitHub Pages. A MLP roda no próprio navegador, com os pesos do modelo escolhido embutidos na página, então não precisa de servidor. Ela mostra os mesmos 6 atributos do treino, a classe estimada e as probabilidades. Se o modelo for retreinado, dá pra regenerar a página com `python src/export_static.py`

A versão Flask (`app/`) faz a mesma coisa mas com servidor. Dá pra publicar no Render, por exemplo (Build: `pip install -r app/requirements.txt`, Start: `cd app && gunicorn app:app`)

## Uso de IA generativa

A gente usou a IA pra ajudar em algumas dúvidas que surgiram com o dataset e em alguns erros que apareceram no meio do caminho