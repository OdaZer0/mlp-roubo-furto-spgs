"""Preparação dos dados: SSP-SP, Celulares Subtraídos 2026 -> amostra rotulada (roubo=1, furto=0).

Uso:  python src/data_prep.py data/CelularesSubtraidos_2026.xlsx
Gera: data/amostra_20k.csv
"""
import sys, unicodedata
import pandas as pd

SEED = 42
N_AMOSTRA = 20_000
TOP_BAIRROS = 100
TOP_SUBTIPOS = 15
TOP_MARCAS = 6
DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def periodo_do_dia(row):
    """Faixa do dia a partir da hora; se ausente, usa DESCR_PERIODO; senão 'não informado'."""
    h = row["HORA_OCORRENCIA"]
    if pd.notna(h):
        h = h.hour if hasattr(h, "hour") else int(str(h).split(":")[0])  # alguns valores vêm como texto 'HH:MM:SS'
        return ("madrugada" if h < 6 else "manhã" if h < 12 else "tarde" if h < 18 else "noite")
    p = row["DESCR_PERIODO"]
    mapa = {"De madrugada": "madrugada", "Pela manhã": "manhã", "A tarde": "tarde", "A noite": "noite"}
    return mapa.get(p, "não informado")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper().strip()
    return " ".join(s.split())


def carregar(path):
    return pd.read_excel(path, sheet_name="CELULAR_2026")


def preparar(df, relatorio=None):
    log = relatorio if relatorio is not None else {}
    log["registros_brutos"] = len(df)
    # 1) recorte geográfico: CIDADE = local dos FATOS (NOME_MUNICIPIO = local de registro do BO)
    df = df[df["CIDADE"] == "S.PAULO"]
    log["capital"] = len(df)
    # 2) rótulo: apenas Roubo (art. 157) e Furto (art. 155)
    df = df[df["RUBRICA"].isin(["Roubo (art. 157)", "Furto (art. 155)"])].copy()
    log["roubo_furto"] = len(df)
    # 3) fatos ocorridos em 2026 (a base traz fatos antigos registrados em 2026)
    df = df[df["DATA_OCORRENCIA_BO"].dt.year == 2026]
    log["fatos_2026"] = len(df)
    # 4) uma linha por boletim; remove BOs com rótulos conflitantes
    df["bo_id"] = df["ANO_BO"].astype(str) + "-" + df["ID_DELEGACIA"].astype(str) + "-" + df["NUM_BO"].astype(str)
    conf = df.groupby("bo_id")["RUBRICA"].nunique()
    conf = conf[conf > 1].index
    log["bos_ambiguos_removidos"] = len(conf)
    df = df[~df["bo_id"].isin(conf)]
    log["linhas_antes_dedup"] = len(df)
    df = df.drop_duplicates("bo_id")
    log["bos_unicos"] = len(df)
    # 5) atributos
    out = pd.DataFrame(index=df.index)
    out["bairro"] = df["BAIRRO"].map(lambda x: norm(x) if pd.notna(x) else "NAO INFORMADO")
    out["periodo"] = df.apply(periodo_do_dia, axis=1)
    out["dia_semana"] = df["DATA_OCORRENCIA_BO"].dt.dayofweek.map(lambda i: DIAS[i])
    out["tipo_local"] = df["DESCR_TIPOLOCAL"].fillna("não informado")
    sub = df["DESCR_SUBTIPOLOCAL"].fillna("não informado")
    out["subtipo_local"] = sub.where(sub.isin(sub.value_counts().index[:TOP_SUBTIPOS]), "Outros")
    marca = df["MARCA_OBJETO"].fillna("Outros")
    out["marca"] = marca.where(marca.isin(marca.value_counts().index[:TOP_MARCAS]), "Outros")
    top_b = out["bairro"].value_counts().index[:TOP_BAIRROS]
    out["bairro"] = out["bairro"].where(out["bairro"].isin(top_b), "OUTROS")
    out["alvo"] = (df["RUBRICA"] == "Roubo (art. 157)").astype(int)  # 1 = roubo, 0 = furto
    return out.reset_index(drop=True), log


def amostrar(out, n=N_AMOSTRA, seed=SEED):
    """Amostra aleatória estratificada pelo alvo (preserva a proporção das classes)."""
    from sklearn.model_selection import train_test_split
    amo, _ = train_test_split(out, train_size=n, stratify=out["alvo"], random_state=seed)
    return amo.reset_index(drop=True)


if __name__ == "__main__":
    df = carregar(sys.argv[1])
    out, log = preparar(df)
    amo = amostrar(out)
    amo.to_csv("data/amostra_20k.csv", index=False)
    print(log, "\namostra:", len(amo), amo.alvo.mean())
