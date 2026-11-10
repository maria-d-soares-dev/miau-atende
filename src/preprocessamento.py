import re
import pandas as pd

COLUNAS_OBRIGATORIAS = {"mensagem", "categoria"}

def limpar_texto(texto: str) -> str:
    """Normaliza caixa e espaços sem remover palavras importantes como 'não'."""
    texto = str(texto).lower().strip()
    return re.sub(r"\s+", " ", texto)

def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")
    df = df.dropna(subset=["mensagem", "categoria"]).copy()
    df["mensagem"] = df["mensagem"].map(limpar_texto)
    df["categoria"] = df["categoria"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["mensagem"]).reset_index(drop=True)
