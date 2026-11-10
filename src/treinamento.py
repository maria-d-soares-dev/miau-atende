from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocessamento import carregar_dados
from src.avaliacao import avaliar_modelo

RAIZ = Path(__file__).resolve().parents[1]

def criar_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classificador", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

def dividir_dados(df: pd.DataFrame):
    X_treino, X_temp, y_treino, y_temp = train_test_split(
        df["mensagem"], df["categoria"], test_size=0.30, stratify=df["categoria"], random_state=42)
    X_valid, X_teste, y_valid, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_treino, X_valid, X_teste, y_treino, y_valid, y_teste

def treinar(caminho_dados=RAIZ/'data/mensagens.csv'):
    df=carregar_dados(caminho_dados)
    X_tr, X_va, X_te, y_tr, y_va, y_te = dividir_dados(df)
    modelo=criar_pipeline(); modelo.fit(X_tr, y_tr)
    metricas_valid, _ = avaliar_modelo(modelo, X_va, y_va)
    metricas_teste, erros = avaliar_modelo(modelo, X_te, y_te)
    (RAIZ/'models').mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ/'models/modelo.joblib')
    erros.to_csv(RAIZ/'models/erros_teste.csv', index=False)
    with open(RAIZ/'models/metricas.json','w',encoding='utf-8') as f:
        json.dump({'validacao':metricas_valid,'teste':metricas_teste},f,ensure_ascii=False,indent=2)
    pd.DataFrame({'mensagem':X_te,'categoria_real':y_te}).to_csv(RAIZ/'models/teste.csv',index=False)
    return modelo, metricas_teste

if __name__ == '__main__':
    _, m=treinar(); print(json.dumps(m,ensure_ascii=False,indent=2))
