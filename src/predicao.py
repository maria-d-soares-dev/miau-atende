from pathlib import Path
import joblib
from src.preprocessamento import limpar_texto
RAIZ=Path(__file__).resolve().parents[1]

def carregar_modelo(caminho=RAIZ/'models/modelo.joblib'):
    if not Path(caminho).exists():
        raise FileNotFoundError("Modelo não encontrado. Execute: python -m src.treinamento")
    return joblib.load(caminho)

def classificar(texto: str, modelo=None):
    if not str(texto).strip():
        raise ValueError("A mensagem não pode estar vazia.")
    modelo=modelo or carregar_modelo()
    texto=limpar_texto(texto)
    categoria=modelo.predict([texto])[0]
    probabilidades=modelo.predict_proba([texto])[0]
    detalhes=dict(sorted(zip(modelo.classes_,map(float,probabilidades)),key=lambda x:x[1],reverse=True))
    return {'categoria':categoria,'confianca':max(detalhes.values()),'probabilidades':detalhes}
