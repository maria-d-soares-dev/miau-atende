import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def avaliar_modelo(modelo, X, y):
    previsoes=modelo.predict(X)
    probabilidades=modelo.predict_proba(X)
    confiancas=probabilidades.max(axis=1)
    relatorio=classification_report(y,previsoes,output_dict=True,zero_division=0)
    labels=list(modelo.classes_)
    metricas={
        'accuracy':float(accuracy_score(y,previsoes)),
        'macro_f1':float(f1_score(y,previsoes,average='macro')),
        'classes':{c:{k:float(v) for k,v in relatorio[c].items() if k in ('precision','recall','f1-score')} for c in labels},
        'labels':labels,
        'matriz_confusao':confusion_matrix(y,previsoes,labels=labels).tolist(),
    }
    resultados=pd.DataFrame({'mensagem':list(X),'categoria_real':list(y),'previsao':previsoes,'confianca':confiancas})
    erros=resultados[resultados.categoria_real != resultados.previsao].sort_values('confianca',ascending=False)
    return metricas, erros
