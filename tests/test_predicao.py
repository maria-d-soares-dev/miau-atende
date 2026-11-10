import pytest
from src.preprocessamento import limpar_texto
from src.predicao import classificar
from src.treinamento import criar_pipeline

def modelo_minimo():
    X=['pedido atrasou muito','comida chegou fria','faltou minha bebida','pedido veio trocado','cobrou duas vezes','adorei estava perfeito']
    y=['ATRASO','QUALIDADE','ITEM_FALTANTE','PEDIDO_INCORRETO','PAGAMENTO','ELOGIO']
    return criar_pipeline().fit(X,y)

def test_limpar_texto_preserva_negacao():
    assert limpar_texto('  NÃO   chegou  ') == 'não chegou'

def test_classificar_retorna_campos():
    r=classificar('a comida chegou fria',modelo_minimo())
    assert set(r)=={'categoria','confianca','probabilidades'}
    assert 0 <= r['confianca'] <= 1

def test_texto_vazio():
    with pytest.raises(ValueError): classificar('   ',modelo_minimo())
