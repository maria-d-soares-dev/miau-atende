# 🐱 Miau Atende

Classificador educacional de mensagens sintéticas de atendimento de delivery usando NLP, TF-IDF e Logistic Regression.

## Objetivo
Demonstrar o ciclo de Machine Learning: definição do problema, criação e inspeção de dados, preparação, divisão estratificada, treinamento, avaliação, análise de erros e disponibilização em Streamlit.

## Categorias
`ATRASO`, `ITEM_FALTANTE`, `PEDIDO_INCORRETO`, `QUALIDADE`, `PAGAMENTO` e `ELOGIO`.

## Arquitetura
1. `preprocessamento.py`: valida, normaliza e remove duplicidades.
2. `treinamento.py`: divide 70/15/15 e treina o pipeline.
3. `avaliacao.py`: calcula accuracy, macro F1, métricas por classe, matriz e erros.
4. `predicao.py`: carrega o modelo e classifica mensagens novas.
5. `app.py`: apresenta classificação, desempenho, erros e limitações.

## Executar
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.treinamento
pytest -q
streamlit run app.py
```

## Decisões técnicas
- **TF-IDF com unigramas e bigramas:** baseline simples, rápido e explicável para texto curto.
- **Logistic Regression:** classificação multiclasse com `predict_proba`, adequada para um baseline.
- **Macro F1:** dá o mesmo peso a todas as classes.
- **Sem remoção de stopwords:** preserva negações como “não”.
- **Dados sintéticos:** protege privacidade, porém limita generalização.

## Limitações
Este modelo não está pronto para produção. A amostra é pequena e sintética; mensagens com múltiplos assuntos recebem apenas um rótulo; a confiança é probabilidade do classificador, não garantia de acerto.

## Uso responsável de IA
A IA foi usada como apoio no planejamento e na implementação. As decisões, testes, avaliação e documentação devem ser compreendidos e revisados pela autora antes de apresentar o projeto.

## 🎨 Identidade Visual

O Miau Atende utiliza uma identidade visual inspirada em um assistente felino.

A proposta é tornar o projeto mais amigável sem comprometer seu caráter técnico.

Foram utilizados:

- tema rosa pastel;
- elementos visuais inspirados em gatos;
- interface Streamlit personalizada;
- foco em experiência do usuário;
- destaque para métricas e avaliação do modelo.

A identidade visual complementa o objetivo educacional do projeto e mantém a análise de Machine Learning como elemento principal.

