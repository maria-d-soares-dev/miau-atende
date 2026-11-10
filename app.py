from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predicao import classificar, carregar_modelo

RAIZ = Path(__file__).resolve().parent

st.set_page_config(
    page_title="🐱 Miau Atende",
    page_icon="🐱",
    layout="wide"
)

# ==========================================
# TEMA VISUAL
# ==========================================

st.markdown("""
<style>

/* ==========================================
   APP
========================================== */

[data-testid="stAppViewContainer"]{
    background-color:#FFF7F3;
}

/* ==========================================
   CONTAINER PRINCIPAL
========================================== */

.main .block-container{
    padding-top:2rem;
}

/* ==========================================
   FONTE GERAL
========================================== */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"] {
    font-family:  "Courier New", "Lucida Console", monospace;

    color:#2F2F2F;
}

/* ==========================================
   SIDEBAR
========================================== */

[data-testid="stSidebar"]{
    background-color:#FCE7F3;
}

[data-testid="stSidebar"] *{
    color:#7A284E !important;
}

/* ==========================================
   TITULOS
========================================== */

h1{
    color:#C2185B !important;
    font-weight:700;
}

h2,
h3{
    color:#A61E4D !important;
}

/* ==========================================
   TEXTO NORMAL
========================================== */

p{
    color:#444444 !important;
}

/* ==========================================
   SELECTBOX
========================================== */

[data-testid="stSelectbox"] div[data-baseweb="select"]{
    background-color:#FFF0F5 !important;
    border-radius:12px !important;
}

/* ==========================================
   TEXTAREA
========================================== */

[data-testid="stTextArea"] textarea{
    background-color:#FFF0F5 !important;
    color:#333333 !important;
    border:2px solid #F9A8D4 !important;
    border-radius:15px !important;
}

/* ==========================================
   BOTÕES
========================================== */

.stButton button{
    background-color:#EC4899 !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:600 !important;
}

.stButton button:hover{
    background-color:#DB2777 !important;
}

/* ==========================================
   METRICAS
========================================== */

[data-testid="stMetric"]{
    background-color:#FFE4EC;
    border-radius:15px;
    padding:15px;
}

[data-testid="stMetricValue"]{
    color:#831843 !important;
    font-weight:700;
}

[data-testid="stMetricLabel"]{
    color:#BE185D !important;
}

/* ==========================================
   DATAFRAME
========================================== */

[data-testid="stDataFrame"]{
    background:white;
    border-radius:12px;
}

/* ==========================================
   ALERTAS
========================================== */

[data-testid="stSuccess"]{
    border-radius:12px;
}

[data-testid="stInfo"]{
    border-radius:12px;
}

[data-testid="stWarning"]{
    border-radius:12px;
}

</style>

""", unsafe_allow_html=True)

# ==========================================
# CACHE
# ==========================================

@st.cache_resource
def modelo():
    return carregar_modelo()

@st.cache_data
def dados():
    return pd.read_csv(
        RAIZ / "data/mensagens.csv"
    )

# ==========================================
# CABEÇALHO
# ==========================================

st.title("🐱 Miau Atende")

st.markdown("""
### Seu assistente felino para triagem de mensagens

O Miau Atende utiliza NLP e Machine Learning para ajudar
na organização de mensagens de atendimento de delivery.

🌸 Projeto educacional desenvolvido com dados sintéticos.
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🐾 Navegação")

st.sidebar.info(
    """
O gatinho analisa mensagens
antes do encaminhamento humano.
"""
)

pagina = st.sidebar.radio(
    "Escolha uma página",
    [
        "Classificar mensagem",
        "Desempenho",
        "Onde o modelo erra",
        "Sobre os dados"
    ]
)

# ==========================================
# PAGINA 1
# ==========================================

if pagina == "Classificar mensagem":

    st.subheader("🐾 Central de Triagem Felina")

    exemplo = st.selectbox(
        "Exemplo opcional",
        [
            "",
            "Meu pedido ainda não chegou.",
            "A comida veio fria.",
            "Esqueceram minha bebida."
        ]
    )

    texto = st.text_area(
        "Mensagem do cliente",
        value=exemplo,
        height=140
    )

    if st.button(
        "🐱 Analisar Mensagem",
        type="primary"
    ):

        if not texto.strip():

            st.warning(
                "Digite uma mensagem."
            )

        else:

            with st.spinner(
                "🐾 O gatinho está analisando a mensagem..."
            ):
                resultado = classificar(
                    texto,
                    modelo()
                )

            st.success(
                f"🐱 Categoria prevista: {resultado['categoria']}"
            )

            st.info(
                f"🌸 Confiança aproximada: {resultado['confianca']:.1%}"
            )

            st.subheader(
                "Probabilidades"
            )

            st.bar_chart(
                pd.Series(
                    resultado["probabilidades"]
                )
            )

            st.warning(
                """
Projeto educacional treinado apenas com dados sintéticos.

Não deve ser utilizado em produção.
"""
            )

# ==========================================
# PAGINA 2
# ==========================================

elif pagina == "Desempenho":

    st.subheader(
        "📊 Desempenho do Modelo"
    )

    df = dados()

    st.metric(
        "Mensagens sintéticas",
        len(df)
    )

    distribuicao = (
        df["categoria"]
        .value_counts()
        .rename_axis("categoria")
        .reset_index(name="quantidade")
    )

    grafico = px.bar(
        distribuicao,
        x="categoria",
        y="quantidade",
        color="categoria",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(
        grafico,
        use_container_width=True
    )

    with open(
        RAIZ / "models/metricas.json",
        encoding="utf-8"
    ) as arquivo:

        metricas = json.load(
            arquivo
        )["teste"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{metricas['accuracy']:.1%}"
    )

    col2.metric(
        "Macro F1",
        f"{metricas['macro_f1']:.1%}"
    )

    st.subheader(
        "Métricas por Categoria"
    )

    st.dataframe(
        pd.DataFrame(
            metricas["classes"]
        ).T,
        use_container_width=True
    )

    matriz = pd.DataFrame(
        metricas["matriz_confusao"],
        index=metricas["labels"],
        columns=metricas["labels"]
    )

    fig = px.imshow(
        matriz,
        text_auto=True,
        color_continuous_scale="Pinkyl",
        labels={
            "x": "Previsto",
            "y": "Real",
            "color": "Quantidade"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# PAGINA 3
# ==========================================

elif pagina == "Onde o modelo erra":

    st.subheader(
        "😿 Onde o gatinho se confunde"
    )

    erros = pd.read_csv(
        RAIZ / "models/erros_teste.csv"
    )

    if erros.empty:

        st.success(
            """
Nenhum erro encontrado nesta divisão.

Isso não significa que o modelo está pronto para uso real.
"""
        )

    else:

        filtro = st.selectbox(
            "Categoria Real",
            ["TODAS"]
            + sorted(
                erros["categoria_real"].unique()
            )
        )

        if filtro != "TODAS":

            erros = erros[
                erros["categoria_real"]
                == filtro
            ]

        st.dataframe(
            erros,
            use_container_width=True
        )

    st.info(
        """
Mensagens ambíguas,
com múltiplos problemas,
ironia,
negações e textos muito curtos
costumam gerar mais erros.
"""
    )

# ==========================================
# PAGINA 4
# ==========================================

else:

    st.subheader(
        "🌸 Sobre o Projeto"
    )

    st.markdown("""
### Dados Sintéticos

🐾 360 mensagens criadas artificialmente

🐾 Nenhum dado real de cliente

🐾 Sem ligação com iFood ou outras plataformas

🐾 Projeto educacional de NLP

🐾 Demonstração do ciclo de Machine Learning

🐾 Não recomendado para produção
""")