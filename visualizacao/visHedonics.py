import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise de Emoções Facial",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #95a5a6; }
</style>
""", unsafe_allow_html=True)

def carregar_dados(uploaded_file):
    """Carrega e processa os dados do arquivo CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def processar_dados_hedonicos(df):
    """Processa dados para análise hedônica"""
    # Classificar emoções em categorias hedônicas
    emocao_hedonica = {
        'positivo': ['happy', 'elated', 'hopeful', 'surprised', 'proud', 'loved'],
        'negativo': ['angry', 'disgusted', 'fearful', 'sad', 'fatigued', 'contempt'],
        'neutro': ['astonished', 'neutral']
    }
    
    df_processed = df.copy()
    
    # Calcular scores hedônicos
    df_processed['score_positivo'] = df_processed[emocao_hedonica['positivo']].sum(axis=1)
    df_processed['score_negativo'] = df_processed[emocao_hedonica['negativo']].sum(axis=1)
    df_processed['score_neutro'] = df_processed[emocao_hedonica['neutro']].sum(axis=1)
    
    # Classificar dominância hedônica
    df_processed['dominancia_hedonica'] = df_processed[['score_positivo', 'score_negativo', 'score_neutro']].idxmax(axis=1)
    df_processed['dominancia_hedonica'] = df_processed['dominancia_hedonica'].str.replace('score_', '')
    
    # Extrair informações da pessoa e frame
    df_processed['pessoa_id'] = df_processed['file'].str.extract(r'pessoa_(\d+)')
    df_processed['frame_num'] = df_processed['file'].str.extract(r'frame_(\d+)').fillna('0').astype(int)
    
    return df_processed, emocao_hedonica

def criar_visualizacao_geral(df, df_processed):
    """Cria visualizações gerais do dataset"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição Geral das Emoções")
        
        # Calcular médias
        emocao_medias = df.drop('file', axis=1).mean().sort_values(ascending=True)
        
        # Gráfico de barras horizontal
        fig = px.bar(
            x=emocao_medias.values,
            y=emocao_medias.index,
            orientation='h',
            title='Probabilidade Média de Cada Emoção',
            labels={'x': 'Probabilidade Média', 'y': 'Emoção'}
        )
        fig.update_traces(marker_color='lightblue')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Análise Hedônica")
        
        # Distribuição hedônica
        hedonica_counts = df_processed['dominancia_hedonica'].value_counts()
        colors = {'positivo': '#2ecc71', 'negativo': '#e74c3c', 'neutro': '#95a5a6'}
        
        fig = px.pie(
            values=hedonica_counts.values,
            names=hedonica_counts.index,
            title='Distribuição Hedônica das Expressões',
            color=hedonica_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)

def criar_metricas_gerais(df, df_processed):
    """Exibe métricas gerais"""
    
    st.subheader("📈 Métricas Gerais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Imagens", len(df))
    
    with col2:
        st.metric("Pessoas Únicas", df_processed['pessoa_id'].nunique())
    
    with col3:
        emocao_dominante_geral = df.drop('file', axis=1).mean().idxmax()
        st.metric("Emoção Mais Frequente", emocao_dominante_geral)
    
    with col4:
        hedonica_dominante = df_processed['dominancia_hedonica'].value_counts().idxmax()
        st.metric("Predominância Hedônica", hedonica_dominante.capitalize())

def criar_analise_individual(df_processed):
    """Cria análise individual por pessoa"""
    
    st.subheader("👤 Análise Individual por Pessoa")
    
    # Selecionar pessoa
    pessoas_unicas = sorted(df_processed['pessoa_id'].unique())
    
    if pessoas_unicas:
        pessoa_selecionada = st.selectbox("Selecione uma pessoa:", pessoas_unicas)
        
        if pessoa_selecionada:
            dados_pessoa = df_processed[df_processed['pessoa_id'] == pessoa_selecionada].sort_values('frame_num')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de pizza das emoções dominantes
                emocao_dominante = dados_pessoa.drop(['file', 'pessoa_id', 'score_positivo', 
                                                   'score_negativo', 'score_neutro', 
                                                   'dominancia_hedonica', 'frame_num'], axis=1)
                emocao_dominante = emocao_dominante.idxmax(axis=1)
                contagem_emocao = emocao_dominante.value_counts()
                
                fig = px.pie(
                    values=contagem_emocao.values,
                    names=contagem_emocao.index,
                    title=f'Distribuição Emocional - Pessoa {pessoa_selecionada}'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Evolução temporal
                st.write(f"**Evolução Temporal - Pessoa {pessoa_selecionada}**")
                
                # Selecionar emoções para plotar
                emocoes_disponiveis = [col for col in dados_pessoa.columns if col not in 
                                     ['file', 'pessoa_id', 'score_positivo', 'score_negativo', 
                                      'score_neutro', 'dominancia_hedonica', 'frame_num']]
                
                emocoes_selecionadas = st.multiselect(
                    "Selecione as emoções para visualizar:",
                    emocoes_disponiveis,
                    default=['hopeful', 'happy', 'neutral'][:min(3, len(emocoes_disponiveis))]
                )
                
                if emocoes_selecionadas and len(dados_pessoa) > 1:
                    fig = go.Figure()
                    for emocao in emocoes_selecionadas:
                        fig.add_trace(go.Scatter(
                            x=dados_pessoa['frame_num'],
                            y=dados_pessoa[emocao],
                            name=emocao,
                            mode='lines+markers'
                        ))
                    
                    fig.update_layout(
                        title=f'Evolução das Emoções - Pessoa {pessoa_selecionada}',
                        xaxis_title='Número do Frame',
                        yaxis_title='Probabilidade',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(dados_pessoa) == 1:
                    st.info("Apenas uma imagem disponível para esta pessoa.")

def criar_analise_detalhada_hedonica(df_processed):
    """Cria análise detalhada hedônica"""
    
    st.subheader("🍽️ Análise Hedônica Detalhada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap de correlação entre emoções positivas
        emocoes_positivas = ['happy', 'elated', 'hopeful', 'surprised', 'proud', 'loved']
        emocoes_positivas_disponiveis = [emo for emo in emocoes_positivas if emo in df_processed.columns]
        
        if len(emocoes_positivas_disponiveis) > 1:
            correlacao_positiva = df_processed[emocoes_positivas_disponiveis].corr()
            
            fig = px.imshow(
                correlacao_positiva,
                title='Correlação entre Emoções Positivas',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Emoções positivas insuficientes para análise de correlação.")
    
    with col2:
        # Heatmap de correlação entre emoções negativas
        emocoes_negativas = ['angry', 'disgusted', 'fearful', 'sad', 'fatigued', 'contempt']
        emocoes_negativas_disponiveis = [emo for emo in emocoes_negativas if emo in df_processed.columns]
        
        if len(emocoes_negativas_disponiveis) > 1:
            correlacao_negativa = df_processed[emocoes_negativas_disponiveis].corr()
            
            fig = px.imshow(
                correlacao_negativa,
                title='Correlação entre Emoções Negativas',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Emoções negativas insuficientes para análise de correlação.")

def criar_relatorio_final(df, df_processed):
    """Cria relatório final com insights"""
    
    st.subheader("📋 Relatório Final - Avaliação do Momento de Consumo")
    
    # Cálculos finais
    emocao_medias = df.drop('file', axis=1).mean().sort_values(ascending=False)
    hedonica_counts = df_processed['dominancia_hedonica'].value_counts()
    hedonica_percent = (hedonica_counts / len(df_processed) * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 5 Emoções Mais Frequentes:**")
        for i, (emocao, prob) in enumerate(emocao_medias.head(5).items(), 1):
            st.write(f"{i}. **{emocao}**: {prob:.3f} ({prob*100:.1f}%)")
    
    with col2:
        st.write("**Distribuição Hedônica:**")
        for categoria in ['positivo', 'negativo', 'neutro']:
            percentual = hedonica_percent.get(categoria, 0)
            cor = "🟢" if categoria == "positivo" else "🔴" if categoria == "negativo" else "⚪"
            st.write(f"{cor} **{categoria.capitalize()}**: {percentual}%")
    
    # Avaliação geral
    st.markdown("---")
    st.subheader("🎯 Avaliação Geral do Momento de Consumo")
    
    percent_positivo = hedonica_percent.get('positivo', 0)
    percent_negativo = hedonica_percent.get('negativo', 0)
    
    if percent_positivo > 60:
        st.success("""
        **✅ EXPERIÊNCIA POSITIVA**
        
        A análise indica uma experiência predominantemente **positiva** durante o consumo. 
        As expressões faciais sugerem satisfação, prazer e engajamento com o produto.
        
        **Recomendações:**
        - Produto bem recebido pelos consumidores
        - Manter características que geram respostas positivas
        - Considerar expansão da linha de produtos
        """)
    elif percent_negativo > 60:
        st.error("""
        **❌ EXPERIÊNCIA NEGATIVA**
        
        A análise indica uma experiência predominantemente **negativa** durante o consumo.
        As expressões faciais sugerem desconforto, desagrado ou insatisfação com o produto.
        
        **Recomendações:**
        - Revisar formulação ou características do produto
        - Realizar pesquisa qualitativa para entender as objeções
        - Considerar reformulação ou reposicionamento
        """)
    else:
        st.warning("""
        **⚠️ EXPERIÊNCIA MISTA**
        
        A análise indica uma experiência **mista** durante o consumo.
        Há uma combinação de respostas positivas e negativas, sugerindo reações variadas ao produto.
        
        **Recomendações:**
        - Segmentar o público-alvo
        - Identificar características específicas que geram respostas positivas/negativas
        - Realizar testes adicionais para entender a variabilidade
        """)

def main():
    """Função principal da aplicação"""
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">😊 Análise de Emoções Faciais</h1>', unsafe_allow_html=True)
    st.markdown("### Upload e análise do arquivo CSV de resultados")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Carregue o arquivo CSV com os resultados das emoções:",
        type=['csv'],
        help="Arquivo deve conter colunas para cada emoção e uma coluna 'file' com o caminho da imagem"
    )
    
    if uploaded_file is not None:
        # Carregar dados
        with st.spinner('Carregando e processando dados...'):
            df = carregar_dados(uploaded_file)
            
            if df is not None:
                # Verificar se as colunas necessárias existem
                colunas_necessarias = ['file']
                colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
                
                if colunas_faltantes:
                    st.error(f"Colunas faltantes no arquivo: {colunas_faltantes}")
                    st.info("O arquivo deve conter uma coluna 'file' com os caminhos das imagens.")
                else:
                    df_processed, emocao_hedonica = processar_dados_hedonicos(df)
                    
                    # Abas para organização
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Visão Geral", 
                        "👤 Análise Individual", 
                        "🍽️ Análise Hedônica",
                        "📋 Relatório Final"
                    ])
                    
                    with tab1:
                        criar_metricas_gerais(df, df_processed)
                        criar_visualizacao_geral(df, df_processed)
                    
                    with tab2:
                        criar_analise_individual(df_processed)
                    
                    with tab3:
                        criar_analise_detalhada_hedonica(df_processed)
                    
                    with tab4:
                        criar_relatorio_final(df, df_processed)
                    
                    # Sidebar com informações adicionais
                    st.sidebar.markdown("## ℹ️ Informações do Dataset")
                    st.sidebar.write(f"**Arquivo:** {uploaded_file.name}")
                    st.sidebar.write(f"**Total de linhas:** {len(df):,}")
                    st.sidebar.write(f"**Total de colunas:** {len(df.columns)}")
                    st.sidebar.write(f"**Pessoas analisadas:** {df_processed['pessoa_id'].nunique()}")
                    
                    st.sidebar.markdown("## 📥 Download de Resultados")
                    
                    # Opção para download dos dados processados
                    csv_processed = df_processed.to_csv(index=False)
                    st.sidebar.download_button(
                        label="📊 Baixar dados processados (CSV)",
                        data=csv_processed,
                        file_name="dados_processados_analise_emocional.csv",
                        mime="text/csv"
                    )
                
    else:
        # Instruções quando não há arquivo
        st.info("""
        ### 📋 Como usar esta aplicação:
        
        1. **Faça upload** do arquivo CSV com os resultados das emoções
        2. **Aguarde** o processamento dos dados
        3. **Navegue** pelas abas para explorar as análises:
           - 📈 **Visão Geral**: Métricas e distribuições gerais
           - 👤 **Análise Individual**: Detalhes por pessoa
           - 🍽️ **Análise Hedônica**: Correlações emocionais
           - 📋 **Relatório Final**: Insights e avaliação
        
        ### 🎯 Formato esperado do CSV:
        - Colunas para cada emoção (happy, sad, angry, etc.)
        - Coluna 'file' com caminho das imagens
        - Valores entre 0 e 1 representando probabilidades
        
        **Exemplo de formato:**
        ```
        happy,angry,sad,neutral,file
        0.8,0.1,0.05,0.05,outputFramesVideo/frames/pessoa_001/frame_000000.jpg
        0.6,0.2,0.1,0.1,outputFramesVideo/frames/pessoa_001/frame_000001.jpg
        ```
        """)

if __name__ == "__main__":
    main()