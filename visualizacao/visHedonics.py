import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Análise de Emoções Facial",
    page_icon="😊",
    layout="wide"
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
    emocao_hedonica = {
        'positivo': ['happy', 'elated', 'hopeful', 'surprised', 'proud', 'loved'],
        'negativo': ['angry', 'disgusted', 'fearful', 'sad', 'fatigued', 'contempt'],
        'neutro': ['astonished', 'neutral']
    }
    
    df_processed = df.copy()
    
    # Calcular scores hedônicos
    for categoria, emocoes in emocao_hedonica.items():
        emocoes_disponiveis = [emo for emo in emocoes if emo in df_processed.columns]
        if emocoes_disponiveis:
            df_processed[f'score_{categoria}'] = df_processed[emocoes_disponiveis].sum(axis=1)
    
    # Classificar dominância hedônica
    score_cols = [col for col in df_processed.columns if col.startswith('score_')]
    if score_cols:
        df_processed['dominancia_hedonica'] = df_processed[score_cols].idxmax(axis=1)
        df_processed['dominancia_hedonica'] = df_processed['dominancia_hedonica'].str.replace('score_', '')
    
    # Extrair informações
    df_processed['pessoa_id'] = df_processed['file'].str.extract(r'pessoa_(\d+)')
    df_processed['frame_num'] = df_processed['file'].str.extract(r'frame_(\d+)').fillna('0').astype(int)
    
    return df_processed, emocao_hedonica

def criar_visualizacao_geral(df, df_processed):
    """Cria visualizações gerais"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição das Emoções")
        emocao_medias = df.drop('file', axis=1).mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=emocao_medias.values,
            y=emocao_medias.index,
            orientation='h',
            title='Probabilidade Média de Cada Emoção',
            labels={'x': 'Probabilidade', 'y': 'Emoção'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Análise Hedônica")
        if 'dominancia_hedonica' in df_processed.columns:
            hedonica_counts = df_processed['dominancia_hedonica'].value_counts()
            colors = {'positivo': '#2ecc71', 'negativo': '#e74c3c', 'neutro': '#95a5a6'}
            
            fig = px.pie(
                values=hedonica_counts.values,
                names=hedonica_counts.index,
                title='Distribuição Hedônica',
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
        pessoas = df_processed['pessoa_id'].nunique() if 'pessoa_id' in df_processed.columns else "N/A"
        st.metric("Pessoas Únicas", pessoas)
    
    with col3:
        emocao_dominante = df.drop('file', axis=1).mean().idxmax()
        st.metric("Emoção Mais Frequente", emocao_dominante)
    
    with col4:
        if 'dominancia_hedonica' in df_processed.columns:
            hedonica_dominante = df_processed['dominancia_hedonica'].value_counts().idxmax()
            st.metric("Predominância", hedonica_dominante.capitalize())

def main():
    st.markdown('<h1 class="main-header">😊 Análise de Emoções Faciais</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Carregue o arquivo CSV:", type=['csv'])
    
    if uploaded_file is not None:
        with st.spinner('Processando dados...'):
            df = carregar_dados(uploaded_file)
            
            if df is not None and 'file' in df.columns:
                df_processed, _ = processar_dados_hedonicos(df)
                
                tab1, tab2 = st.tabs(["📈 Visão Geral", "📋 Relatório"])
                
                with tab1:
                    criar_metricas_gerais(df, df_processed)
                    criar_visualizacao_geral(df, df_processed)
                
                with tab2:
                    st.subheader("🎯 Relatório Final")
                    
                    # Análise final
                    emocao_medias = df.drop('file', axis=1).mean().sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top 5 Emoções:**")
                        for i, (emocao, prob) in enumerate(emocao_medias.head(5).items(), 1):
                            st.write(f"{i}. **{emocao}**: {prob*100:.1f}%")
                    
                    with col2:
                        if 'dominancia_hedonica' in df_processed.columns:
                            st.write("**Distribuição Hedônica:**")
                            hedonica_percent = (df_processed['dominancia_hedonica'].value_counts() / len(df_processed) * 100).round(1)
                            for cat, perc in hedonica_percent.items():
                                st.write(f"• **{cat.capitalize()}**: {perc}%")
                    
                    # Avaliação final
                    if 'dominancia_hedonica' in df_processed.columns:
                        percent_positivo = hedonica_percent.get('positivo', 0)
                        if percent_positivo > 60:
                            st.success("✅ **Experiência Positiva** - Boa aceitação do produto")
                        elif percent_positivo < 40:
                            st.error("❌ **Experiência Negativa** - Necessita melhorias")
                        else:
                            st.warning("⚠️ **Experiência Mista** - Respostas variadas")
                
                # Sidebar info
                st.sidebar.info(f"""
                **📊 Estatísticas:**
                - Imagens: {len(df)}
                - Pessoas: {df_processed['pessoa_id'].nunique()}
                - Emoções: {len(df.columns) - 1}
                """)
                
    else:
        st.info("""
        ### 📋 Como usar:
        1. **Upload** do CSV com resultados de emoções
        2. **Coluna 'file'** com caminhos das imagens
        3. **Colunas de emoções** com valores 0-1
        
        **Exemplo:**
        ```
        happy,angry,neutral,file
        0.8,0.1,0.1,path/frame_001.jpg
        ```
        """)

if __name__ == "__main__":
    main()