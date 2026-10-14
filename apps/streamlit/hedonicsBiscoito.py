import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Visualizador Hedônico de Emoções", layout="wide")

st.title("Visualizador de Predição de Emoções 🎭")

# @st.cache_data garante que o Pandas não tenha que refazer esse processamento a cada clique na interface
@st.cache_data
def load_and_parse_data(file):
    # on_bad_lines='skip' já previne aquele erro da vírgula
    df = pd.read_csv(file, on_bad_lines='skip')
    
    # Função para extrair informações do Path
    def extract_from_path(path_str):
        # Transforma para string e garante que as barras estejam no formato padronizado (/)
        path_str = str(path_str).replace('\\', '/')
        parts = path_str.split('/')
        
        try:
            # [-1] Pega o nome do arquivo, ex: "frame_1795.jpg" -> "frame_1795"
            frame_name = parts[-1].split('.')[0] 
            
            # Extrai apenas os números para ordenar o gráfico de linha temporal corretamente
            match = re.search(r'\d+', frame_name)
            frame_num = int(match.group()) if match else 0
            
            aspecto = parts[-2]
            sujeito = parts[-3]
            experimento = parts[-5]
            
            return pd.Series([experimento, sujeito, aspecto, frame_num])
        except Exception:
            return pd.Series(["Desconhecido", "Desconhecido", "Desconhecido", 0])
            
    # Aplica a função para criar novas colunas no dataframe
    df[['experimento', 'sujeito', 'aspecto', 'frame_num']] = df['file'].apply(extract_from_path)
    return df

# 1. Upload do CSV
uploaded_file = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = load_and_parse_data(uploaded_file)
    emotions = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header("🔍 Filtros de Análise")
    
    experimentos = df['experimento'].unique()
    selected_exp = st.sidebar.selectbox("1. Experimento", experimentos)
    
    # Filtra os sujeitos com base no experimento selecionado
    sujeitos = df[df['experimento'] == selected_exp]['sujeito'].unique()
    selected_subj = st.sidebar.selectbox("2. Sujeito (ID)", sujeitos)
    
    # Filtra os aspectos com base no experimento e sujeito selecionados
    aspectos = df[(df['experimento'] == selected_exp) & (df['sujeito'] == selected_subj)]['aspecto'].unique()
    selected_aspect = st.sidebar.selectbox("3. Aspecto Avaliado (Vídeo)", aspectos)
    
    # --- DADOS FILTRADOS DO VÍDEO ---
    # Filtra o dataframe para o vídeo específico e ORDENA pelos números do frame (para o tempo fluir certo no gráfico)
    df_video = df[(df['experimento'] == selected_exp) & 
                  (df['sujeito'] == selected_subj) & 
                  (df['aspecto'] == selected_aspect)].sort_values('frame_num').reset_index(drop=True)
                  
    st.write(f"### Avaliando aspecto: `{selected_aspect}`")
    st.caption(f"**ID:** `{selected_subj}` | **Experimento:** `{selected_exp}` | **Frames totais:** {len(df_video)}")
    st.divider()
    
    # Cria as abas da interface
    tab1, tab2 = st.tabs(["📊 Análise Geral do Vídeo", "🎞️ Explorar Frames no Detalhe"])
    
    # --- ABA 1: VISÃO GERAL (MÉDIAS E LINHA DO TEMPO) ---
    with tab1:
        col_avg, col_timeline = st.columns([1, 2])
        
        with col_avg:
            st.subheader("Média de Emoções")
            # Calcula a média das emoções de todos os frames desse vídeo
            df_mean = df_video[emotions].mean().reset_index()
            df_mean.columns = ['Emoção', 'Frequência Média']
            df_mean = df_mean.sort_values(by='Frequência Média', ascending=False)
            
            fig_bar_mean = px.bar(
                df_mean, x='Emoção', y='Frequência Média',
                range_y=[0, 1], text_auto='.2%', color='Emoção',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bar_mean.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probabilidade Média")
            st.plotly_chart(fig_bar_mean, use_container_width=True)
            
        with col_timeline:
            st.subheader("Evolução Temporal ao Longo do Vídeo")
            # Gráfico de linha mostrando a variação temporal ao longo dos frames
            fig_line = px.line(
                df_video, x='frame_num', y=emotions,
                labels={'frame_num': 'Número do Frame', 'value': 'Probabilidade', 'variable': 'Emoção'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_line.update_layout(yaxis_range=[0, 1], hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)

    # --- ABA 2: VISUALIZADOR FRAME A FRAME ---
    with tab2:
        if not df_video.empty:
            frame_idx = st.slider("Navegue pelos frames deste vídeo", min_value=0, max_value=len(df_video)-1, value=0, step=1)
            
            row_data = df_video.iloc[frame_idx]
            img_path = row_data['file']
            
            emotion_scores = {emo: row_data[emo] for emo in emotions}
            df_chart = pd.DataFrame(list(emotion_scores.items()), columns=['Emoção', 'Confiança'])
            df_chart = df_chart.sort_values(by='Confiança', ascending=False)
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.write(f"**Frame:** `{row_data['frame_num']}`")
                
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                else:
                    st.error(f"⚠️ Imagem não encontrada no disco.\n\nCaminho esperado:\n`{img_path}`")
                    
            with col2:
                st.write("**Distribuição no instante**")
                fig_bar = px.bar(
                    df_chart, x='Emoção', y='Confiança',
                    range_y=[0, 1], text_auto='.2%', color='Emoção',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probabilidade")
                st.plotly_chart(fig_bar, use_container_width=True)