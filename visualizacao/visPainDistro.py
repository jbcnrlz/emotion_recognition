import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração da página para ocupar a tela toda
st.set_page_config(page_title="Pain Index Analyzer", layout="wide")

st.title("📊 Pain Index Estimation Analysis")
st.markdown("Visualização interativa da dinâmica temporal da dor estimada *vs.* anotação humana.")

# Barra lateral para controles
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Faça o upload do CSV de inferência", type=['csv'])

if uploaded_file is not None:
    # 1. Carregamento dos dados
    with st.spinner('Carregando dados...'):
        df = pd.read_csv(uploaded_file)
    
    # 2. Seleção do Vídeo
    video_ids = sorted(df['video_id'].unique())
    selected_video = st.sidebar.selectbox("Selecione o Vídeo (Video ID)", video_ids)
    
    # 3. Filtragem e ordenação temporal
    video_df = df[df['video_id'] == selected_video].sort_values('filename').reset_index(drop=True)
    
    # 4. Cálculo de Métricas Isoladas para o Vídeo
    mae = mean_absolute_error(video_df['label'], video_df['predicted_pain'])
    mse = mean_squared_error(video_df['label'], video_df['predicted_pain'])
    
    st.write(f"### Desempenho no Vídeo ID: `{selected_video}`")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Frames", len(video_df))
    col2.metric("MAE (Erro Absoluto)", f"{mae:.4f}")
    col3.metric("MSE (Erro Quadrático)", f"{mse:.4f}")
    
    st.markdown("---")
    
    # 5. Plotagem Interativa com Plotly
    fig = go.Figure()
    
    # Adiciona a anotação humana (usando degraus - step 'hv')
    fig.add_trace(go.Scatter(
        x=video_df.index, 
        y=video_df['label'], 
        mode='lines', 
        name='Anotação Humana (Real)',
        line=dict(shape='hv', color='black', width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Adiciona a predição contínua do modelo
    fig.add_trace(go.Scatter(
        x=video_df.index, 
        y=video_df['predicted_pain'], 
        mode='lines', 
        name='Índice de Dor Predito',
        line=dict(color='crimson', width=3)
    ))
    
    # Configurações do layout do gráfico
    fig.update_layout(
        title=dict(text='Curva Temporal de Dor', font=dict(size=20)),
        xaxis_title='Frames (Linha do Tempo)',
        yaxis_title='Intensidade da Dor',
        hovermode='x unified', # Mostra os dois valores ao passar o mouse em um frame
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Renderiza o gráfico ocupando a largura do container
    st.plotly_chart(fig, use_container_width=True)
    
    # 6. Exibição da Tabela de Dados (Opcional para inspeção)
    with st.expander("Ver dados brutos deste vídeo"):
        st.dataframe(video_df[['filename', 'label', 'predicted_pain']])

else:
    st.info("👈 Por favor, faça o upload do arquivo CSV gerado pelo script de inferência para iniciar a análise.")