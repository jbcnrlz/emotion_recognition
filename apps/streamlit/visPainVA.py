import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Valence-Arousal & Pain Analyzer", layout="wide")

st.title("🎯 Análise Dimensional: Valência, Arousal e Índice de Dor")
st.markdown("Inspeção da trajetória emocional no plano cartesiano e estimativa heurística de dor (Zero-Shot).")

st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Faça o upload do CSV gerado (ex: features_va.csv)", type=['csv'])

if uploaded_file is not None:
    with st.spinner('Processando predições...'):
        df = pd.read_csv(uploaded_file)
        
        # Extrai o identificador do vídeo (Sweep) assumindo o padrão de pastas do MIntPAIN
        # Ex: ".../Sub01_Trial01_Sweep01_Label0/RGB/img.jpg" -> "Sub01_Trial01_Sweep01_Label0"
        df['video_id'] = df['file'].apply(lambda x: x.replace('\\', '/').split('/')[-3] if len(x.split('/')) >= 3 else 'Unknown')
        
        # Ordenação temporal garantida pelo nome do frame
        df['filename'] = df['file'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
        df = df.sort_values(by=['video_id', 'filename']).reset_index(drop=True)

        # Cálculo da Heurística de Dor Geométrica
        # P = max(0, -V) * sqrt(V^2 + A^2)
        df['valence_penalty'] = df['valence'].apply(lambda v: max(0, -v))
        df['magnitude'] = np.sqrt(df['valence']**2 + df['arousal']**2)
        df['heuristic_pain'] = df['valence_penalty'] * df['magnitude']

    video_ids = sorted(df['video_id'].unique())
    selected_video = st.sidebar.selectbox("Selecione o Vídeo/Ensaio", video_ids)
    
    video_df = df[df['video_id'] == selected_video].reset_index(drop=True)
    
    st.write(f"### Analisando: `{selected_video}` ({len(video_df)} frames)")
    
    col1, col2 = st.columns([1, 1])
    
    # ==========================================
    # GRÁFICO 1: O PLANO CIRCUNPLEXO DE RUSSELL
    # ==========================================
    with col1:
        st.subheader("1. Trajetória Emocional (Espaço VA)")
        
        fig_2d = go.Figure()
        
        # Linhas dos Eixos
        fig_2d.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_2d.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Trajetória dos frames (linha conectando os pontos no tempo)
        fig_2d.add_trace(go.Scatter(
            x=video_df['valence'],
            y=video_df['arousal'],
            mode='lines+markers',
            marker=dict(
                size=6,
                color=video_df.index, # Colore do início (escuro) ao fim (claro) do vídeo
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Tempo (Frame)")
            ),
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
            text=video_df['filename'],
            hovertemplate="<b>%{text}</b><br>Valence: %{x:.3f}<br>Arousal: %{y:.3f}<extra></extra>"
        ))
        
        # Anotações dos quadrantes
        annotations = [
            dict(x=0.8, y=0.8, text="Feliz / Excitado", showarrow=False, font=dict(color="green")),
            dict(x=-0.8, y=0.8, text="Dor Aguda / Medo", showarrow=False, font=dict(color="red")),
            dict(x=-0.8, y=-0.8, text="Dor Crônica / Triste", showarrow=False, font=dict(color="orange")),
            dict(x=0.8, y=-0.8, text="Calmo / Relaxado", showarrow=False, font=dict(color="blue")),
        ]
        
        fig_2d.update_layout(
            xaxis=dict(title="Valência (Negativa ← → Positiva)", range=[-1.1, 1.1], zeroline=False),
            yaxis=dict(title="Arousal (Baixo ← → Alto)", range=[-1.1, 1.1], zeroline=False),
            annotations=annotations,
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    # ==========================================
    # GRÁFICO 2: EVOLUÇÃO TEMPORAL (V, A e Dor)
    # ==========================================
    with col2:
        st.subheader("2. Evolução Temporal e Dor Heurística")
        
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Linha de Valência
        fig_time.add_trace(
            go.Scatter(x=video_df.index, y=video_df['valence'], name="Valência", line=dict(color='blue', width=2)),
            secondary_y=False,
        )
        
        # Linha de Arousal
        fig_time.add_trace(
            go.Scatter(x=video_df.index, y=video_df['arousal'], name="Arousal", line=dict(color='orange', width=2)),
            secondary_y=False,
        )
        
        # Linha do Índice de Dor Heurístico (Eixo Y Secundário)
        fig_time.add_trace(
            go.Scatter(
                x=video_df.index, 
                y=video_df['heuristic_pain'], 
                name="Dor Heurística (P)", 
                line=dict(color='red', width=3, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            secondary_y=True,
        )
        
        fig_time.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Frames (Linha do Tempo)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_time.update_yaxes(title_text="Escala V/A (-1 a 1)", range=[-1.1, 1.1], secondary_y=False)
        fig_time.update_yaxes(title_text="Índice de Dor (0 a +)", range=[0, 1.5], secondary_y=True)
        
        st.plotly_chart(fig_time, use_container_width=True)

    with st.expander("Visualizar Dados do Dataframe"):
        st.dataframe(video_df[['filename', 'valence', 'arousal', 'heuristic_pain']].style.background_gradient(cmap='Reds', subset=['heuristic_pain']))

else:
    st.info("👈 Faça o upload do arquivo CSV extraído pelo modelo Glint360k-VA para iniciar.")