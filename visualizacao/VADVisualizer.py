import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuração da página
st.set_page_config(
    page_title="Visualizador VAD - Valence, Arousal, Dominance",
    page_icon="📊",
    layout="wide"
)

# Título da aplicação
st.title("📊 Visualizador de Distribuição VAD")
st.markdown("Explore as distribuições de Valence, Arousal e Dominance por classe emocional")

# Função para carregar e processar os dados
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Por favor, faça upload de um arquivo CSV")
            return None
        
        # Verificar se as colunas necessárias estão presentes
        required_columns = ['valence', 'arousal', 'dominance', 'class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colunas necessárias não encontradas: {missing_columns}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Faça upload do seu arquivo CSV com dados VAD",
    type=['csv'],
    help="O arquivo deve conter as colunas: valence, arousal, dominance, class"
)

# Se não há arquivo carregado, usar os dados de exemplo
if uploaded_file is None:
    st.info("💡 **Dica:** Faça upload de um arquivo CSV ou use os dados de exemplo abaixo")
    
    # Criar dados de exemplo baseados no formato fornecido
    sample_data = {
        'valence': [-0.176846, -0.367789, -0.648471, 0.150794, -0.135501],
        'arousal': [-0.0776398, 0.183895, 0.658149, 0.666667, 0.00483933],
        'dominance': [-0.016151582, 0.052005529, 0.024069737, -0.036670412, 0.015107742],
        'class': ['neutral', 'neutral', 'contempt', 'surprise', 'neutral'],
        'path': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']
    }
    df = pd.DataFrame(sample_data)
    st.warning("⚠️ Usando dados de exemplo. Faça upload de um arquivo CSV para visualizar seus dados.")
else:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")

if df is not None:
    # Sidebar para controles
    st.sidebar.header("🎛️ Controles de Visualização")
    
    # Seleção de classes
    all_classes = sorted(df['class'].unique())
    selected_classes = st.sidebar.multiselect(
        "Selecione as classes para visualizar:",
        options=all_classes,
        default=all_classes[:3] if len(all_classes) > 3 else all_classes
    )
    
    # Filtro por faixa de valores
    st.sidebar.subheader("🎚️ Filtros por Faixa")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        valence_range = st.slider(
            "Valence",
            min_value=float(df['valence'].min()),
            max_value=float(df['valence'].max()),
            value=(float(df['valence'].min()), float(df['valence'].max())),
            step=0.1
        )
    
    with col2:
        arousal_range = st.slider(
            "Arousal",
            min_value=float(df['arousal'].min()),
            max_value=float(df['arousal'].max()),
            value=(float(df['arousal'].min()), float(df['arousal'].max())),
            step=0.1
        )
    
    with col3:
        dominance_range = st.slider(
            "Dominance",
            min_value=float(df['dominance'].min()),
            max_value=float(df['dominance'].max()),
            value=(float(df['dominance'].min()), float(df['dominance'].max())),
            step=0.1
        )
    
    # Aplicar filtros
    filtered_df = df[
        (df['class'].isin(selected_classes)) &
        (df['valence'] >= valence_range[0]) & (df['valence'] <= valence_range[1]) &
        (df['arousal'] >= arousal_range[0]) & (df['arousal'] <= arousal_range[1]) &
        (df['dominance'] >= dominance_range[0]) & (df['dominance'] <= dominance_range[1])
    ]
    
    # Estatísticas básicas
    st.sidebar.subheader("📈 Estatísticas")
    st.sidebar.write(f"Registros filtrados: {len(filtered_df)}")
    st.sidebar.write(f"Classes selecionadas: {len(selected_classes)}")
    
    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Gráfico 3D", 
        "📈 Distribuições 2D", 
        "📋 Dados", 
        "ℹ️ Informações"
    ])
    
    with tab1:
        st.subheader("Visualização 3D - Valence, Arousal, Dominance")
        
        if len(filtered_df) > 0:
            # Criar gráfico 3D
            fig_3d = px.scatter_3d(
                filtered_df,
                x='valence',
                y='arousal',
                z='dominance',
                color='class',
                title='Distribuição 3D das Emoções',
                labels={
                    'valence': 'Valence',
                    'arousal': 'Arousal', 
                    'dominance': 'Dominance'
                },
                hover_data=['path'],
                opacity=0.7
            )
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Valence',
                    yaxis_title='Arousal',
                    zaxis_title='Dominance'
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Nenhum dado corresponde aos filtros aplicados.")
    
    with tab2:
        st.subheader("Visualizações 2D")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico Valence vs Arousal
                fig_va = px.scatter(
                    filtered_df,
                    x='valence',
                    y='arousal',
                    color='class',
                    title='Valence vs Arousal',
                    hover_data=['dominance', 'path']
                )
                st.plotly_chart(fig_va, use_container_width=True)
                
                # Histograma Valence
                fig_hist_v = px.histogram(
                    filtered_df,
                    x='valence',
                    color='class',
                    title='Distribuição de Valence',
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig_hist_v, use_container_width=True)
            
            with col2:
                # Gráfico Valence vs Dominance
                fig_vd = px.scatter(
                    filtered_df,
                    x='valence',
                    y='dominance',
                    color='class',
                    title='Valence vs Dominance',
                    hover_data=['arousal', 'path']
                )
                st.plotly_chart(fig_vd, use_container_width=True)
                
                # Histograma Arousal
                fig_hist_a = px.histogram(
                    filtered_df,
                    x='arousal',
                    color='class',
                    title='Distribuição de Arousal',
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig_hist_a, use_container_width=True)
        else:
            st.warning("Nenhum dado corresponde aos filtros aplicados.")
    
    with tab3:
        st.subheader("Dados Filtrados")
        
        if len(filtered_df) > 0:
            # Estatísticas por classe
            st.write("**Estatísticas por Classe:**")
            stats = filtered_df.groupby('class').agg({
                'valence': ['count', 'mean', 'std', 'min', 'max'],
                'arousal': ['mean', 'std', 'min', 'max'],
                'dominance': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            st.dataframe(stats)
            
            # Dados brutos
            st.write("**Dados Brutos:**")
            st.dataframe(filtered_df.reset_index(drop=True))
            
            # Opção para download dos dados filtrados
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download dos dados filtrados (CSV)",
                data=csv,
                file_name="vad_data_filtered.csv",
                mime="text/csv"
            )
        else:
            st.warning("Nenhum dado corresponde aos filtros aplicados.")
    
    with tab4:
        st.subheader("Informações sobre o Dataset VAD")
        
        st.markdown("""
        ### Sobre as Dimensões VAD:
        
        - **Valence (Valência)**: Prazer vs. Desprazer
          - Valores positivos: emoções positivas (felicidade, surpresa)
          - Valores negativos: emoções negativas (raiva, medo, tristeza)
        
        - **Arousal (Excitação)**: Calmo vs. Excitado
          - Valores positivos: alta excitação (medo, raiva, surpresa)
          - Valores negativos: baixa excitação (tristeza, tédio)
        
        - **Dominance (Dominância)**: Submissão vs. Controle
          - Valores positivos: sentimento de controle
          - Valores negativos: sentimento de submissão
        
        ### Interpretação das Emoções:
        
        | Emoção | Valence | Arousal | Dominance |
        |--------|---------|---------|-----------|
        | Felicidade | Alto | Moderado | Alto |
        | Raiva | Baixo | Alto | Alto |
        | Medo | Baixo | Alto | Baixo |
        | Tristeza | Baixo | Baixo | Baixo |
        | Surpresa | Moderado | Alto | Variável |
        | Neutro | Neutro | Neutro | Neutro |
        """)
        
        if len(filtered_df) > 0:
            st.write("**Resumo do Dataset Atual:**")
            st.write(f"- Total de registros: {len(df)}")
            st.write(f"- Classes disponíveis: {', '.join(all_classes)}")
            st.write(f"- Faixa de Valence: {df['valence'].min():.3f} a {df['valence'].max():.3f}")
            st.write(f"- Faixa de Arousal: {df['arousal'].min():.3f} a {df['arousal'].max():.3f}")
            st.write(f"- Faixa de Dominance: {df['dominance'].min():.3f} a {df['dominance'].max():.3f}")

# Instruções de uso
with st.expander("ℹ️ Como usar esta aplicação"):
    st.markdown("""
    1. **Upload de Dados**: Faça upload de um arquivo CSV contendo as colunas:
       - `valence`, `arousal`, `dominance`, `class`
       - A coluna `path` é opcional para informações de hover
    
    2. **Seleção de Classes**: Use o menu lateral para selecionar quais classes emocionais visualizar
    
    3. **Filtros**: Ajuste os sliders para focar em faixas específicas de Valence, Arousal e Dominance
    
    4. **Visualizações**:
       - **Gráfico 3D**: Visualização completa das três dimensões
       - **Distribuições 2D**: Gráficos de dispersão e histogramas
       - **Dados**: Tabelas com estatísticas e dados filtrados
    
    5. **Download**: Baixe os dados filtrados para análise adicional
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido para visualização de dados de emoções usando o modelo VAD (Valence-Arousal-Dominance)"
)