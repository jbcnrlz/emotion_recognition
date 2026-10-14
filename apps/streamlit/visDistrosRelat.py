import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página para ocupar toda a largura
st.set_page_config(page_title="Relatório de Distribuições Emocionais", layout="wide")

st.title("📊 Visualização de Distribuições Emocionais")
st.markdown("""
Esta ferramenta visualiza os dados emocionais. 
**Nota:** Os rótulos nos gráficos mostram apenas a primeira palavra da classe para facilitar a leitura.
Passe o mouse sobre os pontos para ver o nome completo.
""")

# --- Sidebar: Upload de Arquivo ---
st.sidebar.header("Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Escolha o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Carregar dados
        df = pd.read_csv(uploaded_file)

        # Verificação de colunas necessárias
        required_cols = ['class', 'valence mean', 'valence std', 'arousal mean', 'arousal std', 'dominance mean']
        if not all(col in df.columns for col in required_cols):
            st.error(f"O arquivo deve conter as colunas: {', '.join(required_cols)}")
        else:
            # --- Processamento: Extrair a primeira palavra para o label ---
            # Pega a string, divide nos espaços e seleciona o primeiro item
            df['label_short'] = df['class'].apply(lambda x: str(x).split(' ')[0] if isinstance(x, str) else str(x))

            # --- Sidebar: Filtros ---
            st.sidebar.subheader("Filtrar Emoções")
            all_classes = df['class'].unique()
            selected_classes = st.sidebar.multiselect(
                "Selecione as classes para visualizar:",
                options=all_classes,
                default=all_classes
            )
            
            # Filtrar DataFrame
            df_filtered = df[df['class'].isin(selected_classes)]

            if df_filtered.empty:
                st.warning("Nenhuma classe selecionada.")
            else:
                # --- Visualizações Gráficas ---
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("2D: Valência x Ativação")
                    fig_2d = px.scatter(
                        df_filtered,
                        x='valence mean',
                        y='arousal mean',
                        error_x='valence std',
                        error_y='arousal std',
                        text='label_short',  # Usando apenas a primeira palavra
                        hover_name='class',  # Nome completo ao passar o mouse
                        color='dominance mean',
                        color_continuous_scale='RdBu_r',
                        title="Espaço Bidimensional",
                        labels={'valence mean': 'Valência', 'arousal mean': 'Ativação'}
                    )
                    # Ajusta a posição do texto para não ficar em cima do ponto
                    fig_2d.update_traces(textposition='top center')
                    st.plotly_chart(fig_2d, use_container_width=True)

                with col2:
                    st.subheader("3D: Espaço VAD")
                    fig_3d = px.scatter_3d(
                        df_filtered,
                        x='valence mean',
                        y='arousal mean',
                        z='dominance mean',
                        color='valence mean',
                        text='label_short', # Label simplificado no 3D também
                        hover_name='class',
                        title="Espaço Tridimensional Interativo",
                        labels={'valence mean': 'Valência', 'arousal mean': 'Ativação', 'dominance mean': 'Dominância'}
                    )
                    fig_3d.update_traces(textposition='top center')
                    st.plotly_chart(fig_3d, use_container_width=True)

                # --- Tabela de Dados Completa ---
                st.divider() # Linha divisória visual
                st.subheader("📋 Tabela de Dados Detalhada")
                st.markdown("Abaixo estão listados todos os valores numéricos para as emoções filtradas.")
                
                # Exibindo a tabela ocupando a largura total e permitindo ordenação
                st.dataframe(
                    df_filtered, 
                    use_container_width=True,
                    column_config={
                        "class": "Classe Emocional (Nome Completo)",
                        "valence mean": st.column_config.NumberColumn("Valência (Média)", format="%.2f"),
                        "valence std": st.column_config.NumberColumn("Valência (DesvPad)", format="%.2f"),
                        "arousal mean": st.column_config.NumberColumn("Ativação (Média)", format="%.2f"),
                        "arousal std": st.column_config.NumberColumn("Ativação (DesvPad)", format="%.2f"),
                        "dominance mean": st.column_config.NumberColumn("Dominância (Média)", format="%.2f"),
                    }
                )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

else:
    st.info("📂 Por favor, faça o upload do arquivo CSV na barra lateral para iniciar.")