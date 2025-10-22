import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Configurar a página
st.set_page_config(
    page_title="Análise de Distribuição de Emoções",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmotionAnalysisApp:
    def __init__(self):
        self.df = None
        self.emotion_columns = [
            'happy', 'contempt', 'elated', 'hopeful', 'surprised', 
            'proud', 'loved', 'angry', 'astonished', 'disgusted', 
            'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
    def setup_ui(self):
        st.title("😊 Análise de Distribuição de Emoções")
        st.markdown("---")
        
        # Sidebar para upload e configurações
        with st.sidebar:
            st.header("📁 Configurações")
            uploaded_file = st.file_uploader(
                "Carregar arquivo CSV",
                type=['csv'],
                help="Selecione o arquivo CSV contendo os dados das emoções"
            )
            
            if uploaded_file is not None:
                try:
                    self.df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Arquivo carregado com sucesso! ({len(self.df)} registros)")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
                    return
            
            st.markdown("---")
            st.header("📊 Opções de Visualização")
            
            if self.df is not None:
                # Seleção de número de amostras para o gráfico de distribuição
                self.num_samples = st.slider(
                    "Número de amostras para visualização",
                    min_value=5,
                    max_value=min(50, len(self.df)),
                    value=15,
                    help="Número de amostras aleatórias para mostrar no gráfico de distribuição"
                )
        
        # Conteúdo principal
        if self.df is not None:
            # Verificar se as colunas de emoção estão presentes
            missing_columns = [col for col in self.emotion_columns if col not in self.df.columns]
            if missing_columns:
                st.error(f"❌ Colunas faltando no arquivo: {', '.join(missing_columns)}")
                return
            
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Estatísticas", 
                "📊 Gráfico de Barras", 
                "🎨 Heatmap", 
                "📋 Distribuição por Amostra"
            ])
            
            with tab1:
                self.calculate_statistics()
            
            with tab2:
                self.create_bar_chart()
            
            with tab3:
                self.create_heatmap()
            
            with tab4:
                self.create_sample_distribution()
        else:
            # Mensagem inicial quando nenhum arquivo foi carregado
            st.info("👆 Por favor, carregue um arquivo CSV usando o menu lateral para começar a análise.")
            
            # Exemplo de estrutura esperada
            st.subheader("📋 Estrutura esperada do arquivo CSV:")
            example_data = {
                'happy': [0.85, 0.10, 0.05],
                'contempt': [0.02, 0.80, 0.01],
                'elated': [0.01, 0.02, 0.70],
                'hopeful': [0.03, 0.01, 0.10],
                'surprised': [0.02, 0.01, 0.05],
                'proud': [0.01, 0.02, 0.03],
                'loved': [0.02, 0.01, 0.02],
                'angry': [0.01, 0.01, 0.01],
                'astonished': [0.01, 0.01, 0.01],
                'disgusted': [0.00, 0.00, 0.01],
                'fearful': [0.01, 0.00, 0.00],
                'sad': [0.01, 0.01, 0.01],
                'fatigued': [0.00, 0.00, 0.00],
                'neutral': [0.05, 0.10, 0.10],
                'file_path': ['path1.txt', 'path2.txt', 'path3.txt']
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
    
    def calculate_statistics(self):
        st.header("📈 Estatísticas Descritivas")
        
        # Criar colunas para layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Estatísticas detalhadas por emoção
            st.subheader("Estatísticas por Emoção")
            
            stats_data = []
            for emotion in self.emotion_columns:
                values = self.df[emotion]
                stats_data.append({
                    'Emoção': emotion,
                    'Média': f"{values.mean():.6f}",
                    'Mediana': f"{values.median():.6f}",
                    'Desvio Padrão': f"{values.std():.6f}",
                    'Máximo': f"{values.max():.6f}",
                    'Mínimo': f"{values.min():.6f}",
                    'Contagem': values.count()
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, height=600)
        
        with col2:
            # Estatísticas gerais
            st.subheader("📊 Estatísticas Gerais")
            st.metric("Total de Amostras", len(self.df))
            
            # Emoção predominante
            predominant_emotions = self.df[self.emotion_columns].idxmax(axis=1)
            emotion_counts = predominant_emotions.value_counts()
            
            st.subheader("🎯 Emoção Predominante")
            for emotion, count in emotion_counts.head(5).items():
                percentage = (count / len(self.df)) * 100
                st.metric(
                    label=emotion.capitalize(),
                    value=f"{count}",
                    delta=f"{percentage:.1f}%"
                )
            
            # Top emoções por média
            st.subheader("🏆 Top Emoções (Média)")
            means = self.df[self.emotion_columns].mean().sort_values(ascending=False)
            for emotion, mean_val in means.head(3).items():
                st.metric(
                    label=emotion.capitalize(),
                    value=f"{mean_val:.4f}"
                )
    
    def create_bar_chart(self):
        st.header("📊 Distribuição Média das Emoções")
        
        # Calcular médias
        means = self.df[self.emotion_columns].mean().sort_values(ascending=True)
        
        # Criar gráfico com Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=means.index,
            x=means.values,
            orientation='h',
            marker_color=px.colors.qualitative.Set3,
            text=[f'{val:.4f}' for val in means.values],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Valores Médios das Emoções',
            xaxis_title='Valor Médio',
            yaxis_title='Emoções',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar dados em tabela
        with st.expander("📋 Ver dados em tabela"):
            means_df = pd.DataFrame({
                'Emoção': means.index,
                'Valor Médio': means.values
            })
            st.dataframe(means_df, use_container_width=True)
    
    def create_heatmap(self):
        st.header("🎨 Matriz de Correlação entre Emoções")
        
        # Calcular matriz de correlação
        correlation_matrix = self.df[self.emotion_columns].corr()
        
        # Criar heatmap com Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=[[f'{val:.2f}' for val in row] for row in correlation_matrix.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlação entre Emoções',
            xaxis_title='Emoções',
            yaxis_title='Emoções',
            height=700,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights sobre correlações
        with st.expander("💡 Insights sobre Correlações"):
            # Encontrar correlações fortes
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_correlations.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_val
                        ))
            
            if strong_correlations:
                st.write("**Correlações Fortes (|r| > 0.5):**")
                for emo1, emo2, corr in strong_correlations:
                    st.write(f"- {emo1} ↔ {emo2}: {corr:.3f}")
            else:
                st.write("Não foram encontradas correlações fortes entre as emoções.")
    
    def create_sample_distribution(self):
        st.header("📋 Distribuição de Emoções por Amostra")
        
        # Selecionar amostras aleatórias
        sample_indices = np.random.choice(len(self.df), self.num_samples, replace=False)
        sample_df = self.df.iloc[sample_indices]
        
        # Criar gráfico interativo
        fig = go.Figure()
        
        for i, idx in enumerate(sample_indices):
            sample_data = self.df.iloc[idx][self.emotion_columns]
            fig.add_trace(go.Scatter(
                x=self.emotion_columns,
                y=sample_data,
                mode='lines+markers',
                name=f'Amostra {idx}',
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f'Distribuição de Emoções em {self.num_samples} Amostras Aleatórias',
            xaxis_title='Emoções',
            yaxis_title='Valor',
            height=600,
            xaxis=dict(tickangle=45),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com os dados das amostras selecionadas
        with st.expander("📊 Ver dados das amostras selecionadas"):
            display_df = sample_df[self.emotion_columns].copy()
            display_df['Amostra'] = [f'Amostra {idx}' for idx in sample_indices]
            display_df = display_df[['Amostra'] + self.emotion_columns]
            st.dataframe(display_df, use_container_width=True)

# Executar a aplicação
if __name__ == "__main__":
    app = EmotionAnalysisApp()
    app.setup_ui()