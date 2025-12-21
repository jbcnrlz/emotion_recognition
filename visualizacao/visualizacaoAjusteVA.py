# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Visualização de Ajuste de Emoções VA",
    page_icon="😊",
    layout="wide"
)

# Título e descrição
st.title("📊 Visualização de Ajuste de Valence-Arousal")
st.markdown("""
Esta aplicação visualiza o ajuste dos pontos de Valence-Arousal para ficarem dentro das distribuições emocionais definidas.
Carregue os arquivos CSV das distribuições e dos pontos ajustados para visualizar as elipses e os pontos.
""")

# Sidebar para upload de arquivos e configurações
with st.sidebar:
    st.header("📁 Upload de Arquivos")
    
    # Upload do arquivo de distribuições
    distro_file = st.file_uploader(
        "Arquivo de Distribuições (CSV)",
        type=['csv'],
        help="Arquivo com médias e desvios padrão das emoções"
    )
    
    # Upload do arquivo de pontos ajustados
    points_file = st.file_uploader(
        "Arquivo de Pontos Ajustados (CSV)",
        type=['csv'],
        help="Arquivo com pontos originais e ajustados"
    )
    
    st.header("⚙️ Configurações de Visualização")
    
    # Configurações
    radius = st.slider(
        "Raio da Elipse (desvios padrão)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Controla o tamanho das elipses (quantos desvios padrão)"
    )
    
    point_size = st.slider(
        "Tamanho dos Pontos",
        min_value=1,
        max_value=20,
        value=5,
        help="Tamanho dos pontos no gráfico"
    )
    
    show_original = st.checkbox("Mostrar Pontos Originais", value=True)
    show_adjusted = st.checkbox("Mostrar Pontos Ajustados", value=True)
    show_ellipses = st.checkbox("Mostrar Elipses", value=True)
    show_centers = st.checkbox("Mostrar Centros", value=True)
    
    # Seletor de emoções para filtrar
    st.header("🎭 Filtro de Emoções")
    show_all = st.checkbox("Mostrar Todas as Emoções", value=True)

# Função para carregar e processar distribuições
def load_distributions(distro_df):
    """Processa o dataframe de distribuições"""
    distributions = {}
    for _, row in distro_df.iterrows():
        emotion = row['class']
        distributions[emotion] = {
            'valence_mean': row['valence mean'],
            'valence_std': row['valence std'],
            'arousal_mean': row['arousal mean'],
            'arousal_std': row['arousal std'],
            'dominance_mean': row['dominance mean'],
            'dominance_std': row['dominance std']
        }
    
    # Adicionar neutro se não existir
    if 'neutral' not in distributions:
        distributions['neutral'] = {
            'valence_mean': 0.0,
            'valence_std': 0.1,
            'arousal_mean': 0.0,
            'arousal_std': 0.1,
            'dominance_mean': 0.0,
            'dominance_std': 0.1
        }
    
    return distributions

# Função para calcular pontos da elipse
def calculate_ellipse(mean_x, mean_y, std_x, std_y, radius=1.0, n_points=100):
    """Calcula pontos para desenhar uma elipse"""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = mean_x + radius * std_x * np.cos(theta)
    y = mean_y + radius * std_y * np.sin(theta)
    return x, y

# Função para criar figura Plotly interativa
def create_interactive_plot(distributions, points_df, radius, show_original, 
                           show_adjusted, show_ellipses, show_centers, 
                           point_size, selected_emotions=None):
    """Cria gráfico interativo com Plotly"""
    
    # Cores para as emoções
    emotion_colors = {
        'happy': '#FFD700',  # Amarelo ouro
        'neutral': '#808080', # Cinza
        'sad': '#4169E1',    # Azul real
        'surprised': '#FF69B4', # Rosa
        'fearful': '#8A2BE2', # Violeta
        'disgusted': '#228B22', # Verde floresta
        'angry': '#FF4500',   # Laranja vermelho
        'contempt': '#20B2AA' # Verde mar claro
    }
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar elipses se solicitado
    if show_ellipses and distributions:
        for emotion, params in distributions.items():
            if selected_emotions and emotion not in selected_emotions:
                continue
                
            # Calcular pontos da elipse
            x_ellipse, y_ellipse = calculate_ellipse(
                params['valence_mean'], params['arousal_mean'],
                params['valence_std'], params['arousal_std'],
                radius
            )
            
            # Adicionar elipse
            fig.add_trace(go.Scatter(
                x=x_ellipse,
                y=y_ellipse,
                mode='lines',
                name=f'{emotion} (elipse)',
                line=dict(color=emotion_colors.get(emotion, '#000000'), width=1.5),
                opacity=0.7,
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # Adicionar centros se solicitado
    if show_centers and distributions:
        for emotion, params in distributions.items():
            if selected_emotions and emotion not in selected_emotions:
                continue
                
            fig.add_trace(go.Scatter(
                x=[params['valence_mean']],
                y=[params['arousal_mean']],
                mode='markers',
                name=f'{emotion} (centro)',
                marker=dict(
                    size=10,
                    color=emotion_colors.get(emotion, '#000000'),
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                showlegend=True,
                hoverinfo='text',
                hovertext=f'<b>{emotion}</b><br>Centro: ({params["valence_mean"]:.2f}, {params["arousal_mean"]:.2f})'
            ))
    
    # Adicionar pontos se houver dados
    if points_df is not None:
        # Filtrar emoções se necessário
        if selected_emotions:
            points_df = points_df[points_df['emotion'].isin(selected_emotions)]
        
        # Adicionar pontos originais
        if show_original:
            fig.add_trace(go.Scatter(
                x=points_df['valence_original'],
                y=points_df['arousal_original'],
                mode='markers',
                name='Pontos Originais',
                marker=dict(
                    size=point_size,
                    color='lightblue',
                    opacity=0.6,
                    symbol='circle',
                    line=dict(width=1, color='darkblue')
                ),
                text=[f"Emoção: {row['emotion']}<br>Original: ({row['valence_original']:.3f}, {row['arousal_original']:.3f})" 
                      for _, row in points_df.iterrows()],
                hoverinfo='text'
            ))
        
        # Adicionar pontos ajustados
        if show_adjusted:
            # Criar coluna de cor baseada na emoção
            colors = points_df['emotion'].map(lambda x: emotion_colors.get(x, '#000000'))
            
            fig.add_trace(go.Scatter(
                x=points_df['valence_adjusted'],
                y=points_df['arousal_adjusted'],
                mode='markers',
                name='Pontos Ajustados',
                marker=dict(
                    size=point_size,
                    color=colors,
                    opacity=0.8,
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                text=[f"Emoção: {row['emotion']}<br>Ajustado: ({row['valence_adjusted']:.3f}, {row['arousal_adjusted']:.3f})<br>"
                      f"Ratio: {row['distance_ratio']:.2f}" 
                      for _, row in points_df.iterrows()],
                hoverinfo='text'
            ))
            
            # Adicionar linhas conectando pontos originais e ajustados
            if show_original:
                for _, row in points_df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['valence_original'], row['valence_adjusted']],
                        y=[row['arousal_original'], row['arousal_adjusted']],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Atualizar layout
    fig.update_layout(
        title='Distribuições Emocionais e Pontos Ajustados',
        xaxis_title='Valence',
        yaxis_title='Arousal',
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    # Adicionar grid e referências
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
    
    # Adicionar quadrantes
    fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="black", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="black", width=2, dash="dash"))
    
    return fig

# Função para criar gráfico de estatísticas
def create_statistics_plot(points_df):
    """Cria gráficos de estatísticas do ajuste"""
    
    if points_df is None:
        return None
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribuição de Emoções', 'Proporção de Ajustes',
                       'Distância Média por Emoção', 'Histograma de Ratios')
    )
    
    # 1. Distribuição de emoções
    emotion_counts = points_df['emotion'].value_counts()
    fig.add_trace(
        go.Bar(x=emotion_counts.index, y=emotion_counts.values, name='Contagem'),
        row=1, col=1
    )
    
    # 2. Proporção de ajustes por emoção
    adjustment_rates = points_df.groupby('emotion')['was_adjusted'].mean() * 100
    fig.add_trace(
        go.Bar(x=adjustment_rates.index, y=adjustment_rates.values, name='% Ajustado'),
        row=1, col=2
    )
    
    # 3. Distância média por emoção
    avg_ratio = points_df.groupby('emotion')['distance_ratio'].mean()
    fig.add_trace(
        go.Bar(x=avg_ratio.index, y=avg_ratio.values, name='Ratio Médio'),
        row=2, col=1
    )
    
    # 4. Histograma de ratios
    fig.add_trace(
        go.Histogram(x=points_df['distance_ratio'], nbinsx=30, name='Distribuição de Ratios'),
        row=2, col=2
    )
    
    # Atualizar layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Estatísticas do Ajuste",
        template='plotly_white'
    )
    
    return fig

# Função para criar tabela resumo
def create_summary_table(distributions, points_df):
    """Cria tabela resumo das estatísticas"""
    
    if points_df is None or distributions is None:
        return None
    
    summary_data = []
    
    for emotion in points_df['emotion'].unique():
        emotion_points = points_df[points_df['emotion'] == emotion]
        dist = distributions.get(emotion, {})
        
        summary_data.append({
            'Emoção': emotion,
            'Número de Pontos': len(emotion_points),
            '% Ajustado': f"{emotion_points['was_adjusted'].mean() * 100:.1f}%",
            'Ratio Médio': f"{emotion_points['distance_ratio'].mean():.3f}",
            'Valence Médio (Orig)': f"{emotion_points['valence_original'].mean():.3f}",
            'Arousal Médio (Orig)': f"{emotion_points['arousal_original'].mean():.3f}",
            'Valence Centro': f"{dist.get('valence_mean', 0):.3f}" if dist else "N/A",
            'Arousal Centro': f"{dist.get('arousal_mean', 0):.3f}" if dist else "N/A"
        })
    
    return pd.DataFrame(summary_data)

# Main app logic
def main():
    # Inicializar variáveis
    distributions = None
    points_df = None
    
    # Carregar dados se arquivos foram fornecidos
    if distro_file is not None:
        try:
            distro_df = pd.read_csv(distro_file)
            distributions = load_distributions(distro_df)
            st.sidebar.success(f"✅ Distribuições carregadas: {len(distributions)} emoções")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar distribuições: {e}")
    
    if points_file is not None:
        try:
            points_df = pd.read_csv(points_file)
            st.sidebar.success(f"✅ Pontos carregados: {len(points_df)} amostras")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar pontos: {e}")
    
    # Seletor de emoções individual
    if distributions and not show_all:
        selected_emotions = st.sidebar.multiselect(
            "Selecione Emoções",
            options=list(distributions.keys()),
            default=list(distributions.keys())[:3] if distributions else []
        )
    else:
        selected_emotions = None
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if distributions or points_df is not None:
            # Criar gráfico principal
            fig = create_interactive_plot(
                distributions, points_df, radius,
                show_original, show_adjusted, show_ellipses,
                show_centers, point_size, selected_emotions
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Faça upload dos arquivos para visualizar os dados")
            
            # Mostrar exemplo vazio
            fig = go.Figure()
            fig.update_layout(
                title='Exemplo de Visualização',
                xaxis_title='Valence',
                yaxis_title='Arousal',
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Estatísticas Rápidas")
        
        if points_df is not None:
            # Estatísticas gerais
            total_points = len(points_df)
            adjusted_points = points_df['was_adjusted'].sum()
            avg_ratio = points_df['distance_ratio'].mean()
            
            st.metric("Total de Pontos", total_points)
            st.metric("Pontos Ajustados", adjusted_points, 
                     f"{adjusted_points/total_points*100:.1f}%")
            st.metric("Ratio Médio", f"{avg_ratio:.3f}")
            
            # Distribuição de ajustes
            st.subheader("📊 Distribuição")
            adjustment_dist = points_df['was_adjusted'].value_counts()
            
            if len(adjustment_dist) > 0:
                labels = ['Dentro' if not idx else 'Fora' for idx in adjustment_dist.index]
                values = adjustment_dist.values
                
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Mostrar primeiras linhas dos dados
        if points_df is not None:
            with st.expander("👁️ Visualizar Dados (primeiras 10 linhas)"):
                st.dataframe(points_df.head(10))
    
    # Seção de estatísticas detalhadas
    if points_df is not None:
        st.markdown("---")
        st.header("📊 Estatísticas Detalhadas")
        
        # Tabela resumo
        summary_table = create_summary_table(distributions, points_df)
        if summary_table is not None:
            st.subheader("Resumo por Emoção")
            st.dataframe(summary_table, use_container_width=True)
        
        # Gráficos de estatísticas
        st.subheader("Gráficos de Análise")
        stats_fig = create_statistics_plot(points_df)
        if stats_fig:
            st.plotly_chart(stats_fig, use_container_width=True)
        
        # Visualização 3D se houver dados de dominance
        if 'dominance_mean' in distro_df.columns and points_df is not None:
            st.markdown("---")
            st.header("🌐 Visualização 3D (Valence-Arousal-Dominance)")
            
            # Verificar se temos dados de dominance nos pontos
            if all(col in points_df.columns for col in ['valence_original', 'arousal_original', 'valence_adjusted', 'arousal_adjusted']):
                # Adicionar dominance simulado para demonstração
                # (em uma aplicação real, você teria esses dados)
                st.info("Nota: Os valores de dominance são ilustrativos para demonstração 3D")
                
                # Criar gráfico 3D
                fig_3d = go.Figure()
                
                # Adicionar pontos originais (com dominance simulado)
                if show_original:
                    fig_3d.add_trace(go.Scatter3d(
                        x=points_df['valence_original'],
                        y=points_df['arousal_original'],
                        z=np.random.randn(len(points_df)) * 0.5,  # Dominance simulado
                        mode='markers',
                        name='Pontos Originais',
                        marker=dict(size=3, color='lightblue', opacity=0.6)
                    ))
                
                # Adicionar pontos ajustados
                if show_adjusted:
                    fig_3d.add_trace(go.Scatter3d(
                        x=points_df['valence_adjusted'],
                        y=points_df['arousal_adjusted'],
                        z=np.zeros(len(points_df)),  # Dominance no plano médio
                        mode='markers',
                        name='Pontos Ajustados',
                        marker=dict(size=4, color='red', opacity=0.8)
                    ))
                
                fig_3d.update_layout(
                    title='Visualização 3D (Valence-Arousal-Dominance)',
                    scene=dict(
                        xaxis_title='Valence',
                        yaxis_title='Arousal',
                        zaxis_title='Dominance'
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
    
    # Seção de download
    if points_df is not None:
        st.markdown("---")
        st.header("💾 Exportar Dados")
        
        # Converter DataFrame para CSV
        csv = points_df.to_csv(index=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download Pontos Ajustados",
                data=csv,
                file_name="pontos_ajustados_detalhado.csv",
                mime="text/csv",
                help="Baixar todos os pontos ajustados em formato CSV"
            )
        
        with col2:
            # Criar resumo estatístico para download
            if summary_table is not None:
                summary_csv = summary_table.to_csv(index=False)
                st.download_button(
                    label="📊 Download Estatísticas",
                    data=summary_csv,
                    file_name="estatisticas_ajuste.csv",
                    mime="text/csv",
                    help="Baixar estatísticas resumidas"
                )
        
        with col3:
            # Botão para gerar relatório
            if st.button("📄 Gerar Relatório PDF (Simulado)"):
                st.success("📄 Relatório gerado com sucesso! (funcionalidade simulada)")
                st.info("Em uma implementação real, você usaria uma biblioteca como ReportLab ou WeasyPrint para gerar PDFs")

# Rodar aplicação
if __name__ == "__main__":
    main()