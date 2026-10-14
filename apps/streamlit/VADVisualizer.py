import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import io

# Configuração da página
st.set_page_config(
    page_title="Visualizador VAD - Distribuições com Incerteza",
    page_icon="📊",
    layout="wide"
)

# Título da aplicação
st.title("📊 Visualizador de Distribuição VAD - Com Incerteza")
st.markdown("Explore distribuições emocionais com médias e desvios padrão")

# Função para carregar e processar os dados principais
def load_main_data(uploaded_file):
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

# Função para carregar e processar os dados de comparação (com médias e std)
def load_comparison_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Por favor, faça upload de um arquivo CSV")
            return None
        
        # Verificar se as colunas necessárias estão presentes
        required_columns = ['class', 'valence mean', 'valence std', 'arousal mean', 'arousal std', 'dominance mean', 'dominance std']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colunas necessárias não encontradas: {missing_columns}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para gerar pontos amostrais a partir de distribuições normais
def generate_samples_from_distribution(mean_valence, std_valence, mean_arousal, std_arousal, mean_dominance, std_dominance, n_samples=100):
    """Gera pontos amostrais a partir de distribuições normais"""
    valence_samples = np.random.normal(mean_valence, std_valence, n_samples)
    arousal_samples = np.random.normal(mean_arousal, std_arousal, n_samples)
    dominance_samples = np.random.normal(mean_dominance, std_dominance, n_samples)
    
    # Limitar aos valores possíveis no espaço VAD (normalmente -1 a 1)
    valence_samples = np.clip(valence_samples, -1, 1)
    arousal_samples = np.clip(arousal_samples, -1, 1)
    dominance_samples = np.clip(dominance_samples, -1, 1)
    
    return valence_samples, arousal_samples, dominance_samples

# Upload dos arquivos
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Dados Principais (Pontos Individuais)")
    uploaded_file_main = st.file_uploader(
        "Arquivo CSV com dados VAD individuais",
        type=['csv'],
        key="main",
        help="Deve conter: valence, arousal, dominance, class"
    )

with col2:
    st.subheader("🔄 Dados de Comparação (Distribuições)")
    uploaded_file_comparison = st.file_uploader(
        "Arquivo CSV com médias e desvios padrão",
        type=['csv'],
        key="comparison",
        help="Deve conter: class, valence mean, valence std, arousal mean, arousal std, dominance mean, dominance std"
    )

# Se não há arquivo principal carregado, usar dados de exemplo
if uploaded_file_main is None:
    st.info("💡 **Dica:** Faça upload dos arquivos CSV ou use os dados de exemplo")
    
    # Criar dados de exemplo principais
    sample_data_main = {
        'valence': [-0.176846, -0.367789, -0.648471, 0.150794, -0.135501, 0.357143, -0.796971, 0.81, -0.23, 0.4],
        'arousal': [-0.0776398, 0.183895, 0.658149, 0.666667, 0.00483933, 0.0793651, -0.229129, 0.51, 0.31, 0.67],
        'dominance': [-0.016151582, 0.052005529, 0.024069737, -0.036670412, 0.015107742, 0.116616621, -0.193158045, 0.46, 0.18, -0.13],
        'class': ['neutral', 'neutral', 'contempt', 'surprise', 'neutral', 'happy', 'sad', 'happy', 'contempt', 'surprise'],
        'path': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'image10.jpg']
    }
    df_main = pd.DataFrame(sample_data_main)
    st.warning("⚠️ Usando dados principais de exemplo.")
else:
    df_main = load_main_data(uploaded_file_main)
    if df_main is not None:
        st.success(f"✅ Dados principais carregados! {len(df_main)} registros.")

# Carregar dados de comparação se disponíveis
df_comparison = None
if uploaded_file_comparison is not None:
    df_comparison = load_comparison_data(uploaded_file_comparison)
    if df_comparison is not None:
        st.success(f"✅ Dados de comparação carregados! {len(df_comparison)} distribuições.")
        
        # Mostrar preview das distribuições
        st.sidebar.subheader("📋 Distribuições Carregadas")
        for idx, row in df_comparison.iterrows():
            class_name = row['class']
            if len(class_name) > 30:
                display_name = class_name[:27] + "..."
            else:
                display_name = class_name
            st.sidebar.write(f"• {display_name}")

if df_main is not None:
    # Sidebar para controles
    st.sidebar.header("🎛️ Controles de Visualização")
    
    # Seleção de classes principais
    all_classes = sorted(df_main['class'].unique())
    selected_classes = st.sidebar.multiselect(
        "Selecione as classes principais:",
        options=all_classes,
        default=all_classes[:3] if len(all_classes) > 3 else all_classes
    )
    
    # Seleção de distribuições de comparação (se disponíveis)
    selected_comparison_distributions = []
    if df_comparison is not None:
        all_distributions = df_comparison['class'].tolist()
        selected_comparison_distributions = st.sidebar.multiselect(
            "Selecione as distribuições para comparação:",
            options=all_distributions,
            default=all_distributions[:5] if len(all_distributions) > 5 else all_distributions
        )
    
    # Configurações de visualização
    st.sidebar.subheader("🎨 Configurações de Visualização")
    
    n_samples_per_distribution = st.sidebar.slider(
        "Número de pontos por distribuição",
        min_value=10,
        max_value=500,
        value=100,
        help="Quantos pontos gerar para cada distribuição de comparação"
    )
    
    show_ellipsoids = st.sidebar.checkbox(
        "Mostrar elipsoides de incerteza", 
        value=True,
        help="Mostra elipsoides representando 1 desvio padrão"
    )
    
    show_distribution_points = st.sidebar.checkbox(
        "Mostrar pontos das distribuições",
        value=True,
        help="Mostra pontos amostrais das distribuições"
    )
    
    # Filtro por faixa de valores
    st.sidebar.subheader("🎚️ Filtros por Faixa")
    
    # Determinar faixas baseadas nos dados disponíveis
    valence_min = df_main['valence'].min()
    valence_max = df_main['valence'].max()
    arousal_min = df_main['arousal'].min()
    arousal_max = df_main['arousal'].max()
    dominance_min = df_main['dominance'].min()
    dominance_max = df_main['dominance'].max()
    
    if df_comparison is not None:
        comparison_valence_min = (df_comparison['valence mean'] - df_comparison['valence std']).min()
        comparison_valence_max = (df_comparison['valence mean'] + df_comparison['valence std']).max()
        comparison_arousal_min = (df_comparison['arousal mean'] - df_comparison['arousal std']).min()
        comparison_arousal_max = (df_comparison['arousal mean'] + df_comparison['arousal std']).max()
        comparison_dominance_min = (df_comparison['dominance mean'] - df_comparison['dominance std']).min()
        comparison_dominance_max = (df_comparison['dominance mean'] + df_comparison['dominance std']).max()
        
        valence_min = min(valence_min, comparison_valence_min)
        valence_max = max(valence_max, comparison_valence_max)
        arousal_min = min(arousal_min, comparison_arousal_min)
        arousal_max = max(arousal_max, comparison_arousal_max)
        dominance_min = min(dominance_min, comparison_dominance_min)
        dominance_max = max(dominance_max, comparison_dominance_max)
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        valence_range = st.slider(
            "Valence",
            min_value=float(valence_min),
            max_value=float(valence_max),
            value=(float(df_main['valence'].min()), float(df_main['valence'].max())),
            step=0.1
        )
    
    with col2:
        arousal_range = st.slider(
            "Arousal",
            min_value=float(arousal_min),
            max_value=float(arousal_max),
            value=(float(df_main['arousal'].min()), float(df_main['arousal'].max())),
            step=0.1
        )
    
    with col3:
        dominance_range = st.slider(
            "Dominance",
            min_value=float(dominance_min),
            max_value=float(dominance_max),
            value=(float(df_main['dominance'].min()), float(df_main['dominance'].max())),
            step=0.1
        )
    
    # Aplicar filtros aos dados principais
    filtered_df_main = df_main[
        (df_main['class'].isin(selected_classes)) &
        (df_main['valence'] >= valence_range[0]) & (df_main['valence'] <= valence_range[1]) &
        (df_main['arousal'] >= arousal_range[0]) & (df_main['arousal'] <= arousal_range[1]) &
        (df_main['dominance'] >= dominance_range[0]) & (df_main['dominance'] <= dominance_range[1])
    ]
    
    # Processar dados de comparação
    comparison_samples = {}
    comparison_ellipsoids = {}
    
    if df_comparison is not None and len(selected_comparison_distributions) > 0:
        filtered_df_comparison = df_comparison[df_comparison['class'].isin(selected_comparison_distributions)]
        
        for _, row in filtered_df_comparison.iterrows():
            class_name = row['class']
            
            # Gerar pontos amostrais
            valence_samples, arousal_samples, dominance_samples = generate_samples_from_distribution(
                row['valence mean'], row['valence std'],
                row['arousal mean'], row['arousal std'],
                row['dominance mean'], row['dominance std'],
                n_samples_per_distribution
# Continuação do código (parte 2)
            )
            
            # Filtrar pontos pela faixa selecionada
            mask = ((valence_samples >= valence_range[0]) & (valence_samples <= valence_range[1]) &
                   (arousal_samples >= arousal_range[0]) & (arousal_samples <= arousal_range[1]) &
                   (dominance_samples >= dominance_range[0]) & (dominance_samples <= dominance_range[1]))
            
            comparison_samples[class_name] = {
                'valence': valence_samples[mask],
                'arousal': arousal_samples[mask],
                'dominance': dominance_samples[mask],
                'mean': [row['valence mean'], row['arousal mean'], row['dominance mean']],
                'std': [row['valence std'], row['arousal std'], row['dominance std']]
            }
    
    # Estatísticas básicas
    st.sidebar.subheader("📈 Estatísticas")
    st.sidebar.write(f"Registros principais: {len(filtered_df_main)}")
    st.sidebar.write(f"Classes selecionadas: {len(selected_classes)}")
    if comparison_samples:
        st.sidebar.write(f"Distribuições: {len(comparison_samples)}")
    
    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Gráfico 3D", 
        "📈 Visualizações 2D", 
        "📋 Dados e Estatísticas", 
        "ℹ️ Informações"
    ])
    
    with tab1:
        st.subheader("Visualização 3D - Distribuições com Incerteza")
        
        if len(filtered_df_main) > 0 or comparison_samples:
            # Criar gráfico 3D
            fig_3d = go.Figure()
            
            # Adicionar dados principais
            if len(filtered_df_main) > 0:
                for class_name in selected_classes:
                    class_data = filtered_df_main[filtered_df_main['class'] == class_name]
                    fig_3d.add_trace(go.Scatter3d(
                        x=class_data['valence'],
                        y=class_data['arousal'],
                        z=class_data['dominance'],
                        mode='markers',
                        name=f"PRINCIPAL: {class_name}",
                        marker=dict(
                            size=4,
                            opacity=0.6
                        ),
                        hovertemplate=(
                            '<b>%{text}</b><br>' +
                            'Valence: %{x:.3f}<br>' +
                            'Arousal: %{y:.3f}<br>' +
                            'Dominance: %{z:.3f}<extra></extra>'
                        ),
                        text=[f"Classe: {class_name}"] * len(class_data)
                    ))
            
            # Adicionar dados de comparação
            if comparison_samples:
                colors = px.colors.qualitative.Set3
                
                for idx, (class_name, data) in enumerate(comparison_samples.items()):
                    color = colors[idx % len(colors)]
                    
                    # Mostrar pontos das distribuições
                    if show_distribution_points and len(data['valence']) > 0:
                        fig_3d.add_trace(go.Scatter3d(
                            x=data['valence'],
                            y=data['arousal'],
                            z=data['dominance'],
                            mode='markers',
                            name=f"DIST: {class_name[:20]}...",
                            marker=dict(
                                size=3,
                                opacity=0.4,
                                color=color
                            ),
                            hovertemplate=(
                                f'<b>Distribuição: {class_name}</b><br>' +
                                'Valence: %{x:.3f}<br>' +
                                'Arousal: %{y:.3f}<br>' +
                                'Dominance: %{z:.3f}<extra></extra>'
                            ),
                            showlegend=False
                        ))
                    
                    # Mostrar média da distribuição
                    fig_3d.add_trace(go.Scatter3d(
                        x=[data['mean'][0]],
                        y=[data['mean'][1]],
                        z=[data['mean'][2]],
                        mode='markers',
                        name=f"MÉDIA: {class_name[:15]}...",
                        marker=dict(
                            size=8,
                            color=color,
                            symbol='diamond',
                            line=dict(width=2, color='black')
                        ),
                        hovertemplate=(
                            f'<b>MÉDIA: {class_name}</b><br>' +
                            f'Valence: {data["mean"][0]:.3f} ± {data["std"][0]:.3f}<br>' +
                            f'Arousal: {data["mean"][1]:.3f} ± {data["std"][1]:.3f}<br>' +
                            f'Dominance: {data["mean"][2]:.3f} ± {data["std"][2]:.3f}<extra></extra>'
                        )
                    ))
                    
                    # Mostrar elipsoide de incerteza (1 desvio padrão)
                    if show_ellipsoids:
                        # Gerar pontos para o elipsoide
                        u = np.linspace(0, 2 * np.pi, 20)
                        v = np.linspace(0, np.pi, 20)
                        
                        x_ellipsoid = data['std'][0] * np.outer(np.cos(u), np.sin(v)) + data['mean'][0]
                        y_ellipsoid = data['std'][1] * np.outer(np.sin(u), np.sin(v)) + data['mean'][1]
                        z_ellipsoid = data['std'][2] * np.outer(np.ones_like(u), np.cos(v)) + data['mean'][2]
                        
                        fig_3d.add_trace(go.Surface(
                            x=x_ellipsoid,
                            y=y_ellipsoid,
                            z=z_ellipsoid,
                            opacity=0.2,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            name=f"INCERTEZA: {class_name[:15]}...",
                            hoverinfo='skip'
                        ))
            
            fig_3d.update_layout(
                title='Distribuições VAD com Incerteza - Visualização 3D',
                scene=dict(
                    xaxis_title='Valence',
                    yaxis_title='Arousal',
                    zaxis_title='Dominance'
                ),
                height=700,
                showlegend=True
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Nenhum dado corresponde aos filtros aplicados.")
    
    with tab2:
        st.subheader("Visualizações 2D - Projeções")
        
        if len(filtered_df_main) > 0 or comparison_samples:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico Valence vs Arousal
                fig_va = go.Figure()
                
                # Dados principais
                for class_name in selected_classes:
                    class_data = filtered_df_main[filtered_df_main['class'] == class_name]
                    fig_va.add_trace(go.Scatter(
                        x=class_data['valence'],
                        y=class_data['arousal'],
                        mode='markers',
                        name=f"PRINCIPAL: {class_name}",
                        opacity=0.6,
                        marker=dict(size=6)
                    ))
                
                # Dados de comparação
                if comparison_samples:
                    colors = px.colors.qualitative.Set3
                    
                    for idx, (class_name, data) in enumerate(comparison_samples.items()):
                        color = colors[idx % len(colors)]
                        
                        # Pontos das distribuições
                        if show_distribution_points and len(data['valence']) > 0:
                            fig_va.add_trace(go.Scatter(
                                x=data['valence'],
                                y=data['arousal'],
                                mode='markers',
                                name=f"DIST: {class_name[:15]}...",
                                opacity=0.3,
                                marker=dict(size=4, color=color),
                                showlegend=False
                            ))
                        
                        # Médias com barras de erro
                        fig_va.add_trace(go.Scatter(
                            x=[data['mean'][0]],
                            y=[data['mean'][1]],
                            mode='markers',
                            name=f"MÉDIA: {class_name[:15]}...",
                            marker=dict(size=10, color=color, symbol='diamond', line=dict(width=2, color='black')),
                            error_x=dict(
                                type='data',
                                array=[data['std'][0]],
                                color=color,
                                thickness=1.5,
                                width=3
                            ),
                            error_y=dict(
                                type='data',
                                array=[data['std'][1]],
                                color=color,
                                thickness=1.5,
                                width=3
                            )
                        ))
                
                fig_va.update_layout(
                    title='Valence vs Arousal - Com Incerteza',
                    xaxis_title='Valence',
                    yaxis_title='Arousal'
                )
                st.plotly_chart(fig_va, use_container_width=True)
                
            with col2:
                # Gráfico Valence vs Dominance
                fig_vd = go.Figure()
                
                # Dados principais
                for class_name in selected_classes:
                    class_data = filtered_df_main[filtered_df_main['class'] == class_name]
                    fig_vd.add_trace(go.Scatter(
                        x=class_data['valence'],
                        y=class_data['dominance'],
                        mode='markers',
                        name=f"PRINCIPAL: {class_name}",
                        opacity=0.6,
                        marker=dict(size=6),
                        showlegend=False
                    ))
                
                # Dados de comparação
                if comparison_samples:
                    colors = px.colors.qualitative.Set3
                    
                    for idx, (class_name, data) in enumerate(comparison_samples.items()):
                        color = colors[idx % len(colors)]
                        
                        # Pontos das distribuições
                        if show_distribution_points and len(data['valence']) > 0:
                            fig_vd.add_trace(go.Scatter(
                                x=data['valence'],
                                y=data['dominance'],
                                mode='markers',
                                name=f"DIST: {class_name[:15]}...",
                                opacity=0.3,
                                marker=dict(size=4, color=color),
                                showlegend=False
                            ))
                        
                        # Médias com barras de erro
                        fig_vd.add_trace(go.Scatter(
                            x=[data['mean'][0]],
                            y=[data['mean'][2]],
                            mode='markers',
                            name=f"MÉDIA: {class_name[:15]}...",
                            marker=dict(size=10, color=color, symbol='diamond', line=dict(width=2, color='black')),
                            error_x=dict(
                                type='data',
                                array=[data['std'][0]],
                                color=color,
                                thickness=1.5,
                                width=3
                            ),
                            error_y=dict(
                                type='data',
                                array=[data['std'][2]],
                                color=color,
                                thickness=1.5,
                                width=3
                            ),
                            showlegend=False
                        ))
                
                fig_vd.update_layout(
                    title='Valence vs Dominance - Com Incerteza',
                    xaxis_title='Valence',
                    yaxis_title='Dominance'
                )
                st.plotly_chart(fig_vd, use_container_width=True)
            
            # Gráfico Arousal vs Dominance
            fig_ad = go.Figure()
            
            # Dados principais
            for class_name in selected_classes:
                class_data = filtered_df_main[filtered_df_main['class'] == class_name]
                fig_ad.add_trace(go.Scatter(
                    x=class_data['arousal'],
                    y=class_data['dominance'],
                    mode='markers',
                    name=f"PRINCIPAL: {class_name}",
                    opacity=0.6,
                    marker=dict(size=6),
                    showlegend=False
                ))
            
            # Dados de comparação
            if comparison_samples:
                colors = px.colors.qualitative.Set3
                
                for idx, (class_name, data) in enumerate(comparison_samples.items()):
                    color = colors[idx % len(colors)]
                    
                    # Pontos das distribuições
                    if show_distribution_points and len(data['valence']) > 0:
                        fig_ad.add_trace(go.Scatter(
                            x=data['arousal'],
                            y=data['dominance'],
                            mode='markers',
                            name=f"DIST: {class_name[:15]}...",
                            opacity=0.3,
                            marker=dict(size=4, color=color),
                            showlegend=False
                        ))
                    
                    # Médias com barras de erro
                    fig_ad.add_trace(go.Scatter(
                        x=[data['mean'][1]],
                        y=[data['mean'][2]],
                        mode='markers',
                        name=f"MÉDIA: {class_name[:15]}...",
                        marker=dict(size=10, color=color, symbol='diamond', line=dict(width=2, color='black')),
                        error_x=dict(
                            type='data',
                            array=[data['std'][1]],
                            color=color,
                            thickness=1.5,
                            width=3
                        ),
                        error_y=dict(
                            type='data',
                            array=[data['std'][2]],
                            color=color,
                            thickness=1.5,
                            width=3
                        ),
                        showlegend=False
                    ))
            
            fig_ad.update_layout(
                title='Arousal vs Dominance - Com Incerteza',
                xaxis_title='Arousal',
                yaxis_title='Dominance'
            )
            st.plotly_chart(fig_ad, use_container_width=True)
                
        else:
            st.warning("Nenhum dado corresponde aos filtros aplicados.")
    
    with tab3:
        st.subheader("Dados e Estatísticas das Distribuições")
        
        if len(filtered_df_main) > 0:
            # Estatísticas por classe principal
            st.write("**Estatísticas das Classes Principais:**")
            stats_main = filtered_df_main.groupby('class').agg({
                'valence': ['count', 'mean', 'std', 'min', 'max'],
                'arousal': ['mean', 'std', 'min', 'max'],
                'dominance': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            st.dataframe(stats_main)
        
        if comparison_samples:
            # Estatísticas das distribuições de comparação
            st.write("**Estatísticas das Distribuições de Comparação:**")
            comparison_stats = []
            
            for class_name, data in comparison_samples.items():
                comparison_stats.append({
                    'Distribuição': class_name,
                    'Valence Médio': data['mean'][0],
                    'Valence Std': data['std'][0],
                    'Arousal Médio': data['mean'][1],
                    'Arousal Std': data['std'][1],
                    'Dominance Médio': data['mean'][2],
                    'Dominance Std': data['std'][2],
                    'Pontos Gerados': len(data['valence'])
                })
            
            comparison_df = pd.DataFrame(comparison_stats).round(3)
            st.dataframe(comparison_df)
            
            # Análise de similaridade
            st.write("**Análise de Similaridade entre Distribuições:**")
            
            if len(comparison_stats) > 1:
                similarity_data = []
                for i, dist1 in enumerate(comparison_stats):
                    for j, dist2 in enumerate(comparison_stats):
                        if i < j:
                            # Calcular distância euclidiana entre as médias
                            dist_valence = dist1['Valence Médio'] - dist2['Valence Médio']
                            dist_arousal = dist1['Arousal Médio'] - dist2['Arousal Médio']
                            dist_dominance = dist1['Dominance Médio'] - dist2['Dominance Médio']
                            distance = np.sqrt(dist_valence**2 + dist_arousal**2 + dist_dominance**2)
                            
                            similarity_data.append({
                                'Distribuição 1': dist1['Distribuição'][:20] + "...",
                                'Distribuição 2': dist2['Distribuição'][:20] + "...",
                                'Distância': round(distance, 3)
                            })
                
                similarity_df = pd.DataFrame(similarity_data)
                st.dataframe(similarity_df.sort_values('Distância'))
    
    with tab4:
        st.subheader("Informações sobre a Visualização com Incerteza")
        
        st.markdown("""
        ### Visualização de Distribuições com Incerteza:
        
        Esta visualização mostra não apenas as médias, mas também a incerteza (desvio padrão) 
        das distribuições emocionais no espaço VAD.
        
        **Elementos visuais:**
        - **Pontos principais**: Dados individuais das classes emocionais
        - **Pontos de distribuição**: Amostras geradas a partir das distribuições normais
        - **Médias**: Diamantes representando as médias das distribuições
        - **Barras de erro**: Intervalos de ±1 desvio padrão
        - **Elipsoides**: Volumes representando 1 desvio padrão em 3D
        
        **Interpretação:**
        - Distribuições com elipsoides grandes têm alta variabilidade
        - Distribuições sobrepostas são emocionalmente similares
        - A direção do elipsoide indica quais dimensões têm mais variabilidade
        """)

# Rodapé
st.markdown("---")
st.markdown(
    "Visualizador VAD para análise de distribuições emocionais com incerteza"
)