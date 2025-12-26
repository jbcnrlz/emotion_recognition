import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Análise de Emoções e VAD",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da aplicação
st.title("😊 Dashboard de Análise Emocional e VAD")
st.markdown("---")

# Mapeamento dos valores numéricos para nomes de emoções
EMOTION_MAPPING = {
    '0': 'neutral',
    '1': 'happy',
    '2': 'sad',
    '3': 'surprise',
    '4': 'fear',
    '5': 'disgust',
    '6': 'angry',
    '7': 'contempt',
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust',
    'angry': 'angry',
    'contempt': 'contempt'
}

# Função para mapear valores de emoção
def map_emotion_value(value):
    """Mapeia valores de emoção (numéricos ou textuais) para nomes padronizados"""
    if pd.isna(value):
        return 'unknown'
    
    str_value = str(value).strip().lower()
    return EMOTION_MAPPING.get(str_value, str_value)

# Função para carregar e processar dados
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return df, None
        else:
            return None, "Por favor, faça upload de um arquivo CSV."
    except Exception as e:
        return None, f"Erro ao carregar o arquivo: {str(e)}"

# Função para validar e processar o DataFrame
def validate_and_process_dataframe(df):
    required_columns = ['valence', 'arousal', 'dominance', 'emotion', 'path']
    
    # Verificar colunas obrigatórias
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        return False, f"Colunas obrigatórias faltando: {missing_required}"
    
    # Identificar colunas de emoção (baseado no exemplo fornecido)
    emotion_probability_columns = ['happy', 'contempt', 'elated', 'hopeful', 'surprised', 'proud', 
                                  'loved', 'angry', 'astonished', 'disgusted', 'fearful', 'sad', 
                                  'fatigued', 'neutral']
    
    # Manter apenas as colunas de emoção que existem no DataFrame
    emotion_cols_present = [col for col in emotion_probability_columns if col in df.columns]
    
    # Se não encontrou as colunas esperadas, tentar identificar colunas numéricas que não sejam VAD/emotion/path
    if len(emotion_cols_present) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        emotion_cols_present = [col for col in numeric_cols 
                               if col not in ['valence', 'arousal', 'dominance', 'emotion'] 
                               and col in df.columns]
    
    if len(emotion_cols_present) < 3:
        return False, "Não foram encontradas colunas de emoção suficientes"
    
    # Converter colunas de emoção para numérico, forçando erros para NaN
    for col in emotion_cols_present:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher NaN com 0 para colunas de emoção
    df[emotion_cols_present] = df[emotion_cols_present].fillna(0)
    
    # Processar a coluna emotion (rótulo anotado) - aplicar mapeamento
    if 'emotion' in df.columns:
        # Criar uma nova coluna com o rótulo mapeado
        df['emotion_label'] = df['emotion'].apply(map_emotion_value)
        # Manter a coluna original também
        df['emotion_original'] = df['emotion'].astype(str)
    
    return True, emotion_cols_present

# Função para carregar e exibir imagem
def display_image(image_path):
    try:
        # Verificar se o arquivo existe
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"Imagem: {Path(image_path).name}", use_column_width=True)
        else:
            st.warning(f"Arquivo de imagem não encontrado: {image_path}")
            st.info(f"Caminho procurado: {image_path}")
    except Exception as e:
        st.error(f"Erro ao carregar a imagem: {str(e)}")
        st.info(f"Tentativa de carregar: {image_path}")

# Upload do arquivo na sidebar
st.sidebar.header("📁 Carregar Dados")

upload_option = st.sidebar.radio(
    "Escolha a fonte dos dados:",
    ["Upload de arquivo", "Usar arquivo local"]
)

df = None
emotion_columns = []

if upload_option == "Upload de arquivo":
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo CSV",
        type=['csv'],
        help="Upload do arquivo com distribuições emocionais e valores VAD"
    )
    
    if uploaded_file is not None:
        df, error = load_data(uploaded_file)
        if error:
            st.sidebar.error(error)
        else:
            is_valid, emotion_info = validate_and_process_dataframe(df)
            if is_valid:
                emotion_columns = emotion_info
                st.sidebar.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
                st.sidebar.write(f"**Colunas de emoção detectadas:** {len(emotion_columns)}")
            else:
                st.sidebar.error(f"❌ {emotion_info}")

else:  # Usar arquivo local
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if csv_files:
        selected_file = st.sidebar.selectbox(
            "Selecione um arquivo CSV local:",
            csv_files
        )
        
        if st.sidebar.button("Carregar arquivo local"):
            try:
                df = pd.read_csv(selected_file)
                is_valid, emotion_info = validate_and_process_dataframe(df)
                if is_valid:
                    emotion_columns = emotion_info
                    st.sidebar.success(f"✅ Arquivo '{selected_file}' carregado com sucesso! {len(df)} registros.")
                    st.sidebar.write(f"**Colunas de emoção detectadas:** {len(emotion_columns)}")
                else:
                    st.sidebar.error(f"❌ {emotion_info}")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar arquivo: {str(e)}")
    else:
        st.sidebar.warning("Nenhum arquivo CSV encontrado no diretório atual.")

# Se não há dados carregados, mostrar instruções
if df is None:
    st.info("""
    👆 **Por favor, carregue um arquivo CSV para começar**
    
    **Estrutura esperada do CSV:**
    - Colunas de emoções: `happy`, `contempt`, `elated`, `hopeful`, `surprised`, `proud`, `loved`, `angry`, `astonished`, `disgusted`, `fearful`, `sad`, `fatigued`, `neutral`
    - Colunas VAD: `valence`, `arousal`, `dominance`
    - Coluna de rótulo: `emotion` (valores: 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Angry, 7=Contempt)
    - Coluna de caminho: `path`
    """)
    
    with st.expander("📋 Exemplo de estrutura de dados esperada"):
        example_data = {
            'happy': [0.001, 0.002],
            'contempt': [0.15, 0.12],
            'elated': [0.002, 0.001],
            'valence': [-0.17, 0.25],
            'arousal': [0.05, -0.10],
            'dominance': [-0.14, 0.08],
            'emotion': [1, 0],  # Valores numéricos mapeados
            'path': ['/path/to/image1.jpg', '/path/to/image2.jpg']
        }
        st.dataframe(pd.DataFrame(example_data))
    
    st.stop()

# A partir daqui, o DataFrame foi carregado com sucesso
st.sidebar.header("🔧 Controles de Visualização")

# Seletor de modo de visualização
view_mode = st.sidebar.radio(
    "Modo de Visualização:",
    ["Visão Geral", "Análise Individual", "Comparação entre Imagens"]
)

# Filtros na sidebar
st.sidebar.header("🎛️ Filtros")

# Calcular emoção dominante para cada linha de forma segura
def get_dominant_emotion(row, emotion_cols):
    try:
        # Garantir que estamos trabalhando com valores numéricos
        emotion_values = row[emotion_cols]
        # Encontrar a emoção com maior valor
        dominant_idx = emotion_values.astype(float).idxmax()
        return dominant_idx
    except Exception as e:
        return 'unknown'

dominant_emotions = []
for idx, row in df.iterrows():
    dominant_emotion = get_dominant_emotion(row, emotion_columns)
    dominant_emotions.append(dominant_emotion)

df['dominant_emotion'] = dominant_emotions

# Filtro por emoção dominante (probabilidades)
unique_emotions = sorted(list(set(dominant_emotions)))
selected_emotion = st.sidebar.selectbox(
    "Filtrar por emoção dominante (probabilidades):",
    ["Todas"] + unique_emotions
)

# Filtro por rótulo anotado (coluna emotion_label)
if 'emotion_label' in df.columns:
    unique_annotated_emotions = sorted(df['emotion_label'].unique())
    selected_annotated_emotion = st.sidebar.selectbox(
        "Filtrar por rótulo anotado:",
        ["Todos"] + list(unique_annotated_emotions)
    )
else:
    selected_annotated_emotion = "Todos"

# Aplicar filtros
df_filtered = df.copy()

if selected_emotion != "Todas":
    df_filtered = df_filtered[df_filtered['dominant_emotion'] == selected_emotion]

if selected_annotated_emotion != "Todos":
    df_filtered = df_filtered[df_filtered['emotion_label'] == selected_annotated_emotion]

# Filtro por range de valence
if 'valence' in df.columns:
    min_valence, max_valence = st.sidebar.slider(
        "Faixa de Valence:",
        min_value=float(df['valence'].min()),
        max_value=float(df['valence'].max()),
        value=(float(df['valence'].min()), float(df['valence'].max()))
    )
    
    df_filtered = df_filtered[
        (df_filtered['valence'] >= min_valence) & 
        (df_filtered['valence'] <= max_valence)
    ]

# Informações do dataset atual
st.sidebar.markdown("---")
st.sidebar.header("📊 Informações do Dataset")
st.sidebar.write(f"**Total de registros:** {len(df)}")
st.sidebar.write(f"**Registros filtrados:** {len(df_filtered)}")
st.sidebar.write(f"**Colunas de emoção:** {len(emotion_columns)}")
if 'emotion_label' in df.columns:
    st.sidebar.write(f"**Rótulos anotados únicos:** {len(df['emotion_label'].unique())}")

# Layout principal baseado no modo selecionado
if view_mode == "Visão Geral":
    
    st.header("📊 Visão Geral do Dataset")
    
    # Métricas gerais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total de Imagens", len(df_filtered))
    
    if 'valence' in df_filtered.columns:
        with col2:
            avg_valence = df_filtered['valence'].mean()
            st.metric("Valence Médio", f"{avg_valence:.3f}")
    
    if 'arousal' in df_filtered.columns:
        with col3:
            avg_arousal = df_filtered['arousal'].mean()
            st.metric("Arousal Médio", f"{avg_arousal:.3f}")
    
    if 'dominance' in df_filtered.columns:
        with col4:
            avg_dominance = df_filtered['dominance'].mean()
            st.metric("Dominance Médio", f"{avg_dominance:.3f}")
    
    with col5:
        if 'emotion_label' in df_filtered.columns:
            unique_emotions_count = len(df_filtered['emotion_label'].unique())
            st.metric("Rótulos Únicos", unique_emotions_count)
    
    # Gráficos de visão geral
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Emoções (Média - Probabilidades)")
        
        # Calcular médias de forma segura
        emotion_means = df_filtered[emotion_columns].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=emotion_means.values,
            y=[emotion.capitalize() for emotion in emotion_means.index],
            orientation='h',
            labels={'x': 'Probabilidade Média', 'y': 'Emoção'},
            color=emotion_means.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'emotion_label' in df_filtered.columns:
            st.subheader("Distribuição de Rótulos Anotados")
            emotion_counts = df_filtered['emotion_label'].value_counts()
            
            fig = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Proporção de Rótulos Anotados"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Distribuição de Emoções Dominantes (Probabilidades)")
            dominant_counts = df_filtered['dominant_emotion'].value_counts()
            
            fig = px.pie(
                values=dominant_counts.values,
                names=[emotion.capitalize() for emotion in dominant_counts.index],
                title="Proporção de Emoções Dominantes"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Comparação entre emoção dominante e rótulo anotado
    if 'emotion_label' in df_filtered.columns:
        st.subheader("📊 Comparação: Emoção Dominante vs Rótulo Anotado")
        
        comparison_data = []
        for idx, row in df_filtered.iterrows():
            comparison_data.append({
                'Imagem': f"Img {idx}",
                'Emoção Dominante': row['dominant_emotion'],
                'Rótulo Anotado': row['emotion_label'],
                'Concordância': row['dominant_emotion'] == row['emotion_label']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matriz de confusão
            confusion_matrix = pd.crosstab(
                comparison_df['Rótulo Anotado'], 
                comparison_df['Emoção Dominante'],
                margins=True
            )
            st.write("**Matriz de Correspondência:**")
            st.dataframe(confusion_matrix, use_container_width=True)
        
        with col2:
            # Taxa de concordância
            concordance_rate = comparison_df['Concordância'].mean() * 100
            st.metric("Taxa de Concordância", f"{concordance_rate:.1f}%")
            
            # Top discordâncias
            discordances = comparison_df[~comparison_df['Concordância']]
            if not discordances.empty:
                st.write(f"**Discordâncias:** {len(discordances)}")
                discordance_counts = discordances.groupby(['Emoção Dominante', 'Rótulo Anotado']).size().reset_index(name='Contagem')
                discordance_counts = discordance_counts.sort_values('Contagem', ascending=False).head(10)
                st.dataframe(discordance_counts, use_container_width=True)
    
    # Gráficos VAD
    if all(col in df_filtered.columns for col in ['valence', 'arousal', 'dominance']):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Espaço VAD - Valence vs Arousal")
            color_by = 'emotion_label' if 'emotion_label' in df_filtered.columns else 'dominant_emotion'
            fig = px.scatter(
                df_filtered,
                x='valence',
                y='arousal',
                color=color_by,
                size_max=10,
                hover_data=['dominant_emotion', 'emotion_label'] if 'emotion_label' in df_filtered.columns else ['dominant_emotion'],
                title=f"Valence vs Arousal (colorido por {color_by})"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Coordenadas Paralelas - Dimensões VAD")
            color_column = 'emotion_label' if 'emotion_label' in df_filtered.columns else 'valence'
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=df_filtered[color_column] if color_column == 'valence' else df_filtered.index,
                        colorscale='Viridis',
                        showscale=color_column == 'valence',
                        colorbar=dict(title="Valence") if color_column == 'valence' else None
                    ),
                    dimensions=[
                        dict(
                            label='Valence',
                            values=df_filtered['valence'],
                            range=[df_filtered['valence'].min(), df_filtered['valence'].max()]
                        ),
                        dict(
                            label='Arousal', 
                            values=df_filtered['arousal'],
                            range=[df_filtered['arousal'].min(), df_filtered['arousal'].max()]
                        ),
                        dict(
                            label='Dominance',
                            values=df_filtered['dominance'],
                            range=[df_filtered['dominance'].min(), df_filtered['dominance'].max()]
                        )
                    ]
                )
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Análise Individual":
    
    st.header("🔍 Análise Individual por Imagem")
    
    # Seletor de imagem
    image_options = []
    for i, row in df_filtered.iterrows():
        path_str = str(row['path']) if 'path' in row else f"Imagem {i}"
        filename = Path(path_str).name
        emotion_info = f"Anotado: {row['emotion_label']}" if 'emotion_label' in row else ""
        image_options.append(f"{i} - {filename} (Prob: {row['dominant_emotion']}, {emotion_info})")
    
    if df_filtered.empty:
        st.warning("Nenhum dado disponível com os filtros atuais.")
    else:
        selected_image = st.selectbox(
            "Selecione uma imagem para análise:",
            options=df_filtered.index,
            format_func=lambda x: image_options[list(df_filtered.index).index(x)]
        )
        
        if selected_image is not None and selected_image in df_filtered.index:
            selected_row = df_filtered.loc[selected_image]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # EXIBIR A IMAGEM AQUI
                if 'path' in selected_row:
                    image_path = selected_row['path']
                    st.subheader("🖼️ Imagem")
                    display_image(image_path)
                else:
                    st.warning("Nenhum caminho de imagem disponível para esta entrada.")
                
                # Informações da imagem
                st.subheader("📝 Informações da Imagem")
                
                if 'path' in selected_row:
                    st.write(f"**Arquivo:** {Path(selected_row['path']).name}")
                
                st.write(f"**Emoção Dominante (Probabilidades):** {selected_row['dominant_emotion'].capitalize()}")
                
                if 'emotion_label' in selected_row:
                    st.write(f"**Rótulo Anotado:** {selected_row['emotion_label'].capitalize()}")
                    if 'emotion_original' in selected_row:
                        st.write(f"**Valor Original:** {selected_row['emotion_original']}")
                    # Verificar concordância
                    concordance = selected_row['dominant_emotion'] == selected_row['emotion_label']
                    st.write(f"**Concordância:** {'✅ Sim' if concordance else '❌ Não'}")
                
                # Valores VAD
                if all(col in selected_row for col in ['valence', 'arousal', 'dominance']):
                    st.subheader("🎯 Dimensões VAD")
                    vad_data = {
                        'Dimensão': ['Valence', 'Arousal', 'Dominance'],
                        'Valor': [selected_row['valence'], selected_row['arousal'], selected_row['dominance']]
                    }
                    st.dataframe(pd.DataFrame(vad_data), use_container_width=True)
                    
                    # Gráfico radar para VAD
                    categories = ['Valence', 'Arousal', 'Dominance']
                    values = [
                        selected_row['valence'],
                        selected_row['arousal'], 
                        selected_row['dominance']
                    ]
                    
                    fig = go.Figure(data=
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=categories + [categories[0]],
                            fill='toself',
                            name='VAD Values'
                        )
                    )
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[min(values) * 1.1, max(values) * 1.1]
                            )),
                        showlegend=False,
                        height=300,
                        title="Perfil VAD"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribuição Emocional (Probabilidades)")
                
                emotions_data = selected_row[emotion_columns].sort_values(ascending=False)
                dominant_emotion = emotions_data.index[0]
                
                fig = px.bar(
                    x=emotions_data.values,
                    y=[emotion.capitalize() for emotion in emotions_data.index],
                    orientation='h',
                    labels={'x': 'Probabilidade', 'y': 'Emoção'},
                    color=emotions_data.values,
                    color_continuous_scale='viridis',
                    title=f"Emoção Dominante: {dominant_emotion.capitalize()}"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Comparação entre Imagens":
    
    st.header("📈 Comparação entre Imagens")
    
    if df_filtered.empty:
        st.warning("Nenhum dado disponível com os filtros atuais.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Selecionar Imagens para Comparar")
            
            image_options = []
            for i, row in df_filtered.iterrows():
                path_str = str(row['path']) if 'path' in row else f"Imagem {i}"
                filename = Path(path_str).name
                emotion_info = f"Anotado: {row['emotion_label']}" if 'emotion_label' in row else ""
                image_options.append(f"{i} - {filename} (Prob: {row['dominant_emotion']}, {emotion_info})")
            
            selected_images = st.multiselect(
                "Selecione até 5 imagens para comparar:",
                options=df_filtered.index,
                format_func=lambda x: image_options[list(df_filtered.index).index(x)],
                max_selections=5
            )
        
        if selected_images:
            st.subheader("Comparação de Distribuições Emocionais")
            
            comparison_data = []
            for img_idx in selected_images:
                row = df_filtered.loc[img_idx]
                img_label = f"Img {img_idx}"
                for emotion in emotion_columns:
                    comparison_data.append({
                        'Imagem': img_label,
                        'Emoção': emotion.capitalize(),
                        'Probabilidade': row[emotion],
                        'Dominante': emotion == row[emotion_columns].idxmax()
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Emoção',
                y='Probabilidade',
                color='Imagem',
                barmode='group',
                title="Comparação de Distribuições Emocionais",
                hover_data=['Dominante']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de comparação com rótulos anotados
            if 'emotion_label' in df_filtered.columns:
                st.subheader("Comparação de Rótulos Anotados")
                comparison_summary = []
                for img_idx in selected_images:
                    row = df_filtered.loc[img_idx]
                    comparison_summary.append({
                        'Imagem': f"Img {img_idx}",
                        'Arquivo': Path(row['path']).name if 'path' in row else 'N/A',
                        'Emoção Dominante': row['dominant_emotion'],
                        'Rótulo Anotado': row['emotion_label'],
                        'Concordância': '✅' if row['dominant_emotion'] == row['emotion_label'] else '❌'
                    })
                
                st.dataframe(pd.DataFrame(comparison_summary), use_container_width=True)
            
            # Comparação VAD
            if all(col in df_filtered.columns for col in ['valence', 'arousal', 'dominance']):
                st.subheader("Comparação de Dimensões VAD")
                
                vad_comparison = []
                for img_idx in selected_images:
                    row = df_filtered.loc[img_idx]
                    vad_comparison.append({
                        'Imagem': f"Img {img_idx}",
                        'Valence': row['valence'],
                        'Arousal': row['arousal'],
                        'Dominance': row['dominance']
                    })
                
                vad_df = pd.DataFrame(vad_comparison)
                st.dataframe(vad_df.set_index('Imagem'), use_container_width=True)

# Informações adicionais
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Sobre os Dados")
st.sidebar.info("""
**VAD Dimensions:**
- **Valence**: Prazer (positivo) vs Desprazer (negativo)
- **Arousal**: Excitação/Ativação alta vs baixa  
- **Dominance**: Controle alto vs baixo

**Rótulos:**
- **Probabilidades**: Distribuição de emoções calculada
- **Anotado**: Rótulo manual atribuído (0-7 mapeado para nomes)

**Mapeamento:**
- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Surprise
- 4: Fear
- 5: Disgust
- 6: Angry
- 7: Contempt
""")

# Mostrar dados brutos
if st.sidebar.checkbox("Mostrar dados brutos"):
    st.header("📋 Dados Brutos")
    st.dataframe(df_filtered, use_container_width=True)

# Informações da estrutura
with st.expander("📁 Informações da Estrutura do Arquivo"):
    st.write(f"**Colunas detectadas:** {list(df.columns)}")
    st.write(f"**Colunas de emoção (probabilidades):** {emotion_columns}")
    if 'emotion_label' in df.columns:
        st.write(f"**Rótulos anotados únicos:** {list(df['emotion_label'].unique())}")
    st.write(f"**Total de registros:** {len(df)}")

st.markdown("---")
st.caption("Dashboard desenvolvido para análise de distribuições emocionais e dimensões VAD")