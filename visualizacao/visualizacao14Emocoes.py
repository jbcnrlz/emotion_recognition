import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import io

# Configuração da página
st.set_page_config(page_title="Comparação de Modelos de Emoção", layout="wide")
st.title("📊 Análise Comparativa: Previsões vs Ground Truth")

# Função para processar os dados
@st.cache_data
def process_data(preds_file, labels_file):
    """Processa os arquivos CSV carregados"""
    # Ler arquivo de previsões
    preds = pd.read_csv(preds_file)
    
    # Ler arquivo de ground truth
    labels = pd.read_csv(labels_file)
    
    # Identificar automaticamente as colunas de emoção
    # Assumindo que as colunas de emoção são todas as colunas até a última numérica
    emotion_cols = []
    
    # Para previsões: colunas que não são 'file' ou terminam com 'label'
    for col in preds.columns:
        if col not in ['file', 'emotion_label'] and not col.startswith('valence') and not col.startswith('arousal') and not col.startswith('dominance'):
            emotion_cols.append(col)
    
    # Ordenar as colunas de emoção para garantir consistência
    emotion_cols = sorted(emotion_cols)
    
    # Para previsões: pegar apenas colunas de emoção
    preds_emotions = preds[emotion_cols].copy()
    # Para labels: pegar apenas colunas de emoção
    labels_emotions = labels[emotion_cols].copy()
    
    # Adicionar colunas de identificação
    preds_emotions['file'] = preds.get('file', '')
    
    # Tentar identificar a coluna de classe predita
    pred_class_col = None
    for col in preds.columns:
        if 'label' in col.lower() and col not in emotion_cols:
            pred_class_col = col
            break
    
    if pred_class_col:
        preds_emotions['pred_class'] = preds[pred_class_col]
    else:
        # Se não encontrar, usar argmax das probabilidades
        preds_emotions['pred_class'] = preds_emotions[emotion_cols].idxmax(axis=1)
        # Converter nomes para índices se necessário
        # Esta é uma simplificação - pode precisar de ajuste dependendo dos dados
        preds_emotions['pred_class'] = preds_emotions['pred_class'].apply(
            lambda x: emotion_cols.index(x) if x in emotion_cols else 0
        )
    
    labels_emotions['file'] = labels.get('file', '')
    
    # Tentar identificar a coluna de classe verdadeira
    true_class_col = None
    for col in labels.columns:
        if 'label' in col.lower() and col not in emotion_cols:
            true_class_col = col
            break
    
    if true_class_col:
        labels_emotions['true_class'] = labels[true_class_col]
    else:
        # Se não encontrar, usar argmax das probabilidades
        labels_emotions['true_class'] = labels_emotions[emotion_cols].idxmax(axis=1)
        # Converter nomes para índices se necessário
        labels_emotions['true_class'] = labels_emotions['true_class'].apply(
            lambda x: emotion_cols.index(x) if x in emotion_cols else 0
        )
    
    return preds_emotions, labels_emotions, emotion_cols

# Sidebar para upload de arquivos
st.sidebar.header("📁 Upload de Arquivos")

# Upload de arquivos
preds_file = st.sidebar.file_uploader(
    "Arquivo de Previsões (Modelo)",
    type=['csv'],
    help="Selecione o arquivo CSV com as previsões do modelo"
)

labels_file = st.sidebar.file_uploader(
    "Arquivo de Ground Truth (Labels)",
    type=['csv'],
    help="Selecione o arquivo CSV com as labels verdadeiras"
)

# Configurações de análise
st.sidebar.header("⚙️ Configurações de Análise")

# Selecionar colunas específicas se os arquivos forem carregados
if preds_file is not None and labels_file is not None:
    # Pré-visualizar os dados para identificar colunas
    preds_preview = pd.read_csv(preds_file)
    labels_preview = pd.read_csv(labels_file)
    
    # Voltar para o início do arquivo
    preds_file.seek(0)
    labels_file.seek(0)
    
    # Mostrar informações dos arquivos
    st.sidebar.info(f"**Arquivo de Previsões:** {preds_preview.shape[0]} linhas, {preds_preview.shape[1]} colunas")
    st.sidebar.info(f"**Arquivo de Labels:** {labels_preview.shape[0]} linhas, {labels_preview.shape[1]} colunas")
    
    # Permitir seleção manual de colunas de emoção
    st.sidebar.subheader("🔧 Configuração Avançada")
    
    # Mostrar colunas disponíveis
    st.sidebar.write("**Colunas disponíveis no arquivo de previsões:**")
    st.sidebar.write(list(preds_preview.columns))
    
    # Opção para usar detecção automática ou manual
    detection_mode = st.sidebar.radio(
        "Modo de detecção de colunas",
        ["Automático", "Manual"]
    )
    
    if detection_mode == "Manual":
        # Permitir que o usuário especifique as colunas de emoção
        emotion_cols_input = st.sidebar.text_input(
            "Colunas de emoção (separadas por vírgula)",
            value="happy,contempt,elated,surprised,loved,protected,astonished,disgusted,angry,fearful,sad,neutral",
            help="Digite os nomes exatos das colunas de emoção, separados por vírgula"
        )
        emotion_cols_manual = [col.strip() for col in emotion_cols_input.split(',')]
        
        # Permitir especificar coluna de classe
        pred_class_col_manual = st.sidebar.text_input(
            "Coluna de classe predita",
            value="emotion_label",
            help="Nome da coluna que contém a classe predita"
        )
        
        true_class_col_manual = st.sidebar.text_input(
            "Coluna de classe verdadeira",
            value="emotion_label",
            help="Nome da coluna que contém a classe verdadeira"
        )

metric_choice = st.sidebar.selectbox(
    "Métrica de Similaridade Global",
    ["Jensen-Shannon Divergence", "Wasserstein Distance", "Cosine Similarity", "Pearson Correlation"]
)

show_individual = st.sidebar.checkbox("Mostrar Análise por Amostra", value=False)
top_k = st.sidebar.slider("Top-K para Rank", 1, 10, 3)

# Verificar se os arquivos foram carregados
if preds_file is not None and labels_file is not None:
    try:
        # Processar os dados
        with st.spinner("Processando dados..."):
            preds_df, labels_df, emotion_cols = process_data(preds_file, labels_file)
        
        # Se modo manual foi selecionado, ajustar as colunas
        if detection_mode == "Manual" and 'emotion_cols_manual' in locals():
            # Filtrar apenas colunas que existem em ambos dataframes
            available_cols = []
            for col in emotion_cols_manual:
                if col in preds_df.columns and col in labels_df.columns:
                    available_cols.append(col)
            
            if available_cols:
                emotion_cols = available_cols
                st.sidebar.success(f"Usando {len(emotion_cols)} colunas de emoção")
            else:
                st.sidebar.warning("Nenhuma das colunas especificadas foi encontrada. Usando detecção automática.")
        
        # Converter escolha da métrica para formato de código
        metric_map = {
            "Jensen-Shannon Divergence": "js",
            "Wasserstein Distance": "wasserstein",
            "Cosine Similarity": "cosine",
            "Pearson Correlation": "pearson"
        }
        
        # Funções para cálculo de métricas
        def calculate_global_similarity(preds, labels, emotion_cols, metric="js"):
            """Calcula similaridade global entre distribuições"""
            similarities = []
            
            for i in range(len(preds)):
                p = preds.iloc[i][emotion_cols].values.astype(float)
                l = labels.iloc[i][emotion_cols].values.astype(float)
                
                if metric == "js":
                    # Jensen-Shannon Divergence
                    js = jensenshannon(p, l) ** 2
                    similarities.append(1 - js if not np.isnan(js) else 0)
                elif metric == "wasserstein":
                    # Wasserstein Distance
                    wd = wasserstein_distance(p, l)
                    similarities.append(1 / (1 + wd))
                elif metric == "cosine":
                    # Cosine Similarity
                    cos_sim = np.dot(p, l) / (np.linalg.norm(p) * np.linalg.norm(l))
                    similarities.append(cos_sim if not np.isnan(cos_sim) else 0)
                elif metric == "pearson":
                    # Pearson Correlation
                    corr, _ = spearmanr(p, l)
                    similarities.append(corr if not np.isnan(corr) else 0)
            
            return np.mean(similarities), similarities

        def calculate_accuracy(preds, labels, emotion_cols):
            """Calcula acurácia baseada nas classes mais prováveis"""
            pred_classes = []
            true_classes = []
            
            for i in range(len(preds)):
                # Classe mais provável das previsões
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                pred_class = np.argmax(pred_probs)
                pred_classes.append(pred_class)
                
                # Classe verdadeira - garantir que é inteiro
                true_class = labels.iloc[i]['true_class']
                if isinstance(true_class, str):
                    # Tentar converter para inteiro
                    try:
                        true_class = int(float(true_class))
                    except:
                        # Se falhar, usar o índice da emoção máxima
                        true_probs = labels.iloc[i][emotion_cols].values.astype(float)
                        true_class = np.argmax(true_probs)
                else:
                    true_class = int(true_class)
                true_classes.append(true_class)
            
            accuracy = accuracy_score(true_classes, pred_classes)
            
            # Top-K accuracy
            top_k_correct = 0
            for i in range(len(preds)):
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                true_class = true_classes[i]
                
                # Pegar índices das top-K probabilidades
                top_k_indices = np.argsort(pred_probs)[-top_k:][::-1]
                if true_class in top_k_indices:
                    top_k_correct += 1
            
            top_k_accuracy = top_k_correct / len(preds)
            
            return accuracy, top_k_accuracy, pred_classes, true_classes

        def calculate_rank_metrics(preds, labels, emotion_cols, true_classes):
            """Calcula métricas baseadas em rank"""
            rank_positions = []
            
            for i in range(len(preds)):
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                true_class = true_classes[i]
                
                # Ordenar probabilidades em ordem decrescente
                sorted_indices = np.argsort(pred_probs)[::-1]
                
                # Encontrar posição da classe verdadeira
                rank_position = np.where(sorted_indices == true_class)[0]
                if len(rank_position) > 0:
                    rank_positions.append(rank_position[0] + 1)
            
            mean_rank = np.mean(rank_positions) if rank_positions else 0
            median_rank = np.median(rank_positions) if rank_positions else 0
            
            # Distribuição de ranks
            rank_distribution = {}
            for rank in rank_positions:
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
            
            return mean_rank, median_rank, rank_distribution, rank_positions
        
        # Cálculo das métricas
        similarity_metric = metric_map[metric_choice]
        global_similarity, individual_similarities = calculate_global_similarity(
            preds_df, labels_df, emotion_cols, similarity_metric
        )
        accuracy, top_k_accuracy, pred_classes, true_classes = calculate_accuracy(
            preds_df, labels_df, emotion_cols
        )
        mean_rank, median_rank, rank_distribution, rank_positions = calculate_rank_metrics(
            preds_df, labels_df, emotion_cols, true_classes
        )
        
        # Layout principal
        st.success("✅ Dados carregados e processados com sucesso!")
        
        # Mostrar informações dos dados
        with st.expander("📋 Visualizar Dados"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Previsões (primeiras 5 linhas):**")
                st.dataframe(preds_df.head())
            with col2:
                st.write("**Ground Truth (primeiras 5 linhas):**")
                st.dataframe(labels_df.head())
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Similaridade Global",
                value=f"{global_similarity:.3f}",
                delta=f"Métrica: {metric_choice}"
            )
            
            # Gauge chart para similaridade
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=global_similarity * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Similaridade Global (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.metric(
                label="Acurácia (Top-1)",
                value=f"{accuracy:.3f}"
            )
            st.metric(
                label=f"Acurácia (Top-{top_k})",
                value=f"{top_k_accuracy:.3f}"
            )
            
            # Gráfico de barras para acurácia
            fig_acc = go.Figure(data=[
                go.Bar(
                    name='Acurácia',
                    x=['Top-1', f'Top-{top_k}'],
                    y=[accuracy, top_k_accuracy],
                    marker_color=['blue', 'green']
                )
            ])
            fig_acc.update_layout(
                title="Acurácia por Métrica",
                yaxis_title="Acurácia",
                yaxis_range=[0, 1],
                height=300
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col3:
            st.metric(
                label="Rank Médio",
                value=f"{mean_rank:.2f}"
            )
            st.metric(
                label="Rank Mediano",
                value=f"{median_rank:.0f}"
            )
            
            # Distribuição de ranks
            if rank_distribution:
                ranks = list(rank_distribution.keys())
                counts = list(rank_distribution.values())
                
                fig_rank = go.Figure(data=[
                    go.Bar(
                        x=[f'Rank {r}' for r in ranks],
                        y=counts,
                        marker_color='purple'
                    )
                ])
                fig_rank.update_layout(
                    title="Distribuição de Ranks",
                    xaxis_title="Rank",
                    yaxis_title="Frequência",
                    height=300
                )
                st.plotly_chart(fig_rank, use_container_width=True)

        # Matriz de Confusão
        st.subheader("🎯 Matriz de Confusão")
        try:
            cm = confusion_matrix(true_classes, pred_classes, labels=range(len(emotion_cols)))
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred {i}" for i in range(len(emotion_cols))],
                y=[f"True {i}" for i in range(len(emotion_cols))],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_cm.update_layout(
                title="Matriz de Confusão",
                xaxis_title="Classe Prevista",
                yaxis_title="Classe Verdadeira",
                height=500
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível gerar a matriz de confusão: {e}")

        # Comparação de Distribuições
        st.subheader("📈 Comparação de Distribuições de Emoções")

        # Selecionar uma amostra para visualização
        if show_individual:
            sample_idx = st.slider("Selecionar Amostra", 0, len(preds_df)-1, 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição prevista
                pred_sample = preds_df.iloc[sample_idx][emotion_cols]
                true_sample = labels_df.iloc[sample_idx][emotion_cols]
                
                fig_pred = go.Figure(data=[
                    go.Bar(
                        x=emotion_cols,
                        y=pred_sample.values.astype(float),
                        name='Previsto',
                        marker_color='blue'
                    ),
                    go.Bar(
                        x=emotion_cols,
                        y=true_sample.values.astype(float),
                        name='Ground Truth',
                        marker_color='red'
                    )
                ])
                fig_pred.update_layout(
                    title=f"Distribuição - Amostra {sample_idx}",
                    xaxis_title="Emoção",
                    yaxis_title="Probabilidade",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Mostrar métricas dessa amostra
                st.info(f"""
                **Métricas da Amostra {sample_idx}:**
                - Similaridade: {individual_similarities[sample_idx]:.3f}
                - Classe Prevista: {pred_classes[sample_idx]} ({emotion_cols[pred_classes[sample_idx]] if pred_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                - Classe Verdadeira: {true_classes[sample_idx]} ({emotion_cols[true_classes[sample_idx]] if true_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                - Rank: {rank_positions[sample_idx] if sample_idx < len(rank_positions) else 'N/A'}
                """)

            with col2:
                # Gráfico radar para visualização multivariada
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=pred_sample.values.astype(float),
                    theta=emotion_cols,
                    fill='toself',
                    name='Previsto'
                ))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=true_sample.values.astype(float),
                    theta=emotion_cols,
                    fill='toself',
                    name='Ground Truth'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(pred_sample.values.astype(float)), max(true_sample.values.astype(float)))]
                        )),
                    showlegend=True,
                    title=f"Visualização Radar - Amostra {sample_idx}",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # Análise Global das Distribuições
        st.subheader("🌍 Análise Global das Distribuições")

        # Calcular médias globais
        pred_means = preds_df[emotion_cols].apply(pd.to_numeric, errors='coerce').mean()
        label_means = labels_df[emotion_cols].apply(pd.to_numeric, errors='coerce').mean()

        fig_global = go.Figure(data=[
            go.Bar(
                x=emotion_cols,
                y=pred_means.values,
                name='Média Prevista',
                marker_color='lightblue'
            ),
            go.Bar(
                x=emotion_cols,
                y=label_means.values,
                name='Média Ground Truth',
                marker_color='salmon'
            )
        ])

        fig_global.update_layout(
            title="Médias Globais das Distribuições",
            xaxis_title="Emoção",
            yaxis_title="Probabilidade Média",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # Tabela de Métricas Detalhadas
        st.subheader("📋 Métricas Detalhadas por Classe")

        # Calcular métricas por classe
        try:
            class_report = classification_report(
                true_classes, 
                pred_classes, 
                target_names=[f"{emotion_cols[i] if i < len(emotion_cols) else f'Classe {i}'}" for i in range(len(emotion_cols))],
                output_dict=True
            )
            
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
        except Exception as e:
            st.warning(f"Não foi possível gerar o relatório detalhado: {e}")

        # Histograma de Similaridades
        st.subheader("📊 Distribuição das Similaridades Individuais")

        fig_hist = go.Figure(data=[
            go.Histogram(
                x=individual_similarities,
                nbinsx=20,
                marker_color='teal',
                opacity=0.7
            )
        ])
        fig_hist.update_layout(
            title="Distribuição das Similaridades por Amostra",
            xaxis_title="Similaridade",
            yaxis_title="Frequência",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Resumo Executivo
        st.subheader("📝 Resumo Executivo")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Pontos Fortes:**
            - Similaridade global: **{global_similarity:.3f}**
            - Acurácia Top-1: **{accuracy:.3f}**
            - Acurácia Top-{top_k}: **{top_k_accuracy:.3f}**
            - Rank médio: **{mean_rank:.2f}**
            """)

        with col2:
            # Sugestões baseadas nos resultados
            suggestions = []
            if accuracy < 0.5:
                suggestions.append("Acurácia baixa - considere ajustar o modelo")
            if global_similarity < 0.6:
                suggestions.append("Similaridade global moderada - distribuições podem estar diferentes")
            if mean_rank > 3:
                suggestions.append("Rank médio alto - o modelo tem dificuldade em rankear corretamente")
            
            if suggestions:
                st.warning("**Áreas de Melhoria:**")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.success("**Desempenho satisfatório!**")

        # Adicionar download de relatório
        st.subheader("📥 Exportar Resultados")
        
        if st.button("Gerar Relatório Completo"):
            # Criar relatório
            report_data = {
                'file': preds_df['file'],
                'true_class': true_classes,
                'pred_class': pred_classes,
                'similarity': individual_similarities,
                'rank': rank_positions + [np.nan] * (len(preds_df) - len(rank_positions)),
                'correct': [1 if t == p else 0 for t, p in zip(true_classes, pred_classes)]
            }
            
            # Adicionar probabilidades
            for i, emotion in enumerate(emotion_cols):
                report_data[f'pred_{emotion}'] = preds_df[emotion].values
                report_data[f'true_{emotion}'] = labels_df[emotion].values
            
            report_df = pd.DataFrame(report_data)
            
            # Converter para CSV
            csv = report_df.to_csv(index=False)
            
            # Botão de download
            st.download_button(
                label="📥 Download Relatório CSV",
                data=csv,
                file_name="analise_modelo_emocao.csv",
                mime="text/csv"
            )
            
            # Estatísticas resumidas
            summary_stats = {
                'Metrica': ['Similaridade Global', 'Acurácia Top-1', f'Acurácia Top-{top_k}', 'Rank Médio', 'Rank Mediano'],
                'Valor': [f"{global_similarity:.4f}", f"{accuracy:.4f}", f"{top_k_accuracy:.4f}", f"{mean_rank:.2f}", f"{median_rank:.0f}"]
            }
            summary_df = pd.DataFrame(summary_stats)
            
            st.write("**Resumo Estatístico:**")
            st.dataframe(summary_df)
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.info("""
        **Dicas para solução de problemas:**
        1. Verifique se os arquivos têm o mesmo número de linhas
        2. Confirme que as colunas de emoção têm os mesmos nomes em ambos os arquivos
        3. Certifique-se de que as colunas de classe são numéricas
        4. Use o modo manual na sidebar para especificar as colunas
        """)
        
else:
    # Mensagem inicial
    st.markdown("""
    ## Bem-vindo ao Analisador de Modelos de Emoção! 🤖
    
    **Como usar:**
    1. **Faça upload dos arquivos** na sidebar à esquerda:
       - **Arquivo de Previsões**: Resultados do seu modelo
       - **Arquivo de Ground Truth**: Labels verdadeiras
    
    2. **Ajuste as configurações** de análise
    
    3. **Visualize os resultados** automaticamente
    
    **Formatos esperados:**
    - Arquivos CSV com colunas de emoção (happy, sad, angry, etc.)
    - Uma coluna para a classe (normalmente 'emotion_label')
    - Uma coluna 'file' com os caminhos das imagens (opcional)
    
    **Exemplos de formatos:**
    ```
    # Arquivo de previsões:
    happy,contempt,...,neutral,emotion_label,file
    
    # Arquivo de ground truth:
    happy,contempt,...,neutral,valence,arousal,dominance,emotion_label,file
    ```
    """)
    
    # Mostrar exemplo de formato
    with st.expander("👀 Ver exemplo de formato dos dados"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Exemplo - Arquivo de Previsões:**")
            st.code("""
happy,contempt,elated,surprised,loved,protected,astonished,disgusted,angry,fearful,sad,neutral,emotion_label,file
0.0019,0.0567,0.0456,0.0712,0.0303,0.0520,0.0283,0.1426,0.0612,0.1153,0.0745,0.2090,11,image1.jpg
0.0012,0.0752,0.0343,0.0661,0.0864,0.0735,0.0347,0.0778,0.1645,0.0714,0.0411,0.0853,8,image2.jpg
            """)
        with col2:
            st.write("**Exemplo - Arquivo de Ground Truth:**")
            st.code("""
happy,contempt,elated,surprised,loved,protected,astonished,disgusted,angry,fearful,sad,neutral,valence,arousal,dominance,emotion_label,file
2.7e-09,0.1083,0.0028,8.3e-05,0.0045,0.0006,3.0e-06,0.4799,0.0236,0.2314,0.1481,0.0002,0.0006,7.7e-20,-0.4694,0.8178,0.2155,6,image1.jpg
1.4e-05,0.1460,0.0183,0.0033,0.1174,0.0027,0.0019,0.0746,0.5910,0.0205,0.0227,0.0003,0.0013,6.7e-13,-0.1190,0.7778,0.0043,4,image2.jpg
            """)

# CSS personalizado
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    /* Estilo para os botões de upload */
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)