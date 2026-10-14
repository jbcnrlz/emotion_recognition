import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Mapeamento de emoções (baseado no índice e na ordem do cabeçalho)
EMOTION_MAP = {
    0: 'neutral',
    1: 'happy', 
    2: 'sad',
    3: 'surprised',
    4: 'fear',
    5: 'disgust',
    6: 'angry',
    7: 'contempt'
}

# Ordem das colunas conforme o cabeçalho
EMOTION_COLUMNS = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'angry', 'contempt']

def safe_kl_divergence(p, q, eps=1e-10):
    """Calcula KL Divergence de forma segura"""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe))

def js_divergence(p, q, eps=1e-10):
    """Calcula Jensen-Shannon Divergence"""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    m = 0.5 * (p_safe + q_safe)
    return 0.5 * safe_kl_divergence(p_safe, m) + 0.5 * safe_kl_divergence(q_safe, m)

def calculate_accuracy_metrics(df_pred, df_true):
    """Calcula métricas de acurácia baseadas na coluna emotion"""
    # Garantir mesma ordem
    df_pred = df_pred.sort_values('file').reset_index(drop=True)
    df_true = df_true.sort_values('file').reset_index(drop=True)
    
    # Extrair predições (índice da maior probabilidade)
    pred_probs = df_pred[EMOTION_COLUMNS].values
    true_probs = df_true[EMOTION_COLUMNS].values
    
    # Predição: índice da maior probabilidade
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = np.argmax(true_probs, axis=1)
    
    # Se existir coluna emotion, usar ela
    if 'emotion' in df_true.columns:
        y_true = df_true['emotion'].values.astype(int)
    
    if 'emotion' in df_pred.columns:
        y_pred = df_pred['emotion'].values.astype(int)
    
    # Calcular métricas
    accuracy = np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred, labels=range(8))
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=range(8), target_names=EMOTION_COLUMNS)
    
    return accuracy, cm, report, y_pred, y_true

def calculate_distribution_metrics(df_pred, df_true):
    """Calcula métricas entre distribuições de probabilidade"""
    # Garantir mesma ordem
    df_pred = df_pred.sort_values('file').reset_index(drop=True)
    df_true = df_true.sort_values('file').reset_index(drop=True)
    
    # Extrair matrizes
    P = df_pred[EMOTION_COLUMNS].values
    Q = df_true[EMOTION_COLUMNS].values
    
    # Calcular métricas por linha
    metrics = {
        'kl_div': [],
        'js_div': [],
        'cross_entropy': [],
        'mse': [],
        'mae': [],
        'cosine_similarity': []
    }
    
    for i in range(len(P)):
        p = P[i]
        q = Q[i]
        
        metrics['kl_div'].append(safe_kl_divergence(p, q))
        metrics['js_div'].append(js_divergence(p, q))
        metrics['cross_entropy'].append(-np.sum(p * np.log(np.clip(q, 1e-10, 1.0))))
        metrics['mse'].append(np.mean((p - q) ** 2))
        metrics['mae'].append(np.mean(np.abs(p - q)))
        metrics['cosine_similarity'].append(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))
    
    return metrics, df_pred, df_true

def create_metrics_plot(metrics):
    """Cria gráfico com as métricas de distribuição"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['KL Divergence', 'JS Divergence', 'Cross-Entropy', 
                       'MSE', 'MAE', 'Cosine Similarity'],
        vertical_spacing=0.12
    )
    
    # KL Divergence
    fig.add_trace(
        go.Histogram(x=metrics['kl_div'], name='KL Div', nbinsx=30),
        row=1, col=1
    )
    
    # JS Divergence
    fig.add_trace(
        go.Histogram(x=metrics['js_div'], name='JS Div', nbinsx=30),
        row=1, col=2
    )
    
    # Cross-Entropy
    fig.add_trace(
        go.Histogram(x=metrics['cross_entropy'], name='Cross-Entropy', nbinsx=30),
        row=1, col=3
    )
    
    # MSE
    fig.add_trace(
        go.Histogram(x=metrics['mse'], name='MSE', nbinsx=30),
        row=2, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Histogram(x=metrics['mae'], name='MAE', nbinsx=30),
        row=2, col=2
    )
    
    # Cosine Similarity
    fig.add_trace(
        go.Histogram(x=metrics['cosine_similarity'], name='Cosine Sim', nbinsx=30),
        row=2, col=3
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Distribuição das Métricas")
    return fig

def create_confusion_matrix_plot(cm):
    """Cria heatmap da matriz de confusão"""
    emotions_list = EMOTION_COLUMNS
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions_list, 
                yticklabels=emotions_list,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Matriz de Confusão')
    
    return fig

def create_comparison_plot(df_pred, df_true, sample_idx=0):
    """Cria gráfico de comparação para uma amostra específica"""
    pred_probs = df_pred.iloc[sample_idx][EMOTION_COLUMNS].values
    true_probs = df_true.iloc[sample_idx][EMOTION_COLUMNS].values
    filename = df_pred.iloc[sample_idx]['file']
    
    # Obter predições de classe
    pred_class = np.argmax(pred_probs)
    true_class = np.argmax(true_probs)
    
    # Se existir coluna emotion, usar ela
    if 'emotion' in df_pred.columns:
        pred_class = int(df_pred.iloc[sample_idx]['emotion'])
    if 'emotion' in df_true.columns:
        true_class = int(df_true.iloc[sample_idx]['emotion'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=EMOTION_COLUMNS,
        y=pred_probs,
        name=f'Previsto ({EMOTION_MAP[pred_class]})',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=EMOTION_COLUMNS,
        y=true_probs,
        name=f'Verdadeiro ({EMOTION_MAP[true_class]})',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f"Comparação de Distribuições - {filename}",
        xaxis_title="Emoções",
        yaxis_title="Probabilidade",
        barmode='group'
    )
    
    return fig

def main():
    st.set_page_config(page_title="Comparador de Distribuições", layout="wide")
    
    st.title("📊 Comparador de Distribuições de Emoções")
    st.markdown("""
    Faça upload de dois arquivos CSV com distribuições de probabilidade para comparar as métricas.
    **Estrutura esperada:** `neutral, happy, sad, surprised, fear, disgust, angry, contempt, emotion, file`
    """)
    
    # Upload dos arquivos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Arquivo Previsto")
        pred_file = st.file_uploader("Upload CSV com previsões", type=['csv'], key="pred")
    
    with col2:
        st.subheader("📁 Arquivo Verdadeiro")
        true_file = st.file_uploader("Upload CSV com valores verdadeiros", type=['csv'], key="true")
    
    if pred_file and true_file:
        try:
            # Carregar dados
            df_pred = pd.read_csv(pred_file)
            df_true = pd.read_csv(true_file)
            
            # Verificar estrutura mínima
            required_cols = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'angry', 'contempt', 'file']
            
            if not all(col in df_pred.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df_pred.columns]
                st.error(f"❌ Arquivo previsto não tem a estrutura esperada. Colunas faltantes: {missing_cols}")
                return
                
            if not all(col in df_true.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df_true.columns]
                st.error(f"❌ Arquivo verdadeiro não tem a estrutura esperada. Colunas faltantes: {missing_cols}")
                return
            
            st.success("✅ Arquivos carregados com sucesso!")
            
            # Mostrar preview dos dados
            with st.expander("👀 Visualizar Estrutura dos Dados"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Arquivo Previsto (primeiras 5 linhas):**")
                    st.dataframe(df_pred.head())
                with col2:
                    st.write("**Arquivo Verdadeiro (primeiras 5 linhas):**")
                    st.dataframe(df_true.head())
            
            # Abas para diferentes tipos de análise
            tab1, tab2, tab3 = st.tabs(["📈 Métricas de Distribuição", "🎯 Acurácia e Classificação", "🔍 Análise Detalhada"])
            
            with tab1:
                # Métricas de distribuição
                st.subheader("📈 Métricas de Distribuição")
                
                # Mostrar informações básicas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Amostras Previstas", len(df_pred))
                with col2:
                    st.metric("Amostras Verdadeiras", len(df_true))
                with col3:
                    st.metric("Arquivos Únicos", len(set(df_pred['file'])))
                
                # Calcular métricas de distribuição
                with st.spinner("Calculando métricas de distribuição..."):
                    metrics, df_pred_sorted, df_true_sorted = calculate_distribution_metrics(df_pred, df_true)
                
                # Métricas resumidas
                st.subheader("📊 Métricas Resumidas de Distribuição")
                cols = st.columns(6)
                metric_config = {
                    'KL Divergence': ('kl_div', '↓', 'red'),
                    'JS Divergence': ('js_div', '↓', 'orange'),
                    'Cross-Entropy': ('cross_entropy', '↓', 'blue'),
                    'MSE': ('mse', '↓', 'purple'),
                    'MAE': ('mae', '↓', 'green'),
                    'Cosine Similarity': ('cosine_similarity', '↑', 'lightgreen')
                }
                
                for i, (name, (key, direction, color)) in enumerate(metric_config.items()):
                    mean_val = np.mean(metrics[key])
                    with cols[i]:
                        st.metric(
                            label=name,
                            value=f"{mean_val:.4f}",
                            delta=direction
                        )
                
                # Gráficos de distribuição
                st.subheader("📊 Distribuição das Métricas")
                fig_metrics = create_metrics_plot(metrics)
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            with tab2:
                # Métricas de acurácia
                st.subheader("🎯 Métricas de Acurácia")
                
                with st.spinner("Calculando métricas de acurácia..."):
                    accuracy, cm, report, y_pred, y_true = calculate_accuracy_metrics(df_pred, df_true)
                
                # Métricas principais de acurácia
                st.subheader("📈 Métricas Principais")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Acurácia", f"{accuracy:.4f}", f"{(accuracy*100):.2f}%")
                
                with col2:
                    # Precisão média
                    precision_avg = report['macro avg']['precision']
                    st.metric("Precisão Média", f"{precision_avg:.4f}")
                
                with col3:
                    # Recall médio
                    recall_avg = report['macro avg']['recall']
                    st.metric("Recall Médio", f"{recall_avg:.4f}")
                
                with col4:
                    # F1-score médio
                    f1_avg = report['macro avg']['f1-score']
                    st.metric("F1-Score Médio", f"{f1_avg:.4f}")
                
                # Matriz de confusão
                st.subheader("📋 Matriz de Confusão")
                fig_cm = create_confusion_matrix_plot(cm)
                st.pyplot(fig_cm)
                
                # Relatório de classificação
                st.subheader("📊 Relatório de Classificação")
                
                # Converter para DataFrame para melhor visualização
                report_df = pd.DataFrame(report).transpose()
                # Remover a linha 'accuracy' do DataFrame principal
                accuracy_row = report_df.loc['accuracy']
                report_df = report_df.drop('accuracy')
                
                st.dataframe(report_df.style.format({
                    'precision': '{:.4f}',
                    'recall': '{:.4f}',
                    'f1-score': '{:.4f}',
                    'support': '{:.0f}'
                }), use_container_width=True)
                
                # Mostrar acurácia separadamente
                st.metric("Acurácia (Overall)", f"{accuracy_row['precision']:.4f}")
                
                # Distribuição das classes
                st.subheader("📈 Distribuição das Classes")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    true_counts = pd.Series(y_true).value_counts().sort_index()
                    true_counts.index = [EMOTION_MAP[i] for i in true_counts.index]
                    fig_true = go.Figure(go.Bar(x=true_counts.index, y=true_counts.values, marker_color='lightcoral'))
                    fig_true.update_layout(title="Distribuição Verdadeira", xaxis_title="Emoção", yaxis_title="Contagem")
                    st.plotly_chart(fig_true, use_container_width=True)
                
                with col2:
                    pred_counts = pd.Series(y_pred).value_counts().sort_index()
                    pred_counts.index = [EMOTION_MAP[i] for i in pred_counts.index]
                    fig_pred = go.Figure(go.Bar(x=pred_counts.index, y=pred_counts.values, marker_color='lightblue'))
                    fig_pred.update_layout(title="Distribuição Prevista", xaxis_title="Emoção", yaxis_title="Contagem")
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            with tab3:
                # Análise detalhada
                st.subheader("🔍 Análise Detalhada por Amostra")
                
                # Comparação individual
                sample_idx = st.slider(
                    "Selecione a amostra para visualizar:",
                    0, len(df_pred)-1, 0,
                    help="Navegue entre as amostras para comparar as distribuições"
                )
                
                fig_comparison = create_comparison_plot(df_pred, df_true, sample_idx)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Mostrar informações da amostra selecionada
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Informações da Amostra Selecionada:**")
                    st.write(f"**Arquivo:** {df_pred.iloc[sample_idx]['file']}")
                    
                    pred_emotion_idx = int(df_pred.iloc[sample_idx]['emotion']) if 'emotion' in df_pred.columns else np.argmax(df_pred.iloc[sample_idx][EMOTION_COLUMNS].values)
                    true_emotion_idx = int(df_true.iloc[sample_idx]['emotion']) if 'emotion' in df_true.columns else np.argmax(df_true.iloc[sample_idx][EMOTION_COLUMNS].values)
                    
                    st.write(f"**Emoção Prevista:** {EMOTION_MAP[pred_emotion_idx]} (índice {pred_emotion_idx})")
                    st.write(f"**Emoção Verdadeira:** {EMOTION_MAP[true_emotion_idx]} (índice {true_emotion_idx})")
                    st.write(f"**Correto:** {pred_emotion_idx == true_emotion_idx}")
                
                # Tabela com métricas detalhadas
                st.subheader("📋 Métricas Detalhadas por Amostra")
                
                # Calcular métricas de distribuição se ainda não calculadas
                if 'metrics' not in locals():
                    metrics, df_pred_sorted, df_true_sorted = calculate_distribution_metrics(df_pred, df_true)
                
                metrics_df = pd.DataFrame({
                    'file': df_pred_sorted['file'],
                    'predicted_emotion': [EMOTION_MAP[int(df_pred_sorted.iloc[i]['emotion'])] if 'emotion' in df_pred_sorted.columns else [EMOTION_MAP[np.argmax(df_pred_sorted.iloc[i][EMOTION_COLUMNS].values)] for i in range(len(df_pred_sorted))]],
                    'true_emotion': [EMOTION_MAP[int(df_true_sorted.iloc[i]['emotion'])] if 'emotion' in df_true_sorted.columns else [EMOTION_MAP[np.argmax(df_true_sorted.iloc[i][EMOTION_COLUMNS].values)] for i in range(len(df_true_sorted))]],
                    'correct': [int(df_pred_sorted.iloc[i]['emotion']) == int(df_true_sorted.iloc[i]['emotion']) if 'emotion' in df_pred_sorted.columns else np.argmax(df_pred_sorted.iloc[i][EMOTION_COLUMNS].values) == np.argmax(df_true_sorted.iloc[i][EMOTION_COLUMNS].values) for i in range(len(df_pred_sorted))],
                    'kl_div': metrics['kl_div'],
                    'js_div': metrics['js_div'],
                    'cross_entropy': metrics['cross_entropy'],
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'cosine_similarity': metrics['cosine_similarity']
                })
                
                st.dataframe(metrics_df.style.format({
                    'kl_div': '{:.6f}',
                    'js_div': '{:.6f}',
                    'cross_entropy': '{:.6f}',
                    'mse': '{:.6f}',
                    'mae': '{:.6f}',
                    'cosine_similarity': '{:.4f}'
                }), use_container_width=True)
                
                # Download das métricas
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download das Métricas Detalhadas",
                    data=csv,
                    file_name="metricas_detalhadas.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Erro ao processar os arquivos: {str(e)}")
            st.error("Verifique se a estrutura dos arquivos está correta.")
    
    else:
        st.info("👆 Faça upload de ambos os arquivos CSV para começar a análise")
        
        # Exemplo de estrutura
        st.subheader("📝 Estrutura Esperada dos Arquivos")
        example_data = {
            'neutral': [0.15, 0.13],
            'happy': [0.1, 0.2],
            'sad': [0.2, 0.15],
            'surprised': [0.25, 0.3],
            'fear': [0.04, 0.03],
            'disgust': [0.06, 0.04],
            'angry': [0.05, 0.05],
            'contempt': [0.15, 0.1],
            'emotion': [2, 0],  # Nova coluna
            'file': ['image1.jpg', 'image2.jpg']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.info("💡 **Nota:** A coluna `emotion` deve conter o índice da emoção (0-7) de acordo com o mapeamento:")
        
        # Tabela de mapeamento
        mapping_df = pd.DataFrame({
            'Índice': list(EMOTION_MAP.keys()),
            'Emoção': list(EMOTION_MAP.values())
        })
        st.dataframe(mapping_df, use_container_width=True)

if __name__ == "__main__":
    main()