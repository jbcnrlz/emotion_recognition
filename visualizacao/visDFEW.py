import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path

# Configuração da página
st.set_page_config(page_title="Análise de Emoções - Multi Fold", page_icon="😊", layout="wide")

# Título da aplicação
st.title("📊 Análise de Distribuição Emocional - Multi Fold")
st.markdown("---")

# Função para carregar dados
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Função para carregar múltiplos arquivos de uma pasta
def load_files_from_folder(uploaded_files, folder_type):
    files_data = {}
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            # Extrair nome do fold do nome do arquivo
            filename = uploaded_file.name
            fold_name = filename.replace('.csv', '').replace('_predictions', '').replace('_labels', '')
            files_data[fold_name] = df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {uploaded_file.name}: {e}")
    return files_data

# Função para extrair a emoção predominante
def get_predominant_emotion(row, emotion_columns):
    try:
        # Converter para numpy array e garantir que são números
        emotions = row[emotion_columns].values.astype(float)
        max_idx = np.argmax(emotions)
        return emotion_columns[max_idx]
    except Exception as e:
        return None

# Função para processar e validar dados
def preprocess_data(df, suffix):
    # Fazer uma cópia para não modificar o original
    df_processed = df.copy()
    
    # Identificar colunas de emoção (todas exceto 'file')
    emotion_cols = [col for col in df_processed.columns if col != 'file']
    
    # Converter colunas de emoção para float
    for col in emotion_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Renomear colunas com sufixo
    rename_dict = {col: f"{col}_{suffix}" for col in emotion_cols}
    df_processed = df_processed.rename(columns=rename_dict)
    
    return df_processed, [f"{col}_{suffix}" for col in emotion_cols]

# Função para calcular métricas para um fold
def calculate_metrics_single_fold(df_pred, df_true, fold_name):
    # Processar dados
    df_pred_processed, pred_cols = preprocess_data(df_pred, 'pred')
    df_true_processed, true_cols = preprocess_data(df_true, 'true')
    
    # Garantir que os arquivos estejam na mesma ordem
    df_merged = pd.merge(df_pred_processed, df_true_processed, on='file', how='inner')
    
    if len(df_merged) == 0:
        return None
    
    # Obter nomes base das emoções (sem sufixos)
    base_emotions = [col.replace('_pred', '').replace('_true', '') for col in pred_cols]
    
    # Obter emoções predominantes
    predominant_pred = []
    predominant_true = []
    valid_indices = []
    
    for idx, row in df_merged.iterrows():
        pred_emotion = get_predominant_emotion(row, pred_cols)
        true_emotion = get_predominant_emotion(row, true_cols)
        
        if pred_emotion and true_emotion:
            # Remover sufixos para comparação
            pred_emotion_clean = pred_emotion.replace('_pred', '')
            true_emotion_clean = true_emotion.replace('_true', '')
            
            predominant_pred.append(pred_emotion_clean)
            predominant_true.append(true_emotion_clean)
            valid_indices.append(idx)
    
    # Filtrar apenas as linhas válidas
    df_merged_valid = df_merged.loc[valid_indices].reset_index(drop=True)
    df_merged_valid['predominant_pred'] = predominant_pred
    df_merged_valid['predominant_true'] = predominant_true
    
    if len(predominant_pred) == 0:
        return None
    
    # Calcular métricas
    accuracy = accuracy_score(predominant_true, predominant_pred)
    
    class_report = classification_report(
        predominant_true, 
        predominant_pred,
        target_names=base_emotions,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(
        predominant_true, 
        predominant_pred,
        labels=base_emotions
    )
    
    # Calcular diferenças nas distribuições
    distribution_differences = {}
    for emotion in base_emotions:
        pred_col = f"{emotion}_pred"
        true_col = f"{emotion}_true"
        
        if pred_col in df_merged_valid.columns and true_col in df_merged_valid.columns:
            mae = np.mean(np.abs(df_merged_valid[pred_col] - df_merged_valid[true_col]))
            mse = np.mean((df_merged_valid[pred_col] - df_merged_valid[true_col]) ** 2)
            distribution_differences[emotion] = {'MAE': mae, 'MSE': mse}
    
    return {
        'fold_name': fold_name,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'distribution_differences': distribution_differences,
        'merged_data': df_merged_valid,
        'emotion_names': base_emotions,
        'total_samples': len(predominant_pred),
        'predominant_true': predominant_true,
        'predominant_pred': predominant_pred
    }

# Função para calcular métricas agregadas de todos os folds
def calculate_aggregate_metrics(all_results):
    if not all_results:
        return None
    
    # Combinar todos os dados
    all_true = []
    all_pred = []
    all_merged_data = []
    accuracies = []
    
    for result in all_results.values():
        if result is not None:
            all_true.extend(result['predominant_true'])
            all_pred.extend(result['predominant_pred'])
            all_merged_data.append(result['merged_data'])
            accuracies.append(result['accuracy'])
    
    if len(all_true) == 0:
        return None
    
    # Usar emotion_names do primeiro fold válido
    emotion_names = next(iter(all_results.values()))['emotion_names']
    
    # Métricas agregadas
    aggregate_accuracy = accuracy_score(all_true, all_pred)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    aggregate_class_report = classification_report(
        all_true, 
        all_pred,
        target_names=emotion_names,
        output_dict=True,
        zero_division=0
    )
    
    aggregate_cm = confusion_matrix(
        all_true, 
        all_pred,
        labels=emotion_names
    )
    
    # Combinar todos os dados para calcular diferenças de distribuição
    all_merged_combined = pd.concat(all_merged_data, ignore_index=True)
    
    distribution_differences = {}
    for emotion in emotion_names:
        pred_col = f"{emotion}_pred"
        true_col = f"{emotion}_true"
        
        if pred_col in all_merged_combined.columns and true_col in all_merged_combined.columns:
            mae = np.mean(np.abs(all_merged_combined[pred_col] - all_merged_combined[true_col]))
            mse = np.mean((all_merged_combined[pred_col] - all_merged_combined[true_col]) ** 2)
            distribution_differences[emotion] = {'MAE': mae, 'MSE': mse}
    
    return {
        'accuracy': aggregate_accuracy,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'classification_report': aggregate_class_report,
        'confusion_matrix': aggregate_cm,
        'distribution_differences': distribution_differences,
        'emotion_names': emotion_names,
        'total_samples': len(all_true),
        'fold_accuracies': accuracies,
        'all_merged_data': all_merged_combined
    }

# Interface principal
st.header("📁 Upload de Múltiplos Folds")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Arquivos com Previsões")
    pred_files = st.file_uploader(
        "Faça upload dos arquivos CSV com as previsões (múltiplos folds)", 
        type=['csv'], 
        key="pred_multiple",
        accept_multiple_files=True
    )

with col2:
    st.subheader("Arquivos com Rótulos Verdadeiros")
    true_files = st.file_uploader(
        "Faça upload dos arquivos CSV com os rótulos verdadeiros (múltiplos folds)", 
        type=['csv'], 
        key="true_multiple",
        accept_multiple_files=True
    )

if pred_files and true_files:
    # Carregar dados
    preds_data = load_files_from_folder(pred_files, "predictions")
    trues_data = load_files_from_folder(true_files, "labels")
    
    if preds_data and trues_data:
        # Mostrar arquivos carregados
        st.subheader("📋 Arquivos Carregados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Previsões carregadas:**")
            for fold_name in preds_data.keys():
                st.write(f"- {fold_name}: {len(preds_data[fold_name])} amostras")
        
        with col2:
            st.write("**Rótulos carregados:**")
            for fold_name in trues_data.keys():
                st.write(f"- {fold_name}: {len(trues_data[fold_name])} amostras")
        
        # Encontrar folds comuns
        common_folds = set(preds_data.keys()) & set(trues_data.keys())
        
        if not common_folds:
            st.error("Não foram encontrados folds correspondentes entre previsões e rótulos.")
        else:
            st.success(f"Encontrados {len(common_folds)} folds correspondentes: {', '.join(sorted(common_folds))}")
            
            # Calcular métricas para cada fold
            with st.spinner("Calculando métricas para todos os folds..."):
                all_results = {}
                for fold_name in sorted(common_folds):
                    result = calculate_metrics_single_fold(
                        preds_data[fold_name], 
                        trues_data[fold_name], 
                        fold_name
                    )
                    if result is not None:
                        all_results[fold_name] = result
                
                # Calcular métricas agregadas
                aggregate_results = calculate_aggregate_metrics(all_results)
            
            if not all_results:
                st.error("Não foi possível calcular métricas para nenhum fold.")
            else:
                st.markdown("---")
                st.header("📈 Resultados Agregados")
                
                # Métricas principais agregadas
                if aggregate_results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Acurácia Total", f"{aggregate_results['accuracy']:.4f}")
                    
                    with col2:
                        st.metric("Acurácia Média", f"{aggregate_results['mean_accuracy']:.4f}")
                    
                    with col3:
                        st.metric("Desvio Padrão", f"{aggregate_results['std_accuracy']:.4f}")
                    
                    with col4:
                        st.metric("Total de Amostras", aggregate_results['total_samples'])
                
                # Tabs para organização
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Visão Geral dos Folds", 
                    "🎯 Métricas Agregadas", 
                    "📈 Análise por Fold",
                    "🔍 Análise Detalhada"
                ])
                
                with tab1:
                    st.subheader("Desempenho por Fold")
                    
                    # Tabela de acurácias por fold
                    fold_metrics = []
                    for fold_name, result in all_results.items():
                        fold_metrics.append({
                            'Fold': fold_name,
                            'Acurácia': f"{result['accuracy']:.4f}",
                            'Amostras': result['total_samples'],
                            'Previsões Corretas': int(result['accuracy'] * result['total_samples'])
                        })
                    
                    fold_metrics_df = pd.DataFrame(fold_metrics)
                    st.dataframe(fold_metrics_df, use_container_width=True)
                    
                    # Gráfico de acurácias por fold
                    fig, ax = plt.subplots(figsize=(10, 6))
                    folds = [fm['Fold'] for fm in fold_metrics]
                    accuracies = [float(fm['Acurácia']) for fm in fold_metrics]
                    
                    bars = ax.bar(folds, accuracies, color='skyblue', alpha=0.7)
                    ax.axhline(y=aggregate_results['mean_accuracy'] if aggregate_results else np.mean(accuracies), 
                              color='red', linestyle='--', label='Média')
                    ax.set_ylabel('Acurácia')
                    ax.set_xlabel('Folds')
                    ax.set_title('Acurácia por Fold')
                    ax.set_ylim(0, 1)
                    ax.legend()
                    
                    # Adicionar valores nas barras
                    for bar, acc in zip(bars, accuracies):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{acc:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                with tab2:
                    if aggregate_results:
                        st.subheader("Matriz de Confusão Agregada")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(aggregate_results['confusion_matrix'], 
                                   annot=True, fmt='d', 
                                   xticklabels=aggregate_results['emotion_names'],
                                   yticklabels=aggregate_results['emotion_names'],
                                   cmap='Blues', ax=ax)
                        ax.set_xlabel('Previsões')
                        ax.set_ylabel('Valores Verdadeiros')
                        ax.set_title('Matriz de Confusão - Todos os Folds')
                        st.pyplot(fig)
                        
                        st.subheader("Relatório de Classificação Agregado")
                        class_report_df = pd.DataFrame(aggregate_results['classification_report']).transpose()
                        styled_report = class_report_df.style.format({
                            'precision': '{:.4f}',
                            'recall': '{:.4f}',
                            'f1-score': '{:.4f}',
                            'support': '{:.0f}'
                        }).background_gradient(cmap='YlOrBr', subset=['precision', 'recall', 'f1-score'])
                        st.dataframe(styled_report, use_container_width=True)
                
                with tab3:
                    st.subheader("Análise Individual por Fold")
                    
                    selected_fold = st.selectbox(
                        "Selecione um fold para análise detalhada:",
                        list(all_results.keys())
                    )
                    
                    if selected_fold:
                        result = all_results[selected_fold]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Acurácia - {selected_fold}", f"{result['accuracy']:.4f}")
                        with col2:
                            st.metric("Amostras", result['total_samples'])
                        with col3:
                            correct = int(result['accuracy'] * result['total_samples'])
                            st.metric("Previsões Corretas", correct)
                        
                        # Matriz de confusão do fold selecionado
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(result['confusion_matrix'], 
                                   annot=True, fmt='d', 
                                   xticklabels=result['emotion_names'],
                                   yticklabels=result['emotion_names'],
                                   cmap='Blues', ax=ax)
                        ax.set_xlabel('Previsões')
                        ax.set_ylabel('Valores Verdadeiros')
                        ax.set_title(f'Matriz de Confusão - {selected_fold}')
                        st.pyplot(fig)
                
                with tab4:
                    st.subheader("Análise Detalhada por Emoção")
                    
                    if aggregate_results:
                        selected_emotion = st.selectbox(
                            "Selecione uma emoção para análise detalhada:",
                            aggregate_results['emotion_names']
                        )
                        
                        if selected_emotion:
                            # Calcular métricas específicas da emoção
                            emotion_metrics = []
                            for fold_name, result in all_results.items():
                                if selected_emotion in result['emotion_names']:
                                    # Encontrar índice da emoção
                                    emotion_idx = result['emotion_names'].index(selected_emotion)
                                    # TP, FP, FN para esta emoção
                                    tp = result['confusion_matrix'][emotion_idx, emotion_idx]
                                    fp = result['confusion_matrix'][:, emotion_idx].sum() - tp
                                    fn = result['confusion_matrix'][emotion_idx, :].sum() - tp
                                    
                                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                    
                                    emotion_metrics.append({
                                        'Fold': fold_name,
                                        'Precision': f"{precision:.4f}",
                                        'Recall': f"{recall:.4f}",
                                        'F1-Score': f"{f1:.4f}",
                                        'TP': tp,
                                        'FP': fp,
                                        'FN': fn
                                    })
                            
                            if emotion_metrics:
                                emotion_df = pd.DataFrame(emotion_metrics)
                                st.dataframe(emotion_df, use_container_width=True)
    
    else:
        st.error("Erro ao processar os arquivos. Verifique os formatos.")

else:
    st.info("👆 Faça upload dos arquivos de múltiplos folds para começar a análise.")
    
    st.markdown("---")
    st.subheader("ℹ️ Como usar com múltiplos folds")
    
    st.write("""
    **Estrutura esperada:**
    
    - Upload de múltiplos arquivos CSV para previsões e rótulos
    - Os arquivos devem ser pareados por nome (ex: fold1_predictions.csv + fold1_labels.csv)
    - Formato interno igual ao anterior: colunas de emoções + coluna 'file'
    
    **Exemplo de estrutura de arquivos:**
    ```
    fold1_predictions.csv
    fold1_labels.csv
    fold2_predictions.csv
    fold2_labels.csv
    ...
    fold5_predictions.csv
    fold5_labels.csv
    ```
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "**Desenvolvido para análise de distribuições emocionais com múltiplos folds** • "
    "Use arquivos CSV com a mesma estrutura para comparação"
)