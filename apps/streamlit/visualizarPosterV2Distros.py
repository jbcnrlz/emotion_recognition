import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import paired_cosine_distances

st.set_page_config(page_title="Model vs Ground Truth Comparison", layout="wide")

st.title("📊 Análise de Distribuição de Emoções")
st.markdown("Faça o upload dos arquivos CSV contendo as predições do modelo e o *ground truth* para gerar as métricas e visualizações para o seu paper.")

# Seções de upload
col1, col2 = st.columns(2)
with col1:
    pred_file = st.file_uploader("Upload: Predição do Modelo (.csv)", type=['csv'])
with col2:
    gt_file = st.file_uploader("Upload: Ground Truth (.csv)", type=['csv'])

if pred_file and gt_file:
    # Carregar os dados
    df_pred = pd.read_csv(pred_file)
    df_gt = pd.read_csv(gt_file)
    
    st.success("Arquivos carregados com sucesso!")

    # Definir as colunas de emoções alvo (para padronizar a ordem)
    emotions = ['neutral', 'happy', 'sad', 'surprised', 'fearful', 'disgusted', 'angry', 'contempt']
    
    # Verificar se ambos os arquivos contêm a coluna 'file'
    if 'file' not in df_pred.columns or 'file' not in df_gt.columns:
        st.error("Ambos os arquivos devem conter uma coluna chamada 'file' para o alinhamento.")
    else:
        # Fazer o merge (inner join) com base na coluna 'file'
        df_merged = pd.merge(df_pred, df_gt, on='file', suffixes=('_pred', '_gt'))
        st.write(f"**Total de imagens pareadas encontradas:** {len(df_merged)}")

        if len(df_merged) > 0:
            # Extrair as distribuições alinhadas
            pred_dists = df_merged[[f"{emo}_pred" for emo in emotions]].values
            gt_dists = df_merged[[f"{emo}_gt" for emo in emotions]].values
            
            # Extrair o rótulo original do dataset (usaremos a coluna do GT para referência)
            true_labels_discrete = df_merged['emotion_label_gt'].values.astype(int)
            
            # Normalizar distribuições para garantir que somem 1 (necessário para KL e JS)
            epsilon = 1e-9
            pred_dists = np.clip(pred_dists, epsilon, 1)
            pred_dists = pred_dists / pred_dists.sum(axis=1, keepdims=True)
            
            gt_dists = np.clip(gt_dists, epsilon, 1)
            gt_dists = gt_dists / gt_dists.sum(axis=1, keepdims=True)

            # 1. Calcular Acurácias (Argmax)
            pred_labels = np.argmax(pred_dists, axis=1)
            gt_labels = np.argmax(gt_dists, axis=1)
            
            # Desempenho do Modelo
            acc_dist = accuracy_score(gt_labels, pred_labels)
            acc_label = accuracy_score(true_labels_discrete, pred_labels)
            
            # Concordância Interna do Dataset (Rótulo Original vs Distribuição GT)
            acc_dataset_internal = accuracy_score(true_labels_discrete, gt_labels)
            
            # 2. Métricas de Distribuição (Predição vs GT)
            kl_div = np.array([entropy(gt_dists[i], pred_dists[i]) for i in range(len(df_merged))])
            mean_kl = np.mean(kl_div)
            
            js_dist = np.array([jensenshannon(gt_dists[i], pred_dists[i]) for i in range(len(df_merged))])
            mean_js = np.mean(js_dist)
            
            cos_sim = 1 - paired_cosine_distances(gt_dists, pred_dists)
            mean_cos_sim = np.mean(cos_sim)

            # Exibir Métricas em destaque
            st.markdown("### 📈 Métricas de Avaliação Global")
            metric_cols = st.columns(6)
            metric_cols[0].metric("Acc (Pred vs Rótulo)", f"{acc_label*100:.2f}%", help="O modelo acertando o 'emotion_label'")
            metric_cols[1].metric("Acc (Pred vs Dist GT)", f"{acc_dist*100:.2f}%", help="O modelo acertando o Top-1 da Distribuição GT")
            metric_cols[2].metric("Concordância (Rótulo vs Dist)", f"{acc_dataset_internal*100:.2f}%", help="O quanto o rótulo original e o Top-1 da distribuição GT concordam entre si")
            metric_cols[3].metric("Divergência KL", f"{mean_kl:.4f}")
            metric_cols[4].metric("Distância JS", f"{mean_js:.4f}")
            metric_cols[5].metric("Similaridade Cos", f"{mean_cos_sim:.4f}")

            # Visualizações
            st.markdown("### 📊 Visualizações para o Paper")
            
            tab1, tab2, tab3 = st.tabs(["Matrizes de Confusão", "Distribuição das Métricas", "Análise de Amostras"])
            
            with tab1:
                st.markdown("Três visões da matriz de confusão: Desempenho do modelo contra os dois alvos e a discrepância interna do próprio dataset.")
                col_cm1, col_cm2, col_cm3 = st.columns(3)
                
                with col_cm1:
                    fig_lbl, ax_lbl = plt.subplots(figsize=(6, 5))
                    cm_lbl = confusion_matrix(true_labels_discrete, pred_labels, normalize='true')
                    sns.heatmap(cm_lbl, annot=True, fmt=".2f", cmap="Greens", xticklabels=emotions, yticklabels=emotions, ax=ax_lbl, cbar=False)
                    ax_lbl.set_xlabel("Predição do Modelo")
                    ax_lbl.set_ylabel("Rótulo Original")
                    ax_lbl.set_title("1. Pred vs Rótulo Original")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_lbl)
                    
                with col_cm2:
                    fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
                    cm_dist = confusion_matrix(gt_labels, pred_labels, normalize='true')
                    sns.heatmap(cm_dist, annot=True, fmt=".2f", cmap="Blues", xticklabels=emotions, yticklabels=emotions, ax=ax_dist, cbar=False)
                    ax_dist.set_xlabel("Predição do Modelo")
                    ax_dist.set_ylabel("Distribuição GT (Top-1)")
                    ax_dist.set_title("2. Pred vs Distribuição GT")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_dist)
                    
                with col_cm3:
                    fig_gt, ax_gt = plt.subplots(figsize=(6, 5))
                    cm_gt = confusion_matrix(true_labels_discrete, gt_labels, normalize='true')
                    sns.heatmap(cm_gt, annot=True, fmt=".2f", cmap="Purples", xticklabels=emotions, yticklabels=emotions, ax=ax_gt, cbar=False)
                    ax_gt.set_xlabel("Distribuição GT (Top-1)")
                    ax_gt.set_ylabel("Rótulo Original")
                    ax_gt.set_title("3. Discrepância Interna do GT")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_gt)
                
            with tab2:
                st.markdown("Histogramas mostrando como as divergências e similaridades estão distribuídas por todo o dataset.")
                fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                sns.histplot(kl_div, bins=30, ax=axes[0], color='salmon', kde=True)
                axes[0].set_title('Divergência KL (Pred vs Dist)')
                axes[0].set_xlabel('Valor KL')
                
                sns.histplot(js_dist, bins=30, ax=axes[1], color='skyblue', kde=True)
                axes[1].set_title('Distância Jensen-Shannon')
                axes[1].set_xlabel('Valor JS')
                
                sns.histplot(cos_sim, bins=30, ax=axes[2], color='lightgreen', kde=True)
                axes[2].set_title('Similaridade de Cosseno')
                axes[2].set_xlabel('Similaridade')
                
                st.pyplot(fig2)
                
            with tab3:
                st.markdown("Inspecione individualmente a imagem da face junto com as distribuições previstas e alvo.")
                sample_idx = st.slider("Selecione o índice da imagem", 0, len(df_merged)-1, 0)
                file_name = df_merged.iloc[sample_idx]['file']
                st.write(f"**Arquivo:** `{file_name}`")
                
                col_img, col_chart = st.columns([1, 2])
                
                with col_img:
                    try:
                        if os.path.exists(file_name):
                            img = Image.open(file_name)
                            gt_emo_name = emotions[gt_labels[sample_idx]].capitalize()
                            pred_emo_name = emotions[pred_labels[sample_idx]].capitalize()
                            true_label_name = emotions[true_labels_discrete[sample_idx]].capitalize()
                            
                            st.image(img, use_container_width=True)
                            st.markdown(f"**1. Rótulo Original:** {true_label_name}")
                            st.markdown(f"**2. Distribuição GT (Top-1):** {gt_emo_name}")
                            st.markdown(f"**3. Predição Modelo:** {pred_emo_name}")
                            
                            if true_label_name != gt_emo_name:
                                st.warning(f"⚠️ **Discrepância no GT:** A distribuição gerada não tem a mesma emoção majoritária que o rótulo anotado originalmente.")
                        else:
                            st.warning(f"Imagem não encontrada no caminho local: {file_name}")
                    except Exception as e:
                        st.error(f"Erro ao carregar a imagem: {e}")

                with col_chart:
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    x = np.arange(len(emotions))
                    width = 0.35
                    
                    ax3.bar(x - width/2, gt_dists[sample_idx], width, label='Ground Truth Dist', color='gray')
                    ax3.bar(x + width/2, pred_dists[sample_idx], width, label='Predição Modelo', color='royalblue')
                    
                    # Destacar visualmente o rótulo original com uma linha vertical tracejada vermelha
                    label_idx = true_labels_discrete[sample_idx]
                    ax3.axvline(x=label_idx, color='red', linestyle='--', alpha=0.7, label='Rótulo Original', linewidth=2)
                    
                    ax3.set_ylabel('Probabilidade')
                    ax3.set_title('Comparação de Distribuições e Rótulo Original')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(emotions, rotation=45, ha='right')
                    ax3.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                st.write(f"**Métricas para esta amostra:** KL Divergence: `{kl_div[sample_idx]:.4f}` | JS Distance: `{js_dist[sample_idx]:.4f}` | Cosine Sim: `{cos_sim[sample_idx]:.4f}`")