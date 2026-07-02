import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Análise de Classificação de Emoções", layout="wide")

st.title("Estatísticas de Classificação de Emoções")
st.write("Faça o upload do arquivo CSV contendo os logits e os rótulos reais (emotion_label).")

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha o arquivo CSV (ex: logitsOriginal_small.csv)", type=["csv"])

if uploaded_file is not None:
    # Carregar os dados
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Amostra dos Dados Originais")
    st.dataframe(df.head())

    # Dicionário de mapeamento (0 a 7) para os nomes exatos das colunas no CSV
    label_mapping = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'surprised',
        4: 'fearful',
        5: 'disgusted',
        6: 'angry',
        7: 'contempt'
    }
    
    # Lista de colunas de emoção para verificar a probabilidade/logit
    emotion_cols = list(label_mapping.values())
    
    # 1. Obter a classe predita (nome da coluna com o maior valor entre as emoções)
    df['predicted_class'] = df[emotion_cols].idxmax(axis=1)
    
    # 2. Obter a classe real mapeando o 'emotion_label' para string
    df['true_class'] = df['emotion_label'].map(label_mapping)

    # Exibir a comparação
    st.subheader("Comparação: Predição vs Realidade")
    st.dataframe(df[['file', 'emotion_label', 'true_class', 'predicted_class'] + emotion_cols].head())

    # --- Estatísticas e Métricas ---
    st.divider()
    st.header("Métricas de Avaliação")
    
    # Remover possíveis linhas com labels não mapeados (NaNs) por segurança
    df_eval = df.dropna(subset=['true_class', 'predicted_class'])
    
    y_true = df_eval['true_class']
    y_pred = df_eval['predicted_class']
    
    # Acurácia
    acc = accuracy_score(y_true, y_pred)
    st.metric(label="Acurácia Global", value=f"{acc:.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relatório de Classificação")
        # Gerar o relatório de classificação como dicionário para formatar no pandas
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report_dict).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))

    with col2:
        st.subheader("Matriz de Confusão")
        
        # Obter as classes únicas que realmente aparecem no dataset para a matriz
        labels_presentes = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_presentes)
        
        # Plotar a matriz de confusão com Seaborn
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_presentes, yticklabels=labels_presentes, ax=ax)
        ax.set_xlabel('Classe Predita')
        ax.set_ylabel('Classe Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        st.pyplot(fig)