import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuração da página
st.set_page_config(page_title="Validador de Modelos ML", layout="wide")

st.title("📊 Analisador de Performance de Classificação")
st.markdown("Carrega o teu ficheiro CSV com os **logits** e as **labels** para validar o modelo.")

# --- COMPONENTE DE UPLOAD ---
uploaded_file = st.file_uploader("Escolhe o ficheiro CSV", type="csv")

if uploaded_file is not None:
    # Ler o ficheiro selecionado
    df = pd.read_csv(uploaded_file)
    
    st.success("Ficheiro carregado com sucesso!")
    
    # Interface para mapeamento de colunas
    all_columns = df.columns.tolist()
    
    col_selector_1, col_selector_2 = st.columns(2)
    
    with col_selector_1:
        # Tenta pré-selecionar a coluna 'label' se existir
        label_col = st.selectbox("Seleciona a coluna de Rótulos Reais (Label):", 
                                 all_columns, 
                                 index=all_columns.index('label') if 'label' in all_columns else 0)
    
    with col_selector_2:
        # Selecionar quais colunas são os logits (por padrão as numéricas de 0 a 7)
        default_logits = [c for c in all_columns if c.isdigit() or (isinstance(c, str) and c.replace('.','',1).isdigit())]
        logit_cols = st.multiselect("Seleciona as colunas de Logits (Classes):", 
                                    all_columns, 
                                    default=default_logits)

    if label_col and logit_cols:
        # Processamento: Obter a classe predita via Argmax
        # Convertemos para numérico para garantir que o idxmax funcione corretamente
        df_logits = df[logit_cols].apply(pd.to_numeric)
        df['prediction'] = df_logits.idxmax(axis=1).astype(int)
        
        # --- EXIBIÇÃO DE MÉTRICAS ---
        st.divider()
        
        # Métricas principais em destaque
        acc = accuracy_score(df[label_col], df['prediction'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Acurácia (Accuracy)", f"{acc:.2%}")
        m2.metric("Total de Amostras", len(df))
        m3.metric("Nº de Classes", len(logit_cols))

        # --- GRÁFICOS ---
        tab1, tab2, tab3 = st.tabs(["Matriz de Confusão", "Relatório Detalhado", "Visualização de Dados"])

        with tab1:
            st.subheader("Matriz de Confusão")
            cm = confusion_matrix(df[label_col], df['prediction'])
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                        xticklabels=logit_cols, yticklabels=logit_cols)
            plt.xlabel('Predito (Modelo)')
            plt.ylabel('Real (Ground Truth)')
            st.pyplot(fig)

        with tab2:
            st.subheader("Classification Report")
            report = classification_report(df[label_col], df['prediction'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='YlGn'), use_container_width=True)
            
            st.info("💡 **Precision**: De todos os preditos como X, quantos eram realmente X. \n\n"
                    "💡 **Recall**: De todos que eram realmente X, quantos o modelo apanhou.")

        with tab3:
            st.subheader("Exploração de Dados")
            st.dataframe(df, use_container_width=True)

    else:
        st.warning("Por favor, seleciona as colunas de Logits e Label para prosseguir.")

else:
    st.info("Aguardando upload de ficheiro CSV para iniciar a análise.")