import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração básica - sem dependências problemáticas
st.set_page_config(
    page_title="Análise de Emoções",
    page_icon="😊",
    layout="wide"
)

def main():
    st.title("😊 Análise de Emoções Faciais")
    st.markdown("Faça upload do arquivo CSV para análise das emoções")
    
    uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df = pd.read_csv(uploaded_file)
            
            # Verificar colunas básicas
            if 'file' not in df.columns:
                st.error("Arquivo deve conter coluna 'file'")
                return
            
            st.success(f"✅ Arquivo carregado com {len(df)} linhas e {len(df.columns)} colunas")
            
            # Abas de análise
            tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "📈 Emoções", "🎯 Relatório"])
            
            with tab1:
                st.subheader("Visão Geral do Dataset")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Imagens", len(df))
                
                with col2:
                    # Tentar extrair pessoas do arquivo
                    try:
                        pessoas = df['file'].str.extract(r'pessoa_(\d+)')[0].nunique()
                        st.metric("Pessoas Únicas", pessoas)
                    except:
                        st.metric("Pessoas Únicas", "N/A")
                
                with col3:
                    emocao_cols = [col for col in df.columns if col != 'file']
                    st.metric("Emoções Analisadas", len(emocao_cols))
            
            with tab2:
                st.subheader("Distribuição das Emoções")
                
                # Calcular médias
                emocao_cols = [col for col in df.columns if col != 'file']
                medias = df[emocao_cols].mean().sort_values(ascending=True)
                
                # Gráfico de barras
                fig = px.bar(
                    x=medias.values,
                    y=medias.index,
                    orientation='h',
                    title="Probabilidade Média das Emoções",
                    labels={'x': 'Probabilidade', 'y': 'Emoção'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela de dados
                st.subheader("Dados Detalhados")
                st.dataframe(df.head(10))
            
            with tab3:
                st.subheader("Relatório de Análise")
                
                # Análise simples
                emocao_dominante = df[emocao_cols].mean().idxmax()
                valor_dominante = df[emocao_cols].mean().max()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Emoção Mais Presente:**")
                    st.info(f"**{emocao_dominante}**: {valor_dominante:.3f} ({valor_dominante*100:.1f}%)")
                    
                    st.write("**Top 5 Emoções:**")
                    top5 = df[emocao_cols].mean().sort_values(ascending=False).head(5)
                    for i, (emocao, valor) in enumerate(top5.items(), 1):
                        st.write(f"{i}. **{emocao}**: {valor:.3f}")
                
                with col2:
                    # Análise hedônica simples
                    positivas = ['happy', 'hopeful', 'surprised', 'proud', 'loved', 'elated']
                    negativas = ['angry', 'sad', 'fearful', 'disgusted', 'fatigued']
                    
                    emo_positivas = [e for e in positivas if e in emocao_cols]
                    emo_negativas = [e for e in negativas if e in emocao_cols]
                    
                    if emo_positivas and emo_negativas:
                        score_positivo = df[emo_positivas].sum(axis=1).mean()
                        score_negativo = df[emo_negativas].sum(axis=1).mean()
                        
                        st.write("**Tendência Hedônica:**")
                        if score_positivo > score_negativo:
                            st.success("✅ **Positiva** - Boa aceitação")
                        else:
                            st.warning("⚠️ **Neutra/Negativa** - Avaliar produto")
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
    
    else:
        # Instruções
        st.info("""
        ### 📋 Como usar:
        1. **Faça upload** de um arquivo CSV
        2. **Formato esperado:**
           - Coluna 'file' com caminhos das imagens
           - Colunas para cada emoção (happy, sad, angry, etc.)
           - Valores entre 0-1 representando probabilidades
        
        ### 📊 Exemplo de formato:
        ```
        happy,angry,sad,neutral,file
        0.8,0.1,0.05,0.05,caminho/para/imagem.jpg
        ```
        """)

if __name__ == "__main__":
    main()