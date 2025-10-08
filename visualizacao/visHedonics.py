import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise de Emoções Faciais",
    page_icon="😊",
    layout="wide"
)

def main():
    st.title("😊 Análise de Emoções Faciais")
    st.markdown("Faça upload do arquivo CSV para análise das expressões emocionais")
    
    uploaded_file = st.file_uploader("📁 Escolha o arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} imagens analisadas.")
            
            # Verificar coluna obrigatória
            if 'file' not in df.columns:
                st.error("❌ O arquivo deve conter a coluna 'file' com os caminhos das imagens.")
                return
            
            # Identificar colunas de emoção
            emocao_cols = [col for col in df.columns if col != 'file']
            
            if not emocao_cols:
                st.error("❌ Nenhuma coluna de emoção encontrada no arquivo.")
                return
            
            # Abas de análise
            tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "📈 Análise Detalhada", "👤 Por Pessoa"])
            
            with tab1:
                st.subheader("Métricas Gerais")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Imagens", len(df))
                
                with col2:
                    # Extrair pessoas únicas
                    try:
                        pessoas = df['file'].str.extract(r'pessoa_(\d+)')[0].nunique()
                        st.metric("Pessoas Únicas", pessoas)
                    except:
                        st.metric("Pessoas Únicas", "N/A")
                
                with col3:
                    emocao_dominante = df[emocao_cols].mean().idxmax()
                    st.metric("Emoção Dominante", emocao_dominante)
                
                with col4:
                    valor_dominante = df[emocao_cols].mean().max()
                    st.metric("Probabilidade", f"{valor_dominante:.3f}")
                
                # Gráfico de distribuição
                st.subheader("Distribuição das Emoções")
                medias = df[emocao_cols].mean().sort_values(ascending=True)
                
                fig = px.bar(
                    x=medias.values,
                    y=medias.index,
                    orientation='h',
                    title="Probabilidade Média de Cada Emoção",
                    labels={'x': 'Probabilidade Média', 'y': 'Emoção'},
                    color=medias.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Análise Detalhada")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 5 Emoções Mais Frequentes:**")
                    top5 = df[emocao_cols].mean().sort_values(ascending=False).head()
                    for i, (emocao, valor) in enumerate(top5.items(), 1):
                        st.write(f"{i}. **{emocao}**: {valor:.3f} ({valor*100:.1f}%)")
                
                with col2:
                    # Análise hedônica
                    st.write("**Análise Hedônica:**")
                    
                    emo_positivas = ['happy', 'elated', 'hopeful', 'surprised', 'proud', 'loved']
                    emo_negativas = ['angry', 'disgusted', 'fearful', 'sad', 'fatigued', 'contempt']
                    
                    positivas_presentes = [e for e in emo_positivas if e in emocao_cols]
                    negativas_presentes = [e for e in emo_negativas if e in emocao_cols]
                    
                    if positivas_presentes and negativas_presentes:
                        score_pos = df[positivas_presentes].sum(axis=1).mean()
                        score_neg = df[negativas_presentes].sum(axis=1).mean()
                        
                        if score_pos > score_neg:
                            st.success("😊 **Experiência Positiva**")
                            st.write(f"Score positivo: {score_pos:.3f}")
                        else:
                            st.warning("😐 **Experiência Neutra/Negativa**")
                            st.write(f"Score negativo: {score_neg:.3f}")
                
                # Heatmap de correlação
                st.subheader("Correlação entre Emoções")
                if len(emocao_cols) > 1:
                    correlacao = df[emocao_cols].corr()
                    fig = px.imshow(
                        correlacao,
                        title="Matriz de Correlação entre Emoções",
                        aspect="auto",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Análise por Pessoa")
                
                # Extrair pessoas
                try:
                    df_temp = df.copy()
                    df_temp['pessoa'] = df_temp['file'].str.extract(r'pessoa_(\d+)')
                    
                    if 'pessoa' in df_temp.columns and not df_temp['pessoa'].isna().all():
                        pessoas_unicas = sorted(df_temp['pessoa'].unique())
                        pessoa_selecionada = st.selectbox("Selecione uma pessoa:", pessoas_unicas)
                        
                        if pessoa_selecionada:
                            dados_pessoa = df_temp[df_temp['pessoa'] == pessoa_selecionada]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Distribuição para Pessoa {pessoa_selecionada}**")
                                medias_pessoa = dados_pessoa[emocao_cols].mean().sort_values(ascending=True)
                                
                                fig = px.bar(
                                    x=medias_pessoa.values,
                                    y=medias_pessoa.index,
                                    orientation='h',
                                    title=f"Emoções - Pessoa {pessoa_selecionada}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write(f"**Estatísticas:**")
                                st.write(f"- Total de imagens: {len(dados_pessoa)}")
                                emocao_dominante_pessoa = dados_pessoa[emocao_cols].mean().idxmax()
                                st.write(f"- Emoção dominante: **{emocao_dominante_pessoa}**")
                    
                except Exception as e:
                    st.info("Não foi possível extrair informações por pessoa.")
            
            # Amostra dos dados
            st.subheader("Amostra dos Dados")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
    
    else:
        st.info("""
        ### 📋 Instruções de Uso:
        
        1. **Faça upload** de um arquivo CSV com os resultados das emoções
        2. **Formato esperado:**
           - Coluna `file` com caminhos das imagens
           - Colunas para cada emoção (happy, sad, angry, etc.)
           - Valores entre 0 e 1 representando probabilidades
        
        ### 📊 Exemplo de formato:
        ```
        happy,angry,sad,neutral,file
        0.8,0.1,0.05,0.05,outputFramesVideo/frames/pessoa_001/frame_000000.jpg
        0.6,0.2,0.1,0.1,outputFramesVideo/frames/pessoa_001/frame_000001.jpg
        ```
        
        ⚠️ **Importante:** O arquivo deve conter a coluna `file` para análise completa.
        """)

if __name__ == "__main__":
    main()