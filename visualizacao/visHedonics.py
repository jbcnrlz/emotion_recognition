import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback

# Configuração da página
st.set_page_config(
    page_title="Análise de Emoções Faciais",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #2ecc71; font-weight: bold; }
    .negative { color: #e74c3c; font-weight: bold; }
    .neutral { color: #95a5a6; font-weight: bold; }
    .high-freq { color: #e74c3c; font-weight: bold; }
    .medium-freq { color: #f39c12; font-weight: bold; }
    .low-freq { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def analisar_emocoes(df):
    """
    Analisa o DataFrame e retorna todas as métricas
    """
    try:
        # Informações básicas
        emocao_cols = [col for col in df.columns if col != 'file']
        medias = df[emocao_cols].mean().sort_values(ascending=False)
        
        # Extrair pessoas com segurança
        try:
            df_temp = df.copy()
            df_temp['pessoa'] = df_temp['file'].str.extract(r'pessoa_(\d+)')
            pessoas_unicas = df_temp['pessoa'].nunique() if 'pessoa' in df_temp.columns and df_temp['pessoa'].notna().any() else 0
        except:
            pessoas_unicas = 0
        
        # Análise hedônica
        emo_positivas = ['happy', 'elated', 'hopeful', 'surprised', 'proud', 'loved']
        emo_negativas = ['angry', 'disgusted', 'fearful', 'sad', 'fatigued', 'contempt']
        emo_neutras = ['astonished', 'neutral']
        
        positivas_presentes = [e for e in emo_positivas if e in medias.index]
        negativas_presentes = [e for e in emo_negativas if e in medias.index]
        neutras_presentes = [e for e in emo_neutras if e in medias.index]
        
        score_positivo = medias[positivas_presentes].sum() if positivas_presentes else 0
        score_negativo = medias[negativas_presentes].sum() if negativas_presentes else 0
        score_neutro = medias[neutras_presentes].sum() if neutras_presentes else 0
        
        # Obter emoções com verificação de segurança
        emocao_dominante = medias.index[0] if len(medias) > 0 else "N/A"
        valor_dominante = medias.iloc[0] if len(medias) > 0 else 0
        
        emocao_menos_frequente = medias.index[-1] if len(medias) > 1 else emocao_dominante
        valor_menos_frequente = medias.iloc[-1] if len(medias) > 1 else valor_dominante
        
        return {
            'medias': medias,
            'total_imagens': len(df),
            'total_emocoes': len(emocao_cols),
            'pessoas_unicas': pessoas_unicas,
            'score_positivo': score_positivo,
            'score_negativo': score_negativo,
            'score_neutro': score_neutro,
            'emocao_dominante': emocao_dominante,
            'valor_dominante': valor_dominante,
            'emocao_menos_frequente': emocao_menos_frequente,
            'valor_menos_frequente': valor_menos_frequente
        }
    
    except Exception as e:
        st.error(f"❌ Erro na análise: {str(e)}")
        return None

def criar_visualizacoes(df, medias):
    """
    Cria todas as visualizações gráficas da versão original
    """
    try:
        emocao_cols = [col for col in df.columns if col != 'file']
        
        # Configurar o estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Criar figura com subplots (layout original)
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Gráfico de barras horizontal (principal) - TOPO ESQUERDO
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(medias)))
        bars = ax1.barh(range(len(medias)), medias.values, color=colors)
        
        ax1.set_yticks(range(len(medias)))
        ax1.set_yticklabels(medias.index, fontsize=10)
        ax1.set_xlabel('Probabilidade Média', fontsize=12)
        ax1.set_title('📊 Distribuição Média das Emoções', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}\n({width*100:.1f}%)', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Gráfico de pizza (top 6 emoções) - TOPO DIREITO
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        top_n = min(6, len(medias))
        top_emocoes = medias.head(top_n)
        
        if len(top_emocoes) > 1:
            wedges, texts, autotexts = ax2.pie(
                top_emocoes.values, 
                labels=top_emocoes.index, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=plt.cm.Set3(np.linspace(0, 1, len(top_emocoes)))
            )
            ax2.set_title(f'🎯 Top {top_n} Emoções Mais Frequentes', fontsize=14, fontweight='bold', pad=20)
        else:
            ax2.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('🎯 Top Emoções', fontsize=14, fontweight='bold', pad=20)
        
        # 3. Heatmap de correlação - MEIO ESQUERDA
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        
        if len(emocao_cols) > 1:
            correlacao = df[emocao_cols].corr()
            im = ax3.imshow(correlacao, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            ax3.set_xticks(range(len(emocao_cols)))
            ax3.set_yticks(range(len(emocao_cols)))
            ax3.set_xticklabels(emocao_cols, rotation=45, ha='right', fontsize=9)
            ax3.set_yticklabels(emocao_cols, fontsize=9)
            ax3.set_title('🔗 Correlação entre Emoções', fontsize=14, fontweight='bold', pad=20)
            
            # Adicionar valores no heatmap
            for i in range(len(emocao_cols)):
                for j in range(len(emocao_cols)):
                    color = "white" if abs(correlacao.iloc[i, j]) > 0.5 else "black"
                    ax3.text(j, i, f'{correlacao.iloc[i, j]:.2f}',
                            ha="center", va="center", color=color, fontsize=8, fontweight='bold')
            
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(0.5, 0.5, 'Múltiplas emoções\nnecessárias', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('🔗 Correlação entre Emoções', fontsize=14, fontweight='bold', pad=20)
        
        # 4. Distribuição por faixas de probabilidade - MEIO DIREITA
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        bins = [0, 0.05, 0.1, 0.2, 0.3, 1.0]
        labels = ['Muito Baixa\n(0-5%)', 'Baixa\n(5-10%)', 'Moderada\n(10-20%)', 'Alta\n(20-30%)', 'Muito Alta\n(>30%)']
        
        dist_faixas = pd.cut(medias, bins=bins, labels=labels).value_counts().sort_index()
        
        colors_faixas = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars_faixas = ax4.bar(range(len(dist_faixas)), dist_faixas.values, color=colors_faixas)
        
        ax4.set_xticks(range(len(dist_faixas)))
        ax4.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        ax4.set_ylabel('Número de Emoções')
        ax4.set_title('📈 Distribuição por Faixas de Probabilidade', fontsize=14, fontweight='bold', pad=20)
        ax4.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, valor in enumerate(dist_faixas.values):
            ax4.text(i, valor + 0.1, str(valor), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 5. Análise hedônica - INFERIOR (largura total)
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        categorias_hedonicas = ['Positivas', 'Negativas', 'Neutras']
        scores = [analise['score_positivo'], analise['score_negativo'], analise['score_neutro']]
        colors_hedonicas = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        bars_hedonicas = ax5.bar(categorias_hedonicas, scores, color=colors_hedonicas, alpha=0.8)
        ax5.set_ylabel('Score Acumulado')
        ax5.set_title('😊 Análise Hedônica - Scores por Categoria', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(axis='y', alpha=0.3)
        
        # Adicionar valores e porcentagens
        for i, (bar, score) in enumerate(zip(bars_hedonicas, scores)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f'{score:.3f}\n({score*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"❌ Erro ao criar visualizações: {str(e)}")
        return None

def classificar_frequencia(percentual):
    """Classifica a frequência baseada na porcentagem"""
    if percentual > 20:
        return "high-freq", "🔴 MUITO ALTA"
    elif percentual > 10:
        return "medium-freq", "🟡 ALTA"
    elif percentual > 5:
        return "medium-freq", "🟢 MODERADA"
    elif percentual > 2:
        return "low-freq", "🔵 BAIXA"
    else:
        return "low-freq", "⚪ MUITO BAIXA"

def analise_por_pessoa(df):
    """Análise detalhada por pessoa"""
    try:
        if 'pessoa' not in df.columns:
            df_temp = df.copy()
            df_temp['pessoa'] = df_temp['file'].str.extract(r'pessoa_(\d+)')
        
        if 'pessoa' in df.columns and df['pessoa'].notna().any():
            emocao_cols = [col for col in df.columns if col not in ['file', 'pessoa']]
            pessoas_medias = df.groupby('pessoa')[emocao_cols].mean()
            
            resultados = []
            for pessoa in pessoas_medias.index:
                pessoa_medias = pessoas_medias.loc[pessoa]
                emocao_dominante = pessoa_medias.idxmax()
                valor_dominante = pessoa_medias.max()
                resultados.append({
                    'pessoa': pessoa,
                    'emocao_dominante': emocao_dominante,
                    'valor_dominante': valor_dominante,
                    'total_imagens': len(df[df['pessoa'] == pessoa])
                })
            
            return resultados
        return None
    except:
        return None

# Interface principal
def main():
    st.markdown('<h1 class="main-header">😊 Análise de Emoções Faciais</h1>', unsafe_allow_html=True)
    st.markdown("### Carregue seu arquivo CSV para análise detalhada das emoções")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "📁 Selecione o arquivo CSV com os resultados das emoções:",
        type=['csv'],
        help="O arquivo deve conter colunas para cada emoção e uma coluna 'file'"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df = pd.read_csv(uploaded_file)
            
            # Verificar se tem coluna 'file'
            if 'file' not in df.columns:
                st.error("❌ O arquivo deve conter a coluna 'file' com os caminhos das imagens.")
                return
            
            # Realizar análise
            with st.spinner('🔍 Analisando dados...'):
                global analise
                analise = analisar_emocoes(df)
            
            if analise is None:
                st.error("❌ Falha na análise dos dados.")
                return
            
            st.success(f"✅ Análise concluída! {analise['total_imagens']} imagens processadas.")
            
            # Abas para organização (MESMO LAYOUT DA VERSÃO ANTERIOR)
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Visão Geral", 
                "🎭 Tabela de Emoções", 
                "📊 Visualizações",
                "👤 Análise Detalhada"
            ])
            
            with tab1:
                st.subheader("📋 Resumo Executivo")
                
                # Métricas principais (MESMO LAYOUT)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Imagens", f"{analise['total_imagens']:,}")
                
                with col2:
                    st.metric("Emoções Detectadas", analise['total_emocoes'])
                
                with col3:
                    st.metric("Pessoas Únicas", analise['pessoas_unicas'])
                
                with col4:
                    st.metric(
                        "Emoção Dominante", 
                        f"{analise['emocao_dominante']}", 
                        f"{analise['valor_dominante']*100:.1f}%"
                    )
                
                # Análise hedônica (MESMO LAYOUT)
                st.subheader("😊 Análise Hedônica")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'<div class="metric-card positive">'
                               f'<h3>😊 Positivas</h3>'
                               f'<h2>{analise["score_positivo"]*100:.1f}%</h2>'
                               f'</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="metric-card negative">'
                               f'<h3>😠 Negativas</h3>'
                               f'<h2>{analise["score_negativo"]*100:.1f}%</h2>'
                               f'</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'<div class="metric-card neutral">'
                               f'<h3>😐 Neutras</h3>'
                               f'<h2>{analise["score_neutro"]*100:.1f}%</h2>'
                               f'</div>', unsafe_allow_html=True)
                
                # Avaliação geral (MESMA LÓGICA)
                if analise['score_positivo'] > analise['score_negativo']:
                    st.success("""
                    **✅ TENDÊNCIA POSITIVA:** A experiência geral foi predominantemente positiva. 
                    As expressões faciais indicam satisfação e engajamento durante o consumo.
                    """)
                else:
                    st.warning("""
                    **⚠️ TENDÊNCIA NEUTRA/NEGATIVA:** A experiência apresentou respostas mistas ou negativas. 
                    Recomenda-se análise mais detalhada das causas.
                    """)
            
            with tab2:
                st.subheader("📋 Tabela Detalhada das Emoções")
                
                # Criar tabela formatada (MESMO FORMATO)
                tabela_data = []
                for i, (emocao, media) in enumerate(analise['medias'].items(), 1):
                    percentual = media * 100
                    classe, classificacao = classificar_frequencia(percentual)
                    
                    tabela_data.append({
                        'Posição': i,
                        'Emoção': emocao,
                        'Média': f"{media:.4f}",
                        'Porcentagem': f"{percentual:.2f}%",
                        'Classificação': classificacao
                    })
                
                # Exibir tabela (MESMO ESTILO)
                st.dataframe(
                    tabela_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Posição": st.column_config.NumberColumn(width="small"),
                        "Emoção": st.column_config.TextColumn(width="medium"),
                        "Média": st.column_config.TextColumn(width="small"),
                        "Porcentagem": st.column_config.TextColumn(width="small"),
                        "Classificação": st.column_config.TextColumn(width="medium")
                    }
                )
                
                # Estatísticas adicionais (MESMO LAYOUT)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Emoção Mais Frequente", 
                        analise['emocao_dominante'], 
                        f"{analise['valor_dominante']*100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Emoção Menos Frequente", 
                        analise['emocao_menos_frequente'], 
                        f"{analise['valor_menos_frequente']*100:.2f}%"
                    )
            
            with tab3:
                st.subheader("📊 Visualizações Gráficas")
                
                # Criar visualizações (MESMOS GRÁFICOS)
                with st.spinner('🖼️ Gerando gráficos...'):
                    fig = criar_visualizacoes(df, analise['medias'])
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ Não foi possível gerar as visualizações completas")
            
            with tab4:
                st.subheader("👤 Análise Detalhada por Pessoa")
                
                # Análise por pessoa (MESMA LÓGICA)
                resultados_pessoas = analise_por_pessoa(df)
                
                if resultados_pessoas:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top 5 Pessoas - Emoção Dominante:**")
                        for resultado in resultados_pessoas[:5]:
                            st.write(f"• **Pessoa {resultado['pessoa']}**: {resultado['emocao_dominante']} ({resultado['valor_dominante']*100:.1f}%) - {resultado['total_imagens']} imagens")
                    
                    with col2:
                        st.write("**Distribuição de Pessoas:**")
                        st.write(f"• Total de pessoas analisadas: {len(resultados_pessoas)}")
                        st.write(f"• Média de imagens por pessoa: {analise['total_imagens'] // len(resultados_pessoas)}")
                        
                        # Estatísticas de diversidade emocional
                        emocoes_dominantes = [r['emocao_dominante'] for r in resultados_pessoas]
                        st.write(f"• Diversidade emocional: {len(set(emocoes_dominantes))} emoções diferentes dominantes")
                else:
                    st.info("ℹ️ Não foi possível identificar pessoas distintas nos dados.")
                
                # Amostra dos dados (MESMO FORMATO)
                st.subheader("📄 Amostra dos Dados Originais")
                st.dataframe(df.head(10), use_container_width=True)
            
            # Sidebar com informações adicionais (MESMO CONTEÚDO)
            st.sidebar.markdown("## 📊 Estatísticas do Dataset")
            st.sidebar.write(f"**Arquivo:** {uploaded_file.name}")
            st.sidebar.write(f"**Tamanho:** {len(df):,} linhas")
            st.sidebar.write(f"**Colunas:** {len(df.columns)}")
            st.sidebar.write(f"**Pessoas:** {analise['pessoas_unicas']}")
            
            # Download dos resultados (MESMA FUNCIONALIDADE)
            st.sidebar.markdown("## 📥 Exportar Resultados")
            
            relatorio_df = pd.DataFrame({
                'Emoção': analise['medias'].index,
                'Média': analise['medias'].values,
                'Porcentagem': analise['medias'].values * 100
            })
            
            csv = relatorio_df.to_csv(index=False)
            st.sidebar.download_button(
                label="📊 Baixar Relatório (CSV)",
                data=csv,
                file_name="relatorio_emocoes.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            
            # Debug information
            with st.expander("🔍 Detalhes do Erro"):
                st.code(traceback.format_exc())
    
    else:
        # Tela inicial com instruções (MESMO CONTEÚDO)
        st.info("""
        ### 📋 Como usar esta aplicação:
        
        1. **Faça upload** do arquivo CSV com os resultados das emoções
        2. **Aguarde** o processamento automático dos dados
        3. **Navegue** pelas abas para explorar as análises:
           - 📈 **Visão Geral**: Métricas principais e análise hedônica
           - 🎭 **Tabela de Emoções**: Dados detalhados de cada emoção
           - 📊 **Visualizações**: Gráficos e heatmaps interativos
           - 👤 **Análise Detalhada**: Informações por pessoa e dados brutos
        
        ### 🎯 Formato esperado do CSV:
        - Coluna `file` com caminhos das imagens
        - Colunas para cada emoção (happy, sad, angry, etc.)
        - Valores entre 0 e 1 representando probabilidades
        
        **Exemplo:**
        ```
        happy,angry,sad,neutral,file
        0.8,0.1,0.05,0.05,outputFramesVideo/frames/pessoa_001/frame_000000.jpg
        0.6,0.2,0.1,0.1,outputFramesVideo/frames/pessoa_001/frame_000001.jpg
        ```
        """)

if __name__ == "__main__":
    main()