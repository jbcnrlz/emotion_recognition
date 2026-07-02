import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import textwrap
import re
import string
import os
from collections import Counter
from itertools import combinations
import scipy.stats as stats
from PIL import Image

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

# Configuração da página do Streamlit
st.set_page_config(page_title="Plataforma Integrada Sensorial e de Emoções", layout="wide")
st.title("Plataforma Unificada de Análise Sensorial e Biométrica 🎭📊")
st.markdown("---")

STOPWORDS_PT = set([
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "com", "um", "uma", "uns", "umas", "que", "eu",
    "ele", "ela", "nós", "eles", "elas", "me", "se", "por", "como", "mas",
    "ou", "ao", "aos", "mais", "menos", "muito", "pouco", "tem", "foi",
    "ser", "está", "são", "isso", "esse", "este", "esta", "essa", "aquilo",
    "não", "sim", "já", "só", "também", "nem", "quando", "qual", "quais", "quem"
])

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.getvalue()

# ==========================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# ==========================================

@st.cache_data
def load_consolidado(file):
    # Tenta ler com ponto-e-vírgula (padrão sensorial)
    df = pd.read_csv(file, sep=';')
    if len(df.columns) == 1:
        df = pd.read_csv(file, sep=',')
    return df

@st.cache_data
def load_and_parse_emocoes(file):
    df = pd.read_csv(file, on_bad_lines='skip')
    
    def extract_from_path(path_str):
        path_str = str(path_str).replace('\\', '/')
        parts = path_str.split('/')
        try:
            frame_name = parts[-1].split('.')[0] 
            match = re.search(r'\d+', frame_name)
            frame_num = int(match.group()) if match else 0
            
            aspecto = parts[-2].lower().strip()
            token = parts[-3]
            experimento = parts[-5]
            
            return pd.Series([experimento, token, aspecto, frame_num])
        except Exception:
            return pd.Series(["Desconhecido", "Desconhecido", "Desconhecido", 0])
            
    df[['Experimento_Extraido', 'Token', 'Aspecto', 'frame_num']] = df['file'].apply(extract_from_path)
    return df

def calcular_metricas_confianca(df_emocoes, df_consolidado, limiar_neutro):
    mapa_aspectos = {
        'aparência': 'CARAC_Aparência_Quanto você gostou ou desgostou desse aspecto do produto?',
        'aroma': 'CARAC_Aroma_Quanto você gostou ou desgostou desse aspect do produto.', # Tolerante a variações de grafia
        'cor': 'CARAC_Cor_Quanto você gostou ou desgostou desse aspecto do produto?',
        'sabor': 'CARAC_Sabor_Quanto você gostou ou desgostou desse aspecto do produto?',
        'textura': 'CARAC_Textura_Quanto você gostou ou desgostou desse aspecto do produto?'
    }
    
    # Ajuste dinâmico para mapear as colunas reais do dataframe carregado
    colunas_reais = list(df_consolidado.columns)
    mapa_ajustado = {}
    for chave, padrao in mapa_aspectos.items():
        # Busca por aproximação de string (ex: contém CARAC_Aparência)
        col_encontrada = [c for c in colunas_reais if c.startswith(f'CARAC_{chave.capitalize()}')]
        if col_encontrada:
            mapa_ajustado[chave] = col_encontrada[0]
            
    if not mapa_ajustado or 'Token' not in df_consolidado.columns:
        return pd.DataFrame(), pd.DataFrame()

    colunas_necessarias = list(mapa_ajustado.values())
    df_notas = df_consolidado[['Token'] + colunas_necessarias].copy()
    
    df_notas_long = df_notas.melt(
        id_vars=['Token'], value_vars=colunas_necessarias,
        var_name='Coluna_Original', value_name='Nota_Hedonica'
    )
    inv_mapa = {v: k for k, v in mapa_ajustado.items()}
    df_notas_long['Aspecto'] = df_notas_long['Coluna_Original'].map(inv_mapa)
    
    df_merged = pd.merge(df_emocoes, df_notas_long, on=['Token', 'Aspecto'], how='inner')
    df_merged['Nota_Hedonica'] = pd.to_numeric(df_merged['Nota_Hedonica'], errors='coerce')
    df_merged = df_merged.dropna(subset=['Nota_Hedonica']).copy()
    
    if df_merged.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Cálculos Fisiológicos Vetorizados
    df_merged['H_norm'] = 2 * ((df_merged['Nota_Hedonica'] - 1) / (9 - 1)) - 1
    e_pos = df_merged['happy']
    e_neg = df_merged[['contempt', 'angry', 'disgusted', 'fearful', 'sad']].sum(axis=1)
    df_merged['V'] = e_pos - e_neg
    
    emocoes_cols = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    P = df_merged[emocoes_cols] + 1e-9
    df_merged['Entropia'] = -np.sum(P * np.log2(P), axis=1)
    df_merged['W'] = 1 - (df_merged['Entropia'] / 3)
    df_merged['M'] = 1 - (abs(df_merged['H_norm'] - df_merged['V']) / 2)
    df_merged['Confianca_Frame'] = df_merged['M'] * df_merged['W']
    
    # Separação de frames expressivos
    df_expressivo = df_merged[df_merged['neutral'] < limiar_neutro].copy()
    
    df_base = df_merged.groupby(['Token', 'Aspecto']).agg(
        Experimento=('Experimento_Extraido', 'first'),
        Nota_Declarada=('Nota_Hedonica', 'first'),
        Total_Frames_Video=('file', 'count')
    ).reset_index()
    
    df_metricas = df_expressivo.groupby(['Token', 'Aspecto']).agg(
        Certeza_Media_Rede=('W', 'mean'),
        Confianca_Final=('Confianca_Frame', 'mean'),
        Frames_Expressivos=('file', 'count')
    ).reset_index()
    
    df_final = pd.merge(df_base, df_metricas, on=['Token', 'Aspecto'], how='left')
    df_final['Frames_Expressivos'] = df_final['Frames_Expressivos'].fillna(0).astype(int)
    df_final['Sinal_Insuficiente'] = df_final['Frames_Expressivos'] == 0
    df_final['Confianca_Final_%'] = (df_final['Confianca_Final'] * 100).round(2)
    df_final['Certeza_Media_Rede_%'] = (df_final['Certeza_Media_Rede'] * 100).round(2)
    
    return df_final, df_merged

# ==========================================
# SIDEBAR - INPUT DE ARQUIVOS E FILTROS
# ==========================================
st.sidebar.header("📁 Upload de Arquivos")
file_sensorial = st.sidebar.file_uploader("1. Respostas Consolidadas (Sensorial)", type=['csv'])
file_emocoes = st.sidebar.file_uploader("2. Distribuição de Emoções (Frames)", type=['csv'])

limiar_neutro = st.sidebar.slider(
    "⏳ Limiar de Descarte do Neutro",
    min_value=0.50, max_value=0.95, value=0.80, step=0.05,
    help="Frames com probabilidade de Neutro acima deste valor serão tratados como tempo de leitura/operação e ignorados nas médias de valência."
)

if file_sensorial is None:
    st.info("💡 Carregue o arquivo CSV Sensorial (Consolidado) para desbloquear as análises estatísticas.")
    if file_emocoes is None:
        st.stop()

# Inicialização de Variáveis de Controle
df_s = None
df_e = None
df_confianca_global = pd.DataFrame()
df_frames_calculados = pd.DataFrame()

if file_sensorial is not None:
    df_s = load_consolidado(file_sensorial)
if file_emocoes is not None:
    df_e = load_and_parse_emocoes(file_emocoes)

# Se ambos os arquivos existirem, calcula o cruzamento de confiança
if df_s is not None and df_e is not None:
    df_confianca_global, df_frames_calculados = calcular_metricas_confianca(df_e, df_s, limiar_neutro)

# Configuração de Filtros em Cascata na Sidebar
st.sidebar.divider()
st.sidebar.header("⚙️ Filtros Ativos")

if df_s is not None:
    exps_disponiveis = df_s['Experimento'].unique().tolist()
    selected_exps = st.sidebar.multiselect("Filtrar Experimentos (Aba Sensorial)", exps_disponiveis, default=exps_disponiveis)
    df_s_filtered = df_s[df_s['Experimento'].isin(selected_exps)]

if df_e is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Seleção de Amostra Individual")
    exps_e = df_e['Experimento_Extraido'].unique()
    sel_exp_e = st.sidebar.selectbox("Experimento (Vídeo)", exps_e)
    
    sujeitos_e = df_e[df_e['Experimento_Extraido'] == sel_exp_e]['Token'].unique()
    sel_sujeito_e = st.sidebar.selectbox("Sujeito (Token)", sujeitos_e)
    
    aspectos_e = df_e[(df_e['Experimento_Extraido'] == sel_exp_e) & (df_e['Token'] == sel_sujeito_e)]['Aspecto'].unique()
    sel_aspecto_e = st.sidebar.selectbox("Aspecto (Vídeo)", aspectos_e)

# ==========================================
# ESTRUTURA DE ABAS PRINCIPAIS
# ==========================================
abas = ["🔮 Fusão & Confiança Fisiológica", "📊 Características (Escalas)", "💬 Questionário Qualitativo", "🔬 Estatística Avançada", "🎭 Biometria do Vídeo", "🎞️ Linha do Tempo de Frames"]
tab_fusao, tab_carac, tab_perg, tab_estat, tab_biometria, tab_frames = st.tabs(abas)

# ------------------------------------------
# ABA 1: FUSÃO E CONFIANÇA FISIOLÓGICA
# ------------------------------------------
with tab_fusao:
    st.header("Análise de Consistência e Fusão Multimodal")
    st.markdown("Esta seção cruza a resposta consciente declarada no questionário com a reação subconsciente mapeada frame a frame pela inteligência artificial.")
    
    if df_confianca_global.empty:
        st.warning("⚠️ Para visualizar a consistência da base, é necessário fazer o upload de ambos os arquivos CSV (Sensorial e Emoções).")
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        conf_valida = df_confianca_global[~df_confianca_global['Sinal_Insuficiente']]
        
        col_m1.metric("Média de Confiança Global", f"{conf_valida['Confianca_Final_%'].mean():.2f}%")
        col_m2.metric("Vídeos Analisados", len(df_confianca_global))
        col_m3.metric("Usuários Totalmente Neutros", df_confianca_global['Sinal_Insuficiente'].sum())
        
        st.subheader("Índice de Confiança Mapeado por Indivíduo")
        st.dataframe(
            df_confianca_global,
            column_config={
                "Confianca_Final_%": st.column_config.ProgressColumn("Grau de Confiança", format="%.2f%%", min_value=0, max_value=100),
                "Certeza_Media_Rede_%": "Estabilidade da Rede",
                "Nota_Declarada": "Nota Informada"
            },
            hide_index=True, use_container_width=True
        )

# ------------------------------------------
# ABA 2: CARACTERÍSTICAS (ESCALAS HEDÔNICAS)
# ------------------------------------------
with tab_carac:
    st.header("Análise de Atributos Sensoriais")
    if df_s is not None:
        carac_cols = [col for col in df_s.columns if col.startswith('CARAC_')]
        if carac_cols:
            melted_df = df_s_filtered.melt(id_vars=['Experimento'], value_vars=carac_cols, var_name='Characteristic', value_name='Score')
            melted_df['Aspect'] = melted_df['Characteristic'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else x)
            melted_df['Score'] = pd.to_numeric(melted_df['Score'], errors='coerce').dropna()
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Média de Avaliação por Aspecto")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.barplot(data=melted_df, x='Aspect', y='Score', hue='Experimento', errorbar=None, palette='viridis', ax=ax1)
                ax1.set_ylim(0, 9)
                ax1.set_ylabel("Nota Média")
                ax1.set_xlabel("Aspecto")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig1)
            with c2:
                st.subheader("Distribuição das Notas (Boxplot)")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=melted_df, x='Aspect', y='Score', hue='Experimento', palette='Set2', ax=ax2)
                ax2.set_ylim(0, 9)
                ax2.set_ylabel("Nota")
                ax2.set_xlabel("Aspecto")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
        else:
            st.info("Nenhuma coluna iniciando em 'CARAC_' encontrada.")
    else:
        st.info("Aguardando arquivo sensorial.")

# ------------------------------------------
# ABA 3: PERGUNTAS DO QUESTIONÁRIO
# ------------------------------------------
with tab_perg:
    st.header("Análise Qualitativa e Textual")
    if df_s is not None:
        raw_perg_cols = [col for col in df_s.columns if col.startswith('PERG_')]
        if raw_perg_cols:
            base_perg_dict = {}
            for col in raw_perg_cols:
                base_name = re.sub(r'\.\d+$', '', col)
                if base_name not in base_perg_dict:
                    base_perg_dict[base_name] = []
                base_perg_dict[base_name].append(col)
                
            pergunta_selecionada = st.selectbox("Selecione a pergunta para analisar:", list(base_perg_dict.keys()), format_func=lambda x: x.replace('PERG_', ''))
            colunas_reais = base_perg_dict[pergunta_selecionada]
            
            df_perg = df_s_filtered[['Experimento', 'Participante_Email'] + colunas_reais].melt(
                id_vars=['Experimento', 'Participante_Email'], value_vars=colunas_reais, value_name='Resposta'
            ).dropna()
            df_perg = df_perg[df_perg['Resposta'].astype(str).str.strip() != ""]
            
            if not df_perg.empty:
                qtd_opcoes = df_perg['Resposta'].nunique()
                if qtd_opcoes > 15: # TEXTO LIVRE
                    exps_p = df_perg['Experimento'].unique().tolist()
                    tabs_p = st.tabs(exps_p)
                    for idx, exp in enumerate(exps_p):
                        with tabs_p[idx]:
                            df_exp = df_perg[df_perg['Experimento'] == exp].copy()
                            texto_completo = " ".join(df_exp['Resposta'].astype(str).tolist()).lower()
                            texto_limpo = texto_completo.translate(str.maketrans('', '', string.punctuation))
                            palavras = [p for p in texto_limpo.split() if p not in STOPWORDS_PT and len(p) > 2]
                            
                            freq = Counter(palavras)
                            df_freq = pd.DataFrame(freq.most_common(10), columns=['Palavra', 'Ocorrências'])
                            
                            tx1, tx2 = st.columns([2, 1])
                            with tx1:
                                if WordCloud is not None and texto_completo.strip():
                                    wc = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(texto_completo)
                                    fig_w, ax_w = plt.subplots()
                                    ax_w.imshow(wc, interpolation='bilinear')
                                    ax_w.axis('off')
                                    st.pyplot(fig_w)
                                else:
                                    st.info("Nuvem de palavras indisponível.")
                            with tx2:
                                st.dataframe(df_freq, hide_index=True, use_container_width=True)
                                
                            st.markdown("**Respostas na íntegra:**")
                            st.dataframe(df_exp[['Participante_Email', 'Resposta']], hide_index=True, use_container_width=True)
                else: # CATEGÓRICA
                    fig_c, ax_c = plt.subplots(figsize=(10, 5))
                    df_perg['Resp_Format'] = df_perg['Resposta'].astype(str).apply(lambda x: textwrap.fill(x, width=35))
                    sns.countplot(data=df_perg, y='Resp_Format', hue='Experimento', palette='magma', ax=ax_c)
                    st.pyplot(fig_c)
        else:
            st.info("Nenhuma coluna iniciando em 'PERG_' encontrada.")
    else:
        st.info("Aguardando arquivo sensorial.")

# ------------------------------------------
# ABA 4: ANÁLISE ESTATÍSTICA AVANÇADA
# ------------------------------------------
with tab_estat:
    st.header("Análise de Significância Estatística")
    if df_s is not None and carac_cols:
        if len(selected_exps) >= 2:
            pares = list(combinations(selected_exps, 2))
            for exp1, exp2 in pares:
                st.markdown(f"#### 🆚 Comparação Equivalente: `{exp1}` vs `{exp2}`")
                resultados = []
                aspectos_unicos = melted_df['Aspect'].unique()
                
                for asp in aspectos_unicos:
                    d1 = melted_df[(melted_df['Experimento'] == exp1) & (melted_df['Aspect'] == asp)]['Score']
                    d2 = melted_df[(melted_df['Experimento'] == exp2) & (melted_df['Aspect'] == asp)]['Score']
                    
                    if len(d1) > 1 and len(d2) > 1:
                        t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
                        sig = "Sim (p<0.05) 🟢" if p_val < 0.05 else "Não 🔴"
                        resultados.append({"Aspecto": asp, f"Média ({exp1})": round(d1.mean(), 2), f"Média ({exp2})": round(d2.mean(), 2), "Valor p": round(p_val, 4), "Significativo?": sig})
                        
                if resultados:
                    st.dataframe(pd.DataFrame(resultados), hide_index=True, use_container_width=True)
        else:
            st.warning("Selecione pelo menos dois experimentos na barra lateral.")
    else:
        st.info("Aguardando arquivo sensorial estruturado.")

# ------------------------------------------
# ABA 5: BIOMETRIA DO VÍDEO
# ------------------------------------------
with tab_biometria:
    st.header("Visão Geral das Respostas Emocionais Dinâmicas")
    if df_e is not None:
        df_vid = df_e[(df_e['Experimento_Extraido'] == sel_exp_e) & (df_e['Token'] == sel_sujeito_e) & (df_e['Aspecto'] == sel_aspecto_e)].sort_values('frame_num').reset_index(drop=True)
        emotions = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
        
        if not df_vid.empty:
            b1, b2 = st.columns([1, 2])
            with b1:
                st.subheader("Distribuição Média do Vídeo")
                df_m = df_vid[emotions].mean().reset_index()
                df_m.columns = ['Emoção', 'Média']
                fig_m = px.bar(df_m, x='Emoção', y='Média', color='Emoção', range_y=[0,1], text_auto='.1%')
                st.plotly_chart(fig_m, use_container_width=True)
            with b2:
                st.subheader("Série Temporal de Emoções")
                fig_l = px.line(df_vid, x='frame_num', y=emotions, labels={'frame_num': 'Frame', 'value': 'Probabilidade'})
                st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.info("Nenhum frame localizado para o filtro selecionado.")
    else:
        st.info("Aguardando carregamento das emoções dos vídeos.")

# ------------------------------------------
# ABA 6: LINHA DO TEMPO DE FRAMES
# ------------------------------------------
with tab_frames:
    st.header("Inspeção Frame a Frame")
    if df_e is not None and not df_vid.empty:
        idx = st.slider("Selecione o Frame:", 0, len(df_vid)-1, 0)
        row = df_vid.iloc[idx]
        
        f1, f2 = st.columns([1, 1.5])
        with f1:
            st.metric("Número do Frame", int(row['frame_num']))
            if os.path.exists(str(row['file'])):
                st.image(Image.open(str(row['file'])), use_container_width=True)
            else:
                st.error(f"Caminho físico de imagem não localizado:\n`{row['file']}`")
        with f2:
            df_f = pd.DataFrame(list({e: row[e] for e in emotions}.items()), columns=['Emoção', 'Probabilidade'])
            fig_f = px.bar(df_f, x='Emoção', y='Probabilidade', color='Emoção', range_y=[0,1], text_auto='.1%')
            st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Aguardando dados biométricos.")