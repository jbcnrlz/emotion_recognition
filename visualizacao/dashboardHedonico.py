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
import unicodedata
import tempfile
from collections import Counter
from itertools import combinations
import scipy.stats as stats
from PIL import Image

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# Configuração da página do Streamlit
st.set_page_config(page_title="Plataforma Integrada Sensorial e de Emoções", layout="wide")
st.title("Plataforma Unificada de Análise Sensorial e Biométrica 🎭📊")
st.markdown("---")

# Configuração Padrão do Plotly para Edição Fácil (Textos clicáveis e Download em SVG)
PLOTLY_CONFIG = {
    'editable': True,
    'edits': {'titleText': True, 'axisTitleText': True, 'legendText': True},
    'toImageButtonOptions': {
        'format': 'svg', 
        'filename': 'grafico_editavel',
        'height': 600,
        'width': 800,
        'scale': 1
    }
}

STOPWORDS_PT = set([
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "com", "um", "uma", "uns", "umas", "que", "eu",
    "ele", "ela", "nós", "eles", "elas", "me", "se", "por", "como", "mas",
    "ou", "ao", "aos", "mais", "menos", "muito", "pouco", "tem", "foi",
    "ser", "está", "são", "isso", "esse", "este", "esta", "essa", "aquilo",
    "não", "sim", "já", "só", "também", "nem", "quando", "qual", "quais", "quem",
    "gostei", "menos", "mais", "achei"
])

EMO_COLS = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']

def remover_acentos(txt):
    return unicodedata.normalize('NFKD', str(txt)).encode('ASCII', 'ignore').decode('ASCII')

def formatar_texto_celula(texto, max_len=20):
    limpo = remover_acentos(str(texto).capitalize())
    return limpo[:max_len] + ".." if len(limpo) > max_len else limpo

# ==========================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# ==========================================

@st.cache_data
def load_consolidado(file):
    df = pd.read_csv(file, sep=';')
    if len(df.columns) == 1:
        df = pd.read_csv(file, sep=',')
    return df

@st.cache_data
def load_and_parse_emocoes(file):
    df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.lower()
    
    if 'caminho' in df.columns: df = df.rename(columns={'caminho': 'file'})
    elif 'path' in df.columns: df = df.rename(columns={'path': 'file'})
        
    if 'file' not in df.columns:
        st.error(f"🚨 A coluna de caminhos de imagem não foi encontrada! Colunas lidas: {list(df.columns)}")
        st.stop()
    
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

def calcular_metricas_confianca(df_emocoes, df_consolidado, limiar_neutro, v_max):
    try:
        mapa_aspectos = {
            'aparência': 'CARAC_Aparência_Quanto você gostou ou desgostou desse aspecto do produto?',
            'aroma': 'CARAC_Aroma_Quanto você gostou ou desgostou desse aspect do produto.', 
            'cor': 'CARAC_Cor_Quanto você gostou ou desgostou desse aspecto do produto?',
            'sabor': 'CARAC_Sabor_Quanto você gostou ou desgostou desse aspecto do produto?',
            'textura': 'CARAC_Textura_Quanto você gostou ou desgostou desse aspecto do produto?'
        }
        
        colunas_reais = list(df_consolidado.columns)
        mapa_ajustado = {}
        for chave, padrao in mapa_aspectos.items():
            col_encontrada = [c for c in colunas_reais if c.startswith(f'CARAC_{chave.capitalize()}')]
            if col_encontrada: mapa_ajustado[chave] = col_encontrada[0]
                
        if not mapa_ajustado or 'Token' not in df_consolidado.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        df_merged['H_norm'] = 2 * ((df_merged['Nota_Hedonica'] - 1) / (9 - 1)) - 1
        
        e_pos = df_merged['happy']
        e_neg = df_merged[['contempt', 'angry', 'disgusted', 'fearful', 'sad']].sum(axis=1)
        df_merged['V_bruto'] = e_pos - e_neg
        df_merged['V'] = np.clip(df_merged['V_bruto'] / v_max, -1.0, 1.0)
        
        P = df_merged[EMO_COLS] + 1e-9
        df_merged['Entropia'] = -np.sum(P * np.log2(P), axis=1)
        df_merged['W'] = 1 - (df_merged['Entropia'] / 3)
        
        df_merged['M'] = 1 - (abs(df_merged['H_norm'] - df_merged['V']) / 2)
        df_merged['Confianca_Frame'] = df_merged['M'] * df_merged['W']
        
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
        
        df_emo_token = df_merged.groupby('Token')[EMO_COLS].mean().reset_index()
        
        return df_final, df_merged, df_emo_token
    except Exception as e:
        st.sidebar.warning(f"Aviso na Confiança: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ==========================================
# MOTOR DE GERAÇÃO DE PDF (DESIGN PREMIUM)
# ==========================================
def header_tabela(pdf, colunas, larguras):
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 9)
    for col, w in zip(colunas, larguras):
        pdf.cell(w, 8, remover_acentos(col), border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 9)

def gerar_pdf_academico(melted_df_academico, df_confianca_academico, df_frames, df_emo_token, v_max_val, limiar_val, exps, df_s_filtered):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- CAPA ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 10, remover_acentos("RELATÓRIO DE PESQUISA SENSORIAL"), ln=True, align="C")
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, remover_acentos("Estatística Hedônica e Validação Biométrica Multimodal"), ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, remover_acentos("Gerado automaticamente pela Plataforma de IA"), ln=True, align="C")
    pdf.ln(20)

    # --- 1. METODOLOGIA ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, remover_acentos(" 1. Parâmetros Metodológicos e Amostra"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    participantes = df_s_filtered['Participante_Email'].nunique() if 'Participante_Email' in df_s_filtered.columns else len(df_s_filtered)
    exps_str = ", ".join([str(e) for e in exps])
    
    pdf.ln(2)
    pdf.cell(0, 6, remover_acentos(f"  • Experimentos Analisados: {exps_str}"), ln=True)
    pdf.cell(0, 6, remover_acentos(f"  • Tamanho da Amostra (N): {participantes} Sessões Sensoriais"), ln=True)
    pdf.cell(0, 6, remover_acentos(f"  • Teto Fisiológico de Saturação (V_max): {v_max_val}"), ln=True)
    pdf.cell(0, 6, remover_acentos(f"  • Limiar de Descarte Cognitivo (Face Neutra): {limiar_val}"), ln=True)
    pdf.ln(10)

    # --- 2. ESTATÍSTICA HEDÔNICA ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, remover_acentos(" 2. Estatísticas Descritivas (Escala Hedônica)"), ln=True, fill=True)
    pdf.ln(3)

    if not melted_df_academico.empty:
        header_tabela(pdf, ["Aspecto", "N", "Média +/- DP", "Mediana", "Min-Max"], [40, 20, 45, 30, 30])
        stats_hed = melted_df_academico.groupby('Aspect')['Score'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index()
        
        fill = False
        for _, row in stats_hed.iterrows():
            pdf.set_fill_color(245, 245, 245)
            dp = row['std'] if pd.notnull(row['std']) else 0.0
            pdf.cell(40, 8, formatar_texto_celula(row['Aspect'], 20), border=1, fill=fill)
            pdf.cell(20, 8, str(int(row['count'])), border=1, align="C", fill=fill)
            pdf.cell(45, 8, f"{row['mean']:.2f} +/- {dp:.2f}", border=1, align="C", fill=fill)
            pdf.cell(30, 8, f"{row['median']:.1f}", border=1, align="C", fill=fill)
            pdf.cell(30, 8, f"{row['min']:.1f} - {row['max']:.1f}", border=1, align="C", fill=fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    if not melted_df_academico.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=melted_df_academico, x='Aspect', y='Score', hue='Experimento', palette='viridis', ax=ax)
        ax.set_ylim(0, 9)
        ax.set_title(remover_acentos("Média Hedônica por Aspecto"))
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img1:
            fig.savefig(tmp_img1.name, dpi=300)
            pdf.image(tmp_img1.name, w=150)
        os.unlink(tmp_img1.name)
        plt.close(fig)
        pdf.ln(5)

    # --- 3. SIGNIFICÂNCIA ESTATÍSTICA ---
    if len(exps) >= 2 and not melted_df_academico.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 3. Análise de Significância (Teste t de Welch)"), ln=True, fill=True)
        pdf.ln(3)
        
        pares = list(combinations(exps, 2))
        for exp1, exp2 in pares:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 8, remover_acentos(f"Comparação: {exp1} vs {exp2}"), ln=True)
            
            header_tabela(pdf, ["Aspecto", "Estatística t", "Valor p", "Significativo (p<0.05)"], [50, 40, 40, 40])
            
            fill = False
            for asp in melted_df_academico['Aspect'].unique():
                pdf.set_fill_color(245, 245, 245)
                d1 = melted_df_academico[(melted_df_academico['Experimento'] == exp1) & (melted_df_academico['Aspect'] == asp)]['Score']
                d2 = melted_df_academico[(melted_df_academico['Experimento'] == exp2) & (melted_df_academico['Aspect'] == asp)]['Score']
                
                if len(d1) > 1 and len(d2) > 1:
                    t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
                    if pd.isna(t_stat) or pd.isna(p_val):
                        t_str, p_str, sig = "N/A", "N/A", "-"
                    else:
                        t_str, p_str = f"{t_stat:.3f}", f"{p_val:.4f}"
                        sig = "Sim" if p_val < 0.05 else "Nao"
                    
                    pdf.cell(50, 8, formatar_texto_celula(asp, 22), border=1, fill=fill)
                    pdf.cell(40, 8, t_str, border=1, align="C", fill=fill)
                    pdf.cell(40, 8, p_str, border=1, align="C", fill=fill)
                    pdf.cell(40, 8, sig, border=1, align="C", fill=fill)
                    pdf.ln()
                    fill = not fill
            pdf.ln(5)

    # --- 4. PERFIL FISIOLÓGICO DA REDE NEURAL ---
    pdf.add_page()
    if not df_frames.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 4. Perfil Fisiológico Médio (Visão Computacional)"), ln=True, fill=True)
        pdf.ln(3)

        emocoes = ['neutral', 'happy', 'disgusted', 'surprised', 'angry', 'sad']
        emocoes_presentes = [e for e in emocoes if e in df_frames.columns]
        
        if emocoes_presentes:
            col_names = ["Aspecto"] + [e.capitalize() for e in emocoes_presentes]
            widths = [35] + [23] * len(emocoes_presentes)
            header_tabela(pdf, col_names, widths)

            stats_emo = df_frames.groupby('Aspecto')[emocoes_presentes].mean().reset_index()
            fill = False
            for _, row in stats_emo.iterrows():
                pdf.set_fill_color(245, 245, 245)
                pdf.cell(35, 8, formatar_texto_celula(row['Aspecto'], 16), border=1, fill=fill)
                for emo in emocoes_presentes:
                    pdf.cell(23, 8, f"{(row[emo]*100):.1f}%", border=1, align="C", fill=fill)
                pdf.ln()
                fill = not fill
        pdf.ln(8)

    # --- 5. VALIDAÇÃO BIOMÉTRICA DE CONFIANÇA ---
    if not df_confianca_academico.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 5. Validação Biométrica (Consistência Multimodal)"), ln=True, fill=True)
        pdf.ln(3)

        header_tabela(pdf, ["Aspecto", "Frames Totais", "Sinal Expressivo", "Confianca Global"], [45, 40, 45, 45])

        stats_conf = df_confianca_academico.groupby('Aspecto').agg(
            Frames_Tot=('Total_Frames_Video', 'sum'),
            Frames_Exp=('Frames_Expressivos', 'sum'),
            Confianca_Media=('Confianca_Final_%', 'mean')
        ).reset_index()
        
        fill = False
        for _, row in stats_conf.iterrows():
            pdf.set_fill_color(245, 245, 245)
            pct_exp = (row['Frames_Exp'] / row['Frames_Tot']) * 100 if row['Frames_Tot'] > 0 else 0
            pdf.cell(45, 8, formatar_texto_celula(row['Aspecto'], 20), border=1, fill=fill)
            pdf.cell(40, 8, f"{int(row['Frames_Tot'])}", border=1, align="C", fill=fill)
            pdf.cell(45, 8, f"{pct_exp:.1f}%", border=1, align="C", fill=fill)
            pdf.cell(45, 8, f"{row['Confianca_Media']:.2f}%", border=1, align="C", fill=fill)
            pdf.ln()
            fill = not fill
        pdf.ln(8)

    # --- 6. EXTRAÇÃO QUALITATIVA ---
    raw_perg_cols = [col for col in df_s_filtered.columns if col.startswith('PERG_')]
    if raw_perg_cols:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 6. Análise Qualitativa de Questionário"), ln=True, fill=True)
        pdf.ln(4)
        
        for col in raw_perg_cols:
            respostas_validas = df_s_filtered[col].dropna()
            if respostas_validas.empty: continue
            
            nome_perg_bruto = col.replace('PERG_', '')
            nome_perg_bruto = re.sub(r'\.\d+$', '', nome_perg_bruto)
            nome_perg = remover_acentos(nome_perg_bruto.split('_')[-1] if '_' in nome_perg_bruto else nome_perg_bruto)
            
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, remover_acentos(f"Questão: {nome_perg}"), ln=True)
            
            qtd_unicas = respostas_validas.nunique()
            
            # TEXTO LIVRE
            if qtd_unicas > 15: 
                texto_completo = " ".join(respostas_validas.astype(str).tolist()).lower()
                texto_limpo = texto_completo.translate(str.maketrans('', '', string.punctuation))
                palavras = [p for p in texto_limpo.split() if p not in STOPWORDS_PT and len(p) > 2]
                
                freq = Counter(palavras).most_common(8)
                termos_str = ", ".join([f"{p[0].capitalize()} ({p[1]})" for p in freq])
                
                pdf.set_font("Arial", "", 9)
                pdf.multi_cell(0, 6, remover_acentos(f"> Principais termos citados: {termos_str}"))
                
                if not df_emo_token.empty and 'Token' in df_s_filtered.columns:
                    df_tokens_q = df_s_filtered[['Token', col]].dropna()
                    df_emo_text = pd.merge(df_tokens_q, df_emo_token, on='Token', how='inner')
                    if not df_emo_text.empty:
                        emo_means = df_emo_text[EMO_COLS].mean() * 100
                        principais_emo = emo_means.sort_values(ascending=False).head(3)
                        emo_str = ", ".join([f"{k.capitalize()}: {v:.1f}%" for k, v in principais_emo.items()])
                        pdf.set_text_color(100, 100, 100)
                        pdf.multi_cell(0, 6, remover_acentos(f"> Perfil Emocional Predominante deste grupo: {emo_str}"))
                        pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
                
            # MÚLTIPLA ESCOLHA
            else:
                counts = respostas_validas.value_counts().reset_index()
                counts.columns = ['Resposta', 'Votos']
                total_votos = counts['Votos'].sum()
                
                has_bio = False
                if not df_confianca_academico.empty and 'Token' in df_s_filtered.columns:
                    df_perg_cat = df_s_filtered[['Token', col]].dropna().rename(columns={col: 'Resposta'})
                    df_conf_token = df_confianca_academico.groupby('Token')['Confianca_Final_%'].mean().reset_index()
                    df_merged_cat = pd.merge(df_perg_cat, df_conf_token, on='Token', how='inner')
                    
                    if not df_merged_cat.empty:
                        has_bio = True
                        cat_stats = df_merged_cat.groupby('Resposta').agg(Conf_Media=('Confianca_Final_%', 'mean')).reset_index()
                        counts = pd.merge(counts, cat_stats, on='Resposta', how='left')

                if has_bio:
                    header_tabela(pdf, ["Opção Escolhida", "Votos", "%", "Confiança Média"], [95, 20, 20, 40])
                else:
                    header_tabela(pdf, ["Opção Escolhida", "Votos", "%"], [135, 20, 20])
                
                fill = False
                for _, row in counts.iterrows():
                    pdf.set_fill_color(245, 245, 245)
                    resp_str = remover_acentos(textwrap.shorten(str(row['Resposta']), width=50, placeholder="..."))
                    votos = int(row['Votos'])
                    pct = (votos / total_votos) * 100
                    
                    if has_bio:
                        conf_val = f"{row['Conf_Media']:.1f}%" if pd.notnull(row.get('Conf_Media')) else "N/A"
                        pdf.cell(95, 8, resp_str, border=1, fill=fill)
                        pdf.cell(20, 8, str(votos), border=1, align="C", fill=fill)
                        pdf.cell(20, 8, f"{pct:.1f}%", border=1, align="C", fill=fill)
                        pdf.cell(40, 8, conf_val, border=1, align="C", fill=fill)
                    else:
                        pdf.cell(135, 8, resp_str, border=1, fill=fill)
                        pdf.cell(20, 8, str(votos), border=1, align="C", fill=fill)
                        pdf.cell(20, 8, f"{pct:.1f}%", border=1, align="C", fill=fill)
                    pdf.ln()
                    fill = not fill
                pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.unlink(tmp.name)
    return pdf_bytes

# ==========================================
# VARIÁVEIS GLOBAIS DE EXECUÇÃO
# ==========================================
df_s = None
df_e = None
df_confianca_global = pd.DataFrame()
df_frames_calculados = pd.DataFrame()
df_emo_token = pd.DataFrame()
df_s_filtered = pd.DataFrame()

# ==========================================
# SIDEBAR - INPUT DE ARQUIVOS E FILTROS
# ==========================================
st.sidebar.header("📁 Upload de Arquivos")
file_sensorial = st.sidebar.file_uploader("1. Respostas Consolidadas (Sensorial)", type=['csv'])
file_emocoes = st.sidebar.file_uploader("2. Distribuição de Emoções (Frames)", type=['csv'])

st.sidebar.divider()
st.sidebar.header("⚙️ Calibragem do Algoritmo")

limiar_neutro = st.sidebar.slider("⏳ Descarte de Face Neutra", 0.50, 0.95, 0.80, 0.05)
v_max = st.sidebar.slider("🎚️ Teto Fisiológico (V_max)", 0.10, 1.00, 0.50, 0.05)

if file_sensorial is None:
    st.info("💡 Carregue o arquivo CSV Sensorial (Consolidado) para desbloquear as análises estatísticas.")
    if file_emocoes is None:
        st.stop()

if file_sensorial is not None:
    df_s = load_consolidado(file_sensorial)
if file_emocoes is not None:
    df_e = load_and_parse_emocoes(file_emocoes)

if df_s is not None and df_e is not None:
    df_confianca_global, df_frames_calculados, df_emo_token = calcular_metricas_confianca(df_e, df_s, limiar_neutro, v_max)

st.sidebar.divider()
st.sidebar.header("🔍 Filtros Ativos")

if df_s is not None:
    exps_disponiveis = df_s['Experimento'].unique().tolist()
    selected_exps = st.sidebar.multiselect("Filtrar Experimentos (Aba Sensorial)", exps_disponiveis, default=exps_disponiveis)
    df_s_filtered = df_s[df_s['Experimento'].isin(selected_exps)]

df_vid = pd.DataFrame()
if df_e is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Seleção de Amostra Individual (Vídeos)")
    exps_e = df_e['Experimento_Extraido'].unique()
    if len(exps_e) > 0:
        sel_exp_e = st.sidebar.selectbox("Experimento (Vídeo)", exps_e)
        sujeitos_e = df_e[df_e['Experimento_Extraido'] == sel_exp_e]['Token'].unique()
        sel_sujeito_e = st.sidebar.selectbox("Sujeito (Token)", sujeitos_e)
        aspectos_e = df_e[(df_e['Experimento_Extraido'] == sel_exp_e) & (df_e['Token'] == sel_sujeito_e)]['Aspecto'].unique()
        sel_aspecto_e = st.sidebar.selectbox("Aspecto (Vídeo)", aspectos_e)
        
        df_vid = df_e[(df_e['Experimento_Extraido'] == sel_exp_e) & (df_e['Token'] == sel_sujeito_e) & (df_e['Aspecto'] == sel_aspecto_e)].sort_values('frame_num').reset_index(drop=True)

melted_df = pd.DataFrame()
if df_s is not None and not df_s_filtered.empty:
    carac_cols = [col for col in df_s_filtered.columns if col.startswith('CARAC_')]
    if carac_cols:
        melted_df = df_s_filtered.melt(id_vars=['Experimento'], value_vars=carac_cols, var_name='Characteristic', value_name='Score')
        melted_df['Aspect'] = melted_df['Characteristic'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else x)
        melted_df['Score'] = pd.to_numeric(melted_df['Score'], errors='coerce').dropna()

# ==========================================
# ESTRUTURA DE ABAS PRINCIPAIS
# ==========================================
abas = ["🔮 Confiança Fisiológica", "📊 Escalas", "💬 Qualitativo", "🔬 Estatística", "🎭 Biometria", "🎞️ Frames", "📄 Relatório PDF"]
tab_fusao, tab_carac, tab_perg, tab_estat, tab_biometria, tab_frames, tab_pdf = st.tabs(abas)

# ------------------------------------------
# ABA 1: FUSÃO E CONFIANÇA FISIOLÓGICA
# ------------------------------------------
with tab_fusao:
    st.header("Análise de Consistência e Fusão Multimodal")
    
    if df_confianca_global.empty:
        st.warning("⚠️ Para visualizar a consistência da base, é necessário fazer o upload de ambos os arquivos CSV (Sensorial e Emoções).")
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        conf_valida = df_confianca_global[~df_confianca_global['Sinal_Insuficiente']]
        
        col_m1.metric("Média de Confiança Global", f"{conf_valida['Confianca_Final_%'].mean():.2f}%")
        col_m2.metric("Vídeos Analisados", len(df_confianca_global))
        col_m3.metric("Vídeos 100% Neutros (Descartados)", df_confianca_global['Sinal_Insuficiente'].sum())
        
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
# ABA 2: CARACTERÍSTICAS (ESCALAS HEDÔNICAS E ATITUDE DE COMPRA)
# ------------------------------------------
with tab_carac:
    st.header("Análise de Atributos Sensoriais e Intenção de Compra")
    st.info("💡 **Dica:** Clique no título do gráfico ou nos nomes dos eixos para editá-los. Para levar ao PowerPoint, clique no ícone da câmera (Download) e arraste a imagem SVG gerada para o seu slide!")
    
    # 1. GRÁFICOS HEDÔNICOS (CARAC_)
    if not melted_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Média de Avaliação (Hedônica)")
            df_mean = melted_df.groupby(['Aspect', 'Experimento'])['Score'].mean().reset_index()
            fig1 = px.bar(df_mean, x='Aspect', y='Score', color='Experimento', barmode='group', 
                          title='Média Hedônica por Aspecto Sensorial')
            fig1.update_yaxes(range=[0, 9])
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
            
        with c2:
            st.subheader("Distribuição das Notas (Hedônica)")
            fig2 = px.box(melted_df, x='Aspect', y='Score', color='Experimento', 
                          title='Boxplot: Distribuição de Notas por Aspecto')
            fig2.update_yaxes(range=[0, 9])
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    # 2. GRÁFICOS DE ATITUDE DE COMPRA
    if df_s is not None and not df_s_filtered.empty:
        # Busca especificamente a coluna de Atitude de Compra
        compra_cols = [c for c in df_s_filtered.columns if 'atitude de comp' in c.lower() or 'compra' in c.lower()]
        
        if compra_cols:
            st.markdown("---")
            st.subheader("🛒 Atitude de Compra")
            
            # Derrete o dataframe para pegar as respostas
            melted_compra = df_s_filtered.melt(id_vars=['Experimento'], value_vars=compra_cols, var_name='Pergunta', value_name='Score')
            
            # Conversor Exato para a Escala de 5 pontos
            def mapear_escala_compra(val):
                if pd.isna(val): return np.nan
                val_str = str(val).lower().strip()
                
                if "certamente não compraria" in val_str or "certamente nao compraria" in val_str:
                    return 1.0
                elif "provavelmente não compraria" in val_str or "provavelmente nao compraria" in val_str:
                    return 2.0
                elif "dúvida" in val_str or "duvida" in val_str or "dúvidas" in val_str:
                    return 3.0
                elif "provavelmente compraria" in val_str:
                    return 4.0
                elif "certamente compraria" in val_str:
                    return 5.0
                else:
                    match = re.search(r'\d+', val_str)
                    if match: return float(match.group())
                    return np.nan
                
            melted_compra['Score_Num'] = melted_compra['Score'].apply(mapear_escala_compra)
            melted_compra = melted_compra.dropna(subset=['Score_Num'])
            
            if not melted_compra.empty:
                melted_compra['Pergunta_Curta'] = "Intenção de Compra"
                
                c3, c4 = st.columns(2)
                with c3:
                    df_mean_compra = melted_compra.groupby(['Pergunta_Curta', 'Experimento'])['Score_Num'].mean().reset_index()
                    fig3 = px.bar(df_mean_compra, x='Pergunta_Curta', y='Score_Num', color='Experimento', barmode='group', 
                                  title='Média: Atitude de Compra (1 a 5)')
                    fig3.update_yaxes(range=[0, 5], dtick=1) 
                    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)
                    
                with c4:
                    fig4 = px.box(melted_compra, x='Pergunta_Curta', y='Score_Num', color='Experimento', 
                                  title='Boxplot: Distribuição da Atitude de Compra')
                    fig4.update_yaxes(range=[0, 5.5], dtick=1)
                    st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.warning("⚠️ Não foi possível converter as respostas de compra para números (1 a 5). Verifique se a coluna está preenchida corretamente.")

# ------------------------------------------
# ABA 3: PERGUNTAS DO QUESTIONÁRIO
# ------------------------------------------
with tab_perg:
    st.header("Análise Qualitativa e Textual")
    if df_s is not None and not df_s_filtered.empty:
        raw_perg_cols = [col for col in df_s_filtered.columns if col.startswith('PERG_')]
        if raw_perg_cols:
            base_perg_dict = {}
            for col in raw_perg_cols:
                base_name = re.sub(r'\.\d+$', '', col)
                if base_name not in base_perg_dict:
                    base_perg_dict[base_name] = []
                base_perg_dict[base_name].append(col)
                
            pergunta_selecionada = st.selectbox("Selecione a pergunta para analisar:", list(base_perg_dict.keys()), format_func=lambda x: x.replace('PERG_', ''))
            colunas_reais = base_perg_dict[pergunta_selecionada]
            
            base_cols = ['Experimento']
            if 'Participante_Email' in df_s_filtered.columns: base_cols.append('Participante_Email')
            if 'Token' in df_s_filtered.columns: base_cols.append('Token')
                
            df_perg = df_s_filtered[base_cols + colunas_reais].melt(
                id_vars=base_cols, value_vars=colunas_reais, value_name='Resposta'
            ).dropna()
            df_perg = df_perg[df_perg['Resposta'].astype(str).str.strip() != ""]
            
            if not df_confianca_global.empty and 'Token' in df_perg.columns:
                df_conf_token = df_confianca_global.groupby('Token')['Confianca_Final_%'].mean().round(2).reset_index()
                df_conf_token.rename(columns={'Confianca_Final_%': 'Confiança_Fisiológica(%)'}, inplace=True)
                df_perg = pd.merge(df_perg, df_conf_token, on='Token', how='left')
            
            if not df_perg.empty:
                qtd_opcoes = df_perg['Resposta'].nunique()
                
                if qtd_opcoes > 15: 
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
                            
                            if not df_emo_token.empty and 'Token' in df_exp.columns:
                                df_emo_text = pd.merge(df_exp[['Token']].drop_duplicates(), df_emo_token, on='Token', how='inner')
                                if not df_emo_text.empty:
                                    st.markdown("##### 🎭 Perfil Emocional Predominante dos Respondentes")
                                    emo_means = df_emo_text[EMO_COLS].mean().sort_values(ascending=False).head(4)
                                    cols_metric = st.columns(4)
                                    for i, (emo, val) in enumerate(emo_means.items()):
                                        cols_metric[i].metric(label=emo.capitalize(), value=f"{val*100:.1f}%")
                                    st.divider()

                            tx1, tx2 = st.columns([2, 1])
                            with tx1:
                                if WordCloud is not None and texto_completo.strip():
                                    wc = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(texto_completo)
                                    fig_w, ax_w = plt.subplots()
                                    ax_w.imshow(wc, interpolation='bilinear')
                                    ax_w.axis('off')
                                    st.pyplot(fig_w)
                            with tx2:
                                st.dataframe(df_freq, hide_index=True, use_container_width=True)
                                
                            st.markdown("**Respostas na íntegra:**")
                            colunas_tabela = [c for c in ['Participante_Email', 'Token', 'Resposta', 'Confiança_Fisiológica(%)'] if c in df_exp.columns]
                            st.dataframe(
                                df_exp[colunas_tabela], 
                                hide_index=True, 
                                use_container_width=True,
                                column_config={
                                    "Confiança_Fisiológica(%)": st.column_config.ProgressColumn("Confiança (%)", format="%.2f", min_value=0, max_value=100)
                                } if 'Confiança_Fisiológica(%)' in df_exp.columns else None
                            )
                
                else: 
                    df_perg['Resp_Format'] = df_perg['Resposta'].astype(str).apply(lambda x: textwrap.fill(x, width=35))
                    
                    df_counts = df_perg.groupby(['Resp_Format', 'Experimento']).size().reset_index(name='Votos')
                    fig_c = px.bar(df_counts, y='Resp_Format', x='Votos', color='Experimento', barmode='group', 
                                   orientation='h', title='Frequência de Escolhas por Categoria')
                    fig_c.update_layout(yaxis={'categoryorder':'total ascending'})
                    
                    st.info("💡 **Gráfico Editável:** Clique no título ou eixo para alterar.")
                    st.plotly_chart(fig_c, use_container_width=True, config=PLOTLY_CONFIG)
                    
                    if 'Confiança_Fisiológica(%)' in df_perg.columns:
                        st.markdown("##### 🎭 Validação Biométrica por Categoria Escolhida")
                        df_cat_conf = df_perg.groupby('Resposta').agg(
                            Votos=('Resposta', 'count'),
                            Confiança_Média=('Confiança_Fisiológica(%)', 'mean')
                        ).reset_index()
                        df_cat_conf['Confiança_Média'] = df_cat_conf['Confiança_Média'].round(2)
                        st.dataframe(
                            df_cat_conf.sort_values(by='Votos', ascending=False), 
                            hide_index=True, 
                            use_container_width=True,
                            column_config={"Confiança_Média": st.column_config.ProgressColumn("Confiança Global Média (%)", format="%.2f", min_value=0, max_value=100)}
                        )
        else:
            st.info("Nenhuma coluna iniciando em 'PERG_' encontrada.")

# ------------------------------------------
# ABA 4: ANÁLISE ESTATÍSTICA AVANÇADA
# ------------------------------------------
with tab_estat:
    st.header("Análise de Significância Estatística")
    if not melted_df.empty:
        if len(selected_exps) >= 2:
            pares = list(combinations(selected_exps, 2))
            for exp1, exp2 in pares:
                st.markdown(f"#### 🆚 Comparação: `{exp1}` vs `{exp2}`")
                resultados = []
                
                for asp in melted_df['Aspect'].unique():
                    d1 = melted_df[(melted_df['Experimento'] == exp1) & (melted_df['Aspect'] == asp)]['Score']
                    d2 = melted_df[(melted_df['Experimento'] == exp2) & (melted_df['Aspect'] == asp)]['Score']
                    
                    if len(d1) > 1 and len(d2) > 1:
                        t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
                        if pd.isna(t_stat): continue
                        sig = "Sim (p<0.05) 🟢" if p_val < 0.05 else "Não 🔴"
                        resultados.append({"Aspecto": asp, f"Média ({exp1})": round(d1.mean(), 2), f"Média ({exp2})": round(d2.mean(), 2), "Valor p": round(p_val, 4), "Significativo?": sig})
                        
                if resultados:
                    st.dataframe(pd.DataFrame(resultados), hide_index=True, use_container_width=True)
        else:
            st.info("💡 Selecione pelo menos dois experimentos na barra lateral.")

# ------------------------------------------
# ABA 5: BIOMETRIA DO VÍDEO
# ------------------------------------------
with tab_biometria:
    st.header("Respostas Emocionais Dinâmicas")
    if not df_vid.empty:
        emocoes_presentes = [e for e in EMO_COLS if e in df_vid.columns]
        if emocoes_presentes:
            b1, b2 = st.columns([1, 2])
            with b1:
                st.subheader("Distribuição Média")
                df_m = df_vid[emocoes_presentes].mean().reset_index()
                df_m.columns = ['Emoção', 'Média']
                fig_m = px.bar(df_m, x='Emoção', y='Média', color='Emoção', range_y=[0,1], text_auto='.1%', title='Média Emocional da Sessão')
                st.plotly_chart(fig_m, use_container_width=True, config=PLOTLY_CONFIG)
            with b2:
                st.subheader("Série Temporal")
                fig_l = px.line(df_vid, x='frame_num', y=emocoes_presentes, labels={'frame_num': 'Frame'}, title='Variação Emocional ao Longo do Tempo')
                st.plotly_chart(fig_l, use_container_width=True, config=PLOTLY_CONFIG)

# ------------------------------------------
# ABA 6: LINHA DO TEMPO DE FRAMES
# ------------------------------------------
with tab_frames:
    st.header("Inspeção Frame a Frame")
    if not df_vid.empty:
        emocoes_presentes = [e for e in EMO_COLS if e in df_vid.columns]
        idx = st.slider("Selecione o Frame:", 0, len(df_vid)-1, 0)
        row = df_vid.iloc[idx]
        
        f1, f2 = st.columns([1, 1.5])
        with f1:
            st.metric("Número do Frame", int(row['frame_num']))
            img_path = str(row['file'])
            if os.path.exists(img_path): st.image(Image.open(img_path), use_container_width=True)
            else: st.warning(f"⚠️ Imagem não encontrada:\n{img_path}")
        with f2:
            if emocoes_presentes:
                df_f = pd.DataFrame(list({e: row[e] for e in emocoes_presentes}.items()), columns=['Emoção', 'Probabilidade'])
                fig_f = px.bar(df_f, x='Emoção', y='Probabilidade', color='Emoção', range_y=[0,1], text_auto='.1%', title=f'Expressão Facial (Frame {int(row["frame_num"])})')
                st.plotly_chart(fig_f, use_container_width=True, config=PLOTLY_CONFIG)

# ------------------------------------------
# ABA 7: EXPORTAÇÃO PARA ARTIGO CIENTÍFICO
# ------------------------------------------
with tab_pdf:
    st.header("Exportação para Publicação Acadêmica")
    
    if FPDF is None:
        st.error("⚠️ **Biblioteca FPDF não encontrada.** Instale com `pip install fpdf`.")
    else:
        if not melted_df.empty:
            pdf_bytes = gerar_pdf_academico(
                melted_df, 
                df_confianca_global, 
                df_frames_calculados, 
                df_emo_token,
                v_max, 
                limiar_neutro, 
                selected_exps, 
                df_s_filtered
            )
            
            st.success("O relatório acadêmico estilizado está pronto para download.")
            st.download_button(
                label="📥 Baixar Relatório Científico Completo (PDF)",
                data=pdf_bytes,
                file_name="Relatorio_Sensorial_Multimodal.pdf",
                mime="application/pdf"
            )