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

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E STOPWORDS
# ==========================================
st.set_page_config(page_title="Plataforma Sensorial JAR & Biometria", layout="wide")
st.title("Análise Sensorial: Escala do Ideal (JAR 1-9) e Validação Biométrica")
st.markdown("---")

EMO_COLS = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']

STOPWORDS_PT = set([
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", 
    "para", "com", "um", "uma", "uns", "umas", "que", "eu", "ele", "ela", "nós", "eles", 
    "elas", "me", "se", "por", "como", "mas", "ou", "ao", "aos", "mais", "menos", "muito", 
    "pouco", "tem", "foi", "ser", "está", "são", "isso", "esse", "este", "esta", "essa", 
    "aquilo", "não", "sim", "já", "só", "também", "nem", "quando", "qual", "quais", "quem", 
    "gostei", "menos", "mais", "achei"
])

# ==========================================
# FUNÇÕES AUXILIARES E DE PROCESSAMENTO
# ==========================================
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, sep=';', engine='python')
        if len(df.columns) == 1:
            file.seek(0)
            df = pd.read_csv(file, sep=',', engine='python')
        return df
    return pd.read_excel(file)

def extract_from_path(path_str):
    path_str = str(path_str).replace('\\', '/')
    parts = path_str.split('/')
    try:
        token = parts[-3]
        aspecto = parts[-2].lower().strip()
        return pd.Series([token, aspecto])
    except:
        return pd.Series(["Desconhecido", "Desconhecido"])

def mapear_aspecto_inteligente(coluna, pastas_disponiveis):
    """
    Função com Fallback Semântico. Garante que mesmo que o script de extração 
    corte o nome da pasta (ex: "atributos_de_aparencia"), o pareamento funcione.
    """
    coluna_limpa = unicodedata.normalize('NFKD', coluna.lower()).encode('ASCII', 'ignore').decode('utf-8')
    partes = coluna.split('_')
    
    if len(partes) >= 3:
        grupo = partes[1].lower().strip()
        grupo = unicodedata.normalize('NFKD', grupo).encode('ASCII', 'ignore').decode('utf-8')
        grupo = grupo.replace(' ', '_')
        grupo = re.sub(r'[^a-z0-9_]', '', grupo)
        
        # 1. Tenta achar a pasta com o nome idêntico
        if grupo in pastas_disponiveis:
            return grupo
            
        # 2. Tenta achar a pasta por substring (caso tenha cortado um pedaço)
        for pasta in pastas_disponiveis:
            if pasta in grupo or grupo in pasta:
                return pasta
                
    # 3. Fallback Semântico: Se tudo falhar, tenta achar pela categoria
    if 'aparencia' in coluna_limpa or 'textura' in coluna_limpa or 'cor' in coluna_limpa or 'consistencia' in coluna_limpa or 'firmeza' in coluna_limpa:
        for p in pastas_disponiveis:
            if 'aparencia' in p or 'textura' in p: return p
            
    if 'cheiro' in coluna_limpa or 'aroma' in coluna_limpa:
        for p in pastas_disponiveis:
            if 'cheiro' in p or 'aroma' in p: return p
            
    if 'sabor' in coluna_limpa or 'docura' in coluna_limpa or 'acidez' in coluna_limpa or 'amargor' in coluna_limpa:
        for p in pastas_disponiveis:
            if 'sabor' in p: return p

    # Se nada bater, retorna o nome processado pra debug
    return grupo if len(partes) >= 3 else coluna_limpa

def remover_acentos(txt):
    return unicodedata.normalize('NFKD', str(txt)).encode('ASCII', 'ignore').decode('ASCII')

def formatar_texto_celula(texto, max_len=20):
    limpo = remover_acentos(str(texto).capitalize())
    return limpo[:max_len] + ".." if len(limpo) > max_len else limpo

# ==========================================
# MOTOR DE GERAÇÃO DE PDF (DESIGN ACADÊMICO)
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

def gerar_pdf_academico(melted_df_academico, df_confianca_academico, df_frames, df_emo_token, v_max_val, limiar_val, df_s_filtered):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # CAPA
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 10, remover_acentos("RELATORIO DE PESQUISA SENSORIAL"), ln=True, align="C")
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, remover_acentos("Estatistica Hedonica e Validacao Biometrica Multimodal (Escala JAR)"), ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, remover_acentos("Gerado automaticamente pela Plataforma de IA"), ln=True, align="C")
    pdf.ln(20)
    
    # 1. METODOLOGIA
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, remover_acentos(" 1. Parametros Metodologicos e Amostra"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    participantes = df_s_filtered['Token'].nunique() if 'Token' in df_s_filtered.columns else len(df_s_filtered)
    pdf.ln(2)
    pdf.cell(0, 6, remover_acentos(f" Tamanho da Amostra (N): {participantes} Sessoes Sensoriais"), ln=True)
    pdf.cell(0, 6, remover_acentos(f" Teto Fisiologico de Saturacao (V_max): {v_max_val}"), ln=True)
    pdf.cell(0, 6, remover_acentos(f" Limiar de Descarte Cognitivo (Face Neutra): {limiar_val}"), ln=True)
    pdf.ln(10)
    
    # 2. ESTATÍSTICA HEDÔNICA JAR
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, remover_acentos(" 2. Estatisticas Descritivas (Escala JAR 1-9)"), ln=True, fill=True)
    pdf.ln(3)
    
    if not melted_df_academico.empty:
        header_tabela(pdf, ["Aspecto", "N", "Media +/- DP", "Mediana", "Min-Max"], [50, 15, 45, 30, 30])
        stats_hed = melted_df_academico.groupby('Atributo')['Nota'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index()
        fill = False
        for _, row in stats_hed.iterrows():
            pdf.set_fill_color(245, 245, 245)
            dp = row['std'] if pd.notnull(row['std']) else 0.0
            pdf.cell(50, 8, formatar_texto_celula(row['Atributo'], 25), border=1, fill=fill)
            pdf.cell(15, 8, str(int(row['count'])), border=1, align="C", fill=fill)
            pdf.cell(45, 8, f"{row['mean']:.2f} +/- {dp:.2f}", border=1, align="C", fill=fill)
            pdf.cell(30, 8, f"{row['median']:.1f}", border=1, align="C", fill=fill)
            pdf.cell(30, 8, f"{row['min']:.1f} - {row['max']:.1f}", border=1, align="C", fill=fill)
            pdf.ln()
            fill = not fill
        pdf.ln(5)
        
        # Gráfico Embutido
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=melted_df_academico, x='Atributo', y='Nota', errorbar=None, palette='viridis', ax=ax)
        ax.axhline(y=5, color='r', linestyle='--', label='Ideal (5)')
        ax.set_ylim(0, 9)
        ax.set_title(remover_acentos("Media JAR por Aspecto"))
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img1:
            fig.savefig(tmp_img1.name, dpi=300)
            pdf.image(tmp_img1.name, w=150)
            os.unlink(tmp_img1.name)
        plt.close(fig)
        pdf.ln(5)

    # 4. PERFIL FISIOLÓGICO DA REDE NEURAL
    pdf.add_page()
    if not df_frames.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 4. Perfil Fisiologico Medio (Visao Computacional)"), ln=True, fill=True)
        pdf.ln(3)
        # 6 emoções principais para não estourar a largura da página A4
        emocoes_alvo = ['neutral', 'happy', 'disgusted', 'surprised', 'angry', 'sad']
        emocoes_presentes = [e for e in emocoes_alvo if e in df_frames.columns]
        
        if emocoes_presentes:
            col_names = ["Aspecto"] + [e.capitalize() for e in emocoes_presentes]
            widths = [45] + [22] * len(emocoes_presentes)
            header_tabela(pdf, col_names, widths)
            
            stats_emo = df_frames.groupby('Coluna_Original')[emocoes_presentes].mean().reset_index()
            fill = False
            for _, row in stats_emo.iterrows():
                pdf.set_fill_color(245, 245, 245)
                nome_asp = row['Coluna_Original'].split('_')[-1]
                pdf.cell(45, 8, formatar_texto_celula(nome_asp, 25), border=1, fill=fill)
                for emo in emocoes_presentes:
                    pdf.cell(22, 8, f"{(row[emo]*100):.1f}%", border=1, align="C", fill=fill)
                pdf.ln()
                fill = not fill
            pdf.ln(8)

    # 5. VALIDAÇÃO BIOMÉTRICA DE CONFIANÇA
    if not df_confianca_academico.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 5. Validacao Biometrica (Consistencia Multimodal JAR)"), ln=True, fill=True)
        pdf.ln(3)
        header_tabela(pdf, ["Aspecto", "Frames Totais", "Sinal Expressivo", "Confianca Global"], [55, 35, 45, 45])
        
        stats_conf = df_confianca_academico.groupby('Atributo').agg(
            Frames_Tot=('Total_Frames', 'sum'),
            Frames_Exp=('Frames_Expressivos', 'sum'),
            Confianca_Media=('Confianca_Final_%', 'mean')
        ).reset_index()
        
        fill = False
        for _, row in stats_conf.iterrows():
            pdf.set_fill_color(245, 245, 245)
            pct_exp = (row['Frames_Exp'] / row['Frames_Tot']) * 100 if row['Frames_Tot'] > 0 else 0
            
            pdf.cell(55, 8, formatar_texto_celula(row['Atributo'], 28), border=1, fill=fill)
            pdf.cell(35, 8, f"{int(row['Frames_Tot'])}", border=1, align="C", fill=fill)
            pdf.cell(45, 8, f"{pct_exp:.1f}%", border=1, align="C", fill=fill) 
            pdf.cell(45, 8, f"{row['Confianca_Media']:.2f}%", border=1, align="C", fill=fill)
            pdf.ln()
            fill = not fill
        pdf.ln(8)

    # 6. EXTRAÇÃO QUALITATIVA
    raw_perg_cols = [col for col in df_s_filtered.columns if col.startswith('PERG_')]
    if raw_perg_cols:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, remover_acentos(" 6. Analise Qualitativa de Questionario"), ln=True, fill=True)
        pdf.ln(4)
        for col in raw_perg_cols:
            respostas_validas = df_s_filtered[col].dropna()
            if respostas_validas.empty: continue
            
            nome_perg_bruto = col.replace('PERG_', '')
            nome_perg_bruto = re.sub(r'\.\d+$', '', nome_perg_bruto)
            nome_perg = remover_acentos(nome_perg_bruto.split('_')[-1] if '_' in nome_perg_bruto else nome_perg_bruto)
            
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, remover_acentos(f"Questao: {nome_perg}"), ln=True)
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
                    header_tabela(pdf, ["Opcao Escolhida", "Votos", "%", "Confianca Media"], [95, 20, 20, 40])
                else:
                    header_tabela(pdf, ["Opcao Escolhida", "Votos", "%"], [135, 20, 20])
                
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
# SIDEBAR - UPLOADS E PARÂMETROS
# ==========================================
st.sidebar.header("Upload de Arquivos")
file_sensorial = st.sidebar.file_uploader("1. Dados Sensoriais (Escala Ideal 1-9)", type=['csv', 'xlsx'])
file_emocoes = st.sidebar.file_uploader("2. Características de IA (Frames)", type=['csv', 'xlsx'])

st.sidebar.divider()
st.sidebar.header("Calibragem do Algoritmo")
limiar_neutro = st.sidebar.slider("Descarte de Face Neutra", 0.50, 0.95, 0.80, 0.05, 
                                  help="Ignora frames onde a probabilidade de rosto neutro é superior ao limite.")
v_max = st.sidebar.slider("Teto Fisiológico (V_max)", 0.10, 1.00, 0.50, 0.05,
                          help="Intensidade facial máxima assumida para notas extremas.")

if file_sensorial and file_emocoes:
    df_s = load_data(file_sensorial)
    df_e = load_data(file_emocoes)
    
    df_e.columns = df_e.columns.str.strip().str.lower()
    if 'caminho' in df_e.columns: df_e = df_e.rename(columns={'caminho': 'file'})
    if 'path' in df_e.columns: df_e = df_e.rename(columns={'path': 'file'})
        
    df_e[['Token', 'Aspecto_Pasta']] = df_e['file'].apply(extract_from_path)

    jar_cols = [
        'CARAC_ATRIBUTOS DE CHEIRO_Intensidade do Aroma',
        'CARAC_ATRIBUTOS DE APARÊNCIA E TEXTURA_Cor do Iogurte',
        'CARAC_ATRIBUTOS DE APARÊNCIA E TEXTURA_Consistência do Iogurte (Corpo/Cremosidade)',
        'CARAC_ATRIBUTOS DE APARÊNCIA E TEXTURA_Firmeza das Bolinhas (Resistência ao Estouro)',
        'CARAC_ATRIBUTOS DE SABOR_Intensidade do Sabor do Iogurte',
        'CARAC_ATRIBUTOS DE SABOR_Doçura',
        'CARAC_ATRIBUTOS DE SABOR_Acidez',
        'CARAC_ATRIBUTOS DE SABOR_Amargor / Adstringência (“Amarra a boca” igual banana verde)'
    ]
    
    jar_cols_presentes = [col for col in jar_cols if col in df_s.columns]
    
    if not jar_cols_presentes:
        st.error("As colunas com os atributos JAR não foram encontradas no arquivo sensorial.")
        st.stop()

    tab_biometria, tab_penalty, tab_estatistica, tab_pdf = st.tabs([
        "🧠 1. Confiança Biométrica", 
        "📉 2. Análise de Penalidade",
        "📊 3. Estatísticas Descritivas",
        "📄 4. Relatório PDF"
    ])

    df_metricas = pd.DataFrame()
    df_merged = pd.DataFrame()
    df_emo_token = pd.DataFrame()
    df_melted_descritivo = pd.DataFrame()

    with tab_biometria:
        st.header("Consistência entre Notas Declaradas e Emoções")
        if 'Token' not in df_s.columns:
            st.error("A coluna 'Token' é obrigatória no arquivo Sensorial.")
        else:
            df_notas = df_s[['Token'] + jar_cols_presentes].copy()
            df_notas_long = df_notas.melt(id_vars=['Token'], value_vars=jar_cols_presentes, var_name='Coluna_Original', value_name='Nota_Hedonica')
            
            pastas_unicas = df_e['Aspecto_Pasta'].dropna().unique().tolist()
            df_notas_long['Aspecto_Pasta'] = df_notas_long['Coluna_Original'].apply(lambda x: mapear_aspecto_inteligente(x, pastas_unicas))
            
            df_e['Token'] = df_e['Token'].astype(str).str.strip().str.lower()
            df_e['Aspecto_Pasta'] = df_e['Aspecto_Pasta'].astype(str).str.strip().str.lower()
            
            df_notas_long['Token'] = df_notas_long['Token'].astype(str).str.strip().str.lower()
            df_notas_long['Aspecto_Pasta'] = df_notas_long['Aspecto_Pasta'].astype(str).str.strip().str.lower()
            
            df_merged = pd.merge(df_e, df_notas_long, on=['Token', 'Aspecto_Pasta'], how='inner')
            
            if df_merged.empty:
                st.warning("⚠️ Nenhuma correspondência exata encontrada.")
            else:
                df_merged['Nota_Hedonica'] = pd.to_numeric(df_merged['Nota_Hedonica'], errors='coerce')
                df_merged = df_merged.dropna(subset=['Nota_Hedonica']).copy()
                
                # CÁLCULO H_NORM (JAR 1 a 9 -> Ideal 5)
                df_merged['H_norm'] = 1 - (abs(df_merged['Nota_Hedonica'] - 5) / 2)
                
                e_pos = df_merged['happy']
                e_neg = df_merged[['contempt', 'angry', 'disgusted', 'fearful', 'sad']].sum(axis=1)
                df_merged['V_bruto'] = e_pos - e_neg
                df_merged['V'] = np.clip(df_merged['V_bruto'] / v_max, -1.0, 1.0)
                
                P = df_merged[EMO_COLS] + 1e-9
                df_merged['Entropia'] = -np.sum(P * np.log2(P), axis=1)
                df_merged['W'] = 1 - (df_merged['Entropia'] / 3)
                
                df_merged['M'] = 1 - (abs(df_merged['H_norm'] - df_merged['V']) / 2)
                df_merged['Confianca_Frame'] = df_merged['M'] * df_merged['W']
                
                # SEPARAÇÃO DE BASE (Total de frames) e EXPRESSIVOS
                df_base = df_merged.groupby(['Token', 'Coluna_Original']).agg(
                    Nota_Declarada=('Nota_Hedonica', 'first'),
                    Total_Frames=('file', 'count')
                ).reset_index()
                
                df_expressivo = df_merged[df_merged['neutral'] < limiar_neutro].copy()
                df_metricas_exp = df_expressivo.groupby(['Token', 'Coluna_Original']).agg(
                    Frames_Expressivos=('file', 'count'),
                    Confianca_Final=('Confianca_Frame', 'mean')
                ).reset_index()
                
                df_metricas = pd.merge(df_base, df_metricas_exp, on=['Token', 'Coluna_Original'], how='left')
                df_metricas['Frames_Expressivos'] = df_metricas['Frames_Expressivos'].fillna(0).astype(int)
                
                df_metricas['Confianca_Final_%'] = (df_metricas['Confianca_Final'] * 100).round(2)
                df_metricas['Atributo'] = df_metricas['Coluna_Original'].apply(lambda x: x.split('_')[-1])
                
                df_emo_token = df_merged.groupby('Token')[EMO_COLS].mean().reset_index()
                
                st.subheader("Médias por Característica (Geral)")
                df_medias_atributo = df_metricas.groupby('Atributo').agg(
                    Nota_Media=('Nota_Declarada', 'mean'),
                    Confianca_Media=('Confianca_Final_%', 'mean')
                ).reset_index().sort_values(by='Confianca_Media', ascending=False)
                
                st.dataframe(df_medias_atributo, hide_index=True, use_container_width=True)
                
                st.subheader("Detalhamento por Juiz e Atributo")
                metricas_apresentacao = df_metricas[['Token', 'Atributo', 'Nota_Declarada', 'Frames_Expressivos', 'Confianca_Final_%']].sort_values(by='Confianca_Final_%', ascending=False)
                st.dataframe(metricas_apresentacao, hide_index=True, use_container_width=True)

    with tab_penalty:
        st.header("Análise de Penalidade (Penalty Analysis)")
        target_options = [col for col in df_s.columns if "PERG" in col or "Sabor" in col or "Aceitação" in col or "Global" in col]
        if not target_options:
            target_options = df_s.columns.tolist()
            
        sabor_geral_col = st.selectbox("Selecione a Pergunta Hedônica de Aceitação Global:", target_options)

        if st.button("Executar Análise de Penalidade"):
            penalty_data = []
            for col in jar_cols_presentes:
                df_temp = df_s[[col, sabor_geral_col]].dropna()
                df_temp[sabor_geral_col] = pd.to_numeric(df_temp[sabor_geral_col], errors='coerce')
                df_temp = df_temp.dropna()
                
                total_respostas = len(df_temp)
                if total_respostas == 0: continue
                
                media_ideal = df_temp[df_temp[col] == 5][sabor_geral_col].mean()
                if pd.isna(media_ideal): continue
                atributo_nome = col.split('_')[-1]

                df_abaixo = df_temp[df_temp[col] < 5]
                if len(df_abaixo) > 0:
                    pct_abaixo = (len(df_abaixo) / total_respostas) * 100
                    penalty_data.append({"Atributo": f"{atributo_nome} (Fraco/Pouco)", "% de Juízes": pct_abaixo, "Mean Drop (Penalidade)": media_ideal - df_abaixo[sabor_geral_col].mean()})

                df_acima = df_temp[df_temp[col] > 5]
                if len(df_acima) > 0:
                    pct_acima = (len(df_acima) / total_respostas) * 100
                    penalty_data.append({"Atributo": f"{atributo_nome} (Forte/Muito)", "% de Juízes": pct_acima, "Mean Drop (Penalidade)": media_ideal - df_acima[sabor_geral_col].mean()})

            df_penalty = pd.DataFrame(penalty_data)
            if not df_penalty.empty:
                df_penalty['Mean Drop (Penalidade)'] = df_penalty['Mean Drop (Penalidade)'].apply(lambda x: max(x, 0))
                fig = px.scatter(df_penalty, x="% de Juízes", y="Mean Drop (Penalidade)", text="Atributo", size="% de Juízes", color="Mean Drop (Penalidade)", color_continuous_scale="Reds")
                fig.update_traces(textposition='top center')
                fig.add_hline(y=1.0, line_dash="dash", line_color="red")
                fig.add_vline(x=20, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_penalty.sort_values(by="Mean Drop (Penalidade)", ascending=False).round(2), hide_index=True, use_container_width=True)

    with tab_estatistica:
        st.header("Distribuição Geral das Respostas JAR")
        df_descritivo = df_s[jar_cols_presentes].copy()
        df_descritivo.columns = [col.split('_')[-1] for col in df_descritivo.columns]
        df_melted_descritivo = df_descritivo.melt(var_name='Atributo', value_name='Nota').dropna()
        
        fig_box = px.box(df_melted_descritivo, x='Atributo', y='Nota', color='Atributo', points="all", template="plotly_white")
        fig_box.add_hline(y=5, line_dash="dash", line_color="#1a9850", annotation_text="Ideal (5)", annotation_position="top left")
        fig_box.update_layout(height=600, showlegend=False, yaxis=dict(range=[0.5, 9.5], dtick=1))
        st.plotly_chart(fig_box, use_container_width=True)

    with tab_pdf:
        st.header("Exportação para Publicação Acadêmica")
        if FPDF is None:
            st.error("**Biblioteca FPDF não encontrada.** Instale rodando `pip install fpdf` no terminal Linux.")
        else:
            if not df_melted_descritivo.empty and not df_metricas.empty:
                pdf_bytes = gerar_pdf_academico(
                    df_melted_descritivo,
                    df_metricas,
                    df_merged,
                    df_emo_token,
                    v_max,
                    limiar_neutro,
                    df_s
                )
                st.success("O relatório acadêmico estilizado está pronto para download.")
                st.download_button(
                    label="Baixar Relatório Científico Completo (PDF)",
                    data=pdf_bytes,
                    file_name="Relatorio_Sensorial_JAR_Multimodal.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("O processamento precisa ser finalizado nas abas anteriores antes de gerar o PDF.")
else:
    st.info("Aguardando upload dos arquivos no painel lateral.")