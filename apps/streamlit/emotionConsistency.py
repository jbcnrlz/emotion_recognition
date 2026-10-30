import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="LDL Emotion Consistency Analyzer",
    page_icon="📊",
    layout="wide"
)

# Emotion categories constants
EMOTIONS_7 = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad']
EMOTIONS_8 = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
RATINGS_COLS = [f'emotion_{e}' for e in EMOTIONS_7]

# Synonym mappings for robustness across different model export headers
SYNONYM_MAP = {
    'disgust': 'disgusted',
    'fear': 'fearful',
    'surprise': 'surprised',
    'happiness': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'neutral': 'neutral',
    'contempt': 'contempt'
}

# Color palettes
HUMAN_COLOR = '#4f46e5'  # Deep Indigo
MODEL_PALETTE = [
    '#10b981',  # Emerald
    '#f59e0b',  # Amber
    '#06b6d4',  # Cyan
    '#ec4899',  # Pink
    '#8b5cf6',  # Violet
    '#f97316',  # Orange
    '#14b8a6',  # Teal
    '#e11d48',  # Rose
    '#3b82f6',  # Blue
    '#84cc16',  # Lime
    '#a855f7',  # Purple
    '#d97706',  # Dark Amber
]

# Label Distribution Learning (LDL) Core Mathematical Functions
def normalize_distribution(arr, eps=1e-12):
    """Ensures probability vector sums to 1.0 and applies epsilon smoothing to prevent zero-division."""
    arr = np.maximum(0.0, np.array(arr, dtype=np.float64))
    s = np.sum(arr)
    if s <= eps:
        return np.ones_like(arr) / len(arr)
    return arr / s

def compute_ldl_metrics(p, q, eps=1e-12):
    """Computes all 12 standard LDL comparative metrics between distributions P (Human) and Q (Model)."""
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    p_s = np.maximum(p, eps)
    q_s = np.maximum(q, eps)

    # 1. Euclidean Distance (L2)
    l2 = float(np.linalg.norm(p - q))

    # 2. Total Variation (L1)
    total_variation = float(0.5 * np.sum(np.abs(p - q)))

    # 3. Chebyshev Distance (L_inf)
    chebyshev = float(np.max(np.abs(p - q)))

    # 4. Kullback-Leibler Divergence: D_KL(P || Q) = ∑_{P_i > 0} P_i * ln(P_i / Q_i)
    # Sempre P = Anotação Humana (distribuição de referência) e Q = Output do Modelo
    p_mask = p > 0
    p_nz = p[p_mask]
    q_nz = np.maximum(q[p_mask], eps)
    kl = float(np.sum(p_nz * np.log(p_nz / q_nz)))

    # 5. Jensen-Shannon Divergence: JSD(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    # onde M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    m_p = np.maximum(m[p_mask], eps)
    q_mask = q > 0
    q_nz = q[q_mask]
    m_q = np.maximum(m[q_mask], eps)
    jsd = float(0.5 * np.sum(p_nz * np.log(p_nz / m_p)) + 0.5 * np.sum(q_nz * np.log(q_nz / m_q)))

    # 6. Cosine Similarity
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    cosine = float(np.dot(p, q) / (norm_p * norm_q)) if (norm_p > 0 and norm_q > 0) else 0.0

    # 7. Pearson Correlation Coefficient
    p_centered = p - np.mean(p)
    q_centered = q - np.mean(q)
    std_prod = np.linalg.norm(p_centered) * np.linalg.norm(q_centered)
    pearson = float(np.dot(p_centered, q_centered) / std_prod) if std_prod > 0 else 0.0

    # 8. Histogram Overlap / Intersection
    intersection = float(np.sum(np.minimum(p, q)))

    # 9. Fidelity (Bhattacharyya Coefficient)
    fidelity = float(np.sum(np.sqrt(p * q)))
    fidelity = min(1.0, max(0.0, fidelity))

    # 10. Bhattacharyya Distance
    bhattacharyya = float(-np.log(max(fidelity, eps)))

    # 11. Hellinger Distance
    hellinger = float(np.sqrt(max(0.0, 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))))

    # 12. Chi-Square Distance (χ²)
    denom = p + q
    chi2 = float(np.sum(np.divide((p - q) ** 2, denom, out=np.zeros_like(p, dtype=np.float64), where=denom > eps)))

    return {
        "Cosine": cosine,
        "Pearson": pearson,
        "Overlap": intersection,
        "Fidelity": fidelity,
        "JSD": jsd,
        "KL_Divergence": kl,
        "Total_Variation": total_variation,
        "Euclidean_L2": l2,
        "Chebyshev_Linf": chebyshev,
        "Hellinger": hellinger,
        "Bhattacharyya": bhattacharyya,
        "Chi2": chi2
    }

@st.cache_data
def scan_available_local_predictions():
    """Scans repository folders for candidate prediction CSV files containing emotion probabilities."""
    candidates = {}
    search_dirs = ["outputs", "data/predictions", "."]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            try:
                for fname in sorted(os.listdir(s_dir)):
                    if fname.endswith(".csv") and not fname.endswith("_labels.csv"):
                        fpath = os.path.join(s_dir, fname)
                        if os.path.isfile(fpath):
                            try:
                                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                                    first_line = fp.readline().lower()
                                    if any(e in first_line for e in ['happy', 'contempt', 'surprised', 'angry']):
                                        label = f"{fname} ({s_dir})" if s_dir != "." else fname
                                        candidates[label] = fpath
                            except Exception:
                                pass
            except Exception:
                pass
    return candidates

@st.cache_data
def scan_available_local_mappings():
    """Scans repository folders for cross-reference mapping CSV files."""
    candidates = {}
    search_dirs = [".", "data/annotations", "outputs"]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            try:
                for fname in sorted(os.listdir(s_dir)):
                    if fname.endswith(".csv") and "mapping" in fname.lower():
                        fpath = os.path.join(s_dir, fname)
                        if os.path.isfile(fpath):
                            label = f"{fname} ({s_dir})" if s_dir != "." else fname
                            candidates[label] = fpath
            except Exception:
                pass
    return candidates

def standardize_emotion_columns(df):
    """Standardizes column names based on synonyms (e.g. disgust -> disgusted)."""
    rename_dict = {}
    for col in df.columns:
        low = str(col).lower()
        if low in SYNONYM_MAP:
            rename_dict[col] = SYNONYM_MAP[low]
    if rename_dict:
        return df.rename(columns=rename_dict)
    return df

def standardize_mapping_dataframe(df_m):
    """Normalizes mapping file column names for robust joins."""
    df_copy = df_m.copy()
    col_map = {}
    for col in df_copy.columns:
        low = str(col).lower().strip()
        if low in ['user_study_filename', 'filename', 'image_name', 'study_file']:
            if 'image_filename' not in df_copy.columns:
                col_map[col] = 'image_filename'
        elif low in ['affectnet_file', 'dataset_file', 'matched_file', 'target_file']:
            if 'affectnet_file' not in df_copy.columns:
                col_map[col] = 'affectnet_file'
        elif low in ['code', 'stem', 'id', 'face_id']:
            if 'image_code' not in df_copy.columns:
                col_map[col] = 'image_code'
    if col_map:
        df_copy = df_copy.rename(columns=col_map)

    # Ensure affectnet_file and dataset_file both exist as synonyms
    if 'affectnet_file' not in df_copy.columns and 'dataset_file' in df_copy.columns:
        df_copy['affectnet_file'] = df_copy['dataset_file']
    if 'dataset_file' not in df_copy.columns and 'affectnet_file' in df_copy.columns:
        df_copy['dataset_file'] = df_copy['affectnet_file']

    return df_copy

# Main Header
st.title("📊 Emotion Distribution Consistency Analyzer (LDL)")
st.markdown("Quantitative and visual consistency analysis comparing **human prevailing consensus** with **model inferences**, both globally and **face-by-face**.")

# Sidebar (Data Ingestion & Configuration)
with st.sidebar:
    st.header("⚙️ Data & Settings")

    ratings_file = st.file_uploader(
        "1. Human Ratings CSV file",
        type=["csv"],
        help="CSV containing human multi-annotator prevailing scores (ratings_export.csv)."
    )

    st.subheader("2. 🤖 Model Estimations (Multi-Select)")
    uploaded_model_files = st.file_uploader(
        "Upload Model Estimations CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or multiple model prediction CSV files to compare side-by-side."
    )

    local_predictions = scan_available_local_predictions()
    default_selected_local = []
    for k in local_predictions.keys():
        if "simpleNewMarkup.csv" in k:
            default_selected_local = [k]
            break

    selected_local_keys = st.multiselect(
        "Or select from local model predictions:",
        options=list(local_predictions.keys()),
        default=default_selected_local if not uploaded_model_files else [],
        help="Pre-scanned prediction files from outputs/ and data/predictions/."
    )

    st.subheader("3. 🗺️ Cross-Reference Mapping")
    local_mappings = scan_available_local_mappings()

    mapping_options = ["None (Upload below)"] + list(local_mappings.keys())
    # Default selection if mapping_ratings_affectnet.csv exists
    default_map_idx = 0
    for idx, opt in enumerate(mapping_options):
        if "mapping_ratings_affectnet.csv" in opt:
            default_map_idx = idx
            break

    selected_mapping_choice = st.selectbox(
        "Select Mapping File:",
        options=mapping_options,
        index=default_map_idx,
        help="Select pre-existing mapping file generated by match_and_map.py."
    )

    mapping_file = st.file_uploader(
        "Or Upload Mapping CSV",
        type=["csv"],
        help="CSV cross-referencing user study faces with dataset files (match_and_map.py output)."
    )

    # Optional local faces directory to render actual face pictures
    default_faces_dir = "amostras_selecionadas" if os.path.exists("amostras_selecionadas") else ""
    faces_image_dir = st.text_input(
        "Face Images Folder (Optional - to view images):",
        value=default_faces_dir,
        help="Folder containing the face images (e.g. amostras_selecionadas) to render thumbnails."
    )

    st.divider()
    alignment_mode = st.radio(
        "Emotion Space Alignment:",
        options=["7 Shared Classes (Drop 'neutral')", "8 Classes (Impute Human 'neutral')"],
        index=0,
        help="7 Classes: Evaluates across shared 7 universal emotions. 8 Classes: Imputes human neutral residual."
    )

    st.divider()
    st.markdown("💡 **Tip**: Select multiple models to compare their performance rankings, grouped distributions, and radar profiles simultaneously against human annotations.")

# Dataset Loading & Resolution
models_dict = {}

# 1. Load uploaded models
if uploaded_model_files:
    for u_file in uploaded_model_files:
        try:
            m_df = pd.read_csv(u_file)
            m_name = u_file.name.replace(".csv", "")
            models_dict[m_name] = standardize_emotion_columns(m_df)
        except Exception as e:
            st.error(f"Error loading uploaded file '{u_file.name}': {e}")

# 2. Load selected local models
if selected_local_keys:
    for l_key in selected_local_keys:
        fpath = local_predictions[l_key]
        try:
            m_df = pd.read_csv(fpath)
            clean_name = os.path.basename(fpath).replace(".csv", "")
            if clean_name in models_dict:
                clean_name = f"{clean_name} ({os.path.dirname(fpath)})"
            models_dict[clean_name] = standardize_emotion_columns(m_df)
        except Exception as e:
            st.error(f"Error loading local model '{fpath}': {e}")

# 3. Default fallback if nothing selected/uploaded
if not models_dict:
    for fallback in ["outputs/simpleNewMarkup.csv", "data/predictions/simpleNewMarkup.csv", "simpleNewMarkup.csv"]:
        if os.path.exists(fallback):
            try:
                models_dict["simpleNewMarkup (default)"] = standardize_emotion_columns(pd.read_csv(fallback))
                break
            except Exception:
                pass

# Load human ratings
df_ratings = pd.read_csv(ratings_file) if ratings_file else None
if df_ratings is None:
    for default_r in ["ratings_export_20261019_201809.csv", "data/annotations/ratings_export_20261019_201809.csv"]:
        if os.path.exists(default_r):
            try:
                df_ratings = pd.read_csv(default_r)
                break
            except Exception:
                pass

# Load mapping file
df_map = None
if mapping_file:
    try:
        df_map = standardize_mapping_dataframe(pd.read_csv(mapping_file))
    except Exception as e:
        st.error(f"Error loading uploaded mapping file: {e}")
elif selected_mapping_choice != "None (Upload below)":
    map_path = local_mappings.get(selected_mapping_choice)
    if map_path and os.path.exists(map_path):
        try:
            df_map = standardize_mapping_dataframe(pd.read_csv(map_path))
        except Exception as e:
            st.error(f"Error loading local mapping file '{map_path}': {e}")
else:
    # Automatic fallback
    if os.path.exists("mapping_ratings_affectnet.csv") and os.path.getsize("mapping_ratings_affectnet.csv") > 0:
        try:
            df_map = standardize_mapping_dataframe(pd.read_csv("mapping_ratings_affectnet.csv"))
        except Exception:
            pass

# Validation check
if df_ratings is None or not models_dict:
    st.warning("⚠️ Please provide the Human Ratings CSV and select or upload at least one Model Estimations CSV in the sidebar.")
    st.stop()

# Vector Computation based on alignment mode
if alignment_mode == "7 Shared Classes (Drop 'neutral')":
    current_classes = EMOTIONS_7
    human_raw = df_ratings[RATINGS_COLS].mean().values
    p_vector = normalize_distribution(human_raw)
else:
    current_classes = EMOTIONS_8
    df_r_norm = df_ratings[RATINGS_COLS].copy()
    raw_sums = df_r_norm.sum(axis=1)
    neutrals = np.maximum(0.0, 1.0 - raw_sums)
    df_r_norm['emotion_neutral'] = neutrals
    human_raw = df_r_norm[RATINGS_COLS + ['emotion_neutral']].mean().values
    p_vector = normalize_distribution(human_raw)

# Process distributions and metrics for each selected model
q_vectors = {}
model_metrics = {}

for m_name, df_m in models_dict.items():
    if alignment_mode == "7 Shared Classes (Drop 'neutral')":
        missing_classes = [c for c in EMOTIONS_7 if c not in df_m.columns]
        if missing_classes:
            st.warning(f"⚠️ Model '{m_name}' is missing classes: {missing_classes}. Skipping.")
            continue
        m_raw = df_m[EMOTIONS_7].mean().values
        q_vec = normalize_distribution(m_raw)
    else:
        # 8 Classes
        if 'neutral' in df_m.columns:
            missing_classes = [c for c in EMOTIONS_8 if c not in df_m.columns]
            if missing_classes:
                st.warning(f"⚠️ Model '{m_name}' is missing classes: {missing_classes}. Skipping.")
                continue
            m_raw = df_m[EMOTIONS_8].mean().values
            q_vec = normalize_distribution(m_raw)
        else:
            missing_classes = [c for c in EMOTIONS_7 if c not in df_m.columns]
            if missing_classes:
                st.warning(f"⚠️ Model '{m_name}' is missing classes: {missing_classes}. Skipping.")
                continue
            m_raw_7 = df_m[EMOTIONS_7].mean().values
            # Impute neutral as 0.0 and renormalize
            q_vec = normalize_distribution(list(m_raw_7) + [0.0])

    q_vectors[m_name] = q_vec
    model_metrics[m_name] = compute_ldl_metrics(p_vector, q_vec)

if not q_vectors:
    st.error("No valid model predictions could be aligned with the selected emotion space.")
    st.stop()

# ==============================================================================
# Visualization Customization (Titles, Legends, Axes & Emotion Labels)
# ==============================================================================
with st.sidebar:
    st.divider()
    st.header("🎨 Gráficos & Estilização")
    with st.expander("📝 Customizar Legendas, Títulos e Fontes", expanded=False):
        preset_lang = st.radio(
            "Predefinição de Idioma:",
            options=["Inglês (Original)", "Português"],
            index=0,
            horizontal=True,
            help="Altera rapidamente os valores padrão dos títulos, legendas e classes para Português ou Inglês."
        )
        is_pt = (preset_lang == "Português")
        suffix = "pt" if is_pt else "en"

        # Defaults based on preset
        def_title_summary = "Desempenho Geral Comparativo entre Modelos" if is_pt else "Comparative Overall Performance Across Selected Models (Higher is Better)"
        def_title_divergence = "Divergências de Informação (JS e KL) entre Humano e Modelos" if is_pt else "Information Divergences (JS & KL) between Human and Models (Lower is Better)"
        def_title_bar = "Distribuição de Massa de Probabilidade por Classe" if is_pt else "Probability Mass per Class (Grouped Bar Chart)"
        def_title_radar = "Perfil Multidimensional de Consistência (Radar)" if is_pt else "Multidimensional Profile (Radar Chart)"
        def_title_face_bar = "Distribuição de Probabilidades da Face {face_id}" if is_pt else "Emotion Probability Distributions for Face {face_id}"

        def_leg_human = "Humano (Consenso)" if is_pt else "Human (P)"
        def_axis_x = "Categoria de Emoção" if is_pt else "Emotion Category"
        def_axis_y = "Massa de Probabilidade" if is_pt else "Probability Mass"
        def_axis_summary_y = "Pontuação [0, 1]" if is_pt else "Score [0, 1]"
        def_axis_div_y = "Divergência (Menor é Melhor)" if is_pt else "Divergence (Lower is Better)"

        st.markdown("##### 📌 Títulos dos Gráficos")
        title_summary_custom = st.text_input(
            "Título - Comparativo Geral:",
            value=def_title_summary,
            key=f"title_summary_{suffix}"
        )
        title_divergence_custom = st.text_input(
            "Título - Divergências (JS e KL):",
            value=def_title_divergence,
            key=f"title_div_{suffix}"
        )
        title_bar_custom = st.text_input(
            "Título - Distribuição por Classe (Barras):",
            value=def_title_bar,
            key=f"title_bar_{suffix}"
        )
        title_radar_custom = st.text_input(
            "Título - Gráfico Radar:",
            value=def_title_radar,
            key=f"title_radar_{suffix}"
        )
        title_face_bar_custom = st.text_input(
            "Título - Gráfico por Face (use {face_id}):",
            value=def_title_face_bar,
            key=f"title_face_{suffix}"
        )

        st.markdown("##### 🏷️ Legendas dos Dados")
        human_legend_custom = st.text_input(
            "Legenda da Anotação Humana:",
            value=def_leg_human,
            key=f"leg_human_{suffix}"
        )

        st.markdown("**Legenda dos Modelos:**")
        custom_model_legends = {}
        for m_name in q_vectors.keys():
            def_m_label = f"Modelo: {m_name}" if is_pt else f"Model: {m_name}"
            custom_model_legends[m_name] = st.text_input(
                f"Legenda para '{m_name}':",
                value=def_m_label,
                key=f"leg_m_{m_name}_{suffix}"
            )

        st.markdown("##### 📏 Rótulos dos Eixos")
        col_ax1, col_ax2 = st.columns(2)
        with col_ax1:
            axis_x_title = st.text_input("Eixo X (Classes):", value=def_axis_x, key=f"ax_x_{suffix}")
        with col_ax2:
            axis_y_title = st.text_input("Eixo Y (Prob.):", value=def_axis_y, key=f"ax_y_{suffix}")
        axis_summary_y = st.text_input("Eixo Y (Comparativo):", value=def_axis_summary_y, key=f"ax_sum_y_{suffix}")

        st.markdown("##### 🎭 Rótulos das Classes de Emoção")
        emotion_naming_choice = st.selectbox(
            "Estilo dos Nomes das Emoções:",
            options=["Conforme idioma selecionado", "Português", "Inglês (Capitalizado)", "Inglês (Minúsculo Original)", "Personalizado"],
            index=0,
            key=f"emo_naming_{suffix}"
        )

        PT_EMOTIONS = {
            'happy': 'Alegria',
            'contempt': 'Desprezo',
            'surprised': 'Surpresa',
            'angry': 'Raiva',
            'disgusted': 'Nojo',
            'fearful': 'Medo',
            'sad': 'Tristeza',
            'neutral': 'Neutro'
        }
        CAP_EMOTIONS = {
            'happy': 'Happy',
            'contempt': 'Contempt',
            'surprised': 'Surprised',
            'angry': 'Angry',
            'disgusted': 'Disgusted',
            'fearful': 'Fearful',
            'sad': 'Sad',
            'neutral': 'Neutral'
        }

        emotion_name_map = {}
        if emotion_naming_choice == "Personalizado":
            st.markdown("*Edite o rótulo de cada classe individualmente:*")
            emo_cols = st.columns(2)
            for idx, c in enumerate(current_classes):
                col_i = emo_cols[idx % 2]
                with col_i:
                    default_c_val = PT_EMOTIONS.get(c, c) if is_pt else CAP_EMOTIONS.get(c, c)
                    emotion_name_map[c] = st.text_input(
                        f"Classe '{c}':",
                        value=default_c_val,
                        key=f"custom_class_{c}_{suffix}"
                    )
        elif emotion_naming_choice == "Português" or (emotion_naming_choice == "Conforme idioma selecionado" and is_pt):
            emotion_name_map = PT_EMOTIONS
        elif emotion_naming_choice == "Inglês (Capitalizado)":
            emotion_name_map = CAP_EMOTIONS
        else:
            emotion_name_map = {c: c for c in current_classes}

        display_classes = [emotion_name_map.get(c, c) for c in current_classes]

        st.markdown("##### 📍 Layout da Legenda")
        legend_position_choice = st.selectbox(
            "Posição da Legenda nos Gráficos:",
            options=["Horizontal (Superior)", "Vertical (Direita)", "Ocultar Legenda"],
            index=0,
            key=f"leg_pos_{suffix}"
        )

        st.markdown("##### 🔤 Tamanho das Fontes")
        font_adjustment_mode = st.radio(
            "Modo de Ajuste da Fonte:",
            options=["Geral (Escalar Tudo)", "Personalizado (Por Elemento)"],
            index=0,
            horizontal=True,
            key=f"font_mode_{suffix}"
        )

        if font_adjustment_mode == "Geral (Escalar Tudo)":
            base_font_size = st.slider(
                "Tamanho da Fonte:",
                min_value=8,
                max_value=28,
                value=13,
                step=1,
                help="Ajusta proporcionalmente o tamanho de todos os textos nos gráficos.",
                key=f"base_font_{suffix}"
            )
            title_font_size = int(round(base_font_size * 1.25))
            axis_title_font_size = int(round(base_font_size * 1.05))
            tick_font_size = base_font_size
            legend_font_size = max(8, int(round(base_font_size * 0.95)))
            bar_text_font_size = max(8, int(round(base_font_size * 0.85)))
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                title_font_size = st.slider("Título dos Gráficos:", 10, 32, 16, key=f"title_font_{suffix}")
                axis_title_font_size = st.slider("Títulos dos Eixos (X/Y):", 8, 26, 14, key=f"axis_title_font_{suffix}")
                tick_font_size = st.slider("Rótulos / Ticks / Classes:", 8, 24, 12, key=f"tick_font_{suffix}")
            with col_f2:
                legend_font_size = st.slider("Texto da Legenda:", 8, 24, 12, key=f"legend_font_{suffix}")
                bar_text_font_size = st.slider("Valores nas Barras (%):", 6, 20, 10, key=f"bar_text_font_{suffix}")
                base_font_size = tick_font_size

        if legend_position_choice == "Horizontal (Superior)":
            legend_layout = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=dict(text=""),
                font=dict(size=legend_font_size)
            )
        elif legend_position_choice == "Vertical (Direita)":
            legend_layout = dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                title=dict(text=""),
                font=dict(size=legend_font_size)
            )
        else:
            legend_layout = dict(visible=False)

# ==============================================================================
# 1. Primary KPI Cards & Comparative Overview (Macro Consistency)
# ==============================================================================
st.subheader("📈 Global Consistency Overview (Macro Level)")
num_models = len(q_vectors)

if num_models == 1:
    m_name = list(q_vectors.keys())[0]
    metrics = model_metrics[m_name]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Cosine Similarity", f"{metrics['Cosine']:.4f}", help="Vector orientation alignment (Optimal: 1.0)")
    with col2:
        st.metric("Jensen-Shannon (JS)", f"{metrics['JSD']:.4f}", help="Symmetric, bounded relative entropy D_JS(Human, Model) [0, 1] (Optimal: 0.0)")
    with col3:
        st.metric("Kullback-Leibler (KL)", f"{metrics['KL_Divergence']:.4f}", help="Relative entropy D_KL(Human || Model) (Optimal: 0.0)")
    with col4:
        st.metric("Pearson Correlation", f"{metrics['Pearson']:.4f}", help="Linear correlation across emotional class prevalences (Optimal: 1.0)")
    with col5:
        st.metric("Probability Overlap", f"{metrics['Overlap']:.4f}", help="Shared probability volume ∑ min(P_i, Q_i) (Optimal: 1.0)")
else:
    # Multi-model leadership highlights
    best_cos_m = max(model_metrics, key=lambda k: model_metrics[k]['Cosine'])
    best_jsd_m = min(model_metrics, key=lambda k: model_metrics[k]['JSD'])
    best_kl_m = min(model_metrics, key=lambda k: model_metrics[k]['KL_Divergence'])
    best_pear_m = max(model_metrics, key=lambda k: model_metrics[k]['Pearson'])
    best_over_m = max(model_metrics, key=lambda k: model_metrics[k]['Overlap'])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Top Cosine Alignment 🏆",
            f"{model_metrics[best_cos_m]['Cosine']:.4f}",
            delta=f"Best: {custom_model_legends.get(best_cos_m, best_cos_m)}",
            delta_color="normal",
            help="Best vector orientation alignment with human consensus (Optimal: 1.0)"
        )
    with col2:
        st.metric(
            "Lowest JS Divergence 🏆",
            f"{model_metrics[best_jsd_m]['JSD']:.4f}",
            delta=f"Best: {custom_model_legends.get(best_jsd_m, best_jsd_m)}",
            delta_color="inverse",
            help="Lowest relative entropy / Jensen-Shannon Divergence D_JS(Human, Model) (Optimal: 0.0)"
        )
    with col3:
        st.metric(
            "Lowest KL Divergence 🏆",
            f"{model_metrics[best_kl_m]['KL_Divergence']:.4f}",
            delta=f"Best: {custom_model_legends.get(best_kl_m, best_kl_m)}",
            delta_color="inverse",
            help="Lowest Kullback-Leibler Divergence D_KL(Human || Model) (Optimal: 0.0)"
        )
    with col4:
        st.metric(
            "Top Pearson Correlation 🏆",
            f"{model_metrics[best_pear_m]['Pearson']:.4f}",
            delta=f"Best: {custom_model_legends.get(best_pear_m, best_pear_m)}",
            delta_color="normal",
            help="Highest linear correlation across emotion distribution (Optimal: 1.0)"
        )
    with col5:
        st.metric(
            "Top Shared Overlap 🏆",
            f"{model_metrics[best_over_m]['Overlap']:.4f}",
            delta=f"Best: {custom_model_legends.get(best_over_m, best_over_m)}",
            delta_color="normal",
            help="Highest intersection / shared probability volume ∑ min(P, Q) (Optimal: 1.0)"
        )

# Comparative Metric Summary & Information Divergences Bar Charts
c_s1, c_s2 = st.columns(2)

with c_s1:
    summary_records = []
    for m_name, m in model_metrics.items():
        summary_records.append({
            "Model": custom_model_legends.get(m_name, m_name),
            "Cosine Alignment": m["Cosine"],
            "Pearson Correlation": m["Pearson"],
            "Histogram Overlap": m["Overlap"],
            "Fidelity": m["Fidelity"],
            "1 - JSD (Similarity)": max(0.0, 1.0 - m["JSD"])
        })
    df_summary = pd.DataFrame(summary_records).melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig_summary = px.bar(
        df_summary,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title=title_summary_custom,
        color_discrete_sequence=MODEL_PALETTE
    )
    fig_summary.update_layout(
        font=dict(size=base_font_size),
        title=dict(
            text=title_summary_custom,
            font=dict(size=title_font_size)
        ),
        yaxis_range=[0, 1.05],
        yaxis=dict(
            title=dict(text=axis_summary_y, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=tick_font_size)
        ),
        xaxis=dict(
            tickfont=dict(size=tick_font_size)
        ),
        legend=legend_layout,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    st.plotly_chart(fig_summary, use_container_width=True)

with c_s2:
    # Information Divergences Bar Chart (JS and KL Divergences: Human vs Models)
    divergence_records = []
    for m_name, m in model_metrics.items():
        m_label = custom_model_legends.get(m_name, m_name)
        divergence_records.append({
            "Model": m_label,
            "Metric": "JS Divergence (JSD)" if not is_pt else "Divergência JS (JSD)",
            "Divergence": m["JSD"]
        })
        divergence_records.append({
            "Model": m_label,
            "Metric": "KL Divergence (KL)" if not is_pt else "Divergência KL (KL)",
            "Divergence": m["KL_Divergence"]
        })
    df_div = pd.DataFrame(divergence_records)

    fig_div = px.bar(
        df_div,
        x="Metric",
        y="Divergence",
        color="Model",
        barmode="group",
        title=title_divergence_custom,
        color_discrete_sequence=MODEL_PALETTE
    )
    fig_div.update_layout(
        font=dict(size=base_font_size),
        title=dict(
            text=title_divergence_custom,
            font=dict(size=title_font_size)
        ),
        yaxis=dict(
            title=dict(text=def_axis_div_y, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=tick_font_size)
        ),
        xaxis=dict(
            tickfont=dict(size=tick_font_size)
        ),
        legend=legend_layout,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    st.plotly_chart(fig_div, use_container_width=True)

# ==============================================================================
# 2. Detailed Distribution Visualizations
# ==============================================================================
st.divider()
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader(title_bar_custom)
    fig_bar = go.Figure()

    # Human distribution
    fig_bar.add_trace(go.Bar(
        name=human_legend_custom,
        x=display_classes,
        y=p_vector,
        marker_color=HUMAN_COLOR,
        text=[f"{v*100:.1f}%" for v in p_vector],
        textposition='auto',
        textfont=dict(size=bar_text_font_size),
        hoverinfo='x+y+name'
    ))

    # Model distributions
    for idx, (m_name, q_vec) in enumerate(q_vectors.items()):
        c_color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        m_label = custom_model_legends.get(m_name, f'Model: {m_name}')
        fig_bar.add_trace(go.Bar(
            name=m_label,
            x=display_classes,
            y=q_vec,
            marker_color=c_color,
            text=[f"{v*100:.1f}%" for v in q_vec],
            textposition='auto',
            textfont=dict(size=bar_text_font_size),
            hoverinfo='x+y+name'
        ))

    fig_bar.update_layout(
        font=dict(size=base_font_size),
        barmode='group',
        yaxis=dict(
            title=dict(text=axis_y_title, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=tick_font_size)
        ),
        xaxis=dict(
            title=dict(text=axis_x_title, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=tick_font_size)
        ),
        legend=legend_layout,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with g_col2:
    st.subheader(title_radar_custom)
    fig_radar = go.Figure()

    # Human polar trace
    fig_radar.add_trace(go.Scatterpolar(
        r=list(p_vector) + [p_vector[0]],
        theta=display_classes + [display_classes[0]],
        fill='toself',
        name=human_legend_custom,
        line_color=HUMAN_COLOR,
        fillcolor='rgba(79, 70, 229, 0.20)',
        line=dict(width=2.5)
    ))

    max_val = np.max(p_vector)
    for idx, (m_name, q_vec) in enumerate(q_vectors.items()):
        c_color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        max_val = max(max_val, np.max(q_vec))
        m_label = custom_model_legends.get(m_name, m_name)
        fig_radar.add_trace(go.Scatterpolar(
            r=list(q_vec) + [q_vec[0]],
            theta=display_classes + [display_classes[0]],
            fill='none',
            name=m_label,
            line_color=c_color,
            line=dict(width=2.2)
        ))

    fig_radar.update_layout(
        font=dict(size=base_font_size),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.15],
                tickfont=dict(size=tick_font_size)
            ),
            angularaxis=dict(
                tickfont=dict(size=tick_font_size)
            )
        ),
        legend=legend_layout,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==============================================================================
# 3. Comprehensive LDL Benchmark Table (Macro)
# ==============================================================================
st.divider()
st.subheader("📐 Label Distribution Learning (LDL) Benchmark Table (Macro)")

metric_metadata = [
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Cosine Similarity", "key": "Cosine", "Goal": "Higher (max 1.0)", "Formula": "P · Q / (||P|| ||Q||)", "direction": "max"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Pearson Correlation (r)", "key": "Pearson", "Goal": "Higher (max 1.0)", "Formula": "Cov(P, Q) / (σP · σQ)", "direction": "max"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Histogram Overlap / Intersection", "key": "Overlap", "Goal": "Higher (max 1.0)", "Formula": "∑ min(P_i, Q_i)", "direction": "max"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Fidelity (Bhattacharyya Coeff.)", "key": "Fidelity", "Goal": "Higher (max 1.0)", "Formula": "∑ √(P_i · Q_i)", "direction": "max"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Jensen-Shannon Divergence (JSD)", "key": "JSD", "Goal": "Lower (min 0.0)", "Formula": "0.5·D_KL(Human||M) + 0.5·D_KL(Model||M)", "direction": "min"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Kullback-Leibler Divergence (KL)", "key": "KL_Divergence", "Goal": "Lower (min 0.0)", "Formula": "∑ P_human · ln(P_human / Q_model)", "direction": "min"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Total Variation (L1)", "key": "Total_Variation", "Goal": "Lower (min 0.0)", "Formula": "0.5 · ∑ |P_i - Q_i|", "direction": "min"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Euclidean Distance (L2)", "key": "Euclidean_L2", "Goal": "Lower (min 0.0)", "Formula": "√(∑ (P_i - Q_i)²)", "direction": "min"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Chebyshev Distance (L∞)", "key": "Chebyshev_Linf", "Goal": "Lower (min 0.0)", "Formula": "max |P_i - Q_i|", "direction": "min"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Hellinger Distance", "key": "Hellinger", "Goal": "Lower (min 0.0)", "Formula": "√[ 0.5 · ∑ (√P_i - √Q_i)² ]", "direction": "min"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Bhattacharyya Distance", "key": "Bhattacharyya", "Goal": "Lower (min 0.0)", "Formula": "-ln(Fidelity)", "direction": "min"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Chi-Square Distance (χ²)", "key": "Chi2", "Goal": "Lower (min 0.0)", "Formula": "∑ [ (P_i - Q_i)² / (P_i + Q_i) ]", "direction": "min"}
]

table_rows = []
for item in metric_metadata:
    k = item["key"]
    direction = item["direction"]
    row_dict = {
        "Category": item["Category"],
        "Metric": item["Metric"],
        "Goal": item["Goal"],
        "Formula": item["Formula"]
    }

    # Gather values across all models
    model_values = {m_name: model_metrics[m_name][k] for m_name in q_vectors.keys()}

    # Determine winning model
    if len(model_values) > 1:
        if direction == "max":
            best_m = max(model_values, key=model_values.get)
        else:
            best_m = min(model_values, key=model_values.get)
        row_dict["Best Model 🏆"] = custom_model_legends.get(best_m, best_m)

    for m_name, val in model_values.items():
        col_name = custom_model_legends.get(m_name, m_name)
        row_dict[col_name] = f"{val:.5f}"

    table_rows.append(row_dict)

m_df = pd.DataFrame(table_rows)
st.dataframe(m_df, use_container_width=True, hide_index=True)

# ==============================================================================
# 4. Face-by-Face Consistency Analysis (Análise de Consistência por Face)
# ==============================================================================
st.divider()
st.subheader("🔍 Face-by-Face Consistency Analysis")
st.markdown("Detailed consistency metrics calculated for **each individual face instance** evaluated in the user study.")

if df_map is not None and not df_map.empty:
    # 1. Prepare model lookups indexed by basename
    prepared_model_lookups = {}
    for m_name, df_m in models_dict.items():
        df_copy = df_m.copy()
        file_col = next((c for c in ['file', 'path', 'affectnet_file', 'dataset_file', 'filename', 'image'] if c in df_copy.columns), None)
        if file_col:
            df_copy['join_basename'] = df_copy[file_col].astype(str).apply(os.path.basename)
            # Retain only one row per basename if duplicate
            df_dedup = df_copy.drop_duplicates(subset=['join_basename']).set_index('join_basename')
            prepared_model_lookups[m_name] = df_dedup
        else:
            st.warning(f"Could not find file path column in model '{m_name}' to link by face.")

    # 2. Compute human average consensus per face
    # Detect grouping columns in df_ratings
    r_cols_present = [c for c in ['image_code', 'image_filename', 'filename', 'image'] if c in df_ratings.columns]
    if not r_cols_present:
        r_cols_present = [df_ratings.columns[0]]

    human_per_face = df_ratings.groupby(r_cols_present)[RATINGS_COLS].mean().reset_index()

    # 3. Match human ratings with mapping table
    # Prepare clean join keys
    if 'image_filename' in human_per_face.columns and 'image_filename' in df_map.columns:
        human_per_face['join_key'] = human_per_face['image_filename'].astype(str).apply(os.path.basename)
        df_map['join_key'] = df_map['image_filename'].astype(str).apply(os.path.basename)
        merged_faces = human_per_face.merge(df_map, on='join_key', suffixes=('', '_map'))
    elif 'image_code' in human_per_face.columns and 'image_code' in df_map.columns:
        human_per_face['join_key'] = human_per_face['image_code'].astype(str).str.strip()
        df_map['join_key'] = df_map['image_code'].astype(str).str.strip()
        merged_faces = human_per_face.merge(df_map, on='join_key', suffixes=('', '_map'))
    else:
        # Fallback to first text column
        merged_faces = human_per_face.merge(df_map, left_on=human_per_face.columns[0], right_on=df_map.columns[0])

    if not merged_faces.empty and prepared_model_lookups:
        # 4. Compute per-face consistency metrics for all faces
        face_records = []
        face_distributions = {}  # {face_id: {'human': p, 'models': {m_name: q}}}

        for _, row in merged_faces.iterrows():
            face_id = str(row.get('image_code', row.get('image_filename', row.get('join_key'))))
            study_filename = str(row.get('image_filename', ''))
            dataset_filename = str(row.get('affectnet_file', row.get('dataset_file', study_filename)))
            target_ds_basename = os.path.basename(dataset_filename)

            # Human distribution for this face
            p_raw = row[RATINGS_COLS].values
            if alignment_mode == "7 Shared Classes (Drop 'neutral')":
                p_face = normalize_distribution(p_raw)
            else:
                p_neutral = max(0.0, 1.0 - np.sum(p_raw))
                p_face = normalize_distribution(list(p_raw) + [p_neutral])

            top_human_idx = int(np.argmax(p_face))
            top_human_emotion = current_classes[top_human_idx]

            record = {
                "Face ID": face_id,
                "Study File": study_filename,
                "Dataset File": target_ds_basename,
                "Human Top Emotion": emotion_name_map.get(top_human_emotion, top_human_emotion),
                "full_path": row.get('user_study_full_path', row.get('dataset_full_path', ''))
            }

            model_cosines = {}
            model_jsds = {}
            model_kls = {}
            model_top_emotions = {}
            models_matched_count = 0

            face_dists = {'human': p_face, 'models': {}}

            for m_name, df_lookup in prepared_model_lookups.items():
                m_label = custom_model_legends.get(m_name, m_name)
                if target_ds_basename in df_lookup.index:
                    m_row = df_lookup.loc[target_ds_basename]
                    if isinstance(m_row, pd.DataFrame):
                        m_row = m_row.iloc[0]

                    if alignment_mode == "7 Shared Classes (Drop 'neutral')":
                        q_raw = m_row[EMOTIONS_7].values
                        q_face = normalize_distribution(q_raw)
                    else:
                        if 'neutral' in m_row:
                            q_raw = m_row[EMOTIONS_8].values
                            q_face = normalize_distribution(q_raw)
                        else:
                            q_raw = m_row[EMOTIONS_7].values
                            q_face = normalize_distribution(list(q_raw) + [0.0])

                    face_dists['models'][m_name] = q_face

                    top_m_idx = int(np.argmax(q_face))
                    top_m_emotion = current_classes[top_m_idx]
                    top_m_display = emotion_name_map.get(top_m_emotion, top_m_emotion)
                    model_top_emotions[m_name] = top_m_emotion

                    # Compute metrics (Human = P, Model = Q)
                    f_metrics = compute_ldl_metrics(p_face, q_face)
                    cos = f_metrics["Cosine"]
                    jsd = f_metrics["JSD"]
                    kl = f_metrics["KL_Divergence"]
                    over = f_metrics["Overlap"]

                    model_cosines[m_name] = cos
                    model_jsds[m_name] = jsd
                    model_kls[m_name] = kl
                    models_matched_count += 1

                    record[f"{m_label} (Top Emotion)"] = top_m_display
                    record[f"{m_label} (Cosine)"] = round(cos, 4)
                    record[f"{m_label} (JSD)"] = round(jsd, 4)
                    record[f"{m_label} (KL)"] = round(kl, 4)
                    record[f"{m_label} (Overlap)"] = round(over, 4)
                else:
                    record[f"{m_label} (Cosine)"] = np.nan
                    record[f"{m_label} (JSD)"] = np.nan
                    record[f"{m_label} (KL)"] = np.nan

            if models_matched_count > 0:
                # Find best model for this specific face (highest cosine)
                best_face_m = max(model_cosines, key=model_cosines.get) if model_cosines else "N/A"
                record["Best Model 🏆"] = custom_model_legends.get(best_face_m, best_face_m)
                # Check top-1 dominant agreement across models
                agreements = [top_human_emotion == model_top_emotions[m] for m in model_top_emotions]
                record["Dominant Consensus Agreed"] = "Yes" if any(agreements) else "No"
                record["Average Cosine"] = round(float(np.mean(list(model_cosines.values()))), 4)
                record["Average JSD"] = round(float(np.mean(list(model_jsds.values()))), 4)
                record["Average KL"] = round(float(np.mean(list(model_kls.values()))), 4)

                face_records.append(record)
                face_distributions[face_id] = face_dists

        df_face_results = pd.DataFrame(face_records)

        if not df_face_results.empty:
            # ------------------------------------------------------------------
            # Banner with Aggregate Face-by-Face KPIs
            # ------------------------------------------------------------------
            st.markdown("#### 📊 Aggregate Face-by-Face Consistency Scores")
            kpi_cols = st.columns(5)
            with kpi_cols[0]:
                st.metric("Total Faces Linked", f"{len(df_face_results)} faces", help="Faces successfully cross-referenced with model inferences.")
            with kpi_cols[1]:
                mean_cos = df_face_results["Average Cosine"].mean()
                st.metric("Mean Cosine per Face", f"{mean_cos:.4f}", help="Average Cosine Similarity across all evaluated faces.")
            with kpi_cols[2]:
                mean_jsd = df_face_results["Average JSD"].mean()
                st.metric("Mean JS Divergence", f"{mean_jsd:.4f}", help="Average Jensen-Shannon Divergence D_JS(Human, Model) across all evaluated faces.")
            with kpi_cols[3]:
                mean_kl = df_face_results["Average KL"].mean()
                st.metric("Mean KL Divergence", f"{mean_kl:.4f}", help="Average Kullback-Leibler Divergence D_KL(Human || Model) across all evaluated faces.")
            with kpi_cols[4]:
                agree_rate = (df_face_results["Dominant Consensus Agreed"] == "Yes").mean() * 100
                st.metric("Top-1 Agreement Rate", f"{agree_rate:.1f}%", help="Percentage of faces where at least one model matched the human dominant emotion.")

            # ------------------------------------------------------------------
            # Interactive Filter & Search Controls
            # ------------------------------------------------------------------
            st.markdown("#### 📋 Interactive Per-Face Consistency Table")
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 1.5, 2])

            with f_col1:
                emotion_filter = st.selectbox(
                    "Filter by Human Dominant Emotion:",
                    options=["All Emotions"] + current_classes,
                    format_func=lambda x: emotion_name_map.get(x, x) if x != "All Emotions" else x,
                    index=0
                )
            with f_col2:
                agreement_filter = st.selectbox(
                    "Filter by Dominant Agreement:",
                    options=["All Faces", "Agreed on Dominant Emotion", "Disagreed on Dominant Emotion"],
                    index=0
                )
            with f_col3:
                sort_order = st.selectbox(
                    "Sort Faces by:",
                    options=[
                        "Highest Cosine (Most Consistent)",
                        "Lowest Cosine (Most Divergent)",
                        "Lowest JS Divergence (Closest Match)",
                        "Highest JS Divergence (Most Divergent)",
                        "Lowest KL Divergence (Closest Match)",
                        "Highest KL Divergence (Most Divergent)",
                        "Face ID"
                    ],
                    index=0
                )
            with f_col4:
                search_query = st.text_input("Search Face ID or Filename:", placeholder="e.g. 1086")

            # Apply filters
            filtered_df = df_face_results.copy()
            if emotion_filter != "All Emotions":
                filtered_df = filtered_df[filtered_df["Human Top Emotion"] == emotion_name_map.get(emotion_filter, emotion_filter)]

            if agreement_filter == "Agreed on Dominant Emotion":
                filtered_df = filtered_df[filtered_df["Dominant Consensus Agreed"] == "Yes"]
            elif agreement_filter == "Disagreed on Dominant Emotion":
                filtered_df = filtered_df[filtered_df["Dominant Consensus Agreed"] == "No"]

            if search_query:
                q_str = search_query.strip().lower()
                filtered_df = filtered_df[
                    filtered_df["Face ID"].astype(str).str.lower().str.contains(q_str) |
                    filtered_df["Study File"].astype(str).str.lower().str.contains(q_str) |
                    filtered_df["Dataset File"].astype(str).str.lower().str.contains(q_str)
                ]

            # Sorting
            if sort_order == "Highest Cosine (Most Consistent)":
                filtered_df = filtered_df.sort_values(by="Average Cosine", ascending=False)
            elif sort_order == "Lowest Cosine (Most Divergent)":
                filtered_df = filtered_df.sort_values(by="Average Cosine", ascending=True)
            elif sort_order == "Lowest JS Divergence (Closest Match)":
                filtered_df = filtered_df.sort_values(by="Average JSD", ascending=True)
            elif sort_order == "Highest JS Divergence (Most Divergent)":
                filtered_df = filtered_df.sort_values(by="Average JSD", ascending=False)
            elif sort_order == "Lowest KL Divergence (Closest Match)":
                filtered_df = filtered_df.sort_values(by="Average KL", ascending=True)
            elif sort_order == "Highest KL Divergence (Most Divergent)":
                filtered_df = filtered_df.sort_values(by="Average KL", ascending=False)
            else:
                filtered_df = filtered_df.sort_values(by="Face ID", ascending=True)

            # Display table
            display_cols = [c for c in filtered_df.columns if c != "full_path"]
            st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

            # Download CSV button
            csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Per-Face Consistency Table (CSV)",
                data=csv_data,
                file_name="per_face_consistency_results.csv",
                mime="text/csv"
            )

            # ------------------------------------------------------------------
            # Individual Face Visual Inspector
            # ------------------------------------------------------------------
            st.markdown("#### 🔬 Detailed Face Inspection")
            available_face_ids = filtered_df["Face ID"].tolist()

            if available_face_ids:
                selected_face_id = st.selectbox(
                    "Choose face instance to inspect:",
                    options=available_face_ids,
                    format_func=lambda x: f"Face: {x} | File: {filtered_df.loc[filtered_df['Face ID'] == x, 'Dataset File'].iloc[0]} (Human: {filtered_df.loc[filtered_df['Face ID'] == x, 'Human Top Emotion'].iloc[0]})"
                )

                face_row = filtered_df[filtered_df["Face ID"] == selected_face_id].iloc[0]
                face_data = face_distributions.get(selected_face_id)

                if face_data:
                    p_inst = face_data['human']
                    models_inst = face_data['models']

                    ins_col1, ins_col2 = st.columns([1, 2])

                    with ins_col1:
                        # Attempt to find and display face image
                        img_rendered = False
                        candidate_img_paths = [
                            face_row.get('full_path', ''),
                            os.path.join(faces_image_dir, face_row['Study File']) if faces_image_dir else '',
                            os.path.join(faces_image_dir, face_row['Dataset File']) if faces_image_dir else '',
                            os.path.join(faces_image_dir, face_row['Human Top Emotion'], face_row['Study File']) if faces_image_dir else '',
                            os.path.join("amostras_selecionadas", face_row['Human Top Emotion'], face_row['Study File']),
                            os.path.join("amostras_selecionadas", face_row['Study File'])
                        ]

                        for c_path in candidate_img_paths:
                            if c_path and os.path.isfile(c_path):
                                try:
                                    img = Image.open(c_path)
                                    st.image(img, caption=f"Face: {face_row['Dataset File']}", width=230)
                                    img_rendered = True
                                    break
                                except Exception:
                                    pass

                        if not img_rendered:
                            st.info(f"📷 Image `{face_row['Dataset File']}` (Provide folder in sidebar to render)")

                        st.markdown(f"**Face ID / Code:** `{face_row['Face ID']}`")
                        st.markdown(f"**Study Filename:** `{face_row['Study File']}`")
                        st.markdown(f"**Dataset Reference:** `{face_row['Dataset File']}`")
                        st.markdown(f"**Dominant Human Emotion:** **{face_row['Human Top Emotion']}**")
                        st.markdown(f"**Best Performing Model:** 🏆 **{face_row.get('Best Model 🏆', 'N/A')}**")

                        # Small comparison table for this individual face
                        face_cmp_rows = []
                        for m_name, q_val in models_inst.items():
                            f_met = compute_ldl_metrics(p_inst, q_val)
                            m_label = custom_model_legends.get(m_name, m_name)
                            dom_idx = int(np.argmax(q_val))
                            dom_name = current_classes[dom_idx]
                            face_cmp_rows.append({
                                "Model": m_label,
                                "Cosine": f"{f_met['Cosine']:.4f}",
                                "JSD": f"{f_met['JSD']:.4f}",
                                "KL": f"{f_met['KL_Divergence']:.4f}",
                                "Overlap": f"{f_met['Overlap']:.4f}",
                                "Dominant": emotion_name_map.get(dom_name, dom_name)
                            })
                        if face_cmp_rows:
                            st.dataframe(pd.DataFrame(face_cmp_rows), use_container_width=True, hide_index=True)

                    with ins_col2:
                        # Grouped Bar Chart for this specific face
                        fig_face_bar = go.Figure()
                        fig_face_bar.add_trace(go.Bar(
                            name=human_legend_custom,
                            x=display_classes,
                            y=p_inst,
                            marker_color=HUMAN_COLOR,
                            text=[f"{v*100:.1f}%" for v in p_inst],
                            textposition='auto',
                            textfont=dict(size=bar_text_font_size)
                        ))

                        for idx, (m_name, q_val) in enumerate(models_inst.items()):
                            m_color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
                            m_label = custom_model_legends.get(m_name, f'Model: {m_name}')
                            fig_face_bar.add_trace(go.Bar(
                                name=m_label,
                                x=display_classes,
                                y=q_val,
                                marker_color=m_color,
                                text=[f"{v*100:.1f}%" for v in q_val],
                                textposition='auto',
                                textfont=dict(size=bar_text_font_size)
                            ))

                        try:
                            formatted_face_title = title_face_bar_custom.format(
                                face_id=face_row['Face ID'],
                                filename=face_row['Dataset File']
                            )
                        except Exception:
                            formatted_face_title = f"{title_face_bar_custom} ({face_row['Face ID']})"

                        fig_face_bar.update_layout(
                            font=dict(size=base_font_size),
                            barmode='group',
                            height=340,
                            title=dict(text=f"<b>{formatted_face_title}</b>", x=0.5, xanchor='center', font=dict(size=title_font_size)),
                            yaxis=dict(
                                title=dict(text=axis_y_title, font=dict(size=axis_title_font_size)),
                                tickfont=dict(size=tick_font_size)
                            ),
                            xaxis=dict(
                                title=dict(text=axis_x_title, font=dict(size=axis_title_font_size)),
                                tickfont=dict(size=tick_font_size)
                            ),
                            legend=legend_layout,
                            margin=dict(l=10, r=10, t=55, b=10)
                        )
                        st.plotly_chart(fig_face_bar, use_container_width=True)
            else:
                st.info("No faces match the current filter selection.")
        else:
            st.warning("Could not match the mapping file records with the loaded model inferences.")
    else:
        st.info("No matching records found between the provided cross-reference mapping and human ratings.")
else:
    st.info("💡 To enable **Face-by-Face Consistency Verification**, select or upload a valid `mapping_ratings_affectnet.csv` file generated by `match_and_map.py` in the sidebar.")
