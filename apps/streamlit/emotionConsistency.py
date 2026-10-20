import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# Label Distribution Learning (LDL) Core Mathematical Functions
def normalize_distribution(arr, eps=1e-12):
    """Ensures probability vector sums to 1.0 and applies epsilon smoothing to prevent zero-division."""
    arr = np.maximum(0.0, np.array(arr, dtype=np.float64))
    s = np.sum(arr)
    if s <= eps:
        return np.ones_like(arr) / len(arr)
    return arr / s

def compute_ldl_metrics(p, q, eps=1e-12):
    """Computes all 12 standard LDL comparative metrics between distributions P and Q."""
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    n = len(p)

    # 1. Euclidean Distance (L2)
    l2 = float(np.linalg.norm(p - q))

    # 2. Total Variation (L1)
    total_variation = float(0.5 * np.sum(np.abs(p - q)))

    # 3. Chebyshev Distance (L_inf)
    chebyshev = float(np.max(np.abs(p - q)))

    # 4. Kullback-Leibler Divergence
    p_s = np.maximum(p, eps)
    q_s = np.maximum(q, eps)
    kl = float(np.sum(p_s * np.log(p_s / q_s)))

    # 5. Jensen-Shannon Divergence (JSD)
    m_s = 0.5 * (p_s + q_s)
    jsd = float(0.5 * np.sum(p_s * np.log(p_s / m_s)) + 0.5 * np.sum(q_s * np.log(q_s / m_s)))

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
    fidelity = float(np.sum(np.sqrt(p_s * q_s)))
    fidelity = min(1.0, max(0.0, fidelity))

    # 10. Bhattacharyya Distance
    bhattacharyya = float(-np.log(max(fidelity, eps)))

    # 11. Hellinger Distance
    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p_s) - np.sqrt(q_s)) ** 2)))

    # 12. Chi-Square Distance (χ²)
    denom = p + q
    chi2 = float(np.sum(np.where(denom > eps, ((p - q) ** 2) / denom, 0.0)))

    return {
        "JSD": jsd,
        "Cosine": cosine,
        "Pearson": pearson,
        "Overlap": intersection,
        "Fidelity": fidelity,
        "Total_Variation": total_variation,
        "Euclidean_L2": l2,
        "Chebyshev_Linf": chebyshev,
        "KL_Divergence": kl,
        "Hellinger": hellinger,
        "Bhattacharyya": bhattacharyya,
        "Chi2": chi2
    }

# Main Header
st.title("📊 Emotion Distribution Consistency Analyzer (LDL)")
st.markdown("Quantitative and visual consistency analysis between human prevailing ratings and probabilistic model estimates.")

# Sidebar (Data Ingestion & Configuration)
with st.sidebar:
    st.header("⚙️ Data & Settings")
    
    ratings_file = st.file_uploader("Human Ratings CSV file", type=["csv"])
    model_file = st.file_uploader("Model Estimations CSV file", type=["csv"])
    mapping_file = st.file_uploader("Optional Cross-Reference Mapping CSV", type=["csv"])
    
    st.divider()
    alignment_mode = st.radio(
        "Emotion Space Alignment:",
        options=["7 Shared Classes (Drop 'neutral')", "8 Classes (Impute Human 'neutral')"],
        index=0
    )
    
    st.divider()
    st.markdown("💡 **Tip**: If no files are manually uploaded, the system will automatically attempt to load the default local CSV datasets.")

# Dataset Loading
@st.cache_data
def load_datasets(r_file, m_file, map_file):
    df_r = pd.read_csv(r_file) if r_file else (pd.read_csv("ratings_export_20261019_201809.csv") if os.path.exists("ratings_export_20261019_201809.csv") else None)
    df_m = pd.read_csv(m_file) if m_file else (pd.read_csv("simpleNewMarkup.csv") if os.path.exists("simpleNewMarkup.csv") else None)
    df_map = pd.read_csv(map_file) if map_file else (pd.read_csv("mapping_ratings_affectnet.csv") if os.path.exists("mapping_ratings_affectnet.csv") else None)
    return df_r, df_m, df_map

df_ratings, df_model, df_map = load_datasets(ratings_file, model_file, mapping_file)

if df_ratings is None or df_model is None:
    st.warning("Please upload both required CSV datasets in the sidebar to begin the analysis.")
    st.stop()

# Vector Computation based on alignment mode
if alignment_mode == "7 Shared Classes (Drop 'neutral')":
    current_classes = EMOTIONS_7
    
    # Human vector
    human_raw = df_ratings[RATINGS_COLS].mean().values
    p_vector = normalize_distribution(human_raw)
    
    # Model vector (renormalized over the 7 classes)
    model_raw = df_model[EMOTIONS_7].mean().values
    q_vector = normalize_distribution(model_raw)
else:
    current_classes = EMOTIONS_8
    
    # Human vector: impute neutral residual (1 - sum of 7 emotions)
    df_r_norm = df_ratings[RATINGS_COLS].copy()
    raw_sums = df_r_norm.sum(axis=1)
    neutrals = np.maximum(0.0, 1.0 - raw_sums)
    df_r_norm['emotion_neutral'] = neutrals
    human_raw = df_r_norm[RATINGS_COLS + ['emotion_neutral']].mean().values
    p_vector = normalize_distribution(human_raw)
    
    # Model vector: full 8 emotions
    model_raw = df_model[EMOTIONS_8].mean().values
    q_vector = normalize_distribution(model_raw)

# Compute macro metrics
metrics = compute_ldl_metrics(p_vector, q_vector)

# Primary KPI Cards
st.subheader("📈 Global Consistency Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cosine Similarity", f"{metrics['Cosine']:.4f}", help="Vector orientation alignment (Optimal: 1.0)")
with col2:
    st.metric("Jensen-Shannon Divergence", f"{metrics['JSD']:.4f}", help="Symmetric, bounded relative entropy [0, 1] (Optimal: 0.0)")
with col3:
    st.metric("Pearson Correlation", f"{metrics['Pearson']:.4f}", help="Linear correlation across emotional class prevalences (Optimal: 1.0)")
with col4:
    st.metric("Probability Overlap", f"{metrics['Overlap']:.4f}", help="Shared probability volume ∑ min(P_i, Q_i) (Optimal: 1.0)")

# Comparison Charts
st.divider()
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("Probability Mass per Class (Bar Chart)")
    fig_bar = go.Figure(data=[
        go.Bar(name='Human (P)', x=current_classes, y=p_vector, marker_color='#4f46e5'),
        go.Bar(name='Model (Q)', x=current_classes, y=q_vector, marker_color='#10b981')
    ])
    fig_bar.update_layout(
        barmode='group',
        yaxis_title="Probability Mass",
        xaxis_title="Emotion Category",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with g_col2:
    st.subheader("Multidimensional Profile (Radar Chart)")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(p_vector) + [p_vector[0]],
        theta=current_classes + [current_classes[0]],
        fill='toself',
        name='Human (P)',
        line_color='#4f46e5'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=list(q_vector) + [q_vector[0]],
        theta=current_classes + [current_classes[0]],
        fill='toself',
        name='Model (Q)',
        line_color='#10b981'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(np.max(p_vector), np.max(q_vector)) * 1.15])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Comprehensive Benchmark Table
st.divider()
st.subheader("📐 Standard Label Distribution Learning (LDL) Metrics")

m_df = pd.DataFrame([
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Cosine Similarity", "Value": f"{metrics['Cosine']:.5f}", "Formula": "P · Q / (||P|| ||Q||)"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Pearson Correlation (r)", "Value": f"{metrics['Pearson']:.5f}", "Formula": "Cov(P, Q) / (σP · σQ)"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Histogram Overlap / Intersection", "Value": f"{metrics['Overlap']:.5f}", "Formula": "∑ min(P_i, Q_i)"},
    {"Category": "Similarities (Optimal → 1.0)", "Metric": "Fidelity (Bhattacharyya Coeff.)", "Value": f"{metrics['Fidelity']:.5f}", "Formula": "∑ √(P_i · Q_i)"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Jensen-Shannon Divergence (JSD)", "Value": f"{metrics['JSD']:.5f}", "Formula": "0.5·KL(P||M) + 0.5·KL(Q||M)"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Kullback-Leibler Divergence (KL)", "Value": f"{metrics['KL_Divergence']:.5f}", "Formula": "∑ P_i · ln(P_i / Q_i)"},
    {"Category": "Divergences (Optimal → 0.0)", "Metric": "Total Variation (L1)", "Value": f"{metrics['Total_Variation']:.5f}", "Formula": "0.5 · ∑ |P_i - Q_i|"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Euclidean Distance (L2)", "Value": f"{metrics['Euclidean_L2']:.5f}", "Formula": "√(∑ (P_i - Q_i)²)"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Chebyshev Distance (L∞)", "Value": f"{metrics['Chebyshev_Linf']:.5f}", "Formula": "max |P_i - Q_i|"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Hellinger Distance", "Value": f"{metrics['Hellinger']:.5f}", "Formula": "√[ 0.5 · ∑ (√P_i - √Q_i)² ]"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Bhattacharyya Distance", "Value": f"{metrics['Bhattacharyya']:.5f}", "Formula": "-ln(Fidelity)"},
    {"Category": "Distances (Optimal → 0.0)", "Metric": "Chi-Square Distance (χ²)", "Value": f"{metrics['Chi2']:.5f}", "Formula": "∑ [ (P_i - Q_i)² / (P_i + Q_i) ]"}
])

st.dataframe(m_df, use_container_width=True, hide_index=True)

# Sample-by-Sample Inspection (If mapping is available)
st.divider()
st.subheader("🔍 Sample-by-Sample Image Inspection")

if df_map is not None and not df_map.empty:
    df_model['basename'] = df_model['file'].apply(os.path.basename)
    
    # Merge human annotations with mapping and model predictions
    df_merged = df_ratings.groupby(['image_code', 'image_filename'])[RATINGS_COLS].mean().reset_index()
    df_merged = df_merged.merge(df_map, on=['image_code', 'image_filename'])
    df_merged = df_merged.merge(df_model, left_on='affectnet_file', right_on='basename')

    if not df_merged.empty:
        selected_code = st.selectbox("Select mapped image instance:", df_merged['image_code'].unique())
        row = df_merged[df_merged['image_code'] == selected_code].iloc[0]

        img_p = normalize_distribution(row[RATINGS_COLS].values)
        img_q = normalize_distribution(row[EMOTIONS_7].values)
        img_metrics = compute_ldl_metrics(img_p, img_q)

        i_col1, i_col2 = st.columns([1, 2])
        with i_col1:
            st.markdown(f"**Image Code:** `{row['image_code']}`")
            st.markdown(f"**Filename:** `{row['image_filename']}`")
            st.markdown(f"**AffectNet Source:** `{row['affectnet_file']}`")
            st.markdown(f"**Sample JSD:** `{img_metrics['JSD']:.4f}`")
            st.markdown(f"**Sample Cosine:** `{img_metrics['Cosine']:.4f}`")
            st.markdown(f"**Sample Overlap:** `{img_metrics['Overlap']:.4f}`")
        
        with i_col2:
            fig_inst = go.Figure(data=[
                go.Bar(name='Human', x=EMOTIONS_7, y=img_p, marker_color='#4f46e5'),
                go.Bar(name='Model', x=EMOTIONS_7, y=img_q, marker_color='#10b981')
            ])
            fig_inst.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Probability")
            st.plotly_chart(fig_inst, use_container_width=True)
    else:
        st.info("No matching records found between the provided cross-reference mapping and model outputs.")
else:
    st.info("To inspect individual image instances, upload a valid `mapping_ratings_affectnet.csv` file in the sidebar.")