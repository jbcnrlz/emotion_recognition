import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import os

# Page configuration
st.set_page_config(page_title="3D Emotional Distribution Analysis", layout="wide")
st.title("3D Emotional Space Visualization")
st.write("Upload your files to explore data distributions across 3 dimensions (Valence, Arousal, and Dominance).")

# File Uploaders (3 columns)
col1, col2, col3 = st.columns(3)
with col1:
    features_file = st.file_uploader("Features File (e.g., features_test.csv)", type=["csv"])
with col2:
    distros_file = st.file_uploader("Distribution Metrics File (e.g., universal_distros.csv)", type=["csv"])
with col3:
    log_file = st.file_uploader("Dominance Pipeline Log (Optional)", type=["log", "txt"])

if features_file and distros_file:
    # Load DataFrames
    df_features = pd.read_csv(features_file)
    df_distros = pd.read_csv(distros_file)
    
    # Process Log file if uploaded
    log_data = {}
    if log_file:
        log_content = log_file.getvalue().decode("utf-8").splitlines()
        # Regex to parse the pipeline log format
        log_pattern = re.compile(r"Arquivo:\s*(.*?)\s*\| ID:\s*(\d+)\s*\| Emoção Lida:\s*(.*?)\s*\| Dominância Calculada:\s*([-\d.]+)")
        
        for line in log_content:
            match = log_pattern.search(line)
            if match:
                filename = match.group(1).strip()
                log_data[filename] = {
                    "id": match.group(2),
                    "ground_truth_emotion": match.group(3),
                    "calculated_dom": float(match.group(4))
                }
    
    # Inject 'neutral' class into distributions if missing
    if 'neutral' not in df_distros['class'].str.lower().values:
        neutral_row = pd.DataFrame([{
            'class': 'neutral',
            'valence mean': 0.0, 'valence std': 0.01,
            'arousal mean': 0.0, 'arousal std': 0.01,
            'dominance mean': 0.0, 'dominance std': 0.01
        }])
        df_distros = pd.concat([df_distros, neutral_row], ignore_index=True)

    # Strict emotion column mapping to avoid index/file name clipping
    emotion_cols = ['neutral', 'happy', 'sad', 'surprised', 'fearful', 'disgusted', 'angry', 'contempt']
    emotion_cols = [col for col in emotion_cols if col in df_features.columns]
    
    # Determine predicted emotion (highest probability/logit column)
    df_features['predicted_emotion'] = df_features[emotion_cols].idxmax(axis=1)
    
    # Map 'emotion_label' ID to its corresponding string class
    def map_label(val):
        try:
            return emotion_cols[int(val)]
        except:
            return "unknown"
            
    df_features['actual_emotion'] = df_features['emotion_label'].apply(map_label)
    
    # Extract clean baseline filename to match against the Log keys
    df_features['basename'] = df_features['file'].apply(lambda x: os.path.basename(str(x)))
    
    # --- SIDEBAR FILTERS & CONTROLS ---
    st.sidebar.header("3D Environment Controls")
    
    # NEW: View Mode Selector (All vs Only Inconsistent)
    view_mode = st.sidebar.radio(
        "Display Mode:",
        options=["Only Inconsistent Samples", "All Dataset Samples"],
        index=0
    )
    
    # Apply primary filter based on chosen mode
    if view_mode == "Only Inconsistent Samples":
        df_base = df_features[df_features['predicted_emotion'] != df_features['actual_emotion']].copy()
    else:
        df_base = df_features.copy()
        
    # Sidebar Categorical Multiselect Filters
    available_actual = sorted(df_base['actual_emotion'].unique().tolist())
    selected_actual = st.sidebar.multiselect(
        "Ground Truth Labels (Halo Color):",
        options=available_actual,
        default=available_actual
    )
    
    available_predicted = sorted(df_base['predicted_emotion'].unique().tolist())
    selected_predicted = st.sidebar.multiselect(
        "Model Predictions (Core Color):",
        options=available_predicted,
        default=available_predicted
    )
    
    # Final data filtering
    df_filtered = df_base[
        (df_base['actual_emotion'].isin(selected_actual)) & 
        (df_base['predicted_emotion'].isin(selected_predicted))
    ]
    
    st.subheader(f"Rendering {len(df_filtered)} samples in the 3D affective space.")
    if log_file:
        st.success(f"Log synced successfully: {len(log_data)} entries found. Hover over data points to review audit trails.")
    
    # --- COLOR MAP SETUP ---
    all_emotions = list(set(df_distros['class'].tolist() + emotion_cols))
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
    color_map = {emo: color_palette[i % len(color_palette)] for i, emo in enumerate(all_emotions)}
    color_map['neutral'] = 'gray'

    # --- PLOTLY 3D GRAPH BUILDING ---
    fig = go.Figure()

    # 1. Plot reference distribution centroids with error bars (2 Standard Deviations)
    for idx, row in df_distros.iterrows():
        emotion = row['class']
        v_mean, v_std = row['valence mean'], row['valence std']
        a_mean, a_std = row['arousal mean'], row['arousal std']
        d_mean, d_std = row['dominance mean'], row['dominance std']
        
        color = color_map.get(emotion, 'black')
        
        fig.add_trace(go.Scatter3d(
            x=[v_mean], y=[a_mean], z=[d_mean],
            mode='markers+text',
            marker=dict(symbol='diamond', size=8, color=color, line=dict(width=1, color='black')),
            name=f"Centroid: {emotion}",
            text=[emotion],
            textposition="top center",
            error_x=dict(type='constant', value=v_std*2, color=color, thickness=2, width=0),
            error_y=dict(type='constant', value=a_std*2, color=color, thickness=2, width=0),
            error_z=dict(type='constant', value=d_std*2, color=color, thickness=2, width=0),
            legendgroup=emotion,
            hoverinfo='skip'
        ))

    # 2. Plot Filtered Points (Dual layer workaround for WebGL 3D marker line limitation)
    if not df_filtered.empty:
        hover_texts = []
        for _, row in df_filtered.iterrows():
            b_name = row['basename']
            
            # Standard Tooltip construction
            text = (f"<b>File:</b> {b_name}<br>"
                    f"<b>Ground Truth (Halo):</b> {row['actual_emotion']}<br>"
                    f"<b>Prediction (Core):</b> {row['predicted_emotion']}<br>"
                    f"<b>Valence:</b> {row['valence']:.3f}<br>"
                    f"<b>Arousal:</b> {row['arousal']:.3f}<br>"
                    f"<b>Dominance:</b> {row['dominance']:.3f}")
            
            # Append pipeline audit log info if a filename match exists
            if b_name in log_data:
                info = log_data[b_name]
                text += (f"<br><br><b>--- Pipeline Audit Log ---</b><br>"
                         f"<b>Logged Emotion:</b> {info['ground_truth_emotion']}<br>"
                         f"<b>Logged Expression ID:</b> {info['id']}<br>"
                         f"<b>Calculated Dominance:</b> {info['calculated_dom']:.3f}")
                         
            hover_texts.append(text)

        fill_colors = df_filtered['predicted_emotion'].map(color_map).tolist()
        halo_colors = df_filtered['actual_emotion'].map(color_map).tolist()

        # Layer A: The "Halo" outer boundary representing Actual Annotation
        fig.add_trace(go.Scatter3d(
            x=df_filtered['valence'], y=df_filtered['arousal'], z=df_filtered['dominance'],
            mode='markers',
            marker=dict(size=12, color=halo_colors, opacity=0.3), 
            hoverinfo='skip', 
            showlegend=False
        ))

        # Layer B: The "Core" solid inner dot representing Model Prediction
        fig.add_trace(go.Scatter3d(
            x=df_filtered['valence'], y=df_filtered['arousal'], z=df_filtered['dominance'],
            mode='markers',
            marker=dict(size=6, color=fill_colors, opacity=1.0),
            text=hover_texts,
            hoverinfo='text',
            name="Samples (Core=Pred, Halo=Actual)"
        ))

    # Layout Configurations
    fig.update_layout(
        scene=dict(
            xaxis_title="Valence",
            yaxis_title="Arousal",
            zaxis_title="Dominance",
            xaxis=dict(range=[-1.2, 1.2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
            yaxis=dict(range=[-1.2, 1.2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
            zaxis=dict(range=[-1.2, 1.2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05)
    )

    # Render Chart
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload both the Features file and the Distribution metrics file to render the 3D space.")