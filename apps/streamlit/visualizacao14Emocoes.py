import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import base64
from io import BytesIO, StringIO
import json
import tempfile
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Emotion Model Comparison Tool", layout="wide")
st.title("📊 Comparative Analysis: Model Predictions vs Ground Truth")

# Set Matplotlib to use vector graphics backend
plt.switch_backend('Agg')

# Function to process the data
@st.cache_data
def process_data(preds_file, labels_file):
    """Process uploaded CSV files"""
    # Read predictions file
    preds = pd.read_csv(preds_file)
    
    # Read ground truth file
    labels = pd.read_csv(labels_file)
    
    # Automatically identify emotion columns
    emotion_cols = []
    
    # For predictions: columns that are not 'file' or end with 'label'
    for col in preds.columns:
        if col not in ['file', 'emotion_label'] and not col.startswith('valence') and not col.startswith('arousal') and not col.startswith('dominance'):
            emotion_cols.append(col)
    
    # Sort emotion columns for consistency
    emotion_cols = sorted(emotion_cols)
    
    # For predictions: take only emotion columns
    preds_emotions = preds[emotion_cols].copy()
    # For labels: take only emotion columns
    labels_emotions = labels[emotion_cols].copy()
    
    # Add identification columns
    preds_emotions['file'] = preds.get('file', '')
    
    # Try to identify predicted class column
    pred_class_col = None
    for col in preds.columns:
        if 'label' in col.lower() and col not in emotion_cols:
            pred_class_col = col
            break
    
    if pred_class_col:
        preds_emotions['pred_class'] = preds[pred_class_col]
    else:
        # If not found, use argmax of probabilities
        preds_emotions['pred_class'] = preds_emotions[emotion_cols].idxmax(axis=1)
        preds_emotions['pred_class'] = preds_emotions['pred_class'].apply(
            lambda x: emotion_cols.index(x) if x in emotion_cols else 0
        )
    
    labels_emotions['file'] = labels.get('file', '')
    
    # Try to identify true class column
    true_class_col = None
    for col in labels.columns:
        if 'label' in col.lower() and col not in emotion_cols:
            true_class_col = col
            break
    
    if true_class_col:
        labels_emotions['true_class'] = labels[true_class_col]
    else:
        labels_emotions['true_class'] = labels_emotions[emotion_cols].idxmax(axis=1)
        labels_emotions['true_class'] = labels_emotions['true_class'].apply(
            lambda x: emotion_cols.index(x) if x in emotion_cols else 0
        )
    
    return preds_emotions, labels_emotions, emotion_cols

def create_vector_plotly_figure(fig, format='svg'):
    """Convert Plotly figure to vector format"""
    if format == 'svg':
        return fig.to_image(format='svg')
    elif format == 'pdf':
        return fig.to_image(format='pdf')
    elif format == 'eps':
        # EPS requires conversion through SVG
        svg_bytes = fig.to_image(format='svg')
        # Note: Direct EPS export not available in Plotly
        # We'll return SVG for now
        return svg_bytes

def create_matplotlib_figure(fig, format='pdf'):
    """Save Matplotlib figure to vector format"""
    buf = BytesIO()
    if format == 'pdf':
        fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
    elif format == 'svg':
        fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
    elif format == 'eps':
        fig.savefig(buf, format='eps', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def create_comparison_chart_matplotlib(pred_sample, true_sample, emotion_cols, sample_idx, 
                                       figsize=(12, 6), font_sizes=None):
    """Create comparison chart using Matplotlib for vector export"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'xtick_labels': 11,
            'ytick_labels': 11,
            'legend': 11,
            'bar_labels': 8
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(emotion_cols))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pred_sample.values.astype(float), width, 
                   label='Predicted', color='blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, true_sample.values.astype(float), width, 
                   label='Ground Truth', color='red', alpha=0.8)
    
    ax.set_xlabel('Emotion', fontsize=font_sizes['axes_labels'])
    ax.set_ylabel('Probability', fontsize=font_sizes['axes_labels'])
    ax.set_title(f'Distribution Comparison - Sample {sample_idx}', fontsize=font_sizes['title'])
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_cols, rotation=45, ha='right', fontsize=font_sizes['xtick_labels'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick_labels'])
    ax.legend(fontsize=font_sizes['legend'])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if height is significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=font_sizes['bar_labels'])
    
    plt.tight_layout()
    return fig

def create_radar_chart_matplotlib(pred_sample, true_sample, emotion_cols, sample_idx, font_sizes=None):
    """Create radar chart using Matplotlib for vector export"""
    if font_sizes is None:
        font_sizes = {
            'title': 16,
            'tick_labels': 12,
            'legend': 11,
            'radial_labels': 10
        }
    
    angles = np.linspace(0, 2 * np.pi, len(emotion_cols), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    pred_values = pred_sample.values.astype(float).tolist()
    pred_values += pred_values[:1]
    
    true_values = true_sample.values.astype(float).tolist()
    true_values += true_values[:1]
    
    ax.plot(angles, pred_values, 'o-', linewidth=2, label='Predicted')
    ax.fill(angles, pred_values, alpha=0.25)
    
    ax.plot(angles, true_values, 'o-', linewidth=2, label='Ground Truth')
    ax.fill(angles, true_values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotion_cols, fontsize=font_sizes['tick_labels'])
    ax.tick_params(axis='y', labelsize=font_sizes['radial_labels'])
    ax.set_ylim(0, max(max(pred_values), max(true_values)) * 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=font_sizes['legend'])
    ax.set_title(f'Radar Chart - Sample {sample_idx}', size=font_sizes['title'], y=1.1)
    
    plt.tight_layout()
    return fig

def create_global_comparison_matplotlib(pred_means, label_means, emotion_cols, figsize=(14, 8), font_sizes=None):
    """Create global comparison chart using Matplotlib"""
    if font_sizes is None:
        font_sizes = {
            'title': 16,
            'axes_labels': 12,
            'xtick_labels': 11,
            'ytick_labels': 11,
            'legend': 11
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(emotion_cols))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pred_means.values, width, 
                   label='Predicted Average', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, label_means.values, width, 
                   label='Ground Truth Average', color='salmon', alpha=0.8)
    
    ax.set_xlabel('Emotion', fontsize=font_sizes['axes_labels'])
    ax.set_ylabel('Average Probability', fontsize=font_sizes['axes_labels'])
    ax.set_title('Global Averages of Distributions', fontsize=font_sizes['title'])
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_cols, rotation=45, ha='right', fontsize=font_sizes['xtick_labels'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick_labels'])
    ax.legend(fontsize=font_sizes['legend'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_matplotlib(cm, emotion_cols, figsize=(12, 10), font_sizes=None):
    """Create confusion matrix using Matplotlib"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'tick_labels': 10,
            'cell_values': 8
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[f'Pred {i}' for i in range(len(emotion_cols))],
           yticklabels=[f'True {i}' for i in range(len(emotion_cols))],
           title='Confusion Matrix',
           ylabel='True Class',
           xlabel='Predicted Class')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set font sizes
    ax.title.set_fontsize(font_sizes['title'])
    ax.xaxis.label.set_fontsize(font_sizes['axes_labels'])
    ax.yaxis.label.set_fontsize(font_sizes['axes_labels'])
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick_labels'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=font_sizes['cell_values'])
    
    plt.tight_layout()
    return fig

def create_similarity_histogram_matplotlib(similarities, figsize=(10, 6), font_sizes=None):
    """Create similarity histogram using Matplotlib"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'tick_labels': 11,
            'legend': 11
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n, bins, patches = ax.hist(similarities, bins=20, color='teal', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Similarity', fontsize=font_sizes['axes_labels'])
    ax.set_ylabel('Frequency', fontsize=font_sizes['axes_labels'])
    ax.set_title('Distribution of Sample Similarities', fontsize=font_sizes['title'])
    ax.tick_params(axis='both', labelsize=font_sizes['tick_labels'])
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    ax.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    ax.axvline(median_sim, color='green', linestyle='--', label=f'Median: {median_sim:.3f}')
    ax.legend(fontsize=font_sizes['legend'])
    
    plt.tight_layout()
    return fig

# Sidebar for file upload
st.sidebar.header("📁 File Upload")

# File uploaders
preds_file = st.sidebar.file_uploader(
    "Model Predictions File",
    type=['csv'],
    help="Select the CSV file with model predictions"
)

labels_file = st.sidebar.file_uploader(
    "Ground Truth Labels File",
    type=['csv'],
    help="Select the CSV file with ground truth labels"
)

# Export settings
st.sidebar.header("📤 Export Settings")
export_format = st.sidebar.selectbox(
    "Preferred Export Format",
    ["PDF (Recommended)", "SVG", "EPS", "PNG (High-Res)"],
    help="Choose vector format for high-quality export"
)

# Chart font size settings
st.sidebar.header("🔤 Chart Font Sizes")

# Font size controls for bar charts
st.sidebar.subheader("Bar Chart Font Sizes")
bar_title_size = st.sidebar.slider("Title Size", 8, 24, 14)
bar_axes_size = st.sidebar.slider("Axes Labels Size", 8, 20, 12)
bar_tick_size = st.sidebar.slider("Tick Labels Size", 6, 18, 11)
bar_legend_size = st.sidebar.slider("Legend Size", 6, 18, 11)
bar_label_size = st.sidebar.slider("Bar Labels Size", 6, 14, 8)

# Font size controls for other charts
st.sidebar.subheader("Other Charts Font Sizes")
other_title_size = st.sidebar.slider("Other Charts Title Size", 8, 24, 14)
other_labels_size = st.sidebar.slider("Other Charts Labels Size", 8, 20, 11)

# Create font size dictionaries
bar_font_sizes = {
    'title': bar_title_size,
    'axes_labels': bar_axes_size,
    'xtick_labels': bar_tick_size,
    'ytick_labels': bar_tick_size,
    'legend': bar_legend_size,
    'bar_labels': bar_label_size
}

global_font_sizes = {
    'title': other_title_size,
    'axes_labels': other_labels_size,
    'xtick_labels': bar_tick_size,
    'ytick_labels': bar_tick_size,
    'legend': bar_legend_size
}

radar_font_sizes = {
    'title': other_title_size,
    'tick_labels': other_labels_size,
    'legend': bar_legend_size,
    'radial_labels': other_labels_size
}

cm_font_sizes = {
    'title': other_title_size,
    'axes_labels': other_labels_size,
    'tick_labels': bar_tick_size,
    'cell_values': bar_label_size
}

hist_font_sizes = {
    'title': other_title_size,
    'axes_labels': other_labels_size,
    'tick_labels': bar_tick_size,
    'legend': bar_legend_size
}

# Image directory input
st.sidebar.header("🖼️ Image Visualization Settings")
image_dir = st.sidebar.text_input(
    "Image Directory Path",
    help="Enter the path to the directory containing images (e.g., /path/to/train_set/images)"
)

# Analysis settings
st.sidebar.header("⚙️ Analysis Settings")

# Select specific columns if files are uploaded
if preds_file is not None and labels_file is not None:
    preds_preview = pd.read_csv(preds_file)
    labels_preview = pd.read_csv(labels_file)
    
    preds_file.seek(0)
    labels_file.seek(0)
    
    st.sidebar.info(f"**Predictions File:** {preds_preview.shape[0]} rows, {preds_preview.shape[1]} columns")
    st.sidebar.info(f"**Labels File:** {labels_preview.shape[0]} rows, {labels_preview.shape[1]} columns")
    
    st.sidebar.subheader("🔧 Advanced Configuration")
    
    detection_mode = st.sidebar.radio(
        "Column detection mode",
        ["Automatic", "Manual"]
    )
    
    if detection_mode == "Manual":
        emotion_cols_input = st.sidebar.text_input(
            "Emotion columns (comma-separated)",
            value="happy,contempt,elated,surprised,loved,protected,astonished,disgusted,angry,fearful,sad,neutral",
            help="Enter exact names of emotion columns, separated by commas"
        )
        emotion_cols_manual = [col.strip() for col in emotion_cols_input.split(',')]
        
        pred_class_col_manual = st.sidebar.text_input(
            "Predicted class column",
            value="emotion_label",
            help="Name of column containing predicted class"
        )
        
        true_class_col_manual = st.sidebar.text_input(
            "True class column",
            value="emotion_label",
            help="Name of column containing true class"
        )

metric_choice = st.sidebar.selectbox(
    "Global Similarity Metric",
    ["Jensen-Shannon Divergence", "Wasserstein Distance", "Cosine Similarity", "Pearson Correlation"]
)

show_individual = st.sidebar.checkbox("Show Individual Sample Analysis", value=True)
top_k = st.sidebar.slider("Top-K for Rank", 1, 10, 3)

# Check if files were uploaded
if preds_file is not None and labels_file is not None:
    try:
        with st.spinner("Processing data..."):
            preds_df, labels_df, emotion_cols = process_data(preds_file, labels_file)
        
        if detection_mode == "Manual" and 'emotion_cols_manual' in locals():
            available_cols = []
            for col in emotion_cols_manual:
                if col in preds_df.columns and col in labels_df.columns:
                    available_cols.append(col)
            
            if available_cols:
                emotion_cols = available_cols
                st.sidebar.success(f"Using {len(emotion_cols)} emotion columns")
            else:
                st.sidebar.warning("None of the specified columns were found. Using automatic detection.")
        
        metric_map = {
            "Jensen-Shannon Divergence": "js",
            "Wasserstein Distance": "wasserstein",
            "Cosine Similarity": "cosine",
            "Pearson Correlation": "pearson"
        }
        
        def calculate_global_similarity(preds, labels, emotion_cols, metric="js"):
            similarities = []
            
            for i in range(len(preds)):
                p = preds.iloc[i][emotion_cols].values.astype(float)
                l = labels.iloc[i][emotion_cols].values.astype(float)
                
                if metric == "js":
                    js = jensenshannon(p, l) ** 2
                    similarities.append(1 - js if not np.isnan(js) else 0)
                elif metric == "wasserstein":
                    wd = wasserstein_distance(p, l)
                    similarities.append(1 / (1 + wd))
                elif metric == "cosine":
                    cos_sim = np.dot(p, l) / (np.linalg.norm(p) * np.linalg.norm(l))
                    similarities.append(cos_sim if not np.isnan(cos_sim) else 0)
                elif metric == "pearson":
                    corr, _ = spearmanr(p, l)
                    similarities.append(corr if not np.isnan(corr) else 0)
            
            return np.mean(similarities), similarities

        def calculate_accuracy(preds, labels, emotion_cols):
            pred_classes = []
            true_classes = []
            
            for i in range(len(preds)):
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                pred_class = np.argmax(pred_probs)
                pred_classes.append(pred_class)
                
                true_class = labels.iloc[i]['true_class']
                if isinstance(true_class, str):
                    try:
                        true_class = int(float(true_class))
                    except:
                        true_probs = labels.iloc[i][emotion_cols].values.astype(float)
                        true_class = np.argmax(true_probs)
                else:
                    true_class = int(true_class)
                true_classes.append(true_class)
            
            accuracy = accuracy_score(true_classes, pred_classes)
            
            top_k_correct = 0
            for i in range(len(preds)):
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                true_class = true_classes[i]
                
                top_k_indices = np.argsort(pred_probs)[-top_k:][::-1]
                if true_class in top_k_indices:
                    top_k_correct += 1
            
            top_k_accuracy = top_k_correct / len(preds)
            
            return accuracy, top_k_accuracy, pred_classes, true_classes

        def calculate_rank_metrics(preds, labels, emotion_cols, true_classes):
            rank_positions = []
            
            for i in range(len(preds)):
                pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
                true_class = true_classes[i]
                
                sorted_indices = np.argsort(pred_probs)[::-1]
                
                rank_position = np.where(sorted_indices == true_class)[0]
                if len(rank_position) > 0:
                    rank_positions.append(rank_position[0] + 1)
            
            mean_rank = np.mean(rank_positions) if rank_positions else 0
            median_rank = np.median(rank_positions) if rank_positions else 0
            
            rank_distribution = {}
            for rank in rank_positions:
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
            
            return mean_rank, median_rank, rank_distribution, rank_positions
        
        similarity_metric = metric_map[metric_choice]
        global_similarity, individual_similarities = calculate_global_similarity(
            preds_df, labels_df, emotion_cols, similarity_metric
        )
        accuracy, top_k_accuracy, pred_classes, true_classes = calculate_accuracy(
            preds_df, labels_df, emotion_cols
        )
        mean_rank, median_rank, rank_distribution, rank_positions = calculate_rank_metrics(
            preds_df, labels_df, emotion_cols, true_classes
        )
        
        # Calculate global averages for export
        pred_means = preds_df[emotion_cols].apply(pd.to_numeric, errors='coerce').mean()
        label_means = labels_df[emotion_cols].apply(pd.to_numeric, errors='coerce').mean()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Global Analysis", "🖼️ Image & Distribution", "📋 Detailed Metrics", "📤 Export Center"])
        
        with tab1:
            st.success("✅ Data loaded and processed successfully!")
            
            with st.expander("📋 View Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Predictions (first 5 rows):**")
                    st.dataframe(preds_df.head())
                with col2:
                    st.write("**Ground Truth (first 5 rows):**")
                    st.dataframe(labels_df.head())
            
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Global Similarity",
                    value=f"{global_similarity:.3f}",
                    delta=f"Metric: {metric_choice}"
                )
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=global_similarity * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Global Similarity (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.metric(
                    label="Accuracy (Top-1)",
                    value=f"{accuracy:.3f}"
                )
                st.metric(
                    label=f"Accuracy (Top-{top_k})",
                    value=f"{top_k_accuracy:.3f}"
                )
                
                fig_acc = go.Figure(data=[
                    go.Bar(
                        name='Accuracy',
                        x=['Top-1', f'Top-{top_k}'],
                        y=[accuracy, top_k_accuracy],
                        marker_color=['blue', 'green']
                    )
                ])
                fig_acc.update_layout(
                    title="Accuracy by Metric",
                    yaxis_title="Accuracy",
                    yaxis_range=[0, 1],
                    height=300
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col3:
                st.metric(
                    label="Mean Rank",
                    value=f"{mean_rank:.2f}"
                )
                st.metric(
                    label="Median Rank",
                    value=f"{median_rank:.0f}"
                )
                
                if rank_distribution:
                    ranks = list(rank_distribution.keys())
                    counts = list(rank_distribution.values())
                    
                    fig_rank = go.Figure(data=[
                        go.Bar(
                            x=[f'Rank {r}' for r in ranks],
                            y=counts,
                            marker_color='purple'
                        )
                    ])
                    fig_rank.update_layout(
                        title="Rank Distribution",
                        xaxis_title="Rank",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)

            st.subheader("🎯 Confusion Matrix")
            try:
                cm = confusion_matrix(true_classes, pred_classes, labels=range(len(emotion_cols)))
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred {i}" for i in range(len(emotion_cols))],
                    y=[f"True {i}" for i in range(len(emotion_cols))],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Class",
                    yaxis_title="True Class",
                    height=500
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate confusion matrix: {e}")

        with tab2:
            st.subheader("🖼️ Image Visualization with Distribution Comparison")
            
            if image_dir and os.path.exists(image_dir):
                sample_idx = st.slider("Select Sample", 0, len(preds_df)-1, 0)
                filename = preds_df.iloc[sample_idx]['file']
                
                if pd.isna(filename) or filename == '':
                    st.warning("No filename found for this sample.")
                else:
                    image_path = os.path.join(image_dir, filename)
                    
                    if os.path.exists(image_path):
                        try:
                            image = Image.open(image_path)
                            
                            col_img, col_bar, col_radar = st.columns([1, 2, 2])
                            
                            with col_img:
                                st.image(image, caption=f"Image: {filename}", use_column_width=True)
                                
                                st.markdown("### Sample Metrics")
                                st.info(f"""
                                **Sample {sample_idx}:**
                                - Similarity: {individual_similarities[sample_idx]:.3f}
                                - Predicted Class: {pred_classes[sample_idx]} ({emotion_cols[pred_classes[sample_idx]] if pred_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                                - True Class: {true_classes[sample_idx]} ({emotion_cols[true_classes[sample_idx]] if true_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                                - Rank: {rank_positions[sample_idx] if sample_idx < len(rank_positions) else 'N/A'}
                                """)
                            
                            with col_bar:
                                pred_sample = preds_df.iloc[sample_idx][emotion_cols]
                                true_sample = labels_df.iloc[sample_idx][emotion_cols]
                                
                                fig_bar = go.Figure(data=[
                                    go.Bar(
                                        x=emotion_cols,
                                        y=pred_sample.values.astype(float),
                                        name='Predicted',
                                        marker_color='blue'
                                    ),
                                    go.Bar(
                                        x=emotion_cols,
                                        y=true_sample.values.astype(float),
                                        name='Ground Truth',
                                        marker_color='red'
                                    )
                                ])
                                fig_bar.update_layout(
                                    title=f"Distribution Comparison - Sample {sample_idx}",
                                    xaxis_title="Emotion",
                                    yaxis_title="Probability",
                                    barmode='group',
                                    height=400
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            with col_radar:
                                fig_radar = go.Figure()
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=pred_sample.values.astype(float),
                                    theta=emotion_cols,
                                    fill='toself',
                                    name='Predicted'
                                ))
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=true_sample.values.astype(float),
                                    theta=emotion_cols,
                                    fill='toself',
                                    name='Ground Truth'
                                ))
                                
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, max(max(pred_sample.values.astype(float)), max(true_sample.values.astype(float)))]
                                        )),
                                    showlegend=True,
                                    title=f"Radar Visualization - Sample {sample_idx}",
                                    height=400
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                            
                            # Navigation
                            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
                            with col_nav1:
                                if sample_idx > 0:
                                    if st.button("◀ Previous"):
                                        st.session_state['sample_idx'] = sample_idx - 1
                                        st.experimental_rerun()
                            with col_nav2:
                                st.write(f"Sample {sample_idx + 1} of {len(preds_df)}")
                            with col_nav3:
                                if sample_idx < len(preds_df) - 1:
                                    if st.button("Next ▶"):
                                        st.session_state['sample_idx'] = sample_idx + 1
                                        st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
                    else:
                        st.warning(f"Image not found at: {image_path}")
            else:
                st.info("Please enter a valid image directory path in the sidebar to enable image visualization.")
                
                if show_individual:
                    sample_idx = st.slider("Select Sample", 0, len(preds_df)-1, 0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pred_sample = preds_df.iloc[sample_idx][emotion_cols]
                        true_sample = labels_df.iloc[sample_idx][emotion_cols]
                        
                        fig_pred = go.Figure(data=[
                            go.Bar(
                                x=emotion_cols,
                                y=pred_sample.values.astype(float),
                                name='Predicted',
                                marker_color='blue'
                            ),
                            go.Bar(
                                x=emotion_cols,
                                y=true_sample.values.astype(float),
                                name='Ground Truth',
                                marker_color='red'
                            )
                        ])
                        fig_pred.update_layout(
                            title=f"Distribution - Sample {sample_idx}",
                            xaxis_title="Emotion",
                            yaxis_title="Probability",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        st.info(f"""
                        **Sample {sample_idx} Metrics:**
                        - Similarity: {individual_similarities[sample_idx]:.3f}
                        - Predicted Class: {pred_classes[sample_idx]} ({emotion_cols[pred_classes[sample_idx]] if pred_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                        - True Class: {true_classes[sample_idx]} ({emotion_cols[true_classes[sample_idx]] if true_classes[sample_idx] < len(emotion_cols) else 'N/A'})
                        - Rank: {rank_positions[sample_idx] if sample_idx < len(rank_positions) else 'N/A'}
                        """)

                    with col2:
                        fig_radar = go.Figure()
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=pred_sample.values.astype(float),
                            theta=emotion_cols,
                            fill='toself',
                            name='Predicted'
                        ))
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=true_sample.values.astype(float),
                            theta=emotion_cols,
                            fill='toself',
                            name='Ground Truth'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max(max(pred_sample.values.astype(float)), max(true_sample.values.astype(float)))]
                                )),
                            showlegend=True,
                            title=f"Radar Visualization - Sample {sample_idx}",
                            height=400
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

        with tab3:
            st.subheader("📋 Detailed Metrics")
            
            st.subheader("🌍 Global Distribution Analysis")
            
            fig_global = go.Figure(data=[
                go.Bar(
                    x=emotion_cols,
                    y=pred_means.values,
                    name='Predicted Average',
                    marker_color='lightblue'
                ),
                go.Bar(
                    x=emotion_cols,
                    y=label_means.values,
                    name='Ground Truth Average',
                    marker_color='salmon'
                )
            ])

            fig_global.update_layout(
                title="Global Averages of Distributions",
                xaxis_title="Emotion",
                yaxis_title="Average Probability",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_global, use_container_width=True)
            
            st.subheader("📊 Detailed Metrics by Class")
            
            try:
                class_report = classification_report(
                    true_classes, 
                    pred_classes, 
                    target_names=[f"{emotion_cols[i] if i < len(emotion_cols) else f'Class {i}'}" for i in range(len(emotion_cols))],
                    output_dict=True
                )
                
                report_df = pd.DataFrame(class_report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
            except Exception as e:
                st.warning(f"Could not generate detailed report: {e}")
            
            st.subheader("📈 Distribution of Individual Similarities")
            
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=individual_similarities,
                    nbinsx=20,
                    marker_color='teal',
                    opacity=0.7
                )
            ])
            fig_hist.update_layout(
                title="Distribution of Sample Similarities",
                xaxis_title="Similarity",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab4:
            st.subheader("📤 Export Center")
            st.info("Export high-quality vector graphics for publication and presentations")
            
            # Get format extension
            format_map = {
                "PDF (Recommended)": "pdf",
                "SVG": "svg",
                "EPS": "eps",
                "PNG (High-Res)": "png"
            }
            selected_format = format_map[export_format]
            
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.subheader("📊 Current Sample Charts")
                
                if 'sample_idx' in locals():
                    pred_sample = preds_df.iloc[sample_idx][emotion_cols]
                    true_sample = labels_df.iloc[sample_idx][emotion_cols]
                    
                    # Export Comparison Chart
                    st.markdown("**Distribution Comparison Chart**")
                    matplotlib_fig = create_comparison_chart_matplotlib(
                        pred_sample, true_sample, emotion_cols, sample_idx,
                        font_sizes=bar_font_sizes
                    )
                    export_buffer = create_matplotlib_figure(matplotlib_fig, selected_format)
                    
                    st.download_button(
                        label=f"📥 Download Comparison Chart ({selected_format.upper()})",
                        data=export_buffer,
                        file_name=f"comparison_sample_{sample_idx}_{timestamp}.{selected_format}",
                        mime=f"application/{selected_format}" if selected_format != 'png' else "image/png"
                    )
                    
                    # Export Radar Chart
                    st.markdown("**Radar Chart**")
                    radar_fig = create_radar_chart_matplotlib(
                        pred_sample, true_sample, emotion_cols, sample_idx,
                        font_sizes=radar_font_sizes
                    )
                    radar_buffer = create_matplotlib_figure(radar_fig, selected_format)
                    
                    st.download_button(
                        label=f"📥 Download Radar Chart ({selected_format.upper()})",
                        data=radar_buffer,
                        file_name=f"radar_sample_{sample_idx}_{timestamp}.{selected_format}",
                        mime=f"application/{selected_format}" if selected_format != 'png' else "image/png"
                    )
                    
                    # Close figures to free memory
                    plt.close(matplotlib_fig)
                    plt.close(radar_fig)
            
            with col_export2:
                st.subheader("🌍 Global Analysis Charts")
                
                # Export Global Comparison
                st.markdown("**Global Averages Chart**")
                global_fig = create_global_comparison_matplotlib(
                    pred_means, label_means, emotion_cols,
                    font_sizes=global_font_sizes
                )
                global_buffer = create_matplotlib_figure(global_fig, selected_format)
                
                st.download_button(
                    label=f"📥 Download Global Averages ({selected_format.upper()})",
                    data=global_buffer,
                    file_name=f"global_averages_{timestamp}.{selected_format}",
                    mime=f"application/{selected_format}" if selected_format != 'png' else "image/png"
                )
                
                # Export Confusion Matrix
                st.markdown("**Confusion Matrix**")
                if 'cm' in locals():
                    cm_fig = create_confusion_matrix_matplotlib(cm, emotion_cols,
                                                                font_sizes=cm_font_sizes)
                    cm_buffer = create_matplotlib_figure(cm_fig, selected_format)
                    
                    st.download_button(
                        label=f"📥 Download Confusion Matrix ({selected_format.upper()})",
                        data=cm_buffer,
                        file_name=f"confusion_matrix_{timestamp}.{selected_format}",
                        mime=f"application/{selected_format}" if selected_format != 'png' else "image/png"
                    )
                
                # Export Similarity Histogram
                st.markdown("**Similarity Distribution**")
                hist_fig = create_similarity_histogram_matplotlib(individual_similarities,
                                                                 font_sizes=hist_font_sizes)
                hist_buffer = create_matplotlib_figure(hist_fig, selected_format)
                
                st.download_button(
                    label=f"📥 Download Similarity Histogram ({selected_format.upper()})",
                    data=hist_buffer,
                    file_name=f"similarity_histogram_{timestamp}.{selected_format}",
                    mime=f"application/{selected_format}" if selected_format != 'png' else "image/png"
                )
                
                # Close figures
                plt.close(global_fig)
                if 'cm_fig' in locals():
                    plt.close(cm_fig)
                plt.close(hist_fig)
            
            # Export All Data
            st.subheader("📁 Complete Dataset Export")
            
            col_data1, col_data2, col_data3 = st.columns(3)
            
            with col_data1:
                # Export processed data as CSV
                combined_data = pd.DataFrame({
                    'file': preds_df['file'],
                    'true_class': true_classes,
                    'pred_class': pred_classes,
                    'similarity': individual_similarities,
                    'correct': [1 if t == p else 0 for t, p in zip(true_classes, pred_classes)]
                })
                
                csv_data = combined_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download Analysis Results (CSV)",
                    data=csv_data,
                    file_name=f"analysis_results_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col_data2:
                # Export statistics summary
                summary_stats = {
                    'metric': ['Global Similarity', 'Top-1 Accuracy', f'Top-{top_k} Accuracy', 
                              'Mean Rank', 'Median Rank', 'Total Samples', 'AffectNet Samples'],
                    'value': [global_similarity, accuracy, top_k_accuracy, mean_rank, median_rank,
                             len(preds_df), len(preds_df)]  # Add actual counts if available
                }
                summary_df = pd.DataFrame(summary_stats)
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Statistics Summary (CSV)",
                    data=summary_csv,
                    file_name=f"statistics_summary_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col_data3:
                # Export configuration
                config_data = {
                    'export_timestamp': timestamp,
                    'export_format': export_format,
                    'similarity_metric': metric_choice,
                    'top_k': top_k,
                    'emotion_columns': emotion_cols,
                    'font_sizes': {
                        'bar_chart': bar_font_sizes,
                        'global_chart': global_font_sizes,
                        'radar_chart': radar_font_sizes
                    },
                    'total_samples': len(preds_df),
                    'global_similarity': float(global_similarity),
                    'accuracy': float(accuracy)
                }
                config_json = json.dumps(config_data, indent=2)
                
                st.download_button(
                    label="⚙️ Download Configuration (JSON)",
                    data=config_json,
                    file_name=f"export_config_{timestamp}.json",
                    mime="application/json"
                )
            
            # Batch Export Options
            st.subheader("🎯 Batch Export Options")
            
            if st.button("🔄 Generate All Charts in Current Format"):
                with st.spinner("Generating all charts..."):
                    # Create a temporary directory for all exports
                    with tempfile.TemporaryDirectory() as tmpdir:
                        files_to_zip = []
                        
                        # Export all individual comparison charts
                        num_samples = min(10, len(preds_df))  # Limit to 10 samples for performance
                        for i in range(num_samples):
                            pred_sample = preds_df.iloc[i][emotion_cols]
                            true_sample = labels_df.iloc[i][emotion_cols]
                            
                            # Comparison chart
                            comp_fig = create_comparison_chart_matplotlib(
                                pred_sample, true_sample, emotion_cols, i,
                                font_sizes=bar_font_sizes
                            )
                            comp_path = os.path.join(tmpdir, f"comparison_sample_{i}.{selected_format}")
                            comp_fig.savefig(comp_path, format=selected_format, dpi=300, bbox_inches='tight')
                            files_to_zip.append(comp_path)
                            plt.close(comp_fig)
                        
                        # Export global charts
                        global_fig = create_global_comparison_matplotlib(
                            pred_means, label_means, emotion_cols,
                            font_sizes=global_font_sizes
                        )
                        global_path = os.path.join(tmpdir, f"global_averages.{selected_format}")
                        global_fig.savefig(global_path, format=selected_format, dpi=300, bbox_inches='tight')
                        files_to_zip.append(global_path)
                        plt.close(global_fig)
                        
                        # Export confusion matrix
                        if 'cm' in locals():
                            cm_fig = create_confusion_matrix_matplotlib(cm, emotion_cols,
                                                                        font_sizes=cm_font_sizes)
                            cm_path = os.path.join(tmpdir, f"confusion_matrix.{selected_format}")
                            cm_fig.savefig(cm_path, format=selected_format, dpi=300, bbox_inches='tight')
                            files_to_zip.append(cm_path)
                            plt.close(cm_fig)
                        
                        st.success(f"Generated {len(files_to_zip)} charts in {selected_format.upper()} format")
                        
                        # Note: In Streamlit Cloud, direct ZIP creation might be limited
                        st.info("For batch downloads, please download individual charts or run locally for ZIP export")
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        1. Check if files have the same number of rows
        2. Confirm that emotion columns have the same names in both files
        3. Ensure class columns are numeric
        4. Use manual mode in the sidebar to specify columns
        """)
        
else:
    st.markdown("""
    ## Welcome to the Emotion Model Analyzer! 🤖
    
    **How to use:**
    1. **Upload files** in the left sidebar:
       - **Model Predictions**: Your model's results
       - **Ground Truth**: True labels
    
    2. **Configure analysis settings**
    
    3. **Enter image directory path** for visualization (optional)
    
    4. **View results** automatically
    
    **Expected Formats:**
    - CSV files with emotion columns (happy, sad, angry, etc.)
    - One column for class (typically 'emotion_label')
    - A 'file' column with image paths (optional)
    
    **Export Features:**
    - High-quality vector graphics (PDF, SVG, EPS)
    - Publication-ready charts
    - Batch export options
    - Comprehensive data export
    """)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .export-button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)