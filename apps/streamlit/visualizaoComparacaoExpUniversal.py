import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.spatial.distance import jensenshannon as scipy_jensenshannon
from scipy.stats import entropy as scipy_entropy, pearsonr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import hashlib
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from itertools import cycle
import zipfile
import tempfile
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Multi-Experiment FER Comparison",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Multi-Experiment Facial Emotion Recognition Comparison")
st.markdown("""
This tool allows comparison of multiple FER experiments, each with its own ground truth pair.
Upload experiment pairs (predictions + ground truth) for comprehensive comparison.
Perfect for paper analysis with export-ready visualizations.
---
""")

# Color palette for consistent plotting across experiments
COLOR_PALETTE = px.colors.qualitative.Set3 + px.colors.qualitative.Set2 + px.colors.qualitative.Set1
MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon']

# Helper function to convert hex color to rgba
def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba format"""
    try:
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return f'rgba({r}, {g}, {b}, {alpha})'
    except:
        # Return default color if conversion fails
        return f'rgba(100, 149, 237, {alpha})'  # Cornflower blue

# Function to safely calculate KL divergence
def safe_kl_divergence(p, q, symmetric=True):
    """Calculates KL divergence safely, handling zeros"""
    try:
        p_arr = np.asarray(p, dtype=np.float64).flatten()
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        
        if len(p_arr) != len(q_arr):
            return np.nan
        
        # Add small epsilon to avoid zeros
        epsilon = 1e-12
        p_safe = p_arr + epsilon
        q_safe = q_arr + epsilon
        
        # Normalize to sum to 1
        p_sum = np.sum(p_safe)
        q_sum = np.sum(q_safe)
        
        if p_sum <= 0 or q_sum <= 0:
            return np.nan
            
        p_safe = p_safe / p_sum
        q_safe = q_safe / q_sum
        
        # Calculate KL divergence in both directions
        kl_forward = scipy_entropy(p_safe, q_safe)
        kl_backward = scipy_entropy(q_safe, p_safe)
        
        # Handle NaN and Inf
        if np.isnan(kl_forward) or np.isinf(kl_forward):
            kl_forward = np.nan
        if np.isnan(kl_backward) or np.isinf(kl_backward):
            kl_backward = np.nan
        
        if symmetric:
            # Symmetric KL (Jensen-Shannon symmetric KL)
            if np.isnan(kl_forward) or np.isnan(kl_backward):
                return np.nan
            return 0.5 * (kl_forward + kl_backward)
        else:
            return kl_forward if not np.isnan(kl_forward) else np.nan
    except Exception as e:
        return np.nan

# Function to safely calculate Jensen-Shannon divergence
def safe_js_divergence(p, q):
    """Calculates Jensen-Shannon divergence safely"""
    try:
        p_arr = np.asarray(p, dtype=np.float64).flatten()
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        
        if len(p_arr) != len(q_arr):
            return np.nan
        
        epsilon = 1e-12
        p_safe = p_arr + epsilon
        q_safe = q_arr + epsilon
        
        p_sum = np.sum(p_safe)
        q_sum = np.sum(q_safe)
        
        if p_sum <= 0 or q_sum <= 0:
            return np.nan
            
        p_safe = p_safe / p_sum
        q_safe = q_safe / q_sum
        
        js = scipy_jensenshannon(p_safe, q_safe)
        
        return js ** 2 if not np.isnan(js) and not np.isinf(js) else np.nan
    except Exception as e:
        return np.nan

# Function to safely calculate Pearson correlation
def safe_pearson_correlation(p, q):
    """Calculates Pearson correlation safely"""
    try:
        p_arr = np.asarray(p, dtype=np.float64).flatten()
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        
        if len(p_arr) != len(q_arr):
            return np.nan
        
        # Check for constant arrays
        if np.std(p_arr) == 0 or np.std(q_arr) == 0:
            return np.nan
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(p_arr, q_arr)
        
        return correlation if not np.isnan(correlation) else np.nan
    except Exception as e:
        return np.nan

# Function to process single experiment pair
@st.cache_data(ttl=3600)
def process_experiment_pair(predictions_content, groundtruth_content, exp_name):
    """Process a single experiment (predictions + ground truth pair)"""
    try:
        predictions_df = pd.read_csv(io.StringIO(predictions_content))
        groundtruth_df = pd.read_csv(io.StringIO(groundtruth_content))
        
        # Standard emotion columns
        emotion_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
        
        # Check for required columns
        missing_in_pred = [col for col in emotion_columns if col not in predictions_df.columns]
        missing_in_gt = [col for col in emotion_columns if col not in groundtruth_df.columns]
        
        if missing_in_pred:
            st.warning(f"Missing in predictions: {missing_in_pred}")
            return None
        if missing_in_gt:
            st.warning(f"Missing in ground truth: {missing_in_gt}")
            return None
        
        # Convert to numeric
        for col in emotion_columns:
            predictions_df[col] = pd.to_numeric(predictions_df[col], errors='coerce')
            groundtruth_df[col] = pd.to_numeric(groundtruth_df[col], errors='coerce')
        
        # Fill NaN values
        for col in emotion_columns:
            predictions_df[col] = predictions_df[col].fillna(1e-12)
            groundtruth_df[col] = groundtruth_df[col].fillna(1e-12)
        
        # Sort by file for alignment
        predictions_df = predictions_df.sort_values('file').reset_index(drop=True)
        groundtruth_df = groundtruth_df.sort_values('file').reset_index(drop=True)
        
        # Merge dataframes
        if predictions_df['file'].equals(groundtruth_df['file']):
            merged_df = predictions_df.copy()
            for col in emotion_columns:
                merged_df[f"{col}_gt"] = groundtruth_df[col]
            
            # Copy emotion_label if present
            if 'emotion_label' in groundtruth_df.columns:
                merged_df['groundtruth_label'] = groundtruth_df['emotion_label']
            elif 'emotion_label_gt' in merged_df.columns:
                merged_df['groundtruth_label'] = merged_df['emotion_label_gt']
        else:
            # Merge by file path
            merged_df = pd.merge(
                predictions_df, 
                groundtruth_df, 
                on='file', 
                suffixes=('_pred', '_gt'), 
                how='inner'
            )
            # Determine ground truth label column
            if 'emotion_label_gt' in merged_df.columns:
                merged_df['groundtruth_label'] = merged_df['emotion_label_gt']
            elif 'emotion_label' in merged_df.columns and '_gt' in merged_df.columns.get_loc('emotion_label'):
                merged_df['groundtruth_label'] = merged_df['emotion_label']
        
        # Emotion mapping
        column_to_label = {
            0: 1,  # happy -> label 1
            1: 7,  # contempt -> label 7
            2: 3,  # surprised -> label 3
            3: 6,  # angry -> label 6
            4: 5,  # disgusted -> label 5
            5: 4,  # fearful -> label 4
            6: 2,  # sad -> label 2
            7: 0   # neutral -> label 0
        }
        
        emotion_mapping = {
            0: 'neutral',
            1: 'happy',
            2: 'sad',
            3: 'surprised',
            4: 'fearful',
            5: 'disgusted',
            6: 'angry',
            7: 'contempt'
        }
        
        # Calculate predicted label from probabilities (most probable from predictions)
        pred_probs = merged_df[emotion_columns].values
        merged_df['predicted_label'] = np.argmax(pred_probs, axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].map(column_to_label)
        merged_df['predicted_emotion'] = merged_df['predicted_label'].map(emotion_mapping)
        
        # Calculate ground truth most probable label from ground truth probabilities
        # First, identify ground truth probability columns
        gt_prob_columns = []
        for col in emotion_columns:
            # Check for different possible column names for ground truth probabilities
            possible_cols = [f"{col}_gt", col]
            for possible_col in possible_cols:
                if possible_col in merged_df.columns:
                    gt_prob_columns.append(possible_col)
                    break
        
        if len(gt_prob_columns) == len(emotion_columns):
            # Calculate most probable class from ground truth probabilities
            gt_probs = merged_df[gt_prob_columns].values
            merged_df['groundtruth_most_probable_label'] = np.argmax(gt_probs, axis=1)
            merged_df['groundtruth_most_probable_label'] = merged_df['groundtruth_most_probable_label'].map(column_to_label)
            merged_df['groundtruth_most_probable_emotion'] = merged_df['groundtruth_most_probable_label'].map(emotion_mapping)
            
            # Calculate accuracy using most probable class from ground truth
            accuracy_most_probable = accuracy_score(
                merged_df['groundtruth_most_probable_label'], 
                merged_df['predicted_label']
            )
        else:
            st.warning(f"Could not find all ground truth probability columns for {exp_name}")
            merged_df['groundtruth_most_probable_label'] = merged_df['groundtruth_label']
            merged_df['groundtruth_most_probable_emotion'] = merged_df['groundtruth_emotion']
            accuracy_most_probable = None
        
        # Ensure ground truth label exists and filter valid labels
        if 'groundtruth_label' not in merged_df.columns:
            st.error(f"Experiment {exp_name}: No ground truth label column found")
            return None
        
        merged_df = merged_df[merged_df['groundtruth_label'].isin(emotion_mapping.keys())].copy()
        merged_df['groundtruth_emotion'] = merged_df['groundtruth_label'].map(emotion_mapping)
        
        if len(merged_df) == 0:
            st.error(f"Experiment {exp_name}: No valid samples after filtering")
            return None
        
        # Calculate accuracy using annotated label
        accuracy_annotated = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
        
        # Calculate distribution similarity metrics
        def calculate_metrics(row):
            try:
                pred_dist = []
                gt_dist = []
                
                for col in emotion_columns:
                    pred_col = col if col in row.index else f"{col}_pred"
                    gt_col = f"{col}_gt"
                    
                    pred_val = row[pred_col] if pred_col in row.index else 1e-12
                    gt_val = row[gt_col] if gt_col in row.index else 1e-12
                    
                    pred_dist.append(float(pred_val))
                    gt_dist.append(float(gt_val))
                
                pred_dist = np.array(pred_dist, dtype=np.float64) + 1e-12
                gt_dist = np.array(gt_dist, dtype=np.float64) + 1e-12
                
                # Normalize
                pred_dist = pred_dist / pred_dist.sum()
                gt_dist = gt_dist / gt_dist.sum()
                
                # JS divergence
                js_div = safe_js_divergence(pred_dist, gt_dist)
                
                # KL divergence (symmetric)
                kl_div = safe_kl_divergence(pred_dist, gt_dist, symmetric=True)
                
                # Euclidean distance
                euclidean_dist = np.linalg.norm(pred_dist - gt_dist)
                
                # Cosine similarity
                cosine_sim = np.dot(pred_dist, gt_dist) / (np.linalg.norm(pred_dist) * np.linalg.norm(gt_dist))
                
                # Pearson correlation
                pearson_corr = safe_pearson_correlation(pred_dist, gt_dist)
                if pearson_corr is None or np.isnan(pearson_corr):
                    pearson_corr = 0.0
                
                return pd.Series({
                    'js_divergence': js_div if not np.isnan(js_div) else 0.0,
                    'kl_divergence': kl_div if not np.isnan(kl_div) else 0.0,
                    'euclidean_distance': euclidean_dist,
                    'cosine_similarity': cosine_sim,
                    'pearson_correlation': pearson_corr
                })
            except Exception as e:
                return pd.Series({
                    'js_divergence': 0.0,
                    'kl_divergence': 0.0,
                    'euclidean_distance': 0.0,
                    'cosine_similarity': 0.0,
                    'pearson_correlation': 0.0
                })
        
        # Apply metrics calculation
        metrics = merged_df.apply(calculate_metrics, axis=1)
        merged_df = pd.concat([merged_df, metrics], axis=1)
        
        # Confusion matrix (using annotated labels)
        cm = confusion_matrix(
            merged_df['groundtruth_label'], 
            merged_df['predicted_label'], 
            labels=list(emotion_mapping.keys())
        )
        
        # Classification report (using annotated labels)
        class_report = classification_report(
            merged_df['groundtruth_label'], 
            merged_df['predicted_label'],
            target_names=list(emotion_mapping.values()),
            output_dict=True
        )
        
        # Emotion-wise metrics
        emotion_metrics = []
        for emotion_idx, emotion_name in emotion_mapping.items():
            mask = merged_df['groundtruth_label'] == emotion_idx
            if mask.sum() > 0:
                subset = merged_df[mask]
                emotion_accuracy = accuracy_score(subset['groundtruth_label'], subset['predicted_label'])
                
                # Verificar se a emoção existe no classification_report
                if emotion_name in class_report:
                    precision = class_report[emotion_name].get('precision', 0.0)
                    recall = class_report[emotion_name].get('recall', 0.0)
                    f1_score_val = class_report[emotion_name].get('f1-score', 0.0)
                else:
                    # Se não existir, usar valores padrão
                    precision = 0.0
                    recall = 0.0
                    f1_score_val = 0.0
                
                emotion_metrics.append({
                    'emotion': emotion_name,
                    'samples': int(mask.sum()),
                    'accuracy': emotion_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score_val,
                    'js_div_mean': subset['js_divergence'].mean(),
                    'kl_div_mean': subset['kl_divergence'].mean(),
                    'cosine_mean': subset['cosine_similarity'].mean(),
                    'pearson_correlation_mean': subset['pearson_correlation'].mean()
                })
        
        # Summary statistics
        macro_avg = class_report.get('macro avg', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
        weighted_avg = class_report.get('weighted avg', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
        
        summary_metrics = {
            'accuracy_annotated': accuracy_annotated,
            'accuracy_most_probable': accuracy_most_probable,
            'js_div_mean': merged_df['js_divergence'].mean(),
            'kl_div_mean': merged_df['kl_divergence'].mean(),
            'cosine_mean': merged_df['cosine_similarity'].mean(),
            'pearson_correlation_mean': merged_df['pearson_correlation'].mean(),
            'euclidean_mean': merged_df['euclidean_distance'].mean(),
            'samples': len(merged_df),
            'precision_macro': macro_avg.get('precision', 0.0),
            'recall_macro': macro_avg.get('recall', 0.0),
            'f1_macro': macro_avg.get('f1-score', 0.0),
            'precision_weighted': weighted_avg.get('precision', 0.0),
            'recall_weighted': weighted_avg.get('recall', 0.0),
            'f1_weighted': weighted_avg.get('f1-score', 0.0)
        }
        
        return {
            'name': exp_name,
            'data': merged_df,
            'accuracy_annotated': accuracy_annotated,
            'accuracy_most_probable': accuracy_most_probable,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'emotion_metrics': pd.DataFrame(emotion_metrics),
            'emotion_mapping': emotion_mapping,
            'summary_metrics': summary_metrics,
            'n_samples': len(merged_df)
        }
        
    except Exception as e:
        st.error(f"Error processing experiment {exp_name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to create publication-ready plots
def create_publication_plot(fig, title, width=800, height=600):
    """Format plot for publication quality"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial, sans-serif", color='black'),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family="Arial, sans-serif", size=14, color='black'),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            mirror=True,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            mirror=True,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        legend=dict(
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            x=1.02,
            xanchor='left',
            y=1,
            yanchor='top'
        ),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    return fig

# Initialize session state for experiments
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'experiment_counter' not in st.session_state:
    st.session_state.experiment_counter = 0
if 'processed_experiments' not in st.session_state:
    st.session_state.processed_experiments = None

# Sidebar
st.sidebar.header("🔧 Experiment Configuration")

# Add new experiment pair
st.sidebar.subheader("Add Experiment Pair")

with st.sidebar.expander("➕ Add New Experiment", expanded=True):
    exp_name = st.text_input("Experiment Name", value=f"Experiment_{st.session_state.experiment_counter + 1}")
    
    col1, col2 = st.columns(2)
    with col1:
        predictions_file = st.file_uploader(
            "Predictions CSV",
            type=['csv'],
            key=f"pred_{st.session_state.experiment_counter}",
            help="Model predictions file"
        )
    
    with col2:
        groundtruth_file = st.file_uploader(
            "Ground Truth CSV",
            type=['csv'],
            key=f"gt_{st.session_state.experiment_counter}",
            help="Ground truth labels file"
        )
    
    if st.button("Add Experiment Pair", type="primary"):
        if predictions_file and groundtruth_file and exp_name:
            # Store experiment data
            exp_data = {
                'name': exp_name,
                'predictions_file': predictions_file,
                'groundtruth_file': groundtruth_file,
                'processed': False,
                'data': None
            }
            st.session_state.experiments.append(exp_data)
            st.session_state.experiment_counter += 1
            st.success(f"Added {exp_name}")
            st.rerun()
        else:
            st.error("Please provide all files and a name")

# Show added experiments
if st.session_state.experiments:
    st.sidebar.subheader("📋 Added Experiments")
    
    for i, exp in enumerate(st.session_state.experiments):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"**{exp['name']}**")
            st.caption(f"Pred: {exp['predictions_file'].name[:20]}...")
            st.caption(f"GT: {exp['groundtruth_file'].name[:20]}...")
        
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.experiments.pop(i)
                st.rerun()
    
    st.sidebar.markdown("---")

# Analysis controls
st.sidebar.subheader("📈 Analysis Settings")

# Metric selection
metric_options = {
    'accuracy_annotated': {'name': 'Accuracy (Annotated Label)', 'higher_better': True},
    'accuracy_most_probable': {'name': 'Accuracy (Most Probable)', 'higher_better': True},
    'js_divergence': {'name': 'JS Divergence', 'higher_better': False},
    'cosine_similarity': {'name': 'Cosine Similarity', 'higher_better': True},
    'pearson_correlation': {'name': 'Pearson Correlation', 'higher_better': True},
    'kl_divergence': {'name': 'KL Divergence', 'higher_better': False},
    'euclidean_distance': {'name': 'Euclidean Distance', 'higher_better': False},
    'f1_macro': {'name': 'F1-Score (Macro)', 'higher_better': True},
    'precision_macro': {'name': 'Precision (Macro)', 'higher_better': True},
    'recall_macro': {'name': 'Recall (Macro)', 'higher_better': True}
}

selected_metrics = st.sidebar.multiselect(
    "Metrics to Compare:",
    list(metric_options.keys()),
    default=['accuracy_annotated', 'accuracy_most_probable', 'js_divergence', 'cosine_similarity', 'pearson_correlation', 'kl_divergence', 'f1_macro'],
    format_func=lambda x: metric_options[x]['name']
)

# Visualization style
viz_style = st.sidebar.selectbox(
    "Visualization Style:",
    ['Publication', 'Interactive', 'Minimal'],
    index=0
)

st.sidebar.markdown("---")

# Main processing button
if st.sidebar.button("🚀 Process All Experiments", type="primary"):
    if not st.session_state.experiments:
        st.error("No experiments to process!")
    else:
        progress_bar = st.progress(0)
        processed_experiments = []
        
        for i, exp in enumerate(st.session_state.experiments):
            st.info(f"Processing: {exp['name']}")
            
            # Read file contents
            predictions_content = exp['predictions_file'].getvalue().decode('utf-8')
            groundtruth_content = exp['groundtruth_file'].getvalue().decode('utf-8')
            
            # Process experiment
            result = process_experiment_pair(
                predictions_content, 
                groundtruth_content, 
                exp['name']
            )
            
            if result:
                processed_experiments.append(result)
                st.success(f"✓ {exp['name']}: {result['accuracy_annotated']:.2%} accuracy annotated ({result['n_samples']} samples)")
            else:
                st.error(f"✗ Failed to process {exp['name']}")
            
            progress_bar.progress((i + 1) / len(st.session_state.experiments))
        
        if processed_experiments:
            st.session_state.processed_experiments = processed_experiments
            st.success(f"✅ Successfully processed {len(processed_experiments)} experiments")
        else:
            st.error("No experiments were successfully processed")

# Clear all experiments button
if st.sidebar.button("🗑️ Clear All Experiments"):
    st.session_state.experiments = []
    st.session_state.experiment_counter = 0
    st.session_state.processed_experiments = None
    st.rerun()

# Main content area
if 'processed_experiments' in st.session_state and st.session_state.processed_experiments:
    experiments = st.session_state.processed_experiments
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "📈 Comparative Analysis", 
        "🎭 Emotion Performance",
        "🤖 Model Ranking",
        "📉 Distribution Analysis",
        "📋 Detailed Reports",
        "📥 Export Results"
    ])
    
    with tab1:
        st.header("Experiment Overview")
        
        # Summary statistics table
        st.subheader("1. Performance Summary")
        
        summary_data = []
        for exp in experiments:
            summary = {
                'Experiment': exp['name'],
                'Accuracy (Annotated)': exp['accuracy_annotated'],
                'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                'F1-Score (Macro)': exp['summary_metrics']['f1_macro'],
                'Precision (Macro)': exp['summary_metrics']['precision_macro'],
                'Recall (Macro)': exp['summary_metrics']['recall_macro'],
                'JS Divergence (↓)': exp['summary_metrics']['js_div_mean'],
                'KL Divergence (↓)': exp['summary_metrics']['kl_div_mean'],
                'Cosine Similarity (↑)': exp['summary_metrics']['cosine_mean'],
                'Pearson Correlation (↑)': exp['summary_metrics']['pearson_correlation_mean'],
                'Samples': exp['summary_metrics']['samples']
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy (Annotated)', ascending=False)
        
        # Color formatting functions
        def color_accuracy(val):
            if val > 0.8:
                return 'background-color: #90EE90'  # Light green
            elif val > 0.6:
                return 'background-color: #FFD700'  # Gold
            else:
                return 'background-color: #FFB6C1'  # Light red
        
        def color_f1(val):
            if val > 0.8:
                return 'background-color: #90EE90'
            elif val > 0.6:
                return 'background-color: #FFD700'
            else:
                return 'background-color: #FFB6C1'
        
        def color_divergence(val):
            if val < 0.1:
                return 'background-color: #90EE90'
            elif val < 0.3:
                return 'background-color: #FFD700'
            else:
                return 'background-color: #FFB6C1'
        
        def color_correlation(val):
            if val > 0.8:
                return 'background-color: #90EE90'
            elif val > 0.6:
                return 'background-color: #FFD700'
            else:
                return 'background-color: #FFB6C1'
        
        # Apply styling
        styled_df = summary_df.style.format({
            'Accuracy (Annotated)': '{:.2%}',
            'Accuracy (Most Probable)': '{:.2%}',
            'F1-Score (Macro)': '{:.3f}',
            'Precision (Macro)': '{:.3f}',
            'Recall (Macro)': '{:.3f}',
            'JS Divergence (↓)': '{:.4f}',
            'KL Divergence (↓)': '{:.4f}',
            'Cosine Similarity (↑)': '{:.4f}',
            'Pearson Correlation (↑)': '{:.4f}'
        }).applymap(color_accuracy, subset=['Accuracy (Annotated)', 'Accuracy (Most Probable)'])\
          .applymap(color_f1, subset=['F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)'])\
          .applymap(color_divergence, subset=['JS Divergence (↓)', 'KL Divergence (↓)'])\
          .applymap(color_correlation, subset=['Cosine Similarity (↑)', 'Pearson Correlation (↑)'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download summary
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Table",
            data=csv_summary,
            file_name="experiment_summary.csv",
            mime="text/csv",
            key="download_summary"
        )
        
        # Performance radar chart
        if len(experiments) > 1:
            st.subheader("2. Multi-Metric Radar Chart")
            
            # Normalize metrics for radar chart
            radar_metrics = ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score', 'Cosine Similarity', 'Pearson Correlation']
            
            fig_radar = go.Figure()
            
            color_cycle = cycle(COLOR_PALETTE)
            
            for exp in experiments:
                color = next(color_cycle)
                values = [
                    exp['accuracy_annotated'],  # Accuracy (Annotated)
                    exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 0,  # Accuracy (Most Probable)
                    exp['summary_metrics']['f1_macro'],  # F1-Score
                    exp['summary_metrics']['cosine_mean'],  # Cosine Similarity
                    exp['summary_metrics']['pearson_correlation_mean']  # Pearson Correlation
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_metrics,
                    name=exp['name'],
                    fill='toself',
                    fillcolor=hex_to_rgba(color, alpha=0.3),
                    line=dict(color=color, width=2),
                    marker=dict(size=8)
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=12)
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=14)
                    )
                ),
                showlegend=True,
                legend=dict(
                    font=dict(size=12),
                    x=1.1,
                    xanchor='left'
                ),
                title="Multi-Metric Comparison (Normalized to 0-1)",
                height=500
            )
            
            if viz_style == 'Publication':
                fig_radar = create_publication_plot(fig_radar, "Multi-Metric Radar Chart")
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Correlation between KL and JS divergences
            st.subheader("3. KL vs JS Divergence Analysis")
            
            # Prepare data for scatter plot
            kl_js_data = []
            for exp in experiments:
                kl_js_data.append({
                    'Experiment': exp['name'],
                    'KL Divergence': exp['summary_metrics']['kl_div_mean'],
                    'JS Divergence': exp['summary_metrics']['js_div_mean'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 0,
                    'Samples': exp['n_samples']
                })
            
            kl_js_df = pd.DataFrame(kl_js_data)
            
            fig_kl_js = px.scatter(
                kl_js_df,
                x='KL Divergence',
                y='JS Divergence',
                color='Experiment',
                size='Accuracy (Annotated)',
                title='KL Divergence vs JS Divergence by Experiment',
                labels={
                    'KL Divergence': 'KL Divergence (lower is better)',
                    'JS Divergence': 'JS Divergence (lower is better)',
                    'Accuracy (Annotated)': 'Accuracy (Annotated)'
                },
                hover_data=['Experiment', 'Accuracy (Annotated)', 'Accuracy (Most Probable)', 'Samples'],
                color_discrete_sequence=COLOR_PALETTE[:len(experiments)]
            )
            
            # Add diagonal line for reference (JS ≈ KL/2 for similar distributions)
            max_val = max(kl_js_df['KL Divergence'].max(), kl_js_df['JS Divergence'].max())
            fig_kl_js.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='y = x',
                showlegend=True
            ))
            
            fig_kl_js.update_layout(
                height=500,
                xaxis_title="KL Divergence",
                yaxis_title="JS Divergence"
            )
            
            if viz_style == 'Publication':
                fig_kl_js = create_publication_plot(fig_kl_js, "KL vs JS Divergence Comparison")
            
            st.plotly_chart(fig_kl_js, use_container_width=True)
            
            # Explanation of metrics
            with st.expander("📘 Understanding the Metrics"):
                st.markdown("""
                ### Accuracy Metrics
                - **Accuracy (Annotated)**: Uses the manually annotated label from ground truth
                - **Accuracy (Most Probable)**: Uses the most probable class calculated from ground truth probabilities
                
                ### KL Divergence (Kullback-Leibler Divergence)
                - **What it measures**: How one probability distribution diverges from a second, expected probability distribution.
                - **Interpretation**: 
                  - **0**: Distributions are identical
                  - **Higher values**: Greater divergence between distributions
                  - **Asymmetric**: KL(P||Q) ≠ KL(Q||P) in general (we use symmetric KL)
                - **Formula**: KL(P||Q) = Σ P(i) log(P(i)/Q(i))
                
                ### JS Divergence (Jensen-Shannon Divergence)
                - **What it measures**: Symmetric and smoothed version of KL divergence
                - **Interpretation**:
                  - **0**: Distributions are identical
                  - **1**: Distributions are maximally different
                  - **Symmetric**: JS(P,Q) = JS(Q,P)
                - **Formula**: JS(P,Q) = ½ KL(P||M) + ½ KL(Q||M), where M = ½(P+Q)
                
                ### Pearson Correlation
                - **What it measures**: Linear correlation between two sets of values
                - **Interpretation**:
                  - **+1**: Perfect positive correlation
                  - **0**: No correlation
                  - **-1**: Perfect negative correlation
                - **Range**: -1 to 1
                
                ### Cosine Similarity
                - **What it measures**: Cosine of the angle between two vectors
                - **Interpretation**:
                  - **+1**: Vectors point in the same direction
                  - **0**: Vectors are orthogonal
                  - **-1**: Vectors point in opposite directions
                - **Range**: -1 to 1
                """)
    
    with tab2:
        st.header("Comparative Analysis")
        
        # Bar charts for selected metrics
        if selected_metrics:
            # Determine layout
            n_metrics = len(selected_metrics)
            n_cols = 2
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            for i in range(0, n_metrics, n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    if i + j < n_metrics:
                        metric = selected_metrics[i + j]
                        with cols[j]:
                            # Prepare data
                            metric_data = []
                            for exp in experiments:
                                if metric == 'accuracy_annotated':
                                    value = exp['accuracy_annotated']
                                elif metric == 'accuracy_most_probable':
                                    value = exp['accuracy_most_probable']
                                elif metric in exp['summary_metrics']:
                                    value = exp['summary_metrics'][metric]
                                else:
                                    continue
                                
                                metric_data.append({
                                    'Experiment': exp['name'],
                                    'Value': value if value is not None else 0,
                                    'Metric': metric_options[metric]['name']
                                })
                            
                            if metric_data:
                                metric_df = pd.DataFrame(metric_data)
                                
                                # Sort based on whether higher is better
                                ascending = not metric_options[metric]['higher_better']
                                metric_df = metric_df.sort_values('Value', ascending=ascending)
                                
                                # Create bar chart
                                fig = px.bar(
                                    metric_df,
                                    x='Experiment',
                                    y='Value',
                                    color='Value',
                                    color_continuous_scale='Viridis' if metric_options[metric]['higher_better'] else 'Viridis_r',
                                    title=f"{metric_options[metric]['name']} by Experiment",
                                    text_auto='.3f',
                                    labels={'Value': metric_options[metric]['name']}
                                )
                                
                                fig.update_traces(
                                    texttemplate='%{text:.3f}',
                                    textposition='outside'
                                )
                                
                                if viz_style == 'Publication':
                                    fig = create_publication_plot(
                                        fig, 
                                        f"{metric_options[metric]['name']} Comparison",
                                        width=400,
                                        height=400
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison heatmap
            if len(experiments) > 1:
                st.subheader("Performance Comparison Matrix")
                
                # Create comparison matrix
                exp_names = [exp['name'] for exp in experiments]
                comparison_matrix = []
                
                for i, exp1 in enumerate(experiments):
                    row = []
                    for j, exp2 in enumerate(experiments):
                        if i == j:
                            row.append(1.0)
                        else:
                            # Compare using composite score including new metrics
                            score1 = (
                                exp1['accuracy_annotated'] * 0.25 + 
                                (exp1['accuracy_most_probable'] if exp1['accuracy_most_probable'] is not None else exp1['accuracy_annotated']) * 0.20 +
                                exp1['summary_metrics']['f1_macro'] * 0.20 + 
                                exp1['summary_metrics']['cosine_mean'] * 0.15 +
                                exp1['summary_metrics']['pearson_correlation_mean'] * 0.15 +
                                (1 - min(exp1['summary_metrics']['kl_div_mean'], 1)) * 0.05
                            )
                            score2 = (
                                exp2['accuracy_annotated'] * 0.25 + 
                                (exp2['accuracy_most_probable'] if exp2['accuracy_most_probable'] is not None else exp2['accuracy_annotated']) * 0.20 +
                                exp2['summary_metrics']['f1_macro'] * 0.20 + 
                                exp2['summary_metrics']['cosine_mean'] * 0.15 +
                                exp2['summary_metrics']['pearson_correlation_mean'] * 0.15 +
                                (1 - min(exp2['summary_metrics']['kl_div_mean'], 1)) * 0.05
                            )
                            row.append(score1 / score2 if score2 > 0 else 0)
                    comparison_matrix.append(row)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=comparison_matrix,
                    x=exp_names,
                    y=exp_names,
                    colorscale='RdYlGn',
                    zmid=1,
                    text=np.round(comparison_matrix, 2),
                    texttemplate='%{text:.2f}',
                    textfont={"size": 12},
                    hoverongaps=False,
                    colorbar_title="Relative Performance<br>(Row/Column)"
                ))
                
                fig_heatmap.update_layout(
                    title="Relative Performance Comparison Matrix",
                    xaxis_title="Experiment (Denominator)",
                    yaxis_title="Experiment (Numerator)",
                    height=500
                )
                
                if viz_style == 'Publication':
                    fig_heatmap = create_publication_plot(fig_heatmap, "Performance Comparison Matrix")
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Correlation matrix between metrics
                st.subheader("Correlation Between Different Metrics")
                
                # Prepare data for correlation matrix
                corr_data = []
                for exp in experiments:
                    corr_data.append({
                        'Experiment': exp['name'],
                        'Accuracy (Annotated)': exp['accuracy_annotated'],
                        'Accuracy (Most Probable)': exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 0,
                        'F1-Score': exp['summary_metrics']['f1_macro'],
                        'JS Divergence': exp['summary_metrics']['js_div_mean'],
                        'KL Divergence': exp['summary_metrics']['kl_div_mean'],
                        'Cosine Similarity': exp['summary_metrics']['cosine_mean'],
                        'Pearson Correlation': exp['summary_metrics']['pearson_correlation_mean']
                    })
                
                corr_df = pd.DataFrame(corr_data)
                numeric_cols = ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score', 'JS Divergence', 
                              'KL Divergence', 'Cosine Similarity', 'Pearson Correlation']
                correlation_matrix = corr_df[numeric_cols].corr()
                
                fig_corr_matrix = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=numeric_cols,
                    y=numeric_cols,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=np.round(correlation_matrix.values, 2),
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorbar_title="Correlation"
                ))
                
                fig_corr_matrix.update_layout(
                    title="Correlation Between Metrics Across Experiments",
                    height=500
                )
                
                if viz_style == 'Publication':
                    fig_corr_matrix = create_publication_plot(fig_corr_matrix, "Metric Correlations")
                
                st.plotly_chart(fig_corr_matrix, use_container_width=True)
    
    with tab3:
        st.header("Emotion-wise Performance Analysis")
        
        # Emotion accuracy comparison
        st.subheader("1. Emotion-specific Accuracy")
        
        # Prepare data
        emotion_acc_data = []
        for exp in experiments:
            for _, row in exp['emotion_metrics'].iterrows():
                emotion_acc_data.append({
                    'Experiment': exp['name'],
                    'Emotion': row['emotion'],
                    'Accuracy': row['accuracy'],
                    'Samples': row['samples'],
                    'F1-Score': row['f1_score'],
                    'Precision': row['precision'],
                    'Recall': row['recall'],
                    'KL Divergence': row['kl_div_mean'],
                    'Pearson Correlation': row['pearson_correlation_mean']
                })
        
        emotion_acc_df = pd.DataFrame(emotion_acc_data)
        
        # Grouped bar chart
        fig_emotion_acc = px.bar(
            emotion_acc_df,
            x='Emotion',
            y='Accuracy',
            color='Experiment',
            barmode='group',
            title='Accuracy by Emotion and Experiment',
            text_auto='.1%',
            color_discrete_sequence=COLOR_PALETTE[:len(experiments)],
            height=500
        )
        
        fig_emotion_acc.update_traces(
            texttemplate='%{y:.1%}',
            textposition='outside'
        )
        
        if viz_style == 'Publication':
            fig_emotion_acc = create_publication_plot(fig_emotion_acc, "Emotion-wise Accuracy Comparison")
        
        st.plotly_chart(fig_emotion_acc, use_container_width=True)
        
        # Heatmap of emotion performance
        st.subheader("2. Performance Heatmap Across Emotions")
        
        # Select metric for heatmap
        heatmap_metric = st.selectbox(
            "Select metric for heatmap:",
            ['Accuracy', 'F1-Score', 'KL Divergence', 'Pearson Correlation'],
            key="heatmap_metric_select"
        )
        
        # Pivot for heatmap
        if heatmap_metric == 'KL Divergence':
            metric_col = 'KL Divergence'
        elif heatmap_metric == 'Pearson Correlation':
            metric_col = 'Pearson Correlation'
        elif heatmap_metric == 'F1-Score':
            metric_col = 'F1-Score'
        else:
            metric_col = 'Accuracy'
        
        pivot_data = emotion_acc_df.pivot_table(
            index='Experiment', 
            columns='Emotion', 
            values=metric_col,
            aggfunc='mean'
        )
        
        # Determine color scale based on metric
        if metric_col in ['KL Divergence']:
            colorscale = 'Viridis_r'  # Reverse for divergence (lower is better)
        else:
            colorscale = 'Viridis'  # Normal for accuracy/correlation (higher is better)
        
        fig_heatmap_emotion = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=colorscale,
            text=np.round(pivot_data.values, 3),
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar_title=heatmap_metric
        ))
        
        fig_heatmap_emotion.update_layout(
            title=f"{heatmap_metric} by Emotion and Experiment",
            xaxis_title="Emotion",
            yaxis_title="Experiment",
            height=400
        )
        
        if viz_style == 'Publication':
            fig_heatmap_emotion = create_publication_plot(fig_heatmap_emotion, f"{heatmap_metric} Heatmap")
        
        st.plotly_chart(fig_heatmap_emotion, use_container_width=True)
        
        # Scatter plot: KL Divergence vs Accuracy per emotion
        st.subheader("3. KL Divergence vs Accuracy per Emotion")
        
        fig_scatter_kl_acc = px.scatter(
            emotion_acc_df,
            x='KL Divergence',
            y='Accuracy',
            color='Emotion',
            facet_col='Experiment',
            facet_col_wrap=min(3, len(experiments)),
            title='KL Divergence vs Accuracy by Emotion and Experiment',
            labels={
                'KL Divergence': 'KL Divergence (lower is better)',
                'Accuracy': 'Accuracy (higher is better)'
            },
            hover_data=['Emotion', 'Samples'],
            height=400
        )
        
        if viz_style == 'Publication':
            fig_scatter_kl_acc = create_publication_plot(fig_scatter_kl_acc, "KL Divergence vs Accuracy")
        
        st.plotly_chart(fig_scatter_kl_acc, use_container_width=True)
    
    with tab4:
        st.header("Model Ranking and Evaluation")
        
        # Calculate composite scores including new metrics
        st.subheader("1. Model Ranking by Composite Score")
        
        composite_scores = []
        for exp in experiments:
            # Normalize metrics to 0-1 scale
            accuracy_annotated_norm = exp['accuracy_annotated']
            accuracy_most_probable_norm = exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else exp['accuracy_annotated']
            f1_norm = exp['summary_metrics']['f1_macro']
            cosine_norm = exp['summary_metrics']['cosine_mean']
            pearson_norm = (exp['summary_metrics']['pearson_correlation_mean'] + 1) / 2  # Convert -1 to 1 → 0 to 1
            js_norm = 1 - min(exp['summary_metrics']['js_div_mean'], 1)
            kl_norm = 1 - min(exp['summary_metrics']['kl_div_mean'], 1)
            
            # Weighted composite score (adjust weights as needed)
            composite = (
                0.25 * accuracy_annotated_norm + 
                0.20 * accuracy_most_probable_norm + 
                0.15 * f1_norm + 
                0.15 * cosine_norm + 
                0.15 * pearson_norm + 
                0.05 * js_norm + 
                0.05 * kl_norm
            )
            
            composite_scores.append({
                'Experiment': exp['name'],
                'Composite Score': composite,
                'Accuracy (Annotated)': accuracy_annotated_norm,
                'Accuracy (Most Probable)': accuracy_most_probable_norm,
                'F1-Score': f1_norm,
                'Cosine Similarity': cosine_norm,
                'Pearson Correlation': exp['summary_metrics']['pearson_correlation_mean'],
                'JS Divergence': exp['summary_metrics']['js_div_mean'],
                'KL Divergence': exp['summary_metrics']['kl_div_mean'],
                'Rank': 0  # Will be filled after sorting
            })
        
        composite_df = pd.DataFrame(composite_scores)
        composite_df = composite_df.sort_values('Composite Score', ascending=False)
        composite_df['Rank'] = range(1, len(composite_df) + 1)
        
        # Display ranking
        fig_ranking = px.bar(
            composite_df,
            x='Composite Score',
            y='Experiment',
            orientation='h',
            title='Model Ranking by Composite Score',
            color='Composite Score',
            color_continuous_scale='RdYlGn',
            text='Composite Score',
            height=max(300, len(experiments) * 40)
        )
        
        fig_ranking.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside'
        )
        
        if viz_style == 'Publication':
            fig_ranking = create_publication_plot(fig_ranking, "Model Ranking")
        
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Display ranking table
        st.subheader("2. Ranking Details")
        
        rank_columns = ['Rank', 'Experiment', 'Composite Score', 'Accuracy (Annotated)', 'Accuracy (Most Probable)', 
                       'F1-Score', 'Cosine Similarity', 'Pearson Correlation', 'JS Divergence', 'KL Divergence']
        display_rank_df = composite_df[rank_columns].copy()
        
        def highlight_top3(row):
            if row['Rank'] == 1:
                return ['background-color: gold'] * len(row)
            elif row['Rank'] == 2:
                return ['background-color: silver'] * len(row)
            elif row['Rank'] == 3:
                return ['background-color: #cd7f32'] * len(row)  # Bronze
            return [''] * len(row)
        
        styled_rank_df = display_rank_df.style.format({
            'Composite Score': '{:.3f}',
            'Accuracy (Annotated)': '{:.2%}',
            'Accuracy (Most Probable)': '{:.2%}',
            'F1-Score': '{:.3f}',
            'Cosine Similarity': '{:.3f}',
            'Pearson Correlation': '{:.3f}',
            'JS Divergence': '{:.4f}',
            'KL Divergence': '{:.4f}'
        }).apply(highlight_top3, axis=1)
        
        st.dataframe(styled_rank_df, use_container_width=True)
        
        # Parallel coordinates plot for multi-dimensional comparison
        if len(experiments) > 1:
            st.subheader("3. Multi-Dimensional Comparison")
            
            # Prepare data for parallel coordinates
            parallel_data = composite_df.copy()
            
            # Normalize all metrics to 0-1 scale for parallel coordinates
            for col in ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score', 'Cosine Similarity', 'Pearson Correlation']:
                if col in parallel_data.columns:
                    parallel_data[col] = parallel_data[col].astype(float)
            
            # For divergence metrics, invert so higher is better
            parallel_data['JS Divergence_inv'] = 1 - parallel_data['JS Divergence']
            parallel_data['KL Divergence_inv'] = 1 - parallel_data['KL Divergence']
            
            # Create parallel coordinates plot
            dimensions = []
            for col in ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score', 'Cosine Similarity', 'Pearson Correlation', 
                       'JS Divergence_inv', 'KL Divergence_inv']:
                if col in parallel_data.columns:
                    dim_label = col.replace('_inv', ' (Inv)')
                    dimensions.append(
                        dict(range=[parallel_data[col].min(), parallel_data[col].max()],
                             label=dim_label,
                             values=parallel_data[col])
                    )
            
            fig_parallel = go.Figure(data=
                go.Parcoords(
                    line=dict(color=parallel_data['Composite Score'],
                             colorscale='Viridis',
                             showscale=True,
                             colorbar=dict(title='Composite Score')),
                    dimensions=dimensions,
                    labelangle=30,
                    labelside='bottom'
                )
            )
            
            fig_parallel.update_layout(
                title="Parallel Coordinates Plot - Multi-Dimensional Comparison",
                height=500
            )
            
            if viz_style == 'Publication':
                fig_parallel = create_publication_plot(fig_parallel, "Multi-Dimensional Comparison")
            
            st.plotly_chart(fig_parallel, use_container_width=True)
    
    with tab5:
        st.header("Distribution Analysis")
        
        # Distribution of metrics across experiments
        st.subheader("1. Metric Distributions Comparison")
        
        # Prepare data for violin/box plots
        dist_data = []
        for exp in experiments:
            for metric in ['js_divergence', 'kl_divergence', 'cosine_similarity', 'pearson_correlation']:
                values = exp['data'][metric].dropna()
                for val in values:
                    dist_data.append({
                        'Experiment': exp['name'],
                        'Metric': metric_options[metric]['name'],
                        'Value': val
                    })
        
        dist_df = pd.DataFrame(dist_data)
        
        # Create violin plots
        fig_violin = px.violin(
            dist_df,
            x='Experiment',
            y='Value',
            color='Experiment',
            facet_col='Metric',
            facet_col_wrap=2,
            box=True,
            points=False,
            title="Distribution of Metrics Across Experiments",
            color_discrete_sequence=COLOR_PALETTE[:len(experiments)],
            height=600
        )
        
        if viz_style == 'Publication':
            fig_violin = create_publication_plot(fig_violin, "Metric Distributions")
        
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Correlation between accuracy and distribution metrics
        st.subheader("2. Correlation Analysis: Accuracy vs Distribution Metrics")
        
        # Prepare data for scatter matrix
        scatter_data = []
        for exp in experiments:
            scatter_data.append({
                'Experiment': exp['name'],
                'Accuracy (Annotated)': exp['accuracy_annotated'],
                'Accuracy (Most Probable)': exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 0,
                'F1-Score': exp['summary_metrics']['f1_macro'],
                'JS Divergence': exp['summary_metrics']['js_div_mean'],
                'KL Divergence': exp['summary_metrics']['kl_div_mean'],
                'Cosine Similarity': exp['summary_metrics']['cosine_mean'],
                'Pearson Correlation': exp['summary_metrics']['pearson_correlation_mean']
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create scatter matrix
        fig_scatter_matrix = px.scatter_matrix(
            scatter_df,
            dimensions=['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score', 'JS Divergence', 
                       'KL Divergence', 'Cosine Similarity', 'Pearson Correlation'],
            color='Experiment',
            title="Scatter Matrix: Accuracy vs Distribution Metrics",
            height=700
        )
        
        if viz_style == 'Publication':
            fig_scatter_matrix.update_layout(
                font=dict(size=10),
                title_font=dict(size=16)
            )
        
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
        
        # Time series or progression of metrics (if applicable)
        st.subheader("3. Metric Relationships")
        
        # Select x and y axes for scatter plot
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X-axis metric:",
                ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'JS Divergence', 'KL Divergence', 
                 'Cosine Similarity', 'Pearson Correlation', 'F1-Score'],
                key="x_axis_select"
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-axis metric:",
                ['Accuracy (Annotated)', 'Accuracy (Most Probable)', 'JS Divergence', 'KL Divergence', 
                 'Cosine Similarity', 'Pearson Correlation', 'F1-Score'],
                key="y_axis_select",
                index=3  # Default to KL Divergence
            )
        
        # Map selection to actual column names
        col_map = {
            'Accuracy (Annotated)': 'Accuracy (Annotated)',
            'Accuracy (Most Probable)': 'Accuracy (Most Probable)',
            'JS Divergence': 'JS Divergence',
            'KL Divergence': 'KL Divergence',
            'Cosine Similarity': 'Cosine Similarity',
            'Pearson Correlation': 'Pearson Correlation',
            'F1-Score': 'F1-Score'
        }
        
        x_col = col_map[x_axis]
        y_col = col_map[y_axis]
        
        fig_relationship = px.scatter(
            scatter_df,
            x=x_col,
            y=y_col,
            color='Experiment',
            size='F1-Score',
            title=f'{x_axis} vs {y_axis} by Experiment',
            hover_data=['Experiment', 'Accuracy (Annotated)', 'F1-Score'],
            trendline="ols",
            height=500
        )
        
        if viz_style == 'Publication':
            fig_relationship = create_publication_plot(fig_relationship, f"{x_axis} vs {y_axis}")
        
        st.plotly_chart(fig_relationship, use_container_width=True)
    
    with tab6:
        st.header("Detailed Experiment Reports")
        
        # Select experiment for detailed view
        selected_exp_name = st.selectbox(
            "Select Experiment for Detailed Report:",
            [exp['name'] for exp in experiments],
            key="detailed_report_select"
        )
        
        selected_exp = next(exp for exp in experiments if exp['name'] == selected_exp_name)
        
        if selected_exp:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy (Annotated)", f"{selected_exp['accuracy_annotated']:.2%}")
            
            with col2:
                if selected_exp['accuracy_most_probable'] is not None:
                    st.metric("Accuracy (Most Probable)", f"{selected_exp['accuracy_most_probable']:.2%}")
                else:
                    st.metric("Accuracy (Most Probable)", "N/A")
            
            with col3:
                st.metric("F1-Score (Macro)", f"{selected_exp['summary_metrics']['f1_macro']:.3f}")
            
            with col4:
                st.metric("Cosine Similarity", f"{selected_exp['summary_metrics']['cosine_mean']:.3f}")
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("JS Divergence", f"{selected_exp['summary_metrics']['js_div_mean']:.4f}")
            
            with col6:
                st.metric("KL Divergence", f"{selected_exp['summary_metrics']['kl_div_mean']:.4f}")
            
            with col7:
                st.metric("Pearson Correlation", f"{selected_exp['summary_metrics']['pearson_correlation_mean']:.3f}")
            
            with col8:
                st.metric("Samples", f"{selected_exp['n_samples']:,}")
            
            # Classification report
            st.subheader("Classification Report")
            
            class_report_df = pd.DataFrame(selected_exp['classification_report']).T
            
            # Style the classification report
            def color_classification(val):
                if isinstance(val, (int, float)):
                    if val > 0.8:
                        return 'background-color: #90EE90'
                    elif val > 0.6:
                        return 'background-color: #FFD700'
                    else:
                        return 'background-color: #FFB6C1'
                return ''
            
            styled_report = class_report_df.style.format({
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1-score': '{:.3f}',
                'support': '{:.0f}'
            }).applymap(color_classification, subset=['precision', 'recall', 'f1-score'])
            
            st.dataframe(styled_report, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            
            cm = selected_exp['confusion_matrix']
            emotion_names = list(selected_exp['emotion_mapping'].values())
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_normalized,
                x=emotion_names,
                y=emotion_names,
                colorscale='Blues',
                text=np.round(cm_normalized, 2),
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar_title="Proportion"
            ))
            
            fig_cm.update_layout(
                title=f"Normalized Confusion Matrix - {selected_exp_name}",
                xaxis_title="Predicted Emotion",
                yaxis_title="True Emotion",
                height=500
            )
            
            if viz_style == 'Publication':
                fig_cm = create_publication_plot(fig_cm, f"Confusion Matrix - {selected_exp_name}")
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Emotion metrics table
            st.subheader("Emotion-wise Performance")
            
            emotion_metrics_df = selected_exp['emotion_metrics']
            
            styled_emotion_df = emotion_metrics_df.style.format({
                'accuracy': '{:.2%}',
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1_score': '{:.3f}',
                'js_div_mean': '{:.4f}',
                'kl_div_mean': '{:.4f}',
                'cosine_mean': '{:.3f}',
                'pearson_correlation_mean': '{:.3f}'
            }).applymap(color_classification, subset=['accuracy', 'precision', 'recall', 'f1_score'])
            
            st.dataframe(styled_emotion_df, use_container_width=True)
            
            # Sample-level distribution metrics
            st.subheader("Sample-level Distribution Metrics")
            
            # Select a few random samples to show
            sample_data = selected_exp['data'].sample(min(5, len(selected_exp['data'])))
            
            for idx, row in sample_data.iterrows():
                with st.expander(f"Sample: {row['file'].split('/')[-1] if isinstance(row['file'], str) else 'Sample ' + str(idx)}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Distribution Metrics:**")
                        metrics_table = pd.DataFrame({
                            'Metric': ['KL Divergence', 'JS Divergence', 'Pearson Correlation', 
                                      'Cosine Similarity', 'Euclidean Distance'],
                            'Value': [
                                f"{row['kl_divergence']:.4f}",
                                f"{row['js_divergence']:.4f}",
                                f"{row['pearson_correlation']:.4f}",
                                f"{row['cosine_similarity']:.4f}",
                                f"{row['euclidean_distance']:.4f}"
                            ]
                        })
                        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**Prediction vs Ground Truth:**")
                        pred_emotion = row['predicted_emotion']
                        true_emotion = row['groundtruth_emotion']
                        true_most_probable_emotion = row.get('groundtruth_most_probable_emotion', true_emotion)
                        correct_annotated = pred_emotion == true_emotion
                        correct_most_probable = pred_emotion == true_most_probable_emotion
                        
                        st.metric("Predicted", pred_emotion)
                        st.metric("Ground Truth (Annotated)", true_emotion)
                        st.metric("Ground Truth (Most Probable)", true_most_probable_emotion)
                        st.metric("Correct (Annotated)", "✓" if correct_annotated else "✗")
                        st.metric("Correct (Most Probable)", "✓" if correct_most_probable else "✗")
    
    with tab7:
        st.header("Export Results for Paper")
        
        st.info("""
        **Export Options for Publication:**
        
        1. **Summary Tables**: CSV format for inclusion in papers
        2. **High-Quality Figures**: Publication-ready visualizations
        3. **Complete Results**: All metrics for supplementary materials
        """)
        
        # Export options
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.subheader("📊 Summary Data")
            
            # Create comprehensive summary
            all_summary_data = []
            for exp in experiments:
                for _, row in exp['emotion_metrics'].iterrows():
                    all_summary_data.append({
                        'Experiment': exp['name'],
                        'Emotion': row['emotion'],
                        'Accuracy_Annotated': exp['accuracy_annotated'],
                        'Accuracy_Most_Probable': exp['accuracy_most_probable'],
                        'Precision': row['precision'],
                        'Recall': row['recall'],
                        'F1_Score': row['f1_score'],
                        'KL_Divergence': row['kl_div_mean'],
                        'Pearson_Correlation': row['pearson_correlation_mean'],
                        'Samples': row['samples']
                    })
            
            all_summary_df = pd.DataFrame(all_summary_data)
            csv_all_summary = all_summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download Complete Summary",
                data=csv_all_summary,
                file_name="complete_experiment_summary.csv",
                mime="text/csv",
                key="download_complete_summary"
            )
        
        with col_exp2:
            st.subheader("📈 Publication Figures")
            
            # Figure format selection
            fig_format = st.selectbox(
                "Figure Format:",
                ["PNG (High-Res)", "SVG (Vector)", "PDF (Vector)"],
                key="fig_format_select"
            )
            
            # Figure selection
            fig_options = {
                'performance_comparison': 'Performance Comparison Bar Chart',
                'kl_js_comparison': 'KL vs JS Divergence Scatter Plot',
                'emotion_heatmap': 'Emotion Performance Heatmap',
                'model_ranking': 'Model Ranking Chart',
                'metric_distributions': 'Metric Distributions Violin Plot'
            }
            
            selected_fig = st.selectbox(
                "Select Figure to Export:",
                list(fig_options.keys()),
                format_func=lambda x: fig_options[x],
                key="figure_export_select"
            )
            
            if st.button("Generate Figure for Export", key="generate_fig_button"):
                # Generate the selected figure
                if selected_fig == 'performance_comparison':
                    # Generate performance comparison figure
                    fig = go.Figure()
                    for i, exp in enumerate(experiments):
                        fig.add_trace(go.Bar(
                            name=exp['name'],
                            x=['Accuracy (Ann)', 'Accuracy (Prob)', 'F1-Score', 'Cosine Sim', 'Pearson Corr'],
                            y=[
                                exp['accuracy_annotated'],
                                exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 0,
                                exp['summary_metrics']['f1_macro'],
                                exp['summary_metrics']['cosine_mean'],
                                exp['summary_metrics']['pearson_correlation_mean']
                            ],
                            marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)]
                        ))
                    
                    fig.update_layout(
                        barmode='group',
                        title='Performance Comparison',
                        yaxis_title='Score',
                        height=500
                    )
                    
                elif selected_fig == 'kl_js_comparison':
                    # Generate KL vs JS divergence figure
                    kl_js_data = []
                    for exp in experiments:
                        kl_js_data.append({
                            'Experiment': exp['name'],
                            'KL Divergence': exp['summary_metrics']['kl_div_mean'],
                            'JS Divergence': exp['summary_metrics']['js_div_mean'],
                            'Accuracy (Annotated)': exp['accuracy_annotated']
                        })
                    
                    kl_js_df = pd.DataFrame(kl_js_data)
                    fig = px.scatter(
                        kl_js_df,
                        x='KL Divergence',
                        y='JS Divergence',
                        color='Experiment',
                        size='Accuracy (Annotated)',
                        title='KL Divergence vs JS Divergence'
                    )
                    
                elif selected_fig == 'emotion_heatmap':
                    # Generate emotion heatmap
                    emotion_acc_data = []
                    for exp in experiments:
                        for _, row in exp['emotion_metrics'].iterrows():
                            emotion_acc_data.append({
                                'Experiment': exp['name'],
                                'Emotion': row['emotion'],
                                'Accuracy': row['accuracy']
                            })
                    
                    emotion_acc_df = pd.DataFrame(emotion_acc_data)
                    pivot_data = emotion_acc_df.pivot_table(
                        index='Experiment', 
                        columns='Emotion', 
                        values='Accuracy',
                        aggfunc='mean'
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        colorscale='Viridis',
                        colorbar_title="Accuracy"
                    ))
                    fig.update_layout(title='Emotion Accuracy Heatmap')
                
                st.success(f"Figure '{fig_options[selected_fig]}' generated!")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"In a full implementation, this would export as {fig_format}")
        
        with col_exp3:
            st.subheader("📋 LaTeX Tables")
            
            # Generate LaTeX formatted tables
            if st.button("Generate LaTeX Tables", key="latex_button"):
                # Create LaTeX formatted summary table
                latex_table = r"\begin{table}[h!]"
                latex_table += r"\centering"
                latex_table += r"\caption{Performance Comparison of FER Models}"
                latex_table += r"\label{tab:performance_comparison}"
                latex_table += r"\begin{tabular}{lcccccc}"
                latex_table += r"\hline"
                latex_table += r"Model & Acc. (Ann.) & Acc. (Prob.) & F1-Score & Cosine Sim. & Pearson Corr. & JS Div. \\"
                latex_table += r"\hline"
                
                for exp in experiments:
                    latex_table += f"{exp['name']} & "
                    latex_table += f"{exp['accuracy_annotated']:.2%} & "
                    latex_table += f"{exp['accuracy_most_probable']:.2% if exp['accuracy_most_probable'] is not None else 'N/A'} & "
                    latex_table += f"{exp['summary_metrics']['f1_macro']:.3f} & "
                    latex_table += f"{exp['summary_metrics']['cosine_mean']:.3f} & "
                    latex_table += f"{exp['summary_metrics']['pearson_correlation_mean']:.3f} & "
                    latex_table += f"{exp['summary_metrics']['js_div_mean']:.4f} \\\\"
                
                latex_table += r"\hline"
                latex_table += r"\end{tabular}"
                latex_table += r"\end{table}"
                
                st.code(latex_table, language='latex')
                
                st.info("Copy and paste this LaTeX code into your paper")
        
        # Batch export all data
        st.subheader("🎯 Batch Export All Results")
        
        if st.button("📦 Export Complete Analysis Package", type="primary", key="export_package_button"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "fer_analysis_package.zip")
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # Add summary data
                    summary_path = os.path.join(tmpdir, "experiment_summary.csv")
                    summary_df.to_csv(summary_path, index=False)
                    zipf.write(summary_path, "experiment_summary.csv")
                    
                    # Add detailed reports for each experiment
                    for exp in experiments:
                        # Experiment summary
                        exp_summary = pd.DataFrame({
                            'Metric': [
                                'Accuracy (Annotated)', 'Accuracy (Most Probable)', 'F1-Score (Macro)', 
                                'Precision (Macro)', 'Recall (Macro)',
                                'JS Divergence', 'KL Divergence', 'Cosine Similarity', 'Pearson Correlation',
                                'Euclidean Distance'
                            ],
                            'Value': [
                                exp['accuracy_annotated'],
                                exp['accuracy_most_probable'] if exp['accuracy_most_probable'] is not None else 'N/A',
                                exp['summary_metrics']['f1_macro'],
                                exp['summary_metrics']['precision_macro'],
                                exp['summary_metrics']['recall_macro'],
                                exp['summary_metrics']['js_div_mean'],
                                exp['summary_metrics']['kl_div_mean'],
                                exp['summary_metrics']['cosine_mean'],
                                exp['summary_metrics']['pearson_correlation_mean'],
                                exp['summary_metrics']['euclidean_mean']
                            ]
                        })
                        
                        exp_path = os.path.join(tmpdir, f"{exp['name']}_summary.csv")
                        exp_summary.to_csv(exp_path, index=False)
                        zipf.write(exp_path, f"{exp['name']}_summary.csv")
                        
                        # Add emotion metrics
                        emo_path = os.path.join(tmpdir, f"{exp['name']}_emotion_metrics.csv")
                        exp['emotion_metrics'].to_csv(emo_path, index=False)
                        zipf.write(emo_path, f"{exp['name']}_emotion_metrics.csv")
                        
                        # Add classification report
                        class_report_path = os.path.join(tmpdir, f"{exp['name']}_classification_report.csv")
                        pd.DataFrame(exp['classification_report']).T.to_csv(class_report_path)
                        zipf.write(class_report_path, f"{exp['name']}_classification_report.csv")
                
                # Read ZIP file for download
                with open(zip_path, 'rb') as f:
                    zip_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Complete Package (ZIP)",
                    data=zip_bytes,
                    file_name="fer_analysis_package.zip",
                    mime="application/zip",
                    key="download_zip_package"
                )
                
                st.success("Package created successfully! Download using the button above.")

else:
    # Initial state - show instructions
    st.info("👈 **Start by adding experiment pairs in the sidebar**")
    
    with st.expander("📋 Instructions for Use"):
        st.markdown("""
        ## Multi-Experiment FER Comparison Tool
        
        ### **Step 1: Add Experiment Pairs**
        1. Click "➕ Add New Experiment" in the sidebar
        2. Give your experiment a meaningful name (e.g., "ResNet50_FocalLoss")
        3. Upload **two files** for each experiment:
           - **Predictions CSV**: Model predictions with emotion probabilities
           - **Ground Truth CSV**: Corresponding ground truth labels
        
        ### **Step 2: Configure Analysis**
        - Select which metrics to compare (including new KL Divergence and Pearson Correlation)
        - Choose visualization style ("Publication" for paper-ready plots)
        - Select color palette for consistent visuals
        
        ### **Step 3: Process Experiments**
        - Click "🚀 Process All Experiments" to analyze all added pairs
        - View progress as each experiment is processed
        
        ### **Step 4: Explore Results**
        The tool provides 7 analysis tabs:
        
        1. **📊 Overview**: Summary table and multi-metric radar chart
        2. **📈 Comparative Analysis**: Bar charts for selected metrics, performance matrix
        3. **🎭 Emotion Performance**: Emotion-wise accuracy, heatmaps, detailed metrics
        4. **🤖 Model Ranking**: Composite scoring, statistical significance tests
        5. **📉 Distribution Analysis**: Violin plots, correlation analysis
        6. **📋 Detailed Reports**: Individual experiment reports with confusion matrices
        7. **📥 Export Results**: Export data and figures for publication
        
        ### **📁 Expected File Format:**
        Both prediction and ground truth files should contain:
        ```
        Columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral, file
        Optional: emotion_label, valence, arousal, dominance
        ```
        
        ### **🎯 New Features:**
        - **Two Accuracy Metrics**:
          1. **Accuracy (Annotated)**: Uses the manually annotated label from ground truth
          2. **Accuracy (Most Probable)**: Uses the most probable class calculated from ground truth probabilities
        
        ### **🎯 Perfect for Paper Writing:**
        - Publication-ready visualizations
        - Statistical significance testing
        - LaTeX table generation
        - Batch export of all results
        - High-resolution figure export
        """)
    
    # Quick start example
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📁 Add Experiment Pairs**")
        st.write("Each experiment = Predictions + Ground Truth")
    
    with col2:
        st.markdown("**⚙️ Configure Analysis**")
        st.write("Select metrics and visualization style")
    
    with col3:
        st.markdown("**📊 Get Results**")
        st.write("Compare multiple models side by side")