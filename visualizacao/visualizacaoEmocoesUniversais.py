import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import jensenshannon as scipy_jensenshannon
from scipy.stats import entropy as scipy_entropy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import hashlib
import io
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition Performance Analysis",
    page_icon="😊",
    layout="wide"
)

# Application title
st.title("📊 Facial Emotion Recognition Performance Analysis")
st.markdown("---")

# ============================================================================
# Export Settings & Fonts
# ============================================================================

# Sidebar for export settings
st.sidebar.header("📤 Export Settings")

# Vector format selection
export_format = st.sidebar.selectbox(
    "Preferred Export Format",
    ["PDF (Recommended)", "SVG", "EPS", "PNG (High-Res)"],
    help="Choose vector format for high-quality export"
)

# Font size controls for bar charts
st.sidebar.header("🔤 Bar Chart Font Sizes")

bar_title_size = st.sidebar.slider("Title Size", 8, 24, 14, key="bar_title")
bar_axes_size = st.sidebar.slider("Axes Labels Size", 8, 20, 12, key="bar_axes")
bar_tick_size = st.sidebar.slider("Tick Labels Size", 6, 18, 11, key="bar_tick")
bar_legend_size = st.sidebar.slider("Legend Size", 6, 18, 11, key="bar_legend")
bar_label_size = st.sidebar.slider("Bar Labels Size", 6, 14, 8, key="bar_label")

# Font size controls for other charts
st.sidebar.header("🔤 Other Charts Font Sizes")
other_title_size = st.sidebar.slider("Other Charts Title Size", 8, 24, 14, key="other_title")
other_labels_size = st.sidebar.slider("Other Charts Labels Size", 8, 20, 11, key="other_labels")

# ============================================================================
# FUNCTIONS: Vector export and font control
# ============================================================================

def create_vector_figure_matplotlib(fig, format='pdf'):
    """Save Matplotlib figure to vector format"""
    buf = BytesIO()
    if format == 'pdf' or format == 'PDF (Recommended)':
        fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
        mime = 'application/pdf'
    elif format == 'svg':
        fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
        mime = 'image/svg+xml'
    elif format == 'eps':
        fig.savefig(buf, format='eps', dpi=300, bbox_inches='tight')
        mime = 'application/eps'
    elif format == 'png' or format == 'PNG (High-Res)':
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        mime = 'image/png'
    buf.seek(0)
    return buf, mime

def create_comparison_bar_chart_matplotlib(pred_probs, gt_probs, emotion_names, 
                                          filename="sample", sample_idx=0,
                                          font_sizes=None):
    """Create comparison bar chart using Matplotlib for vector export"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'tick_labels': 11,
            'legend': 11,
            'bar_labels': 8
        }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(emotion_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pred_probs, width, 
                   label='Predicted', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, gt_probs, width, 
                   label='Ground Truth', color='red', alpha=0.7)
    
    ax.set_xlabel('Emotion', fontsize=font_sizes.get('axes_labels', 12))
    ax.set_ylabel('Probability', fontsize=font_sizes.get('axes_labels', 12))
    ax.set_title(f'Distribution Comparison - {filename} (Sample {sample_idx})', 
                 fontsize=font_sizes.get('title', 14), pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right', 
                       fontsize=font_sizes.get('tick_labels', 11))
    ax.tick_params(axis='y', labelsize=font_sizes.get('tick_labels', 11))
    ax.legend(fontsize=font_sizes.get('legend', 11), loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=font_sizes.get('bar_labels', 8))
    
    ax.yaxis.grid(True, alpha=0.2)
    max_prob = max(max(pred_probs), max(gt_probs))
    ax.set_ylim(0, min(1.0, max_prob * 1.2))
    
    plt.tight_layout()
    return fig

def create_radar_chart_matplotlib(pred_probs, gt_probs, emotion_names, 
                                 filename="sample", sample_idx=0,
                                 font_sizes=None):
    """Create radar chart using Matplotlib for vector export"""
    if font_sizes is None:
        font_sizes = {
            'title': 16,
            'tick_labels': 12,
            'legend': 11,
            'radial_labels': 10
        }
    
    angles = np.linspace(0, 2 * np.pi, len(emotion_names), endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    pred_probs_closed = pred_probs.tolist() if hasattr(pred_probs, 'tolist') else list(pred_probs)
    gt_probs_closed = gt_probs.tolist() if hasattr(gt_probs, 'tolist') else list(gt_probs)
    
    pred_probs_closed += pred_probs_closed[:1]
    gt_probs_closed += gt_probs_closed[:1]
    
    ax.plot(angles, pred_probs_closed, 'o-', linewidth=2, label='Predicted')
    ax.fill(angles, pred_probs_closed, alpha=0.25, color='blue')
    
    ax.plot(angles, gt_probs_closed, 'o-', linewidth=2, label='Ground Truth')
    ax.fill(angles, gt_probs_closed, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotion_names, fontsize=font_sizes.get('tick_labels', 12))
    
    ax.set_ylim(0, max(max(pred_probs_closed), max(gt_probs_closed)) * 1.1)
    ax.tick_params(axis='y', labelsize=font_sizes.get('radial_labels', 10))
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=font_sizes.get('legend', 11))
    ax.set_title(f'Radar Chart - {filename} (Sample {sample_idx})', 
                 size=font_sizes.get('title', 16), y=1.1, pad=20)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_matplotlib(cm, emotion_names, font_sizes=None):
    """Create confusion matrix using Matplotlib for vector export"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'tick_labels': 10,
            'cell_values': 9
        }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=font_sizes.get('axes_labels', 12))
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=emotion_names,
           yticklabels=emotion_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Predicted Emotion', fontsize=font_sizes.get('axes_labels', 12), labelpad=10)
    ax.set_ylabel('True Emotion', fontsize=font_sizes.get('axes_labels', 12), labelpad=10)
    ax.set_title('Confusion Matrix', fontsize=font_sizes.get('title', 14), pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=font_sizes.get('tick_labels', 10))
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=font_sizes.get('cell_values', 9))
    
    plt.tight_layout()
    return fig

def create_metrics_histogram_matplotlib(metric_data, metric_name, font_sizes=None):
    """Create histogram for distribution metrics using Matplotlib"""
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'axes_labels': 12,
            'tick_labels': 11,
            'legend': 11
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(metric_data, bins=30, color='teal', 
                               alpha=0.7, edgecolor='black')
    
    ax.set_xlabel(metric_name, fontsize=font_sizes.get('axes_labels', 12))
    ax.set_ylabel('Frequency', fontsize=font_sizes.get('axes_labels', 12))
    ax.set_title(f'Distribution of {metric_name}', fontsize=font_sizes.get('title', 14))
    ax.tick_params(axis='both', labelsize=font_sizes.get('tick_labels', 11))
    ax.grid(True, alpha=0.3)
    
    mean_val = np.mean(metric_data)
    median_val = np.median(metric_data)
    ax.axvline(mean_val, color='red', linestyle='--', 
               label=f'Mean: {mean_val:.3f}', linewidth=2)
    ax.axvline(median_val, color='green', linestyle='--', 
               label=f'Median: {median_val:.3f}', linewidth=2)
    ax.legend(fontsize=font_sizes.get('legend', 11))
    
    plt.tight_layout()
    return fig

# ============================================================================
# ORIGINAL FUNCTIONS
# ============================================================================

def safe_kl_divergence(p, q):
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
        
        kl = scipy_entropy(p_safe, q_safe)
        
        return kl if not np.isnan(kl) and not np.isinf(kl) else np.nan
    except Exception as e:
        return np.nan

def safe_js_divergence(p, q):
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

def generate_data_hash(predictions_content, groundtruth_content):
    combined = predictions_content + groundtruth_content
    return hashlib.md5(combined.encode()).hexdigest()

@st.cache_data(ttl=3600)
def process_data(predictions_content, groundtruth_content, data_hash):
    predictions_df = pd.read_csv(io.StringIO(predictions_content))
    groundtruth_df = pd.read_csv(io.StringIO(groundtruth_content))
    
    emotion_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    missing_in_pred = [col for col in emotion_columns if col not in predictions_df.columns]
    missing_in_gt = [col for col in emotion_columns if col not in groundtruth_df.columns]
    
    if missing_in_pred:
        st.error(f"⚠️ Missing columns in predictions file: {missing_in_pred}")
        return None, None, None, None, None, None, None
    if missing_in_gt:
        st.error(f"⚠️ Missing columns in ground truth file: {missing_in_gt}")
        return None, None, None, None, None, None, None
    
    for col in emotion_columns:
        predictions_df[col] = pd.to_numeric(predictions_df[col], errors='coerce')
        groundtruth_df[col] = pd.to_numeric(groundtruth_df[col], errors='coerce')
    
    for col in emotion_columns:
        predictions_df[col] = predictions_df[col].fillna(1e-12)
        groundtruth_df[col] = groundtruth_df[col].fillna(1e-12)
    
    predictions_df = predictions_df.sort_values('file').reset_index(drop=True)
    groundtruth_df = groundtruth_df.sort_values('file').reset_index(drop=True)
    
    extra_columns = []
    for col in ['valence', 'arousal', 'dominance']:
        if col in groundtruth_df.columns:
            extra_columns.append(col)
            groundtruth_df[col] = pd.to_numeric(groundtruth_df[col], errors='coerce')
    
    if not predictions_df['file'].equals(groundtruth_df['file']):
        st.warning("⚠️ Files are not perfectly aligned by image. Merging by file paths...")
        merged_df = pd.merge(predictions_df, groundtruth_df, on='file', suffixes=('_pred', '_gt'), how='inner')
    else:
        merged_df = predictions_df.copy()
        for col in emotion_columns:
            merged_df[f"{col}_gt"] = groundtruth_df[col]
        for col in extra_columns:
            merged_df[f"{col}_gt"] = groundtruth_df[col]
        if 'emotion_label' in groundtruth_df.columns:
            merged_df['emotion_label_gt'] = groundtruth_df['emotion_label']
    
    rename_dict = {}
    if 'emotion_label_pred' in merged_df.columns:
        rename_dict['emotion_label_pred'] = 'pred_label'
    elif 'emotion_label' in merged_df.columns and 'emotion_label_gt' in merged_df.columns:
        rename_dict['emotion_label'] = 'pred_label'
        rename_dict['emotion_label_gt'] = 'true_label'
    elif 'emotion_label' in merged_df.columns:
        rename_dict['emotion_label'] = 'pred_label'
    
    if rename_dict:
        merged_df = merged_df.rename(columns=rename_dict)
    
    column_to_label = {
        0: 1,  # happy
        1: 7,  # contempt
        2: 3,  # surprised
        3: 6,  # angry
        4: 5,  # disgusted
        5: 4,  # fearful
        6: 2,  # sad
        7: 0   # neutral
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
    
    pred_probs = merged_df[emotion_columns].values
    merged_df['predicted_label'] = np.argmax(pred_probs, axis=1)
    merged_df['predicted_label'] = merged_df['predicted_label'].map(column_to_label)
    merged_df['predicted_emotion'] = merged_df['predicted_label'].map(emotion_mapping)
    
    if 'true_label' in merged_df.columns:
        merged_df['groundtruth_label'] = merged_df['true_label']
    elif 'emotion_label_gt' in merged_df.columns:
        merged_df['groundtruth_label'] = merged_df['emotion_label_gt']
    else:
        st.error("❌ Ground truth file doesn't have emotion_label column")
        return None, None, None, None, None, None, None
    
    gt_emotion_columns = [f"{col}_gt" for col in emotion_columns]
    gt_probs = merged_df[gt_emotion_columns].values
    
    merged_df['gt_distribution_label'] = np.argmax(gt_probs, axis=1)
    merged_df['gt_distribution_label'] = merged_df['gt_distribution_label'].map(column_to_label)
    merged_df['gt_distribution_emotion'] = merged_df['gt_distribution_label'].map(emotion_mapping)
    
    merged_df['groundtruth_emotion'] = merged_df['groundtruth_label'].map(emotion_mapping)
    merged_df['annotation_distribution_discrepancy'] = merged_df['groundtruth_label'] != merged_df['gt_distribution_label']
    merged_df['gt_max_probability'] = np.max(gt_probs, axis=1)
    
    def calculate_entropy(row):
        probs = row[gt_emotion_columns].values.astype(np.float64)
        probs_safe = probs + 1e-12
        probs_sum = np.sum(probs_safe)
        if probs_sum <= 0: return np.nan
        probs_safe = probs_safe / probs_sum
        return scipy_entropy(probs_safe)
    
    merged_df['gt_distribution_entropy'] = merged_df.apply(calculate_entropy, axis=1)
    
    original_count = len(merged_df)
    merged_df = merged_df[merged_df['groundtruth_label'].isin(emotion_mapping.keys())].copy()
    filtered_count = len(merged_df)
    
    if original_count != filtered_count:
        st.warning(f"⚠️ Removed {original_count - filtered_count} samples with invalid labels.")
    
    vad_expectations = {
        'neutral': {'valence': 'neutral', 'arousal': 'low', 'dominance': 'neutral'},
        'happy': {'valence': 'high', 'arousal': 'medium/high', 'dominance': 'high'},
        'sad': {'valence': 'low', 'arousal': 'low', 'dominance': 'low'},
        'surprised': {'valence': 'medium', 'arousal': 'high', 'dominance': 'medium'},
        'fearful': {'valence': 'low', 'arousal': 'high', 'dominance': 'low'},
        'disgusted': {'valence': 'low', 'arousal': 'medium', 'dominance': 'medium'},
        'angry': {'valence': 'low', 'arousal': 'high', 'dominance': 'high'},
        'contempt': {'valence': 'low', 'arousal': 'low', 'dominance': 'high'}
    }
    
    def calculate_vad_consistency(row):
        if not extra_columns: return pd.Series({'vad_consistency_score': np.nan})
        consistency_scores = []
        emotion = row['groundtruth_emotion']
        if emotion not in vad_expectations: return pd.Series({'vad_consistency_score': np.nan})
        expected = vad_expectations[emotion]
        
        for dim in extra_columns:
            if f"{dim}_gt" not in row.index: continue
            value = row[f"{dim}_gt"]
            if pd.isna(value): continue
            norm_value = (value + 1) / 2
            
            if dim == 'valence':
                if expected['valence'] == 'high': expected_range = (0.7, 1.0)
                elif expected['valence'] == 'low': expected_range = (0.0, 0.3)
                elif expected['valence'] == 'medium': expected_range = (0.3, 0.7)
                else: expected_range = (0.4, 0.6)
            elif dim == 'arousal':
                if expected['arousal'] == 'high': expected_range = (0.7, 1.0)
                elif expected['arousal'] == 'low': expected_range = (0.0, 0.3)
                elif 'medium' in expected['arousal']: expected_range = (0.3, 0.7)
                else: expected_range = (0.4, 0.6)
            elif dim == 'dominance':
                if expected['dominance'] == 'high': expected_range = (0.7, 1.0)
                elif expected['dominance'] == 'low': expected_range = (0.0, 0.3)
                elif expected['dominance'] == 'medium': expected_range = (0.3, 0.7)
                else: expected_range = (0.4, 0.6)
            
            if expected_range[0] <= norm_value <= expected_range[1]:
                consistency_scores.append(1.0)
            else:
                if norm_value < expected_range[0]: distance = expected_range[0] - norm_value
                else: distance = norm_value - expected_range[1]
                penalty = min(distance * 2, 0.5)
                consistency_scores.append(1.0 - penalty)
        
        if consistency_scores: return pd.Series({'vad_consistency_score': np.mean(consistency_scores)})
        else: return pd.Series({'vad_consistency_score': np.nan})
    
    def calculate_distribution_metrics(row):
        try:
            pred_dist = []
            gt_dist = []
            for col in emotion_columns:
                pred_val = row[col]
                gt_val = row[f"{col}_gt"]
                pred_dist.append(float(pred_val) if isinstance(pred_val, (int, float, np.number)) else 1e-12)
                gt_dist.append(float(gt_val) if isinstance(gt_val, (int, float, np.number)) else 1e-12)
            
            pred_dist = np.array(pred_dist, dtype=np.float64)
            gt_dist = np.array(gt_dist, dtype=np.float64)
            
            pred_dist = np.nan_to_num(pred_dist, nan=1e-12, posinf=1e-12, neginf=1e-12)
            gt_dist = np.nan_to_num(gt_dist, nan=1e-12, posinf=1e-12, neginf=1e-12)
            
            pred_dist_safe = pred_dist + 1e-12
            gt_dist_safe = gt_dist + 1e-12
            
            pred_sum = np.sum(pred_dist_safe)
            gt_sum = np.sum(gt_dist_safe)
            
            if pred_sum > 0: pred_dist_safe = pred_dist_safe / pred_sum
            else: pred_dist_safe = np.ones_like(pred_dist_safe) / len(pred_dist_safe)
            
            if gt_sum > 0: gt_dist_safe = gt_dist_safe / gt_sum
            else: gt_dist_safe = np.ones_like(gt_dist_safe) / len(gt_dist_safe)
            
            js_div = safe_js_divergence(pred_dist_safe, gt_dist_safe)
            kl_div_1 = safe_kl_divergence(pred_dist_safe, gt_dist_safe)
            kl_div_2 = safe_kl_divergence(gt_dist_safe, pred_dist_safe)
            kl_div = 0.5 * (kl_div_1 + kl_div_2) if not np.isnan(kl_div_1) and not np.isnan(kl_div_2) else np.nan
            
            euclidean_dist = np.linalg.norm(pred_dist_safe - gt_dist_safe)
            dot_product = np.dot(pred_dist_safe, gt_dist_safe)
            norm_pred = np.linalg.norm(pred_dist_safe)
            norm_gt = np.linalg.norm(gt_dist_safe)
            cosine_sim = dot_product / (norm_pred * norm_gt) if norm_pred > 0 and norm_gt > 0 else 0.0
            
            if len(pred_dist_safe) > 1:
                correlation = np.corrcoef(pred_dist_safe, gt_dist_safe)[0, 1]
                if np.isnan(correlation): correlation = 0.0
            else:
                correlation = 1.0 if pred_dist_safe[0] == gt_dist_safe[0] else 0.0
            
            return pd.Series({
                'js_divergence': 0.0 if np.isnan(js_div) else js_div,
                'kl_divergence': 0.0 if np.isnan(kl_div) else kl_div,
                'euclidean_distance': 0.0 if np.isnan(euclidean_dist) else euclidean_dist,
                'cosine_similarity': cosine_sim,
                'pearson_correlation': correlation
            })
        except Exception as e:
            return pd.Series({'js_divergence': 0.0, 'kl_divergence': 0.0, 'euclidean_distance': 0.0, 'cosine_similarity': 0.0, 'pearson_correlation': 0.0})
    
    distribution_metrics = merged_df.apply(calculate_distribution_metrics, axis=1)
    merged_df = pd.concat([merged_df, distribution_metrics], axis=1)
    
    if extra_columns:
        vad_consistency = merged_df.apply(calculate_vad_consistency, axis=1)
        merged_df = pd.concat([merged_df, vad_consistency], axis=1)
    
    return merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.sidebar.header("📁 File Upload")

predictions_file = st.sidebar.file_uploader(
    "Predictions File (simpleNetwork_FocalLoss.csv)",
    type=['csv'],
    help="File with model predictions - should have columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral, emotion_label, file"
)

groundtruth_file = st.sidebar.file_uploader(
    "Ground Truth File (simpleNetwork_FocalLoss_labels.csv)",
    type=['csv'],
    help="File with ground truth values - should have columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral, valence, arousal, dominance, emotion_label, file"
)

if predictions_file is not None and groundtruth_file is not None:
    predictions_content = predictions_file.getvalue().decode('utf-8')
    groundtruth_content = groundtruth_file.getvalue().decode('utf-8')
    
    data_hash = generate_data_hash(predictions_content, groundtruth_content)
    
    try:
        result = process_data(predictions_content, groundtruth_content, data_hash)
        
        if result[0] is None:
            st.error("❌ Error processing data. Please check file formats.")
            st.stop()
        
        merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations = result
        
        metrics_with_nan = merged_df[['js_divergence', 'kl_divergence', 'cosine_similarity', 'pearson_correlation']].isna().sum()
        if metrics_with_nan.sum() > 0:
            st.warning(f"⚠️ Some metrics contain NaN values: {metrics_with_nan.to_dict()}")
            merged_df['js_divergence'] = merged_df['js_divergence'].fillna(0)
            merged_df['kl_divergence'] = merged_df['kl_divergence'].fillna(0)
            merged_df['cosine_similarity'] = merged_df['cosine_similarity'].fillna(0)
            merged_df['pearson_correlation'] = merged_df['pearson_correlation'].fillna(0)
        
        accuracy = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
        
        non_discrepant_df = merged_df[~merged_df['annotation_distribution_discrepancy']]
        discrepant_df = merged_df[merged_df['annotation_distribution_discrepancy']]
        
        clean_accuracy = accuracy_score(non_discrepant_df['groundtruth_label'], non_discrepant_df['predicted_label']) if len(non_discrepant_df) > 0 else 0.0
        discrepant_accuracy = accuracy_score(discrepant_df['groundtruth_label'], discrepant_df['predicted_label']) if len(discrepant_df) > 0 else 0.0
        
        valid_labels = list(emotion_mapping.keys())
        cm = confusion_matrix(merged_df['groundtruth_label'], merged_df['predicted_label'], labels=valid_labels)
        
        st.success(f"✅ Data loaded successfully! Total of {len(merged_df)} samples.")
        
        with st.expander("📋 Correct Emotion Mapping"):
            st.markdown("""
            | Label | Emotion |
            |-------|---------|
            | 0 | neutral |
            | 1 | happy |
            | 2 | sad |
            | 3 | surprised |
            | 4 | fearful |
            | 5 | disgusted |
            | 6 | angry |
            | 7 | contempt |
            """)
        
        if extra_columns:
            st.info(f"📊 Extra columns detected in ground truth: {', '.join(extra_columns)}")
        
        st.sidebar.header("🔧 Analysis Controls")
        
        metric_options = {
            'js_divergence': 'Jensen-Shannon Divergence',
            'kl_divergence': 'KL Divergence (Symmetric)',
            'euclidean_distance': 'Euclidean Distance',
            'cosine_similarity': 'Cosine Similarity',
            'pearson_correlation': 'Pearson Correlation'
        }
        
        selected_metric = st.sidebar.selectbox(
            "Select distribution metric for analysis:",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        num_samples = st.sidebar.slider("Number of samples to visualize:", min_value=5, max_value=min(50, len(merged_df)), value=min(10, len(merged_df)))
        all_emotions = list(emotion_mapping.values())
        selected_emotion = st.sidebar.multiselect("Filter by true emotion:", all_emotions, default=all_emotions)
        
        if selected_emotion:
            filtered_df = merged_df[merged_df['groundtruth_emotion'].isin(selected_emotion)].copy()
        else:
            filtered_df = merged_df.copy()
        
        # TAB DECLARATION
        tab1, tab2, tab_err_inspect, tab_err_pattern, tab_err_dist_pattern, tab_confidence, tab3, tab4, tab5, tab6, tab_clean, tab7, tab8, tab9 = st.tabs([
            "📈 Overview", 
            "🔍 Detailed Analysis", 
            "🖼️ Error Inspector",
            "📉 Error Patterns",
            "📉 Dist. Error Patterns", # <--- NOVA ABA AQUI
            "⚖️ Confidence & Thresholds",
            "📊 Distributions", 
            "🎭 Affective Dimensions",
            "🔎 Label-VAD Consistency",
            "⚠️ Annotation-Distribution Discrepancy",            
            "✨ Clean Performance",
            "📊 Distribution Metrics",
            "📋 Complete Data",
            "📤 Export Center"
        ])
        
        with tab1:
            st.header("Performance Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Overall Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Clean Accuracy", f"{clean_accuracy:.2%}", help="Accuracy on non-discrepant samples only.")
            with col3:
                avg_js = merged_df['js_divergence'].mean()
                st.metric("JS Divergence (avg)", f"{avg_js:.4f}")
            with col4:
                avg_cosine = merged_df['cosine_similarity'].mean()
                st.metric("Cosine Similarity (avg)", f"{avg_cosine:.4f}")
            with col5:
                correct_predictions = (merged_df['groundtruth_label'] == merged_df['predicted_label']).sum()
                total = len(merged_df)
                st.metric("Correct Predictions", f"{correct_predictions}/{total}")
            
            st.markdown("---")
            st.subheader("Confusion Matrix")
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=[emotion_mapping[i] for i in range(8)], y=[emotion_mapping[i] for i in range(8)],
                colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size": 10}, hoverongaps=False, colorbar_title="Count"
            ))
            fig_cm.update_layout(title="Confusion Matrix - Predicted vs True Emotions", xaxis_title="Predicted Emotion", yaxis_title="True Emotion", height=600)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.subheader("Distribution of Similarity Metrics")
            metrics_to_plot = ['js_divergence', 'cosine_similarity', 'pearson_correlation']
            fig_metrics = make_subplots(rows=1, cols=3, subplot_titles=['JS Divergence', 'Cosine Similarity', 'Pearson Correlation'])
            for i, metric in enumerate(metrics_to_plot, 1):
                fig_metrics.add_trace(go.Histogram(x=merged_df[metric], name=metric_options.get(metric, metric)), row=1, col=i)
            fig_metrics.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with tab2:
            st.header("Detailed Analysis by Emotion")
            
            emotion_stats = []
            for emotion_idx, emotion_name in emotion_mapping.items():
                mask = merged_df['groundtruth_label'] == emotion_idx
                if mask.sum() > 0:
                    subset = merged_df[mask]
                    accuracy_subset = accuracy_score(subset['groundtruth_label'], subset['predicted_label'])
                    stats = {
                        'Emotion': emotion_name, 'Samples': mask.sum(), 'Accuracy': accuracy_subset,
                        'JS Divergence (avg)': subset['js_divergence'].mean(), 'Cosine Similarity (avg)': subset['cosine_similarity'].mean(),
                        'Correlation (avg)': subset['pearson_correlation'].mean()
                    }
                    emotion_stats.append(stats)
            
            stats_df = pd.DataFrame(emotion_stats)
            st.dataframe(stats_df.style.format({'Accuracy': '{:.2%}', 'JS Divergence (avg)': '{:.4f}', 'Cosine Similarity (avg)': '{:.4f}', 'Correlation (avg)': '{:.4f}'}), use_container_width=True)
            
            fig_acc = px.bar(stats_df, x='Emotion', y='Accuracy', title='Accuracy by Emotion', color='Accuracy', color_continuous_scale='Viridis')
            fig_acc.update_layout(height=400)
            st.plotly_chart(fig_acc, use_container_width=True)
            
            st.subheader("Most Common Errors")
            error_pairs = merged_df[merged_df['groundtruth_label'] != merged_df['predicted_label']]
            confusion_pairs = error_pairs.groupby(['groundtruth_emotion', 'predicted_emotion']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False).head(10)
            
            if not confusion_pairs.empty:
                fig_errors = px.bar(confusion_pairs, x='count', y='groundtruth_emotion', color='predicted_emotion', orientation='h', title='Top 10 Confusion Pairs (True → Predicted)', labels={'count': 'Error Count', 'groundtruth_emotion': 'True Emotion'})
                fig_errors.update_layout(height=500)
                st.plotly_chart(fig_errors, use_container_width=True)
            else:
                st.info("No errors to display!")
                
        with tab_err_inspect:
            st.header("🖼️ Error Inspector: Visualizing Incorrect Classifications")
            st.markdown("""
            Analyze individual samples where the model failed. Compare the image, manual annotation, 
            estimated distribution (Ground Truth), and model's guess (Prediction).
            """)
            
            errors_df = filtered_df[filtered_df['groundtruth_label'] != filtered_df['predicted_label']].copy()
            
            if errors_df.empty:
                st.success("🎉 No classification errors in the filtered samples!")
            else:
                img_dir = st.text_input("Base image directory (optional - leave blank if you don't want to load the actual image):", 
                                      help="Example: C:/datasets/FER/images/ or ./data/images/")
                
                error_options = errors_df['file'].tolist()
                selected_error_file = st.selectbox("Select the incorrectly classified sample:", error_options)
                
                if selected_error_file:
                    err_data = errors_df[errors_df['file'] == selected_error_file].iloc[0]
                    
                    col_img, col_info = st.columns([1, 2])
                    
                    with col_img:
                        st.subheader("Image")
                        if img_dir:
                            img_path = os.path.join(img_dir, selected_error_file)
                            try:
                                st.image(img_path, caption=selected_error_file, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load image at: {img_path}")
                                st.code(selected_error_file)
                        else:
                            st.info("Image directory not provided.")
                            st.code(selected_error_file)
                            
                        st.markdown(f"**Original Annotation:** `{err_data['groundtruth_emotion'].upper()}`")
                        st.markdown(f"**Model's Guess:** `{err_data['predicted_emotion'].upper()}`")
                    
                    with col_info:
                        st.subheader("Distributions: Ground Truth vs Prediction")
                        
                        column_to_label_err = {0: 1, 1: 7, 2: 3, 3: 6, 4: 5, 5: 4, 6: 2, 7: 0}
                        p_probs, g_probs = [0.0]*8, [0.0]*8
                        
                        for i, col in enumerate(emotion_columns):
                            lbl = column_to_label_err[i]
                            p_probs[lbl] = float(err_data[col])
                            g_probs[lbl] = float(err_data[f"{col}_gt"])
                        
                        emotion_names_err = [emotion_mapping[i] for i in range(8)]
                        
                        fig_err_dist = go.Figure()
                        fig_err_dist.add_trace(go.Bar(x=emotion_names_err, y=p_probs, name='Predicted Distribution', marker_color='indianred'))
                        fig_err_dist.add_trace(go.Bar(x=emotion_names_err, y=g_probs, name='Original Distribution (GT)', marker_color='royalblue'))
                        
                        fig_err_dist.update_layout(barmode='group', height=400, yaxis_title="Probability")
                        st.plotly_chart(fig_err_dist, use_container_width=True)
                        
                        col_met1, col_met2 = st.columns(2)
                        with col_met1:
                            st.metric("Error Confidence (Pred Max)", f"{np.max(p_probs):.3f}")
                        with col_met2:
                            st.metric("Most Probable Value (GT Max)", f"{np.max(g_probs):.3f}")

        with tab_err_pattern:
            st.header("📉 Error Patterns: Behavioral Patterns in Errors")
            st.markdown("Exclusive global analysis of samples where the model failed, helping to identify systemic biases.")
            
            global_errors_df = merged_df[merged_df['groundtruth_label'] != merged_df['predicted_label']].copy()
            
            if global_errors_df.empty:
                st.success("No errors to analyze patterns!")
            else:
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.subheader("Bias Heatmap (Errors Only)")
                    st.markdown("Read as: 'When the real emotion was X (row), the model guessed Y (column) N times'.")
                    
                    error_cm = confusion_matrix(global_errors_df['groundtruth_label'], global_errors_df['predicted_label'], labels=valid_labels)
                    
                    fig_err_cm = go.Figure(data=go.Heatmap(
                        z=error_cm, 
                        x=[emotion_mapping[i] for i in range(8)], 
                        y=[emotion_mapping[i] for i in range(8)],
                        colorscale='Reds', text=error_cm, texttemplate='%{text}'
                    ))
                    fig_err_cm.update_layout(xaxis_title="What the model guessed", yaxis_title="Real Annotation", height=500)
                    st.plotly_chart(fig_err_cm, use_container_width=True)
                
                with col_p2:
                    st.subheader("Where do the errors go?")
                    error_grouped = global_errors_df.groupby(['groundtruth_emotion', 'predicted_emotion']).size().reset_index(name='count')
                    
                    fig_err_bar = px.bar(
                        error_grouped, x='groundtruth_emotion', y='count', color='predicted_emotion',
                        title="Error Composition by Real Emotion",
                        labels={'groundtruth_emotion': 'Real Emotion', 'count': 'Error Count', 'predicted_emotion': 'Incorrect Guess'},
                        barmode='stack'
                    )
                    fig_err_bar.update_layout(height=500)
                    st.plotly_chart(fig_err_bar, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Confidence Level: Correct vs Incorrect")
                st.markdown("Is the model too 'confident' when it errs? (Max probability extracted from the predicted distribution)")
                
                merged_df['prediction_confidence'] = merged_df[emotion_columns].max(axis=1)
                merged_df['is_correct'] = merged_df['groundtruth_label'] == merged_df['predicted_label']
                merged_df['Status'] = merged_df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
                
                fig_conf_violin = px.violin(
                    merged_df, y='prediction_confidence', x='Status', color='Status',
                    box=True, points="all",
                    color_discrete_map={'Correct': 'mediumseagreen', 'Incorrect': 'indianred'},
                    labels={'prediction_confidence': 'Prediction Confidence (0 to 1)'}
                )
                fig_conf_violin.update_layout(height=400)
                st.plotly_chart(fig_conf_violin, use_container_width=True)
                
        with tab_confidence:
            st.header("⚖️ Model Confidence & Calibration")
            st.markdown("""
            Analyze how well the model's confidence scores align with actual accuracy (Calibration) 
            and simulate what happens if you reject predictions below a certain confidence threshold.
            """)
            
            if 'prediction_confidence' not in merged_df.columns:
                merged_df['prediction_confidence'] = merged_df[emotion_columns].max(axis=1)
            if 'is_correct' not in merged_df.columns:
                merged_df['is_correct'] = merged_df['groundtruth_label'] == merged_df['predicted_label']
                
            col_calib, col_thresh = st.columns(2)
            
            with col_calib:
                st.subheader("Reliability Diagram (Calibration Curve)")
                st.markdown("Perfect calibration follows the dashed diagonal line. Points below the line indicate the model is overconfident.")
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    merged_df['is_correct'], 
                    merged_df['prediction_confidence'], 
                    n_bins=10, 
                    strategy='uniform'
                )
                
                fig_calib = go.Figure()
                fig_calib.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines', 
                    name='Perfectly Calibrated', 
                    line=dict(dash='dash', color='gray')
                ))
                fig_calib.add_trace(go.Scatter(
                    x=mean_predicted_value, y=fraction_of_positives, 
                    mode='lines+markers', name='Model Calibration', 
                    marker=dict(size=10, color='royalblue'),
                    line=dict(width=2)
                ))
                
                fig_calib.update_layout(
                    xaxis_title="Mean Predicted Probability (Confidence)",
                    yaxis_title="Fraction of True Positives (Accuracy)",
                    height=500,
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    legend=dict(x=0.05, y=0.95)
                )
                st.plotly_chart(fig_calib, use_container_width=True)
                
            with col_thresh:
                st.subheader("Rejection Threshold Simulator")
                st.markdown("What happens to accuracy if we reject predictions where the model is uncertain?")
                
                threshold = st.slider("Minimum Confidence Threshold", min_value=0.0, max_value=0.99, value=0.50, step=0.05)
                
                kept_samples = merged_df[merged_df['prediction_confidence'] >= threshold]
                rejected_samples = merged_df[merged_df['prediction_confidence'] < threshold]
                
                coverage = len(kept_samples) / len(merged_df) if len(merged_df) > 0 else 0
                new_accuracy = kept_samples['is_correct'].mean() if len(kept_samples) > 0 else 0
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        "Accuracy on Kept Samples", 
                        f"{new_accuracy:.2%}", 
                        delta=f"{(new_accuracy - accuracy):.2%} vs Overall" if len(kept_samples) > 0 else None
                    )
                with col_m2:
                    st.metric(
                        "Samples Kept (Coverage)", 
                        f"{coverage:.2%}", 
                        delta=f"-{len(rejected_samples)} rejected samples" if len(rejected_samples) > 0 else None,
                        delta_color="inverse"
                    )
                
                thresholds = np.linspace(0, 0.99, 50)
                accs, covs = [], []
                for t in thresholds:
                    kept = merged_df[merged_df['prediction_confidence'] >= t]
                    covs.append(len(kept) / len(merged_df) if len(merged_df) > 0 else 0)
                    accs.append(kept['is_correct'].mean() if len(kept) > 0 else 0)
                    
                fig_tradeoff = go.Figure()
                fig_tradeoff.add_trace(go.Scatter(x=thresholds, y=accs, name='Accuracy', line=dict(color='mediumseagreen', width=3)))
                fig_tradeoff.add_trace(go.Scatter(x=thresholds, y=covs, name='Coverage', line=dict(color='orange', width=3)))
                fig_tradeoff.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Selected Threshold")
                
                fig_tradeoff.update_layout(
                    title="Accuracy vs Coverage Trade-off",
                    xaxis_title="Confidence Threshold",
                    yaxis_title="Percentage",
                    hovermode="x unified",
                    height=350,
                    margin=dict(t=40, b=0),
                    yaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig_tradeoff, use_container_width=True)
        with tab_err_dist_pattern:
            st.header("📉 Distribution Error Patterns")
            st.markdown("""
            This analysis evaluates errors using the **most probable emotion from the Ground Truth distribution**, 
            rather than the manual annotation. This helps identify if the model is failing against the actual 
            distributional consensus.
            """)
            
            # Filtrando erros com base na distribuição (GT) em vez da anotação
            dist_errors_df = merged_df[merged_df['gt_distribution_label'] != merged_df['predicted_label']].copy()
            
            if dist_errors_df.empty:
                st.success("No distribution-based errors to analyze!")
            else:
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.subheader("Bias Heatmap (Distribution Errors)")
                    st.markdown("Read as: 'When the GT distribution peak was X, the model guessed Y'.")
                    
                    dist_error_cm = confusion_matrix(dist_errors_df['gt_distribution_label'], dist_errors_df['predicted_label'], labels=valid_labels)
                    
                    fig_dist_err_cm = go.Figure(data=go.Heatmap(
                        z=dist_error_cm, 
                        x=[emotion_mapping[i] for i in range(8)], 
                        y=[emotion_mapping[i] for i in range(8)],
                        colorscale='Oranges', text=dist_error_cm, texttemplate='%{text}'
                    ))
                    fig_dist_err_cm.update_layout(xaxis_title="What the model guessed", yaxis_title="GT Distribution Peak", height=500)
                    st.plotly_chart(fig_dist_err_cm, use_container_width=True)
                
                with col_p2:
                    st.subheader("Where do the errors go?")
                    dist_error_grouped = dist_errors_df.groupby(['gt_distribution_emotion', 'predicted_emotion']).size().reset_index(name='count')
                    
                    fig_dist_err_bar = px.bar(
                        dist_error_grouped, x='gt_distribution_emotion', y='count', color='predicted_emotion',
                        title="Error Composition by Distribution Consensus",
                        labels={'gt_distribution_emotion': 'GT Dist. Emotion', 'count': 'Error Count', 'predicted_emotion': 'Incorrect Guess'},
                        barmode='stack'
                    )
                    fig_dist_err_bar.update_layout(height=500)
                    st.plotly_chart(fig_dist_err_bar, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Confidence Level: Correct vs Incorrect (vs Distribution)")
                
                dist_merged_df = merged_df.copy()
                if 'prediction_confidence' not in dist_merged_df.columns:
                    dist_merged_df['prediction_confidence'] = dist_merged_df[emotion_columns].max(axis=1)
                    
                dist_merged_df['is_dist_correct'] = dist_merged_df['gt_distribution_label'] == dist_merged_df['predicted_label']
                dist_merged_df['Dist_Status'] = dist_merged_df['is_dist_correct'].map({True: 'Correct (vs Dist)', False: 'Incorrect (vs Dist)'})
                
                fig_dist_conf_violin = px.violin(
                    dist_merged_df, y='prediction_confidence', x='Dist_Status', color='Dist_Status',
                    box=True, points="all",
                    color_discrete_map={'Correct (vs Dist)': 'mediumseagreen', 'Incorrect (vs Dist)': 'darkorange'},
                    labels={'prediction_confidence': 'Prediction Confidence (0 to 1)'}
                )
                fig_dist_conf_violin.update_layout(height=400)
                st.plotly_chart(fig_dist_conf_violin, use_container_width=True)
        with tab3:
            st.header("Probability Distribution Comparison")
            sample_options = filtered_df['file'].tolist()
            if sample_options:
                selected_sample = st.selectbox("Select a sample for detailed visualization:", sample_options[:50])
                if selected_sample:
                    sample_data = filtered_df[filtered_df['file'] == selected_sample].iloc[0]
                    
                    column_to_label = {0: 1, 1: 7, 2: 3, 3: 6, 4: 5, 5: 4, 6: 2, 7: 0}
                    pred_probs_ordered = [0] * 8
                    gt_probs_ordered = [0] * 8
                    
                    for i, col in enumerate(emotion_columns):
                        label = column_to_label[i]
                        pred_val = sample_data[col]
                        gt_val = sample_data[f"{col}_gt"]
                        pred_probs_ordered[label] = float(pred_val) if isinstance(pred_val, (int, float, np.number)) else 0.0
                        gt_probs_ordered[label] = float(gt_val) if isinstance(gt_val, (int, float, np.number)) else 0.0
                    
                    pred_probs_sample = np.nan_to_num(np.array(pred_probs_ordered, dtype=np.float64), nan=0.0)
                    gt_probs_sample = np.nan_to_num(np.array(gt_probs_ordered, dtype=np.float64), nan=0.0)
                    
                    pred_sum = np.sum(pred_probs_sample)
                    gt_sum = np.sum(gt_probs_sample)
                    
                    if pred_sum > 0: pred_probs_sample = pred_probs_sample / pred_sum
                    else: pred_probs_sample = np.ones_like(pred_probs_sample) / len(pred_probs_sample)
                    
                    if gt_sum > 0: gt_probs_sample = gt_probs_sample / gt_sum
                    else: gt_probs_sample = np.ones_like(gt_probs_sample) / len(gt_probs_sample)
                    
                    emotion_names_ordered = [emotion_mapping[i] for i in range(8)]
                    plot_data = pd.DataFrame({'Emotion': emotion_names_ordered, 'Predicted': pred_probs_sample, 'Ground Truth': gt_probs_sample})
                    
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Bar(x=plot_data['Emotion'], y=plot_data['Predicted'], name='Predicted', marker_color='blue', opacity=0.7))
                    fig_comparison.add_trace(go.Bar(x=plot_data['Emotion'], y=plot_data['Ground Truth'], name='Ground Truth', marker_color='red', opacity=0.7))
                    
                    pred_max_idx = np.argmax(pred_probs_sample)
                    gt_max_idx = np.argmax(gt_probs_sample)
                    
                    fig_comparison.add_annotation(x=emotion_names_ordered[pred_max_idx], y=pred_probs_sample[pred_max_idx], text="▲ Pred", showarrow=True, arrowhead=1, ax=0, ay=-40, font=dict(color="blue", size=12))
                    fig_comparison.add_annotation(x=emotion_names_ordered[gt_max_idx], y=gt_probs_sample[gt_max_idx], text="▲ GT", showarrow=True, arrowhead=1, ax=0, ay=-60, font=dict(color="red", size=12))
                    
                    fig_comparison.update_layout(title=f"Distribution Comparison - {selected_sample.split('/')[-1]}", xaxis_title="Emotion", yaxis_title="Probability", barmode='group', height=500)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    st.subheader("Emotion Source Comparison")
                    comparison_data = {
                        'Source': ['Annotation Label', 'Ground Truth Distribution', 'Predicted Distribution'],
                        'Emotion': [sample_data['groundtruth_emotion'], sample_data['gt_distribution_emotion'], sample_data['predicted_emotion']],
                        'Confidence/Probability': ['N/A', f"{sample_data['gt_max_probability']:.3f}", f"{np.max(pred_probs_sample):.3f}"],
                        'Details': [f"Label: {sample_data['groundtruth_label']}", f"Entropy: {sample_data['gt_distribution_entropy']:.3f}", f"JS Div: {sample_data['js_divergence']:.3f}"]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    consistency_notes = []
                    for i, row in comparison_df.iterrows():
                        if i == 0:
                            matches_dist = (sample_data['groundtruth_emotion'] == sample_data['gt_distribution_emotion'])
                            matches_pred = (sample_data['groundtruth_emotion'] == sample_data['predicted_emotion'])
                            if matches_dist and matches_pred: consistency_notes.append("✅ Consistent with both")
                            elif matches_dist: consistency_notes.append("✅ Matches GT distribution")
                            elif matches_pred: consistency_notes.append("✅ Matches prediction")
                            else: consistency_notes.append("⚠️ Differs from both")
                        elif i == 1:
                            matches_pred = (sample_data['gt_distribution_emotion'] == sample_data['predicted_emotion'])
                            if matches_pred: consistency_notes.append("✅ Matches prediction")
                            else: consistency_notes.append("⚠️ Differs from prediction")
                        else:
                            consistency_notes.append("")
                    
                    comparison_df['Consistency'] = consistency_notes
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Detailed Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Annotation Label", sample_data['groundtruth_emotion'], delta=f"Label: {sample_data['groundtruth_label']}")
                    with col2:
                        has_discrepancy = sample_data['annotation_distribution_discrepancy']
                        discrepancy_icon = "⚠️" if has_discrepancy else "✅"
                        st.metric(f"{discrepancy_icon} GT Distribution", sample_data['gt_distribution_emotion'], delta=f"Confidence: {sample_data['gt_max_probability']:.3f}")
                    with col3:
                        correct = sample_data['groundtruth_label'] == sample_data['predicted_label']
                        correct_icon = "✅" if correct else "❌"
                        st.metric(f"{correct_icon} Predicted", sample_data['predicted_emotion'], delta="Correct" if correct else "Incorrect")
                    
                    col4, col5, col6 = st.columns(3)
                    with col4: st.metric("JS Divergence", f"{sample_data['js_divergence']:.4f}")
                    with col5: st.metric("Cosine Similarity", f"{sample_data['cosine_similarity']:.4f}")
                    with col6: st.metric("KL Divergence", f"{sample_data['kl_divergence']:.4f}")
                    
                    col7, col8, col9 = st.columns(3)
                    with col7: st.metric("GT Entropy", f"{sample_data['gt_distribution_entropy']:.3f}")
                    with col8: st.metric("Euclidean Dist", f"{sample_data['euclidean_distance']:.3f}")
                    with col9: st.metric("Correlation", f"{sample_data['pearson_correlation']:.3f}")
                    
                    st.subheader(f"{metric_options[selected_metric]} vs Accuracy per Sample")
                    filtered_df['correct'] = filtered_df['groundtruth_label'] == filtered_df['predicted_label']
                    scatter_df = filtered_df.head(100).dropna(subset=[selected_metric, 'cosine_similarity']).copy()
                    
                    if len(scatter_df) > 0:
                        fig_scatter = px.scatter(scatter_df, x=selected_metric, y='cosine_similarity', color='correct', hover_data=['file', 'groundtruth_emotion', 'predicted_emotion'], title=f"{metric_options[selected_metric]} vs Cosine Similarity", labels={'correct': 'Correct Prediction'})
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("Insufficient data for scatter plot after removing NaN values.")
            else:
                st.warning("No samples available after applying filters.")   
        
        with tab4:
            st.header("Affective Dimensions Analysis (VAD)")
            if extra_columns:
                st.info(f"📊 Analyzing affective dimensions: {', '.join(extra_columns)}")
                vad_stats = []
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        valid_values = merged_df[f"{dimension}_gt"].dropna()
                        if len(valid_values) > 0:
                            vad_stats.append({'Dimension': dimension, 'Mean': valid_values.mean(), 'Std Dev': valid_values.std(), 'Min': valid_values.min(), 'Max': valid_values.max()})
                
                if vad_stats:
                    stats_df = pd.DataFrame(vad_stats)
                    st.dataframe(stats_df.style.format({'Mean': '{:.4f}', 'Std Dev': '{:.4f}', 'Min': '{:.4f}', 'Max': '{:.4f}'}), use_container_width=True)
                
                st.subheader("Distribution of Affective Dimensions by Emotion")
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        plot_df = merged_df.dropna(subset=[f'{dimension}_gt', 'groundtruth_emotion']).copy()
                        if len(plot_df) > 0:
                            fig = px.box(plot_df, x='groundtruth_emotion', y=f'{dimension}_gt', title=f'Distribution of {dimension} by Emotion', points='all', color='groundtruth_emotion')
                            fig.update_layout(xaxis_title="Emotion", yaxis_title=dimension, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                st.subheader("Correlation between Affective Dimensions and Model Performance")
                correlation_data = []
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        corr_df = merged_df[[f"{dimension}_gt", 'js_divergence', 'cosine_similarity']].dropna()
                        if len(corr_df) > 1:
                            correlation_data.append({
                                'Dimension': dimension, 'Correlation with JS Divergence': corr_df[f"{dimension}_gt"].corr(corr_df['js_divergence']),
                                'Correlation with Cosine Similarity': corr_df[f"{dimension}_gt"].corr(corr_df['cosine_similarity'])
                            })
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    st.dataframe(corr_df.style.format({'Correlation with JS Divergence': '{:.4f}', 'Correlation with Cosine Similarity': '{:.4f}'}), use_container_width=True)
                    fig_corr = px.bar(corr_df.melt(id_vars=['Dimension'], value_vars=['Correlation with JS Divergence', 'Correlation with Cosine Similarity']), x='Dimension', y='value', color='variable', barmode='group', title='Correlation between Affective Dimensions and Model Metrics', labels={'value': 'Correlation', 'variable': 'Metric'})
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("ℹ️ No affective dimensions (valence, arousal, dominance) found in ground truth file.")
        
        with tab5:
            st.header("🔍 Label-VAD Consistency Analysis")
            if extra_columns and len(extra_columns) >= 2:
                st.subheader("Theoretical VAD Expectations by Emotion")
                expectations_df = pd.DataFrame(vad_expectations).T
                st.dataframe(expectations_df, use_container_width=True)
                
                vad_means = []
                for emotion in emotion_mapping.values():
                    subset = merged_df[merged_df['groundtruth_emotion'] == emotion]
                    if len(subset) > 0:
                        means = {'Emotion': emotion}
                        for dim in extra_columns:
                            if f"{dim}_gt" in subset.columns: means[dim] = subset[f"{dim}_gt"].mean()
                        vad_means.append(means)
                
                if vad_means:
                    means_df = pd.DataFrame(vad_means)
                    st.subheader("1. Average VAD Values by Emotion (Observed)")
                    st.dataframe(means_df.style.format({dim: '{:.4f}' for dim in extra_columns}), use_container_width=True)
                    
                    fig_means = go.Figure()
                    for dim in extra_columns:
                        if dim in means_df.columns: fig_means.add_trace(go.Bar(x=means_df['Emotion'], y=means_df[dim], name=dim, text=means_df[dim].round(3), textposition='auto'))
                    fig_means.update_layout(title='Average VAD Values by Emotion', xaxis_title='Emotion', yaxis_title='Value', barmode='group', height=500)
                    st.plotly_chart(fig_means, use_container_width=True)
                
                st.subheader("2. Valence vs Arousal Scatter")
                scatter_df = merged_df.dropna(subset=['valence_gt', 'arousal_gt', 'groundtruth_emotion']).copy()
                if len(scatter_df) > 0:
                    fig_scatter_2d = px.scatter(scatter_df, x='valence_gt', y='arousal_gt', color='groundtruth_emotion', title='Valence vs Arousal by Emotion', labels={'valence_gt': 'Valence', 'arousal_gt': 'Arousal'}, hover_data=['file', 'groundtruth_emotion'])
                    fig_scatter_2d.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_scatter_2d.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_scatter_2d.update_layout(height=600)
                    st.plotly_chart(fig_scatter_2d, use_container_width=True)
                
                if 'vad_consistency_score' in merged_df.columns:
                    st.subheader("3. Label-VAD Consistency Score")
                    consistency_stats = merged_df['vad_consistency_score'].describe()
                    st.metric("Average Consistency", f"{consistency_stats['mean']:.2%}")
                    fig_consistency = px.histogram(merged_df, x='vad_consistency_score', title='Label-VAD Consistency Score Distribution', nbins=20)
                    st.plotly_chart(fig_consistency, use_container_width=True)
            else:
                st.info("ℹ️ At least valence and arousal are required in ground truth for consistency analysis.")
        
        with tab6:
            st.header("⚠️ Annotation-Distribution Discrepancy Analysis")
            st.markdown("""
            This analysis compares the **annotated emotion label** with the **most probable emotion from the distribution** in the ground truth.
            """)
            
            total_samples = len(merged_df)
            discrepancy_count = merged_df['annotation_distribution_discrepancy'].sum()
            discrepancy_percentage = discrepancy_count / total_samples * 100
            
            st.subheader("1. Overall Discrepancy Statistics")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Samples", total_samples)
            with col2: st.metric("Discrepancy Count", f"{discrepancy_count}")
            with col3: st.metric("Discrepancy Percentage", f"{discrepancy_percentage:.1f}%")
            
            st.subheader("2. Discrepancy by Emotion")
            discrepancy_by_emotion = []
            for emotion_idx, emotion_name in emotion_mapping.items():
                emotion_samples = merged_df[merged_df['groundtruth_label'] == emotion_idx]
                if len(emotion_samples) > 0:
                    discrepancy_rate = emotion_samples['annotation_distribution_discrepancy'].mean()
                    discrepancy_by_emotion.append({'Emotion': emotion_name, 'Total Samples': len(emotion_samples), 'Discrepancy Count': emotion_samples['annotation_distribution_discrepancy'].sum(), 'Discrepancy Rate': discrepancy_rate})
            
            if discrepancy_by_emotion:
                discrepancy_df = pd.DataFrame(discrepancy_by_emotion).sort_values('Discrepancy Rate', ascending=False)
                st.dataframe(discrepancy_df.style.format({'Discrepancy Rate': '{:.2%}'}), use_container_width=True)
                fig_discrepancy = px.bar(discrepancy_df, x='Emotion', y='Discrepancy Rate', title='Annotation-Distribution Discrepancy Rate by Emotion', color='Discrepancy Rate', color_continuous_scale='RdYlGn_r', text='Discrepancy Rate')
                fig_discrepancy.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                st.plotly_chart(fig_discrepancy, use_container_width=True)
            
            st.subheader("3. Confidence and Entropy Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Avg Max Probability", f"{merged_df['gt_max_probability'].mean():.3f}")
            with col2: st.metric("Avg Distribution Entropy", f"{merged_df['gt_distribution_entropy'].mean():.3f}")
            with col3: st.metric("Avg Confidence (Discrepant)", f"{merged_df[merged_df['annotation_distribution_discrepancy']]['gt_max_probability'].mean():.3f}")
            with col4: st.metric("Avg Entropy (Discrepant)", f"{merged_df[merged_df['annotation_distribution_discrepancy']]['gt_distribution_entropy'].mean():.3f}")
            
            st.subheader("4. Confusion Matrix: Annotation vs Distribution")
            cm_annotation_dist = confusion_matrix(merged_df['groundtruth_label'], merged_df['gt_distribution_label'], labels=valid_labels)
            fig_cm_annotation = go.Figure(data=go.Heatmap(z=cm_annotation_dist, x=[emotion_mapping[i] for i in range(8)], y=[emotion_mapping[i] for i in range(8)], colorscale='Blues', text=cm_annotation_dist, texttemplate='%{text}'))
            fig_cm_annotation.update_layout(title="Confusion Matrix: Annotated Label vs Distribution Label", height=600)
            st.plotly_chart(fig_cm_annotation, use_container_width=True)
            
            discrepancy_samples = merged_df[merged_df['annotation_distribution_discrepancy']].copy()
            if len(discrepancy_samples) > 0:
                st.subheader("5. Most Common Discrepancies")
                discrepancy_types = discrepancy_samples.groupby(['groundtruth_emotion', 'gt_distribution_emotion']).size().reset_index(name='count').sort_values('count', ascending=False).head(15)
                fig_discrepancy_types = px.bar(discrepancy_types, x='count', y='groundtruth_emotion', color='gt_distribution_emotion', orientation='h')
                st.plotly_chart(fig_discrepancy_types, use_container_width=True)

        with tab_clean:
            st.header("✨ Clean vs Discrepant Performance Analysis")
            st.markdown("""
            This tab isolates the model's performance on "clean" samples (where the manual annotation perfectly agrees 
            with the most probable emotion from the Ground Truth distribution) versus "discrepant" samples 
            (where there is divergence or ambiguity in the dataset).
            """)
            
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.metric("Overall Accuracy", f"{accuracy:.2%}", help="Accuracy across the entire dataset")
            with col_c2:
                st.metric("✨ Clean Accuracy", f"{clean_accuracy:.2%}", delta=f"{(clean_accuracy - accuracy):.2%}", help=f"Based on {len(non_discrepant_df)} clean samples")
            with col_c3:
                st.metric("⚠️ Discrepant Accuracy", f"{discrepant_accuracy:.2%}", delta=f"{(discrepant_accuracy - accuracy):.2%}", delta_color="inverse", help=f"Based on {len(discrepant_df)} discrepant samples")
                
            st.markdown("---")
            
            st.subheader("Accuracy by Emotion: Clean vs Discrepant")
            
            clean_emotions = []
            for em_idx, em_name in emotion_mapping.items():
                c_mask = non_discrepant_df['groundtruth_label'] == em_idx
                if c_mask.sum() > 0:
                    c_acc = accuracy_score(non_discrepant_df[c_mask]['groundtruth_label'], non_discrepant_df[c_mask]['predicted_label'])
                    clean_emotions.append({'Emotion': em_name, 'Accuracy': c_acc, 'Type': 'Clean (Match)', 'Count': c_mask.sum()})
                
                d_mask = discrepant_df['groundtruth_label'] == em_idx
                if d_mask.sum() > 0:
                    d_acc = accuracy_score(discrepant_df[d_mask]['groundtruth_label'], discrepant_df[d_mask]['predicted_label'])
                    clean_emotions.append({'Emotion': em_name, 'Accuracy': d_acc, 'Type': 'Discrepant (Mismatch)', 'Count': d_mask.sum()})
                    
            if clean_emotions:
                comp_df = pd.DataFrame(clean_emotions)
                fig_comp = px.bar(comp_df, x='Emotion', y='Accuracy', color='Type', barmode='group',
                                  hover_data=['Count'], title="Performance Comparison per Emotion",
                                  color_discrete_map={'Clean (Match)': 'mediumseagreen', 'Discrepant (Mismatch)': 'indianred'})
                fig_comp.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_comp, use_container_width=True)
                
            st.subheader("Confusion Matrices Isolation")
            col_cm1, col_cm2 = st.columns(2)
            
            with col_cm1:
                st.markdown("**Clean Data Confusion Matrix**")
                if len(non_discrepant_df) > 0:
                    cm_clean = confusion_matrix(non_discrepant_df['groundtruth_label'], non_discrepant_df['predicted_label'], labels=valid_labels)
                    fig_cm_clean = go.Figure(data=go.Heatmap(z=cm_clean, x=[emotion_mapping[i] for i in range(8)], y=[emotion_mapping[i] for i in range(8)], colorscale='Greens', text=cm_clean, texttemplate='%{text}'))
                    st.plotly_chart(fig_cm_clean, use_container_width=True)
                    
            with col_cm2:
                st.markdown("**Discrepant Data Confusion Matrix**")
                if len(discrepant_df) > 0:
                    cm_disc = confusion_matrix(discrepant_df['groundtruth_label'], discrepant_df['predicted_label'], labels=valid_labels)
                    fig_cm_disc = go.Figure(data=go.Heatmap(z=cm_disc, x=[emotion_mapping[i] for i in range(8)], y=[emotion_mapping[i] for i in range(8)], colorscale='Reds', text=cm_disc, texttemplate='%{text}'))
                    st.plotly_chart(fig_cm_disc, use_container_width=True)

        with tab7:
            st.header("📊 Distribution Metrics Analysis")
            if merged_df.columns.duplicated().any():
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            
            distribution_metrics_cols = ['js_divergence', 'kl_divergence', 'euclidean_distance', 'cosine_similarity', 'pearson_correlation']
            available_metrics = [col for col in distribution_metrics_cols if col in merged_df.columns]
            
            if not available_metrics:
                st.warning("No distribution Metrics found in the data.")
            else:
                stats_df = merged_df[available_metrics].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                metric_names = {'js_divergence': 'JS Divergence', 'kl_divergence': 'KL Divergence', 'euclidean_distance': 'Euclidean Distance', 'cosine_similarity': 'Cosine Similarity', 'pearson_correlation': 'Pearson Correlation'}
                stats_df.index = [metric_names.get(col, col) for col in stats_df.index]
                
                st.subheader("1. Descriptive Statistics")
                st.dataframe(stats_df.style.format({'mean': '{:.4f}', 'std': '{:.4f}', 'min': '{:.4f}', '25%': '{:.4f}', '50%': '{:.4f}', '75%': '{:.4f}', 'max': '{:.4f}'}), use_container_width=True)
                
                st.subheader("2. Distribution of Metrics")
                metric_to_analyze = st.selectbox("Select metric for detailed analysis:", available_metrics, format_func=lambda x: metric_names.get(x, x))
                metric_data = merged_df[metric_to_analyze].dropna()
                
                if len(metric_data) > 0:
                    fig_dist = make_subplots(rows=2, cols=2, subplot_titles=['Histogram', 'Cumulative Distribution', 'Box Plot by Emotion', 'Violin Plot by Emotion'])
                    fig_dist.add_trace(go.Histogram(x=metric_data, nbinsx=50, name='Histogram', marker_color='lightblue'), row=1, col=1)
                    
                    sorted_vals = np.sort(metric_data)
                    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                    fig_dist.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines', name='CDF', line=dict(color='darkblue', width=2)), row=1, col=2)
                    
                    for emotion in emotion_mapping.values():
                        emotion_data = merged_df[merged_df['groundtruth_emotion'] == emotion][metric_to_analyze].dropna()
                        if len(emotion_data) > 0:
                            fig_dist.add_trace(go.Box(y=emotion_data, name=emotion, boxpoints='outliers'), row=2, col=1)
                            fig_dist.add_trace(go.Violin(y=emotion_data, name=emotion, box_visible=True, meanline_visible=True), row=2, col=2)
                    
                    fig_dist.update_layout(height=800, showlegend=False, title_text=f"Detailed Analysis: {metric_names.get(metric_to_analyze, metric_to_analyze)}")
                    st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab8:
            st.header("Complete Data")
            view_option = st.radio("Select view:", ["Combined Data", "Incorrect Samples Only", "Correct Samples Only", "Discrepant Samples Only"])
            
            if view_option == "Incorrect Samples Only": display_df = merged_df[merged_df['groundtruth_label'] != merged_df['predicted_label']]
            elif view_option == "Correct Samples Only": display_df = merged_df[merged_df['groundtruth_label'] == merged_df['predicted_label']]
            elif view_option == "Discrepant Samples Only": display_df = merged_df[merged_df['annotation_distribution_discrepancy']]
            else: display_df = merged_df
            
            display_columns = ['file', 'groundtruth_emotion', 'predicted_emotion', 'js_divergence', 'cosine_similarity', 'pearson_correlation']
            
            if not display_df.empty:
                st.dataframe(display_df[display_columns].head(100), use_container_width=True)
                csv = display_df[display_columns].to_csv(index=False)
                st.download_button("📥 Download Filtered Data", data=csv, file_name="filtered_data.csv", mime="text/csv")
            else:
                st.info("No data to display.")

        with tab9:
            st.header("📤 Export Center")
            st.info("Export high-quality vector graphics for publication and presentations")
            
            format_map = {"PDF (Recommended)": "pdf", "SVG": "svg", "EPS": "eps", "PNG (High-Res)": "png"}
            selected_format = format_map[export_format]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.subheader("📊 Current Sample Charts")
                export_sample_options = merged_df['file'].tolist()
                selected_export_sample = st.selectbox("Select sample for export:", export_sample_options, key="export_sample")
                
                if selected_export_sample:
                    export_sample_data = merged_df[merged_df['file'] == selected_export_sample].iloc[0]
                    sample_idx = merged_df[merged_df['file'] == selected_export_sample].index[0]
                    
                    column_to_label = {0: 1, 1: 7, 2: 3, 3: 6, 4: 5, 5: 4, 6: 2, 7: 0}
                    pred_probs_ordered, gt_probs_ordered = [0]*8, [0]*8
                    
                    for i, col in enumerate(emotion_columns):
                        label = column_to_label[i]
                        pred_probs_ordered[label] = export_sample_data[col]
                        gt_probs_ordered[label] = export_sample_data[f"{col}_gt"]
                    
                    pred_probs_ordered = np.array(pred_probs_ordered, dtype=np.float64)
                    gt_probs_ordered = np.array(gt_probs_ordered, dtype=np.float64)
                    
                    pred_sum, gt_sum = np.sum(pred_probs_ordered), np.sum(gt_probs_ordered)
                    if pred_sum > 0: pred_probs_ordered = pred_probs_ordered / pred_sum
                    if gt_sum > 0: gt_probs_ordered = gt_probs_ordered / gt_sum
                    
                    emotion_names_ordered = [emotion_mapping[i] for i in range(8)]
                    font_sizes = {'title': bar_title_size, 'axes_labels': bar_axes_size, 'tick_labels': bar_tick_size, 'legend': bar_legend_size, 'bar_labels': bar_label_size, 'radial_labels': other_labels_size}
                    
                    st.markdown("**Distribution Comparison Chart**")
                    matplotlib_fig = create_comparison_bar_chart_matplotlib(pred_probs_ordered, gt_probs_ordered, emotion_names_ordered, filename=selected_export_sample.split('/')[-1], sample_idx=sample_idx, font_sizes=font_sizes)
                    export_buffer, mime_type = create_vector_figure_matplotlib(matplotlib_fig, selected_format)
                    st.download_button(label=f"📥 Download Comparison Chart ({selected_format.upper()})", data=export_buffer, file_name=f"comparison_sample_{sample_idx}_{timestamp}.{selected_format}", mime=mime_type)
                    plt.close(matplotlib_fig)
            
            with col_export2:
                st.subheader("🌍 Global Analysis Charts")
                st.markdown("**Confusion Matrix**")
                cm_font_sizes = {'title': other_title_size, 'axes_labels': other_labels_size, 'tick_labels': bar_tick_size, 'cell_values': bar_label_size}
                cm_fig = create_confusion_matrix_matplotlib(cm, [emotion_mapping[i] for i in range(8)], font_sizes=cm_font_sizes)
                cm_buffer, mime_type_cm = create_vector_figure_matplotlib(cm_fig, selected_format)
                st.download_button(label=f"📥 Download Confusion Matrix ({selected_format.upper()})", data=cm_buffer, file_name=f"confusion_matrix_{timestamp}.{selected_format}", mime=mime_type_cm)
                plt.close(cm_fig)
                
            st.subheader("📁 Complete Dataset Export")
            col_data1, col_data2, col_data3 = st.columns(3)
            with col_data1:
                combined_data = merged_df[['file', 'groundtruth_emotion', 'predicted_emotion', 'js_divergence']].copy()
                csv_data = combined_data.to_csv(index=False)
                st.download_button(label="📊 Download Analysis Results (CSV)", data=csv_data, file_name=f"analysis_results_{timestamp}.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
elif predictions_file is not None and groundtruth_file is None:
    st.warning("⚠️ Please upload the Ground Truth file")
elif predictions_file is None and groundtruth_file is not None:
    st.warning("⚠️ Please upload the Predictions file")
else:
    st.info("👈 Please upload both CSV files to begin analysis")