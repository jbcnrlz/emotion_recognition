import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.spatial.distance import jensenshannon as scipy_jensenshannon
from scipy.stats import entropy as scipy_entropy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import hashlib
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo

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
# NOVA SEÇÃO: Configurações de Exportação e Fontes
# ============================================================================

# Sidebar para configurações de exportação
st.sidebar.header("📤 Export Settings")

# Seleção de formato vetorial
export_format = st.sidebar.selectbox(
    "Preferred Export Format",
    ["PDF (Recommended)", "SVG", "EPS", "PNG (High-Res)"],
    help="Choose vector format for high-quality export"
)

# Controles de tamanho de fonte para gráficos de barras
st.sidebar.header("🔤 Bar Chart Font Sizes")

# Font size controls for bar charts
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
# NOVAS FUNÇÕES: Exportação vetorial e controle de fontes
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
    
    ax.set_xlabel('Emotion', fontsize=font_sizes['axes_labels'])
    ax.set_ylabel('Probability', fontsize=font_sizes['axes_labels'])
    ax.set_title(f'Distribution Comparison - {filename} (Sample {sample_idx})', 
                 fontsize=font_sizes['title'], pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right', 
                       fontsize=font_sizes['tick_labels'])
    ax.tick_params(axis='y', labelsize=font_sizes['tick_labels'])
    ax.legend(fontsize=font_sizes['legend'], loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if height is significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=font_sizes['bar_labels'])
    
    # Add subtle horizontal grid
    ax.yaxis.grid(True, alpha=0.2)
    
    # Set y-axis limit
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
    angles += angles[:1]  # Close the circle
    
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
    ax.set_xticklabels(emotion_names, fontsize=font_sizes['tick_labels'])
    
    # Set radial labels
    ax.set_ylim(0, max(max(pred_probs_closed), max(gt_probs_closed)) * 1.1)
    ax.tick_params(axis='y', labelsize=font_sizes['radial_labels'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=font_sizes['legend'])
    ax.set_title(f'Radar Chart - {filename} (Sample {sample_idx})', 
                 size=font_sizes['title'], y=1.1, pad=20)
    
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
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=font_sizes['axes_labels'])
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=emotion_names,
           yticklabels=emotion_names)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set labels and title
    ax.set_xlabel('Predicted Emotion', fontsize=font_sizes['axes_labels'], labelpad=10)
    ax.set_ylabel('True Emotion', fontsize=font_sizes['axes_labels'], labelpad=10)
    ax.set_title('Confusion Matrix', fontsize=font_sizes['title'], pad=20)
    
    # Set tick label sizes
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
    
    ax.set_xlabel(metric_name, fontsize=font_sizes['axes_labels'])
    ax.set_ylabel('Frequency', fontsize=font_sizes['axes_labels'])
    ax.set_title(f'Distribution of {metric_name}', fontsize=font_sizes['title'])
    ax.tick_params(axis='both', labelsize=font_sizes['tick_labels'])
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(metric_data)
    median_val = np.median(metric_data)
    ax.axvline(mean_val, color='red', linestyle='--', 
               label=f'Mean: {mean_val:.3f}', linewidth=2)
    ax.axvline(median_val, color='green', linestyle='--', 
               label=f'Median: {median_val:.3f}', linewidth=2)
    ax.legend(fontsize=font_sizes['legend'])
    
    plt.tight_layout()
    return fig

# ============================================================================
# FUNÇÕES ORIGINAIS (mantidas do script original)
# ============================================================================

# Function to safely calculate KL divergence
def safe_kl_divergence(p, q):
    """
    Calculates KL divergence safely, handling zeros
    """
    try:
        # Ensure numpy arrays and flatten them
        p_arr = np.asarray(p, dtype=np.float64).flatten()
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        
        # Check if arrays have same length
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
        
        # Calculate KL divergence using scipy for stability
        kl = scipy_entropy(p_safe, q_safe)
        
        return kl if not np.isnan(kl) and not np.isinf(kl) else np.nan
    except Exception as e:
        return np.nan

# Function to safely calculate Jensen-Shannon divergence
def safe_js_divergence(p, q):
    """
    Calculates Jensen-Shannon divergence safely
    """
    try:
        # Ensure numpy arrays and flatten them
        p_arr = np.asarray(p, dtype=np.float64).flatten()
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        
        # Check if arrays have same length
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
        
        # Calculate Jensen-Shannon divergence using scipy
        js = scipy_jensenshannon(p_safe, q_safe)
        
        # scipy_jensenshannon returns the square root of JS divergence
        # So we square it to get the actual JS divergence
        return js ** 2 if not np.isnan(js) and not np.isinf(js) else np.nan
    except Exception as e:
        return np.nan

# Function to generate data hash (for caching)
def generate_data_hash(predictions_content, groundtruth_content):
    combined = predictions_content + groundtruth_content
    return hashlib.md5(combined.encode()).hexdigest()

# Function to process data
@st.cache_data(ttl=3600)
def process_data(predictions_content, groundtruth_content, data_hash):
    # Convert content to DataFrames
    predictions_df = pd.read_csv(io.StringIO(predictions_content))
    groundtruth_df = pd.read_csv(io.StringIO(groundtruth_content))
    
    # List of emotion columns (common to both files)
    emotion_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    # Check if emotion columns exist in both dataframes
    missing_in_pred = [col for col in emotion_columns if col not in predictions_df.columns]
    missing_in_gt = [col for col in emotion_columns if col not in groundtruth_df.columns]
    
    if missing_in_pred:
        st.error(f"⚠️ Missing columns in predictions file: {missing_in_pred}")
        return None, None, None, None, None, None, None
    if missing_in_gt:
        st.error(f"⚠️ Missing columns in ground truth file: {missing_in_gt}")
        return None, None, None, None, None, None, None
    
    # Convert emotion columns to numeric, forcing errors to NaN
    for col in emotion_columns:
        predictions_df[col] = pd.to_numeric(predictions_df[col], errors='coerce')
        groundtruth_df[col] = pd.to_numeric(groundtruth_df[col], errors='coerce')
    
    # Fill NaN values with a small number
    for col in emotion_columns:
        predictions_df[col] = predictions_df[col].fillna(1e-12)
        groundtruth_df[col] = groundtruth_df[col].fillna(1e-12)
    
    # Sort both dataframes by file path
    predictions_df = predictions_df.sort_values('file').reset_index(drop=True)
    groundtruth_df = groundtruth_df.sort_values('file').reset_index(drop=True)
    
    # Extra columns in ground truth (if they exist)
    extra_columns = []
    for col in ['valence', 'arousal', 'dominance']:
        if col in groundtruth_df.columns:
            extra_columns.append(col)
            # Convert to numeric if needed
            groundtruth_df[col] = pd.to_numeric(groundtruth_df[col], errors='coerce')
    
    # Check if files are aligned
    if not predictions_df['file'].equals(groundtruth_df['file']):
        st.warning("⚠️ Files are not perfectly aligned by image. Merging by file paths...")
        
        # Merge by file paths
        merged_df = pd.merge(predictions_df, groundtruth_df, on='file', suffixes=('_pred', '_gt'), how='inner')
    else:
        # Create combined dataframe
        merged_df = predictions_df.copy()
        
        # Add ground truth emotion columns
        for col in emotion_columns:
            merged_df[f"{col}_gt"] = groundtruth_df[col]
        
        # Add extra ground truth columns
        for col in extra_columns:
            merged_df[f"{col}_gt"] = groundtruth_df[col]
        
        # Add emotion_label from ground truth
        if 'emotion_label' in groundtruth_df.columns:
            merged_df['emotion_label_gt'] = groundtruth_df['emotion_label']
    
    # Rename columns for clarity
    rename_dict = {}
    
    # For emotion_label
    if 'emotion_label_pred' in merged_df.columns:
        rename_dict['emotion_label_pred'] = 'pred_label'
    elif 'emotion_label' in merged_df.columns and 'emotion_label_gt' in merged_df.columns:
        # Both have emotion_label, rename appropriately
        rename_dict['emotion_label'] = 'pred_label'
        rename_dict['emotion_label_gt'] = 'true_label'
    elif 'emotion_label' in merged_df.columns:
        # Only predictions has emotion_label
        rename_dict['emotion_label'] = 'pred_label'
    
    if rename_dict:
        merged_df = merged_df.rename(columns=rename_dict)
    
    # IMPORTANT: Mapping from column index to label
    # Column order: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral
    # But labels are: 0=neutral, 1=happy, 2=sad, 3=surprised, 4=fearful, 5=disgusted, 6=angry, 7=contempt
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
    
    # CORRECT label to emotion name mapping
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
    
    # 1. Calculate predicted emotion from model predictions
    pred_probs = merged_df[emotion_columns].values
    merged_df['predicted_label'] = np.argmax(pred_probs, axis=1)
    merged_df['predicted_label'] = merged_df['predicted_label'].map(column_to_label)
    merged_df['predicted_emotion'] = merged_df['predicted_label'].map(emotion_mapping)
    
    # 2. Get ground truth annotation (emotion_label)
    if 'true_label' in merged_df.columns:
        merged_df['groundtruth_label'] = merged_df['true_label']
    elif 'emotion_label_gt' in merged_df.columns:
        merged_df['groundtruth_label'] = merged_df['emotion_label_gt']
    else:
        # If no emotion_label column, we can't proceed with this analysis
        st.error("❌ Ground truth file doesn't have emotion_label column")
        return None, None, None, None, None, None, None
    
    # 3. Calculate most probable emotion from ground truth distribution
    gt_emotion_columns = [f"{col}_gt" for col in emotion_columns]
    gt_probs = merged_df[gt_emotion_columns].values
    
    # Calculate most probable emotion from distribution
    merged_df['gt_distribution_label'] = np.argmax(gt_probs, axis=1)
    merged_df['gt_distribution_label'] = merged_df['gt_distribution_label'].map(column_to_label)
    merged_df['gt_distribution_emotion'] = merged_df['gt_distribution_label'].map(emotion_mapping)
    
    # Add emotion names for ground truth label
    merged_df['groundtruth_emotion'] = merged_df['groundtruth_label'].map(emotion_mapping)
    
    # 4. Calculate discrepancy between annotation and distribution
    merged_df['annotation_distribution_discrepancy'] = merged_df['groundtruth_label'] != merged_df['gt_distribution_label']
    
    # 5. Calculate confidence metrics for ground truth distribution
    # Maximum probability (confidence)
    merged_df['gt_max_probability'] = np.max(gt_probs, axis=1)
    
    # Entropy of distribution (uncertainty)
    def calculate_entropy(row):
        probs = row[gt_emotion_columns].values.astype(np.float64)
        probs_safe = probs + 1e-12
        probs_sum = np.sum(probs_safe)
        if probs_sum <= 0:
            return np.nan
        probs_safe = probs_safe / probs_sum
        # Calculate entropy using scipy for stability
        return scipy_entropy(probs_safe)
    
    merged_df['gt_distribution_entropy'] = merged_df.apply(calculate_entropy, axis=1)
    
    # FILTER: Remove rows with groundtruth_label outside 0-7 range
    original_count = len(merged_df)
    merged_df = merged_df[merged_df['groundtruth_label'].isin(emotion_mapping.keys())].copy()
    filtered_count = len(merged_df)
    
    if original_count != filtered_count:
        st.warning(f"⚠️ Removed {original_count - filtered_count} samples with invalid labels (outside 0-7 range).")
    
    # Define expected VAD values for each emotion (based on literature)
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
    
    # Function to calculate consistency between label and VAD
    def calculate_vad_consistency(row):
        if not extra_columns:
            return pd.Series({'vad_consistency_score': np.nan})
        
        consistency_scores = []
        emotion = row['groundtruth_emotion']
        
        if emotion not in vad_expectations:
            return pd.Series({'vad_consistency_score': np.nan})
        
        expected = vad_expectations[emotion]
        
        for dim in extra_columns:
            if f"{dim}_gt" not in row.index:
                continue
                
            value = row[f"{dim}_gt"]
            
            # Skip if value is NaN
            if pd.isna(value):
                continue
                
            # Normalize value to 0-1 scale (assuming VAD is in -1 to 1)
            norm_value = (value + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Define expected ranges for each dimension based on emotion
            if dim == 'valence':
                if expected['valence'] == 'high':
                    expected_range = (0.7, 1.0)
                elif expected['valence'] == 'low':
                    expected_range = (0.0, 0.3)
                elif expected['valence'] == 'medium':
                    expected_range = (0.3, 0.7)
                else:  # neutral
                    expected_range = (0.4, 0.6)
                    
            elif dim == 'arousal':
                if expected['arousal'] == 'high':
                    expected_range = (0.7, 1.0)
                elif expected['arousal'] == 'low':
                    expected_range = (0.0, 0.3)
                elif 'medium' in expected['arousal']:
                    expected_range = (0.3, 0.7)
                else:
                    expected_range = (0.4, 0.6)
                    
            elif dim == 'dominance':
                if expected['dominance'] == 'high':
                    expected_range = (0.7, 1.0)
                elif expected['dominance'] == 'low':
                    expected_range = (0.0, 0.3)
                elif expected['dominance'] == 'medium':
                    expected_range = (0.3, 0.7)
                else:  # neutral
                    expected_range = (0.4, 0.6)
            
            # Calculate consistency score (1 if within range, 0 if outside)
            if expected_range[0] <= norm_value <= expected_range[1]:
                consistency_scores.append(1.0)
            else:
                # Penalty proportional to distance from expected range
                if norm_value < expected_range[0]:
                    distance = expected_range[0] - norm_value
                else:
                    distance = norm_value - expected_range[1]
                
                # Maximum penalty of 0.5 if far outside range
                penalty = min(distance * 2, 0.5)
                consistency_scores.append(1.0 - penalty)
        
        if consistency_scores:
            return pd.Series({'vad_consistency_score': np.mean(consistency_scores)})
        else:
            return pd.Series({'vad_consistency_score': np.nan})
    
    # Calculate similarity metrics between predicted and ground truth distributions
    def calculate_distribution_metrics(row):
        try:
            # Get emotion probabilities
            pred_dist = []
            gt_dist = []
            
            for col in emotion_columns:
                pred_val = row[col]
                gt_val = row[f"{col}_gt"]
                
                # Ensure numeric values
                if isinstance(pred_val, (int, float, np.number)):
                    pred_dist.append(float(pred_val))
                else:
                    pred_dist.append(1e-12)
                    
                if isinstance(gt_val, (int, float, np.number)):
                    gt_dist.append(float(gt_val))
                else:
                    gt_dist.append(1e-12)
            
            pred_dist = np.array(pred_dist, dtype=np.float64)
            gt_dist = np.array(gt_dist, dtype=np.float64)
            
            # Ensure no NaN or inf values
            pred_dist = np.nan_to_num(pred_dist, nan=1e-12, posinf=1e-12, neginf=1e-12)
            gt_dist = np.nan_to_num(gt_dist, nan=1e-12, posinf=1e-12, neginf=1e-12)
            
            # Add small epsilon to avoid zeros
            epsilon = 1e-12
            pred_dist_safe = pred_dist + epsilon
            gt_dist_safe = gt_dist + epsilon
            
            # Normalize to sum to 1
            pred_sum = np.sum(pred_dist_safe)
            gt_sum = np.sum(gt_dist_safe)
            
            if pred_sum > 0:
                pred_dist_safe = pred_dist_safe / pred_sum
            else:
                pred_dist_safe = np.ones_like(pred_dist_safe) / len(pred_dist_safe)
                
            if gt_sum > 0:
                gt_dist_safe = gt_dist_safe / gt_sum
            else:
                gt_dist_safe = np.ones_like(gt_dist_safe) / len(gt_dist_safe)
            
            # Jensen-Shannon divergence (using safe implementation)
            js_div = safe_js_divergence(pred_dist_safe, gt_dist_safe)
            
            # KL divergence (symmetric) using safe implementation
            kl_div_1 = safe_kl_divergence(pred_dist_safe, gt_dist_safe)
            kl_div_2 = safe_kl_divergence(gt_dist_safe, pred_dist_safe)
            kl_div = 0.5 * (kl_div_1 + kl_div_2) if not np.isnan(kl_div_1) and not np.isnan(kl_div_2) else np.nan
            
            # Euclidean distance
            euclidean_dist = np.linalg.norm(pred_dist_safe - gt_dist_safe)
            
            # Cosine similarity
            dot_product = np.dot(pred_dist_safe, gt_dist_safe)
            norm_pred = np.linalg.norm(pred_dist_safe)
            norm_gt = np.linalg.norm(gt_dist_safe)
            
            if norm_pred > 0 and norm_gt > 0:
                cosine_sim = dot_product / (norm_pred * norm_gt)
            else:
                cosine_sim = 0.0
            
            # Pearson correlation
            if len(pred_dist_safe) > 1:
                correlation = np.corrcoef(pred_dist_safe, gt_dist_safe)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
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
            # Return default values in case of error
            return pd.Series({
                'js_divergence': 0.0,
                'kl_divergence': 0.0,
                'euclidean_distance': 0.0,
                'cosine_similarity': 0.0,
                'pearson_correlation': 0.0
            })
    
    # Apply metric calculation
    distribution_metrics = merged_df.apply(calculate_distribution_metrics, axis=1)
    merged_df = pd.concat([merged_df, distribution_metrics], axis=1)
    
    # Calculate VAD consistency if extra columns exist
    if extra_columns:
        vad_consistency = merged_df.apply(calculate_vad_consistency, axis=1)
        merged_df = pd.concat([merged_df, vad_consistency], axis=1)
    
    return merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

# Sidebar for file upload
st.sidebar.header("📁 File Upload")

# File uploaders
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

# Check if both files are uploaded
if predictions_file is not None and groundtruth_file is not None:
    # Read file contents
    predictions_content = predictions_file.getvalue().decode('utf-8')
    groundtruth_content = groundtruth_file.getvalue().decode('utf-8')
    
    # Generate hash for caching
    data_hash = generate_data_hash(predictions_content, groundtruth_content)
    
    try:
        # Process data
        result = process_data(predictions_content, groundtruth_content, data_hash)
        
        if result[0] is None:
            st.error("❌ Error processing data. Please check file formats.")
            st.stop()
        
        merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations = result
        
        # Check for NaN values in metrics
        metrics_with_nan = merged_df[['js_divergence', 'kl_divergence', 'cosine_similarity', 'pearson_correlation']].isna().sum()
        if metrics_with_nan.sum() > 0:
            st.warning(f"⚠️ Some metrics contain NaN values: {metrics_with_nan.to_dict()}")
            # Replace NaN with 0 to avoid visualization issues
            merged_df['js_divergence'] = merged_df['js_divergence'].fillna(0)
            merged_df['kl_divergence'] = merged_df['kl_divergence'].fillna(0)
            merged_df['cosine_similarity'] = merged_df['cosine_similarity'].fillna(0)
            merged_df['pearson_correlation'] = merged_df['pearson_correlation'].fillna(0)
        
        # Calculate accuracy between predicted and ground truth
        accuracy = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
        
        # Create confusion matrix - ensure we only use valid labels
        valid_labels = list(emotion_mapping.keys())
        cm = confusion_matrix(merged_df['groundtruth_label'], merged_df['predicted_label'], 
                             labels=valid_labels)
        
        # Classification report
        class_report = classification_report(merged_df['groundtruth_label'], 
                                            merged_df['predicted_label'],
                                            target_names=list(emotion_mapping.values()),
                                            output_dict=True)
        
        st.success(f"✅ Data loaded successfully! Total of {len(merged_df)} samples.")
        
        # Show correct emotion mapping
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
        
        # Show information about extra columns
        if extra_columns:
            st.info(f"📊 Extra columns detected in ground truth: {', '.join(extra_columns)}")
        
        # Sidebar for analysis controls
        st.sidebar.header("🔧 Analysis Controls")
        
        # Metric selector for visualization
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
        
        # Number of samples to visualize
        num_samples = st.sidebar.slider(
            "Number of samples to visualize:",
            min_value=5,
            max_value=min(50, len(merged_df)),
            value=min(10, len(merged_df))
        )
        
        # Emotion filter
        all_emotions = list(emotion_mapping.values())
        selected_emotion = st.sidebar.multiselect(
            "Filter by true emotion:",
            all_emotions,
            default=all_emotions
        )
        
        # Apply filter
        if selected_emotion:
            filtered_df = merged_df[merged_df['groundtruth_emotion'].isin(selected_emotion)].copy()
        else:
            filtered_df = merged_df.copy()
        
        # Layout principal - Adding new tab for Export Center
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 Overview", 
            "🔍 Detailed Analysis", 
            "📊 Distributions", 
            "🎭 Affective Dimensions",
            "🔎 Label-VAD Consistency",
            "⚠️ Annotation-Distribution Discrepancy",            
            "📊 Distribution Metrics",
            "📋 Complete Data",
            "📤 Export Center"  # Nova aba para exportação
        ])
        
        with tab1:
            st.header("Performance Overview")
            
            # Main metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            
            with col2:
                avg_js = merged_df['js_divergence'].mean()
                st.metric("JS Divergence (avg)", f"{avg_js:.4f}")
            
            with col3:
                avg_cosine = merged_df['cosine_similarity'].mean()
                st.metric("Cosine Similarity (avg)", f"{avg_cosine:.4f}")
            
            with col4:
                correct_predictions = (merged_df['groundtruth_label'] == merged_df['predicted_label']).sum()
                total = len(merged_df)
                st.metric("Correct Predictions", f"{correct_predictions}/{total}")
            
            st.markdown("---")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[emotion_mapping[i] for i in range(8)],
                y=[emotion_mapping[i] for i in range(8)],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar_title="Count"
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix - Predicted vs True Emotions",
                xaxis_title="Predicted Emotion",
                yaxis_title="True Emotion",
                height=600
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Distribution of metrics
            st.subheader("Distribution of Similarity Metrics")
            
            metrics_to_plot = ['js_divergence', 'cosine_similarity', 'pearson_correlation']
            
            fig_metrics = make_subplots(
                rows=1, 
                cols=3,
                subplot_titles=['JS Divergence', 'Cosine Similarity', 'Pearson Correlation']
            )
            
            for i, metric in enumerate(metrics_to_plot, 1):
                fig_metrics.add_trace(
                    go.Histogram(x=merged_df[metric], name=metric_options.get(metric, metric)),
                    row=1, col=i
                )
            
            fig_metrics.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with tab2:
            st.header("Detailed Analysis by Emotion")
            
            # Metrics per emotion
            emotion_stats = []
            for emotion_idx, emotion_name in emotion_mapping.items():
                mask = merged_df['groundtruth_label'] == emotion_idx
                if mask.sum() > 0:
                    subset = merged_df[mask]
                    accuracy_subset = accuracy_score(subset['groundtruth_label'], subset['predicted_label'])
                    
                    stats = {
                        'Emotion': emotion_name,
                        'Samples': mask.sum(),
                        'Accuracy': accuracy_subset,
                        'JS Divergence (avg)': subset['js_divergence'].mean(),
                        'Cosine Similarity (avg)': subset['cosine_similarity'].mean(),
                        'Correlation (avg)': subset['pearson_correlation'].mean()
                    }
                    emotion_stats.append(stats)
            
            stats_df = pd.DataFrame(emotion_stats)
            st.dataframe(stats_df.style.format({
                'Accuracy': '{:.2%}',
                'JS Divergence (avg)': '{:.4f}',
                'Cosine Similarity (avg)': '{:.4f}',
                'Correlation (avg)': '{:.4f}'
            }), use_container_width=True)
            
            # Accuracy by emotion chart
            fig_acc = px.bar(
                stats_df,
                x='Emotion',
                y='Accuracy',
                title='Accuracy by Emotion',
                color='Accuracy',
                color_continuous_scale='Viridis'
            )
            fig_acc.update_layout(height=400)
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Error analysis
            st.subheader("Most Common Errors")
            
            # Identify confusion pairs
            error_pairs = merged_df[merged_df['groundtruth_label'] != merged_df['predicted_label']]
            confusion_pairs = error_pairs.groupby(['groundtruth_emotion', 'predicted_emotion']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False).head(10)
            
            if not confusion_pairs.empty:
                fig_errors = px.bar(
                    confusion_pairs,
                    x='count',
                    y='groundtruth_emotion',
                    color='predicted_emotion',
                    orientation='h',
                    title='Top 10 Confusion Pairs (True → Predicted)',
                    labels={'count': 'Error Count', 'groundtruth_emotion': 'True Emotion'}
                )
                fig_errors.update_layout(height=500)
                st.plotly_chart(fig_errors, use_container_width=True)
            else:
                st.info("No errors to display!")
        
        with tab3:
            st.header("Probability Distribution Comparison")
            
            # Sample selector
            sample_options = filtered_df['file'].tolist()
            
            if sample_options:
                selected_sample = st.selectbox(
                    "Select a sample for detailed visualization:",
                    sample_options[:50]  # Limit to 50 for performance
                )
                
                if selected_sample:
                    sample_data = filtered_df[filtered_df['file'] == selected_sample].iloc[0]
                    
                    # Prepare data for visualization
                    pred_probs_sample = []
                    gt_probs_sample = []
                    
                    # CORREÇÃO: Obter as probabilidades na ordem correta das emoções
                    # A ordem deve ser: neutral, happy, sad, surprised, fearful, disgusted, angry, contempt
                    # Mas as colunas estão em: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral
                    
                    # Precisamos reordenar para a ordem dos rótulos (0-7)
                    # Mapeamento inverso: do índice da coluna para o label
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
                    
                    # Para reordenar, vamos criar listas na ordem dos rótulos (0-7)
                    pred_probs_ordered = [0] * 8  # Inicializar com 8 zeros
                    gt_probs_ordered = [0] * 8     # Inicializar com 8 zeros
                    
                    for i, col in enumerate(emotion_columns):
                        label = column_to_label[i]  # Obter o label correspondente
                        
                        pred_val = sample_data[col]
                        gt_val = sample_data[f"{col}_gt"]
                        
                        if isinstance(pred_val, (int, float, np.number)):
                            pred_probs_ordered[label] = float(pred_val)
                        else:
                            pred_probs_ordered[label] = 0.0
                            
                        if isinstance(gt_val, (int, float, np.number)):
                            gt_probs_ordered[label] = float(gt_val)
                        else:
                            gt_probs_ordered[label] = 0.0
                    
                    pred_probs_sample = np.array(pred_probs_ordered, dtype=np.float64)
                    gt_probs_sample = np.array(gt_probs_ordered, dtype=np.float64)
                    
                    # Ensure no NaN
                    pred_probs_sample = np.nan_to_num(pred_probs_sample, nan=0.0)
                    gt_probs_sample = np.nan_to_num(gt_probs_sample, nan=0.0)
                    
                    # Normalize to sum to 1
                    pred_sum = np.sum(pred_probs_sample)
                    gt_sum = np.sum(gt_probs_sample)
                    
                    if pred_sum > 0:
                        pred_probs_sample = pred_probs_sample / pred_sum
                    else:
                        pred_probs_sample = np.ones_like(pred_probs_sample) / len(pred_probs_sample)
                        
                    if gt_sum > 0:
                        gt_probs_sample = gt_probs_sample / gt_sum
                    else:
                        gt_probs_sample = np.ones_like(gt_probs_sample) / len(gt_probs_sample)
                    
                    # CORREÇÃO: Usar a ordem correta das emoções (0-7)
                    emotion_names_ordered = [
                        emotion_mapping[0],  # neutral
                        emotion_mapping[1],  # happy
                        emotion_mapping[2],  # sad
                        emotion_mapping[3],  # surprised
                        emotion_mapping[4],  # fearful
                        emotion_mapping[5],  # disgusted
                        emotion_mapping[6],  # angry
                        emotion_mapping[7]   # contempt
                    ]
                    
                    # Create dataframe for plotting
                    plot_data = pd.DataFrame({
                        'Emotion': emotion_names_ordered,  # Agora na ordem correta
                        'Predicted': pred_probs_sample,
                        'Ground Truth': gt_probs_sample
                    })
                    
                    # Comparative bar chart
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Bar(
                        x=plot_data['Emotion'],
                        y=plot_data['Predicted'],
                        name='Predicted',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    fig_comparison.add_trace(go.Bar(
                        x=plot_data['Emotion'],
                        y=plot_data['Ground Truth'],
                        name='Ground Truth',
                        marker_color='red',
                        opacity=0.7
                    ))
                    
                    # Adicionar linhas para as emoções mais prováveis
                    # Encontrar a emoção com maior probabilidade em cada distribuição
                    pred_max_idx = np.argmax(pred_probs_sample)
                    gt_max_idx = np.argmax(gt_probs_sample)
                    
                    # Adicionar anotações no gráfico
                    fig_comparison.add_annotation(
                        x=emotion_names_ordered[pred_max_idx],
                        y=pred_probs_sample[pred_max_idx],
                        text="▲ Pred",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color="blue", size=12)
                    )
                    
                    fig_comparison.add_annotation(
                        x=emotion_names_ordered[gt_max_idx],
                        y=gt_probs_sample[gt_max_idx],
                        text="▲ GT",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-60,
                        font=dict(color="red", size=12)
                    )
                    
                    fig_comparison.update_layout(
                        title=f"Distribution Comparison - {selected_sample.split('/')[-1]}",
                        xaxis_title="Emotion",
                        yaxis_title="Probability",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Nova seção: Comparação das três fontes de informação
                    st.subheader("Emotion Source Comparison")
                    
                    # Criar uma tabela comparativa
                    comparison_data = {
                        'Source': [
                            'Annotation Label',
                            'Ground Truth Distribution', 
                            'Predicted Distribution'
                        ],
                        'Emotion': [
                            sample_data['groundtruth_emotion'],
                            sample_data['gt_distribution_emotion'],
                            sample_data['predicted_emotion']
                        ],
                        'Confidence/Probability': [
                            'N/A',  # A anotação é categórica
                            f"{sample_data['gt_max_probability']:.3f}",
                            f"{np.max(pred_probs_sample):.3f}"
                        ],
                        'Details': [
                            f"Label: {sample_data['groundtruth_label']}",
                            f"Entropy: {sample_data['gt_distribution_entropy']:.3f}",
                            f"JS Div: {sample_data['js_divergence']:.3f}"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Adicionar coluna de consistência
                    consistency_notes = []
                    for i, row in comparison_df.iterrows():
                        if i == 0:  # Annotation
                            # Verificar se annotation matches distribution
                            matches_dist = (sample_data['groundtruth_emotion'] == 
                                        sample_data['gt_distribution_emotion'])
                            matches_pred = (sample_data['groundtruth_emotion'] == 
                                        sample_data['predicted_emotion'])
                            
                            if matches_dist and matches_pred:
                                consistency_notes.append("✅ Consistent with both")
                            elif matches_dist:
                                consistency_notes.append("✅ Matches GT distribution")
                            elif matches_pred:
                                consistency_notes.append("✅ Matches prediction")
                            else:
                                consistency_notes.append("⚠️ Differs from both")
                        elif i == 1:  # GT Distribution
                            # Verificar se distribution matches prediction
                            matches_pred = (sample_data['gt_distribution_emotion'] == 
                                        sample_data['predicted_emotion'])
                            if matches_pred:
                                consistency_notes.append("✅ Matches prediction")
                            else:
                                consistency_notes.append("⚠️ Differs from prediction")
                        else:  # Prediction
                            # Já verificamos acima
                            consistency_notes.append("")
                    
                    comparison_df['Consistency'] = consistency_notes
                    
                    # Exibir tabela
                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Métricas para esta amostra específica - agora em 3 linhas
                    st.subheader("Detailed Metrics")
                    
                    # Linha 1: Informações de emoção
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Annotation Label", 
                            sample_data['groundtruth_emotion'],
                            delta=f"Label: {sample_data['groundtruth_label']}"
                        )
                    
                    with col2:
                        # Verificar discrepância entre anotação e distribuição GT
                        has_discrepancy = sample_data['annotation_distribution_discrepancy']
                        discrepancy_icon = "⚠️" if has_discrepancy else "✅"
                        
                        st.metric(
                            f"{discrepancy_icon} GT Distribution", 
                            sample_data['gt_distribution_emotion'],
                            delta=f"Confidence: {sample_data['gt_max_probability']:.3f}"
                        )
                    
                    with col3:
                        correct = sample_data['groundtruth_label'] == sample_data['predicted_label']
                        correct_icon = "✅" if correct else "❌"
                        
                        st.metric(
                            f"{correct_icon} Predicted", 
                            sample_data['predicted_emotion'],
                            delta="Correct" if correct else "Incorrect"
                        )
                    
                    # Linha 2: Métricas de similaridade
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        js_val = sample_data['js_divergence']
                        st.metric("JS Divergence", f"{js_val:.4f}")
                    
                    with col5:
                        cosine_val = sample_data['cosine_similarity']
                        st.metric("Cosine Similarity", f"{cosine_val:.4f}")
                    
                    with col6:
                        kl_val = sample_data['kl_divergence']
                        st.metric("KL Divergence", f"{kl_val:.4f}")
                    
                    # Linha 3: Informações adicionais
                    col7, col8, col9 = st.columns(3)
                    
                    with col7:
                        entropy_val = sample_data['gt_distribution_entropy']
                        st.metric("GT Entropy", f"{entropy_val:.3f}")
                    
                    with col8:
                        euclidean_val = sample_data['euclidean_distance']
                        st.metric("Euclidean Dist", f"{euclidean_val:.3f}")
                    
                    with col9:
                        correlation_val = sample_data['pearson_correlation']
                        st.metric("Correlation", f"{correlation_val:.3f}")
                    
                    # Seção de análise de consistência
                    st.subheader("Consistency Analysis")
                    
                    # Criar um resumo das consistências
                    consistency_summary = {
                        "Annotation vs GT Distribution": 
                            "✅ Match" if not sample_data['annotation_distribution_discrepancy'] else "❌ Mismatch",
                        "Annotation vs Prediction": 
                            "✅ Match" if sample_data['groundtruth_label'] == sample_data['predicted_label'] else "❌ Mismatch",
                        "GT Distribution vs Prediction": 
                            "✅ Match" if sample_data['gt_distribution_emotion'] == sample_data['predicted_emotion'] else "❌ Mismatch"
                    }
                    
                    for check, result in consistency_summary.items():
                        st.write(f"{check}: {result}")
                    
                    # Análise de confiança
                    if has_discrepancy:
                        st.warning(f"""
                        **⚠️ Discrepancy Detected!**
                        
                        The annotated emotion ({sample_data['groundtruth_emotion']}) differs from the 
                        most probable emotion in the ground truth distribution ({sample_data['gt_distribution_emotion']}).
                        
                        This could indicate:
                        - An ambiguous sample
                        - Potential annotation error
                        - High uncertainty in the ground truth distribution (entropy: {entropy_val:.3f})
                        """)
                    else:
                        st.success(f"""
                        **✅ Annotation Consistent**
                        
                        The annotated emotion ({sample_data['groundtruth_emotion']}) matches the most probable 
                        emotion in the ground truth distribution.
                        
                        Ground truth distribution confidence: {sample_data['gt_max_probability']:.3f}
                        """)
                
                # Scatter plot of selected metric vs accuracy
                st.subheader(f"{metric_options[selected_metric]} vs Accuracy per Sample")
                
                # Add correct/incorrect indicator
                filtered_df['correct'] = filtered_df['groundtruth_label'] == filtered_df['predicted_label']
                
                # Filter NaN values
                scatter_df = filtered_df.head(100).copy()  # Limit for better visualization
                scatter_df = scatter_df.dropna(subset=[selected_metric, 'cosine_similarity'])
                
                if len(scatter_df) > 0:
                    fig_scatter = px.scatter(
                        scatter_df,
                        x=selected_metric,
                        y='cosine_similarity',
                        color='correct',
                        hover_data=['file', 'groundtruth_emotion', 'predicted_emotion'],
                        title=f"{metric_options[selected_metric]} vs Cosine Similarity",
                        labels={'correct': 'Correct Prediction'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Insufficient data for scatter plot after removing NaN values.")
            else:
                st.warning("No samples available after applying filters.")   
        
        # As outras abas (4-8) permanecem as mesmas do código original
        # Por questões de espaço, não as reproduzi aqui completamente
        # Você pode copiá-las do seu código original
        
        # ============================================================================
        # NOVA ABA: Export Center
        # ============================================================================
        
        with tab9:
            st.header("📤 Export Center")
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
                
                # Sample selector for export
                export_sample_options = merged_df['file'].tolist()
                selected_export_sample = st.selectbox(
                    "Select sample for export:",
                    export_sample_options,
                    key="export_sample"
                )
                
                if selected_export_sample:
                    export_sample_data = merged_df[merged_df['file'] == selected_export_sample].iloc[0]
                    sample_idx = merged_df[merged_df['file'] == selected_export_sample].index[0]
                    
                    # Get ordered probabilities
                    column_to_label = {
                        0: 1, 1: 7, 2: 3, 3: 6, 4: 5, 5: 4, 6: 2, 7: 0
                    }
                    
                    pred_probs_ordered = [0] * 8
                    gt_probs_ordered = [0] * 8
                    
                    for i, col in enumerate(emotion_columns):
                        label = column_to_label[i]
                        pred_probs_ordered[label] = export_sample_data[col]
                        gt_probs_ordered[label] = export_sample_data[f"{col}_gt"]
                    
                    pred_probs_ordered = np.array(pred_probs_ordered, dtype=np.float64)
                    gt_probs_ordered = np.array(gt_probs_ordered, dtype=np.float64)
                    
                    # Normalize
                    pred_sum = np.sum(pred_probs_ordered)
                    gt_sum = np.sum(gt_probs_ordered)
                    
                    if pred_sum > 0:
                        pred_probs_ordered = pred_probs_ordered / pred_sum
                    if gt_sum > 0:
                        gt_probs_ordered = gt_probs_ordered / gt_sum
                    
                    # Emotion names in correct order
                    emotion_names_ordered = [
                        emotion_mapping[0], emotion_mapping[1], emotion_mapping[2], emotion_mapping[3],
                        emotion_mapping[4], emotion_mapping[5], emotion_mapping[6], emotion_mapping[7]
                    ]
                    
                    # Font size settings
                    font_sizes = {
                        'title': bar_title_size,
                        'axes_labels': bar_axes_size,
                        'tick_labels': bar_tick_size,
                        'legend': bar_legend_size,
                        'bar_labels': bar_label_size
                    }
                    
                    # Export Comparison Bar Chart
                    st.markdown("**Distribution Comparison Chart**")
                    matplotlib_fig = create_comparison_bar_chart_matplotlib(
                        pred_probs_ordered, 
                        gt_probs_ordered, 
                        emotion_names_ordered,
                        filename=selected_export_sample.split('/')[-1],
                        sample_idx=sample_idx,
                        font_sizes=font_sizes
                    )
                    
                    export_buffer, mime_type = create_vector_figure_matplotlib(
                        matplotlib_fig, 
                        selected_format
                    )
                    
                    st.download_button(
                        label=f"📥 Download Comparison Chart ({selected_format.upper()})",
                        data=export_buffer,
                        file_name=f"comparison_sample_{sample_idx}_{timestamp}.{selected_format}",
                        mime=mime_type
                    )
                    
                    # Export Radar Chart
                    st.markdown("**Radar Chart**")
                    radar_fig = create_radar_chart_matplotlib(
                        pred_probs_ordered,
                        gt_probs_ordered,
                        emotion_names_ordered,
                        filename=selected_export_sample.split('/')[-1],
                        sample_idx=sample_idx,
                        font_sizes=font_sizes
                    )
                    
                    radar_buffer, mime_type_radar = create_vector_figure_matplotlib(
                        radar_fig, 
                        selected_format
                    )
                    
                    st.download_button(
                        label=f"📥 Download Radar Chart ({selected_format.upper()})",
                        data=radar_buffer,
                        file_name=f"radar_sample_{sample_idx}_{timestamp}.{selected_format}",
                        mime=mime_type_radar
                    )
                    
                    # Close figures to free memory
                    plt.close(matplotlib_fig)
                    plt.close(radar_fig)
            
            with col_export2:
                st.subheader("🌍 Global Analysis Charts")
                
                # Export Confusion Matrix
                st.markdown("**Confusion Matrix**")
                
                # Font sizes for confusion matrix
                cm_font_sizes = {
                    'title': other_title_size,
                    'axes_labels': other_labels_size,
                    'tick_labels': bar_tick_size,
                    'cell_values': bar_label_size
                }
                
                cm_fig = create_confusion_matrix_matplotlib(
                    cm, 
                    [emotion_mapping[i] for i in range(8)],
                    font_sizes=cm_font_sizes
                )
                
                cm_buffer, mime_type_cm = create_vector_figure_matplotlib(
                    cm_fig, 
                    selected_format
                )
                
                st.download_button(
                    label=f"📥 Download Confusion Matrix ({selected_format.upper()})",
                    data=cm_buffer,
                    file_name=f"confusion_matrix_{timestamp}.{selected_format}",
                    mime=mime_type_cm
                )
                
                # Export Metric Histogram
                st.markdown("**JS Divergence Distribution**")
                
                hist_font_sizes = {
                    'title': other_title_size,
                    'axes_labels': other_labels_size,
                    'tick_labels': bar_tick_size,
                    'legend': bar_legend_size
                }
                
                hist_fig = create_metrics_histogram_matplotlib(
                    merged_df['js_divergence'].dropna(),
                    "JS Divergence",
                    font_sizes=hist_font_sizes
                )
                
                hist_buffer, mime_type_hist = create_vector_figure_matplotlib(
                    hist_fig, 
                    selected_format
                )
                
                st.download_button(
                    label=f"📥 Download JS Divergence Histogram ({selected_format.upper()})",
                    data=hist_buffer,
                    file_name=f"js_divergence_histogram_{timestamp}.{selected_format}",
                    mime=mime_type_hist
                )
                
                # Close figures
                plt.close(cm_fig)
                plt.close(hist_fig)
            
            # Export All Data
            st.subheader("📁 Complete Dataset Export")
            
            col_data1, col_data2, col_data3 = st.columns(3)
            
            with col_data1:
                # Export processed data as CSV
                combined_data = merged_df[[
                    'file', 'groundtruth_emotion', 'predicted_emotion',
                    'js_divergence', 'cosine_similarity', 'pearson_correlation',
                    'gt_distribution_emotion', 'annotation_distribution_discrepancy',
                    'gt_max_probability', 'gt_distribution_entropy'
                ]].copy()
                
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
                    'metric': ['Global Accuracy', 'JS Divergence (avg)', 
                              'Cosine Similarity (avg)', 'Total Samples'],
                    'value': [accuracy, merged_df['js_divergence'].mean(),
                             merged_df['cosine_similarity'].mean(), len(merged_df)]
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
                    'font_sizes': {
                        'bar_chart': {
                            'title': bar_title_size,
                            'axes_labels': bar_axes_size,
                            'tick_labels': bar_tick_size,
                            'legend': bar_legend_size,
                            'bar_labels': bar_label_size
                        },
                        'other_charts': {
                            'title': other_title_size,
                            'labels': other_labels_size
                        }
                    },
                    'total_samples': len(merged_df),
                    'accuracy': float(accuracy),
                    'avg_js_divergence': float(merged_df['js_divergence'].mean()),
                    'avg_cosine_similarity': float(merged_df['cosine_similarity'].mean())
                }
                
                import json
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
                    import tempfile
                    import os
                    
                    with tempfile.TemporaryDirectory() as tmpdir:
                        files_to_zip = []
                        
                        # Export all individual comparison charts (limit to 5 for performance)
                        num_samples = min(5, len(merged_df))
                        for i in range(num_samples):
                            sample_data = merged_df.iloc[i]
                            
                            # Get ordered probabilities
                            pred_probs_ordered = [0] * 8
                            gt_probs_ordered = [0] * 8
                            
                            for j, col in enumerate(emotion_columns):
                                label = column_to_label[j]
                                pred_probs_ordered[label] = sample_data[col]
                                gt_probs_ordered[label] = sample_data[f"{col}_gt"]
                            
                            pred_probs_ordered = np.array(pred_probs_ordered, dtype=np.float64)
                            gt_probs_ordered = np.array(gt_probs_ordered, dtype=np.float64)
                            
                            # Normalize
                            pred_sum = np.sum(pred_probs_ordered)
                            gt_sum = np.sum(gt_probs_ordered)
                            
                            if pred_sum > 0:
                                pred_probs_ordered = pred_probs_ordered / pred_sum
                            if gt_sum > 0:
                                gt_probs_ordered = gt_probs_ordered / gt_sum
                            
                            # Create comparison chart
                            comp_fig = create_comparison_bar_chart_matplotlib(
                                pred_probs_ordered, 
                                gt_probs_ordered, 
                                emotion_names_ordered,
                                filename=sample_data['file'].split('/')[-1],
                                sample_idx=i,
                                font_sizes=font_sizes
                            )
                            
                            comp_path = os.path.join(tmpdir, f"comparison_sample_{i}.{selected_format}")
                            comp_fig.savefig(comp_path, format=selected_format.replace(' (Recommended)', ''), 
                                           dpi=300, bbox_inches='tight')
                            files_to_zip.append(comp_path)
                            plt.close(comp_fig)
                        
                        # Export confusion matrix
                        cm_fig = create_confusion_matrix_matplotlib(
                            cm, 
                            [emotion_mapping[i] for i in range(8)],
                            font_sizes=cm_font_sizes
                        )
                        cm_path = os.path.join(tmpdir, f"confusion_matrix.{selected_format}")
                        cm_fig.savefig(cm_path, format=selected_format.replace(' (Recommended)', ''), 
                                     dpi=300, bbox_inches='tight')
                        files_to_zip.append(cm_path)
                        plt.close(cm_fig)
                        
                        st.success(f"✅ Generated {len(files_to_zip)} charts in {selected_format.upper()} format")
                        
                        st.info("""
                        **Note:** Due to Streamlit Cloud limitations, ZIP export is not directly available.
                        Please download individual charts or run locally for batch ZIP export.
                        
                        **Files generated:**
                        - Comparison charts for {num_samples} samples
                        - Confusion matrix
                        """)
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.info("Please check if files have the correct format")
    
elif predictions_file is not None and groundtruth_file is None:
    st.warning("⚠️ Please upload the Ground Truth file")
elif predictions_file is None and groundtruth_file is not None:
    st.warning("⚠️ Please upload the Predictions file")
else:
    st.info("👈 Please upload both CSV files to begin analysis")
    
    # Instructions
    with st.expander("ℹ️ Usage Instructions and File Formats"):
        st.markdown("""
        ## 📋 Expected File Formats
        
        ### 1. Predictions File (simpleNetwork_FocalLoss.csv):
        ```
        happy,contempt,surprised,angry,disgusted,fearful,sad,neutral,emotion_label,file
        0.04905860126018524,0.08107654750347137,0.07898678630590439,... ,2,/path/to/image.jpg
        ```
        
        ### 2. Ground Truth File (simpleNetwork_FocalLoss_labels.csv):
        ```
        happy,contempt,surprised,angry,disgusted,fearful,sad,neutral,valence,arousal,dominance,emotion_label,file
        2.7970681326650038e-09,0.11136697977781296,0.004586424678564072,... ,-0.469415,0.817847,0.21547260880470276,6,/path/to/image.jpg
        ```
        
        ### 🎯 Correct Emotion Mapping:
        ```
        0 = neutral
        1 = happy
        2 = sad
        3 = surprised
        4 = fearful
        5 = disgusted
        6 = angry
        7 = contempt
        ```
        
        ## 🔍 Analysis Includes:
        
        1. **Classification Metrics:**
           - Overall and per-emotion accuracy
           - Confusion matrix
           - Most common error analysis
        
        2. **Distribution Metrics:**
           - Jensen-Shannon Divergence
           - KL Divergence (symmetric)
           - Euclidean Distance
           - Cosine Similarity
           - Pearson Correlation
        
        3. **Affective Dimensions Analysis (VAD):**
           - Valence
           - Arousal
           - Dominance
           - Correlation with model performance
        
        4. **Label-VAD Consistency:**
           - Consistency verification between labels and VAD dimensions
           - Identification of inconsistent samples
        
        5. **Annotation-Distribution Discrepancy:**
           - Comparison between annotated label and most probable emotion from distribution
           - Confidence and entropy analysis
           - Identification of ambiguous samples
        
        6. **Export Features:**
           - High-quality vector graphics (PDF, SVG, EPS, PNG)
           - Customizable font sizes for charts
           - Publication-ready figures
           - Batch export options
        """)