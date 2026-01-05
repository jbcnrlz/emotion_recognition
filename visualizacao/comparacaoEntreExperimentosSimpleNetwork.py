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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition Performance Analysis",
    page_icon="😊",
    layout="wide"
)

# Application title
st.title("📊 Facial Emotion Recognition Performance Analysis - Multi-Experiment")
st.markdown("---")

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

# Function to process single experiment
@st.cache_data(ttl=3600)
def process_experiment(predictions_content, groundtruth_content, experiment_name, data_hash):
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
        return None
    if missing_in_gt:
        st.error(f"⚠️ Missing columns in ground truth file: {missing_in_gt}")
        return None
    
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
        st.warning(f"⚠️ Files are not perfectly aligned by image in experiment '{experiment_name}'. Merging by file paths...")
        
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
        st.error(f"❌ Ground truth file doesn't have emotion_label column in experiment '{experiment_name}'")
        return None
    
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
        st.warning(f"⚠️ Removed {original_count - filtered_count} samples with invalid labels (outside 0-7 range) in experiment '{experiment_name}'.")
    
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
    
    # Add experiment name as a column
    merged_df['experiment_name'] = experiment_name
    
    return merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations

# Function to calculate aggregated metrics for each experiment
def calculate_experiment_metrics(experiments_data):
    metrics_list = []
    
    for exp_name, exp_data in experiments_data.items():
        merged_df, _, emotion_mapping, _, extra_columns, _ = exp_data
        
        if merged_df is None:
            continue
            
        # Calculate accuracy
        accuracy = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
        
        # Calculate per-emotion accuracy
        emotion_accuracies = {}
        for emotion_idx, emotion_name in emotion_mapping.items():
            mask = merged_df['groundtruth_label'] == emotion_idx
            if mask.sum() > 0:
                subset = merged_df[mask]
                emotion_acc = accuracy_score(subset['groundtruth_label'], subset['predicted_label'])
                emotion_accuracies[emotion_name] = emotion_acc
        
        # Calculate average metrics
        avg_metrics = {
            'js_divergence': merged_df['js_divergence'].mean(),
            'cosine_similarity': merged_df['cosine_similarity'].mean(),
            'pearson_correlation': merged_df['pearson_correlation'].mean(),
            'kl_divergence': merged_df['kl_divergence'].mean(),
            'euclidean_distance': merged_df['euclidean_distance'].mean(),
        }
        
        # Calculate annotation-distribution discrepancy
        discrepancy_rate = merged_df['annotation_distribution_discrepancy'].mean()
        
        # Calculate average confidence and entropy
        avg_confidence = merged_df['gt_max_probability'].mean()
        avg_entropy = merged_df['gt_distribution_entropy'].mean()
        
        # Calculate VAD consistency if available
        vad_consistency = merged_df['vad_consistency_score'].mean() if 'vad_consistency_score' in merged_df.columns else np.nan
        
        # Create metrics dictionary
        metrics_dict = {
            'experiment_name': exp_name,
            'samples': len(merged_df),
            'accuracy': accuracy,
            'discrepancy_rate': discrepancy_rate,
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy,
            'vad_consistency': vad_consistency,
            **avg_metrics,
        }
        
        # Add per-emotion accuracies
        for emotion, acc in emotion_accuracies.items():
            metrics_dict[f'acc_{emotion}'] = acc
        
        metrics_list.append(metrics_dict)
    
    return pd.DataFrame(metrics_list)

# Sidebar for file upload
st.sidebar.header("📁 File Upload")

# Ground truth file uploader (single)
groundtruth_file = st.sidebar.file_uploader(
    "Ground Truth File (simpleNetwork_FocalLoss_labels.csv)",
    type=['csv'],
    help="File with ground truth values - should have columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral, valence, arousal, dominance, emotion_label, file"
)

# Multiple prediction files uploader
prediction_files = st.sidebar.file_uploader(
    "Prediction Files (CSV format)",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload multiple prediction files for comparison"
)

# Dictionary to store experiment names
experiment_names = {}

if prediction_files:
    st.sidebar.header("🔧 Experiment Naming")
    
    for i, file in enumerate(prediction_files):
        default_name = file.name.replace('.csv', '').replace('_', ' ')
        exp_name = st.sidebar.text_input(
            f"Name for {file.name}",
            value=default_name,
            key=f"exp_name_{i}"
        )
        experiment_names[file.name] = exp_name

# Check if files are uploaded
if groundtruth_file is not None and prediction_files:
    # Read ground truth content
    groundtruth_content = groundtruth_file.getvalue().decode('utf-8')
    
    # Process each experiment
    experiments_data = {}
    progress_bar = st.progress(0)
    
    for idx, pred_file in enumerate(prediction_files):
        experiment_name = experiment_names.get(pred_file.name, pred_file.name)
        
        with st.spinner(f"Processing experiment: {experiment_name}..."):
            # Read prediction content
            predictions_content = pred_file.getvalue().decode('utf-8')
            
            # Generate hash for caching
            data_hash = generate_data_hash(predictions_content, groundtruth_content)
            
            try:
                # Process experiment
                result = process_experiment(predictions_content, groundtruth_content, experiment_name, data_hash)
                
                if result[0] is not None:
                    experiments_data[experiment_name] = result
                    st.success(f"✅ Experiment '{experiment_name}' processed successfully!")
                else:
                    st.error(f"❌ Error processing experiment '{experiment_name}'")
            except Exception as e:
                st.error(f"❌ Error processing experiment '{experiment_name}': {str(e)}")
        
        progress_bar.progress((idx + 1) / len(prediction_files))
    
    # Check if we have any valid experiments
    if not experiments_data:
        st.error("❌ No experiments were successfully processed. Please check your files.")
        st.stop()
    
    # Calculate aggregated metrics for all experiments
    metrics_df = calculate_experiment_metrics(experiments_data)
    
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
    
    # Main tabs for multi-experiment analysis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Experiment Comparison", 
        "📊 Performance Metrics", 
        "🎭 Per-Emotion Analysis", 
        "📉 Distribution Metrics",
        "⚠️ Discrepancy Analysis",
        "🔍 Individual Experiments",
        "📋 Download Results"
    ])
    
    with tab1:
        st.header("Experiment Comparison Overview")
        
        # Display metrics table
        st.subheader("Aggregated Metrics by Experiment")
        
        # Format the metrics dataframe for display
        display_cols = ['experiment_name', 'samples', 'accuracy', 'cosine_similarity', 
                       'js_divergence', 'discrepancy_rate', 'avg_confidence', 'avg_entropy']
        
        if 'vad_consistency' in metrics_df.columns:
            display_cols.append('vad_consistency')
        
        display_df = metrics_df[display_cols].copy()
        
        # Format percentages
        for col in ['accuracy', 'discrepancy_rate', 'avg_confidence', 'vad_consistency']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
        
        # Format other metrics
        for col in ['cosine_similarity', 'js_divergence', 'avg_entropy']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Bar chart comparison
        st.subheader("Key Metrics Comparison")
        
        # Select metrics to compare
        metric_options_comparison = {
            'accuracy': 'Accuracy',
            'cosine_similarity': 'Cosine Similarity',
            'js_divergence': 'JS Divergence',
            'discrepancy_rate': 'Annotation-Distribution Discrepancy',
            'avg_confidence': 'Average Confidence',
            'avg_entropy': 'Average Entropy'
        }
        
        selected_comparison_metrics = st.multiselect(
            "Select metrics to compare:",
            list(metric_options_comparison.keys()),
            default=['accuracy', 'cosine_similarity'],
            format_func=lambda x: metric_options_comparison[x]
        )
        
        if selected_comparison_metrics:
            fig_comparison = go.Figure()
            
            for metric in selected_comparison_metrics:
                fig_comparison.add_trace(go.Bar(
                    x=metrics_df['experiment_name'],
                    y=metrics_df[metric],
                    name=metric_options_comparison[metric],
                    text=metrics_df[metric].apply(lambda x: f"{x:.3f}" if metric in ['accuracy', 'discrepancy_rate'] else f"{x:.4f}"),
                    textposition='auto'
                ))
            
            fig_comparison.update_layout(
                title='Experiment Comparison',
                xaxis_title='Experiment',
                yaxis_title='Value',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Radar chart for comprehensive comparison
        st.subheader("Comprehensive Radar Chart Comparison")
        
        # Select experiments for radar chart
        selected_experiments = st.multiselect(
            "Select experiments for radar chart:",
            metrics_df['experiment_name'].tolist(),
            default=metrics_df['experiment_name'].tolist()[:3]
        )
        
        # Select metrics for radar chart
        radar_metrics = ['accuracy', 'cosine_similarity', 'js_divergence', 'discrepancy_rate']
        
        if selected_experiments:
            fig_radar = go.Figure()
            
            for exp_name in selected_experiments:
                exp_data = metrics_df[metrics_df['experiment_name'] == exp_name].iloc[0]
                
                values = []
                for metric in radar_metrics:
                    val = exp_data[metric]
                    # Normalize JS divergence and discrepancy rate (lower is better)
                    if metric in ['js_divergence', 'discrepancy_rate']:
                        # Invert so higher is better for radar chart
                        values.append(1 - val)
                    else:
                        values.append(val)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[metric_options_comparison.get(m, m) for m in radar_metrics],
                    fill='toself',
                    name=exp_name
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title='Radar Chart Comparison (Higher is Better)',
                height=600
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.header("Performance Metrics Analysis")
        
        # Accuracy comparison with statistical significance
        st.subheader("Accuracy Comparison")
        
        # Calculate accuracy for each experiment
        accuracy_data = []
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, _, _, _, _ = exp_data
            if merged_df is not None:
                accuracy = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
                accuracy_data.append({
                    'experiment': exp_name,
                    'accuracy': accuracy,
                    'samples': len(merged_df)
                })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Bar chart
        fig_accuracy = px.bar(
            accuracy_df.sort_values('accuracy', ascending=False),
            x='experiment',
            y='accuracy',
            title='Accuracy by Experiment',
            color='accuracy',
            text='accuracy',
            color_continuous_scale='Viridis'
        )
        fig_accuracy.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_accuracy.update_layout(height=500)
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Statistical comparison
        st.subheader("Statistical Comparison")
        
        # Calculate confidence intervals for accuracy
        import scipy.stats as stats
        
        ci_data = []
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, _, _, _, _ = exp_data
            if merged_df is not None:
                n = len(merged_df)
                correct = (merged_df['groundtruth_label'] == merged_df['predicted_label']).sum()
                p = correct / n
                
                # Wilson score interval
                z = stats.norm.ppf(0.975)  # 95% confidence
                denominator = 1 + z**2 / n
                centre_adjusted_probability = p + z**2 / (2 * n)
                adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
                
                lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
                upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
                
                ci_data.append({
                    'experiment': exp_name,
                    'accuracy': p,
                    'lower_ci': lower_bound,
                    'upper_ci': upper_bound
                })
        
        ci_df = pd.DataFrame(ci_data)
        
        # Plot with confidence intervals
        fig_ci = go.Figure()
        
        for _, row in ci_df.iterrows():
            fig_ci.add_trace(go.Scatter(
                x=[row['experiment'], row['experiment']],
                y=[row['lower_ci'], row['upper_ci']],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig_ci.add_trace(go.Scatter(
            x=ci_df['experiment'],
            y=ci_df['accuracy'],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Accuracy'
        ))
        
        fig_ci.update_layout(
            title='Accuracy with 95% Confidence Intervals',
            xaxis_title='Experiment',
            yaxis_title='Accuracy',
            height=500
        )
        st.plotly_chart(fig_ci, use_container_width=True)
    
    with tab3:
        st.header("Per-Emotion Analysis")
        
        # Emotion accuracy comparison across experiments
        st.subheader("Per-Emotion Accuracy Comparison")
        
        # Get emotion mapping from first experiment
        first_exp = list(experiments_data.values())[0]
        emotion_mapping = first_exp[2]
        
        # Prepare data for heatmap
        heatmap_data = []
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, emotion_mapping, _, _, _ = exp_data
            if merged_df is not None:
                for emotion_idx, emotion_name in emotion_mapping.items():
                    mask = merged_df['groundtruth_label'] == emotion_idx
                    if mask.sum() > 0:
                        subset = merged_df[mask]
                        emotion_acc = accuracy_score(subset['groundtruth_label'], subset['predicted_label'])
                        heatmap_data.append({
                            'experiment': exp_name,
                            'emotion': emotion_name,
                            'accuracy': emotion_acc,
                            'samples': mask.sum()
                        })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        if not heatmap_df.empty:
            # Create pivot table for heatmap
            pivot_df = heatmap_df.pivot(index='experiment', columns='emotion', values='accuracy')
            
            # Heatmap
            fig_heatmap = px.imshow(
                pivot_df,
                labels=dict(x="Emotion", y="Experiment", color="Accuracy"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title='Per-Emotion Accuracy Heatmap'
            )
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Bar chart for best/worst emotions per experiment
            st.subheader("Best and Worst Performing Emotions")
            
            for exp_name in pivot_df.index:
                exp_row = pivot_df.loc[exp_name]
                best_emotion = exp_row.idxmax()
                worst_emotion = exp_row.idxmin()
                best_acc = exp_row.max()
                worst_acc = exp_row.min()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"Best in {exp_name}",
                        value=best_emotion,
                        delta=f"{best_acc:.2%}"
                    )
                with col2:
                    st.metric(
                        label=f"Worst in {exp_name}",
                        value=worst_emotion,
                        delta=f"{worst_acc:.2%}",
                        delta_color="inverse"
                    )
    
    with tab4:
        st.header("Distribution Metrics Comparison")
        
        # Select distribution metric to compare
        distribution_metric_options = {
            'js_divergence': 'Jensen-Shannon Divergence',
            'kl_divergence': 'KL Divergence',
            'euclidean_distance': 'Euclidean Distance',
            'cosine_similarity': 'Cosine Similarity',
            'pearson_correlation': 'Pearson Correlation'
        }
        
        selected_dist_metric = st.selectbox(
            "Select distribution metric:",
            list(distribution_metric_options.keys()),
            format_func=lambda x: distribution_metric_options[x]
        )
        
        # Box plot comparison
        st.subheader(f"{distribution_metric_options[selected_dist_metric]} Distribution Comparison")
        
        # Prepare data for box plot
        box_data = []
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, _, _, _, _ = exp_data
            if merged_df is not None:
                for value in merged_df[selected_dist_metric].values:
                    box_data.append({
                        'experiment': exp_name,
                        'value': value
                    })
        
        box_df = pd.DataFrame(box_data)
        
        if not box_df.empty:
            fig_box = px.box(
                box_df,
                x='experiment',
                y='value',
                title=f'Distribution of {distribution_metric_options[selected_dist_metric]} by Experiment',
                points='all',
                color='experiment'
            )
            fig_box.update_layout(height=500)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Violin plot for better distribution visualization
            fig_violin = px.violin(
                box_df,
                x='experiment',
                y='value',
                title=f'Violin Plot of {distribution_metric_options[selected_dist_metric]} by Experiment',
                color='experiment',
                box=True,
                points='all'
            )
            fig_violin.update_layout(height=500)
            st.plotly_chart(fig_violin, use_container_width=True)
    
    with tab5:
        st.header("Annotation-Distribution Discrepancy Analysis")
        
        # Compare discrepancy rates
        st.subheader("Discrepancy Rate Comparison")
        
        discrepancy_data = []
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, _, _, _, _ = exp_data
            if merged_df is not None:
                discrepancy_rate = merged_df['annotation_distribution_discrepancy'].mean()
                avg_confidence = merged_df['gt_max_probability'].mean()
                avg_entropy = merged_df['gt_distribution_entropy'].mean()
                
                discrepancy_data.append({
                    'experiment': exp_name,
                    'discrepancy_rate': discrepancy_rate,
                    'avg_confidence': avg_confidence,
                    'avg_entropy': avg_entropy
                })
        
        discrepancy_df = pd.DataFrame(discrepancy_data)
        
        # Bar chart for discrepancy rates
        fig_discrepancy = px.bar(
            discrepancy_df.sort_values('discrepancy_rate', ascending=False),
            x='experiment',
            y='discrepancy_rate',
            title='Annotation-Distribution Discrepancy Rate by Experiment',
            color='discrepancy_rate',
            text='discrepancy_rate',
            color_continuous_scale='RdYlGn_r'
        )
        fig_discrepancy.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_discrepancy.update_layout(height=500)
        st.plotly_chart(fig_discrepancy, use_container_width=True)
        
        # Scatter plot: Confidence vs Entropy
        st.subheader("Confidence vs Entropy Analysis")
        
        fig_scatter_ce = px.scatter(
            discrepancy_df,
            x='avg_confidence',
            y='avg_entropy',
            size='discrepancy_rate',
            color='experiment',
            title='Average Confidence vs Average Entropy by Experiment',
            labels={
                'avg_confidence': 'Average Confidence (Max Probability)',
                'avg_entropy': 'Average Entropy',
                'discrepancy_rate': 'Discrepancy Rate'
            },
            hover_data=['experiment', 'discrepancy_rate']
        )
        fig_scatter_ce.update_layout(height=500)
        st.plotly_chart(fig_scatter_ce, use_container_width=True)
    
    with tab6:
        st.header("Individual Experiment Analysis")
        
        # Select experiment to view in detail
        selected_exp = st.selectbox(
            "Select experiment for detailed analysis:",
            list(experiments_data.keys())
        )
        
        if selected_exp:
            merged_df, emotion_columns, emotion_mapping, gt_emotion_columns, extra_columns, vad_expectations = experiments_data[selected_exp]
            
            st.subheader(f"Detailed Analysis: {selected_exp}")
            
            # Calculate metrics for this experiment
            accuracy = accuracy_score(merged_df['groundtruth_label'], merged_df['predicted_label'])
            
            # Create confusion matrix
            valid_labels = list(emotion_mapping.keys())
            cm = confusion_matrix(merged_df['groundtruth_label'], merged_df['predicted_label'], 
                                 labels=valid_labels)
            
            # Display key metrics
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
                discrepancy_rate = merged_df['annotation_distribution_discrepancy'].mean()
                st.metric("Discrepancy Rate", f"{discrepancy_rate:.2%}")
            
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
                title=f"Confusion Matrix - {selected_exp}",
                xaxis_title="Predicted Emotion",
                yaxis_title="True Emotion",
                height=500
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Sample distribution comparison for this experiment
            st.subheader("Sample Distribution Analysis")
            
            # Select a sample to view
            sample_options = merged_df['file'].tolist()
            
            if sample_options:
                selected_sample = st.selectbox(
                    "Select a sample for detailed visualization:",
                    sample_options[:20]  # Limit to 20 for performance
                )
                
                if selected_sample:
                    sample_data = merged_df[merged_df['file'] == selected_sample].iloc[0]
                    
                    # Prepare data for visualization
                    pred_probs_sample = []
                    gt_probs_sample = []
                    
                    for col in emotion_columns:
                        pred_val = sample_data[col]
                        gt_val = sample_data[f"{col}_gt"]
                        
                        if isinstance(pred_val, (int, float, np.number)):
                            pred_probs_sample.append(float(pred_val))
                        else:
                            pred_probs_sample.append(0.0)
                            
                        if isinstance(gt_val, (int, float, np.number)):
                            gt_probs_sample.append(float(gt_val))
                        else:
                            gt_probs_sample.append(0.0)
                    
                    pred_probs_sample = np.array(pred_probs_sample, dtype=np.float64)
                    gt_probs_sample = np.array(gt_probs_sample, dtype=np.float64)
                    
                    # Normalize
                    pred_sum = np.sum(pred_probs_sample)
                    gt_sum = np.sum(gt_probs_sample)
                    
                    if pred_sum > 0:
                        pred_probs_sample = pred_probs_sample / pred_sum
                    if gt_sum > 0:
                        gt_probs_sample = gt_probs_sample / gt_sum
                    
                    # Create dataframe for plotting
                    plot_data = pd.DataFrame({
                        'Emotion': list(emotion_mapping.values()),
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
                    
                    fig_comparison.update_layout(
                        title=f"Distribution Comparison - {selected_sample.split('/')[-1]}",
                        xaxis_title="Emotion",
                        yaxis_title="Probability",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab7:
        st.header("Download Results")
        
        # Option to download aggregated metrics
        st.subheader("Aggregated Metrics")
        csv_metrics = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Aggregated Metrics",
            data=csv_metrics,
            file_name="experiment_comparison_metrics.csv",
            mime="text/csv"
        )
        
        # Option to download detailed results for each experiment
        st.subheader("Detailed Results by Experiment")
        
        selected_exp_download = st.selectbox(
            "Select experiment to download:",
            list(experiments_data.keys())
        )
        
        if selected_exp_download:
            merged_df, _, _, _, _, _ = experiments_data[selected_exp_download]
            
            # Select columns to include
            download_columns = [
                'file', 'experiment_name', 'groundtruth_emotion', 'predicted_emotion',
                'gt_distribution_emotion', 'annotation_distribution_discrepancy',
                'js_divergence', 'cosine_similarity', 'pearson_correlation',
                'gt_max_probability', 'gt_distribution_entropy'
            ]
            
            # Add VAD columns if available
            for col in ['valence_gt', 'arousal_gt', 'dominance_gt', 'vad_consistency_score']:
                if col in merged_df.columns:
                    download_columns.append(col)
            
            download_df = merged_df[download_columns].copy()
            csv_detailed = download_df.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Download {selected_exp_download} Results",
                data=csv_detailed,
                file_name=f"{selected_exp_download.replace(' ', '_')}_detailed_results.csv",
                mime="text/csv"
            )
        
        # Option to download all experiments combined
        st.subheader("All Experiments Combined")
        
        # Combine all experiments
        all_experiments_df = pd.DataFrame()
        for exp_name, exp_data in experiments_data.items():
            merged_df, _, _, _, _, _ = exp_data
            if merged_df is not None:
                # Select key columns
                cols_to_include = [
                    'file', 'experiment_name', 'groundtruth_emotion', 'predicted_emotion',
                    'js_divergence', 'cosine_similarity'
                ]
                # Only include columns that exist
                cols_to_include = [col for col in cols_to_include if col in merged_df.columns]
                
                exp_df_subset = merged_df[cols_to_include].copy()
                all_experiments_df = pd.concat([all_experiments_df, exp_df_subset], ignore_index=True)
        
        if not all_experiments_df.empty:
            csv_all = all_experiments_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Experiments Combined",
                data=csv_all,
                file_name="all_experiments_combined.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        **Summary:**
        - **Total Experiments**: {len(experiments_data)}
        - **Total Samples**: {sum([len(exp_data[0]) for exp_data in experiments_data.values() if exp_data[0] is not None])}
        - **Best Accuracy**: {metrics_df['accuracy'].max():.2%} ({metrics_df.loc[metrics_df['accuracy'].idxmax(), 'experiment_name']})
        - **Average Accuracy**: {metrics_df['accuracy'].mean():.2%}
        
        **Notes:**
        - **Correct Mapping**: 0=neutral, 1=happy, 2=sad, 3=surprised, 4=fearful, 5=disgusted, 6=angry, 7=contempt
        - **Jensen-Shannon Divergence**: Measure of similarity between distributions (0 = identical, 1 = maximally different)
        - **Cosine Similarity**: Measures similarity in direction of probability vectors (1 = same direction, 0 = orthogonal)
        - **Annotation-Distribution Discrepancy**: Compares annotated label with most probable emotion from ground truth distribution
        """
    )
    
elif groundtruth_file is None:
    st.warning("⚠️ Please upload the Ground Truth file")
elif not prediction_files:
    st.warning("⚠️ Please upload at least one Prediction file")
else:
    st.info("👈 Please upload both Ground Truth and Prediction files to begin analysis")
    
    # Instructions
    with st.expander("ℹ️ Usage Instructions and File Formats"):
        st.markdown("""
        ## 📋 Expected File Formats
        
        ### 1. Ground Truth File (single):
        ```
        happy,contempt,surprised,angry,disgusted,fearful,sad,neutral,valence,arousal,dominance,emotion_label,file
        2.7970681326650038e-09,0.11136697977781296,0.004586424678564072,... ,-0.469415,0.817847,0.21547260880470276,6,/path/to/image.jpg
        ```
        
        ### 2. Prediction Files (multiple):
        ```
        happy,contempt,surprised,angry,disgusted,fearful,sad,neutral,emotion_label,file
        0.04905860126018524,0.08107654750347137,0.07898678630590439,... ,2,/path/to/image.jpg
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
        
        ## 🔍 Multi-Experiment Analysis Includes:
        
        1. **Experiment Comparison:**
           - Aggregated metrics table
           - Bar chart comparison
           - Radar chart for comprehensive comparison
        
        2. **Performance Metrics:**
           - Accuracy comparison with confidence intervals
           - Statistical analysis
        
        3. **Per-Emotion Analysis:**
           - Heatmap of accuracy by emotion across experiments
           - Best and worst performing emotions
        
        4. **Distribution Metrics:**
           - Comparison of distribution similarity metrics
           - Box plots and violin plots
        
        5. **Discrepancy Analysis:**
           - Annotation-distribution discrepancy comparison
           - Confidence vs entropy analysis
        
        6. **Individual Experiment Analysis:**
           - Detailed view of each experiment
           - Confusion matrices
           - Sample-level analysis
        
        7. **Download Results:**
           - Download aggregated metrics
           - Download detailed results per experiment
           - Download all experiments combined
        """)