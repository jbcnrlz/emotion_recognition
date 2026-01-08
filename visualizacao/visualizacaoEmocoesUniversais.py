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
st.title("📊 Facial Emotion Recognition Performance Analysis")
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
        
        # Layout principal - Adding new tab for Annotation-Distribution Discrepancy
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📈 Overview", 
            "🔍 Detailed Analysis", 
            "📊 Distributions", 
            "🎭 Affective Dimensions",
            "🔎 Label-VAD Consistency",
            "⚠️ Annotation-Distribution Discrepancy",            
            "📊 Distribution Metrics",  # Nova aba
            "📋 Complete Data"
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
        with tab4:
            st.header("Affective Dimensions Analysis (VAD)")
            
            if extra_columns:
                st.info(f"📊 Analyzing affective dimensions: {', '.join(extra_columns)}")
                
                # Show VAD dimension statistics
                vad_stats = []
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        # Remove NaN before calculating statistics
                        valid_values = merged_df[f"{dimension}_gt"].dropna()
                        if len(valid_values) > 0:
                            stats = {
                                'Dimension': dimension,
                                'Mean': valid_values.mean(),
                                'Std Dev': valid_values.std(),
                                'Min': valid_values.min(),
                                'Max': valid_values.max()
                            }
                            vad_stats.append(stats)
                
                if vad_stats:
                    stats_df = pd.DataFrame(vad_stats)
                    st.dataframe(stats_df.style.format({
                        'Mean': '{:.4f}',
                        'Std Dev': '{:.4f}',
                        'Min': '{:.4f}',
                        'Max': '{:.4f}'
                    }), use_container_width=True)
                else:
                    st.warning("Could not calculate statistics for VAD dimensions.")
                
                # VAD dimension plots by emotion
                st.subheader("Distribution of Affective Dimensions by Emotion")
                
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        # Remove NaN
                        plot_df = merged_df.dropna(subset=[f'{dimension}_gt', 'groundtruth_emotion']).copy()
                        
                        if len(plot_df) > 0:
                            fig = px.box(
                                plot_df,
                                x='groundtruth_emotion',
                                y=f'{dimension}_gt',
                                title=f'Distribution of {dimension} by Emotion',
                                points='all',
                                color='groundtruth_emotion'
                            )
                            fig.update_layout(
                                xaxis_title="Emotion",
                                yaxis_title=dimension,
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Insufficient data to plot {dimension}.")
                
                # Correlation between VAD dimensions and model metrics
                st.subheader("Correlation between Affective Dimensions and Model Performance")
                
                # Calculate correlations
                correlation_data = []
                for dimension in extra_columns:
                    if f"{dimension}_gt" in merged_df.columns:
                        # Remove NaN before calculating correlation
                        corr_df = merged_df[[f"{dimension}_gt", 'js_divergence', 'cosine_similarity']].dropna()
                        
                        if len(corr_df) > 1:  # Need at least 2 points for correlation
                            corr_js = corr_df[f"{dimension}_gt"].corr(corr_df['js_divergence'])
                            corr_cosine = corr_df[f"{dimension}_gt"].corr(corr_df['cosine_similarity'])
                            correlation_data.append({
                                'Dimension': dimension,
                                'Correlation with JS Divergence': corr_js,
                                'Correlation with Cosine Similarity': corr_cosine
                            })
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    st.dataframe(corr_df.style.format({
                        'Correlation with JS Divergence': '{:.4f}',
                        'Correlation with Cosine Similarity': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Correlation chart
                    fig_corr = px.bar(
                        corr_df.melt(id_vars=['Dimension'], 
                                     value_vars=['Correlation with JS Divergence', 
                                                'Correlation with Cosine Similarity']),
                        x='Dimension',
                        y='value',
                        color='variable',
                        barmode='group',
                        title='Correlation between Affective Dimensions and Model Metrics',
                        labels={'value': 'Correlation', 'variable': 'Metric'}
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Could not calculate correlations due to insufficient data.")
            else:
                st.info("ℹ️ No affective dimensions (valence, arousal, dominance) found in ground truth file.")
        
        with tab5:
            st.header("🔍 Label-VAD Consistency Analysis")
            
            if extra_columns and len(extra_columns) >= 2:  # Need at least valence and arousal
                
                # Show theoretical expectations
                st.subheader("Theoretical VAD Expectations by Emotion")
                expectations_df = pd.DataFrame(vad_expectations).T
                st.dataframe(expectations_df, use_container_width=True)
                
                # 1. VAD means by emotion
                st.subheader("1. Average VAD Values by Emotion (Observed)")
                
                vad_means = []
                for emotion in emotion_mapping.values():
                    subset = merged_df[merged_df['groundtruth_emotion'] == emotion]
                    if len(subset) > 0:
                        means = {'Emotion': emotion}
                        for dim in extra_columns:
                            col_name = f"{dim}_gt"
                            if col_name in subset.columns:
                                means[dim] = subset[col_name].mean()
                        vad_means.append(means)
                
                if vad_means:
                    means_df = pd.DataFrame(vad_means)
                    st.dataframe(means_df.style.format({dim: '{:.4f}' for dim in extra_columns}), use_container_width=True)
                    
                    # Bar chart of means
                    fig_means = go.Figure()
                    for dim in extra_columns:
                        if dim in means_df.columns:
                            fig_means.add_trace(go.Bar(
                                x=means_df['Emotion'],
                                y=means_df[dim],
                                name=dim,
                                text=means_df[dim].round(3),
                                textposition='auto'
                            ))
                    
                    fig_means.update_layout(
                        title='Average VAD Values by Emotion',
                        xaxis_title='Emotion',
                        yaxis_title='Value',
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig_means, use_container_width=True)
                
                # 2. 2D scatter: Valence vs Arousal
                st.subheader("2. Valence vs Arousal Scatter (Colored by Emotion)")
                
                scatter_df = merged_df.dropna(subset=['valence_gt', 'arousal_gt', 'groundtruth_emotion']).copy()
                
                if len(scatter_df) > 0:
                    fig_scatter_2d = px.scatter(
                        scatter_df,
                        x='valence_gt',
                        y='arousal_gt',
                        color='groundtruth_emotion',
                        title='Valence vs Arousal by Emotion',
                        labels={'valence_gt': 'Valence', 'arousal_gt': 'Arousal'},
                        hover_data=['file', 'groundtruth_emotion']
                    )
                    
                    # Add quadrants
                    fig_scatter_2d.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_scatter_2d.add_vline(x=0, line_dash="dash", line_color="gray")
                    
                    fig_scatter_2d.update_layout(height=600)
                    st.plotly_chart(fig_scatter_2d, use_container_width=True)
                    
                    # Outlier analysis based on centroid distance
                    st.subheader("3. Identification of Possible Inconsistencies")
                    
                    # Calculate centroid per emotion
                    outliers_info = []
                    for emotion in scatter_df['groundtruth_emotion'].unique():
                        emotion_data = scatter_df[scatter_df['groundtruth_emotion'] == emotion]
                        if len(emotion_data) > 1:
                            centroid_valence = emotion_data['valence_gt'].mean()
                            centroid_arousal = emotion_data['arousal_gt'].mean()
                            
                            # Calculate Euclidean distance from centroid
                            emotion_data = emotion_data.copy()
                            emotion_data['distance'] = np.sqrt(
                                (emotion_data['valence_gt'] - centroid_valence)**2 + 
                                (emotion_data['arousal_gt'] - centroid_arousal)**2
                            )
                            
                            # Identify outliers (more than 2 standard deviations)
                            threshold = emotion_data['distance'].mean() + 2 * emotion_data['distance'].std()
                            outliers = emotion_data[emotion_data['distance'] > threshold]
                            
                            if len(outliers) > 0:
                                outliers_info.append({
                                    'Emotion': emotion,
                                    'Total Samples': len(emotion_data),
                                    'Possible Inconsistencies': len(outliers),
                                    'Percentage': f"{len(outliers)/len(emotion_data)*100:.1f}%"
                                })
                    
                    if outliers_info:
                        outliers_df = pd.DataFrame(outliers_info)
                        st.dataframe(outliers_df, use_container_width=True)
                        
                        # Show some problematic samples
                        st.subheader("4. Examples of Samples with Possible Inconsistency")
                        
                        all_outliers = []
                        for emotion in scatter_df['groundtruth_emotion'].unique():
                            emotion_data = scatter_df[scatter_df['groundtruth_emotion'] == emotion]
                            if len(emotion_data) > 1:
                                centroid_valence = emotion_data['valence_gt'].mean()
                                centroid_arousal = emotion_data['arousal_gt'].mean()
                                
                                emotion_data = emotion_data.copy()
                                emotion_data['distance'] = np.sqrt(
                                    (emotion_data['valence_gt'] - centroid_valence)**2 + 
                                    (emotion_data['arousal_gt'] - centroid_arousal)**2
                                )
                                
                                threshold = emotion_data['distance'].mean() + emotion_data['distance'].std()
                                outliers = emotion_data[emotion_data['distance'] > threshold]
                                
                                for _, row in outliers.head(3).iterrows():  # Take up to 3 per emotion
                                    all_outliers.append({
                                        'File': row['file'].split('/')[-1],
                                        'Emotion': row['groundtruth_emotion'],
                                        'Valence': row['valence_gt'],
                                        'Arousal': row['arousal_gt'],
                                        'Distance from Centroid': row['distance']
                                    })
                        
                        if all_outliers:
                            outliers_examples_df = pd.DataFrame(all_outliers)
                            st.dataframe(
                                outliers_examples_df.style.format({
                                    'Valence': '{:.4f}',
                                    'Arousal': '{:.4f}',
                                    'Distance from Centroid': '{:.4f}'
                                }),
                                use_container_width=True
                            )
                    else:
                        st.info("No obvious inconsistencies identified based on centroid distance.")
                
                # 3. Consistency analysis using calculated score
                if 'vad_consistency_score' in merged_df.columns:
                    st.subheader("5. Label-VAD Consistency Score")
                    
                    # Score statistics
                    consistency_stats = merged_df['vad_consistency_score'].describe()
                    st.metric("Average Consistency", f"{consistency_stats['mean']:.2%}")
                    
                    # Score distribution
                    fig_consistency = px.histogram(
                        merged_df,
                        x='vad_consistency_score',
                        title='Label-VAD Consistency Score Distribution',
                        labels={'vad_consistency_score': 'Consistency Score'},
                        nbins=20
                    )
                    fig_consistency.update_layout(height=400)
                    st.plotly_chart(fig_consistency, use_container_width=True)
                    
                    # Score by emotion
                    consistency_by_emotion = merged_df.groupby('groundtruth_emotion')['vad_consistency_score'].mean().reset_index()
                    consistency_by_emotion = consistency_by_emotion.sort_values('vad_consistency_score', ascending=False)
                    
                    fig_consistency_emotion = px.bar(
                        consistency_by_emotion,
                        x='groundtruth_emotion',
                        y='vad_consistency_score',
                        title='Consistency Score by Emotion',
                        labels={'vad_consistency_score': 'Consistency Score', 'groundtruth_emotion': 'Emotion'},
                        color='vad_consistency_score',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_consistency_emotion.update_layout(height=400)
                    st.plotly_chart(fig_consistency_emotion, use_container_width=True)
                    
                    # Samples with low consistency
                    low_consistency_threshold = 0.5
                    low_consistency_samples = merged_df[merged_df['vad_consistency_score'] < low_consistency_threshold]
                    
                    if len(low_consistency_samples) > 0:
                        st.subheader(f"6. Samples with Low Consistency (Score < {low_consistency_threshold})")
                        
                        low_consistency_display = low_consistency_samples[['file', 'groundtruth_emotion', 'vad_consistency_score'] + 
                                                                         [f"{dim}_gt" for dim in extra_columns]].copy()
                        low_consistency_display = low_consistency_display.sort_values('vad_consistency_score')
                        
                        st.dataframe(
                            low_consistency_display.head(20).style.format({
                                'vad_consistency_score': '{:.3f}',
                                **{f"{dim}_gt": '{:.4f}' for dim in extra_columns}
                            }),
                            use_container_width=True
                        )
                        
                        st.info(f"Total of {len(low_consistency_samples)} samples ({len(low_consistency_samples)/len(merged_df)*100:.1f}%) with low consistency.")
                    else:
                        st.success("All samples have reasonable consistency between labels and VAD dimensions!")
                
                # 4. Clustering analysis to see if emotions form distinct groups
                st.subheader("7. VAD Dimension Clustering Analysis")
                
                # Prepare data for clustering
                cluster_data = merged_df[['valence_gt', 'arousal_gt', 'groundtruth_emotion']].dropna().copy()
                
                if len(cluster_data) > 10:  # Need sufficient data
                    # Apply KMeans
                    X = cluster_data[['valence_gt', 'arousal_gt']].values
                    
                    # Normalize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply clustering
                    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Add labels to dataframe
                    cluster_data = cluster_data.copy()
                    cluster_data['cluster'] = cluster_labels
                    
                    # Cluster plot
                    fig_clusters = px.scatter(
                        cluster_data,
                        x='valence_gt',
                        y='arousal_gt',
                        color='cluster',
                        title='KMeans Clusters of VAD Dimensions (8 clusters)',
                        labels={'valence_gt': 'Valence', 'arousal_gt': 'Arousal'},
                        hover_data=['groundtruth_emotion']
                    )
                    
                    # Add centroids
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    fig_clusters.add_trace(go.Scatter(
                        x=centroids[:, 0],
                        y=centroids[:, 1],
                        mode='markers',
                        marker=dict(size=15, color='black', symbol='x'),
                        name='Centroids'
                    ))
                    
                    fig_clusters.update_layout(height=600)
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Confusion matrix between clusters and emotions
                    confusion_cluster = pd.crosstab(
                        cluster_data['groundtruth_emotion'],
                        cluster_data['cluster'],
                        normalize='index'
                    )
                    
                    fig_cluster_cm = go.Figure(data=go.Heatmap(
                        z=confusion_cluster.values,
                        x=[f'Cluster {i}' for i in confusion_cluster.columns],
                        y=confusion_cluster.index,
                        colorscale='Viridis',
                        text=confusion_cluster.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorbar_title="Proportion"
                    ))
                    
                    fig_cluster_cm.update_layout(
                        title='Emotion Distribution by Cluster',
                        xaxis_title="Cluster",
                        yaxis_title="Emotion",
                        height=500
                    )
                    
                    st.plotly_chart(fig_cluster_cm, use_container_width=True)
                    
                    # Cluster purity analysis
                    st.subheader("8. Cluster Purity")
                    
                    cluster_purity = []
                    for cluster in sorted(cluster_data['cluster'].unique()):
                        cluster_subset = cluster_data[cluster_data['cluster'] == cluster]
                        total = len(cluster_subset)
                        if total > 0:
                            # Most common emotion in cluster
                            dominant_emotion = cluster_subset['groundtruth_emotion'].mode()[0]
                            dominant_count = (cluster_subset['groundtruth_emotion'] == dominant_emotion).sum()
                            purity = dominant_count / total
                            
                            cluster_purity.append({
                                'Cluster': cluster,
                                'Samples': total,
                                'Dominant Emotion': dominant_emotion,
                                'Purity': purity
                            })
                    
                    if cluster_purity:
                        purity_df = pd.DataFrame(cluster_purity)
                        st.dataframe(purity_df.style.format({'Purity': '{:.2%}'}), use_container_width=True)
                        
                        # Clusters with low purity may indicate inconsistencies
                        low_purity_clusters = purity_df[purity_df['Purity'] < 0.5]
                        if len(low_purity_clusters) > 0:
                            st.warning(f"{len(low_purity_clusters)} clusters have purity < 50%, indicating possible mixing of emotions with similar VAD.")
                
                else:
                    st.warning("Insufficient data for clustering analysis.")
                
            else:
                st.info("ℹ️ At least valence and arousal are required in ground truth for consistency analysis.")
        
        with tab6:
            st.header("⚠️ Annotation-Distribution Discrepancy Analysis")
            st.markdown("""
            This analysis compares the **annotated emotion label** with the **most probable emotion from the distribution** in the ground truth.
            
            **Why this matters:**
            - If there's high discrepancy, annotations might be inconsistent with the actual emotion probabilities
            - High entropy distributions might indicate ambiguous samples
            - This helps identify potential annotation errors or ambiguous cases
            """)
            
            # 1. Overall discrepancy statistics
            st.subheader("1. Overall Discrepancy Statistics")
            
            total_samples = len(merged_df)
            discrepancy_count = merged_df['annotation_distribution_discrepancy'].sum()
            discrepancy_percentage = discrepancy_count / total_samples * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", total_samples)
            
            with col2:
                st.metric("Discrepancy Count", f"{discrepancy_count}")
            
            with col3:
                st.metric("Discrepancy Percentage", f"{discrepancy_percentage:.1f}%")
            
            # 2. Discrepancy by emotion
            st.subheader("2. Discrepancy by Emotion")
            
            discrepancy_by_emotion = []
            for emotion_idx, emotion_name in emotion_mapping.items():
                emotion_samples = merged_df[merged_df['groundtruth_label'] == emotion_idx]
                if len(emotion_samples) > 0:
                    discrepancy_rate = emotion_samples['annotation_distribution_discrepancy'].mean()
                    discrepancy_by_emotion.append({
                        'Emotion': emotion_name,
                        'Total Samples': len(emotion_samples),
                        'Discrepancy Count': emotion_samples['annotation_distribution_discrepancy'].sum(),
                        'Discrepancy Rate': discrepancy_rate
                    })
            
            if discrepancy_by_emotion:
                discrepancy_df = pd.DataFrame(discrepancy_by_emotion)
                discrepancy_df = discrepancy_df.sort_values('Discrepancy Rate', ascending=False)
                
                st.dataframe(discrepancy_df.style.format({
                    'Discrepancy Rate': '{:.2%}'
                }), use_container_width=True)
                
                # Bar chart of discrepancy rates
                fig_discrepancy = px.bar(
                    discrepancy_df,
                    x='Emotion',
                    y='Discrepancy Rate',
                    title='Annotation-Distribution Discrepancy Rate by Emotion',
                    color='Discrepancy Rate',
                    color_continuous_scale='RdYlGn_r',  # Red for high discrepancy
                    text='Discrepancy Rate'
                )
                fig_discrepancy.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig_discrepancy.update_layout(height=400)
                st.plotly_chart(fig_discrepancy, use_container_width=True)
            
            # 3. Confidence and Entropy Analysis
            st.subheader("3. Confidence and Entropy Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_confidence = merged_df['gt_max_probability'].mean()
                st.metric("Avg Max Probability", f"{avg_confidence:.3f}")
            
            with col2:
                avg_entropy = merged_df['gt_distribution_entropy'].mean()
                st.metric("Avg Distribution Entropy", f"{avg_entropy:.3f}")
            
            with col3:
                confidence_discrepancy = merged_df[merged_df['annotation_distribution_discrepancy']]['gt_max_probability'].mean()
                st.metric("Avg Confidence (Discrepant)", f"{confidence_discrepancy:.3f}")
            
            with col4:
                entropy_discrepancy = merged_df[merged_df['annotation_distribution_discrepancy']]['gt_distribution_entropy'].mean()
                st.metric("Avg Entropy (Discrepant)", f"{entropy_discrepancy:.3f}")
            
            # Scatter plot: Confidence vs Entropy colored by discrepancy
            st.subheader("4. Confidence vs Entropy Scatter Plot")
            
            scatter_confidence = merged_df.copy()
            scatter_confidence = scatter_confidence.dropna(subset=['gt_max_probability', 'gt_distribution_entropy'])
            
            if len(scatter_confidence) > 0:
                fig_confidence = px.scatter(
                    scatter_confidence,
                    x='gt_max_probability',
                    y='gt_distribution_entropy',
                    color='annotation_distribution_discrepancy',
                    title='Confidence (Max Probability) vs Entropy',
                    labels={
                        'gt_max_probability': 'Maximum Probability (Confidence)',
                        'gt_distribution_entropy': 'Distribution Entropy (Uncertainty)',
                        'annotation_distribution_discrepancy': 'Has Discrepancy'
                    },
                    hover_data=['file', 'groundtruth_emotion', 'gt_distribution_emotion']
                )
                
                # Add reference lines
                fig_confidence.add_vline(x=0.5, line_dash="dash", line_color="gray", 
                                        annotation_text="50% Confidence")
                fig_confidence.add_hline(y=1.5, line_dash="dash", line_color="gray", 
                                        annotation_text="High Entropy")
                
                fig_confidence.update_layout(height=500)
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # 5. Confusion matrix between annotation and distribution
            st.subheader("5. Confusion Matrix: Annotation vs Distribution")
            
            # Create confusion matrix
            cm_annotation_dist = confusion_matrix(
                merged_df['groundtruth_label'], 
                merged_df['gt_distribution_label'],
                labels=valid_labels
            )
            
            fig_cm_annotation = go.Figure(data=go.Heatmap(
                z=cm_annotation_dist,
                x=[emotion_mapping[i] for i in range(8)],
                y=[emotion_mapping[i] for i in range(8)],
                colorscale='Blues',
                text=cm_annotation_dist,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar_title="Count"
            ))
            
            fig_cm_annotation.update_layout(
                title="Confusion Matrix: Annotated Label vs Distribution Label",
                xaxis_title="Distribution Label (Most Probable)",
                yaxis_title="Annotated Label",
                height=600
            )
            
            st.plotly_chart(fig_cm_annotation, use_container_width=True)
            
            # 6. Most common annotation-distribution discrepancies
            st.subheader("6. Most Common Annotation-Distribution Discrepancies")
            
            # Get samples with discrepancy
            discrepancy_samples = merged_df[merged_df['annotation_distribution_discrepancy']].copy()
            
            if len(discrepancy_samples) > 0:
                # Count discrepancy types
                discrepancy_types = discrepancy_samples.groupby(['groundtruth_emotion', 'gt_distribution_emotion']).size().reset_index(name='count')
                discrepancy_types = discrepancy_types.sort_values('count', ascending=False).head(15)
                
                if not discrepancy_types.empty:
                    fig_discrepancy_types = px.bar(
                        discrepancy_types,
                        x='count',
                        y='groundtruth_emotion',
                        color='gt_distribution_emotion',
                        orientation='h',
                        title='Top Annotation-Distribution Discrepancies (Annotated → Distribution)',
                        labels={'count': 'Frequency', 'groundtruth_emotion': 'Annotated Emotion'}
                    )
                    fig_discrepancy_types.update_layout(height=500)
                    st.plotly_chart(fig_discrepancy_types, use_container_width=True)
                else:
                    st.info("No discrepancy patterns to display.")
                
                # 7. Examples of discrepant samples
                st.subheader("7. Examples of Discrepant Samples")
                
                # Sort by entropy (most uncertain first) or by confidence (least confident first)
                display_discrepant = discrepancy_samples.sort_values('gt_distribution_entropy', ascending=False).head(20)
                
                display_columns = [
                    'file', 'groundtruth_emotion', 'gt_distribution_emotion',
                    'gt_max_probability', 'gt_distribution_entropy'
                ]
                
                # Add probability distribution columns
                display_discrepant_detailed = display_discrepant.copy()
                
                # Extract probabilities for display
                for i, emotion in enumerate(emotion_columns):
                    display_discrepant_detailed[f"{emotion}_prob"] = display_discrepant_detailed[gt_emotion_columns[i]]
                
                # Create a formatted display
                display_data = []
                for _, row in display_discrepant_detailed.iterrows():
                    # Get top 3 probabilities
                    probs = []
                    for i, emotion in enumerate(emotion_columns):
                        probs.append((emotion, row[gt_emotion_columns[i]]))
                    
                    probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)[:3]
                    top_probs_str = ", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in probs_sorted])
                    
                    display_data.append({
                        'File': row['file'].split('/')[-1],
                        'Annotated': row['groundtruth_emotion'],
                        'Distribution': row['gt_distribution_emotion'],
                        'Max Probability': row['gt_max_probability'],
                        'Entropy': row['gt_distribution_entropy'],
                        'Top Probabilities': top_probs_str
                    })
                
                display_df_formatted = pd.DataFrame(display_data)
                st.dataframe(
                    display_df_formatted.style.format({
                        'Max Probability': '{:.3f}',
                        'Entropy': '{:.3f}'
                    }),
                    use_container_width=True
                )
                
                # 8. Download discrepant samples
                st.subheader("8. Download Discrepant Samples")
                
                # Prepare data for download
                download_columns = [
                    'file', 'groundtruth_emotion', 'gt_distribution_emotion',
                    'gt_max_probability', 'gt_distribution_entropy'
                ] + gt_emotion_columns
                
                download_df = discrepancy_samples[download_columns].copy()
                csv_discrepant = download_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Discrepant Samples",
                    data=csv_discrepant,
                    file_name="annotation_distribution_discrepancies.csv",
                    mime="text/csv"
                )
            else:
                st.success("🎉 No discrepancies found! All annotations match the most probable emotion from the distribution.")
        with tab7:
            st.header("📊 Distribution Metrics Analysis")
            st.markdown("""
            Esta seção foca especificamente nas métricas de distribuição que comparam as probabilidades previstas 
            pelo modelo com as distribuições do ground truth.
            
            **Métricas disponíveis:**
            - **Jensen-Shannon Divergence**: Mede a similaridade entre duas distribuições (0 = idênticas, 1 = máximamente diferentes)
            - **KL Divergence (Simétrica)**: Versão simétrica da divergência Kullback-Leibler
            - **Euclidean Distance**: Distância euclidiana entre os vetores de probabilidade
            - **Cosine Similarity**: Similaridade de cosseno entre os vetores (1 = mesma direção, 0 = ortogonais)
            - **Pearson Correlation**: Correlação linear entre as distribuições
            """)
            
            # Garantir que não há colunas duplicadas no merged_df
            if merged_df.columns.duplicated().any():
                st.warning("⚠️ Found duplicate columns in data. Removing duplicates...")
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            
            # 1. Estatísticas descritivas de todas as métricas
            st.subheader("1. Descriptive Statistics of Distribution Metrics")
            
            # Selecionar apenas as métricas de distribuição
            distribution_metrics_cols = ['js_divergence', 'kl_divergence', 'euclidean_distance', 
                                        'cosine_similarity', 'pearson_correlation']
            
            # Verificar se as colunas existem no DataFrame e não são duplicadas
            available_metrics = []
            for col in distribution_metrics_cols:
                if col in merged_df.columns:
                    # Verificar se a coluna não está duplicada
                    col_count = list(merged_df.columns).count(col)
                    if col_count == 1:
                        available_metrics.append(col)
                    else:
                        st.warning(f"Column '{col}' appears {col_count} times. Using first occurrence.")
            
            if not available_metrics:
                st.warning("No distribution metrics found in the data.")
                # Verificar quais colunas realmente temos
                st.write("Available columns:", list(merged_df.columns))
                st.stop()
            
            stats_df = merged_df[available_metrics].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            
            # Renomear as métricas para nomes mais amigáveis
            metric_names = {
                'js_divergence': 'JS Divergence',
                'kl_divergence': 'KL Divergence',
                'euclidean_distance': 'Euclidean Distance',
                'cosine_similarity': 'Cosine Similarity',
                'pearson_correlation': 'Pearson Correlation'
            }
            
            stats_df.index = [metric_names.get(col, col) for col in stats_df.index]
            
            st.dataframe(
                stats_df.style.format({
                    'mean': '{:.4f}',
                    'std': '{:.4f}',
                    'min': '{:.4f}',
                    '25%': '{:.4f}',
                    '50%': '{:.4f}',
                    '75%': '{:.4f}',
                    'max': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # 2. Distribuição das métricas
            st.subheader("2. Distribution of Metrics")
            
            # Seleção de métrica para visualização detalhada
            metric_to_analyze = st.selectbox(
                "Select metric for detailed analysis:",
                available_metrics,
                format_func=lambda x: metric_names.get(x, x)
            )
            
            # Criar subplots para a métrica selecionada
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Histogram', 
                    'Cumulative Distribution',
                    'Box Plot by Emotion',
                    'Violin Plot by Emotion'
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 2.1 Histograma
            metric_data = merged_df[metric_to_analyze].dropna()
            
            if len(metric_data) > 0:
                fig_dist.add_trace(
                    go.Histogram(
                        x=metric_data,
                        nbinsx=50,
                        name='Histogram',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Adicionar linha da média
                mean_val = metric_data.mean()
                fig_dist.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_val:.3f}",
                    annotation_position="top right",
                    row=1, col=1
                )
                
                # 2.2 Distribuição cumulativa
                sorted_vals = np.sort(metric_data)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                
                fig_dist.add_trace(
                    go.Scatter(
                        x=sorted_vals,
                        y=cdf,
                        mode='lines',
                        name='CDF',
                        line=dict(color='darkblue', width=2)
                    ),
                    row=1, col=2
                )
                
                # Adicionar percentis
                for percentile in [25, 50, 75, 90, 95]:
                    pctl_val = np.percentile(sorted_vals, percentile)
                    fig_dist.add_vline(
                        x=pctl_val,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"{percentile}%: {pctl_val:.3f}",
                        annotation_position="bottom right",
                        row=1, col=2
                    )
                
                # 2.3 Box plot por emoção
                for emotion in emotion_mapping.values():
                    emotion_data = merged_df[merged_df['groundtruth_emotion'] == emotion][metric_to_analyze].dropna()
                    if len(emotion_data) > 0:
                        fig_dist.add_trace(
                            go.Box(
                                y=emotion_data,
                                name=emotion,
                                boxpoints='outliers',
                                marker_color='lightcoral'
                            ),
                            row=2, col=1
                        )
                
                # 2.4 Violin plot por emoção
                for emotion in emotion_mapping.values():
                    emotion_data = merged_df[merged_df['groundtruth_emotion'] == emotion][metric_to_analyze].dropna()
                    if len(emotion_data) > 0:
                        fig_dist.add_trace(
                            go.Violin(
                                y=emotion_data,
                                name=emotion,
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor='lightseagreen',
                                opacity=0.6,
                                line_color='black'
                            ),
                            row=2, col=2
                        )
                
                fig_dist.update_layout(
                    height=800,
                    showlegend=False,
                    title_text=f"Detailed Analysis: {metric_names.get(metric_to_analyze, metric_to_analyze)}"
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.warning(f"No data available for {metric_names.get(metric_to_analyze, metric_to_analyze)}")
            
            # 3. Matriz de correlação entre métricas
            st.subheader("3. Correlation Between Distribution Metrics")
            
            # Calcular matriz de correlação apenas com as métricas disponíveis
            corr_matrix = merged_df[available_metrics].corr()
            
            # Plotar heatmap de correlação
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[metric_names.get(col, col) for col in corr_matrix.columns],
                y=[metric_names.get(col, col) for col in corr_matrix.index],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False,
                colorbar_title="Correlation"
            ))
            
            fig_corr.update_layout(
                title="Correlation Matrix of Distribution Metrics",
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Análise das correlações
            st.info("""
            **Interpretação das correlações:**
            - **Cosine Similarity e Pearson Correlation**: Normalmente altamente correlacionados (ambos medem similaridade)
            - **JS/KL Divergence e Euclidean Distance**: Normalmente correlacionados (todos medem distância/divergência)
            - **Similarity vs Divergence metrics**: Normalmente negativamente correlacionados
            """)
            
            # 4. Relação entre métricas e acurácia
            st.subheader("4. Metrics Relationship with Classification Accuracy")
            
            # Criar dataframe com métricas e acurácia
            merged_df['correct'] = merged_df['groundtruth_label'] == merged_df['predicted_label']
            accuracy_by_metric = []
            
            # Analisar cada métrica disponível
            for metric in available_metrics:
                # Separar corretos vs incorretos
                correct_vals = merged_df[merged_df['correct']][metric].dropna()
                incorrect_vals = merged_df[~merged_df['correct']][metric].dropna()
                
                if len(correct_vals) > 0 and len(incorrect_vals) > 0:
                    # Teste t para diferença de médias
                    try:
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(correct_vals, incorrect_vals, equal_var=False)
                        
                        effect_size = 0
                        if (correct_vals.std()**2 + incorrect_vals.std()**2) > 0:
                            effect_size = (correct_vals.mean() - incorrect_vals.mean()) / np.sqrt(
                                (correct_vals.std()**2 + incorrect_vals.std()**2) / 2
                            )
                        
                        accuracy_by_metric.append({
                            'Metric': metric_names.get(metric, metric),
                            'Correct Mean': correct_vals.mean(),
                            'Incorrect Mean': incorrect_vals.mean(),
                            'Difference': correct_vals.mean() - incorrect_vals.mean(),
                            'Effect Size': effect_size,
                            'p-value': p_value,
                            'Significant': p_value < 0.05
                        })
                    except Exception as e:
                        # Se houver erro no teste t, pular esta métrica
                        continue
            
            if accuracy_by_metric:
                accuracy_df = pd.DataFrame(accuracy_by_metric)
                
                # Função para colorir linhas significativas
                def highlight_significant(row):
                    if row['Significant']:
                        return ['background-color: lightgreen'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    accuracy_df.style.format({
                        'Correct Mean': '{:.4f}',
                        'Incorrect Mean': '{:.4f}',
                        'Difference': '{:.4f}',
                        'Effect Size': '{:.3f}',
                        'p-value': '{:.6f}'
                    }).apply(highlight_significant, axis=1),
                    use_container_width=True
                )
                
                # Gráfico de barras comparando médias
                fig_accuracy = go.Figure()
                
                fig_accuracy.add_trace(go.Bar(
                    x=accuracy_df['Metric'],
                    y=accuracy_df['Correct Mean'],
                    name='Correct Predictions',
                    marker_color='green',
                    opacity=0.7
                ))
                
                fig_accuracy.add_trace(go.Bar(
                    x=accuracy_df['Metric'],
                    y=accuracy_df['Incorrect Mean'],
                    name='Incorrect Predictions',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_accuracy.update_layout(
                    title='Average Metric Values by Prediction Accuracy',
                    xaxis_title='Metric',
                    yaxis_title='Average Value',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_accuracy, use_container_width=True)
            else:
                st.warning("Could not calculate accuracy relationships for any metrics.")
            
            # 5. Threshold analysis para cada métrica
            st.subheader("5. Threshold Analysis for Anomaly Detection")
            
            # Selecionar métrica para análise de threshold
            threshold_metric = st.selectbox(
                "Select metric for threshold analysis:",
                available_metrics,
                key='threshold_metric',
                format_func=lambda x: metric_names.get(x, x)
            )
            
            if threshold_metric in merged_df.columns:
                # Slider para definir threshold
                metric_min = merged_df[threshold_metric].min()
                metric_max = merged_df[threshold_metric].max()
                metric_mean = merged_df[threshold_metric].mean()
                metric_std = merged_df[threshold_metric].std()
                
                # Determinar threshold padrão baseado na métrica
                if threshold_metric in ['js_divergence', 'kl_divergence', 'euclidean_distance']:
                    default_threshold = metric_mean + metric_std  # Para métricas de divergência
                else:
                    default_threshold = metric_mean - metric_std  # Para métricas de similaridade
                
                threshold = st.slider(
                    f"Threshold for {metric_names.get(threshold_metric, threshold_metric)}:",
                    min_value=float(metric_min),
                    max_value=float(metric_max),
                    value=float(default_threshold),
                    step=0.01
                )
                
                # Analisar amostras acima/abaixo do threshold
                if threshold_metric in ['js_divergence', 'kl_divergence', 'euclidean_distance']:
                    # Valores altos são ruins
                    problematic = merged_df[merged_df[threshold_metric] > threshold].copy()
                    comparison_text = "above"
                else:
                    # Valores baixos são ruins (para similaridade/correlação)
                    problematic = merged_df[merged_df[threshold_metric] < threshold].copy()
                    comparison_text = "below"
                
                # Estatísticas do threshold
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"Samples {comparison_text} threshold", len(problematic))
                
                with col2:
                    percentage = len(problematic) / len(merged_df) * 100 if len(merged_df) > 0 else 0
                    st.metric("Percentage", f"{percentage:.1f}%")
                
                with col3:
                    accuracy_problematic = accuracy_score(
                        problematic['groundtruth_label'], 
                        problematic['predicted_label']
                    ) if len(problematic) > 0 else 0
                    st.metric("Accuracy in subset", f"{accuracy_problematic:.2%}")
                
                with col4:
                    overall_accuracy = accuracy_score(
                        merged_df['groundtruth_label'], 
                        merged_df['predicted_label']
                    )
                    st.metric("Overall accuracy", f"{overall_accuracy:.2%}")
                
                # Distribuição de emoções nas amostras problemáticas
                if len(problematic) > 0:
                    st.subheader(f"6. Emotion Distribution in {comparison_text.capitalize()} Threshold Samples")
                    
                    emotion_dist = problematic['groundtruth_emotion'].value_counts().reset_index()
                    emotion_dist.columns = ['Emotion', 'Count']
                    emotion_dist['Percentage'] = emotion_dist['Count'] / len(problematic) * 100
                    
                    # Comparar com distribuição geral
                    overall_dist = merged_df['groundtruth_emotion'].value_counts().reset_index()
                    overall_dist.columns = ['Emotion', 'Count']
                    overall_dist['Percentage'] = overall_dist['Count'] / len(merged_df) * 100
                    
                    # Juntar as distribuições
                    comparison_dist = pd.merge(
                        emotion_dist, 
                        overall_dist, 
                        on='Emotion', 
                        suffixes=('_problematic', '_overall')
                    )
                    
                    # Calcular over/under representation
                    comparison_dist['Over_Representation'] = (
                        comparison_dist['Percentage_problematic'] - 
                        comparison_dist['Percentage_overall']
                    )
                    
                    st.dataframe(
                        comparison_dist.style.format({
                            'Percentage_problematic': '{:.1f}%',
                            'Percentage_overall': '{:.1f}%',
                            'Over_Representation': '{:.1f}%'
                        }).apply(
                            lambda x: ['background-color: lightcoral' if x['Over_Representation'] > 5 else 
                                    'background-color: lightgreen' if x['Over_Representation'] < -5 else '' 
                                    for _ in x], axis=1
                        ),
                        use_container_width=True
                    )
                    
                    # Mostrar top amostras problemáticas
                    st.subheader(f"7. Top Problematic Samples ({comparison_text} threshold)")
                    
                    if threshold_metric in ['js_divergence', 'kl_divergence', 'euclidean_distance']:
                        # Ordenar por valores mais altos
                        top_problematic = problematic.nlargest(10, threshold_metric)
                    else:
                        # Ordenar por valores mais baixos
                        top_problematic = problematic.nsmallest(10, threshold_metric)
                    
                    display_cols = [
                        'file', 'groundtruth_emotion', 'predicted_emotion', 
                        threshold_metric, 'js_divergence', 'cosine_similarity',
                        'gt_max_probability', 'gt_distribution_entropy'
                    ]
                    
                    # Filtrar colunas que existem e remover duplicatas
                    existing_cols = []
                    seen = set()
                    for col in display_cols:
                        if col in problematic.columns and col not in seen:
                            existing_cols.append(col)
                            seen.add(col)
                    
                    display_df = top_problematic[existing_cols].copy()
                    display_df['file'] = display_df['file'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else str(x))
                    
                    st.dataframe(
                        display_df.style.format({
                            threshold_metric: '{:.4f}',
                            'js_divergence': '{:.4f}',
                            'cosine_similarity': '{:.4f}',
                            'gt_max_probability': '{:.3f}',
                            'gt_distribution_entropy': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Botão para download
                    csv_problematic = problematic.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Problematic Samples",
                        data=csv_problematic,
                        file_name=f"problematic_samples_{threshold_metric}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No samples found {comparison_text} the threshold of {threshold:.3f}")
            else:
                st.warning(f"Selected metric '{threshold_metric}' not found in data.")
            
            # 8. Análise de métricas por nível de confiança do ground truth
            st.subheader("8. Metrics Analysis by Ground Truth Confidence")
            
            # Verificar se gt_max_probability existe
            if 'gt_max_probability' in merged_df.columns:
                # Criar bins de confiança
                merged_df['confidence_bin'] = pd.cut(
                    merged_df['gt_max_probability'],
                    bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                    labels=['Very Low (0-0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 
                            'High (0.7-0.9)', 'Very High (0.9-1.0)']
                )
                
                # Calcular métricas médias por bin de confiança
                confidence_analysis = []
                for metric in available_metrics:
                    for bin_name in merged_df['confidence_bin'].cat.categories:
                        bin_data = merged_df[merged_df['confidence_bin'] == bin_name][metric].dropna()
                        if len(bin_data) > 0:
                            confidence_analysis.append({
                                'Confidence Bin': bin_name,
                                'Metric': metric_names.get(metric, metric),
                                'Mean': bin_data.mean(),
                                'Std': bin_data.std(),
                                'Samples': len(bin_data)
                            })
                
                if confidence_analysis:
                    conf_df = pd.DataFrame(confidence_analysis)
                    
                    # Pivot para heatmap
                    pivot_df = conf_df.pivot(index='Metric', columns='Confidence Bin', values='Mean')
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis',
                        text=np.round(pivot_df.values, 3),
                        texttemplate='%{text}',
                        textfont={"size": 11},
                        hoverongaps=False,
                        colorbar_title="Metric Value"
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Distribution Metrics by Ground Truth Confidence Level",
                        height=500
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Análise de tendências
                    st.info("""
                    **Tendências esperadas:**
                    - **Divergência/Similaridade**: Amostras com alta confiança no ground truth devem ter:
                    - Menor JS/KL Divergence
                    - Maior Cosine Similarity e Pearson Correlation
                    - Se este padrão não for observado, pode indicar problemas no modelo ou nos dados
                    """)
                else:
                    st.warning("Could not calculate metrics by confidence bins.")
            else:
                st.info("Ground truth confidence (gt_max_probability) not available for analysis.")
        with tab8:
            st.header("Complete Data")
            
            # View options
            view_option = st.radio(
                "Select view:",
                ["Combined Data", "Incorrect Samples Only", "Correct Samples Only", "Discrepant Samples Only"]
            )
            
            if view_option == "Incorrect Samples Only":
                display_df = merged_df[merged_df['groundtruth_label'] != merged_df['predicted_label']]
            elif view_option == "Correct Samples Only":
                display_df = merged_df[merged_df['groundtruth_label'] == merged_df['predicted_label']]
            elif view_option == "Discrepant Samples Only":
                display_df = merged_df[merged_df['annotation_distribution_discrepancy']]
            else:
                display_df = merged_df
            
            # Select columns to display
            display_columns = [
                'file', 'groundtruth_emotion', 'predicted_emotion',
                'js_divergence', 'cosine_similarity', 'pearson_correlation'
            ]
            
            # Add annotation-distribution comparison columns
            display_columns.extend(['gt_distribution_emotion', 'annotation_distribution_discrepancy', 
                                   'gt_max_probability', 'gt_distribution_entropy'])
            
            # Add extra ground truth columns
            for col in extra_columns:
                if f"{col}_gt" in merged_df.columns:
                    display_columns.append(f"{col}_gt")
            
            # Add consistency score if available
            if 'vad_consistency_score' in merged_df.columns:
                display_columns.append('vad_consistency_score')
            
            # Add all probability columns if desired
            show_all_columns = st.checkbox("Show all probability columns")
            
            if show_all_columns:
                all_cols = list(merged_df.columns)
                selected_cols = st.multiselect(
                    "Select additional columns:",
                    [col for col in all_cols if col not in display_columns],
                    default=[]
                )
                display_columns.extend(selected_cols)
            
            # Display dataframe
            if not display_df.empty:
                # Format numbers for better display
                display_df_formatted = display_df.copy()
                for col in display_df_formatted.columns:
                    if display_df_formatted[col].dtype in [np.float64, np.float32]:
                        display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "NaN")
                
                st.dataframe(
                    display_df_formatted[display_columns].head(100),  # Limit to 100 rows for performance
                    use_container_width=True
                )
                
                # Download option
                csv = display_df[display_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name="emotion_analysis_filtered.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data to display with selected filters.")
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            **Notes:**
            - **Correct Mapping**: 0=neutral, 1=happy, 2=sad, 3=surprised, 4=fearful, 5=disgusted, 6=angry, 7=contempt
            - **Jensen-Shannon Divergence**: Measure of similarity between distributions (0 = identical, 1 = maximally different)
            - **Cosine Similarity**: Measures similarity in direction of probability vectors (1 = same direction, 0 = orthogonal)
            - **Pearson Correlation**: Measures linear correlation between distributions
            - **VAD Dimensions**: Valence, Arousal, Dominance - only in ground truth
            - **Label-VAD Consistency**: Measures how much emotion labels are consistent with VAD dimensions
            - **Annotation-Distribution Discrepancy**: Compares annotated label with most probable emotion from ground truth distribution
            """
        )
        
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
        
        6. **Interactive Visualizations:**
           - Comparative distribution plots
           - Sample-specific analysis
           - VAD dimension visualization by emotion
        """)