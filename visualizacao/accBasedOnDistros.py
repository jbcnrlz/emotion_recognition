import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configuração da página
st.set_page_config(
    page_title="Emotion Recognition Analysis",
    page_icon="😊",
    layout="wide"
)

def calculate_ccc(y_true, y_pred):
    """Calculates Concordance Correlation Coefficient"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Ensure they are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values if any exist
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 0.0
        
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    covariance = np.cov(y_true, y_pred)[0, 1]
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    if (var_true + var_pred + (mean_true - mean_pred) ** 2) == 0:
        return 1.0  # Perfect concordance
    
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def load_true_labels(labels_file, emotion_columns):
    """Load true labels from file in the same format as estimates"""
    try:
        labels_df = pd.read_csv(labels_file)
        
        # Verify the format has the required columns
        if 'file' not in labels_df.columns:
            st.error("Labels file must contain 'file' column")
            return None
        
        # Check if we have emotion probability columns
        has_emotion_probs = all(emotion in labels_df.columns for emotion in emotion_columns)
        
        if not has_emotion_probs:
            # Check if we have an 'emotion' column with indices
            if 'emotion' in labels_df.columns:
                st.info("Converting emotion indices to probability distributions...")
                # Convert emotion indices to one-hot probability distributions
                emotion_probs = []
                for idx in labels_df['emotion']:
                    prob_vector = np.zeros(len(emotion_columns))
                    try:
                        idx_int = int(idx)
                        if 0 <= idx_int < len(emotion_columns):
                            prob_vector[idx_int] = 1.0
                        emotion_probs.append(prob_vector)
                    except (ValueError, TypeError):
                        # If conversion fails, use uniform distribution
                        prob_vector = np.ones(len(emotion_columns)) / len(emotion_columns)
                        emotion_probs.append(prob_vector)
                
                # Add probability columns to dataframe
                for i, emotion in enumerate(emotion_columns):
                    labels_df[emotion] = [prob[i] for prob in emotion_probs]
            else:
                st.error("Labels file must contain either emotion probability columns or 'emotion' column with indices")
                return None
        
        return labels_df
    
    except Exception as e:
        st.error(f"Error loading labels file: {str(e)}")
        return None

def safe_kl_divergence(true_probs, pred_probs):
    """Safely calculate KL divergence with error handling"""
    try:
        # Ensure numpy arrays
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Add small epsilon to avoid log(0) and division by zero
        epsilon = 1e-12
        true_probs = np.clip(true_probs, epsilon, 1.0)
        pred_probs = np.clip(pred_probs, epsilon, 1.0)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Calculate KL divergence
        kl = np.sum(true_probs * np.log(true_probs / pred_probs))
        
        # Handle edge cases
        if np.isnan(kl) or np.isinf(kl):
            return 0.0
            
        return kl
        
    except Exception as e:
        return 0.0

def calculate_chebyshev_distance(true_probs, pred_probs):
    """Calculates Chebyshev distance between two probability distributions"""
    try:
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Chebyshev distance is the maximum absolute difference
        chebyshev = np.max(np.abs(true_probs - pred_probs))
        
        return chebyshev
        
    except Exception as e:
        return 0.0

def calculate_clark_distance(true_probs, pred_probs):
    """Calculates Clark distance between two probability distributions"""
    try:
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Clark distance formula
        numerator = np.abs(true_probs - pred_probs)
        denominator = np.abs(true_probs + pred_probs)
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-12, denominator)
        
        clark = np.sqrt(np.sum((numerator / denominator) ** 2))
        
        return clark
        
    except Exception as e:
        return 0.0

def calculate_canberra_distance(true_probs, pred_probs):
    """Calculates Canberra distance between two probability distributions"""
    try:
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Canberra distance formula
        numerator = np.abs(true_probs - pred_probs)
        denominator = np.abs(true_probs) + np.abs(pred_probs)
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-12, denominator)
        
        canberra = np.sum(numerator / denominator)
        
        return canberra
        
    except Exception as e:
        return 0.0

def calculate_cosine_similarity(true_probs, pred_probs):
    """Calculates Cosine similarity between two probability distributions"""
    try:
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Cosine similarity formula
        dot_product = np.sum(true_probs * pred_probs)
        norm_true = np.sqrt(np.sum(true_probs ** 2))
        norm_pred = np.sqrt(np.sum(pred_probs ** 2))
        
        # Avoid division by zero
        if norm_true == 0 or norm_pred == 0:
            return 0.0
            
        cosine_sim = dot_product / (norm_true * norm_pred)
        
        return cosine_sim
        
    except Exception as e:
        return 0.0

def calculate_intersection_similarity(true_probs, pred_probs):
    """Calculates Intersection similarity between two probability distributions"""
    try:
        true_probs = np.array(true_probs, dtype=np.float64)
        pred_probs = np.array(pred_probs, dtype=np.float64)
        
        # Normalize to ensure they are proper probability distributions
        true_probs = true_probs / np.sum(true_probs)
        pred_probs = pred_probs / np.sum(pred_probs)
        
        # Intersection similarity is the sum of minimums
        intersection = np.sum(np.minimum(true_probs, pred_probs))
        
        return intersection
        
    except Exception as e:
        return 0.0

def calculate_arousal_valence_metrics(estimates_df, labels_df, emotion_columns, emotion_labels):
    """Calculates arousal and valence metrics for each face"""
    results = {
        'valence_ccc': [],
        'arousal_ccc': [],
        'valence_pearson': [],
        'arousal_pearson': [],
        'valence_spearman': [],
        'arousal_spearman': [],
        'emotion_accuracy': [],
        'file_names': [],
        'per_emotion_accuracy': {emotion: [] for emotion in emotion_columns}
    }
    
    # Emotion to arousal and valence mapping (theoretical values)
    emotion_av_mapping = {
        'happy': {'valence': 0.8, 'arousal': 0.6},
        'sad': {'valence': 0.2, 'arousal': 0.3},
        'angry': {'valence': 0.3, 'arousal': 0.8},
        'fearful': {'valence': 0.2, 'arousal': 0.9},
        'surprised': {'valence': 0.6, 'arousal': 0.8},
        'disgusted': {'valence': 0.1, 'arousal': 0.7},
        'contempt': {'valence': 0.3, 'arousal': 0.4},
        'neutral': {'valence': 0.5, 'arousal': 0.5}
    }
    
    missing_files = 0
    processed_files = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create mapping from filename to label row for efficient lookup
    labels_dict = {}
    for idx, row in labels_df.iterrows():
        labels_dict[row['file']] = row
    
    for index, estimates_row in estimates_df.iterrows():
        try:
            image_path = estimates_row['file']
            
            # Find corresponding true label
            if image_path not in labels_dict:
                # Try to find by filename only (without path)
                filename_only = Path(image_path).name
                matching_files = [f for f in labels_dict.keys() if Path(f).name == filename_only]
                if matching_files:
                    true_row = labels_dict[matching_files[0]]
                else:
                    missing_files += 1
                    continue
            else:
                true_row = labels_dict[image_path]
            
            # Calculate predicted valence and arousal from estimates
            pred_valence = 0.0
            pred_arousal = 0.0
            total_prob_pred = 0.0
            
            for emotion in emotion_columns:
                prob = estimates_row[emotion]
                if emotion in emotion_av_mapping:
                    pred_valence += prob * emotion_av_mapping[emotion]['valence']
                    pred_arousal += prob * emotion_av_mapping[emotion]['arousal']
                    total_prob_pred += prob
            
            if total_prob_pred > 0:
                pred_valence /= total_prob_pred
                pred_arousal /= total_prob_pred
            
            # Calculate true valence and arousal from labels
            true_valence = 0.0
            true_arousal = 0.0
            total_prob_true = 0.0
            
            for emotion in emotion_columns:
                prob = true_row[emotion]
                if emotion in emotion_av_mapping:
                    true_valence += prob * emotion_av_mapping[emotion]['valence']
                    true_arousal += prob * emotion_av_mapping[emotion]['arousal']
                    total_prob_true += prob
            
            if total_prob_true > 0:
                true_valence /= total_prob_true
                true_arousal /= total_prob_true
            
            # Calculate metrics for this face
            results['valence_ccc'].append((true_valence, pred_valence))
            results['arousal_ccc'].append((true_arousal, pred_arousal))
            
            # Check emotion accuracy
            pred_emotion_probs = estimates_row[emotion_columns].values
            true_emotion_probs = true_row[emotion_columns].values
            
            predicted_emotion = emotion_columns[np.argmax(pred_emotion_probs)]
            true_emotion = emotion_columns[np.argmax(true_emotion_probs)]
            
            is_correct = 1 if predicted_emotion == true_emotion else 0
            results['emotion_accuracy'].append(is_correct)
            results['file_names'].append(image_path)
            
            # Track per-emotion accuracy
            results['per_emotion_accuracy'][true_emotion].append(is_correct)
            
            processed_files += 1
            
            # Update progress
            if index % max(1, len(estimates_df) // 100) == 0:
                progress = processed_files / len(estimates_df)
                progress_bar.progress(progress)
                status_text.text(f"Processing files... {processed_files}/{len(estimates_df)}")
            
        except Exception as e:
            missing_files += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results, missing_files

def calculate_distribution_metrics(estimates_df, labels_df, emotion_columns, emotion_labels):
    """Calculates metrics between probability distributions"""
    all_true_emotions = []
    all_pred_emotions = []
    all_true_probs = []
    all_pred_probs = []
    
    # Create mapping from filename to label row for efficient lookup
    labels_dict = {}
    for idx, row in labels_df.iterrows():
        labels_dict[row['file']] = row
    
    for index, estimates_row in estimates_df.iterrows():
        try:
            image_path = estimates_row['file']
            
            # Find corresponding true label
            if image_path not in labels_dict:
                # Try to find by filename only (without path)
                filename_only = Path(image_path).name
                matching_files = [f for f in labels_dict.keys() if Path(f).name == filename_only]
                if matching_files:
                    true_row = labels_dict[matching_files[0]]
                else:
                    continue
            else:
                true_row = labels_dict[image_path]
            
            # Ensure probabilities are floats
            pred_emotion_probs = np.array(estimates_row[emotion_columns].values, dtype=np.float64)
            true_emotion_probs = np.array(true_row[emotion_columns].values, dtype=np.float64)
            
            predicted_emotion = emotion_columns[np.argmax(pred_emotion_probs)]
            true_emotion = emotion_columns[np.argmax(true_emotion_probs)]
            
            all_true_emotions.append(true_emotion)
            all_pred_emotions.append(predicted_emotion)
            all_true_probs.append(true_emotion_probs)
            all_pred_probs.append(pred_emotion_probs)
            
        except Exception as e:
            continue
    
    # Classification metrics
    accuracy = np.mean([1 if true == pred else 0 for true, pred in zip(all_true_emotions, all_pred_emotions)])
    f1_macro = f1_score(all_true_emotions, all_pred_emotions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true_emotions, all_pred_emotions, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_true_emotions, all_pred_emotions, labels=emotion_columns)
    
    # Calculate EMD (Earth Mover's Distance) between distributions
    emd_scores = []
    for true_probs, pred_probs in zip(all_true_probs, all_pred_probs):
        try:
            # EMD simplified for discrete distributions
            emd = np.sum(np.abs(np.cumsum(true_probs) - np.cumsum(pred_probs)))
            emd_scores.append(emd)
        except:
            emd_scores.append(0.0)
    
    avg_emd = np.mean(emd_scores) if emd_scores else 0
    
    # Calculate KL divergence safely
    kl_scores = []
    for true_probs, pred_probs in zip(all_true_probs, all_pred_probs):
        kl = safe_kl_divergence(true_probs, pred_probs)
        kl_scores.append(kl)
    
    avg_kl = np.mean(kl_scores) if kl_scores else 0
    
    # Calculate new distance metrics
    chebyshev_scores = []
    clark_scores = []
    canberra_scores = []
    
    # Calculate new similarity metrics
    cosine_scores = []
    intersection_scores = []
    
    for true_probs, pred_probs in zip(all_true_probs, all_pred_probs):
        chebyshev = calculate_chebyshev_distance(true_probs, pred_probs)
        clark = calculate_clark_distance(true_probs, pred_probs)
        canberra = calculate_canberra_distance(true_probs, pred_probs)
        cosine = calculate_cosine_similarity(true_probs, pred_probs)
        intersection = calculate_intersection_similarity(true_probs, pred_probs)
        
        chebyshev_scores.append(chebyshev)
        clark_scores.append(clark)
        canberra_scores.append(canberra)
        cosine_scores.append(cosine)
        intersection_scores.append(intersection)
    
    avg_chebyshev = np.mean(chebyshev_scores) if chebyshev_scores else 0
    avg_clark = np.mean(clark_scores) if clark_scores else 0
    avg_canberra = np.mean(canberra_scores) if canberra_scores else 0
    avg_cosine = np.mean(cosine_scores) if cosine_scores else 0
    avg_intersection = np.mean(intersection_scores) if intersection_scores else 0
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'avg_emd': avg_emd,
        'avg_kl': avg_kl,
        'avg_chebyshev': avg_chebyshev,
        'avg_clark': avg_clark,
        'avg_canberra': avg_canberra,
        'avg_cosine': avg_cosine,
        'avg_intersection': avg_intersection,
        'true_emotions': all_true_emotions,
        'pred_emotions': all_pred_emotions,
        'true_probs': all_true_probs,
        'pred_probs': all_pred_probs,
        'all_chebyshev': chebyshev_scores,
        'all_clark': clark_scores,
        'all_canberra': canberra_scores,
        'all_cosine': cosine_scores,
        'all_intersection': intersection_scores
    }

def main():
    st.title("😊 Emotion Recognition Analysis")
    st.markdown("---")
    
    # Sidebar for file uploads
    st.sidebar.header("Settings")
    
    uploaded_estimates = st.sidebar.file_uploader(
        "📁 Upload Estimates CSV", 
        type=['csv'],
        help="CSV file containing emotion probabilities from the model"
    )
    
    uploaded_labels = st.sidebar.file_uploader(
        "📁 Upload True Labels CSV", 
        type=['csv'],
        help="CSV file with true emotion probabilities (same format as estimates)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Options")
    
    generate_plots = st.sidebar.checkbox("📊 Generate plots", value=True)
    show_details = st.sidebar.checkbox("🔍 Show file details", value=False)
    show_per_emotion = st.sidebar.checkbox("🎭 Show per-emotion metrics", value=True)
    show_distribution_metrics = st.sidebar.checkbox("📏 Show distribution metrics", value=True)
    
    # Emotion columns
    emotion_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    if uploaded_estimates is not None and uploaded_labels is not None:
        try:
            # Load data
            estimates_df = pd.read_csv(uploaded_estimates)
            labels_df = load_true_labels(uploaded_labels, emotion_columns)
            
            if labels_df is None:
                st.stop()
            
            # Show basic dataset information
            st.subheader("📋 Dataset Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Estimates Samples", len(estimates_df))
            
            with col2:
                st.metric("Labels Samples", len(labels_df))
            
            with col3:
                st.metric("Emotion Columns", len(emotion_columns))
            
            with col4:
                common_files = len(set(estimates_df['file']).intersection(set(labels_df['file'])))
                st.metric("Common Files", common_files)
            
            # Data preview
            with st.expander("👀 Estimates Data Preview"):
                st.dataframe(estimates_df.head(10))
            
            with st.expander("👀 Labels Data Preview"):
                st.dataframe(labels_df.head(10))
            
            # Check data quality
            with st.expander("🔍 Data Quality Check"):
                st.write("**Estimates Summary:**")
                st.write(estimates_df[emotion_columns].describe())
                st.write("**Labels Summary:**")
                st.write(labels_df[emotion_columns].describe())
            
            # Calculate metrics
            if st.button("🚀 Calculate Metrics", type="primary"):
                st.markdown("---")
                
                # Container for main metrics
                with st.container():
                    st.subheader("📈 Main Metrics")
                    
                    # Calculate all metrics
                    with st.spinner("Calculating arousal and valence metrics..."):
                        av_results, missing_files = calculate_arousal_valence_metrics(estimates_df, labels_df, emotion_columns, {})
                    
                    with st.spinner("Calculating distribution metrics..."):
                        dist_results = calculate_distribution_metrics(estimates_df, labels_df, emotion_columns, {})
                    
                    if len(av_results['valence_ccc']) == 0:
                        st.error("No files could be processed. Please check if the file paths match between estimates and labels.")
                        return
                    
                    # Calculate aggregate CCC for arousal and valence
                    valence_true = [x[0] for x in av_results['valence_ccc']]
                    valence_pred = [x[1] for x in av_results['valence_ccc']]
                    arousal_true = [x[0] for x in av_results['arousal_ccc']]
                    arousal_pred = [x[1] for x in av_results['arousal_ccc']]
                    
                    valence_ccc = calculate_ccc(valence_true, valence_pred)
                    arousal_ccc = calculate_ccc(arousal_true, arousal_pred)
                    
                    try:
                        valence_pearson = pearsonr(valence_true, valence_pred)[0] if len(valence_true) > 1 else 0
                        arousal_pearson = pearsonr(arousal_true, arousal_pred)[0] if len(arousal_true) > 1 else 0
                    except:
                        valence_pearson = 0
                        arousal_pearson = 0
                    
                    try:
                        valence_spearman = spearmanr(valence_true, valence_pred)[0] if len(valence_true) > 1 else 0
                        arousal_spearman = spearmanr(arousal_true, arousal_pred)[0] if len(arousal_true) > 1 else 0
                    except:
                        valence_spearman = 0
                        arousal_spearman = 0
                    
                    # Show metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("CCC Valence", f"{valence_ccc:.4f}")
                        st.metric("CCC Arousal", f"{arousal_ccc:.4f}")
                        st.metric("Accuracy", f"{dist_results['accuracy']:.4f}")
                    
                    with col2:
                        st.metric("Pearson Valence", f"{valence_pearson:.4f}")
                        st.metric("Pearson Arousal", f"{arousal_pearson:.4f}")
                        st.metric("F1-Score Macro", f"{dist_results['f1_macro']:.4f}")
                    
                    with col3:
                        st.metric("Spearman Valence", f"{valence_spearman:.4f}")
                        st.metric("Spearman Arousal", f"{arousal_spearman:.4f}")
                        st.metric("Avg EMD", f"{dist_results['avg_emd']:.4f}")
                    
                    # Additional distribution metrics
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Avg KL Divergence", f"{dist_results['avg_kl']:.4f}")
                    with col5:
                        st.metric("F1-Score Weighted", f"{dist_results['f1_weighted']:.4f}")
                
                # Distribution Metrics Section
                if show_distribution_metrics:
                    st.markdown("---")
                    st.subheader("📏 Distribution Metrics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Chebyshev Distance", f"{dist_results['avg_chebyshev']:.4f}",
                                help="Maximum absolute difference between distributions")
                    
                    with col2:
                        st.metric("Clark Distance", f"{dist_results['avg_clark']:.4f}",
                                help="Square root of sum of squared relative differences")
                    
                    with col3:
                        st.metric("Canberra Distance", f"{dist_results['avg_canberra']:.4f}",
                                help="Sum of absolute differences divided by sum of absolute values")
                    
                    with col4:
                        st.metric("Cosine Similarity", f"{dist_results['avg_cosine']:.4f}",
                                help="Cosine of angle between distribution vectors")
                    
                    with col5:
                        st.metric("Intersection Similarity", f"{dist_results['avg_intersection']:.4f}",
                                help="Sum of minimum values between distributions")
                
                # Per-emotion metrics
                if show_per_emotion and av_results['per_emotion_accuracy']:
                    st.markdown("---")
                    st.subheader("🎭 Per-Emotion Accuracy")
                    
                    per_emotion_acc = {}
                    per_emotion_count = {}
                    for emotion, acc_list in av_results['per_emotion_accuracy'].items():
                        if acc_list:  # Only include emotions with samples
                            per_emotion_acc[emotion] = np.mean(acc_list)
                            per_emotion_count[emotion] = len(acc_list)
                    
                    if per_emotion_acc:
                        # Display as metrics
                        cols = st.columns(len(per_emotion_acc))
                        for (emotion, acc), col in zip(per_emotion_acc.items(), cols):
                            with col:
                                count = per_emotion_count[emotion]
                                st.metric(
                                    f"{emotion.capitalize()} Accuracy", 
                                    f"{acc:.4f}",
                                    help=f"Based on {count} samples"
                                )
                
                # Plots
                if generate_plots:
                    st.markdown("---")
                    st.subheader("📊 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confusion matrix
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(dist_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                   xticklabels=emotion_columns, yticklabels=emotion_columns, ax=ax)
                        ax.set_title('Confusion Matrix')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        st.pyplot(fig)
                    
                    with col2:
                        # Valence scatter plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(valence_true, valence_pred, alpha=0.6)
                        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
                        ax.set_xlabel('True Valence')
                        ax.set_ylabel('Predicted Valence')
                        ax.set_title(f'Valence (CCC = {valence_ccc:.3f})')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Arousal scatter plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(arousal_true, arousal_pred, alpha=0.6)
                    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
                    ax.set_xlabel('True Arousal')
                    ax.set_ylabel('Predicted Arousal')
                    ax.set_title(f'Arousal (CCC = {arousal_ccc:.3f})')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Distribution metrics comparison
                    st.markdown("#### Distribution Metrics Comparison")
                    
                    # Create a dataframe for distribution metrics
                    dist_metrics_data = {
                        'Metric': ['Chebyshev', 'Clark', 'Canberra', 'KL Divergence', 'EMD', 'Cosine', 'Intersection'],
                        'Value': [
                            dist_results['avg_chebyshev'],
                            dist_results['avg_clark'],
                            dist_results['avg_canberra'],
                            dist_results['avg_kl'],
                            dist_results['avg_emd'],
                            dist_results['avg_cosine'],
                            dist_results['avg_intersection']
                        ],
                        'Type': ['Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Similarity', 'Similarity']
                    }
                    
                    dist_metrics_df = pd.DataFrame(dist_metrics_data)
                    
                    # Plot distribution metrics
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Distances
                    distances_df = dist_metrics_df[dist_metrics_df['Type'] == 'Distance']
                    ax1.bar(distances_df['Metric'], distances_df['Value'], color='lightcoral', alpha=0.7)
                    ax1.set_title('Distance Metrics (Lower is Better)')
                    ax1.set_ylabel('Value')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Similarities
                    similarities_df = dist_metrics_df[dist_metrics_df['Type'] == 'Similarity']
                    ax2.bar(similarities_df['Metric'], similarities_df['Value'], color='lightgreen', alpha=0.7)
                    ax2.set_title('Similarity Metrics (Higher is Better)')
                    ax2.set_ylabel('Value')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # File details
                if show_details:
                    st.markdown("---")
                    st.subheader("📄 File Details")
                    
                    # Create dataframe with detailed results
                    detailed_results = []
                    for i, file_name in enumerate(av_results['file_names']):
                        detailed_results.append({
                            'File': file_name,
                            'Valence_True': valence_true[i],
                            'Valence_Pred': valence_pred[i],
                            'Arousal_True': arousal_true[i],
                            'Arousal_Pred': arousal_pred[i],
                            'Emotion_Correct': av_results['emotion_accuracy'][i],
                            'Chebyshev_Distance': dist_results['all_chebyshev'][i] if i < len(dist_results['all_chebyshev']) else 0,
                            'Clark_Distance': dist_results['all_clark'][i] if i < len(dist_results['all_clark']) else 0,
                            'Canberra_Distance': dist_results['all_canberra'][i] if i < len(dist_results['all_canberra']) else 0,
                            'Cosine_Similarity': dist_results['all_cosine'][i] if i < len(dist_results['all_cosine']) else 0,
                            'Intersection_Similarity': dist_results['all_intersection'][i] if i < len(dist_results['all_intersection']) else 0
                        })
                    
                    detailed_df = pd.DataFrame(detailed_results)
                    st.dataframe(detailed_df)
                    
                    # Option to download detailed results
                    csv = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detailed Results",
                        data=csv,
                        file_name="detailed_results.csv",
                        mime="text/csv"
                    )
                
                # General statistics
                st.markdown("---")
                st.subheader("📊 General Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Successfully processed files:** {len(av_results['emotion_accuracy'])}")
                    st.info(f"**Files with errors:** {missing_files}")
                    st.info(f"**Success rate:** {(len(av_results['emotion_accuracy']) / len(estimates_df) * 100):.1f}%")
                
                with col2:
                    st.success(f"**Per-face accuracy:** {np.mean(av_results['emotion_accuracy']):.4f}")
                    st.success(f"**Total estimates samples:** {len(estimates_df)}")
                    st.success(f"**Total labels samples:** {len(labels_df)}")
        
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.error("Please check that both files have the same format and contain the required columns.")
    
    else:
        # Initial screen when no files are uploaded
        st.markdown("""
        ## 👋 Welcome to Emotion Recognition Analysis
        
        This tool allows you to analyze performance metrics for emotion recognition models,
        including:
        
        - **Arousal and Valence Metrics** (CCC, Pearson, Spearman)
        - **Classification Metrics** (Accuracy, F1-Score, Confusion Matrix)
        - **Distribution Metrics** (Earth Mover's Distance, KL Divergence)
        - **New Distance Metrics** (Chebyshev, Clark, Canberra)
        - **New Similarity Metrics** (Cosine, Intersection)
        - **Interactive Visualizations**
        
        ### 📁 How to use:
        1. Upload a CSV file with model estimates using the sidebar
        2. Upload a CSV file with true labels (same format as estimates)
        3. Configure the desired analysis options
        4. Click **"Calculate Metrics"** to start the analysis
        
        ### 🎯 Expected CSV format:
        Both files should contain the same columns:
        - `file`: image file paths
        - `happy`, `sad`, `angry`, `fearful`, `surprised`, `disgusted`, `contempt`, `neutral`: emotion probabilities
        """)
        
        # Format example
        example_data = {
            'file': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
            'happy': [0.1, 0.8, 0.2],
            'sad': [0.7, 0.1, 0.1],
            'angry': [0.05, 0.05, 0.6],
            'fearful': [0.05, 0.02, 0.05],
            'surprised': [0.05, 0.01, 0.02],
            'disgusted': [0.02, 0.01, 0.01],
            'contempt': [0.01, 0.0, 0.01],
            'neutral': [0.02, 0.01, 0.01]
        }
        example_df = pd.DataFrame(example_data)
        
        with st.expander("👁️ CSV Format Example"):
            st.dataframe(example_df)

if __name__ == '__main__':
    main()