import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import io
from itertools import cycle
import warnings
import plotly.colors
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Multi-Experiment Emotion Distribution Comparison", layout="wide")
st.title("Comparative Analysis: Emotion Distribution Metrics")

# Initialize session state
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'experiment_counter' not in st.session_state:
    st.session_state.experiment_counter = 0
if 'processed_experiments' not in st.session_state:
    st.session_state.processed_experiments = None

# Function to process experiment data
@st.cache_data
def process_experiment_data(preds_file, labels_file, exp_name, emotion_cols=None):
    """Process loaded CSV files for an experiment"""
    try:
        # Read predictions file
        preds = pd.read_csv(preds_file)
        
        # Read ground truth file
        labels = pd.read_csv(labels_file)
        
        # Automatically identify emotion columns if not provided
        if emotion_cols is None:
            emotion_cols = []
            # For predictions: columns that aren't 'file' or end with 'label' or start with VAD
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
        labels_emotions['file'] = labels.get('file', '')
        
        return {
            'name': exp_name,
            'preds_df': preds_emotions,
            'labels_df': labels_emotions,
            'emotion_cols': emotion_cols,
            'n_samples': len(preds_emotions),
            'original_preds': preds,  # Keep original for reference
            'original_labels': labels  # Keep original for reference
        }
    except Exception as e:
        st.error(f"Error processing {exp_name}: {str(e)}")
        return None

# Function to calculate all distribution metrics (raw values)
def calculate_all_metrics(preds, labels, emotion_cols):
    """Calculate all distribution metrics (raw values)"""
    metrics = {
        "kl": {"global": 0, "individual": []},
        "js": {"global": 0, "individual": []},
        "cosine": {"global": 0, "individual": []},
        "euclidean": {"global": 0, "individual": []},
        "pearson": {"global": 0, "individual": []}
    }
    
    for i in range(len(preds)):
        p = preds.iloc[i][emotion_cols].values.astype(float)
        l = labels.iloc[i][emotion_cols].values.astype(float)
        
        # Normalize to probability distribution
        p = p / np.sum(p) if np.sum(p) > 0 else p
        l = l / np.sum(l) if np.sum(l) > 0 else l
        
        # KL Divergence (symmetric)
        epsilon = 1e-10
        p_safe = p + epsilon
        l_safe = l + epsilon
        p_safe = p_safe / np.sum(p_safe)
        l_safe = l_safe / np.sum(l_safe)
        kl1 = entropy(p_safe, l_safe)
        kl2 = entropy(l_safe, p_safe)
        kl_sym = 0.5 * (kl1 + kl2)
        metrics["kl"]["individual"].append(kl_sym if not np.isnan(kl_sym) else 0)
        
        # JS Divergence
        js = jensenshannon(p, l) ** 2
        metrics["js"]["individual"].append(js if not np.isnan(js) else 0)
        
        # Cosine Similarity
        cos_sim = np.dot(p, l) / (np.linalg.norm(p) * np.linalg.norm(l) + 1e-10)
        metrics["cosine"]["individual"].append(cos_sim if not np.isnan(cos_sim) else 0)
        
        # Euclidean Distance
        euc_dist = euclidean(p, l)
        metrics["euclidean"]["individual"].append(euc_dist if not np.isnan(euc_dist) else 0)
        
        # Pearson Correlation
        if len(p) > 1:
            corr = np.corrcoef(p, l)[0, 1]
            metrics["pearson"]["individual"].append(corr if not np.isnan(corr) else 0)
        else:
            metrics["pearson"]["individual"].append(1.0 if p[0] == l[0] else 0.0)
    
    # Calculate global averages
    for metric in metrics:
        if metrics[metric]["individual"]:
            metrics[metric]["global"] = np.mean(metrics[metric]["individual"])
        else:
            metrics[metric]["global"] = 0
    
    return metrics

def calculate_accuracy(preds, labels, emotion_cols, top_k=3):
    """Calculate accuracy based on most probable classes"""
    pred_classes = []
    true_classes = []
    
    for i in range(len(preds)):
        # Most probable class from predictions
        pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
        pred_class = np.argmax(pred_probs)
        pred_classes.append(pred_class)
        
        # Most probable class from ground truth probabilities
        true_probs = labels.iloc[i][emotion_cols].values.astype(float)
        true_class = np.argmax(true_probs)
        true_classes.append(true_class)
    
    accuracy = accuracy_score(true_classes, pred_classes)
    
    # Top-K accuracy
    top_k_correct = 0
    for i in range(len(preds)):
        pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
        true_class = true_classes[i]
        
        # Get indices of top-K probabilities
        top_k_indices = np.argsort(pred_probs)[-top_k:][::-1]
        if true_class in top_k_indices:
            top_k_correct += 1
    
    top_k_accuracy = top_k_correct / len(preds) if len(preds) > 0 else 0
    
    return accuracy, top_k_accuracy, pred_classes, true_classes

def calculate_accuracy_with_annotated_label(preds, labels, emotion_cols, original_labels):
    """Calculate accuracy using annotated label from ground truth"""
    pred_classes = []
    true_classes_annotated = []
    
    for i in range(len(preds)):
        # Most probable class from predictions
        pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
        pred_class = np.argmax(pred_probs)
        pred_classes.append(pred_class)
        
        # Try to get annotated label from original labels file
        if i < len(original_labels):
            # Look for emotion_label column
            if 'emotion_label' in original_labels.columns:
                true_class = original_labels.iloc[i]['emotion_label']
            else:
                # If not found, use most probable from ground truth
                true_probs = labels.iloc[i][emotion_cols].values.astype(float)
                true_class = np.argmax(true_probs)
            
            # Convert to integer if needed
            if isinstance(true_class, str):
                try:
                    true_class = int(float(true_class))
                except:
                    true_probs = labels.iloc[i][emotion_cols].values.astype(float)
                    true_class = np.argmax(true_probs)
            else:
                true_class = int(true_class)
        else:
            # If index out of bounds, use most probable
            true_probs = labels.iloc[i][emotion_cols].values.astype(float)
            true_class = np.argmax(true_probs)
        
        true_classes_annotated.append(true_class)
    
    accuracy_annotated = accuracy_score(true_classes_annotated, pred_classes) if true_classes_annotated else 0
    
    return accuracy_annotated, pred_classes, true_classes_annotated

def calculate_rank_metrics(preds, labels, emotion_cols, true_classes):
    """Calculate rank-based metrics"""
    rank_positions = []
    
    for i in range(len(preds)):
        pred_probs = preds.iloc[i][emotion_cols].values.astype(float)
        true_class = true_classes[i]
        
        # Sort probabilities in descending order
        sorted_indices = np.argsort(pred_probs)[::-1]
        
        # Find position of true class
        rank_position = np.where(sorted_indices == true_class)[0]
        if len(rank_position) > 0:
            rank_positions.append(rank_position[0] + 1)
    
    mean_rank = np.mean(rank_positions) if rank_positions else 0
    median_rank = np.median(rank_positions) if rank_positions else 0
    
    # Rank distribution
    rank_distribution = {}
    for rank in rank_positions:
        rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
    
    return mean_rank, median_rank, rank_distribution, rank_positions

# Sidebar for configuration
st.sidebar.header("Experiment Configuration")

# Add new experiment
st.sidebar.subheader("Add New Experiment")

with st.sidebar.expander("Add Experiment", expanded=True):
    exp_name = st.text_input(
        "Experiment Name",
        value=f"Experiment_{st.session_state.experiment_counter + 1}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        predictions_file = st.file_uploader(
            "Predictions File",
            type=['csv'],
            key=f"pred_{st.session_state.experiment_counter}"
        )
    
    with col2:
        groundtruth_file = st.file_uploader(
            "Ground Truth File",
            type=['csv'],
            key=f"gt_{st.session_state.experiment_counter}"
        )
    
    if st.button("Add Experiment", type="primary"):
        if predictions_file and groundtruth_file and exp_name:
            exp_data = {
                'name': exp_name,
                'predictions_file': predictions_file,
                'groundtruth_file': groundtruth_file
            }
            st.session_state.experiments.append(exp_data)
            st.session_state.experiment_counter += 1
            st.success(f"Added {exp_name}!")
            st.rerun()

# Show added experiments
if st.session_state.experiments:
    st.sidebar.subheader("Added Experiments")
    
    for i, exp in enumerate(st.session_state.experiments):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"**{exp['name']}**")
            st.caption(f"Pred: {exp['predictions_file'].name[:15]}...")
            st.caption(f"GT: {exp['groundtruth_file'].name[:15]}...")
        
        with col2:
            if st.button("X", key=f"remove_{i}"):
                st.session_state.experiments.pop(i)
                st.rerun()

# Analysis settings
st.sidebar.markdown("---")
st.sidebar.header("Analysis Settings")

# Specify emotion columns (14 emotions)
st.sidebar.subheader("Emotion Columns")
emotion_cols_input = st.sidebar.text_input(
    "Emotion columns (comma-separated)",
    value="happy,contempt,elated,hopeful,surprised,proud,loved,angry,astonished,disgusted,fearful,sad,fatigued,neutral",
    help="List all 14 emotion columns in correct order"
)

if emotion_cols_input:
    emotion_cols = [col.strip() for col in emotion_cols_input.split(',')]
else:
    emotion_cols = None

# Distribution metrics
st.sidebar.subheader("Distribution Metrics")
primary_metric = st.sidebar.selectbox(
    "Primary Metric for Analysis",
    ["KL Divergence", "JS Divergence", "Cosine Similarity", 
     "Euclidean Distance", "Pearson Correlation"]
)

metric_map = {
    "KL Divergence": "kl",
    "JS Divergence": "js",
    "Cosine Similarity": "cosine",
    "Euclidean Distance": "euclidean",
    "Pearson Correlation": "pearson"
}

# Button to process all experiments
st.sidebar.markdown("---")
if st.sidebar.button("Process All Experiments", type="primary"):
    if not st.session_state.experiments:
        st.error("No experiments to process!")
    else:
        processed_experiments = []
        
        for i, exp in enumerate(st.session_state.experiments):
            with st.spinner(f"Processing {exp['name']}..."):
                # Process experiment data
                exp_result = process_experiment_data(
                    exp['predictions_file'],
                    exp['groundtruth_file'],
                    exp['name'],
                    emotion_cols
                )
                
                if exp_result:
                    # Calculate all distribution metrics
                    distribution_metrics = calculate_all_metrics(
                        exp_result['preds_df'],
                        exp_result['labels_df'],
                        exp_result['emotion_cols']
                    )
                    
                    # Calculate accuracy using most probable classes (both from predictions and ground truth)
                    accuracy_most_probable, top_k_accuracy, pred_classes, true_classes_most_probable = calculate_accuracy(
                        exp_result['preds_df'],
                        exp_result['labels_df'],
                        exp_result['emotion_cols'],
                        3
                    )
                    
                    # Calculate accuracy using annotated label (if available)
                    accuracy_annotated, pred_classes_annotated, true_classes_annotated = calculate_accuracy_with_annotated_label(
                        exp_result['preds_df'],
                        exp_result['labels_df'],
                        exp_result['emotion_cols'],
                        exp_result['original_labels']
                    )
                    
                    # Use most probable classes for rank metrics
                    mean_rank, median_rank, rank_dist, rank_pos = calculate_rank_metrics(
                        exp_result['preds_df'],
                        exp_result['labels_df'],
                        exp_result['emotion_cols'],
                        true_classes_most_probable
                    )
                    
                    # Confusion matrix using most probable classes
                    cm = confusion_matrix(
                        true_classes_most_probable, 
                        pred_classes, 
                        labels=range(len(exp_result['emotion_cols']))
                    )
                    
                    # Classification report using most probable classes
                    try:
                        class_report = classification_report(
                            true_classes_most_probable,
                            pred_classes,
                            target_names=exp_result['emotion_cols'],
                            output_dict=True
                        )
                    except:
                        class_report = {}
                    
                    # Store results
                    exp_result.update({
                        'distribution_metrics': distribution_metrics,
                        'primary_metric_value': distribution_metrics[metric_map[primary_metric]]["global"],
                        'primary_metric_individual': distribution_metrics[metric_map[primary_metric]]["individual"],
                        'accuracy_most_probable': accuracy_most_probable,  # Using most probable classes
                        'accuracy_annotated': accuracy_annotated,  # Using annotated label
                        'top_k_accuracy': top_k_accuracy,
                        'mean_rank': mean_rank,
                        'median_rank': median_rank,
                        'rank_positions': rank_pos,
                        'pred_classes': pred_classes,
                        'true_classes_most_probable': true_classes_most_probable,
                        'true_classes_annotated': true_classes_annotated,
                        'confusion_matrix': cm,
                        'classification_report': class_report
                    })
                    
                    processed_experiments.append(exp_result)
                    st.success(f"{exp['name']}: Most Probable Acc = {accuracy_most_probable:.4f}, Annotated Acc = {accuracy_annotated:.4f}")
        
        if processed_experiments:
            st.session_state.processed_experiments = processed_experiments
            st.success(f"{len(processed_experiments)} experiments processed successfully!")
        else:
            st.error("No experiments were successfully processed.")

# Button to clear all experiments
if st.sidebar.button("Clear All Experiments"):
    st.session_state.experiments = []
    st.session_state.experiment_counter = 0
    st.session_state.processed_experiments = None
    st.rerun()

# Main content area
if 'processed_experiments' in st.session_state and st.session_state.processed_experiments:
    experiments = st.session_state.processed_experiments
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Metrics Overview", 
        "Metric Comparison", 
        "Emotion Analysis",
        "Model Ranking",
        "Sample Analysis",
        "Export Results"
    ])
    
    with tab1:
        st.header("Distribution Metrics Overview")
        
        # Summary table
        st.subheader("Distribution Metrics Summary")
        
        summary_data = []
        for exp in experiments:
            metrics = exp['distribution_metrics']
            summary_data.append({
                'Experiment': exp['name'],
                'Samples': exp['n_samples'],
                'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                'Accuracy (Annotated)': exp['accuracy_annotated'],
                'KL Divergence': metrics['kl']['global'],
                'JS Divergence': metrics['js']['global'],
                'Cosine Similarity': metrics['cosine']['global'],
                'Euclidean Distance': metrics['euclidean']['global'],
                'Pearson Correlation': metrics['pearson']['global']
            })

        summary_df = pd.DataFrame(summary_data)
        
        # Sort based on primary metric (lower is better for divergences/distance, higher for similarity/correlation)
        if primary_metric in ["KL Divergence", "JS Divergence", "Euclidean Distance"]:
            summary_df = summary_df.sort_values(primary_metric, ascending=True)
        else:
            summary_df = summary_df.sort_values(primary_metric, ascending=False)

        # Formatting
        styled_df = summary_df.style.format({
            'Accuracy (Most Probable)': '{:.4f}',
            'Accuracy (Annotated)': '{:.4f}',
            'KL Divergence': '{:.4f}',
            'JS Divergence': '{:.4f}',
            'Cosine Similarity': '{:.4f}',
            'Euclidean Distance': '{:.4f}',
            'Pearson Correlation': '{:.4f}'
        })

        st.dataframe(styled_df, use_container_width=True)
        
        # Metric descriptions
        st.markdown("""
        ### Metric Interpretations:
        
        | Metric | Range | Ideal Value | Interpretation |
        |--------|-------|-------------|----------------|
        | **Accuracy (Most Probable)** | 0 to 1 | 1 | Accuracy using most probable class from ground truth |
        | **Accuracy (Annotated)** | 0 to 1 | 1 | Accuracy using annotated label from ground truth |
        | **KL Divergence** | 0 to ∞ | 0 | Measures information loss (lower is better) |
        | **JS Divergence** | 0 to 1 | 0 | Measures distance between distributions (lower is better) |
        | **Cosine Similarity** | -1 to 1 | 1 | Measures angle between vectors (higher is better) |
        | **Euclidean Distance** | 0 to ∞ | 0 | Measures straight-line distance (lower is better) |
        | **Pearson Correlation** | -1 to 1 | 1 | Measures linear correlation (higher is better) |
        
        **Note**: "Most Probable" accuracy uses the class with highest probability from ground truth distribution.
        "Annotated" accuracy uses the manually annotated label from the ground truth file.
        """)
        
        # Comparison of the two accuracies
        if len(experiments) > 1:
            st.subheader("Accuracy Comparison: Most Probable vs Annotated")
            
            # Prepare data for comparison
            acc_comparison_data = []
            for exp in experiments:
                acc_comparison_data.append({
                    'Experiment': exp['name'],
                    'Most Probable': exp['accuracy_most_probable'],
                    'Annotated': exp['accuracy_annotated'],
                    'Difference': exp['accuracy_most_probable'] - exp['accuracy_annotated']
                })
            
            acc_comparison_df = pd.DataFrame(acc_comparison_data)
            acc_comparison_df = acc_comparison_df.sort_values('Difference', ascending=False)
            
            # Create grouped bar chart
            fig_acc_comparison = go.Figure()
            
            for metric in ['Most Probable', 'Annotated']:
                fig_acc_comparison.add_trace(go.Bar(
                    name=metric,
                    x=acc_comparison_df['Experiment'],
                    y=acc_comparison_df[metric],
                    text=np.round(acc_comparison_df[metric], 4),
                    textposition='outside'
                ))
            
            fig_acc_comparison.update_layout(
                barmode='group',
                title='Accuracy Comparison: Most Probable vs Annotated Labels',
                xaxis_title='Experiment',
                yaxis_title='Accuracy',
                yaxis_range=[0, 1],
                height=500
            )
            
            st.plotly_chart(fig_acc_comparison, use_container_width=True)
            
            # Difference analysis
            st.subheader("Accuracy Difference Analysis")
            
            col_diff1, col_diff2, col_diff3 = st.columns(3)
            
            with col_diff1:
                avg_diff = acc_comparison_df['Difference'].mean()
                st.metric("Average Difference", f"{avg_diff:.4f}")
            
            with col_diff2:
                max_diff = acc_comparison_df['Difference'].max()
                max_exp = acc_comparison_df.loc[acc_comparison_df['Difference'].idxmax(), 'Experiment']
                st.metric(f"Max Diff ({max_exp})", f"{max_diff:.4f}")
            
            with col_diff3:
                min_diff = acc_comparison_df['Difference'].min()
                min_exp = acc_comparison_df.loc[acc_comparison_df['Difference'].idxmin(), 'Experiment']
                st.metric(f"Min Diff ({min_exp})", f"{min_diff:.4f}")
            
            st.info("""
            **Interpretation of Accuracy Differences:**
            - **Positive Difference**: Model performs better when using most probable class vs annotated label
            - **Negative Difference**: Model performs better with annotated label vs most probable class
            - **Near Zero**: Annotated label aligns well with most probable class
            """)
    
    with tab2:
        st.header("Distribution Metric Comparison")
        
        # Charts for each metric
        metrics_list = [
            ('KL Divergence', 'kl'),
            ('JS Divergence', 'js'),
            ('Cosine Similarity', 'cosine'),
            ('Euclidean Distance', 'euclidean'),
            ('Pearson Correlation', 'pearson')
        ]
        
        # Add accuracy metrics to the comparison
        metrics_list_with_acc = [
            ('Accuracy (Most Probable)', 'accuracy_most_probable'),
            ('Accuracy (Annotated)', 'accuracy_annotated')
        ] + metrics_list
        
        cols = st.columns(2)
        
        for idx, (metric_name, metric_key) in enumerate(metrics_list_with_acc):
            with cols[idx % 2]:
                metric_data = []
                
                for exp in experiments:
                    if metric_key in ['accuracy_most_probable', 'accuracy_annotated']:
                        value = exp[metric_key]
                    else:
                        value = exp['distribution_metrics'][metric_key]['global']
                    
                    metric_data.append({
                        'Experiment': exp['name'],
                        'Value': value,
                        'Metric': metric_name
                    })
                
                if metric_data:
                    metric_df = pd.DataFrame(metric_data)
                    
                    # Sort appropriately for each metric
                    if metric_name in ["KL Divergence", "JS Divergence", "Euclidean Distance"]:
                        metric_df = metric_df.sort_values('Value', ascending=True)
                        better_direction = "lower is better"
                    else:
                        metric_df = metric_df.sort_values('Value', ascending=False)
                        better_direction = "higher is better"
                    
                    fig = px.bar(
                        metric_df,
                        x='Experiment',
                        y='Value',
                        title=f'{metric_name} by Experiment ({better_direction})',
                        color='Value',
                        color_continuous_scale='Viridis_r' if metric_name in ["KL Divergence", "JS Divergence", "Euclidean Distance"] else 'Viridis',
                        text_auto='.4f'
                    )
                    
                    fig.update_traces(
                        texttemplate='%{text:.4f}',
                        textposition='outside'
                    )
                    
                    # Set appropriate y-axis ranges
                    if metric_name == "JS Divergence":
                        fig.update_layout(yaxis_range=[0, 1])
                    elif metric_name == "Cosine Similarity":
                        fig.update_layout(yaxis_range=[-1, 1])
                    elif metric_name == "Pearson Correlation":
                        fig.update_layout(yaxis_range=[-1, 1])
                    elif 'Accuracy' in metric_name:
                        fig.update_layout(yaxis_range=[0, 1])
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Parallel coordinates plot for comparing all metrics
        if len(experiments) > 1:
            st.subheader("Parallel Coordinates Comparison")
            
            # Prepare data for parallel coordinates
            parallel_data = []
            for exp in experiments:
                metrics = exp['distribution_metrics']
                parallel_data.append({
                    'Experiment': exp['name'],
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            
            parallel_df = pd.DataFrame(parallel_data)
            
            fig_parallel = px.parallel_coordinates(
                parallel_df,
                color='Accuracy (Most Probable)',
                dimensions=['Accuracy (Most Probable)', 'Accuracy (Annotated)', 
                           'KL Divergence', 'JS Divergence', 'Cosine Similarity', 
                           'Euclidean Distance', 'Pearson Correlation'],
                color_continuous_scale=px.colors.diverging.Tealrose,
                title="Parallel Coordinates Plot of All Metrics"
            )
            
            st.plotly_chart(fig_parallel, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Between Metrics")
            
            # Calculate correlation matrix
            corr_cols = ['Accuracy (Most Probable)', 'Accuracy (Annotated)', 
                        'KL Divergence', 'JS Divergence', 'Cosine Similarity', 
                        'Euclidean Distance', 'Pearson Correlation']
            
            # Create dataframe for correlation
            corr_data = []
            for exp in experiments:
                metrics = exp['distribution_metrics']
                corr_data.append({
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto='.2f',
                aspect='auto',
                title='Correlation Matrix Between Metrics',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.header("Emotion Analysis")
        
        # Select experiment for detailed analysis
        selected_exp_name = st.selectbox(
            "Select Experiment for Detailed Analysis:",
            [exp['name'] for exp in experiments],
            key="emotion_analysis_select"
        )
        
        selected_exp = next(exp for exp in experiments if exp['name'] == selected_exp_name)
        
        if selected_exp:
            # Show accuracy metrics
            col_acc1, col_acc2, col_acc3 = st.columns(3)
            
            with col_acc1:
                st.metric("Accuracy (Most Probable)", f"{selected_exp['accuracy_most_probable']:.4f}")
            
            with col_acc2:
                st.metric("Accuracy (Annotated)", f"{selected_exp['accuracy_annotated']:.4f}")
            
            with col_acc3:
                diff = selected_exp['accuracy_most_probable'] - selected_exp['accuracy_annotated']
                st.metric("Difference", f"{diff:.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix (Most Probable Classes)")
            
            cm = selected_exp['confusion_matrix']
            emotion_names = selected_exp['emotion_cols']
            
            # Normalize by row
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
                yaxis_title="True Emotion (Most Probable)",
                height=600
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Performance by emotion (if classification report is available)
            if selected_exp['classification_report']:
                st.subheader("Performance by Emotion (Most Probable)")
                
                # Extract metrics per emotion
                emotion_metrics = []
                for emotion in emotion_names:
                    if emotion in selected_exp['classification_report']:
                        metrics = selected_exp['classification_report'][emotion]
                        emotion_metrics.append({
                            'Emotion': emotion,
                            'Precision': metrics.get('precision', 0),
                            'Recall': metrics.get('recall', 0),
                            'F1-Score': metrics.get('f1-score', 0),
                            'Support': metrics.get('support', 0)
                        })
                
                if emotion_metrics:
                    emotion_df = pd.DataFrame(emotion_metrics)
                    
                    # Bar chart for F1-Score by emotion
                    fig_f1 = px.bar(
                        emotion_df.sort_values('F1-Score', ascending=False),
                        x='Emotion',
                        y='F1-Score',
                        title=f'F1-Score by Emotion - {selected_exp_name}',
                        color='F1-Score',
                        color_continuous_scale='Viridis',
                        text_auto='.2f'
                    )
                    
                    fig_f1.update_layout(height=400)
                    st.plotly_chart(fig_f1, use_container_width=True)
                    
                    # Detailed table
                    st.dataframe(
                        emotion_df.style.format({
                            'Precision': '{:.3f}',
                            'Recall': '{:.3f}',
                            'F1-Score': '{:.3f}',
                            'Support': '{:.0f}'
                        }).highlight_max(subset=['Precision', 'Recall', 'F1-Score']),
                        use_container_width=True
                    )
            
            # Discrepancy analysis between most probable and annotated
            st.subheader("Label Discrepancy Analysis")
            
            if 'true_classes_annotated' in selected_exp and 'true_classes_most_probable' in selected_exp:
                # Calculate discrepancies
                discrepancies = []
                for i in range(min(len(selected_exp['true_classes_annotated']), 
                                 len(selected_exp['true_classes_most_probable']))):
                    if selected_exp['true_classes_annotated'][i] != selected_exp['true_classes_most_probable'][i]:
                        discrepancies.append({
                            'Sample': i,
                            'Annotated': selected_exp['emotion_cols'][selected_exp['true_classes_annotated'][i]],
                            'Most Probable': selected_exp['emotion_cols'][selected_exp['true_classes_most_probable'][i]]
                        })
                
                if discrepancies:
                    discrepancy_df = pd.DataFrame(discrepancies)
                    
                    st.write(f"**Total Discrepancies**: {len(discrepancies)} out of {selected_exp['n_samples']} samples ({len(discrepancies)/selected_exp['n_samples']:.2%})")
                    
                    # Most common discrepancies
                    discrepancy_counts = discrepancy_df.groupby(['Annotated', 'Most Probable']).size().reset_index(name='Count')
                    discrepancy_counts = discrepancy_counts.sort_values('Count', ascending=False)
                    
                    fig_discrepancies = px.bar(
                        discrepancy_counts.head(20),
                        x='Count',
                        y='Most Probable',
                        color='Annotated',
                        orientation='h',
                        title='Most Common Label Discrepancies',
                        height=400
                    )
                    
                    st.plotly_chart(fig_discrepancies, use_container_width=True)
                    
                    # Show detailed table
                    with st.expander("View Detailed Discrepancy Table"):
                        st.dataframe(discrepancy_df, use_container_width=True)
                else:
                    st.success("No discrepancies found between annotated and most probable labels.")
    
    with tab4:
        st.header("Model Ranking")
        
        # Let user choose ranking method
        ranking_method = st.selectbox(
            "Select Ranking Method:",
            ["Accuracy (Most Probable)", "Accuracy (Annotated)",
             "KL Divergence (lower is better)", "JS Divergence (lower is better)", 
             "Cosine Similarity (higher is better)", "Euclidean Distance (lower is better)",
             "Pearson Correlation (higher is better)", "Composite Score"]
        )
        
        if ranking_method == "Composite Score":
            st.info("""
            **Composite Score Calculation:**
            - For KL, JS, and Euclidean: normalized to 0-1 where 0 is worst, 1 is best
            - For Cosine, Pearson, and Accuracies: normalized to 0-1 where 0 is worst, 1 is best
            - All metrics weighted equally
            """)
        
        ranking_data = []
        for exp in experiments:
            metrics = exp['distribution_metrics']
            
            if ranking_method == "Accuracy (Most Probable)":
                score = exp['accuracy_most_probable']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "Accuracy (Annotated)":
                score = exp['accuracy_annotated']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "KL Divergence (lower is better)":
                score = metrics['kl']['global']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "JS Divergence (lower is better)":
                score = metrics['js']['global']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "Cosine Similarity (higher is better)":
                score = metrics['cosine']['global']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "Euclidean Distance (lower is better)":
                score = metrics['euclidean']['global']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            elif ranking_method == "Pearson Correlation (higher is better)":
                score = metrics['pearson']['global']
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': metrics['kl']['global'],
                    'JS Divergence': metrics['js']['global'],
                    'Cosine Similarity': metrics['cosine']['global'],
                    'Euclidean Distance': metrics['euclidean']['global'],
                    'Pearson Correlation': metrics['pearson']['global']
                })
            else:  # Composite Score
                # Normalize each metric to 0-1 scale
                all_acc_most_probable = [e['accuracy_most_probable'] for e in experiments]
                all_acc_annotated = [e['accuracy_annotated'] for e in experiments]
                all_kl = [e['distribution_metrics']['kl']['global'] for e in experiments]
                all_js = [e['distribution_metrics']['js']['global'] for e in experiments]
                all_cosine = [e['distribution_metrics']['cosine']['global'] for e in experiments]
                all_euclidean = [e['distribution_metrics']['euclidean']['global'] for e in experiments]
                all_pearson = [e['distribution_metrics']['pearson']['global'] for e in experiments]
                
                # For accuracies and similarities: keep as is (higher is better)
                acc_most_probable_norm = (exp['accuracy_most_probable'] - min(all_acc_most_probable)) / (max(all_acc_most_probable) - min(all_acc_most_probable) + 1e-10)
                acc_annotated_norm = (exp['accuracy_annotated'] - min(all_acc_annotated)) / (max(all_acc_annotated) - min(all_acc_annotated) + 1e-10)
                cosine_norm = (exp['distribution_metrics']['cosine']['global'] - min(all_cosine)) / (max(all_cosine) - min(all_cosine) + 1e-10)
                pearson_norm = (exp['distribution_metrics']['pearson']['global'] - min(all_pearson)) / (max(all_pearson) - min(all_pearson) + 1e-10)
                
                # For divergences/distances: invert so higher is better
                kl_norm = 1 - ((exp['distribution_metrics']['kl']['global'] - min(all_kl)) / (max(all_kl) - min(all_kl) + 1e-10))
                js_norm = 1 - ((exp['distribution_metrics']['js']['global'] - min(all_js)) / (max(all_js) - min(all_js) + 1e-10))
                euclidean_norm = 1 - ((exp['distribution_metrics']['euclidean']['global'] - min(all_euclidean)) / (max(all_euclidean) - min(all_euclidean) + 1e-10))
                
                score = (acc_most_probable_norm + acc_annotated_norm + kl_norm + js_norm + cosine_norm + euclidean_norm + pearson_norm) / 7
                
                ranking_data.append({
                    'Experiment': exp['name'],
                    'Score': score,
                    'Accuracy (Most Probable)': exp['accuracy_most_probable'],
                    'Accuracy (Annotated)': exp['accuracy_annotated'],
                    'KL Divergence': exp['distribution_metrics']['kl']['global'],
                    'JS Divergence': exp['distribution_metrics']['js']['global'],
                    'Cosine Similarity': exp['distribution_metrics']['cosine']['global'],
                    'Euclidean Distance': exp['distribution_metrics']['euclidean']['global'],
                    'Pearson Correlation': exp['distribution_metrics']['pearson']['global'],
                    'KL (norm)': kl_norm,
                    'JS (norm)': js_norm,
                    'Cosine (norm)': cosine_norm,
                    'Euclidean (norm)': euclidean_norm,
                    'Pearson (norm)': pearson_norm
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Sort based on score (direction depends on metric)
        if ranking_method in ["KL Divergence (lower is better)", "JS Divergence (lower is better)", 
                            "Euclidean Distance (lower is better)"]:
            ranking_df = ranking_df.sort_values('Score', ascending=True)
            better_text = "Lower is better"
        else:
            ranking_df = ranking_df.sort_values('Score', ascending=False)
            better_text = "Higher is better"
        
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        # Ranking chart
        fig_ranking = px.bar(
            ranking_df,
            x='Score',
            y='Experiment',
            orientation='h',
            title=f'Model Ranking by {ranking_method}',
            color='Score',
            color_continuous_scale='RdYlGn' if ranking_method == "Composite Score" else 
                                 ('Viridis_r' if "lower is better" in ranking_method else 'Viridis'),
            text='Score',
            height=max(300, len(experiments) * 40)
        )
        
        fig_ranking.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Detailed ranking table
        st.subheader("Detailed Metrics")
        
        if ranking_method == "Composite Score":
            display_columns = ['Rank', 'Experiment', 'Score', 'Accuracy (Most Probable)', 'Accuracy (Annotated)',
                             'KL Divergence', 'JS Divergence', 'Cosine Similarity', 'Euclidean Distance', 'Pearson Correlation']
        else:
            display_columns = ['Rank', 'Experiment', 'Score', 'Accuracy (Most Probable)', 'Accuracy (Annotated)',
                             'KL Divergence', 'JS Divergence', 'Cosine Similarity', 'Euclidean Distance', 'Pearson Correlation']
        
        display_rank_df = ranking_df[display_columns].copy()
        
        def highlight_top3(row):
            if row['Rank'] == 1:
                return ['background-color: gold'] * len(row)
            elif row['Rank'] == 2:
                return ['background-color: silver'] * len(row)
            elif row['Rank'] == 3:
                return ['background-color: #cd7f32'] * len(row)
            return [''] * len(row)
        
        styled_rank_df = display_rank_df.style.format({
            'Score': '{:.4f}',
            'Accuracy (Most Probable)': '{:.4f}',
            'Accuracy (Annotated)': '{:.4f}',
            'KL Divergence': '{:.4f}',
            'JS Divergence': '{:.4f}',
            'Cosine Similarity': '{:.4f}',
            'Euclidean Distance': '{:.4f}',
            'Pearson Correlation': '{:.4f}'
        }).apply(highlight_top3, axis=1)
        
        st.dataframe(styled_rank_df, use_container_width=True)
    
    with tab5:
        st.header("Detailed Sample Analysis")
        
        # Select experiment
        selected_exp_name = st.selectbox(
            "Select Experiment:",
            [exp['name'] for exp in experiments],
            key="sample_analysis_select"
        )
        
        selected_exp = next(exp for exp in experiments if exp['name'] == selected_exp_name)
        
        if selected_exp:
            # Distribution metrics for this experiment
            st.subheader(f"Distribution Metrics - {selected_exp_name}")
            
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("KL Divergence", f"{selected_exp['distribution_metrics']['kl']['global']:.4f}")
                st.metric("Accuracy (Most Probable)", f"{selected_exp['accuracy_most_probable']:.4f}")
            
            with col_metrics2:
                st.metric("JS Divergence", f"{selected_exp['distribution_metrics']['js']['global']:.4f}")
                st.metric("Accuracy (Annotated)", f"{selected_exp['accuracy_annotated']:.4f}")
            
            with col_metrics3:
                st.metric("Cosine Similarity", f"{selected_exp['distribution_metrics']['cosine']['global']:.4f}")
                st.metric("Difference", f"{selected_exp['accuracy_most_probable'] - selected_exp['accuracy_annotated']:.4f}")
            
            # Distribution of individual metric values
            st.subheader(f"Distribution of Individual {primary_metric} Values")
            
            fig_dist = px.histogram(
                x=selected_exp['primary_metric_individual'],
                nbins=30,
                title=f"Distribution of {primary_metric} - {selected_exp_name}",
                labels={'x': primary_metric, 'y': 'Count'},
                opacity=0.7
            )
            
            fig_dist.add_vline(
                x=selected_exp['primary_metric_value'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {selected_exp['primary_metric_value']:.4f}",
                annotation_position="top"
            )
            
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Select sample for detailed view
            st.subheader("Sample-Level Analysis")
            
            sample_idx = st.slider(
                "Select Sample", 
                0, 
                min(100, selected_exp['n_samples'] - 1), 
                0
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution of selected sample
                pred_sample = selected_exp['preds_df'].iloc[sample_idx][selected_exp['emotion_cols']]
                true_sample = selected_exp['labels_df'].iloc[sample_idx][selected_exp['emotion_cols']]
                
                fig_sample = go.Figure(data=[
                    go.Bar(
                        x=selected_exp['emotion_cols'],
                        y=pred_sample.values.astype(float),
                        name='Predicted',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    go.Bar(
                        x=selected_exp['emotion_cols'],
                        y=true_sample.values.astype(float),
                        name='Ground Truth',
                        marker_color='red',
                        opacity=0.7
                    )
                ])
                
                fig_sample.update_layout(
                    title=f"Distribution - Sample {sample_idx}",
                    xaxis_title="Emotion",
                    yaxis_title="Probability",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_sample, use_container_width=True)
            
            with col2:
                # Radar chart for multivariate visualization
                fig_radar_sample = go.Figure()
                
                fig_radar_sample.add_trace(go.Scatterpolar(
                    r=pred_sample.values.astype(float),
                    theta=selected_exp['emotion_cols'],
                    fill='toself',
                    name='Predicted'
                ))
                
                fig_radar_sample.add_trace(go.Scatterpolar(
                    r=true_sample.values.astype(float),
                    theta=selected_exp['emotion_cols'],
                    fill='toself',
                    name='Ground Truth'
                ))
                
                fig_radar_sample.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(pred_sample.values.astype(float)), max(true_sample.values.astype(float)))]
                        )
                    ),
                    showlegend=True,
                    title=f"Radar Visualization - Sample {sample_idx}",
                    height=400
                )
                
                st.plotly_chart(fig_radar_sample, use_container_width=True)
            
            # Sample information
            st.subheader("Sample Information")
            
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            
            with col_info1:
                pred_class = selected_exp['pred_classes'][sample_idx] if sample_idx < len(selected_exp['pred_classes']) else -1
                true_class_most_probable = selected_exp['true_classes_most_probable'][sample_idx] if sample_idx < len(selected_exp['true_classes_most_probable']) else -1
                true_class_annotated = selected_exp['true_classes_annotated'][sample_idx] if sample_idx < len(selected_exp['true_classes_annotated']) else -1
                
                pred_emotion = selected_exp['emotion_cols'][pred_class] if 0 <= pred_class < len(selected_exp['emotion_cols']) else "N/A"
                true_emotion_most_probable = selected_exp['emotion_cols'][true_class_most_probable] if 0 <= true_class_most_probable < len(selected_exp['emotion_cols']) else "N/A"
                true_emotion_annotated = selected_exp['emotion_cols'][true_class_annotated] if 0 <= true_class_annotated < len(selected_exp['emotion_cols']) else "N/A"
                
                st.metric("Predicted Class", pred_emotion)
            
            with col_info2:
                st.metric("True Class (Most Probable)", true_emotion_most_probable)
            
            with col_info3:
                st.metric("True Class (Annotated)", true_emotion_annotated)
            
            with col_info4:
                correct_most_probable = pred_class == true_class_most_probable
                correct_annotated = pred_class == true_class_annotated
                st.metric("Correct (Most Probable)", "Yes" if correct_most_probable else "No")
                st.metric("Correct (Annotated)", "Yes" if correct_annotated else "No")
            
            # Individual metric values for this sample
            st.subheader("Individual Metric Values for This Sample")
            
            metrics_cols = st.columns(5)
            metrics_data = [
                ("KL Divergence", selected_exp['distribution_metrics']['kl']['individual'][sample_idx] 
                 if sample_idx < len(selected_exp['distribution_metrics']['kl']['individual']) else 0),
                ("JS Divergence", selected_exp['distribution_metrics']['js']['individual'][sample_idx] 
                 if sample_idx < len(selected_exp['distribution_metrics']['js']['individual']) else 0),
                ("Cosine Similarity", selected_exp['distribution_metrics']['cosine']['individual'][sample_idx] 
                 if sample_idx < len(selected_exp['distribution_metrics']['cosine']['individual']) else 0),
                ("Euclidean Distance", selected_exp['distribution_metrics']['euclidean']['individual'][sample_idx] 
                 if sample_idx < len(selected_exp['distribution_metrics']['euclidean']['individual']) else 0),
                ("Pearson Correlation", selected_exp['distribution_metrics']['pearson']['individual'][sample_idx] 
                 if sample_idx < len(selected_exp['distribution_metrics']['pearson']['individual']) else 0)
            ]
            
            for idx, (metric_name, value) in enumerate(metrics_data):
                with metrics_cols[idx]:
                    st.metric(metric_name, f"{value:.4f}")
    
    with tab6:
        st.header("Export Results")
        
        st.info("""
        **Export Options:**
        
        1. **Summary Tables**: Formatted CSV for inclusion in papers
        2. **Complete Data**: All results for further analysis
        3. **LaTeX Code**: Ready-to-use tables for academic papers
        """)
        
        # Export summary table
        st.subheader("Export Summary Table")
        
        # Create complete summary table
        export_data = []
        for exp in experiments:
            metrics = exp['distribution_metrics']
            export_data.append({
                'Experiment': exp['name'],
                'Samples': exp['n_samples'],
                'Accuracy_Most_Probable': exp['accuracy_most_probable'],
                'Accuracy_Annotated': exp['accuracy_annotated'],
                'Accuracy_Difference': exp['accuracy_most_probable'] - exp['accuracy_annotated'],
                'KL_Divergence': metrics['kl']['global'],
                'JS_Divergence': metrics['js']['global'],
                'Cosine_Similarity': metrics['cosine']['global'],
                'Euclidean_Distance': metrics['euclidean']['global'],
                'Pearson_Correlation': metrics['pearson']['global'],
                'Top_3_Accuracy': exp['top_k_accuracy'],
                'Mean_Rank': exp['mean_rank']
            })
        
        export_df = pd.DataFrame(export_data)
        csv_export = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Metrics Summary (CSV)",
            data=csv_export,
            file_name="distribution_metrics_summary.csv",
            mime="text/csv",
            key="download_summary"
        )
        
        # Export complete data per experiment
        st.subheader("Export Complete Data")
        
        for exp in experiments:
            # Create dataframe with all metrics per sample
            sample_data = []
            max_samples = min(1000, exp['n_samples'])  # Limit for performance
            
            for i in range(max_samples):
                sample_data.append({
                    'Sample': i,
                    'File': exp['preds_df'].iloc[i]['file'] if 'file' in exp['preds_df'].columns else f"sample_{i}",
                    'True_Class_Most_Probable': exp['true_classes_most_probable'][i] if i < len(exp['true_classes_most_probable']) else -1,
                    'True_Class_Annotated': exp['true_classes_annotated'][i] if i < len(exp['true_classes_annotated']) else -1,
                    'Predicted_Class': exp['pred_classes'][i] if i < len(exp['pred_classes']) else -1,
                    'Correct_Most_Probable': 1 if (i < len(exp['true_classes_most_probable']) and i < len(exp['pred_classes']) and 
                                                  exp['true_classes_most_probable'][i] == exp['pred_classes'][i]) else 0,
                    'Correct_Annotated': 1 if (i < len(exp['true_classes_annotated']) and i < len(exp['pred_classes']) and 
                                              exp['true_classes_annotated'][i] == exp['pred_classes'][i]) else 0,
                    'KL_Divergence': exp['distribution_metrics']['kl']['individual'][i] if i < len(exp['distribution_metrics']['kl']['individual']) else 0,
                    'JS_Divergence': exp['distribution_metrics']['js']['individual'][i] if i < len(exp['distribution_metrics']['js']['individual']) else 0,
                    'Cosine_Similarity': exp['distribution_metrics']['cosine']['individual'][i] if i < len(exp['distribution_metrics']['cosine']['individual']) else 0,
                    'Euclidean_Distance': exp['distribution_metrics']['euclidean']['individual'][i] if i < len(exp['distribution_metrics']['euclidean']['individual']) else 0,
                    'Pearson_Correlation': exp['distribution_metrics']['pearson']['individual'][i] if i < len(exp['distribution_metrics']['pearson']['individual']) else 0,
                    'Rank_Position': exp['rank_positions'][i] if i < len(exp['rank_positions']) else -1
                })
            
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                csv_samples = sample_df.to_csv(index=False)
                
                st.download_button(
                    label=f"Download {exp['name']} - Sample Data",
                    data=csv_samples,
                    file_name=f"{exp['name']}_detailed_metrics.csv",
                    mime="text/csv",
                    key=f"download_{exp['name']}"
                )
        
        # Generate LaTeX code for tables
        st.subheader("LaTeX Code for Papers")
        
        if st.button("Generate LaTeX Code"):
            latex_code = r"\begin{table}[h!]"
            latex_code += r"\centering"
            latex_code += r"\caption{Distribution Metrics and Accuracy Comparison of Emotion Models}"
            latex_code += r"\label{tab:distribution_metrics}"
            latex_code += r"\begin{tabular}{lccccccc}"
            latex_code += r"\hline"
            latex_code += r"Model & Acc. (Prob.) $\uparrow$ & Acc. (Ann.) $\uparrow$ & KL $\downarrow$ & JS $\downarrow$ & Cosine $\uparrow$ & Euclidean $\downarrow$ & Pearson $\uparrow$ \\"
            latex_code += r"\hline"
            
            for exp in experiments:
                metrics = exp['distribution_metrics']
                latex_code += f"{exp['name']} & "
                latex_code += f"{exp['accuracy_most_probable']:.4f} & "
                latex_code += f"{exp['accuracy_annotated']:.4f} & "
                latex_code += f"{metrics['kl']['global']:.4f} & "
                latex_code += f"{metrics['js']['global']:.4f} & "
                latex_code += f"{metrics['cosine']['global']:.4f} & "
                latex_code += f"{metrics['euclidean']['global']:.4f} & "
                latex_code += f"{metrics['pearson']['global']:.4f} \\\\"
            
            latex_code += r"\hline"
            latex_code += r"\end{tabular}"
            latex_code += r"\end{table}"
            
            st.code(latex_code, language='latex')
            st.info("Copy and paste this LaTeX code into your paper. Arrows indicate direction: down for lower is better, up for higher is better.")

else:
    # Initial instructions
    st.markdown("""
    ## Multi-Experiment Emotion Distribution Metrics Analysis
    
    **How to use this tool:**
    
    1. **Add Experiments**: Use the sidebar to add multiple file pairs (predictions + ground truth)
    2. **Configure Emotions**: Specify the 14 emotion columns in the sidebar
    3. **Process**: Click "Process All Experiments" to analyze all models
    4. **Analyze**: Explore the different comparison tabs focused on distribution metrics
    
    ### New Feature: Two Accuracy Metrics
    
    This tool now calculates **two different accuracy metrics**:
    
    1. **Accuracy (Most Probable)**: Uses the most probable class from the ground truth distribution (argmax of probabilities)
    2. **Accuracy (Annotated)**: Uses the manually annotated label from the ground truth file
    
    ### Expected File Format:
    
    **Predictions File (Model):**
    ```
    happy,contempt,elated,hopeful,surprised,proud,loved,angry,astonished,disgusted,fearful,sad,fatigued,neutral,file
    0.0019,0.0567,0.0456,0.0712,0.0303,0.0520,0.0283,0.1426,0.0612,0.1153,0.0745,0.2090,0.01,0.02,image1.jpg
    ```
    
    **Ground Truth File:**
    ```
    happy,contempt,elated,hopeful,surprised,proud,loved,angry,astonished,disgusted,fearful,sad,fatigued,neutral,emotion_label,file
    2.7e-09,0.1083,0.0028,8.3e-05,0.0045,0.0006,3.0e-06,0.4799,0.0236,0.2314,0.1481,0.0002,0.01,0.02,6,image1.jpg
    ```
    
    ### Supported Emotions (14):
    
    | Emotion | Description |
    |--------|-----------|
    | happy | Happy |
    | contempt | Contempt |
    | elated | Elated/Exultant |
    | hopeful | Hopeful |
    | surprised | Surprised |
    | proud | Proud |
    | loved | Loved |
    | angry | Angry |
    | astonished | Astonished |
    | disgusted | Disgusted |
    | fearful | Fearful |
    | sad | Sad |
    | fatigued | Fatigued |
    | neutral | Neutral |
    
    ### Distribution Metrics Calculated (Raw Values):
    
    - **KL Divergence**: Information loss between distributions (0 to ∞, lower is better)
    - **JS Divergence**: Jensen-Shannon divergence between distributions (0 to 1, lower is better)
    - **Cosine Similarity**: Angle between distribution vectors (-1 to 1, higher is better)
    - **Euclidean Distance**: Straight-line distance between distributions (0 to ∞, lower is better)
    - **Pearson Correlation**: Linear correlation between distributions (-1 to 1, higher is better)
    
    ### Analysis Features:
    
    - **Two Accuracy Metrics**: Compare performance using different ground truth references
    - **Multiple Ranking Methods**: Choose which metric to use for ranking or use a composite score
    - **Detailed Sample Analysis**: Examine individual samples with all metrics
    - **Parallel Coordinates**: Visual comparison across all metrics simultaneously
    """)
    
    # Quick example
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Add Experiments**")
        st.write("Each experiment = Predictions + Ground Truth")
    
    with col2:
        st.markdown("**2. Configure**")
        st.write("Specify the 14 emotions")
    
    with col3:
        st.markdown("**3. Analyze Both Accuracies**")
        st.write("Compare models using two accuracy metrics")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Style for buttons */
    .stButton > button {
        width: 100%;
        margin: 5px 0;
    }
    
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    
    /* Style for metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Better table formatting */
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)