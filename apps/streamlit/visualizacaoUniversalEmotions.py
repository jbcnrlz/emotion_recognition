import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Emotion Model Analysis", layout="wide")

# Main title
st.title("📊 Emotion Model Performance Analysis")
st.markdown("""
This application compares emotional distributions between ground truth and model estimates.
Upload the required files and visualize various performance metrics.
""")

# Function to load files
@st.cache_data
def load_file(file):
    return pd.read_csv(file)

# Function to load and display image
def load_image(image_path):
    """
    Try to load image from various possible locations
    """
    # Try the full path first
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            return img
        except:
            pass
    
    # Try relative path (just filename)
    filename = os.path.basename(image_path)
    if os.path.exists(filename):
        try:
            img = Image.open(filename)
            return img
        except:
            pass
    
    # Try in a local 'images' folder
    local_path = os.path.join("images", filename)
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path)
            return img
        except:
            pass
    
    return None

# Function to calculate distribution metrics
def calculate_distribution_metrics(true_dist, pred_dist):
    """
    Calculate various metrics to compare probability distributions
    """
    metrics = {}
    
    # KL Divergence (symmetric)
    eps = 1e-10
    true_safe = np.clip(true_dist, eps, 1)
    pred_safe = np.clip(pred_dist, eps, 1)
    
    # KL Divergence in both directions
    kl_forward = np.sum(true_safe * np.log(true_safe / pred_safe))
    kl_backward = np.sum(pred_safe * np.log(pred_safe / true_safe))
    metrics['KL_Divergence'] = (kl_forward + kl_backward) / 2
    
    # Jensen-Shannon Divergence
    m = 0.5 * (true_safe + pred_safe)
    jsd = 0.5 * kl_forward + 0.5 * kl_backward
    metrics['Jensen_Shannon_Divergence'] = jsd
    
    # Correlation
    metrics['Pearson_Correlation'] = np.corrcoef(true_dist, pred_dist)[0, 1]
    metrics['Spearman_Correlation'] = stats.spearmanr(true_dist, pred_dist)[0]
    
    # Distances
    metrics['Euclidean_Distance'] = np.linalg.norm(true_dist - pred_dist)
    metrics['Cosine_Similarity'] = np.dot(true_dist, pred_dist) / (np.linalg.norm(true_dist) * np.linalg.norm(pred_dist))
    
    # Earth Mover's Distance (approximation)
    metrics['EMD'] = np.sum(np.abs(np.cumsum(true_dist) - np.cumsum(pred_dist)))
    
    # Bhattacharyya Distance
    metrics['Bhattacharyya_Distance'] = -np.log(np.sum(np.sqrt(true_safe * pred_safe)))
    
    # Hellinger Distance
    metrics['Hellinger_Distance'] = np.sqrt(0.5 * np.sum((np.sqrt(true_safe) - np.sqrt(pred_safe)) ** 2))
    
    # Intersection
    metrics['Intersection'] = np.sum(np.minimum(true_dist, pred_dist))
    
    return metrics

# Sidebar for file upload
st.sidebar.header("📁 File Upload")

# File uploaders
gt_file = st.sidebar.file_uploader("Ground Truth File (labels.csv)", type=['csv'])
estim_file = st.sidebar.file_uploader("Estimates File (estim.csv)", type=['csv'])

# Base image path input
base_image_path = st.sidebar.text_input(
    "Base Image Path (optional)",
    help="Enter the base path where images are stored. Leave empty to use paths from CSV."
)

# Emotion mapping upload (optional)
emotion_map_file = st.sidebar.file_uploader("Emotion Mapping (CSV - optional)", type=['csv'])

# Default emotion mapping
default_emotion_map = {
    0: "neutral",
    1: "happy", 
    2: "sad",
    3: "surprised",
    4: "fearful",
    5: "disgusted",
    6: "angry",
    7: "contempt"
}

if emotion_map_file:
    emotion_df = pd.read_csv(emotion_map_file)
    emotion_map = dict(zip(emotion_df['code'], emotion_df['emotion']))
else:
    emotion_map = default_emotion_map

# Load data if files are provided
if gt_file and estim_file:
    gt_df = load_file(gt_file)
    estim_df = load_file(estim_file)
    
    # List of emotions (assuming same order in both files)
    emotions = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 General Metrics", 
        "📊 Classification", 
        "🎯 Distribution Analysis",
        "📊 Valence-Arousal Space",
        "🖼️ Sample Details with Images",
        "📁 File Information"
    ])
    
    with tab1:
        st.header("General Performance Metrics")
        
        # Calculate metrics for each sample
        all_metrics = []
        for i in range(len(gt_df)):
            true_dist = gt_df[emotions].iloc[i].values
            pred_dist = estim_df[emotions].iloc[i].values
            metrics = calculate_distribution_metrics(true_dist, pred_dist)
            all_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Average metrics
        st.subheader("📊 Average Distribution Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("KL Divergence", f"{metrics_df['KL_Divergence'].mean():.4f}")
            st.metric("Pearson Correlation", f"{metrics_df['Pearson_Correlation'].mean():.4f}")
            
        with col2:
            st.metric("Jensen-Shannon", f"{metrics_df['Jensen_Shannon_Divergence'].mean():.4f}")
            st.metric("Spearman Correlation", f"{metrics_df['Spearman_Correlation'].mean():.4f}")
            
        with col3:
            st.metric("Cosine Similarity", f"{metrics_df['Cosine_Similarity'].mean():.4f}")
            st.metric("Intersection", f"{metrics_df['Intersection'].mean():.4f}")
            
        with col4:
            st.metric("Hellinger Distance", f"{metrics_df['Hellinger_Distance'].mean():.4f}")
            st.metric("Earth Mover's Distance", f"{metrics_df['EMD'].mean():.4f}")
        
        # Separate boxplots for distances and similarities
        st.subheader("Distribution of Distance Metrics")
        
        # Distance metrics (lower is better)
        distance_metrics = ['KL_Divergence', 'Jensen_Shannon_Divergence', 
                          'Euclidean_Distance', 'EMD', 
                          'Bhattacharyya_Distance', 'Hellinger_Distance']
        
        # Filter only available metrics
        available_distance_metrics = [m for m in distance_metrics if m in metrics_df.columns]
        
        if available_distance_metrics:
            fig_distances = go.Figure()
            for metric in available_distance_metrics:
                fig_distances.add_trace(go.Box(
                    y=metrics_df[metric],
                    name=metric,
                    boxpoints='all'
                ))
            
            fig_distances.update_layout(
                title="Box Plot of Distance Metrics (Lower is Better)",
                yaxis_title="Value",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_distances, use_container_width=True)
        else:
            st.info("No distance metrics available to display.")
        
        # Similarity metrics (higher is better)
        st.subheader("Distribution of Similarity Metrics")
        
        similarity_metrics = ['Pearson_Correlation', 'Spearman_Correlation', 
                            'Cosine_Similarity', 'Intersection']
        
        # Filter only available metrics
        available_similarity_metrics = [m for m in similarity_metrics if m in metrics_df.columns]
        
        if available_similarity_metrics:
            fig_similarities = go.Figure()
            for metric in available_similarity_metrics:
                fig_similarities.add_trace(go.Box(
                    y=metrics_df[metric],
                    name=metric,
                    boxpoints='all'
                ))
            
            fig_similarities.update_layout(
                title="Box Plot of Similarity Metrics (Higher is Better)",
                yaxis_title="Value",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_similarities, use_container_width=True)
        else:
            st.info("No similarity metrics available to display.")
        
        # Summary statistics table
        st.subheader("Summary Statistics of All Metrics")
        
        # Calculate summary statistics
        summary_stats = pd.DataFrame({
            'Mean': metrics_df.mean(),
            'Std': metrics_df.std(),
            'Min': metrics_df.min(),
            '25%': metrics_df.quantile(0.25),
            'Median': metrics_df.median(),
            '75%': metrics_df.quantile(0.75),
            'Max': metrics_df.max()
        })
        
        # Categorize metrics
        summary_stats['Category'] = 'Other'
        for metric in distance_metrics:
            if metric in summary_stats.index:
                summary_stats.loc[metric, 'Category'] = 'Distance'
        for metric in similarity_metrics:
            if metric in summary_stats.index:
                summary_stats.loc[metric, 'Category'] = 'Similarity'
        
        st.dataframe(summary_stats.round(4), use_container_width=True)
    
    with tab2:
        st.header("Classification Analysis")
        
        # Extract predictions (emotion with highest probability)
        pred_emotions = estim_df[emotions].idxmax(axis=1)
        true_emotions_gt = gt_df['emotion_label']
        
        # Convert names to codes (if needed)
        emotion_to_code = {v: k for k, v in emotion_map.items()}
        
        # For ground truth, we already have codes
        y_true = true_emotions_gt.values
        
        # For predictions, convert names to codes
        y_pred = []
        for emotion_name in pred_emotions:
            # Find corresponding code
            code = None
            for k, v in emotion_map.items():
                if v == emotion_name:
                    code = k
                    break
            y_pred.append(code)
        
        y_pred = np.array(y_pred)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("F1-Score (weighted)", f"{f1:.4f}")
        with col3:
            st.metric("Number of Samples", len(y_true))
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(emotion_map.keys()))
        
        # Create labels for the matrix
        labels = [emotion_map[i] for i in emotion_map.keys()]
        
        fig_cm = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=500
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_true, y_pred, 
                                      target_names=[emotion_map[i] for i in emotion_map.keys()],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}").highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
        
        # Per-class accuracy
        st.subheader("Per-Class Accuracy")
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for emotion_code in emotion_map.keys():
            mask = y_true == emotion_code
            if mask.sum() > 0:
                class_acc = (y_true[mask] == y_pred[mask]).mean()
                class_accuracy[emotion_map[emotion_code]] = class_acc
        
        class_acc_df = pd.DataFrame(list(class_accuracy.items()), columns=['Emotion', 'Accuracy'])
        
        fig_class_acc = px.bar(
            class_acc_df,
            x='Emotion',
            y='Accuracy',
            color='Accuracy',
            color_continuous_scale='Viridis',
            text_auto='.2%'
        )
        fig_class_acc.update_layout(
            title="Accuracy per Emotion Class",
            yaxis_title="Accuracy",
            yaxis_tickformat='.0%',
            height=400
        )
        st.plotly_chart(fig_class_acc, use_container_width=True)
    
    with tab3:
        st.header("Distribution Analysis")
        
        # Sample selector
        sample_idx = st.slider("Select sample to visualize:", 0, len(gt_df)-1, 0)
        
        # Try to display image for selected sample
        if 'file' in gt_df.columns:
            image_path = gt_df['file'].iloc[sample_idx]
            
            # Use base path if provided
            if base_image_path and os.path.exists(base_image_path):
                # Extract just the filename
                filename = os.path.basename(image_path)
                full_image_path = os.path.join(base_image_path, filename)
            else:
                full_image_path = image_path
            
            # Display image if available
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(f"Image - Sample {sample_idx}")
                img = load_image(full_image_path)
                
                if img:
                    st.image(img, caption=f"Sample {sample_idx}: {os.path.basename(image_path)}", 
                            use_column_width=True)
                else:
                    st.warning(f"Image not found at: {full_image_path}")
                    st.write(f"Tried path: {full_image_path}")
            
            with col2:
                # Distribution for selected sample
                st.subheader(f"Emotion Distribution - Sample {sample_idx}")
                
                true_sample = gt_df[emotions].iloc[sample_idx]
                pred_sample = estim_df[emotions].iloc[sample_idx]
                
                fig_dist = go.Figure()
                
                fig_dist.add_trace(go.Bar(
                    x=emotions,
                    y=true_sample.values,
                    name='Ground Truth',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_dist.add_trace(go.Bar(
                    x=emotions,
                    y=pred_sample.values,
                    name='Estimate',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    title=f"Distribution Comparison - Sample {sample_idx}",
                    xaxis_title="Emotion",
                    yaxis_title="Probability",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            # Distribution for selected sample (without image)
            st.subheader(f"Emotion Distribution - Sample {sample_idx}")
            
            true_sample = gt_df[emotions].iloc[sample_idx]
            pred_sample = estim_df[emotions].iloc[sample_idx]
            
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Bar(
                x=emotions,
                y=true_sample.values,
                name='Ground Truth',
                marker_color='blue',
                opacity=0.7
            ))
            
            fig_dist.add_trace(go.Bar(
                x=emotions,
                y=pred_sample.values,
                name='Estimate',
                marker_color='red',
                opacity=0.7
            ))
            
            fig_dist.update_layout(
                title=f"Distribution Comparison - Sample {sample_idx}",
                xaxis_title="Emotion",
                yaxis_title="Probability",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Show metrics for this sample
        st.subheader("Metrics for this sample:")
        true_sample = gt_df[emotions].iloc[sample_idx].values
        pred_sample = estim_df[emotions].iloc[sample_idx].values
        sample_metrics = calculate_distribution_metrics(true_sample, pred_sample)
        
        # Separate distance and similarity metrics
        distance_sample = {k: v for k, v in sample_metrics.items() 
                          if k in ['KL_Divergence', 'Jensen_Shannon_Divergence', 
                                  'Euclidean_Distance', 'EMD', 
                                  'Bhattacharyya_Distance', 'Hellinger_Distance']}
        
        similarity_sample = {k: v for k, v in sample_metrics.items() 
                           if k in ['Pearson_Correlation', 'Spearman_Correlation', 
                                   'Cosine_Similarity', 'Intersection']}
        
        st.markdown("**Distance Metrics (lower is better):**")
        cols = st.columns(3)
        for i, (metric, value) in enumerate(distance_sample.items()):
            with cols[i % 3]:
                st.metric(metric, f"{value:.4f}")
        
        st.markdown("**Similarity Metrics (higher is better):**")
        cols = st.columns(4)
        for i, (metric, value) in enumerate(similarity_sample.items()):
            with cols[i % 4]:
                st.metric(metric, f"{value:.4f}")
        
        # Show emotion labels for this sample
        if 'emotion_label' in gt_df.columns:
            true_emotion_code = gt_df['emotion_label'].iloc[sample_idx]
            true_emotion_name = emotion_map.get(true_emotion_code, f"Unknown ({true_emotion_code})")
            
            pred_emotion_name = emotions[np.argmax(pred_sample)]
            pred_emotion_code = None
            for k, v in emotion_map.items():
                if v == pred_emotion_name:
                    pred_emotion_code = k
                    break
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Emotion", f"{true_emotion_name}", 
                         delta="Correct" if true_emotion_code == pred_emotion_code else "Incorrect")
            with col2:
                st.metric("Predicted Emotion", f"{pred_emotion_name}")
            with col3:
                match = true_emotion_code == pred_emotion_code
                st.metric("Classification", "✓ Match" if match else "✗ Mismatch")
    
    with tab4:
        st.header("📊 Valence-Arousal Space Analysis")
        
        # Check if valence and arousal columns exist
        if 'valence' in gt_df.columns and 'arousal' in gt_df.columns:
            
            # Create two columns for overall and per-sample views
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Overall valence-arousal scatter plot
                st.subheader("Overall Valence-Arousal Space")
                
                # Create the scatter plot
                fig_va = go.Figure()
                
                # Add scatter points
                fig_va.add_trace(go.Scatter(
                    x=gt_df['valence'],
                    y=gt_df['arousal'],
                    mode='markers',
                    name='Samples',
                    marker=dict(
                        size=10,
                        color=gt_df['emotion_label'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Emotion Label",
                            tickvals=list(emotion_map.keys()),
                            ticktext=list(emotion_map.values())
                        ),
                        opacity=0.7
                    ),
                    text=[f"Sample {i}<br>Emotion: {emotion_map.get(gt_df['emotion_label'].iloc[i], 'Unknown')}<br>Valence: {gt_df['valence'].iloc[i]:.3f}<br>Arousal: {gt_df['arousal'].iloc[i]:.3f}" 
                          for i in range(len(gt_df))],
                    hoverinfo='text'
                ))
                
                # Add quadrant lines
                fig_va.add_shape(type="line",
                    x0=0, y0=-1, x1=0, y1=1,
                    line=dict(color="Gray", width=2, dash="dash")
                )
                fig_va.add_shape(type="line",
                    x0=-1, y0=0, x1=1, y1=0,
                    line=dict(color="Gray", width=2, dash="dash")
                )
                
                # Add quadrant labels
                fig_va.add_annotation(x=0.5, y=0.5, text="High Arousal<br>Positive Valence",
                                     showarrow=False, font=dict(size=10))
                fig_va.add_annotation(x=0.5, y=-0.5, text="Low Arousal<br>Positive Valence",
                                     showarrow=False, font=dict(size=10))
                fig_va.add_annotation(x=-0.5, y=0.5, text="High Arousal<br>Negative Valence",
                                     showarrow=False, font=dict(size=10))
                fig_va.add_annotation(x=-0.5, y=-0.5, text="Low Arousal<br>Negative Valence",
                                     showarrow=False, font=dict(size=10))
                
                # Set axis limits and labels
                fig_va.update_layout(
                    title="Valence-Arousal Space (-1 to 1)",
                    xaxis_title="Valence (Negative ↔ Positive)",
                    yaxis_title="Arousal (Low ↔ High)",
                    xaxis_range=[-1, 1],
                    yaxis_range=[-1, 1],
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig_va, use_container_width=True)
            
            with col2:
                # Statistics
                st.subheader("Statistics")
                
                st.metric("Mean Valence", f"{gt_df['valence'].mean():.4f}")
                st.metric("Mean Arousal", f"{gt_df['arousal'].mean():.4f}")
                st.metric("Valence Std", f"{gt_df['valence'].std():.4f}")
                st.metric("Arousal Std", f"{gt_df['arousal'].std():.4f}")
                
                # Quadrant distribution
                st.subheader("Quadrant Distribution")
                
                # Calculate quadrant counts
                quadrant_counts = {
                    "Q1 (+,+)": ((gt_df['valence'] >= 0) & (gt_df['arousal'] >= 0)).sum(),
                    "Q2 (-,+)": ((gt_df['valence'] < 0) & (gt_df['arousal'] >= 0)).sum(),
                    "Q3 (-,-)": ((gt_df['valence'] < 0) & (gt_df['arousal'] < 0)).sum(),
                    "Q4 (+,-)": ((gt_df['valence'] >= 0) & (gt_df['arousal'] < 0)).sum()
                }
                
                for quadrant, count in quadrant_counts.items():
                    percentage = (count / len(gt_df)) * 100
                    st.write(f"**{quadrant}:** {count} samples ({percentage:.1f}%)")
            
            # Per-sample valence-arousal analysis
            st.subheader("Per-Sample Valence-Arousal Analysis")
            
            # Sample selector
            va_sample_idx = st.slider("Select sample for detailed view:", 
                                      0, len(gt_df)-1, 0, key="va_sample")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display valence-arousal values for selected sample
                st.markdown("### Selected Sample Values")
                
                valence_val = gt_df['valence'].iloc[va_sample_idx]
                arousal_val = gt_df['arousal'].iloc[va_sample_idx]
                
                st.metric("Valence", f"{valence_val:.4f}")
                st.metric("Arousal", f"{arousal_val:.4f}")
                
                # Determine quadrant
                if valence_val >= 0 and arousal_val >= 0:
                    quadrant = "Q1: Positive Valence, High Arousal"
                    quadrant_color = "green"
                elif valence_val < 0 and arousal_val >= 0:
                    quadrant = "Q2: Negative Valence, High Arousal"
                    quadrant_color = "orange"
                elif valence_val < 0 and arousal_val < 0:
                    quadrant = "Q3: Negative Valence, Low Arousal"
                    quadrant_color = "red"
                else:
                    quadrant = "Q4: Positive Valence, Low Arousal"
                    quadrant_color = "blue"
                
                st.markdown(f"**Quadrant:** <span style='color:{quadrant_color}'>{quadrant}</span>", 
                           unsafe_allow_html=True)
                
                # Emotion label for this sample
                if 'emotion_label' in gt_df.columns:
                    emotion_code = gt_df['emotion_label'].iloc[va_sample_idx]
                    emotion_name = emotion_map.get(emotion_code, f"Unknown ({emotion_code})")
                    st.metric("Emotion Label", emotion_name)
            
            with col2:
                # Single sample on valence-arousal space
                st.markdown("### Sample in Valence-Arousal Space")
                
                fig_single_va = go.Figure()
                
                # Add all samples as background
                fig_single_va.add_trace(go.Scatter(
                    x=gt_df['valence'],
                    y=gt_df['arousal'],
                    mode='markers',
                    name='All Samples',
                    marker=dict(
                        size=6,
                        color='lightgray',
                        opacity=0.3
                    ),
                    showlegend=True
                ))
                
                # Highlight selected sample
                fig_single_va.add_trace(go.Scatter(
                    x=[valence_val],
                    y=[arousal_val],
                    mode='markers',
                    name='Selected Sample',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star'
                    ),
                    text=[f"Sample {va_sample_idx}<br>Valence: {valence_val:.3f}<br>Arousal: {arousal_val:.3f}"],
                    hoverinfo='text'
                ))
                
                # Add quadrant lines
                fig_single_va.add_shape(type="line",
                    x0=0, y0=-1, x1=0, y1=1,
                    line=dict(color="Gray", width=2, dash="dash")
                )
                fig_single_va.add_shape(type="line",
                    x0=-1, y0=0, x1=1, y1=0,
                    line=dict(color="Gray", width=2, dash="dash")
                )
                
                # Set axis limits and labels
                fig_single_va.update_layout(
                    title=f"Sample {va_sample_idx} in Valence-Arousal Space",
                    xaxis_title="Valence (Negative ↔ Positive)",
                    yaxis_title="Arousal (Low ↔ High)",
                    xaxis_range=[-1, 1],
                    yaxis_range=[-1, 1],
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_single_va, use_container_width=True)
            
            # Valence-Arousal by Emotion Class
            st.subheader("Valence-Arousal by Emotion Class")
            
            if 'emotion_label' in gt_df.columns:
                # Calculate mean valence and arousal per emotion
                emotion_stats = []
                for emotion_code, emotion_name in emotion_map.items():
                    mask = gt_df['emotion_label'] == emotion_code
                    if mask.any():
                        mean_valence = gt_df.loc[mask, 'valence'].mean()
                        mean_arousal = gt_df.loc[mask, 'arousal'].mean()
                        count = mask.sum()
                        emotion_stats.append({
                            'Emotion': emotion_name,
                            'Code': emotion_code,
                            'Mean Valence': mean_valence,
                            'Mean Arousal': mean_arousal,
                            'Count': count
                        })
                
                emotion_stats_df = pd.DataFrame(emotion_stats)
                
                # Create scatter plot of emotion means
                fig_emotion_va = px.scatter(
                    emotion_stats_df,
                    x='Mean Valence',
                    y='Mean Arousal',
                    size='Count',
                    color='Emotion',
                    text='Emotion',
                    title="Mean Valence-Arousal by Emotion Class",
                    labels={'Mean Valence': 'Mean Valence', 'Mean Arousal': 'Mean Arousal'}
                )
                
                # Add quadrant lines
                fig_emotion_va.add_shape(type="line",
                    x0=0, y0=-1, x1=0, y1=1,
                    line=dict(color="Gray", width=1, dash="dash")
                )
                fig_emotion_va.add_shape(type="line",
                    x0=-1, y0=0, x1=1, y1=0,
                    line=dict(color="Gray", width=1, dash="dash")
                )
                
                fig_emotion_va.update_traces(
                    textposition='top center',
                    marker=dict(size=20)
                )
                
                fig_emotion_va.update_layout(
                    xaxis_range=[-1, 1],
                    yaxis_range=[-1, 1],
                    height=500
                )
                
                st.plotly_chart(fig_emotion_va, use_container_width=True)
                
                # Show table of emotion statistics
                st.dataframe(
                    emotion_stats_df[['Emotion', 'Mean Valence', 'Mean Arousal', 'Count']]
                    .round(4)
                    .sort_values('Mean Valence'),
                    use_container_width=True
                )
        
        else:
            st.warning("Valence and/or arousal columns not found in the ground truth file.")
            st.info("The ground truth CSV should contain 'valence' and 'arousal' columns with values between -1 and 1.")
    
    with tab5:
        st.header("🖼️ Sample Details with Images")
        
        # Create detailed dataframe
        detailed_df = pd.DataFrame()
        
        for i in range(len(gt_df)):
            true_dist = gt_df[emotions].iloc[i].values
            pred_dist = estim_df[emotions].iloc[i].values
            
            # Main emotions
            true_main = emotions[np.argmax(true_dist)]
            pred_main = emotions[np.argmax(pred_dist)]
            
            # Codes for classification
            true_code = gt_df['emotion_label'].iloc[i]
            pred_code = None
            for k, v in emotion_map.items():
                if v == pred_main:
                    pred_code = k
                    break
            
            # Metrics
            metrics = calculate_distribution_metrics(true_dist, pred_dist)
            
            # File path
            file_path = gt_df['file'].iloc[i] if 'file' in gt_df.columns else f"sample_{i}"
            
            # Add valence and arousal if available
            row_data = {
                'Sample': i,
                'File': os.path.basename(file_path),
                'Full_Path': file_path,
                'True_Emotion': emotion_map.get(true_code, true_code),
                'True_Code': true_code,
                'Pred_Emotion': pred_main,
                'Pred_Code': pred_code,
                'Match': true_code == pred_code,
                **metrics
            }
            
            # Add valence and arousal if available
            if 'valence' in gt_df.columns:
                row_data['Valence'] = gt_df['valence'].iloc[i]
            if 'arousal' in gt_df.columns:
                row_data['Arousal'] = gt_df['arousal'].iloc[i]
            
            detailed_df = pd.concat([detailed_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Sample selector for detailed view
        st.subheader("Select Sample for Detailed View")
        
        selected_sample = st.selectbox(
            "Choose a sample to view details:",
            options=range(len(detailed_df)),
            format_func=lambda x: f"Sample {x}: {detailed_df.loc[x, 'True_Emotion']} → {detailed_df.loc[x, 'Pred_Emotion']} ({'✓' if detailed_df.loc[x, 'Match'] else '✗'})"
        )
        
        if selected_sample is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image for selected sample
                image_path = detailed_df.loc[selected_sample, 'Full_Path']
                
                # Use base path if provided
                if base_image_path and os.path.exists(base_image_path):
                    filename = detailed_df.loc[selected_sample, 'File']
                    full_image_path = os.path.join(base_image_path, filename)
                else:
                    full_image_path = image_path
                
                st.subheader(f"Image - Sample {selected_sample}")
                img = load_image(full_image_path)
                
                if img:
                    st.image(img, caption=f"Sample {selected_sample}: {detailed_df.loc[selected_sample, 'File']}", 
                            use_column_width=True)
                else:
                    st.warning(f"Image not found at: {full_image_path}")
                    st.write(f"Image path from data: {image_path}")
                    st.write(f"Tried path: {full_image_path}")
                
                # Show key information
                st.markdown("### Sample Information")
                st.write(f"**File:** {detailed_df.loc[selected_sample, 'File']}")
                st.write(f"**True Emotion:** {detailed_df.loc[selected_sample, 'True_Emotion']}")
                st.write(f"**Predicted Emotion:** {detailed_df.loc[selected_sample, 'Pred_Emotion']}")
                
                # Show valence and arousal if available
                if 'Valence' in detailed_df.columns and 'Arousal' in detailed_df.columns:
                    st.write(f"**Valence:** {detailed_df.loc[selected_sample, 'Valence']:.4f}")
                    st.write(f"**Arousal:** {detailed_df.loc[selected_sample, 'Arousal']:.4f}")
                
                match_status = detailed_df.loc[selected_sample, 'Match']
                if match_status:
                    st.success("✓ Classification Match")
                else:
                    st.error("✗ Classification Mismatch")
            
            with col2:
                # Show detailed metrics for this sample
                st.subheader(f"Detailed Metrics - Sample {selected_sample}")
                
                # Get data for this sample
                sample_row = detailed_df.loc[selected_sample]
                
                # Create metrics display
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown("**Distribution Metrics:**")
                    st.metric("KL Divergence", f"{sample_row['KL_Divergence']:.4f}")
                    st.metric("Jensen-Shannon", f"{sample_row['Jensen_Shannon_Divergence']:.4f}")
                    st.metric("Cosine Similarity", f"{sample_row['Cosine_Similarity']:.4f}")
                    st.metric("Pearson Correlation", f"{sample_row['Pearson_Correlation']:.4f}")
                
                with metrics_col2:
                    st.markdown("**Distance Metrics:**")
                    st.metric("Euclidean Distance", f"{sample_row['Euclidean_Distance']:.4f}")
                    st.metric("Hellinger Distance", f"{sample_row['Hellinger_Distance']:.4f}")
                    st.metric("EMD", f"{sample_row['EMD']:.4f}")
                    st.metric("Intersection", f"{sample_row['Intersection']:.4f}")
                
                # Show valence-arousal for this sample if available
                if 'Valence' in sample_row and 'Arousal' in sample_row:
                    st.markdown("### Valence-Arousal Coordinates")
                    
                    va_col1, va_col2 = st.columns(2)
                    with va_col1:
                        st.metric("Valence", f"{sample_row['Valence']:.4f}")
                    with va_col2:
                        st.metric("Arousal", f"{sample_row['Arousal']:.4f}")
                
                # Show distributions
                true_dist = gt_df[emotions].iloc[selected_sample]
                pred_dist = estim_df[emotions].iloc[selected_sample]
                
                fig_single_dist = go.Figure()
                
                fig_single_dist.add_trace(go.Bar(
                    x=emotions,
                    y=true_dist.values,
                    name='Ground Truth',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_single_dist.add_trace(go.Bar(
                    x=emotions,
                    y=pred_dist.values,
                    name='Estimate',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_single_dist.update_layout(
                    title=f"Distribution Comparison",
                    xaxis_title="Emotion",
                    yaxis_title="Probability",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_single_dist, use_container_width=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("---")
            st.markdown("**Navigate between samples:**")
            
            nav_col1, nav_col2, nav_col3 = st.columns(3)
            with nav_col1:
                if st.button("← Previous", key="prev_sample"):
                    if selected_sample > 0:
                        st.session_state.selected_sample = selected_sample - 1
                        st.rerun()
            
            with nav_col2:
                st.write(f"**{selected_sample + 1} / {len(detailed_df)}**")
            
            with nav_col3:
                if st.button("Next →", key="next_sample"):
                    if selected_sample < len(detailed_df) - 1:
                        st.session_state.selected_sample = selected_sample + 1
                        st.rerun()
        
        # Option to download all results
        st.markdown("---")
        csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Results (CSV)",
            data=csv,
            file_name="detailed_results.csv",
            mime="text/csv"
        )
    
    with tab6:
        st.header("File Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ground Truth (labels.csv)")
            st.write(f"**Total samples:** {len(gt_df)}")
            st.write(f"**Columns:** {list(gt_df.columns)}")
            st.write("**Label distribution:**")
            label_counts = gt_df['emotion_label'].value_counts().sort_index()
            for label, count in label_counts.items():
                st.write(f"  {emotion_map.get(label, label)} ({label}): {count} samples")
            
            # Check if image paths exist
            if 'file' in gt_df.columns:
                st.write("**Image paths check:**")
                first_path = gt_df['file'].iloc[0]
                st.write(f"First path: `{first_path}`")
                
                # Try to load first image
                img = load_image(first_path)
                if img:
                    st.success("✓ First image path is accessible")
                else:
                    st.warning("⚠ First image path not accessible")
            
            # Check valence-arousal ranges
            if 'valence' in gt_df.columns and 'arousal' in gt_df.columns:
                st.write("**Valence-Arousal Ranges:**")
                st.write(f"Valence: [{gt_df['valence'].min():.3f}, {gt_df['valence'].max():.3f}]")
                st.write(f"Arousal: [{gt_df['arousal'].min():.3f}, {gt_df['arousal'].max():.3f}]")
                
                # Check if values are within -1 to 1
                valence_in_range = (gt_df['valence'] >= -1).all() and (gt_df['valence'] <= 1).all()
                arousal_in_range = (gt_df['arousal'] >= -1).all() and (gt_df['arousal'] <= 1).all()
                
                if valence_in_range and arousal_in_range:
                    st.success("✓ Valence and arousal values are within [-1, 1] range")
                else:
                    st.warning("⚠ Some valence/arousal values are outside [-1, 1] range")
            
            st.dataframe(gt_df.head(), use_container_width=True)
        
        with col2:
            st.subheader("Estimates (estim.csv)")
            st.write(f"**Total samples:** {len(estim_df)}")
            st.write(f"**Columns:** {list(estim_df.columns)}")
            
            # Check consistency
            if len(gt_df) != len(estim_df):
                st.warning(f"⚠️ Different number of samples: GT={len(gt_df)}, Est={len(estim_df)}")
            
            st.dataframe(estim_df.head(), use_container_width=True)
        
        # Emotion mapping
        st.subheader("Emotion Mapping")
        mapping_df = pd.DataFrame(list(emotion_map.items()), columns=['Code', 'Emotion'])
        st.dataframe(mapping_df, use_container_width=True)
        
        # Image loading instructions
        st.subheader("📸 Image Display Configuration")
        st.markdown("""
        To display images properly:
        1. **Option 1:** Enter the base path to your images folder in the sidebar
        2. **Option 2:** Ensure image paths in the CSV are absolute and accessible
        3. **Option 3:** Place images in a folder named `images` in the same directory as this app
        
        The app will try to load images from:
        - The full path in the CSV
        - Base path + filename (if base path provided)
        - Local `images` folder
        """)

else:
    st.info("👈 Please upload the CSV files in the sidebar to start the analysis.")
    
    # Show expected structure
    st.subheader("Expected File Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **labels.csv (Ground Truth):**
        - Distribution columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral
        - Additional columns: valence, arousal, dominance, emotion_label, file
        - emotion_label: numeric emotion code (0-7)
        - file: path to image file
        - valence, arousal: values between -1 and 1
        """)
    
    with col2:
        st.markdown("""
        **estim.csv (Estimates):**
        - Distribution columns: happy, contempt, surprised, angry, disgusted, fearful, sad, neutral
        - Additional columns: emotion_label (optional), file
        - Same emotion order as labels.csv
        """)
    
    st.markdown("---")
    st.markdown("**Default emotion mapping:**")
    default_mapping_df = pd.DataFrame(list(default_emotion_map.items()), columns=['Code', 'Emotion'])
    st.dataframe(default_mapping_df, use_container_width=True)
    
    # Image display instructions
    st.markdown("---")
    st.markdown("""
    ### 🖼️ Image Display Feature
    
    This app can display facial images alongside the emotion analysis. To use this feature:
    
    1. **CSV files must contain a `file` column** with image paths
    2. **Images should be accessible** from the file system
    3. **Use the sidebar input** to specify the base folder if needed
    
    ### 📊 Valence-Arousal Space
    
    The app includes a dedicated tab for visualizing samples in the valence-arousal space:
    - **Valence**: Negative ↔ Positive (-1 to 1)
    - **Arousal**: Low ↔ High (-1 to 1)
    - **Quadrant analysis** for emotional states
    """)

# Footer
st.markdown("---")
st.markdown("""
**Implemented Features:**
- 📊 Distribution metrics comparison
- 🎯 Classification accuracy analysis
- 📊 Valence-Arousal space visualization
- 🖼️ Image display for samples
- 📈 Interactive visualizations
- 📋 Detailed sample inspection
- 📁 File validation and consistency checks

**Valence-Arousal Space**: Visualize emotional states in the 2D space of valence (negative-positive) and arousal (low-high).
""")