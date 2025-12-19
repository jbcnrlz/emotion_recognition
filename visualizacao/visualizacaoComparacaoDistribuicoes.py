import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis, euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Emotional Distributions Analysis", layout="wide")
st.title("📊 Comparative Analysis of Emotional Distributions")
st.markdown("""
This application compares emotional distributions between two CSV files containing means and standard deviations 
for valence, arousal and dominance of different emotions.
""")

# Functions for statistical calculations
def calculate_metrics_2d(mean1, cov1, mean2, cov2):
    """Calculates metrics for 2D distributions"""
    metrics = {}
    
    # Euclidean distance between means
    metrics['euclidean_distance'] = euclidean(mean1, mean2)
    
    # Mahalanobis distance (using covariance of first distribution)
    try:
        inv_cov = np.linalg.inv(cov1)
        metrics['mahalanobis_distance'] = mahalanobis(mean1, mean2, inv_cov)
    except:
        metrics['mahalanobis_distance'] = np.nan
    
    # Symmetric KL divergence
    try:
        kl_1_2 = 0.5 * (np.trace(np.linalg.inv(cov2) @ cov1) + 
                        (mean2 - mean1).T @ np.linalg.inv(cov2) @ (mean2 - mean1) - 
                        2 + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
        kl_2_1 = 0.5 * (np.trace(np.linalg.inv(cov1) @ cov2) + 
                        (mean1 - mean2).T @ np.linalg.inv(cov1) @ (mean1 - mean2) - 
                        2 + np.log(np.linalg.det(cov1) / np.linalg.det(cov2)))
        metrics['kl_divergence_symmetric'] = (kl_1_2 + kl_2_1) / 2
    except:
        metrics['kl_divergence_symmetric'] = np.nan
    
    # Probability overlap (approximation)
    try:
        # Using bivariate normal distribution
        diff_mean = mean2 - mean1
        combined_cov = cov1 + cov2
        mahal_sq = diff_mean.T @ np.linalg.inv(combined_cov) @ diff_mean
        metrics['probability_overlap'] = 1 - stats.chi2.cdf(mahal_sq, df=2)
    except:
        metrics['probability_overlap'] = np.nan
    
    # Pearson correlation between mean vectors
    metrics['pearson_correlation'] = np.corrcoef(mean1, mean2)[0, 1]
    
    return metrics

def calculate_metrics_3d(mean1, cov1, mean2, cov2):
    """Calculates metrics for 3D distributions"""
    metrics = {}
    
    # Euclidean distance between means
    metrics['euclidean_distance'] = euclidean(mean1, mean2)
    
    # Mahalanobis distance
    try:
        inv_cov = np.linalg.inv(cov1)
        metrics['mahalanobis_distance'] = mahalanobis(mean1, mean2, inv_cov)
    except:
        metrics['mahalanobis_distance'] = np.nan
    
    # Symmetric KL divergence
    try:
        kl_1_2 = 0.5 * (np.trace(np.linalg.inv(cov2) @ cov1) + 
                        (mean2 - mean1).T @ np.linalg.inv(cov2) @ (mean2 - mean1) - 
                        3 + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
        kl_2_1 = 0.5 * (np.trace(np.linalg.inv(cov1) @ cov2) + 
                        (mean1 - mean2).T @ np.linalg.inv(cov1) @ (mean1 - mean2) - 
                        3 + np.log(np.linalg.det(cov1) / np.linalg.det(cov2)))
        metrics['kl_divergence_symmetric'] = (kl_1_2 + kl_2_1) / 2
    except:
        metrics['kl_divergence_symmetric'] = np.nan
    
    # Wasserstein distance (for normal distributions)
    try:
        trace_term = np.trace(cov1 + cov2 - 2 * np.sqrt(np.sqrt(cov1) @ cov2 @ np.sqrt(cov1)))
        metrics['wasserstein_distance'] = np.sqrt(euclidean(mean1, mean2)**2 + trace_term)
    except:
        metrics['wasserstein_distance'] = np.nan
    
    # Relative volume of confidence ellipsoids (95%)
    try:
        det1 = np.linalg.det(cov1)
        det2 = np.linalg.det(cov2)
        metrics['volume_ratio'] = det2 / det1 if det1 != 0 else np.nan
    except:
        metrics['volume_ratio'] = np.nan
    
    # Cosine similarity between mean vectors
    metrics['cosine_similarity'] = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
    
    return metrics

def create_covariance_matrix(valence_std, arousal_std, dominance_std=None):
    """Creates diagonal covariance matrix from standard deviations"""
    if dominance_std is not None:
        variances = [valence_std**2, arousal_std**2, dominance_std**2]
        return np.diag(variances)
    else:
        variances = [valence_std**2, arousal_std**2]
        return np.diag(variances)

def create_russell_circumplex_background():
    """Creates Russell's Circumplex Model as background for 2D plot"""
    # Define quadrants and emotions in Russell's Circumplex
    quadrants = {
        'High Arousal, Positive Valence': {'x_range': [0, 1], 'y_range': [0, 1], 'color': 'rgba(255, 200, 200, 0.1)'},
        'High Arousal, Negative Valence': {'x_range': [-1, 0], 'y_range': [0, 1], 'color': 'rgba(200, 200, 255, 0.1)'},
        'Low Arousal, Negative Valence': {'x_range': [-1, 0], 'y_range': [-1, 0], 'color': 'rgba(200, 255, 200, 0.1)'},
        'Low Arousal, Positive Valence': {'x_range': [0, 1], 'y_range': [-1, 0], 'color': 'rgba(255, 255, 200, 0.1)'}
    }
    
    # Russell's circumplex model - typical emotional positions
    russell_emotions = {
        'Excited': {'valence': 0.8, 'arousal': 0.8},
        'Happy': {'valence': 0.7, 'arousal': 0.4},
        'Content': {'valence': 0.6, 'arousal': 0.1},
        'Relaxed': {'valence': 0.4, 'arousal': -0.3},
        'Calm': {'valence': 0.2, 'arousal': -0.6},
        'Bored': {'valence': -0.2, 'arousal': -0.7},
        'Sad': {'valence': -0.6, 'arousal': -0.4},
        'Depressed': {'valence': -0.8, 'arousal': -0.2},
        'Distressed': {'valence': -0.7, 'arousal': 0.3},
        'Angry': {'valence': -0.6, 'arousal': 0.7},
        'Afraid': {'valence': -0.4, 'arousal': 0.8},
        'Aroused': {'valence': 0.1, 'arousal': 0.9}
    }
    
    return quadrants, russell_emotions

# Main interface
st.sidebar.header("📁 File Upload")
file1 = st.sidebar.file_uploader("First CSV file", type=['csv'])
file2 = st.sidebar.file_uploader("Second CSV file", type=['csv'])

analysis_type = st.sidebar.selectbox(
    "Analysis type",
    ["Both (2D and 3D)", "Only Valence and Arousal (2D)", "Valence, Arousal and Dominance (3D)"]
)

threshold = st.sidebar.slider(
    "Threshold for high similarity",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="Similarity above this value is considered high"
)

# Add option to show emotion labels
show_emotion_labels = st.sidebar.checkbox("Show emotion labels on plot", value=True)

if file1 and file2:
    try:
        # Load data
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Check structure
        required_cols = ['class', 'valence mean', 'valence std', 'arousal mean', 'arousal std', 
                        'dominance mean', 'dominance std']
        
        if not all(col in df1.columns for col in required_cols):
            st.error("First file does not have the expected structure")
            st.stop()
            
        if not all(col in df2.columns for col in required_cols):
            st.error("Second file does not have the expected structure")
            st.stop()
        
        # Ensure same emotion order
        df1 = df1.sort_values('class').reset_index(drop=True)
        df2 = df2.sort_values('class').reset_index(drop=True)
        
        # Check if classes are equal
        if not df1['class'].equals(df2['class']):
            st.warning("⚠️ Classes in files are not identical. Comparing only common classes.")
            common_classes = list(set(df1['class']) & set(df2['class']))
            df1 = df1[df1['class'].isin(common_classes)].sort_values('class').reset_index(drop=True)
            df2 = df2[df2['class'].isin(common_classes)].sort_values('class').reset_index(drop=True)
        
        if len(df1) == 0:
            st.error("No common classes for comparison")
            st.stop()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 General Analysis", "🔍 Details by Emotion", "📊 Visualizations", "📋 Complete Metrics"])
        
        with tab1:
            st.header("General Distribution Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Number of Emotions", len(df1))
            
            with col2:
                # Average similarity of means (valence + arousal)
                mean_diff = np.mean([
                    euclidean([row1['valence mean'], row1['arousal mean']], 
                             [row2['valence mean'], row2['arousal mean']])
                    for _, row1 in df1.iterrows() for _, row2 in df2[df2['class'] == row1['class']].iterrows()
                ])
                st.metric("Average 2D Difference", f"{mean_diff:.3f}")
            
            with col3:
                # Average similarity of means (3D)
                mean_diff_3d = np.mean([
                    euclidean([row1['valence mean'], row1['arousal mean'], row1['dominance mean']], 
                             [row2['valence mean'], row2['arousal mean'], row2['dominance mean']])
                    for _, row1 in df1.iterrows() for _, row2 in df2[df2['class'] == row1['class']].iterrows()
                ])
                st.metric("Average 3D Difference", f"{mean_diff_3d:.3f}")
            
            # 2D Analysis
            if analysis_type in ["Both (2D and 3D)", "Only Valence and Arousal (2D)"]:
                st.subheader("2D Analysis (Valence and Arousal)")
                
                metrics_2d = []
                for idx in range(len(df1)):
                    class_name = df1.iloc[idx]['class']
                    row1 = df1.iloc[idx]
                    row2 = df2[df2['class'] == class_name].iloc[0]
                    
                    mean1 = np.array([row1['valence mean'], row1['arousal mean']])
                    mean2 = np.array([row2['valence mean'], row2['arousal mean']])
                    cov1 = create_covariance_matrix(row1['valence std'], row1['arousal std'])
                    cov2 = create_covariance_matrix(row2['valence std'], row2['arousal std'])
                    
                    metrics = calculate_metrics_2d(mean1, cov1, mean2, cov2)
                    metrics['class'] = class_name
                    metrics_2d.append(metrics)
                
                df_metrics_2d = pd.DataFrame(metrics_2d)
                
                # Statistical summary
                st.write("**2D Metrics Summary:**")
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    avg_euclidean = df_metrics_2d['euclidean_distance'].mean()
                    st.metric("Average Euclidean Distance", f"{avg_euclidean:.3f}")
                
                with summary_cols[1]:
                    avg_mahal = df_metrics_2d['mahalanobis_distance'].mean()
                    st.metric("Average Mahalanobis Distance", f"{avg_mahal:.3f}")
                
                with summary_cols[2]:
                    avg_kl = df_metrics_2d['kl_divergence_symmetric'].mean()
                    st.metric("Average KL Divergence", f"{avg_kl:.3f}")
                
                with summary_cols[3]:
                    avg_overlap = df_metrics_2d['probability_overlap'].mean()
                    st.metric("Average Overlap", f"{avg_overlap:.3f}")
                
                # Interpretation
                st.write("**2D Interpretation:**")
                if avg_overlap > threshold:
                    st.success("✅ 2D distributions tell similar stories (high overlap)")
                elif avg_overlap > 0.3:
                    st.warning("⚠️ 2D distributions have moderate similarity")
                else:
                    st.error("❌ 2D distributions tell different stories (low overlap)")
            
            # 3D Analysis
            if analysis_type in ["Both (2D and 3D)", "Valence, Arousal and Dominance (3D)"]:
                st.subheader("3D Analysis (Valence, Arousal and Dominance)")
                
                metrics_3d = []
                for idx in range(len(df1)):
                    class_name = df1.iloc[idx]['class']
                    row1 = df1.iloc[idx]
                    row2 = df2[df2['class'] == class_name].iloc[0]
                    
                    mean1 = np.array([row1['valence mean'], row1['arousal mean'], row1['dominance mean']])
                    mean2 = np.array([row2['valence mean'], row2['arousal mean'], row2['dominance mean']])
                    cov1 = create_covariance_matrix(row1['valence std'], row1['arousal std'], row1['dominance std'])
                    cov2 = create_covariance_matrix(row2['valence std'], row2['arousal std'], row2['dominance std'])
                    
                    metrics = calculate_metrics_3d(mean1, cov1, mean2, cov2)
                    metrics['class'] = class_name
                    metrics_3d.append(metrics)
                
                df_metrics_3d = pd.DataFrame(metrics_3d)
                
                # Statistical summary
                st.write("**3D Metrics Summary:**")
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    avg_euclidean = df_metrics_3d['euclidean_distance'].mean()
                    st.metric("Average Euclidean Distance", f"{avg_euclidean:.3f}")
                
                with summary_cols[1]:
                    avg_wasserstein = df_metrics_3d['wasserstein_distance'].mean()
                    st.metric("Average Wasserstein Distance", f"{avg_wasserstein:.3f}")
                
                with summary_cols[2]:
                    avg_kl = df_metrics_3d['kl_divergence_symmetric'].mean()
                    st.metric("Average KL Divergence", f"{avg_kl:.3f}")
                
                with summary_cols[3]:
                    avg_cosine = df_metrics_3d['cosine_similarity'].mean()
                    st.metric("Average Cosine Similarity", f"{avg_cosine:.3f}")
                
                # Interpretation
                st.write("**3D Interpretation:**")
                if avg_cosine > threshold:
                    st.success("✅ 3D distributions tell similar stories (high similarity)")
                elif avg_cosine > 0.3:
                    st.warning("⚠️ 3D distributions have moderate similarity")
                else:
                    st.error("❌ 3D distributions tell different stories (low similarity)")
        
        with tab2:
            st.header("Detailed Analysis by Emotion")
            
            # Emotion selector
            selected_class = st.selectbox("Select an emotion:", df1['class'].unique())
            
            if selected_class:
                row1 = df1[df1['class'] == selected_class].iloc[0]
                row2 = df2[df2['class'] == selected_class].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{selected_class} - File 1:**")
                    st.write(f"Valence: {row1['valence mean']:.3f} ± {row1['valence std']:.3f}")
                    st.write(f"Arousal: {row1['arousal mean']:.3f} ± {row1['arousal std']:.3f}")
                    st.write(f"Dominance: {row1['dominance mean']:.3f} ± {row1['dominance std']:.3f}")
                
                with col2:
                    st.write(f"**{selected_class} - File 2:**")
                    st.write(f"Valence: {row2['valence mean']:.3f} ± {row2['valence std']:.3f}")
                    st.write(f"Arousal: {row2['arousal mean']:.3f} ± {row2['arousal std']:.3f}")
                    st.write(f"Dominance: {row2['dominance mean']:.3f} ± {row2['dominance std']:.3f}")
                
                # Calculate metrics for this emotion
                st.subheader("Similarity Metrics")
                
                # 2D
                mean1_2d = np.array([row1['valence mean'], row1['arousal mean']])
                mean2_2d = np.array([row2['valence mean'], row2['arousal mean']])
                cov1_2d = create_covariance_matrix(row1['valence std'], row1['arousal std'])
                cov2_2d = create_covariance_matrix(row2['valence std'], row2['arousal std'])
                
                metrics_2d = calculate_metrics_2d(mean1_2d, cov1_2d, mean2_2d, cov2_2d)
                
                # 3D
                mean1_3d = np.array([row1['valence mean'], row1['arousal mean'], row1['dominance mean']])
                mean2_3d = np.array([row2['valence mean'], row2['arousal mean'], row2['dominance mean']])
                cov1_3d = create_covariance_matrix(row1['valence std'], row1['arousal std'], row1['dominance std'])
                cov2_3d = create_covariance_matrix(row2['valence std'], row2['arousal std'], row2['dominance std'])
                
                metrics_3d = calculate_metrics_3d(mean1_3d, cov1_3d, mean2_3d, cov2_3d)
                
                # Display metrics
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.write("**2D Metrics:**")
                    for key, value in metrics_2d.items():
                        if key != 'class':
                            st.write(f"{key}: {value:.4f}")
                
                with metric_cols[1]:
                    st.write("**3D Metrics:**")
                    for key, value in metrics_3d.items():
                        if key != 'class':
                            st.write(f"{key}: {value:.4f}")
                
                # Interpretation
                st.subheader("Interpretation")
                if metrics_2d.get('probability_overlap', 0) > threshold and metrics_3d.get('cosine_similarity', 0) > threshold:
                    st.success(f"✅ **{selected_class}**: Distributions are very similar in both spaces")
                elif metrics_2d.get('probability_overlap', 0) > threshold:
                    st.info(f"ℹ️ **{selected_class}**: Similar in 2D space, but may differ in dominance")
                elif metrics_3d.get('cosine_similarity', 0) > threshold:
                    st.info(f"ℹ️ **{selected_class}**: Similar in 3D space, but with differences in 2D distribution")
                else:
                    st.warning(f"⚠️ **{selected_class}**: Distributions are significantly different")
        
        with tab3:
            st.header("Graphical Visualizations")
            
            # Create Russell Circumplex background
            quadrants, russell_emotions = create_russell_circumplex_background()
            
            # Create the 2D plot with Russell Circumplex
            fig_circumplex = go.Figure()
            
            # Add quadrant backgrounds
            for quadrant_name, quadrant_data in quadrants.items():
                fig_circumplex.add_shape(
                    type="rect",
                    x0=quadrant_data['x_range'][0],
                    y0=quadrant_data['y_range'][0],
                    x1=quadrant_data['x_range'][1],
                    y1=quadrant_data['y_range'][1],
                    fillcolor=quadrant_data['color'],
                    line=dict(width=0),
                    layer="below"
                )
            
            # Add axis lines
            fig_circumplex.add_shape(
                type="line",
                x0=-1, y0=0, x1=1, y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            fig_circumplex.add_shape(
                type="line",
                x0=0, y0=-1, x1=0, y1=1,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Add Russell's emotion labels
            for emotion, coords in russell_emotions.items():
                fig_circumplex.add_annotation(
                    x=coords['valence'],
                    y=coords['arousal'],
                    text=emotion,
                    showarrow=False,
                    font=dict(size=9, color="rgba(100, 100, 100, 0.7)"),
                    xshift=10,
                    yshift=10
                )
            
            # Prepare data for plotting
            # Add data points from both files with different markers and labels
            if show_emotion_labels:
                # With labels
                fig_circumplex.add_trace(go.Scatter(
                    x=df1['valence mean'],
                    y=df1['arousal mean'],
                    mode='markers+text',
                    name='File 1',
                    text=df1['class'],
                    textposition="top center",
                    marker=dict(size=12, color='blue', symbol='circle'),
                    showlegend=True
                ))
                
                fig_circumplex.add_trace(go.Scatter(
                    x=df2['valence mean'],
                    y=df2['arousal mean'],
                    mode='markers+text',
                    name='File 2',
                    text=df2['class'],
                    textposition="bottom center",
                    marker=dict(size=10, color='red', symbol='x'),
                    showlegend=True
                ))
            else:
                # Without labels
                fig_circumplex.add_trace(go.Scatter(
                    x=df1['valence mean'],
                    y=df1['arousal mean'],
                    mode='markers',
                    name='File 1',
                    text=df1['class'],
                    marker=dict(size=12, color='blue', symbol='circle'),
                    showlegend=True
                ))
                
                fig_circumplex.add_trace(go.Scatter(
                    x=df2['valence mean'],
                    y=df2['arousal mean'],
                    mode='markers',
                    name='File 2',
                    text=df2['class'],
                    marker=dict(size=10, color='red', symbol='x'),
                    showlegend=True
                ))
            
            # Add error bars for File 1
            for idx, row in df1.iterrows():
                fig_circumplex.add_shape(
                    type="circle",
                    x0=row['valence mean'] - row['valence std'],
                    y0=row['arousal mean'] - row['arousal std'],
                    x1=row['valence mean'] + row['valence std'],
                    y1=row['arousal mean'] + row['arousal std'],
                    line=dict(color="blue", width=1, dash="dot"),
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    opacity=0.3
                )
            
            # Add error bars for File 2
            for idx, row in df2.iterrows():
                fig_circumplex.add_shape(
                    type="circle",
                    x0=row['valence mean'] - row['valence std'],
                    y0=row['arousal mean'] - row['arousal std'],
                    x1=row['valence mean'] + row['valence std'],
                    y1=row['arousal mean'] + row['arousal std'],
                    line=dict(color="red", width=1, dash="dot"),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    opacity=0.3
                )
            
            # Add connecting lines between corresponding emotions
            for idx, (row1, row2) in enumerate(zip(df1.iterrows(), df2.iterrows())):
                _, r1 = row1
                _, r2 = row2
                
                if r1['class'] == r2['class']:
                    fig_circumplex.add_trace(go.Scatter(
                        x=[r1['valence mean'], r2['valence mean']],
                        y=[r1['arousal mean'], r2['arousal mean']],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=(idx == 0),
                        name='Distance between files',
                        opacity=0.5
                    ))
            
            # Update layout
            fig_circumplex.update_layout(
                title="2D Emotional Distribution on Russell's Circumplex Model",
                xaxis_title="Valence (Negative ↔ Positive)",
                yaxis_title="Arousal (Calm ↔ Excited)",
                xaxis=dict(range=[-1.1, 1.1]),
                yaxis=dict(range=[-1.1, 1.1]),
                height=600,
                showlegend=True,
                plot_bgcolor='white',
                hovermode='closest'
            )
            
            # Add hover information
            fig_circumplex.update_traces(
                hovertemplate="<b>%{text}</b><br>Valence: %{x:.3f}<br>Arousal: %{y:.3f}<extra></extra>"
            )
            
            # Add quadrant labels
            fig_circumplex.add_annotation(
                x=0.75, y=0.75,
                text="High Arousal<br>Positive Valence",
                showarrow=False,
                font=dict(size=10, color="darkred"),
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
            fig_circumplex.add_annotation(
                x=-0.75, y=0.75,
                text="High Arousal<br>Negative Valence",
                showarrow=False,
                font=dict(size=10, color="darkblue"),
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
            fig_circumplex.add_annotation(
                x=-0.75, y=-0.75,
                text="Low Arousal<br>Negative Valence",
                showarrow=False,
                font=dict(size=10, color="darkgreen"),
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
            fig_circumplex.add_annotation(
                x=0.75, y=-0.75,
                text="Low Arousal<br>Positive Valence",
                showarrow=False,
                font=dict(size=10, color="darkorange"),
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
            
            st.plotly_chart(fig_circumplex, use_container_width=True)
            
            # Add explanation about the plot
            with st.expander("About this visualization"):
                st.markdown("""
                **Russell's Circumplex Model of Emotions:**
                - **Valence (X-axis):** Negative (left) to Positive (right) emotional experience
                - **Arousal (Y-axis):** Calm/Low energy (bottom) to Excited/High energy (top)
                
                **What you're seeing:**
                - **Blue circles (File 1):** First file's emotional distributions
                - **Red X marks (File 2):** Second file's emotional distributions  
                - **Gray dashed lines:** Connect corresponding emotions between files
                - **Faded circles:** Represent ±1 standard deviation (uncertainty)
                - **Quadrant colors:** Different emotional categories in Russell's model
                
                **Interpretation:**
                - Emotions close together = similar distributions
                - Long connecting lines = significant differences
                - Large circles = high uncertainty/variability
                """)
            
            # Original box plots
            st.subheader("Dimensional Comparison")
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Valence', 'Arousal', 'Dominance'),
                specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}]]
            )
            
            for idx, dim in enumerate(['valence mean', 'arousal mean', 'dominance mean']):
                fig.add_trace(go.Box(
                    y=df1[dim],
                    name='File 1',
                    marker_color='blue',
                    showlegend=(idx == 0)
                ), row=1, col=idx+1)
                
                fig.add_trace(go.Box(
                    y=df2[dim],
                    name='File 2',
                    marker_color='red',
                    showlegend=(idx == 0)
                ), row=1, col=idx+1)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Matrix Between Files")
            
            corr_data = []
            for dim in ['valence mean', 'arousal mean', 'dominance mean']:
                corr = np.corrcoef(df1[dim], df2[dim])[0, 1]
                corr_data.append([dim.replace(' mean', ''), corr])
            
            corr_df = pd.DataFrame(corr_data, columns=['Dimension', 'Correlation'])
            
            fig2 = go.Figure(data=go.Heatmap(
                z=[corr_df['Correlation']],
                x=corr_df['Dimension'],
                y=['Correlation'],
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_df['Correlation'], 3),
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            
            fig2.update_layout(title="Correlation between files by dimension")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab4:
            st.header("Complete Metrics Table")
            
            # Combine all metrics
            all_metrics = []
            for idx in range(len(df1)):
                class_name = df1.iloc[idx]['class']
                row1 = df1.iloc[idx]
                row2 = df2[df2['class'] == class_name].iloc[0]
                
                # 2D
                mean1_2d = np.array([row1['valence mean'], row1['arousal mean']])
                mean2_2d = np.array([row2['valence mean'], row2['arousal mean']])
                cov1_2d = create_covariance_matrix(row1['valence std'], row1['arousal std'])
                cov2_2d = create_covariance_matrix(row2['valence std'], row2['arousal std'])
                metrics_2d = calculate_metrics_2d(mean1_2d, cov1_2d, mean2_2d, cov2_2d)
                
                # 3D
                mean1_3d = np.array([row1['valence mean'], row1['arousal mean'], row1['dominance mean']])
                mean2_3d = np.array([row2['valence mean'], row2['arousal mean'], row2['dominance mean']])
                cov1_3d = create_covariance_matrix(row1['valence std'], row1['arousal std'], row1['dominance std'])
                cov2_3d = create_covariance_matrix(row2['valence std'], row2['arousal std'], row2['dominance std'])
                metrics_3d = calculate_metrics_3d(mean1_3d, cov1_3d, mean2_3d, cov2_3d)
                
                # Combine
                combined = {'class': class_name}
                for key, value in metrics_2d.items():
                    if key != 'class':
                        combined[f'2d_{key}'] = value
                for key, value in metrics_3d.items():
                    if key != 'class':
                        combined[f'3d_{key}'] = value
                
                all_metrics.append(combined)
            
            df_all_metrics = pd.DataFrame(all_metrics)
            st.dataframe(df_all_metrics, use_container_width=True)
            
            # Option to download
            csv = df_all_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download complete metrics as CSV",
                data=csv,
                file_name="complete_metrics.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.write("Make sure files have the same structure as the provided example.")

else:
    st.info("👈 Please upload two CSV files in the sidebar menu to start analysis.")
    
    # Show example of expected structure
    st.subheader("Expected CSV file structure:")
    example_data = {
        'class': ['happy', 'sad', 'angry'],
        'valence mean': [0.8, -0.6, -0.5],
        'valence std': [0.2, 0.2, 0.2],
        'arousal mean': [0.5, -0.3, 0.6],
        'arousal std': [0.3, 0.3, 0.3],
        'dominance mean': [0.4, -0.3, 0.2],
        'dominance std': [0.4, 0.2, 0.4]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)