import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Emotion Dimensions Visualizer", layout="wide")

# Application title
st.title("📊 Emotion Dimensions Visualizer")
st.markdown("Upload a CSV file in the emotions format and visualize distributions in 2D or 3D.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns are present
        required_columns = ['class', 'valence mean', 'valence std', 'arousal mean', 'arousal std']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV file must contain the columns: {', '.join(required_columns)}")
        else:
            # Sidebar for settings
            st.sidebar.header("Visualization Settings")
            
            # Dimension selection
            dimension = st.sidebar.radio(
                "Select dimensionality:",
                ["2D (Valence-Arousal)", "3D (Valence-Arousal-Dominance)"]
            )
            
            # Emotion selection for plotting
            available_emotions = df['class'].unique().tolist()
            selected_emotions = st.sidebar.multiselect(
                "Select emotions to visualize:",
                available_emotions,
                default=available_emotions
            )
            
            # Filter data based on selection
            filtered_df = df[df['class'].isin(selected_emotions)]
            
            # Plot settings
            st.sidebar.subheader("Plot Settings")
            show_std = st.sidebar.checkbox("Show standard deviation", value=True)
            
            if dimension == "2D (Valence-Arousal)":
                # Interactive 2D Plot with Plotly
                fig = go.Figure()
                
                # Colors for different emotions
                colors = px.colors.qualitative.Set3
                
                for i, (idx, row) in enumerate(filtered_df.iterrows()):
                    color = colors[i % len(colors)]
                    
                    # Add mean point
                    fig.add_trace(go.Scatter(
                        x=[row['valence mean']],
                        y=[row['arousal mean']],
                        mode='markers+text',
                        name=row['class'],
                        marker=dict(
                            size=15,
                            color=color,
                            line=dict(width=2, color='black')
                        ),
                        text=row['class'],
                        textposition="top center",
                        hoverinfo='text',
                        hovertext=f"{row['class']}<br>Valence: {row['valence mean']:.2f}±{row['valence std']:.2f}<br>Arousal: {row['arousal mean']:.2f}±{row['arousal std']:.2f}"
                    ))
                    
                    # Add standard deviation ellipse if requested
                    if show_std:
                        # Generate ellipse points
                        theta = np.linspace(0, 2*np.pi, 100)
                        x_ellipse = row['valence mean'] + row['valence std'] * np.cos(theta)
                        y_ellipse = row['arousal mean'] + row['arousal std'] * np.sin(theta)
                        
                        fig.add_trace(go.Scatter(
                            x=x_ellipse,
                            y=y_ellipse,
                            mode='lines',
                            line=dict(color=color, width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip',
                            opacity=0.5
                        ))
                
                # Update layout
                fig.update_layout(
                    title='Emotion Dimensional Space (Valence-Arousal)',
                    xaxis_title='Valence',
                    yaxis_title='Arousal',
                    width=800,
                    height=600,
                    showlegend=True,
                    hovermode='closest'
                )
                
                # Add reference lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Set axis limits
                fig.update_xaxes(range=[-1, 1])
                fig.update_yaxes(range=[-1, 1])
                
                # Display the interactive plot
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # 3D
                # Check if dominance columns are present
                if 'dominance mean' not in df.columns or 'dominance std' not in df.columns:
                    st.error("For 3D visualization, the file must contain 'dominance mean' and 'dominance std' columns")
                else:
                    # Interactive 3D Plot with Plotly
                    fig = go.Figure()
                    
                    # Colors for different emotions
                    colors = px.colors.qualitative.Set3
                    
                    # Additional 3D settings
                    st.sidebar.subheader("3D Visualization Options")
                    show_error_bars = st.sidebar.checkbox("Show error bars instead of ellipsoids", value=True)
                    
                    for i, (idx, row) in enumerate(filtered_df.iterrows()):
                        color = colors[i % len(colors)]
                        
                        # Add mean point
                        fig.add_trace(go.Scatter3d(
                            x=[row['valence mean']],
                            y=[row['arousal mean']],
                            z=[row['dominance mean']],
                            mode='markers+text',
                            name=row['class'],
                            marker=dict(
                                size=8,
                                color=color,
                                line=dict(width=2, color='black')
                            ),
                            text=row['class'],
                            textposition="top center",
                            hoverinfo='text',
                            hovertext=f"""
                            {row['class']}<br>
                            Valence: {row['valence mean']:.2f}±{row['valence std']:.2f}<br>
                            Arousal: {row['arousal mean']:.2f}±{row['arousal std']:.2f}<br>
                            Dominance: {row['dominance mean']:.2f}±{row['dominance std']:.2f}
                            """
                        ))
                        
                        # Add error bars or ellipsoids for standard deviation
                        if show_std:
                            if show_error_bars:
                                # Add error bars (much clearer in 3D)
                                fig.add_trace(go.Scatter3d(
                                    x=[row['valence mean'], row['valence mean']],
                                    y=[row['arousal mean'], row['arousal mean']],
                                    z=[row['dominance mean'] - row['dominance std'], 
                                       row['dominance mean'] + row['dominance std']],
                                    mode='lines',
                                    line=dict(color=color, width=4),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig.add_trace(go.Scatter3d(
                                    x=[row['valence mean'], row['valence mean']],
                                    y=[row['arousal mean'] - row['arousal std'], 
                                       row['arousal mean'] + row['arousal std']],
                                    z=[row['dominance mean'], row['dominance mean']],
                                    mode='lines',
                                    line=dict(color=color, width=4),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig.add_trace(go.Scatter3d(
                                    x=[row['valence mean'] - row['valence std'], 
                                       row['valence mean'] + row['valence std']],
                                    y=[row['arousal mean'], row['arousal mean']],
                                    z=[row['dominance mean'], row['dominance mean']],
                                    mode='lines',
                                    line=dict(color=color, width=4),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                            else:
                                # Add transparent ellipsoid (optional - can be computationally heavy)
                                # Generate ellipsoid points
                                u = np.linspace(0, 2 * np.pi, 20)
                                v = np.linspace(0, np.pi, 20)
                                
                                x_ellipsoid = row['valence mean'] + row['valence std'] * np.outer(np.cos(u), np.sin(v))
                                y_ellipsoid = row['arousal mean'] + row['arousal std'] * np.outer(np.sin(u), np.sin(v))
                                z_ellipsoid = row['dominance mean'] + row['dominance std'] * np.outer(np.ones(np.size(u)), np.cos(v))
                                
                                fig.add_trace(go.Surface(
                                    x=x_ellipsoid,
                                    y=y_ellipsoid,
                                    z=z_ellipsoid,
                                    colorscale=[[0, color], [1, color]],
                                    showscale=False,
                                    opacity=0.2,
                                    hoverinfo='skip'
                                ))
                
                    # Update 3D layout
                    fig.update_layout(
                        title='Three-Dimensional Emotion Space (Valence-Arousal-Dominance)',
                        scene=dict(
                            xaxis_title='Valence',
                            yaxis_title='Arousal',
                            zaxis_title='Dominance',
                            xaxis=dict(range=[-1, 1]),
                            yaxis=dict(range=[-1, 1]),
                            zaxis=dict(range=[-1, 1]),
                        ),
                        width=800,
                        height=700,
                        showlegend=True
                    )
                    
                    # Display the interactive 3D plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add 3D interaction instructions
                    st.sidebar.info("""
                    **3D Interaction Guide:**
                    - **Rotate**: Click and drag
                    - **Zoom**: Scroll wheel
                    - **Pan**: Shift + click and drag
                    - **Reset**: Double click
                    """)
            
            # Show tabular data in an expandable section
            with st.expander("View Loaded Data"):
                st.subheader("Loaded Data")
                st.dataframe(filtered_df)
                
                st.subheader("Descriptive Statistics")
                st.dataframe(filtered_df.describe())
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Instructions when no file has been uploaded
    st.info("""
    ### Instructions:
    1. **Upload** a CSV file in the specified format
    2. **Select** dimensionality (2D or 3D)
    3. **Choose** which emotions to visualize
    4. **Adjust** plot settings as needed
    
    ### Expected CSV format:
    - Columns: `class`, `valence mean`, `valence std`, `arousal mean`, `arousal std`
    - For 3D: also `dominance mean` and `dominance std`
    - Format example:
    ```
    class,valence mean,valence std,arousal mean,arousal std,dominance mean,dominance std
    happy,0.81,0.21,0.51,0.26,0.46,0.38
    sad,-0.63,0.23,-0.27,0.34,-0.33,0.22
    ```
    """)