import streamlit as st
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Application title
st.title("Emotion Distribution Comparator")
st.markdown("""
This application allows you to compare emotion distributions between two different CSV files for the same image.
Upload the files and select an image to view the corresponding distributions and the image itself.
""")

# File upload
st.sidebar.header("File Upload")
file1 = st.sidebar.file_uploader("Upload first CSV file (simpleNetwork_FocalLoss...)", type="csv")
file2 = st.sidebar.file_uploader("Upload second CSV file (comparacao_affectnet...)", type="csv")

# Initialize session state for image caching
if 'image_cache' not in st.session_state:
    st.session_state.image_cache = {}

def safe_float_conversion(value):
    """Safely convert a value to float"""
    try:
        if isinstance(value, (int, float, np.number)):
            return float(value)
        elif isinstance(value, str):
            # Remove any non-numeric characters except decimal point and minus sign
            cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
            if cleaned:
                return float(cleaned)
            else:
                return 0.0
        else:
            return 0.0
    except:
        return 0.0

def load_image(image_path):
    """Load image from path with caching"""
    if image_path in st.session_state.image_cache:
        return st.session_state.image_cache[image_path]
    
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.session_state.image_cache[image_path] = img
            return img
        else:
            # Try to find the image in different locations
            base_name = os.path.basename(image_path)
            possible_paths = [
                image_path,
                f"./{base_name}",
                f"images/{base_name}",
                f"../images/{base_name}",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    st.session_state.image_cache[image_path] = img
                    return img
            
            return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

if file1 and file2:
    # Load data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Display column info for debugging
    with st.sidebar.expander("Debug Info"):
        st.write("First file columns:", list(df1.columns))
        st.write("Second file columns:", list(df2.columns))
    
    # Process df1 - extract filename from 'file' column
    if 'file' not in df1.columns:
        # Try to find the file column
        file_col = None
        for col in df1.columns:
            if 'file' in col.lower() or col.lower().endswith('file'):
                file_col = col
                break
        
        if file_col:
            df1 = df1.rename(columns={file_col: 'file'})
        else:
            st.error("First file doesn't contain 'file' column. Please check the file format.")
            st.write("Available columns:", list(df1.columns))
            st.stop()
    
    df1 = df1.copy()
    df1['filename'] = df1['file'].apply(lambda x: os.path.basename(str(x)) if pd.notnull(x) else '')
    
    # Process df2 - extract filename from affectnet_image_path column
    if 'affectnet_image_path' not in df2.columns:
        # Try alternative column names
        alt_cols = ['image_path', 'path', 'image_filename', 'filename']
        found_col = None
        for col in alt_cols:
            if col in df2.columns:
                found_col = col
                break
        
        if found_col:
            df2 = df2.rename(columns={found_col: 'affectnet_image_path'})
        else:
            st.error("Second file doesn't contain 'affectnet_image_path' column. Please check the file format.")
            st.write("Available columns:", list(df2.columns))
            st.stop()
    
    df2 = df2.copy()
    df2['filename'] = df2['affectnet_image_path'].apply(lambda x: os.path.basename(str(x)) if pd.notnull(x) else '')
    
    # Clean filenames - remove any directory paths that might be included
    df1['filename'] = df1['filename'].apply(lambda x: os.path.basename(x) if pd.notnull(x) else '')
    df2['filename'] = df2['filename'].apply(lambda x: os.path.basename(x) if pd.notnull(x) else '')
    
    # Create sets of filenames from both files
    df1_filenames = set(df1['filename'].dropna().unique())
    df2_filenames = set(df2['filename'].dropna().unique())
    
    # Find intersection - images present in both files
    common_images = sorted(list(df1_filenames.intersection(df2_filenames)))
    
    # List of emotions from first file (first 8 columns are emotions)
    # Get emotion columns from first file
    emotion_cols_df1 = []
    for col in df1.columns:
        if col in ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']:
            emotion_cols_df1.append(col)
    
    # If we couldn't find them by name, try the first 8 columns
    if not emotion_cols_df1 and len(df1.columns) >= 8:
        emotion_cols_df1 = list(df1.columns[:8])
    
    # List of emotions from second file (csv_ and affectnet_ prefixes)
    csv_emotions = [col for col in df2.columns if col.startswith('csv_')]
    affectnet_emotions = [col for col in df2.columns if col.startswith('affectnet_')]
    
    # Image selection
    st.header("Image Selection")
    
    if len(common_images) > 0:
        # Display count of common images
        st.info(f"Found {len(common_images)} images present in both files.")
        
        selected_image = st.selectbox(
            "Select an image (showing only images present in both files):",
            options=common_images,
            format_func=lambda x: f"{x}"
        )
        
        if selected_image:
            # Find data in first file
            df1_matches = df1[df1['filename'] == selected_image]
            if len(df1_matches) > 0:
                df1_row = df1_matches.iloc[0]
            else:
                st.error(f"Image {selected_image} not found in first file after matching.")
                st.stop()
            
            # Find data in second file
            df2_matches = df2[df2['filename'] == selected_image]
            if len(df2_matches) > 0:
                df2_row = df2_matches.iloc[0]
            else:
                st.error(f"Image {selected_image} not found in second file after matching.")
                st.stop()
            
            # Get image path from first file
            image_path = df1_row['file']
            
            # Display in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Image: {selected_image}")
                
                # Try to load and display image
                img = load_image(image_path)
                
                if img:
                    st.image(img, caption=selected_image, use_column_width=True)
                    
                    # Display image info
                    with st.expander("Image Information"):
                        st.write(f"**Dimensions:** {img.size[0]} x {img.size[1]}")
                        st.write(f"**Mode:** {img.mode}")
                        st.write(f"**Format:** {img.format if hasattr(img, 'format') else 'Unknown'}")
                else:
                    st.warning(f"Could not load image from path: {image_path}")
                    st.write("**Image path stored in data:**")
                    st.code(image_path, language='text')
            
            with col2:
                # Display emotion label if available
                if 'emotion_label' in df1.columns:
                    emotion_label = df1_row['emotion_label']
                    # Convert to int if it's a string representation
                    try:
                        if isinstance(emotion_label, str):
                            emotion_label = int(float(emotion_label))
                        elif isinstance(emotion_label, (int, float)):
                            emotion_label = int(emotion_label)
                    except:
                        pass
                    
                    emotion_labels_map = {
                        0: 'Neutral',
                        1: 'Happy',
                        2: 'Sad',
                        3: 'Surprise',
                        4: 'Fear',
                        5: 'Anger',
                        6: 'Disgust',
                        7: 'Contempt'
                    }
                    label_text = emotion_labels_map.get(int(emotion_label), f"Unknown ({emotion_label})")
                    st.metric("Primary Emotion Label", label_text)
                
                st.metric("Images in common", len(common_images))
                
                # Display file info
                with st.expander("File Information"):
                    st.write(f"**First file path:**")
                    st.code(str(image_path), language='text')
                    if 'affectnet_image_path' in df2_row.index:
                        st.write(f"**Second file path:**")
                        st.code(str(df2_row['affectnet_image_path']), language='text')
            
            # Create tabs for better organization
            tab1, tab2, tab3, tab4 = st.tabs(["First File Distribution", "Second File Distributions", "Side-by-Side Comparison", "Advanced Analysis"])
            
            with tab1:
                st.subheader("Emotion Distribution - First File")
                
                # Get emotion values from first file and ensure they are floats
                emotion_values = []
                for col in emotion_cols_df1:
                    if col in df1_row.index:
                        value = df1_row[col]
                        emotion_values.append(safe_float_conversion(value))
                    else:
                        emotion_values.append(0.0)
                
                # Create dataframe for display
                df1_emotions = pd.DataFrame({
                    'Emotion': emotion_cols_df1,
                    'Value': emotion_values
                })
                
                # Ensure Value column is numeric
                df1_emotions['Value'] = pd.to_numeric(df1_emotions['Value'], errors='coerce').fillna(0)
                
                # Sort by value for better visualization
                df1_emotions = df1_emotions.sort_values('Value', ascending=False)
                
                # Display as dataframe
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.dataframe(df1_emotions.style.format({'Value': '{:.6f}'}), use_container_width=True)
                
                with col2:
                    # Display as pie chart
                    import plotly.express as px
                    fig_pie = px.pie(df1_emotions, values='Value', names='Emotion', 
                                     title='Emotion Distribution (First File)',
                                     hole=0.3)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Bar chart
                fig_bar = px.bar(df1_emotions, x='Emotion', y='Value', 
                                 title='Emotion Distribution Bar Chart',
                                 color='Value',
                                 color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Display affective dimensions if available
                if all(col in df1.columns for col in ['valence', 'arousal', 'dominance']):
                    st.subheader("Affective Dimensions")
                    
                    # Safely get values
                    valence_val = safe_float_conversion(df1_row.get('valence', 0))
                    arousal_val = safe_float_conversion(df1_row.get('arousal', 0))
                    dominance_val = safe_float_conversion(df1_row.get('dominance', 0))
                    
                    dim_data = pd.DataFrame({
                        'Dimension': ['Valence', 'Arousal', 'Dominance'],
                        'Value': [valence_val, arousal_val, dominance_val],
                        'Description': ['Positive/Negative', 'Calm/Excited', 'Submissive/Dominant']
                    })
                    
                    st.dataframe(dim_data.style.format({'Value': '{:.4f}'}), use_container_width=True)
                    
                    # Radar chart
                    fig_radar = px.line_polar(dim_data, r='Value', theta='Dimension', 
                                              line_close=True, 
                                              title='Affective Dimensions Radar Chart',
                                              range_r=[-1, 1])
                    fig_radar.update_traces(fill='toself')
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab2:
                st.subheader("Second File Distributions")
                
                # CSV Distribution
                st.markdown("#### CSV Distribution")
                
                # Get values safely
                csv_values = []
                csv_emotion_names = []
                for col in csv_emotions:
                    if col in df2_row.index:
                        value = df2_row[col]
                        csv_values.append(safe_float_conversion(value))
                        csv_emotion_names.append(col.replace('csv_', ''))
                
                df2_csv_emotions = pd.DataFrame({
                    'Emotion': csv_emotion_names,
                    'Value': csv_values
                })
                
                # Ensure Value column is numeric
                df2_csv_emotions['Value'] = pd.to_numeric(df2_csv_emotions['Value'], errors='coerce').fillna(0)
                df2_csv_emotions = df2_csv_emotions.sort_values('Value', ascending=False)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.dataframe(df2_csv_emotions.style.format({'Value': '{:.6f}'}), use_container_width=True)
                with col2:
                    fig_csv_pie = px.pie(df2_csv_emotions, values='Value', names='Emotion',
                                        title='CSV Distribution', hole=0.3)
                    st.plotly_chart(fig_csv_pie, use_container_width=True)
                
                # AffectNet Distribution
                st.markdown("#### AffectNet Distribution")
                
                # Get values safely
                affectnet_values = []
                affectnet_emotion_names = []
                for col in affectnet_emotions:
                    if col in df2_row.index:
                        value = df2_row[col]
                        affectnet_values.append(safe_float_conversion(value))
                        affectnet_emotion_names.append(col.replace('affectnet_', ''))
                
                df2_affectnet_emotions = pd.DataFrame({
                    'Emotion': affectnet_emotion_names,
                    'Value': affectnet_values
                })
                
                # Ensure Value column is numeric
                df2_affectnet_emotions['Value'] = pd.to_numeric(df2_affectnet_emotions['Value'], errors='coerce').fillna(0)
                df2_affectnet_emotions = df2_affectnet_emotions.sort_values('Value', ascending=False)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.dataframe(df2_affectnet_emotions.style.format({'Value': '{:.6f}'}), use_container_width=True)
                with col2:
                    fig_aff_pie = px.pie(df2_affectnet_emotions, values='Value', names='Emotion',
                                        title='AffectNet Distribution', hole=0.3)
                    st.plotly_chart(fig_aff_pie, use_container_width=True)
                
                # Side-by-side bar chart
                st.markdown("#### Comparison: CSV vs AffectNet")
                
                # Create comparison data
                comparison_data = pd.DataFrame({
                    'Emotion': csv_emotion_names,
                    'CSV': csv_values,
                    'AffectNet': affectnet_values[:len(csv_emotion_names)]  # Ensure same length
                })
                
                # Melt for plotting
                comparison_melted = comparison_data.melt(id_vars=['Emotion'], 
                                                        value_vars=['CSV', 'AffectNet'],
                                                        var_name='Source', 
                                                        value_name='Value')
                
                fig_side = px.bar(comparison_melted, x='Emotion', y='Value', color='Source',
                                 barmode='group', title='CSV vs AffectNet Comparison',
                                 color_discrete_sequence=['#1f77b4', '#ff7f0e'])
                st.plotly_chart(fig_side, use_container_width=True)
            
            with tab3:
                st.subheader("Side-by-Side Comparison")
                
                # Create comparison dataframe for all three sources
                comparison_all = []
                
                for emotion in emotion_cols_df1:
                    # Get values from all sources safely
                    first_file_val = safe_float_conversion(df1_row.get(emotion, 0))
                    csv_val = safe_float_conversion(df2_row.get(f'csv_{emotion}', 0))
                    affectnet_val = safe_float_conversion(df2_row.get(f'affectnet_{emotion}', 0))
                    
                    comparison_all.append({
                        'Emotion': emotion,
                        'First File': first_file_val,
                        'Second CSV': csv_val,
                        'Second AffectNet': affectnet_val
                    })
                
                comparison_df = pd.DataFrame(comparison_all)
                
                # Display comparison table
                st.dataframe(comparison_df.style.format({
                    'First File': '{:.6f}',
                    'Second CSV': '{:.6f}',
                    'Second AffectNet': '{:.6f}'
                }), use_container_width=True)
                
                # Melt for plotting
                comparison_melted = comparison_df.melt(
                    id_vars=['Emotion'],
                    value_vars=['First File', 'Second CSV', 'Second AffectNet'],
                    var_name='Source',
                    value_name='Value'
                )
                
                # Grouped bar chart
                fig_grouped = px.bar(comparison_melted, x='Emotion', y='Value', color='Source',
                                    barmode='group', title='Three-Way Emotion Distribution Comparison',
                                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
                st.plotly_chart(fig_grouped, use_container_width=True)
                
                # Line chart for trends
                fig_line = px.line(comparison_melted, x='Emotion', y='Value', color='Source',
                                  markers=True, title='Distribution Trends Comparison')
                st.plotly_chart(fig_line, use_container_width=True)
                
                # Calculate differences
                st.markdown("#### Differences Analysis")
                
                comparison_df['Diff_CSV_vs_First'] = comparison_df['Second CSV'] - comparison_df['First File']
                comparison_df['Diff_AffectNet_vs_First'] = comparison_df['Second AffectNet'] - comparison_df['First File']
                
                diff_cols = ['Emotion', 'First File', 'Second CSV', 'Second AffectNet', 
                            'Diff_CSV_vs_First', 'Diff_AffectNet_vs_First']
                
                st.dataframe(comparison_df[diff_cols].style.format({
                    'First File': '{:.6f}',
                    'Second CSV': '{:.6f}',
                    'Second AffectNet': '{:.6f}',
                    'Diff_CSV_vs_First': '{:.6f}',
                    'Diff_AffectNet_vs_First': '{:.6f}'
                }).applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green', 
                          subset=['Diff_CSV_vs_First', 'Diff_AffectNet_vs_First']),
                use_container_width=True)
            
            with tab4:
                st.subheader("Advanced Analysis")
                
                # Calculate similarity metrics
                try:
                    from scipy.spatial.distance import cosine, euclidean
                    
                    # Get vectors for comparison (ensure same emotions)
                    first_vec = np.array([safe_float_conversion(df1_row.get(emotion, 0)) for emotion in emotion_cols_df1])
                    
                    # For CSV and AffectNet, match the same emotions
                    csv_vec = []
                    affectnet_vec = []
                    for emotion in emotion_cols_df1:
                        csv_vec.append(safe_float_conversion(df2_row.get(f'csv_{emotion}', 0)))
                        affectnet_vec.append(safe_float_conversion(df2_row.get(f'affectnet_{emotion}', 0)))
                    
                    csv_vec = np.array(csv_vec)
                    affectnet_vec = np.array(affectnet_vec)
                    
                    # Calculate metrics
                    metrics_data = []
                    
                    # Cosine similarity (handle zero vectors)
                    def safe_cosine(vec1, vec2):
                        try:
                            if np.all(vec1 == 0) or np.all(vec2 == 0):
                                return 0.0
                            return 1 - cosine(vec1, vec2)
                        except:
                            return 0.0
                    
                    cos_sim_csv = safe_cosine(first_vec, csv_vec)
                    cos_sim_aff = safe_cosine(first_vec, affectnet_vec)
                    
                    # Euclidean distance
                    euc_dist_csv = euclidean(first_vec, csv_vec) if not (np.all(first_vec == 0) and np.all(csv_vec == 0)) else 0.0
                    euc_dist_aff = euclidean(first_vec, affectnet_vec) if not (np.all(first_vec == 0) and np.all(affectnet_vec == 0)) else 0.0
                    
                    # Manhattan distance
                    man_dist_csv = np.sum(np.abs(first_vec - csv_vec))
                    man_dist_aff = np.sum(np.abs(first_vec - affectnet_vec))
                    
                    metrics_data.extend([
                        {'Metric': 'Cosine Similarity (CSV)', 'Value': cos_sim_csv, 'Type': 'Similarity'},
                        {'Metric': 'Cosine Similarity (AffectNet)', 'Value': cos_sim_aff, 'Type': 'Similarity'},
                        {'Metric': 'Euclidean Distance (CSV)', 'Value': euc_dist_csv, 'Type': 'Distance'},
                        {'Metric': 'Euclidean Distance (AffectNet)', 'Value': euc_dist_aff, 'Type': 'Distance'},
                        {'Metric': 'Manhattan Distance (CSV)', 'Value': man_dist_csv, 'Type': 'Distance'},
                        {'Metric': 'Manhattan Distance (AffectNet)', 'Value': man_dist_aff, 'Type': 'Distance'}
                    ])
                    
                    # Add original metrics from second file if available
                    similarity_cols = ['cosine_similarity', 'euclidean_distance', 
                                      'manhattan_distance', 'correlation', 'js_divergence']
                    
                    for col in similarity_cols:
                        if col in df2.columns:
                            metrics_data.append({
                                'Metric': f'Original {col.replace("_", " ").title()}',
                                'Value': safe_float_conversion(df2_row[col]),
                                'Type': 'Original'
                            })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Display metrics
                    st.dataframe(metrics_df.style.format({'Value': '{:.6f}'}), use_container_width=True)
                    
                    # Visualize metrics
                    fig_metrics = px.bar(metrics_df, x='Metric', y='Value', color='Type',
                                        title='Similarity and Distance Metrics',
                                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("#### Correlation Matrix")
                    
                    # Create correlation matrix
                    vectors_df = pd.DataFrame({
                        'First File': first_vec,
                        'CSV': csv_vec,
                        'AffectNet': affectnet_vec
                    }, index=emotion_cols_df1).T
                    
                    corr_matrix = vectors_df.corr()
                    
                    fig_heatmap = px.imshow(corr_matrix, 
                                           text_auto='.3f',
                                           title='Correlation Between Emotion Vectors',
                                           color_continuous_scale='RdBu',
                                           aspect='auto')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                except ImportError:
                    st.warning("Scipy is not installed. Advanced metrics will not be calculated.")
                    st.info("To install scipy, run: `pip install scipy`")
                except Exception as e:
                    st.error(f"Error in advanced analysis: {e}")
            
            # Download section
            st.divider()
            st.subheader("Export Data")
            
            # Prepare data for download
            export_data = pd.DataFrame({
                'Emotion': emotion_cols_df1,
                'First_File': [safe_float_conversion(df1_row.get(emotion, 0)) for emotion in emotion_cols_df1],
                'Second_CSV': [safe_float_conversion(df2_row.get(f'csv_{emotion}', 0)) for emotion in emotion_cols_df1],
                'Second_AffectNet': [safe_float_conversion(df2_row.get(f'affectnet_{emotion}', 0)) for emotion in emotion_cols_df1],
                'Difference_CSV_vs_First': [safe_float_conversion(df2_row.get(f'csv_{emotion}', 0)) - safe_float_conversion(df1_row.get(emotion, 0)) for emotion in emotion_cols_df1],
                'Difference_AffectNet_vs_First': [safe_float_conversion(df2_row.get(f'affectnet_{emotion}', 0)) - safe_float_conversion(df1_row.get(emotion, 0)) for emotion in emotion_cols_df1]
            })
            
            csv_export = export_data.to_csv(index=False)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.download_button(
                    label="📥 Download Comparison Data",
                    data=csv_export,
                    file_name=f"{selected_image.replace('.', '_')}_emotion_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.warning("No common images found between the two files.")
        
        with st.expander("Debug Information"):
            st.write(f"**Total images in first file:** {len(df1_filenames)}")
            st.write(f"**Total images in second file:** {len(df2_filenames)}")
            
            if len(df1_filenames) > 0 and len(df2_filenames) > 0:
                st.write("### Sample filenames for comparison:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First file (first 5):**")
                    for img in list(df1_filenames)[:5]:
                        st.write(f"- {img}")
                
                with col2:
                    st.write("**Second file (first 5):**")
                    for img in list(df2_filenames)[:5]:
                        st.write(f"- {img}")
                
                # Check if filenames might have different extensions or naming conventions
                st.write("### Filename matching issues:")
                
                # Check for partial matches
                st.write("Looking for partial matches...")
                partial_matches = []
                for img1 in list(df1_filenames)[:10]:
                    for img2 in list(df2_filenames)[:10]:
                        if img1.split('.')[0] == img2.split('.')[0]:
                            partial_matches.append((img1, img2))
                
                if partial_matches:
                    st.write(f"Found {len(partial_matches)} partial matches (same base name):")
                    for img1, img2 in partial_matches[:5]:
                        st.write(f"- {img1} ↔ {img2}")
else:
    st.info("Please upload both CSV files to begin analysis.")
    
# Add help information
with st.sidebar.expander("📋 Usage Instructions"):
    st.markdown("""
    1. **Upload both CSV files** in the sidebar
    2. **Select an image** from the dropdown (only images in both files)
    3. **View the image** and its emotion distributions
    4. **Compare distributions** across different tabs
    5. **Analyze differences** and similarity metrics
    6. **Export data** for further analysis
    """)

with st.sidebar.expander("📊 About the Data"):
    st.markdown("""
    **First File (simpleNetwork_FocalLoss):**
    - 8 basic emotions
    - Affective dimensions (valence, arousal, dominance)
    - Emotion label (numeric)
    - Full file path
    
    **Second File (comparacao_affectnet):**
    - Two sets of emotion distributions (CSV and AffectNet)
    - Similarity metrics
    - File paths for matching
    
    **Matching Process:**
    - Extracts filename from 'file' column (first file)
    - Extracts filename from 'affectnet_image_path' (second file)
    - Only shows images present in both files
    """)

with st.sidebar.expander("⚙️ Settings"):
    auto_refresh = st.checkbox("Auto-refresh image cache", value=False)
    if auto_refresh:
        st.session_state.image_cache = {}
        st.success("Image cache cleared!")
    
    show_debug = st.checkbox("Show debug information", value=False)
    if show_debug and 'df1' in locals() and 'df2' in locals():
        st.write("First file sample data:")
        st.write(df1.head(3))
        st.write("Second file sample data:")
        st.write(df2.head(3))