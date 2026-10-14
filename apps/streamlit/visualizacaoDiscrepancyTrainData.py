import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(page_title="Emotion Distribution Analysis", layout="wide")

st.title("📊 Emotion Distribution Analysis")
st.markdown("The charts below show the percentage (%) of occurrences to allow for better comparison between datasets of different sizes.")

# File uploader
uploaded_file = st.file_uploader("Choose the generated CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Distribution columns (8 standard emotions)
    emotion_cols = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'angry', 'contempt']
    
    # 1. Data Processing
    # Identify the Top 1 emotion from the distribution
    df['top_1_dist'] = df[emotion_cols].idxmax(axis=1)
    
    # Calculate Discrepancy (Match vs Mismatch)
    df['is_discrepant'] = df['emotion'] != df['top_1_dist']
    
    # Calculate the Rank of the actual label within the distribution
    def get_rank(row):
        # Sort intensities from highest to lowest
        sorted_dist = row[emotion_cols].sort_values(ascending=False)
        try:
            # Find the position of the ground truth 'emotion'
            return list(sorted_dist.index).index(row['emotion']) + 1
        except:
            return np.nan

    df['emotion_rank'] = df.apply(get_rank, axis=1)

    # ==========================================
    # NOVO: Processamento de Distância VAD
    # ==========================================
    # 1. Calcular os centroides empíricos (Média de VAD para cada emoção anotada)
    centroids = df.groupby('emotion')[['valence', 'arousal', 'dominance']].mean().to_dict('index')

    def analyze_vad_distance(row):
        # Analisar apenas os casos onde houve discrepância (erro do modelo)
        if not row['is_discrepant']:
            return np.nan
        
        gt_emo = row['emotion']
        pred_emo = row['top_1_dist']
        
        # VAD atual da imagem
        v, a, d = row['valence'], row['arousal'], row['dominance']
        
        # Centroides
        gt_c = centroids.get(gt_emo, {'valence': v, 'arousal': a, 'dominance': d})
        pred_c = centroids.get(pred_emo, {'valence': v, 'arousal': a, 'dominance': d})
        
        # Distância Euclidiana 3D
        dist_gt = np.sqrt((v - gt_c['valence'])**2 + (a - gt_c['arousal'])**2 + (d - gt_c['dominance'])**2)
        dist_pred = np.sqrt((v - pred_c['valence'])**2 + (a - pred_c['arousal'])**2 + (d - pred_c['dominance'])**2)
        
        if dist_pred < dist_gt:
            return 'Closer to Prediction'
        elif dist_gt < dist_pred:
            return 'Closer to Ground Truth'
        else:
            return 'Equidistant'
            
    df['vad_closer_to'] = df.apply(analyze_vad_distance, axis=1)

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        # 1 - Histogram of Ground Truth Emotions (Relative)
        st.subheader("1. Relative Frequency of Labels (Ground Truth)")
        fig1 = px.histogram(df, x='emotion', 
                           category_orders={'emotion': df['emotion'].value_counts().index},
                           histnorm='percent',
                           labels={'emotion': 'Emotion', 'percent': 'Percentage (%)'},
                           color_discrete_sequence=['#636EFA'])
        fig1.update_yaxes(title_text="Percentage (%)")
        st.plotly_chart(fig1, use_container_width=True)

        # 3 - Discrepancy Ratio (Pie Chart)
        st.subheader("3. Discrepancy Proportion")
        discrepancy_df = df['is_discrepant'].value_counts(normalize=True).reset_index()
        discrepancy_df.columns = ['Status', 'Proportion']
        discrepancy_df['Status'] = discrepancy_df['Status'].map({True: 'Discrepant (Mismatch)', False: 'Match'})
        
        fig3 = px.pie(discrepancy_df, names='Status', values='Proportion', 
                     color='Status', color_discrete_map={'Match': '#00CC96', 'Discrepant (Mismatch)': '#EF553B'},
                     hole=0.4)
        fig3.update_traces(textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # 2 - Histogram of Top 1 Predictions (Relative)
        st.subheader("2. Relative Frequency of Top 1 Distribution")
        fig2 = px.histogram(df, x='top_1_dist', 
                           category_orders={'top_1_dist': df['top_1_dist'].value_counts().index},
                           histnorm='percent',
                           labels={'top_1_dist': 'Predominant Emotion', 'percent': 'Percentage (%)'},
                           color_discrete_sequence=['#AB63FA'])
        fig2.update_yaxes(title_text="Percentage (%)")
        st.plotly_chart(fig2, use_container_width=True)

        # 4 - Discrepancy by Rank (Relative)
        st.subheader("4. Percentage Distribution by Rank")
        st.info("Shows where the ground truth label placed in the predicted distribution.")
        fig4 = px.histogram(df, x='emotion_rank', 
                           nbins=8,
                           histnorm='percent',
                           labels={'emotion_rank': 'Rank Position (1 = Perfect Match)', 'percent': 'Percentage (%)'})
        fig4.update_yaxes(title_text="Percentage (%)")
        fig4.update_xaxes(dtick=1) 
        st.plotly_chart(fig4, use_container_width=True)

    # ==========================================
    # NOVO: Sessão de Justificativa VAD (Gráfico 5)
    # ==========================================
    st.divider()
    st.subheader("5. VAD Distance Analysis on Discrepant Cases (Proof of Annotation Noise)")
    st.markdown("""
    **What does this mean?** When the model's Top-1 prediction contradicts the human label, we check the actual Valence, Arousal, and Dominance values of the image. 
    We calculate if the image's VAD is mathematically closer to the centroid of the **Model's Prediction** or the **Ground Truth Label**. 
    If it's closer to the prediction, the model is mathematically correct, and the human annotation was likely noisy or inconsistent.
    """)
    
    # Filter only discrepant cases
    df_discrepant = df[df['is_discrepant'] == True]
    
    if not df_discrepant.empty:
        vad_closer_counts = df_discrepant['vad_closer_to'].value_counts(normalize=True).reset_index()
        vad_closer_counts.columns = ['Closer to', 'Proportion']
        
        # Set colors to emphasize when prediction is right
        color_map = {'Closer to Prediction': '#AB63FA', 'Closer to Ground Truth': '#636EFA', 'Equidistant': '#B6E880'}
        
        fig5 = px.pie(vad_closer_counts, names='Closer to', values='Proportion', 
                     color='Closer to', color_discrete_map=color_map, hole=0.4)
        fig5.update_traces(textinfo='percent+label')
        
        col_text, col_chart = st.columns([1, 2])
        with col_text:
            st.metric("Label Accuracy (Argmax Match)", f"{(~df['is_discrepant']).mean() * 100:.2f}%")
            
            # Cálculo de "Acurácia Justificada"
            prediction_justified = (df_discrepant['vad_closer_to'] == 'Closer to Prediction').sum()
            total_cases = len(df)
            justified_acc = ((~df['is_discrepant']).sum() + prediction_justified) / total_cases * 100
            
            st.metric("VAD-Justified Accuracy", f"{justified_acc:.2f}%", help="Argmax Matches + Discrepancies mathematically justified by VAD distance")
            
        with col_chart:
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.success("No discrepancies found! 100% accuracy.")

    # Raw data preview
    st.divider()
    if st.checkbox("Show processed raw data"):
        st.dataframe(df[['emotion', 'top_1_dist', 'is_discrepant', 'emotion_rank', 'vad_closer_to', 'valence', 'arousal', 'dominance']].head(50))

else:
    st.info("Awaiting CSV file upload to begin processing.")