import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from PIL import Image
import io

# Configuração da página
st.set_page_config(page_title="Emotion Analysis System", layout="wide")

# Título da aplicação
st.title("📊 Emotion Analysis System")
st.markdown("Visualize and compare predictions with ground truth for emotion analysis")

# Funções para cálculo de métricas
def calcular_metricas_categoricas(df_comparacao, emocoes):
    """Calculate metrics for categorical emotion distributions"""
    y_true = df_comparacao[[f'{emo}_true' for emo in emocoes]].values
    y_pred = df_comparacao[[f'{emo}_pred' for emo in emocoes]].values
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Pearson correlation for each emotion
    correlacoes = []
    for i, emo in enumerate(emocoes):
        corr, _ = pearsonr(df_comparacao[f'{emo}_true'], df_comparacao[f'{emo}_pred'])
        correlacoes.append(corr)
    
    correlacao_media = np.mean(correlacoes)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Average Correlation': correlacao_media,
        'Correlations by Emotion': dict(zip(emocoes, correlacoes))
    }

def calcular_metricas_dimensionais(df_comparacao):
    """Calculate metrics for valence and arousal dimensions"""
    # Cross Correlation Coefficient (CCC)
    def cross_correlation_coefficient(x, y):
        correlation = np.corrcoef(x, y)[0, 1]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        
        ccc = (2 * correlation * np.sqrt(var_x) * np.sqrt(var_y)) / (var_x + var_y + (mean_x - mean_y)**2)
        return ccc
    
    ccc_valence = cross_correlation_coefficient(df_comparacao['valence_true'], df_comparacao['valence_pred'])
    ccc_arousal = cross_correlation_coefficient(df_comparacao['arousal_true'], df_comparacao['arousal_pred'])
    ccc_medio = (ccc_valence + ccc_arousal) / 2
    
    # Other metrics
    mse_valence = mean_squared_error(df_comparacao['valence_true'], df_comparacao['valence_pred'])
    mse_arousal = mean_squared_error(df_comparacao['arousal_true'], df_comparacao['arousal_pred'])
    mse_medio = (mse_valence + mse_arousal) / 2
    
    mae_valence = mean_absolute_error(df_comparacao['valence_true'], df_comparacao['valence_pred'])
    mae_arousal = mean_absolute_error(df_comparacao['arousal_true'], df_comparacao['arousal_pred'])
    mae_medio = (mae_valence + mae_arousal) / 2
    
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    rmse_medio = (rmse_valence + rmse_arousal) / 2
    
    # Pearson correlation
    corr_valence, _ = pearsonr(df_comparacao['valence_true'], df_comparacao['valence_pred'])
    corr_arousal, _ = pearsonr(df_comparacao['arousal_true'], df_comparacao['arousal_pred'])
    corr_medio = (corr_valence + corr_arousal) / 2
    
    return {
        'CCC Valence': ccc_valence,
        'CCC Arousal': ccc_arousal,
        'Average CCC': ccc_medio,
        'MSE Valence': mse_valence,
        'MSE Arousal': mse_arousal,
        'Average MSE': mse_medio,
        'MAE Valence': mae_valence,
        'MAE Arousal': mae_arousal,
        'Average MAE': mae_medio,
        'RMSE Valence': rmse_valence,
        'RMSE Arousal': rmse_arousal,
        'Average RMSE': rmse_medio,
        'Correlation Valence': corr_valence,
        'Correlation Arousal': corr_arousal,
        'Average Correlation': corr_medio
    }

def criar_dataframe_comparacao(df_pred, df_gt, emocoes, dimensoes):
    """Create dataframe with direct comparison between prediction and ground truth for each image"""
    # Find common images
    imagens_comuns = set(df_pred['file']).intersection(set(df_gt['file']))
    
    if len(imagens_comuns) == 0:
        st.error("❌ No common images found between files!")
        return None
    
    st.info(f"📊 Found {len(imagens_comuns)} common images for analysis")
    
    # Filter only common images
    df_pred_filtrado = df_pred[df_pred['file'].isin(imagens_comuns)].copy()
    df_gt_filtrado = df_gt[df_gt['file'].isin(imagens_comuns)].copy()
    
    # Sort by same files to ensure matching
    df_pred_filtrado = df_pred_filtrado.sort_values('file').reset_index(drop=True)
    df_gt_filtrado = df_gt_filtrado.sort_values('file').reset_index(drop=True)
    
    # Create comparison dataframe
    df_comparacao = pd.DataFrame()
    df_comparacao['file'] = df_pred_filtrado['file']
    
    # Add columns for each emotion (true and pred)
    for emo in emocoes:
        df_comparacao[f'{emo}_true'] = df_gt_filtrado[emo].values
        df_comparacao[f'{emo}_pred'] = df_pred_filtrado[emo].values
        df_comparacao[f'{emo}_diff'] = df_comparacao[f'{emo}_pred'] - df_comparacao[f'{emo}_true']
    
    # Add columns for dimensions
    for dim in dimensoes:
        df_comparacao[f'{dim}_true'] = df_gt_filtrado[dim].values
        df_comparacao[f'{dim}_pred'] = df_pred_filtrado[dim].values
        df_comparacao[f'{dim}_diff'] = df_comparacao[f'{dim}_pred'] - df_comparacao[f'{dim}_true']
    
    return df_comparacao

# Sidebar for file upload
st.sidebar.header("📁 File Upload")

arquivo_pred = st.sidebar.file_uploader("Prediction File", type=['csv'])
arquivo_gt = st.sidebar.file_uploader("Ground Truth File", type=['csv'])

# List of emotions and dimensions
emocoes = ['happy', 'contempt', 'elated', 'hopeful', 'surprised', 'proud', 
           'loved', 'angry', 'astonished', 'disgusted', 'fearful', 'sad', 
           'fatigued', 'neutral']

dimensoes = ['valence', 'arousal', 'dominance']

if arquivo_pred and arquivo_gt:
    try:
        # Load data
        df_pred = pd.read_csv(arquivo_pred)
        df_gt = pd.read_csv(arquivo_gt)
        
        # Check if required columns exist
        colunas_necessarias = emocoes + dimensoes + ['file']
        colunas_faltantes_pred = [col for col in colunas_necessarias if col not in df_pred.columns]
        colunas_faltantes_gt = [col for col in colunas_necessarias if col not in df_gt.columns]
        
        if colunas_faltantes_pred or colunas_faltantes_gt:
            st.error(f"Missing columns: Prediction - {colunas_faltantes_pred}, Ground Truth - {colunas_faltantes_gt}")
        else:
            st.success("✅ Files loaded successfully!")
            
            # Create comparison dataframe
            with st.spinner("Processing comparison between files..."):
                df_comparacao = criar_dataframe_comparacao(df_pred, df_gt, emocoes, dimensoes)
            
            if df_comparacao is not None:
                # Select image for visualization
                st.sidebar.header("🎯 Image Selection")
                imagem_selecionada = st.sidebar.selectbox("Select an image:", df_comparacao['file'].tolist())
                
                # Main layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Selected Image")
                    try:
                        # Try to load and display the image
                        if os.path.exists(imagem_selecionada):
                            img = Image.open(imagem_selecionada)
                            st.image(img, caption=os.path.basename(imagem_selecionada), use_column_width=True)
                        else:
                            st.warning("Image path not found. Checking only filename...")
                            st.info(f"File: {os.path.basename(imagem_selecionada)}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                        st.info(f"File: {os.path.basename(imagem_selecionada)}")
                
                with col2:
                    st.subheader("📊 Comparison for Selected Image")
                    
                    # Get data for selected image from comparison dataframe
                    dados_imagem = df_comparacao[df_comparacao['file'] == imagem_selecionada].iloc[0]
                    
                    # Emotion distributions chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Emotions - Prediction vs Ground Truth
                    x = range(len(emocoes))
                    width = 0.35
                    
                    valores_true = [dados_imagem[f'{emo}_true'] for emo in emocoes]
                    valores_pred = [dados_imagem[f'{emo}_pred'] for emo in emocoes]
                    
                    ax1.bar([i - width/2 for i in x], valores_pred, 
                           width, label='Prediction', alpha=0.7, color='blue')
                    ax1.bar([i + width/2 for i in x], valores_true, 
                           width, label='Ground Truth', alpha=0.7, color='orange')
                    
                    ax1.set_xlabel('Emotions')
                    ax1.set_ylabel('Probability')
                    ax1.set_title(f'Emotion Distributions - {os.path.basename(imagem_selecionada)}')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(emocoes, rotation=45, ha='right')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Dimensions
                    dimensoes_plot = ['valence', 'arousal']
                    x_dims = range(len(dimensoes_plot))
                    
                    valores_true_dims = [dados_imagem[f'{dim}_true'] for dim in dimensoes_plot]
                    valores_pred_dims = [dados_imagem[f'{dim}_pred'] for dim in dimensoes_plot]
                    
                    ax2.bar([i - width/2 for i in x_dims], valores_pred_dims, 
                           width, label='Prediction', alpha=0.7, color='blue')
                    ax2.bar([i + width/2 for i in x_dims], valores_true_dims, 
                           width, label='Ground Truth', alpha=0.7, color='orange')
                    
                    ax2.set_xlabel('Dimensions')
                    ax2.set_ylabel('Value')
                    ax2.set_title('Dimensions - Valence and Arousal')
                    ax2.set_xticks(x_dims)
                    ax2.set_xticklabels(dimensoes_plot)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics for individual image
                    st.subheader("📈 Metrics for Individual Image")
                    
                    col_met1, col_met2, col_met3 = st.columns(3)
                    
                    with col_met1:
                        mse_emo = mean_squared_error(valores_true, valores_pred)
                        st.metric("MSE Emotions", f"{mse_emo:.4f}")
                    
                    with col_met2:
                        mae_emo = mean_absolute_error(valores_true, valores_pred)
                        st.metric("MAE Emotions", f"{mae_emo:.4f}")
                    
                    with col_met3:
                        mse_dims = mean_squared_error(valores_true_dims, valores_pred_dims)
                        st.metric("MSE Dimensions", f"{mse_dims:.4f}")
                
                # Global Analysis
                st.markdown("---")
                st.header("🌍 Global Database Analysis")
                st.info(f"Analyzing {len(df_comparacao)} matching images")
                
                # Calculate global metrics
                with st.spinner("Calculating global metrics..."):
                    metricas_emocionais = calcular_metricas_categoricas(df_comparacao, emocoes)
                    metricas_dimensionais = calcular_metricas_dimensionais(df_comparacao)
                
                # Display global metrics
                col_glob1, col_glob2 = st.columns(2)
                
                with col_glob1:
                    st.subheader("📊 Global Metrics - Emotions")
                    
                    met_emo_df = pd.DataFrame({
                        'Metric': ['MSE', 'MAE', 'RMSE', 'Average Correlation'],
                        'Value': [
                            metricas_emocionais['MSE'],
                            metricas_emocionais['MAE'],
                            metricas_emocionais['RMSE'],
                            metricas_emocionais['Average Correlation']
                        ]
                    })
                    st.dataframe(met_emo_df, use_container_width=True)
                    
                    # Correlation chart by emotion
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                    correlacoes_emo = [metricas_emocionais['Correlations by Emotion'][emo] for emo in emocoes]
                    bars = ax_corr.bar(emocoes, correlacoes_emo)
                    
                    # Color bars based on correlation value
                    for bar, corr in zip(bars, correlacoes_emo):
                        if corr > 0.7:
                            bar.set_color('green')
                        elif corr > 0.5:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
                    
                    ax_corr.set_xlabel('Emotions')
                    ax_corr.set_ylabel('Pearson Correlation')
                    ax_corr.set_title('Correlation by Emotion (Prediction vs Ground Truth)')
                    ax_corr.set_xticklabels(emocoes, rotation=45, ha='right')
                    ax_corr.grid(True, alpha=0.3)
                    ax_corr.set_ylim(0, 1)
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                
                with col_glob2:
                    st.subheader("📈 Global Metrics - Dimensions")
                    
                    # Main metrics
                    met_dim_principais = pd.DataFrame({
                        'Metric': ['Average CCC', 'Average MSE', 'Average MAE', 'Average RMSE', 'Average Correlation'],
                        'Value': [
                            metricas_dimensionais['Average CCC'],
                            metricas_dimensionais['Average MSE'],
                            metricas_dimensionais['Average MAE'],
                            metricas_dimensionais['Average RMSE'],
                            metricas_dimensionais['Average Correlation']
                        ]
                    })
                    st.dataframe(met_dim_principais, use_container_width=True)
                    
                    # Detailed metrics
                    with st.expander("View detailed metrics by dimension"):
                        met_dim_detalhadas = pd.DataFrame({
                            'Dimension': ['Valence', 'Arousal'],
                            'CCC': [metricas_dimensionais['CCC Valence'], metricas_dimensionais['CCC Arousal']],
                            'MSE': [metricas_dimensionais['MSE Valence'], metricas_dimensionais['MSE Arousal']],
                            'MAE': [metricas_dimensionais['MAE Valence'], metricas_dimensionais['MAE Arousal']],
                            'RMSE': [metricas_dimensionais['RMSE Valence'], metricas_dimensionais['RMSE Arousal']],
                            'Correlation': [metricas_dimensionais['Correlation Valence'], metricas_dimensionais['Correlation Arousal']]
                        })
                        st.dataframe(met_dim_detalhadas, use_container_width=True)
                
                # Scatter plots for dimensions
                st.subheader("🔍 Scatter Analysis - Dimensions")
                
                fig_scatter, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Valence
                ax1.scatter(df_comparacao['valence_true'], df_comparacao['valence_pred'], alpha=0.6)
                min_val = min(df_comparacao['valence_true'].min(), df_comparacao['valence_pred'].min())
                max_val = max(df_comparacao['valence_true'].max(), df_comparacao['valence_pred'].max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Reference Line')
                ax1.set_xlabel('Ground Truth - Valence')
                ax1.set_ylabel('Prediction - Valence')
                ax1.set_title(f'Valence (CCC: {metricas_dimensionais["CCC Valence"]:.3f})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Arousal
                ax2.scatter(df_comparacao['arousal_true'], df_comparacao['arousal_pred'], alpha=0.6)
                min_val = min(df_comparacao['arousal_true'].min(), df_comparacao['arousal_pred'].min())
                max_val = max(df_comparacao['arousal_true'].max(), df_comparacao['arousal_pred'].max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Reference Line')
                ax2.set_xlabel('Ground Truth - Arousal')
                ax2.set_ylabel('Prediction - Arousal')
                ax2.set_title(f'Arousal (CCC: {metricas_dimensionais["CCC Arousal"]:.3f})')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_scatter)
                
                # Detailed comparison table
                st.subheader("📋 Detailed Comparison Table")
                
                with st.expander("View complete comparison data"):
                    # Create simplified version for display
                    colunas_exibicao = ['file']
                    for emo in emocoes[:5]:  # Show only first 5 emotions to avoid overload
                        colunas_exibicao.extend([f'{emo}_true', f'{emo}_pred', f'{emo}_diff'])
                    
                    for dim in dimensoes:
                        colunas_exibicao.extend([f'{dim}_true', f'{dim}_pred', f'{dim}_diff'])
                    
                    st.dataframe(df_comparacao[colunas_exibicao].head(20), use_container_width=True)
                    st.info(f"Showing 20 of {len(df_comparacao)} rows. Use scroll to see more.")
    
    except Exception as e:
        st.error(f"❌ Error processing files: {e}")
        st.info("Make sure files are in correct format and contain required columns.")

else:
    st.info("👆 Upload prediction and ground truth files to start analysis.")
    
    # Instructions
    with st.expander("ℹ️ Usage Instructions"):
        st.markdown("""
        ### How to use this application:
        
        1. **File Upload**: Upload prediction CSV file and ground truth CSV file
        2. **Expected Format**: Files should contain the same columns:
           - Emotions: happy, contempt, elated, hopeful, surprised, proud, loved, angry, astonished, disgusted, fearful, sad, fatigued, neutral
           - Dimensions: valence, arousal, dominance
           - Image path: file
        3. **Image-based Comparison**: Analysis compares values of the SAME image between prediction and ground truth
        4. **Global Metrics**: Calculated over ALL matching images
        
        ### Calculated Metrics:
        - **Emotion Distributions**: MSE, MAE, RMSE, Pearson Correlation
        - **Dimensions**: Cross Correlation Coefficient (CCC), MSE, MAE, RMSE, Correlation
        """)

# Footer
st.markdown("---")
st.markdown("Developed for emotion recognition model analysis")