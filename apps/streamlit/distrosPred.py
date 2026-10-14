import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from PIL import Image
import base64
import io

# Configuração da página
st.set_page_config(
    page_title="Emotion Distribution Comparison Viewer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .ground-truth { 
        color: #1f77b4; 
        font-weight: bold; 
        border-left: 4px solid #1f77b4;
    }
    .predictions { 
        color: #e74c3c; 
        font-weight: bold; 
        border-left: 4px solid #e74c3c;
    }
    .positive-diff { color: #2ecc71; font-weight: bold; }
    .negative-diff { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class EmotionComparisonViewer:
    def __init__(self):
        self.df1 = None
        self.df2 = None
        self.common_images = []
        
        # Inicializar session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'global_metrics_calculated' not in st.session_state:
            st.session_state.global_metrics_calculated = False
    
    def clean_dataframe(self, df):
        """Limpa o dataframe removendo linhas com problemas na coluna 'file'"""
        if df is None:
            return df
        
        # Fazer uma cópia para não modificar o original
        df_clean = df.copy()
        
        # Converter a coluna 'file' para string e remover linhas com valores NaN
        df_clean['file'] = df_clean['file'].astype(str)
        df_clean = df_clean[df_clean['file'] != 'nan']
        df_clean = df_clean[df_clean['file'].str.strip() != '']
        
        return df_clean
    
    def calculate_error_metrics(self, values1, values2):
        """Calculate various error metrics between two distributions"""
        # Mean Absolute Error
        mae = mean_absolute_error(values1, values2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(values1, values2))
        
        # Jensen-Shannon Divergence
        js_div = jensenshannon(values1, values2)
        
        # KL Divergence (with smoothing to avoid division by zero)
        epsilon = 1e-10
        smoothed_p = np.array(values1) + epsilon
        smoothed_q = np.array(values2) + epsilon
        smoothed_p = smoothed_p / np.sum(smoothed_p)
        smoothed_q = smoothed_q / np.sum(smoothed_q)
        kl_div = np.sum(smoothed_p * np.log(smoothed_p / smoothed_q))
        
        # Cosine Similarity
        cosine_sim = np.dot(values1, values2) / (np.linalg.norm(values1) * np.linalg.norm(values2))
        
        # Correlation
        correlation = np.corrcoef(values1, values2)[0, 1]
        
        # Maximum difference
        differences = [abs(values1[i] - values2[i]) for i in range(len(values1))]
        max_diff = max(differences)
        max_diff_index = differences.index(max_diff)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'js_divergence': js_div,
            'kl_divergence': kl_div,
            'cosine_similarity': cosine_sim,
            'correlation': correlation,
            'max_difference': max_diff,
            'max_difference_index': max_diff_index
        }
    
    def find_common_images(self):
        """Find common images between the two dataframes"""
        if self.df1 is None or self.df2 is None:
            return []
        
        try:
            # Extrair nomes de arquivos válidos
            df1_files = []
            for _, row in self.df1.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and not filename.startswith('nan'):
                    df1_files.append(os.path.basename(filename))
            
            df2_files = []
            for _, row in self.df2.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and not filename.startswith('nan'):
                    df2_files.append(os.path.basename(filename))
            
            # Encontrar interseção
            common_files = list(set(df1_files) & set(df2_files))
            return sorted(common_files)
            
        except Exception as e:
            st.error(f"Erro ao encontrar imagens comuns: {str(e)}")
            return []
    
    def load_image(self, image_path, max_size=400):
        """Load and display image"""
        try:
            # Verificar se o caminho é válido
            if not image_path or str(image_path).strip() in ['', 'nan']:
                return None
                
            image_path = str(image_path).strip()
            
            if os.path.exists(image_path):
                img = Image.open(image_path)
                # Redimensionar mantendo a proporção
                img.thumbnail((max_size, max_size))
                return img
            else:
                # Tentar encontrar o arquivo em outros locais
                filename = os.path.basename(image_path)
                possible_paths = [
                    filename,  # Arquivo no diretório atual
                    os.path.join('images', filename),
                    os.path.join('data', filename),
                    os.path.join('output', filename),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        img = Image.open(path)
                        img.thumbnail((max_size, max_size))
                        return img
                
                return None
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {str(e)}")
            return None
    
    def create_comparison_chart(self, row1, row2, emotions):
        """Create comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extrair valores, lidando com colunas possivelmente faltantes
        values1 = []
        values2 = []
        available_emotions = []
        
        for emotion in emotions:
            if emotion in row1 and emotion in row2:
                values1.append(row1[emotion])
                values2.append(row2[emotion])
                available_emotions.append(emotion)
        
        if not available_emotions:
            ax.text(0.5, 0.5, 'No emotion data available', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            return fig
        
        # Configurar posições das barras
        x = np.arange(len(available_emotions))
        width = 0.35
        
        # Criar gráfico de barras comparativo
        bars1 = ax.bar(x - width/2, values1, width, label='Ground Truth', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, values2, width, label='Predictions', color='red', alpha=0.7)
        
        ax.set_title('Emotion Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(available_emotions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars1, values1):
            height = bar.get_height()
            if height > 0.05:  # Só mostrar texto se a barra for alta o suficiente
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, values2):
            height = bar.get_height()
            if height > 0.05:  # Só mostrar texto se a barra for alta o suficiente
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_diff_chart(self, row1, row2, emotions):
        """Create difference chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calcular diferenças, lidando com colunas possivelmente faltantes
        differences = []
        available_emotions = []
        
        for emotion in emotions:
            if emotion in row1 and emotion in row2:
                diff = row1[emotion] - row2[emotion]
                differences.append(diff)
                available_emotions.append(emotion)
        
        if not available_emotions:
            ax.text(0.5, 0.5, 'No emotion data available', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            return fig
        
        colors = ['green' if diff >= 0 else 'red' for diff in differences]
        
        # Criar gráfico de barras para diferenças
        bars = ax.bar(available_emotions, differences, color=colors, alpha=0.7)
        
        ax.set_title('Distribution Differences (Ground Truth - Predictions)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Difference', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, differences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.01 if height >= 0 else -0.02),
                   f'{value:.3f}', 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=9, 
                   fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def calculate_global_metrics(self):
        """Calculate global metrics across all images"""
        emotions = ['happy', 'contempt', 'elated', 'surprised', 'love', 'protected', 
                   'astonished', 'disgusted', 'angry', 'fearfull', 'sad', 'neutral']
        
        all_mae = []
        all_rmse = []
        all_js = []
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_name in enumerate(self.common_images):
            # Atualizar progresso
            progress = (i + 1) / len(self.common_images)
            progress_bar.progress(progress)
            status_text.text(f"Processando: {os.path.basename(image_name)} ({i+1}/{len(self.common_images)})")
            
            # Encontrar linhas correspondentes
            row1 = None
            row2 = None
            
            for _, row in self.df1.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and os.path.basename(filename) == image_name:
                    row1 = row
                    break
                    
            for _, row in self.df2.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and os.path.basename(filename) == image_name:
                    row2 = row
                    break
            
            if row1 is not None and row2 is not None:
                # Coletar apenas emoções disponíveis
                emotion_values1 = []
                emotion_values2 = []
                available_emotions = []
                
                for emotion in emotions:
                    if emotion in row1 and emotion in row2:
                        emotion_values1.append(row1[emotion])
                        emotion_values2.append(row2[emotion])
                        available_emotions.append(emotion)
                
                if available_emotions:
                    # Calcular métricas
                    metrics = self.calculate_error_metrics(emotion_values1, emotion_values2)
                    
                    all_mae.append(metrics['mae'])
                    all_rmse.append(metrics['rmse'])
                    all_js.append(metrics['js_divergence'])
                    
                    # Verificar se a emoção principal coincide
                    gt_top = available_emotions[emotion_values1.index(max(emotion_values1))]
                    pred_top = available_emotions[emotion_values2.index(max(emotion_values2))]
                    
                    if gt_top == pred_top:
                        correct_predictions += 1
                    total_predictions += 1
        
        # Limpar barra de progresso
        progress_bar.empty()
        status_text.empty()
        
        # Calcular médias globais
        global_mae = np.mean(all_mae) if all_mae else 0
        global_rmse = np.mean(all_rmse) if all_rmse else 0
        global_js = np.mean(all_js) if all_js else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'global_mae': global_mae,
            'global_rmse': global_rmse,
            'global_js': global_js,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def render_interface(self):
        """Render the main interface"""
        st.markdown('<h1 class="main-header">😊 Emotion Distribution Comparison Viewer</h1>', unsafe_allow_html=True)
        
        # Seção de upload de arquivos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Ground Truth CSV")
            uploaded_file1 = st.file_uploader("Select Ground Truth CSV", type=['csv'], key="file1")
            if uploaded_file1:
                try:
                    self.df1 = pd.read_csv(uploaded_file1)
                    self.df1 = self.clean_dataframe(self.df1)
                    st.success(f"✅ Ground Truth loaded: {len(self.df1)} valid rows")
                    
                    # Mostrar prévia dos dados
                    with st.expander("🔍 Preview Ground Truth Data"):
                        st.dataframe(self.df1.head())
                        
                except Exception as e:
                    st.error(f"❌ Error loading Ground Truth CSV: {str(e)}")
        
        with col2:
            st.subheader("📁 Predictions CSV")
            uploaded_file2 = st.file_uploader("Select Predictions CSV", type=['csv'], key="file2")
            if uploaded_file2:
                try:
                    self.df2 = pd.read_csv(uploaded_file2)
                    self.df2 = self.clean_dataframe(self.df2)
                    st.success(f"✅ Predictions loaded: {len(self.df2)} valid rows")
                    
                    # Mostrar prévia dos dados
                    with st.expander("🔍 Preview Predictions Data"):
                        st.dataframe(self.df2.head())
                        
                except Exception as e:
                    st.error(f"❌ Error loading Predictions CSV: {str(e)}")
        
        # Verificar se ambos os arquivos foram carregados
        if self.df1 is not None and self.df2 is not None:
            self.common_images = self.find_common_images()
            
            if not self.common_images:
                st.error("""
                ❌ No common images found between the two files. Please check:
                - Both files have a 'file' column with valid paths
                - File names match between the two datasets
                - There are no NaN values in the 'file' column
                """)
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write("Ground Truth files sample:", [str(f) for f in self.df1['file'].head().tolist()])
                    st.write("Predictions files sample:", [str(f) for f in self.df2['file'].head().tolist()])
                return
            
            # Controles de navegação
            st.subheader("🎮 Navigation Controls")
            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 2])
            
            with nav_col1:
                if st.button("⏮️ First", use_container_width=True):
                    st.session_state.current_index = 0
                    st.rerun()
            
            with nav_col2:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_index = max(0, st.session_state.current_index - 1)
                    st.rerun()
            
            with nav_col3:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_index = min(len(self.common_images) - 1, st.session_state.current_index + 1)
                    st.rerun()
            
            with nav_col4:
                if st.button("Last ⏭️", use_container_width=True):
                    st.session_state.current_index = len(self.common_images) - 1
                    st.rerun()
            
            with nav_col5:
                selected_image = st.selectbox(
                    "Select Image:",
                    options=self.common_images,
                    index=st.session_state.current_index,
                    key="image_selector"
                )
                if selected_image and selected_image in self.common_images:
                    new_index = self.common_images.index(selected_image)
                    if new_index != st.session_state.current_index:
                        st.session_state.current_index = new_index
                        st.rerun()
            
            # Mostrar posição atual
            st.info(f"📊 Showing image {st.session_state.current_index + 1} of {len(self.common_images)}")
            
            # Obter dados da imagem atual
            current_file = self.common_images[st.session_state.current_index]
            
            # Encontrar linhas correspondentes
            row1 = None
            row2 = None
            
            for _, row in self.df1.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and os.path.basename(filename) == current_file:
                    row1 = row
                    break
                    
            for _, row in self.df2.iterrows():
                filename = str(row['file']).strip()
                if filename and filename != 'nan' and os.path.basename(filename) == current_file:
                    row2 = row
                    break
            
            if row1 is None or row2 is None:
                st.error("❌ Could not find matching data for selected image.")
                return
            
            # Área principal de conteúdo
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Exibição da imagem
                st.subheader("🖼️ Image Preview")
                actual_path = row1['file'] if 'file' in row1 else ''
                img = self.load_image(actual_path)
                if img:
                    st.image(img, use_column_width=True)
                    st.caption(f"Image: {os.path.basename(str(actual_path))}")
                else:
                    st.warning("⚠️ Image not found or cannot be displayed")
                    st.info(f"Tried to load: {actual_path}")
                
                # Caminhos dos arquivos
                st.subheader("📁 File Information")
                st.markdown(f'<div class="metric-card ground-truth">'
                           f'<h4>Ground Truth Path:</h4>'
                           f'<p style="word-wrap: break-word; font-size: 0.8em;">{actual_path}</p>'
                           f'</div>', unsafe_allow_html=True)
                
                pred_path = row2['file'] if 'file' in row2 else ''
                st.markdown(f'<div class="metric-card predictions">'
                           f'<h4>Predictions Path:</h4>'
                           f'<p style="word-wrap: break-word; font-size: 0.8em;">{pred_path}</p>'
                           f'</div>', unsafe_allow_html=True)
            
            with col2:
                # Análise de emoções
                st.subheader("😊 Emotion Analysis")
                
                emotions = ['happy', 'contempt', 'elated', 'surprised', 'love', 'protected', 
                           'astonished', 'disgusted', 'angry', 'fearfull', 'sad', 'neutral']
                
                # Emoções dominantes
                col2_1, col2_2 = st.columns(2)
                
                with col2_1:
                    # Emoção dominante do Ground Truth
                    emotion_values1 = []
                    available_emotions1 = []
                    for emotion in emotions:
                        if emotion in row1:
                            emotion_values1.append(row1[emotion])
                            available_emotions1.append(emotion)
                    
                    if available_emotions1:
                        max_index1 = emotion_values1.index(max(emotion_values1))
                        dominant_emotion1 = available_emotions1[max_index1]
                        dominant_value1 = emotion_values1[max_index1]
                        
                        st.markdown(f'<div class="metric-card ground-truth">'
                                   f'<h3>Ground Truth</h3>'
                                   f'<p><strong>Dominant Emotion:</strong> {dominant_emotion1}</p>'
                                   f'<p><strong>Value:</strong> {dominant_value1:.4f}</p>'
                                   f'</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No emotion data found in Ground Truth")
                
                with col2_2:
                    # Emoção dominante das Predictions
                    emotion_values2 = []
                    available_emotions2 = []
                    for emotion in emotions:
                        if emotion in row2:
                            emotion_values2.append(row2[emotion])
                            available_emotions2.append(emotion)
                    
                    if available_emotions2:
                        max_index2 = emotion_values2.index(max(emotion_values2))
                        dominant_emotion2 = available_emotions2[max_index2]
                        dominant_value2 = emotion_values2[max_index2]
                        
                        st.markdown(f'<div class="metric-card predictions">'
                                   f'<h3>Predictions</h3>'
                                   f'<p><strong>Dominant Emotion:</strong> {dominant_emotion2}</p>'
                                   f'<p><strong>Value:</strong> {dominant_value2:.4f}</p>'
                                   f'</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No emotion data found in Predictions")
                
                # Métricas de erro
                if available_emotions1 and available_emotions2:
                    st.subheader("📊 Error Metrics")
                    metrics = self.calculate_error_metrics(emotion_values1, emotion_values2)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.6f}")
                        st.metric("Jensen-Shannon Divergence", f"{metrics['js_divergence']:.6f}")
                        st.metric("Cosine Similarity", f"{metrics['cosine_similarity']:.6f}")
                        st.metric("Maximum Difference", f"{metrics['max_difference']:.6f}")
                    
                    with metric_col2:
                        st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.6f}")
                        st.metric("KL Divergence", f"{metrics['kl_divergence']:.6f}")
                        st.metric("Correlation", f"{metrics['correlation']:.6f}")
                        if available_emotions1:
                            st.metric("Emotion with Max Difference", available_emotions1[metrics['max_difference_index']])
            
            # Seção de gráficos
            if available_emotions1 and available_emotions2:
                st.subheader("📈 Visualizations")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig1 = self.create_comparison_chart(row1, row2, emotions)
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                with chart_col2:
                    fig2 = self.create_diff_chart(row1, row2, emotions)
                    st.pyplot(fig2)
                    plt.close(fig2)
            
            # Seção de métricas globais
            st.subheader("🌍 Global Metrics (All Images)")
            
            if st.button("📊 Calculate Global Metrics", type="primary"):
                with st.spinner("Calculating global metrics across all images..."):
                    global_metrics = self.calculate_global_metrics()
                    st.session_state.global_metrics_calculated = True
                    st.session_state.global_metrics = global_metrics
                    st.rerun()
            
            if st.session_state.global_metrics_calculated and 'global_metrics' in st.session_state:
                global_metrics = st.session_state.global_metrics
                
                global_col1, global_col2, global_col3, global_col4 = st.columns(4)
                
                with global_col1:
                    st.metric("Global MAE", f"{global_metrics['global_mae']:.6f}")
                
                with global_col2:
                    st.metric("Global RMSE", f"{global_metrics['global_rmse']:.6f}")
                
                with global_col3:
                    st.metric("Avg JS Divergence", f"{global_metrics['global_js']:.6f}")
                
                with global_col4:
                    st.metric("Accuracy (Top Emotion)", 
                             f"{global_metrics['accuracy']:.4f}", 
                             f"{global_metrics['correct_predictions']}/{global_metrics['total_predictions']}")
        
        else:
            # Instruções quando nenhum arquivo está carregado
            st.info("""
            ### 📋 How to use this application:
            
            1. **Upload both CSV files**: 
               - **Ground Truth CSV**: Contains the actual emotion distributions
               - **Predictions CSV**: Contains the predicted emotion distributions
            
            2. **Navigate through images**: Use the navigation controls to browse through common images
            
            3. **Analyze results**: View detailed comparisons, error metrics, and visualizations
            
            4. **Calculate global metrics**: Get overall performance metrics across all images
            
            ### 🎯 Expected CSV format:
            - Must contain a `file` column with image paths
            - Emotion columns: `happy`, `contempt`, `elated`, `surprised`, `love`, `protected`, 
              `astonished`, `disgusted`, `angry`, `fearfull`, `sad`, `neutral`
            - Values should be probabilities between 0 and 1
            
            ### ⚠️ Troubleshooting:
            - If you get errors, check that your CSV files have valid file paths
            - Remove any rows with NaN values in the 'file' column
            - Ensure both files have the same emotion column names
            """)

def main():
    app = EmotionComparisonViewer()
    app.render_interface()

if __name__ == "__main__":
    main()