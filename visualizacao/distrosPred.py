import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error

class EmotionComparisonViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Distribution Comparison Viewer")
        self.root.geometry("1400x1000")
        
        self.df1 = None
        self.df2 = None
        self.current_index = 0
        
        # Configure the interface
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure row and column weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Buttons to load CSVs
        ttk.Label(main_frame, text="Ground Truth:").grid(row=0, column=0, pady=5, sticky=tk.W)
        ttk.Button(main_frame, text="Load Ground Truth CSV", command=lambda: self.load_csv(1)).grid(row=0, column=1, pady=5, sticky=tk.W)
        
        ttk.Label(main_frame, text="Predictions:").grid(row=0, column=2, pady=5, sticky=tk.W)
        ttk.Button(main_frame, text="Load Predictions CSV", command=lambda: self.load_csv(2)).grid(row=0, column=3, pady=5, sticky=tk.W)
        
        # Labels to show loaded files
        self.file1_label = ttk.Label(main_frame, text="No file loaded", foreground="blue")
        self.file1_label.grid(row=1, column=1, pady=5, sticky=tk.W)
        
        self.file2_label = ttk.Label(main_frame, text="No file loaded", foreground="red")
        self.file2_label.grid(row=1, column=3, pady=5, sticky=tk.W)
        
        # Navigation controls
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, columnspan=4, pady=10, sticky=tk.W)
        
        ttk.Button(nav_frame, text="First", command=self.first_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Last", command=self.last_image).pack(side=tk.LEFT, padx=5)
        
        # Dropdown for direct selection
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(nav_frame, textvariable=self.image_var, state="readonly", width=50)
        self.image_dropdown.pack(side=tk.LEFT, padx=10)
        self.image_dropdown.bind('<<ComboboxSelected>>', self.on_image_select)
        
        # Label to show current index
        self.index_label = ttk.Label(nav_frame, text="0/0")
        self.index_label.pack(side=tk.LEFT, padx=10)
        
        # Frame for images and charts
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        
        # Frame for images
        images_frame = ttk.Frame(content_frame)
        images_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
        # Frame for image 1 (Ground Truth)
        self.image1_frame = ttk.LabelFrame(images_frame, text="Ground Truth Image", padding="5")
        self.image1_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.image1_frame.columnconfigure(0, weight=1)
        self.image1_frame.rowconfigure(0, weight=1)
        
        self.image1_label = ttk.Label(self.image1_frame, text="Image not available", background="white")
        self.image1_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame for image 2 (Predictions)
        self.image2_frame = ttk.LabelFrame(images_frame, text="Prediction Image", padding="5")
        self.image2_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.image2_frame.columnconfigure(0, weight=1)
        self.image2_frame.rowconfigure(0, weight=1)
        
        self.image2_label = ttk.Label(self.image2_frame, text="Image not available", background="white")
        self.image2_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame for charts
        charts_frame = ttk.Frame(content_frame)
        charts_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Frame for comparison chart
        self.comparison_frame = ttk.LabelFrame(charts_frame, text="Distribution Comparison", padding="5")
        self.comparison_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.comparison_frame.columnconfigure(0, weight=1)
        self.comparison_frame.rowconfigure(0, weight=1)
        
        # Frame for difference chart
        self.diff_frame = ttk.LabelFrame(charts_frame, text="Distribution Differences", padding="5")
        self.diff_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.diff_frame.columnconfigure(0, weight=1)
        self.diff_frame.rowconfigure(0, weight=1)
        
        # Frame for image information
        info_frame = ttk.LabelFrame(main_frame, text="Comparative Information", padding="5")
        info_frame.grid(row=4, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(3, weight=1)
        
        # Ground Truth information
        ttk.Label(info_frame, text="Ground Truth:", foreground="blue").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.file1_path_label = ttk.Label(info_frame, text="", foreground="blue")
        self.file1_path_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Dominant emotion:", foreground="blue").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dominant_emotion1_label = ttk.Label(info_frame, text="", foreground="blue")
        self.dominant_emotion1_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Value:", foreground="blue").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.dominant_value1_label = ttk.Label(info_frame, text="", foreground="blue")
        self.dominant_value1_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Predictions information
        ttk.Label(info_frame, text="Predictions:", foreground="red").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.file2_path_label = ttk.Label(info_frame, text="", foreground="red")
        self.file2_path_label.grid(row=0, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Dominant emotion:", foreground="red").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.dominant_emotion2_label = ttk.Label(info_frame, text="", foreground="red")
        self.dominant_emotion2_label.grid(row=1, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Value:", foreground="red").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.dominant_value2_label = ttk.Label(info_frame, text="", foreground="red")
        self.dominant_value2_label.grid(row=2, column=3, sticky=tk.W, pady=2)
        
        # Error metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Error Metrics", padding="5")
        metrics_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.columnconfigure(3, weight=1)
        
        # Error metrics
        ttk.Label(metrics_frame, text="Mean Absolute Error (MAE):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mae_label = ttk.Label(metrics_frame, text="")
        self.mae_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Root Mean Squared Error (RMSE):").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.rmse_label = ttk.Label(metrics_frame, text="")
        self.rmse_label.grid(row=0, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Jensen-Shannon Divergence:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.js_div_label = ttk.Label(metrics_frame, text="")
        self.js_div_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="KL Divergence:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.kl_div_label = ttk.Label(metrics_frame, text="")
        self.kl_div_label.grid(row=1, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Cosine Similarity:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.cosine_label = ttk.Label(metrics_frame, text="")
        self.cosine_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Correlation:").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.correlation_label = ttk.Label(metrics_frame, text="")
        self.correlation_label.grid(row=2, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Maximum Difference:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.max_diff_label = ttk.Label(metrics_frame, text="")
        self.max_diff_label.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Emotion with Max Difference:").grid(row=3, column=2, sticky=tk.W, pady=2)
        self.max_diff_emotion_label = ttk.Label(metrics_frame, text="")
        self.max_diff_emotion_label.grid(row=3, column=3, sticky=tk.W, pady=2)
        
        # Global metrics frame (for all images)
        global_frame = ttk.LabelFrame(main_frame, text="Global Metrics (All Images)", padding="5")
        global_frame.grid(row=6, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        global_frame.columnconfigure(1, weight=1)
        global_frame.columnconfigure(3, weight=1)
        
        ttk.Button(global_frame, text="Calculate Global Metrics", command=self.calculate_global_metrics).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        ttk.Label(global_frame, text="Global MAE:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.global_mae_label = ttk.Label(global_frame, text="")
        self.global_mae_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(global_frame, text="Global RMSE:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.global_rmse_label = ttk.Label(global_frame, text="")
        self.global_rmse_label.grid(row=1, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(global_frame, text="Avg JS Divergence:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.global_js_label = ttk.Label(global_frame, text="")
        self.global_js_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(global_frame, text="Accuracy (Top Emotion):").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.global_accuracy_label = ttk.Label(global_frame, text="")
        self.global_accuracy_label.grid(row=2, column=3, sticky=tk.W, pady=2)
    
    def load_csv(self, file_num):
        file_path = filedialog.askopenfilename(
            title=f"Select CSV File {file_num}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_num == 1:
                    self.df1 = pd.read_csv(file_path)
                    self.file1_label.config(text=f"File: {os.path.basename(file_path)}")
                else:
                    self.df2 = pd.read_csv(file_path)
                    self.file2_label.config(text=f"File: {os.path.basename(file_path)}")
                
                # Configure dropdown when both files are loaded
                if self.df1 is not None and self.df2 is not None:
                    # Find common images
                    common_images = self.find_common_images()
                    self.image_dropdown['values'] = common_images
                    
                    if common_images:
                        self.current_index = 0
                        self.update_display()
                    else:
                        messagebox.showwarning("Warning", "No common images found between the two files.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def find_common_images(self):
        """Find common images between the two dataframes"""
        if self.df1 is None or self.df2 is None:
            return []
        
        # Extract base filenames
        df1_files = [os.path.basename(row['file']) for _, row in self.df1.iterrows()]
        df2_files = [os.path.basename(row['file']) for _, row in self.df2.iterrows()]
        
        # Find intersection
        common_files = list(set(df1_files) & set(df2_files))
        return sorted(common_files)
    
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
    
    def update_display(self):
        if self.df1 is None or self.df2 is None or len(self.df1) == 0 or len(self.df2) == 0:
            return
        
        # Get current filename
        current_file = self.image_var.get()
        if not current_file:
            common_images = self.find_common_images()
            if common_images:
                current_file = common_images[0]
                self.image_var.set(current_file)
            else:
                return
        
        # Find corresponding rows in both dataframes
        row1 = None
        row2 = None
        
        for _, row in self.df1.iterrows():
            if os.path.basename(row['file']) == current_file:
                row1 = row
                break
                
        for _, row in self.df2.iterrows():
            if os.path.basename(row['file']) == current_file:
                row2 = row
                break
        
        if row1 is None or row2 is None:
            return
        
        # Update index
        common_images = self.find_common_images()
        self.current_index = common_images.index(current_file)
        self.index_label.config(text=f"{self.current_index + 1}/{len(common_images)}")
        
        # Try to load images
        self.load_image(row1['file'], self.image1_label, self.image1_frame, 400)
        self.load_image(row2['file'], self.image2_label, self.image2_frame, 400)
        
        # Update image information
        self.file1_path_label.config(text=row1['file'])
        self.file2_path_label.config(text=row2['file'])
        
        # Find dominant emotions
        emotions = ['happy', 'contempt', 'elated', 'surprised', 'love', 'protected', 
                   'astonished', 'disgusted', 'angry', 'fearfull', 'sad', 'neutral']
        
        # For file 1 (Ground Truth)
        emotion_values1 = [row1[emotion] for emotion in emotions]
        max_index1 = emotion_values1.index(max(emotion_values1))
        dominant_emotion1 = emotions[max_index1]
        dominant_value1 = emotion_values1[max_index1]
        
        self.dominant_emotion1_label.config(text=dominant_emotion1)
        self.dominant_value1_label.config(text=f"{dominant_value1:.4f}")
        
        # For file 2 (Predictions)
        emotion_values2 = [row2[emotion] for emotion in emotions]
        max_index2 = emotion_values2.index(max(emotion_values2))
        dominant_emotion2 = emotions[max_index2]
        dominant_value2 = emotion_values2[max_index2]
        
        self.dominant_emotion2_label.config(text=dominant_emotion2)
        self.dominant_value2_label.config(text=f"{dominant_value2:.4f}")
        
        # Calculate error metrics
        metrics = self.calculate_error_metrics(emotion_values1, emotion_values2)
        
        # Update metrics display
        self.mae_label.config(text=f"{metrics['mae']:.6f}")
        self.rmse_label.config(text=f"{metrics['rmse']:.6f}")
        self.js_div_label.config(text=f"{metrics['js_divergence']:.6f}")
        self.kl_div_label.config(text=f"{metrics['kl_divergence']:.6f}")
        self.cosine_label.config(text=f"{metrics['cosine_similarity']:.6f}")
        self.correlation_label.config(text=f"{metrics['correlation']:.6f}")
        self.max_diff_label.config(text=f"{metrics['max_difference']:.6f}")
        self.max_diff_emotion_label.config(text=f"{emotions[metrics['max_difference_index']]}")
        
        # Update charts
        self.update_comparison_chart(row1, row2, emotions)
        self.update_diff_chart(row1, row2, emotions)
    
    def calculate_global_metrics(self):
        """Calculate global metrics across all images"""
        if self.df1 is None or self.df2 is None:
            messagebox.showwarning("Warning", "Please load both files first.")
            return
        
        emotions = ['happy', 'contempt', 'elated', 'surprised', 'love', 'protected', 
                   'astonished', 'disgusted', 'angry', 'fearfull', 'sad', 'neutral']
        
        all_mae = []
        all_rmse = []
        all_js = []
        correct_predictions = 0
        total_predictions = 0
        
        common_images = self.find_common_images()
        
        for image_name in common_images:
            # Find corresponding rows
            row1 = None
            row2 = None
            
            for _, row in self.df1.iterrows():
                if os.path.basename(row['file']) == image_name:
                    row1 = row
                    break
                    
            for _, row in self.df2.iterrows():
                if os.path.basename(row['file']) == image_name:
                    row2 = row
                    break
            
            if row1 is not None and row2 is not None:
                emotion_values1 = [row1[emotion] for emotion in emotions]
                emotion_values2 = [row2[emotion] for emotion in emotions]
                
                # Calculate metrics
                metrics = self.calculate_error_metrics(emotion_values1, emotion_values2)
                
                all_mae.append(metrics['mae'])
                all_rmse.append(metrics['rmse'])
                all_js.append(metrics['js_divergence'])
                
                # Check if top emotion matches
                gt_top = emotions[emotion_values1.index(max(emotion_values1))]
                pred_top = emotions[emotion_values2.index(max(emotion_values2))]
                
                if gt_top == pred_top:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            # Calculate global averages
            global_mae = np.mean(all_mae)
            global_rmse = np.mean(all_rmse)
            global_js = np.mean(all_js)
            accuracy = correct_predictions / total_predictions
            
            # Update global metrics display
            self.global_mae_label.config(text=f"{global_mae:.6f}")
            self.global_rmse_label.config(text=f"{global_rmse:.6f}")
            self.global_js_label.config(text=f"{global_js:.6f}")
            self.global_accuracy_label.config(text=f"{accuracy:.4f} ({correct_predictions}/{total_predictions})")
        else:
            messagebox.showwarning("Warning", "No common images found for global metrics calculation.")
    
    def load_image(self, image_path, label, frame, size=300):
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                # Resize while maintaining aspect ratio
                img.thumbnail((size, size))
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo, text="")
                label.image = photo  # Keep a reference
            else:
                label.config(image="", text="Image not found")
        except Exception as e:
            label.config(image="", text=f"Error loading image: {str(e)}")
    
    def update_comparison_chart(self, row1, row2, emotions):
        # Clear the chart frame
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Extract values
        values1 = [row1[emotion] for emotion in emotions]
        values2 = [row2[emotion] for emotion in emotions]
        
        # Configure bar positions
        x = np.arange(len(emotions))
        width = 0.35
        
        # Create comparative bar chart
        bars1 = ax.bar(x - width/2, values1, width, label='Ground Truth', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, values2, width, label='Predictions', color='red', alpha=0.7)
        
        ax.set_title('Emotion Distribution Comparison')
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Embed chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_diff_chart(self, row1, row2, emotions):
        # Clear the chart frame
        for widget in self.diff_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Calculate differences
        differences = [row1[emotion] - row2[emotion] for emotion in emotions]
        colors = ['green' if diff >= 0 else 'red' for diff in differences]
        
        # Create bar chart for differences
        bars = ax.bar(emotions, differences, color=colors, alpha=0.7)
        
        ax.set_title('Distribution Differences (Ground Truth - Predictions)')
        ax.set_ylabel('Difference')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add values on bars
        for bar, value in zip(bars, differences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Embed chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.diff_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def first_image(self):
        common_images = self.find_common_images()
        if common_images:
            self.current_index = 0
            self.image_var.set(common_images[self.current_index])
            self.update_display()
    
    def previous_image(self):
        common_images = self.find_common_images()
        if common_images:
            self.current_index = max(0, self.current_index - 1)
            self.image_var.set(common_images[self.current_index])
            self.update_display()
    
    def next_image(self):
        common_images = self.find_common_images()
        if common_images:
            self.current_index = min(len(common_images) - 1, self.current_index + 1)
            self.image_var.set(common_images[self.current_index])
            self.update_display()
    
    def last_image(self):
        common_images = self.find_common_images()
        if common_images:
            self.current_index = len(common_images) - 1
            self.image_var.set(common_images[self.current_index])
            self.update_display()
    
    def on_image_select(self, event):
        common_images = self.find_common_images()
        if common_images:
            selected_image = self.image_var.get()
            self.current_index = common_images.index(selected_image)
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionComparisonViewer(root)
    root.mainloop()