import tkinter as tk
from tkinter import filedialog, ttk
import torch
import os
import sys
import numpy as np
from torchvision import transforms
import cv2
import pandas as pd
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torch import nn
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM

# Mapeamento de emoções para valence e arousal
emotion_to_va = {
    "happy": (0.8, 0.6),
    "contempt": (-0.3, 0.4),
    "elated": (0.9, 0.7),
    "hopeful": (0.7, 0.5),
    "surprised": (0.4, 0.8),
    "proud": (0.6, 0.4),
    "loved": (0.9, 0.3),
    "angry": (-0.8, 0.7),
    "astonished": (0.3, 0.9),
    "disgusted": (-0.7, 0.5),
    "fearful": (-0.6, 0.9),
    "sad": (-0.8, -0.3),
    "fatigued": (-0.5, -0.6),
    "neutral": (0.0, 0.0)
}

# Cores para cada emoção no gráfico de barras
emotion_colors = {
    "happy": "#FFD700",       # Amarelo dourado
    "contempt": "#A9A9A9",    # Cinza escuro
    "elated": "#FF69B4",      # Rosa quente
    "hopeful": "#32CD32",     # Verde Lima
    "surprised": "#00BFFF",   # Azul claro
    "proud": "#FF8C00",       # Laranja escuro
    "loved": "#FF1493",       # Rosa profundo
    "angry": "#FF4500",       # Vermelho laranja
    "astonished": "#4B0082",  # Índigo
    "disgusted": "#8B4513",   # Marrom
    "fearful": "#800080",     # Roxo
    "sad": "#1E90FF",         # Azul dodger
    "fatigued": "#696969",    # Cinza escuro
    "neutral": "#D3D3D3"      # Cinza claro
}

class VideoFeatureExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Feature Extractor")
        self.root.geometry("1200x800")
        
        self.model_path = ""
        self.weights_path = ""
        self.video_path = ""
        self.output_path = ""
        self.output_video_path = ""
        self.current_frame = None
        self.cap = None
        self.features_df = None
        self.frame_thumbnails = []
        self.thumbnail_labels = []
        self.current_selected_frame = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Extraction tab
        self.extraction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.extraction_frame, text="Extraction")
        
        # Visualization tab
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Visualization")
        
        # Video Output tab
        self.output_video_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_video_frame, text="Video Output")
        
        self.setup_extraction_tab()
        self.setup_visualization_tab()
        self.setup_video_output_tab()
        
    def setup_extraction_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.extraction_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right frame for preview
        right_frame = ttk.Frame(self.extraction_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(left_frame, text="Model Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value="resnetBayesGMM")
        model_combo = ttk.Combobox(left_frame, textvariable=self.model_var, 
                                  values=["resnetBayesGMM", "resnetBayes", "resnetAttentionGMM"])
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # ResNet inner model
        ttk.Label(left_frame, text="ResNet Model:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.resnet_var = tk.IntVar(value=18)
        resnet_combo = ttk.Combobox(left_frame, textvariable=self.resnet_var, values=[18, 50])
        resnet_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Weights file selection
        ttk.Label(left_frame, text="Weights File:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(left_frame, text="Browse Weights", command=self.browse_weights).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.weights_label = ttk.Label(left_frame, text="No file selected", wraplength=200)
        self.weights_label.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Video file selection
        ttk.Label(left_frame, text="Video File:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(left_frame, text="Browse Video", command=self.browse_video).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.video_label = ttk.Label(left_frame, text="No file selected", wraplength=200)
        self.video_label.grid(row=5, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Output file selection
        ttk.Label(left_frame, text="Output File:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(left_frame, text="Browse Output", command=self.browse_output).grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.output_label = ttk.Label(left_frame, text="No file selected", wraplength=200)
        self.output_label.grid(row=7, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Batch size
        ttk.Label(left_frame, text="Batch Size:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.batch_var = tk.IntVar(value=16)
        batch_spin = ttk.Spinbox(left_frame, from_=1, to=64, textvariable=self.batch_var)
        batch_spin.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        
        # Progress bar
        ttk.Label(left_frame, text="Progress:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=9, column=1, padx=5, pady=5, sticky="ew")
        
        # Status label
        self.status_label = ttk.Label(left_frame, text="Ready", wraplength=200)
        self.status_label.grid(row=10, column=0, columnspan=2, padx=5, pady=10)
        
        # Buttons frame
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=11, column=0, columnspan=2, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Extract Features", command=self.extract_features).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Load Results", command=self.load_results).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Go to Visualization", 
                  command=lambda: self.notebook.select(1)).pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Preview area in right frame
        ttk.Label(right_frame, text="Video Preview", font=('Arial', 12, 'bold')).pack(pady=5)
        self.video_label_display = ttk.Label(right_frame, text="Video will appear here", background="black")
        self.video_label_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(nav_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.frame_spin = ttk.Spinbox(nav_frame, from_=0, to=0, textvariable=self.frame_var, width=8)
        self.frame_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Show Frame", command=self.show_frame).pack(side=tk.LEFT, padx=5)
        
        # Results display
        ttk.Label(right_frame, text="Frame Details", font=('Arial', 12, 'bold')).pack(pady=5)
        self.results_text = tk.Text(right_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure grid weights
        left_frame.columnconfigure(1, weight=1)
        
    def setup_visualization_tab(self):
        # Main frame for visualization
        main_viz_frame = ttk.Frame(self.visualization_frame)
        main_viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_viz_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Refresh Visualization", 
                  command=self.refresh_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Back to Extraction", 
                  command=lambda: self.notebook.select(0)).pack(side=tk.LEFT, padx=5)
        
        # Frame range controls
        range_frame = ttk.Frame(controls_frame)
        range_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(range_frame, text="Start Frame:").pack(side=tk.LEFT)
        self.start_frame_var = tk.IntVar(value=0)
        self.start_frame_spin = ttk.Spinbox(range_frame, from_=0, to=0, 
                                           textvariable=self.start_frame_var, width=6)
        self.start_frame_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_frame, text="End Frame:").pack(side=tk.LEFT, padx=(10,0))
        self.end_frame_var = tk.IntVar(value=50)
        self.end_frame_spin = ttk.Spinbox(range_frame, from_=0, to=0, 
                                         textvariable=self.end_frame_var, width=6)
        self.end_frame_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(range_frame, text="Update Range", 
                  command=self.update_visualization_range).pack(side=tk.LEFT, padx=5)
        
        # Thumbnails frame with scrollbar
        thumbnails_container = ttk.Frame(main_viz_frame)
        thumbnails_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas and scrollbar for thumbnails
        self.canvas = tk.Canvas(thumbnails_container, bg='white')
        scrollbar = ttk.Scrollbar(thumbnails_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel to scroll
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Details frame at bottom
        details_frame = ttk.LabelFrame(main_viz_frame, text="Frame Details", padding=10)
        details_frame.pack(fill=tk.X, pady=5)
        
        self.details_text = tk.Text(details_frame, height=6, width=80)
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_video_output_tab(self):
        """Setup tab for generating output video with emotional data"""
        main_frame = ttk.Frame(self.output_video_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel - preview
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Generate Video with Emotional Data", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Output video path
        ttk.Label(left_frame, text="Output Video:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(left_frame, text="Browse Output", command=self.browse_output_video).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.output_video_label = ttk.Label(left_frame, text="No file selected", wraplength=200)
        self.output_video_label.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Video quality settings
        ttk.Label(left_frame, text="Video Quality:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.quality_var = tk.StringVar(value="medium")
        quality_combo = ttk.Combobox(left_frame, textvariable=self.quality_var,
                                    values=["low", "medium", "high", "original"])
        quality_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Display options
        ttk.Label(left_frame, text="Display Options:", font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, pady=(20,5))
        
        self.show_emotions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Show Emotions", variable=self.show_emotions_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=5)
        
        self.show_vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Show VAD Values", variable=self.show_vad_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=5)
        
        self.show_progress_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Show Progress Bar", variable=self.show_progress_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=5)
        
        self.show_dominant_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Highlight Dominant Emotion", variable=self.show_dominant_var).grid(row=8, column=0, columnspan=2, sticky="w", padx=5)
        
        # Number of emotions to show
        ttk.Label(left_frame, text="Max Emotions:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.top_emotions_var = tk.IntVar(value=5)
        top_emotions_spin = ttk.Spinbox(left_frame, from_=1, to=10, textvariable=self.top_emotions_var, width=5)
        top_emotions_spin.grid(row=9, column=1, padx=5, pady=5, sticky="w")
        
        # Font size
        ttk.Label(left_frame, text="Font Size:").grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.font_size_var = tk.IntVar(value=20)
        font_size_spin = ttk.Spinbox(left_frame, from_=12, to=36, textvariable=self.font_size_var, width=5)
        font_size_spin.grid(row=10, column=1, padx=5, pady=5, sticky="w")
        
        # Info panel width
        ttk.Label(left_frame, text="Info Panel Width:").grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.panel_width_var = tk.IntVar(value=600)
        panel_width_spin = ttk.Spinbox(left_frame, from_=300, to=800, textvariable=self.panel_width_var, width=5)
        panel_width_spin.grid(row=11, column=1, padx=5, pady=5, sticky="w")
        
        # Generate button
        ttk.Button(left_frame, text="Generate Video", command=self.generate_output_video).grid(row=12, column=0, columnspan=2, pady=20, sticky="ew")
        
        # Progress for video generation
        ttk.Label(left_frame, text="Video Generation Progress:").grid(row=13, column=0, columnspan=2, pady=5)
        self.video_progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.video_progress.grid(row=14, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.video_status_label = ttk.Label(left_frame, text="Ready to generate video", wraplength=200)
        self.video_status_label.grid(row=15, column=0, columnspan=2, padx=5, pady=10)
        
        # Preview area
        ttk.Label(right_frame, text="Sample Frame Preview", font=('Arial', 12, 'bold')).pack(pady=5)
        self.sample_frame_label = ttk.Label(right_frame, text="Sample frame will be shown here", 
                                           background="black", foreground="white")
        self.sample_frame_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(right_frame, text="Generate Sample Frame", 
                  command=self.generate_sample_frame).pack(pady=10)
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def browse_weights(self):
        filename = filedialog.askopenfilename(
            title="Select weights file",
            filetypes=[("PyTorch files", "*.pth *.pt *.tar"), ("All files", "*.*")]
        )
        if filename:
            self.weights_path = filename
            self.weights_label.config(text=os.path.basename(filename))
            
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path = filename
            self.video_label.config(text=os.path.basename(filename))
            self.load_video_preview()
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_path = filename
            self.output_label.config(text=os.path.basename(filename))
            
    def browse_output_video(self):
        filename = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            self.output_video_path = filename
            self.output_video_label.config(text=os.path.basename(filename))
            
    def load_video_preview(self):
        """Load video for preview"""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_spin.config(to=max(0, total_frames-1))
        self.start_frame_spin.config(to=max(0, total_frames-1))
        self.end_frame_spin.config(to=max(0, total_frames-1))
        self.show_frame()
            
    def show_frame(self):
        """Show specific frame from video"""
        if not self.cap:
            return
            
        frame_idx = self.frame_var.get()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.display_frame(frame, frame_idx, self.video_label_display)
            self.display_frame_features(frame_idx)
            
    def display_frame(self, frame, frame_idx, label_widget):
        """Display frame in a label widget"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        h, w = frame_rgb.shape[:2]
        max_size = 400
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
            
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        label_widget.configure(image=imgtk)
        label_widget.image = imgtk
        return frame_resized
        
    def get_dominant_emotion(self, frame_data):
        """Get the dominant emotion from frame data"""
        emotions = [
            "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
            'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        max_emotion = ""
        max_value = -1
        
        for emotion in emotions:
            if emotion in frame_data:
                if frame_data[emotion] > max_value:
                    max_value = frame_data[emotion]
                    max_emotion = emotion
                    
        return max_emotion, max_value
        
    def display_frame_features(self, frame_idx):
        """Display features for specific frame"""
        if self.features_df is not None and frame_idx < len(self.features_df):
            frame_data = self.features_df.iloc[frame_idx]
            dominant_emotion, dominant_value = self.get_dominant_emotion(frame_data)
            
            emotions = [
                "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
                'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
            ]
            
            results_text = f"Frame: {frame_idx}\n"
            results_text += f"Timestamp: {frame_data['timestamp']:.2f}s\n"
            results_text += f"Dominant Emotion: {dominant_emotion} ({dominant_value:.3f})\n\n"
            results_text += "Emotions:\n"
            
            for emotion in emotions:
                if emotion in frame_data:
                    marker = "▶" if emotion == dominant_emotion else " "
                    results_text += f"{marker} {emotion}: {frame_data[emotion]:.4f}\n"
            
            results_text += f"\nVAD:\n"
            results_text += f"  Valence: {frame_data['valence']:.4f}\n"
            results_text += f"  Arousal: {frame_data['arousal']:.4f}\n"
            results_text += f"  Dominance: {frame_data['dominance']:.4f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_text)
            
    def create_thumbnail(self, frame_idx, width=120, height=90):
        """Create thumbnail for a specific frame"""
        if not self.cap or frame_idx >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            return None
            
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if ret:
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            return ImageTk.PhotoImage(image=img)
        return None
        
    def refresh_visualization(self):
        """Refresh the visualization with current data"""
        if self.features_df is None:
            self.status_label.config(text="Error: No results loaded")
            return
            
        self.update_visualization_range()
        
    def update_visualization_range(self):
        """Update visualization for the selected frame range"""
        if self.features_df is None or not self.cap:
            return
            
        start_frame = self.start_frame_var.get()
        end_frame = self.end_frame_var.get()
        
        # Clear existing thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.thumbnail_labels = []
        self.frame_thumbnails = []
        
        # Create thumbnails for the range
        frames_per_row = 5
        row = 0
        col = 0
        
        for frame_idx in range(start_frame, min(end_frame + 1, len(self.features_df))):
            # Create thumbnail
            thumbnail = self.create_thumbnail(frame_idx)
            if thumbnail is None:
                continue
                
            frame_data = self.features_df.iloc[frame_idx]
            dominant_emotion, dominant_value = self.get_dominant_emotion(frame_data)
            
            # Create frame for thumbnail
            thumb_frame = ttk.Frame(self.scrollable_frame, relief='raised', borderwidth=1)
            thumb_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Create label for thumbnail
            label = ttk.Label(thumb_frame, image=thumbnail, cursor="hand2")
            label.image = thumbnail  # Keep a reference
            label.bind("<Button-1>", lambda e, idx=frame_idx: self.show_frame_details(idx))
            label.pack(padx=2, pady=2)
            
            # Add frame info
            info_text = f"Frame: {frame_idx}\n{dominant_emotion}\n({dominant_value:.2f})"
            info_label = ttk.Label(thumb_frame, text=info_text, font=('Arial', 8), 
                                 justify=tk.CENTER, wraplength=100)
            info_label.pack(padx=2, pady=2)
            
            self.thumbnail_labels.append(label)
            self.frame_thumbnails.append(thumbnail)
            
            # Update grid position
            col += 1
            if col >= frames_per_row:
                col = 0
                row += 1
                
        # Configure grid weights for responsive layout
        for i in range(frames_per_row):
            self.scrollable_frame.columnconfigure(i, weight=1)
            
    def show_frame_details(self, frame_idx):
        """Show detailed information for a clicked frame"""
        if self.features_df is not None and frame_idx < len(self.features_df):
            frame_data = self.features_df.iloc[frame_idx]
            dominant_emotion, dominant_value = self.get_dominant_emotion(frame_data)
            
            emotions = [
                "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
                'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
            ]
            
            details_text = f"=== Frame {frame_idx} Details ===\n"
            details_text += f"Timestamp: {frame_data['timestamp']:.2f}s\n"
            details_text += f"Dominant Emotion: {dominant_emotion} ({dominant_value:.3f})\n\n"
            
            details_text += "Emotion Probabilities:\n"
            for emotion in emotions:
                if emotion in frame_data:
                    marker = "▶" if emotion == dominant_emotion else " "
                    details_text += f"{marker} {emotion:12}: {frame_data[emotion]:.4f}\n"
            
            details_text += f"\nVAD Values:\n"
            details_text += f"  Valence:   {frame_data['valence']:.4f}\n"
            details_text += f"  Arousal:   {frame_data['arousal']:.4f}\n"
            details_text += f"  Dominance: {frame_data['dominance']:.4f}\n"
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details_text)
            
            # Update the main frame view as well
            self.frame_var.set(frame_idx)
            self.show_frame()

    def load_model(self):
        """Load the selected model with weights - supports .tar files"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model based on selection
        model = None
        if self.model_var.get() == "resnetBayesGMM":
            model = ResnetWithBayesianGMMHead(classes=13, resnetModel=self.resnet_var.get())
        elif self.model_var.get() == "resnetBayes":
            model = ResnetWithBayesianHead(13, resnetModel=self.resnet_var.get())
        elif self.model_var.get() == "resnetAttentionGMM":
            model = ResNet50WithAttentionGMM(num_classes=14, bottleneck='none', bayesianHeadType='VAD')
        
        # Load weights - support for .tar files
        checkpoint = torch.load(self.weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # Standard .pth/.pt format
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif 'model_state_dict' in checkpoint:
            # Alternative format
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            # Try loading directly (for .tar files or custom formats)
            model.load_state_dict(checkpoint, strict=True)
            
        model.to(device)
        model.eval()
        
        return model, device
        
    def extract_features(self):
        """Extract features from video frame by frame"""
        if not all([self.weights_path, self.video_path, self.output_path]):
            self.status_label.config(text="Error: Please select all required files")
            return
            
        try:
            self.status_label.config(text="Loading model...")
            self.root.update()
            
            model, device = self.load_model()
            
            # Define transforms
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.status_label.config(text="Opening video...")
            self.root.update()
            
            # Open video file
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                self.status_label.config(text="Error: Could not read video file")
                return
                
            self.progress['maximum'] = total_frames
            
            # Initialize lists to store results
            all_predictions = []
            all_vad_preds = []
            frame_numbers = []
            timestamps = []
            
            batch_size = self.batch_var.get()
            batch_frames = []
            batch_indices = []
            
            soft = nn.Softmax(dim=1)
            frame_count = 0
            
            self.status_label.config(text="Processing frames...")
            
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Apply transforms
                    frame_tensor = data_transforms(frame_rgb).unsqueeze(0)
                    batch_frames.append(frame_tensor)
                    batch_indices.append(frame_count)
                    
                    # Process batch when full
                    if len(batch_frames) >= batch_size:
                        batch_tensor = torch.cat(batch_frames, 0).to(device)
                        
                        # Get model predictions
                        outputs, _, vad = model(batch_tensor)
                        outputs = soft(outputs)
                        
                        # Store results
                        predictions = outputs.cpu().numpy()
                        vad_preds = vad.cpu().numpy()
                        
                        all_predictions.extend(predictions)
                        all_vad_preds.extend(vad_preds)
                        frame_numbers.extend(batch_indices)
                        timestamps.extend([i / fps for i in batch_indices])
                        
                        # Clear batch
                        batch_frames = []
                        batch_indices = []
                        
                    frame_count += 1
                    self.progress['value'] = frame_count
                    self.status_label.config(text=f"Processing frame {frame_count}/{total_frames}")
                    self.root.update()
                
                # Process remaining frames in batch
                if batch_frames:
                    batch_tensor = torch.cat(batch_frames, 0).to(device)
                    outputs, _, vad = model(batch_tensor)
                    outputs = soft(outputs)
                    
                    predictions = outputs.cpu().numpy()
                    vad_preds = vad.cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_vad_preds.extend(vad_preds)
                    frame_numbers.extend(batch_indices)
                    timestamps.extend([i / fps for i in batch_indices])
            
            cap.release()
            
            # Save results to CSV
            self.save_results(all_predictions, all_vad_preds, frame_numbers, timestamps)
            
            self.status_label.config(text=f"Completed! Processed {frame_count} frames")
            self.progress['value'] = 0
            
            # Load results for visualization
            self.load_results()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            
    def save_results(self, predictions, vad_preds, frame_numbers, timestamps):
        """Save extracted features to CSV file"""
        emotions = [
            "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
            'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        # Create DataFrame
        data = {}
        
        # Add emotion predictions
        for i, emotion in enumerate(emotions):
            if i < len(predictions[0]):  # Ensure we don't exceed prediction dimensions
                data[emotion] = [pred[i] for pred in predictions]
        
        # Add VAD predictions
        data['valence'] = [vad[0] for vad in vad_preds]
        data['arousal'] = [vad[1] for vad in vad_preds]
        data['dominance'] = [vad[2] for vad in vad_preds]
        
        # Add frame info
        data['frame_number'] = frame_numbers
        data['timestamp'] = timestamps
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_path, index=False)
        
    def load_results(self):
        """Load results from CSV file for visualization"""
        if not self.output_path or not os.path.exists(self.output_path):
            self.status_label.config(text="Error: Output file not found")
            return
            
        try:
            self.features_df = pd.read_csv(self.output_path)
            self.status_label.config(text=f"Loaded {len(self.features_df)} frames")
            
            # Update frame spinner range
            if self.cap:
                total_frames = len(self.features_df)
                self.frame_spin.config(to=max(0, total_frames-1))
                self.start_frame_spin.config(to=max(0, total_frames-1))
                self.end_frame_spin.config(to=max(0, total_frames-1))
                
            # Show first frame features
            self.show_frame()
            
        except Exception as e:
            self.status_label.config(text=f"Error loading results: {str(e)}")

    def get_top_emotions(self, frame_data, n=5):
        """Get top n emotions with highest probabilities"""
        emotions = [
            "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
            'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        emotion_values = []
        for emotion in emotions:
            if emotion in frame_data:
                emotion_values.append((emotion, frame_data[emotion]))
        
        # Sort by probability descending
        emotion_values.sort(key=lambda x: x[1], reverse=True)
        return emotion_values[:n]

    def create_valence_arousal_plot(self, frame_data):
        """Cria um gráfico do plano valence-arousal com as emoções plotadas"""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Configurar o gráfico
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_title('Valence-Arousal')
        
        # Plotar as emoções baseadas nas probabilidades
        for emotion in emotion_to_va.keys():
            if emotion in frame_data:
                va = emotion_to_va[emotion]
                prob = frame_data[emotion]
                size = 30 + (prob * 200)  # Tamanho do marcador baseado na probabilidade
                alpha = 0.5 + (prob * 0.5)  # Transparência baseada na probabilidade
                ax.scatter(va[0], va[1], s=size, alpha=alpha, 
                          c=emotion_colors[emotion], edgecolors='black')
        
        # Plotar o ponto atual VAD
        current_va = (frame_data['valence'], frame_data['arousal'])
        ax.scatter(current_va[0], current_va[1], s=100, c='red', 
                  marker='X', edgecolors='black', label='Current')
        
        # Adicionar quadrantes
        ax.text(0.5, 0.5, 'Happy\nAroused', ha='center', va='center', 
                fontsize=8, alpha=0.7, bbox=dict(facecolor='yellow', alpha=0.2))
        ax.text(-0.5, 0.5, 'Sad\nAngry', ha='center', va='center', 
                fontsize=8, alpha=0.7, bbox=dict(facecolor='red', alpha=0.2))
        ax.text(-0.5, -0.5, 'Calm\nUnpleasant', ha='center', va='center', 
                fontsize=8, alpha=0.7, bbox=dict(facecolor='blue', alpha=0.2))
        ax.text(0.5, -0.5, 'Relaxed\nPleasant', ha='center', va='center', 
                fontsize=8, alpha=0.7, bbox=dict(facecolor='green', alpha=0.2))
        
        plt.tight_layout()
        return fig

    def create_emotion_bar_chart(self, frame_data):
        """Cria um gráfico de barras para as probabilidades das emoções"""
        emotions = [
            "happy", "contempt", "elated", "hopeful", "surprised", 'proud', 'loved',
            'angry', 'astonished', 'disgusted', 'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        # Filtrar emoções com probabilidade > 0.01
        filtered_emotions = []
        filtered_probs = []
        colors = []
        
        for emotion in emotions:
            if emotion in frame_data and frame_data[emotion] > 0.01:
                filtered_emotions.append(emotion)
                filtered_probs.append(frame_data[emotion])
                colors.append(emotion_colors[emotion])
        
        if not filtered_emotions:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Criar barras coloridas
        bars = ax.bar(filtered_emotions, filtered_probs, color=colors, alpha=0.7)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{filtered_probs[i]:.2f}', ha='center', va='bottom', 
                    rotation=0, fontsize=7)
        
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Distribution')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        
        return fig

    def fig_to_array(self, fig):
        """Converte uma figura matplotlib para um array numpy"""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_arr = np.asarray(buf)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return img_arr

    def calculate_panel_height(self, frame_data, total_frames):
        """Calculate the required height for the info panel based on content"""
        base_height = 100  # Header and basic info
        
        # Dominant emotion section
        if self.show_dominant_var.get():
            base_height += 130
        
        # Emotion bar chart
        if self.show_emotions_var.get():
            base_height += 220  # Graph height + spacing
        
        # VAD plot and values
        if self.show_vad_var.get():
            base_height += 250  # Graph height + values + spacing
        
        # Progress bar
        if self.show_progress_var.get() and total_frames > 0:
            base_height += 60
        
        # Add margin
        base_height += 40
        
        return base_height

    def create_combined_frame(self, video_frame, frame_data, frame_idx, total_frames):
        """Create a combined frame with video on left and emotional data on right"""
        # Redimensionar o vídeo para 255x255
        target_video_size = 255
        video_resized = cv2.resize(video_frame, (target_video_size, target_video_size))
        
        # Convert BGR to RGB for PIL
        video_frame_rgb = cv2.cvtColor(video_resized, cv2.COLOR_BGR2RGB)
        video_pil = Image.fromarray(video_frame_rgb)
        
        # Get dimensions
        video_width, video_height = target_video_size, target_video_size
        panel_width = self.panel_width_var.get()
        
        # Calcular altura necessária para o painel baseado no conteúdo
        panel_height = self.calculate_panel_height(frame_data, total_frames)
        
        # Usar a maior altura entre vídeo e painel
        combined_height = max(video_height, panel_height)
        combined_width = video_width + panel_width
        
        # Create combined image
        combined_image = Image.new('RGB', (combined_width, combined_height), 'black')
        
        # Paste video frame on the left (centralizado verticalmente)
        video_y = (combined_height - video_height) // 2
        combined_image.paste(video_pil, (0, video_y))
        
        # Create info panel on the right
        info_panel = Image.new('RGB', (panel_width, combined_height), (40, 40, 60))
        draw = ImageDraw.Draw(info_panel)
        
        # Define fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", self.font_size_var.get() + 4)
            font_medium = ImageFont.truetype("arial.ttf", self.font_size_var.get())
            font_small = ImageFont.truetype("arial.ttf", self.font_size_var.get() - 2)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Starting position for text
        x_pos = 20
        y_pos = 20
        
        # Header
        draw.text((x_pos, y_pos), "EMOTIONAL ANALYSIS", fill='white', font=font_large)
        y_pos += 50
        
        # Frame information
        draw.text((x_pos, y_pos), f"Frame: {frame_idx}/{total_frames}", fill='cyan', font=font_medium)
        y_pos += 30
        draw.text((x_pos, y_pos), f"Time: {frame_data['timestamp']:.2f}s", fill='cyan', font=font_medium)
        y_pos += 50
        
        # Dominant emotion
        if self.show_dominant_var.get():
            dominant_emotion, dominant_value = self.get_dominant_emotion(frame_data)
            if dominant_emotion:
                draw.text((x_pos, y_pos), "DOMINANT EMOTION:", fill='yellow', font=font_medium)
                y_pos += 30
                draw.text((x_pos + 20, y_pos), f"★ {dominant_emotion.upper()}", fill='yellow', font=font_large)
                y_pos += 40
                draw.text((x_pos + 20, y_pos), f"Confidence: {dominant_value:.3f}", fill='lightgreen', font=font_medium)
                y_pos += 60
        
        # Create matplotlib visualizations
        viz_y_pos = y_pos
        
        if self.show_emotions_var.get():
            # Emotion bar chart
            bar_fig = self.create_emotion_bar_chart(frame_data)
            if bar_fig is not None:
                bar_img = self.fig_to_array(bar_fig)
                # Convert numpy array to PIL Image
                bar_pil = Image.fromarray(cv2.cvtColor(bar_img, cv2.COLOR_BGR2RGB))
                # Resize to fit panel
                bar_width = panel_width - 40
                bar_height = 200
                bar_pil = bar_pil.resize((bar_width, bar_height), Image.LANCZOS)
                info_panel.paste(bar_pil, (x_pos, viz_y_pos))
                viz_y_pos += bar_height + 20
        
        if self.show_vad_var.get():
            # Valence-Arousal plot
            va_fig = self.create_valence_arousal_plot(frame_data)
            if va_fig is not None:
                va_img = self.fig_to_array(va_fig)
                # Convert numpy array to PIL Image
                va_pil = Image.fromarray(cv2.cvtColor(va_img, cv2.COLOR_BGR2RGB))
                # Resize to fit panel
                va_width = panel_width - 40
                va_height = 200
                va_pil = va_pil.resize((va_width, va_height), Image.LANCZOS)
                info_panel.paste(va_pil, (x_pos, viz_y_pos))
                viz_y_pos += va_height + 30
            
            # VAD values text
            draw.text((x_pos, viz_y_pos), "VAD VALUES:", fill='magenta', font=font_medium)
            viz_y_pos += 30
            draw.text((x_pos + 10, viz_y_pos), f"Valence: {frame_data['valence']:.3f}", fill='lightblue', font=font_small)
            viz_y_pos += 25
            draw.text((x_pos + 10, viz_y_pos), f"Arousal: {frame_data['arousal']:.3f}", fill='lightcoral', font=font_small)
            viz_y_pos += 25
            draw.text((x_pos + 10, viz_y_pos), f"Dominance: {frame_data['dominance']:.3f}", fill='lightgreen', font=font_small)
            viz_y_pos += 40
        
        # Progress bar at bottom
        if self.show_progress_var.get() and total_frames > 0:
            progress_width = panel_width - 40
            progress_height = 15
            progress_x = 20
            progress_y = combined_height - 40
            
            # Progress text
            progress = frame_idx / total_frames
            progress_text = f"Overall Progress: {progress*100:.1f}%"
            draw.text((progress_x, progress_y - 20), progress_text, fill='white', font=font_small)
            
            # Progress bar background
            draw.rectangle([progress_x, progress_y, progress_x + progress_width, progress_y + progress_height], 
                         fill=(100, 100, 100))
            
            # Progress bar fill
            fill_width = int(progress_width * progress)
            draw.rectangle([progress_x, progress_y, progress_x + fill_width, progress_y + progress_height], 
                         fill=(0, 255, 0))
        
        # Paste info panel on the right side of combined image
        combined_image.paste(info_panel, (video_width, 0))
        
        return cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)
    
    def generate_sample_frame(self):
        """Generate a sample frame with emotional data panel for preview"""
        if self.features_df is None or not self.cap:
            self.video_status_label.config(text="Error: No results or video loaded")
            return
        
        frame_idx = min(0, len(self.features_df) - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            combined_frame = self.create_combined_frame(
                frame, self.features_df.iloc[frame_idx], frame_idx, len(self.features_df)
            )
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            frame_resized = self.resize_frame_for_display(frame_rgb, 600)
            
            # Display in preview
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.sample_frame_label.configure(image=imgtk)
            self.sample_frame_label.image = imgtk
            
            self.video_status_label.config(text="Sample frame generated successfully")
    
    def resize_frame_for_display(self, frame, max_size=600):
        """Resize frame for display in GUI"""
        h, w = frame.shape[:2]
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        return cv2.resize(frame, (new_w, new_h))
    
    def generate_output_video(self):
        """Generate output video with emotional data panel"""
        if not hasattr(self, 'output_video_path') or not self.output_video_path:
            self.video_status_label.config(text="Error: Please select output video path")
            return
        
        if self.features_df is None or not self.cap:
            self.video_status_label.config(text="Error: No results or video loaded")
            return
        
        try:
            self.video_status_label.config(text="Initializing video writer...")
            self.root.update()
            
            # Get video properties
            total_frames = len(self.features_df)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Definir dimensões fixas para garantir que tudo caiba
            video_width = 255  # Tamanho fixo do vídeo
            panel_width = self.panel_width_var.get()
            
            # Calcular altura máxima necessária
            max_panel_height = 0
            for frame_idx in range(min(10, total_frames)):  # Amostrar os primeiros 10 frames
                if frame_idx < len(self.features_df):
                    frame_data = self.features_df.iloc[frame_idx]
                    panel_height = self.calculate_panel_height(frame_data, total_frames)
                    max_panel_height = max(max_panel_height, panel_height)
            
            # Garantir altura mínima
            video_height = 255
            output_height = max(video_height, max_panel_height)
            output_width = video_width + panel_width
            
            # Ajustar para qualidades específicas
            if self.quality_var.get() == "low":
                output_width = 854
                output_height = 480
            elif self.quality_var.get() == "medium":
                output_width = 1280
                output_height = 720
            elif self.quality_var.get() == "high":
                output_width = 1920
                output_height = 1080
            elif self.quality_var.get() == "original":
                # Manter as dimensões calculadas para original
                pass
            
            output_fps = fps if self.quality_var.get() == "original" else 25
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video_path, fourcc, output_fps, (output_width, output_height))
            
            self.video_progress['maximum'] = total_frames
            self.video_progress['value'] = 0
            
            # Process each frame
            for frame_idx in range(total_frames):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Create combined frame with emotional data panel
                combined_frame = self.create_combined_frame(
                    frame, self.features_df.iloc[frame_idx], frame_idx, total_frames
                )
                
                # Resize to output dimensions
                frame_resized = cv2.resize(combined_frame, (output_width, output_height))
                
                # Write frame
                out.write(frame_resized)
                
                # Update progress
                self.video_progress['value'] = frame_idx + 1
                self.video_status_label.config(text=f"Processing frame {frame_idx + 1}/{total_frames}")
                self.root.update()
            
            # Release resources
            out.release()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
            
            self.video_status_label.config(text=f"Video generated successfully: {self.output_video_path}")
            
        except Exception as e:
            self.video_status_label.config(text=f"Error generating video: {str(e)}")

def main():
    root = tk.Tk()
    app = VideoFeatureExtractor(root)
    root.mainloop()

if __name__ == '__main__':
    main()