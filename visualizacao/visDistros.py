import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, ttk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

class EmotionSphereVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Esferas Emocionais 3D")
        self.root.geometry("1200x900")
        
        self.df = None
        self.current_figure = None
        self.sphere_artists = []
        self.label_artists = []  # Para armazenar os rótulos
        self.show_labels = True  # Estado inicial - rótulos visíveis
        
        self.create_widgets()
        
        # Variáveis de controle 3D
        self.elevation = 30
        self.azimuth = 45
        self.sphere_quality = 20
    
    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de controles
        control_frame = tk.Frame(main_frame, padx=10, pady=10, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Frame do gráfico
        self.graph_frame = tk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Controles de arquivo
        file_frame = tk.LabelFrame(control_frame, text="Arquivo", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file_frame, text="Carregar CSV", command=self.load_file).pack(fill=tk.X)
        self.file_label = tk.Label(file_frame, text="Nenhum arquivo carregado")
        self.file_label.pack(fill=tk.X)
        
        # Controles 3D
        view_frame = tk.LabelFrame(control_frame, text="Visualização 3D", padx=5, pady=5)
        view_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(view_frame, text="Elevação:").pack()
        self.elev_slider = tk.Scale(view_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                                   command=self.update_view)
        self.elev_slider.set(30)
        self.elev_slider.pack(fill=tk.X)
        
        tk.Label(view_frame, text="Azimute:").pack()
        self.azim_slider = tk.Scale(view_frame, from_=0, to=360, orient=tk.HORIZONTAL,
                                    command=self.update_view)
        self.azim_slider.set(45)
        self.azim_slider.pack(fill=tk.X)
        
        # Controles de visualização
        display_frame = tk.LabelFrame(control_frame, text="Exibição", padx=5, pady=5)
        display_frame.pack(fill=tk.X, pady=5)
        
        # Botão para mostrar/esconder rótulos
        self.toggle_labels_btn = tk.Button(
            display_frame, 
            text="Ocultar Rótulos", 
            command=self.toggle_labels
        )
        self.toggle_labels_btn.pack(fill=tk.X)
        
        # Controles de esferas
        sphere_frame = tk.LabelFrame(control_frame, text="Configurações das Esferas", padx=5, pady=5)
        sphere_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sphere_frame, text="Tamanho base:").pack()
        self.size_slider = tk.Scale(sphere_frame, from_=0.1, to=2, resolution=0.1,
                                   orient=tk.HORIZONTAL, command=self.update_spheres)
        self.size_slider.set(0.5)
        self.size_slider.pack(fill=tk.X)
        
        tk.Label(sphere_frame, text="Transparência:").pack()
        self.alpha_slider = tk.Scale(sphere_frame, from_=0.1, to=1, resolution=0.1,
                                    orient=tk.HORIZONTAL, command=self.update_spheres)
        self.alpha_slider.set(0.6)
        self.alpha_slider.pack(fill=tk.X)
        
        # Lista de emoções
        emotion_frame = tk.LabelFrame(control_frame, text="Emoções", padx=5, pady=5)
        emotion_frame.pack(fill=tk.BOTH, expand=True)
        
        self.emotion_listbox = tk.Listbox(emotion_frame, selectmode=tk.MULTIPLE, height=15)
        self.emotion_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(emotion_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.emotion_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.emotion_listbox.yview)
        
        # Botões de ação
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Selecionar Todas", command=self.select_all).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Limpar Seleção", command=self.clear_selection).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Plotar Esferas", command=self.plot_spheres).pack(side=tk.LEFT, expand=True)
        tk.Button(button_frame, text="Salvar Visualização", command=self.save_visualization).pack(side=tk.LEFT, expand=True)
    
    def toggle_labels(self):
        """Alterna a visibilidade dos rótulos"""
        self.show_labels = not self.show_labels
        
        if self.current_figure:
            for label in self.label_artists:
                label.set_visible(self.show_labels)
            
            # Atualiza o texto do botão
            self.toggle_labels_btn.config(
                text="Mostrar Rótulos" if not self.show_labels else "Ocultar Rótulos"
            )
            
            # Redesenha o canvas
            self.current_figure.canvas.draw()
    
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                self.df = pd.read_csv(filepath)
                
                # Verificar e criar colunas de dominância se necessário
                if 'dominance mean' not in self.df.columns:
                    self.df['dominance mean'] = np.random.uniform(-1, 1, size=len(self.df))
                if 'dominance std' not in self.df.columns:
                    self.df['dominance std'] = np.random.uniform(0.1, 0.3, size=len(self.df))
                
                self.file_label.config(text=f"Arquivo: {filepath.split('/')[-1]}")
                self.update_emotion_list()
            except Exception as e:
                tk.messagebox.showerror("Erro", f"Não foi possível carregar o arquivo:\n{str(e)}")
    
    def update_emotion_list(self):
        if self.df is not None and 'class' in self.df.columns:
            self.emotion_listbox.delete(0, tk.END)
            for emotion in self.df['class']:
                self.emotion_listbox.insert(tk.END, emotion)
    
    def select_all(self):
        self.emotion_listbox.selection_set(0, tk.END)
    
    def clear_selection(self):
        self.emotion_listbox.selection_clear(0, tk.END)
    
    def update_view(self, *args):
        if self.current_figure:
            self.elevation = self.elev_slider.get()
            self.azimuth = self.azim_slider.get()
            for artist in self.sphere_artists:
                artist.remove()
            self.sphere_artists = []
            self.label_artists = []  # Limpa os rótulos antigos
            self.plot_spheres()
    
    def update_spheres(self, *args):
        if self.current_figure:
            for artist in self.sphere_artists:
                artist.remove()
            self.sphere_artists = []
            self.label_artists = []  # Limpa os rótulos antigos
            self.plot_spheres()
    
    def plot_spheres(self):
        if self.df is None:
            tk.messagebox.showwarning("Aviso", "Por favor, carregue um arquivo CSV primeiro.")
            return
        
        selected_indices = self.emotion_listbox.curselection()
        if not selected_indices:
            tk.messagebox.showwarning("Aviso", "Por favor, selecione pelo menos uma emoção.")
            return
        
        emotions_to_plot = [self.emotion_listbox.get(i) for i in selected_indices]
        df_plot = self.df[self.df['class'].isin(emotions_to_plot)]
        
        # Limpar frame do gráfico
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Criar figura 3D
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        self.current_figure = fig
        
        # Configurar ângulo de visualização
        self.elevation = self.elev_slider.get()
        self.azimuth = self.azim_slider.get()
        ax.view_init(elev=self.elevation, azim=self.azimuth)
        
        # Configurações de plotagem
        base_size = self.size_slider.get()
        alpha = self.alpha_slider.get()
        
        # Normalização para cores baseadas na valência
        norm = plt.Normalize(-1, 1)
        color_map = cm.coolwarm
        
        # Limpar lista de rótulos
        self.label_artists = []
        
        # Plotar esferas para cada emoção
        for _, row in df_plot.iterrows():
            # Posição central
            x, y, z = row['valence mean'], row['arousal mean'], row['dominance mean']
            
            # Tamanhos baseados no desvio padrão
            dx, dy, dz = row['valence std'], row['arousal std'], row['dominance std']
            radius = base_size * np.mean([dx, dy, dz])
            
            # Cor baseada na valência
            color = color_map(norm(x))
            
            # Criar esfera
            u = np.linspace(0, 2 * np.pi, self.sphere_quality)
            v = np.linspace(0, np.pi, self.sphere_quality)
            x_sphere = x + radius * np.outer(np.cos(u), np.sin(v)).flatten()
            y_sphere = y + radius * np.outer(np.sin(u), np.sin(v)).flatten()
            z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
            
            # Plotar esfera
            sphere = ax.plot_surface(
                x_sphere.reshape((self.sphere_quality, self.sphere_quality)),
                y_sphere.reshape((self.sphere_quality, self.sphere_quality)),
                z_sphere.reshape((self.sphere_quality, self.sphere_quality)),
                color=color,
                alpha=alpha,
                shade=True
            )
            self.sphere_artists.append(sphere)
            
            # Rótulo (só adiciona se show_labels for True)
            label = ax.text(
                x, y, z, row['class'], 
                fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, pad=1),
                visible=self.show_labels
            )
            self.label_artists.append(label)
        
        # Configurações dos eixos
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        
        ax.set_xlabel('Valence (Negativo ↔ Positivo)')
        ax.set_ylabel('Arousal (Calmo ↔ Ativo)')
        ax.set_zlabel('Dominance (Submisso ↔ Dominante)')
        ax.set_title('Distribuições Emocionais como Esferas 3D')
        
        # Adicionar barra de cores
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Valence (Vermelho: Negativo, Azul: Positivo)')
        
        # Adicionar canvas ao frame
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Adicionar toolbar de navegação
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def save_visualization(self):
        if self.current_figure is None:
            tk.messagebox.showwarning("Aviso", "Nenhuma visualização para salvar. Plote algumas esferas primeiro.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Salvar visualização 3D"
        )
        
        if filepath:
            try:
                self.current_figure.savefig(filepath, dpi=300, bbox_inches='tight')
                tk.messagebox.showinfo("Sucesso", f"Visualização salva com sucesso em:\n{filepath}")
            except Exception as e:
                tk.messagebox.showerror("Erro", f"Não foi possível salvar o arquivo:\n{str(e)}")

# Executar a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionSphereVisualizer(root)
    root.mainloop()