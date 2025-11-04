import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.stats import multivariate_normal

# Data from file
data = {
    'class': ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad'],
    'valence mean': [0.81, -0.23, 0.4, -0.51, -0.6, -0.64, -0.63],
    'valence std': [0.21, 0.39, 0.3, 0.2, 0.2, 0.2, 0.23],
    'arousal mean': [0.51, 0.31, 0.67, 0.59, 0.35, 0.6, -0.27],
    'arousal std': [0.26, 0.33, 0.27, 0.33, 0.41, 0.32, 0.34],
    'dominance mean': [0.46, 0.18, -0.13, 0.25, 0.11, -0.21, -0.33],
    'dominance std': [0.38, 0.29, 0.38, 0.39, 0.34, 0.27, 0.22]
}

df = pd.DataFrame(data)

# Colors for each emotion
colors = ['yellow', 'orange', 'cyan', 'red', 'brown', 'purple', 'blue']

# Create figure with subplots
fig = plt.figure(figsize=(18, 8))

# Subplot 1: 2D Projection (Valence vs Arousal)
ax1 = fig.add_subplot(121)

# Plot each emotion with standard deviation ellipses
for i, (idx, row) in enumerate(df.iterrows()):
    # Central point
    ax1.scatter(row['valence mean'], row['arousal mean'], 
               color=colors[i], s=100, label=row['class'], alpha=0.8, edgecolors='black')
    
    # Standard deviation ellipse
    ellipse = patches.Ellipse((row['valence mean'], row['arousal mean']),
                             width=row['valence std']*1.5, height=row['arousal std']*1.5,
                             alpha=0.3, color=colors[i])
    ax1.add_patch(ellipse)
    
    # Annotation
    ax1.annotate(row['class'], (row['valence mean'], row['arousal mean']),
                xytext=(8, 8), textcoords='offset points', fontsize=10, 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

ax1.set_xlabel('Valence', fontsize=12, fontweight='bold')
ax1.set_ylabel('Arousal', fontsize=12, fontweight='bold')
ax1.set_title('Emotional Distribution - Valence vs Arousal (2D)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Subplot 2: 3D Distribution with Intersection Visualization
ax2 = fig.add_subplot(122, projection='3d')

# Create a 3D grid for probability density visualization
x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)
z = np.linspace(-1, 1, 30)
X, Y, Z = np.meshgrid(x, y, z)

# Plot each emotion in 3D with transparent spheres showing distribution overlap
for i, (idx, row) in enumerate(df.iterrows()):
    # Central point
    ax2.scatter(row['valence mean'], row['arousal mean'], row['dominance mean'],
               color=colors[i], s=150, label=row['class'], alpha=0.9, edgecolors='black')
    
    # Create transparent sphere representing distribution
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    # Use average standard deviation for sphere radius
    avg_std = np.mean([row['valence std'], row['arousal std'], row['dominance std']])
    radius = avg_std * 1.2
    
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + row['valence mean']
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + row['arousal mean']
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + row['dominance mean']
    
    # Plot transparent sphere
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, 
                    color=colors[i], alpha=0.15, linewidth=0, antialiased=True)

ax2.set_xlabel('Valence', fontsize=12, fontweight='bold')
ax2.set_ylabel('Arousal', fontsize=12, fontweight='bold')
ax2.set_zlabel('Dominance', fontsize=12, fontweight='bold')
ax2.set_title('3D Emotional Distribution with Overlap Visualization', fontsize=14, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

# Add reference lines
ax2.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
ax2.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
ax2.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.show()

# Create a detailed intersection analysis
fig2 = plt.figure(figsize=(16, 6))

# Subplot 1: 2D with intersection regions highlighted
ax3 = fig2.add_subplot(121)

# Plot points and ellipses
for i, (idx, row) in enumerate(df.iterrows()):
    ax3.scatter(row['valence mean'], row['arousal mean'], 
               color=colors[i], s=100, label=row['class'], alpha=0.8)
    
    ellipse = patches.Ellipse((row['valence mean'], row['arousal mean']),
                             width=row['valence std']*1.8, height=row['arousal std']*1.8,
                             alpha=0.2, color=colors[i])
    ax3.add_patch(ellipse)

# Highlight intersection regions by drawing lines between close distributions
for i in range(len(df)):
    for j in range(i+1, len(df)):
        dist = np.sqrt((df.iloc[i]['valence mean'] - df.iloc[j]['valence mean'])**2 + 
                      (df.iloc[i]['arousal mean'] - df.iloc[j]['arousal mean'])**2)
        
        # If distributions are close, draw a connection line
        if dist < 0.8:  # Threshold for considering overlap
            ax3.plot([df.iloc[i]['valence mean'], df.iloc[j]['valence mean']],
                    [df.iloc[i]['arousal mean'], df.iloc[j]['arousal mean']],
                    'k--', alpha=0.5, linewidth=1)
            
            # Mark intersection point
            mid_x = (df.iloc[i]['valence mean'] + df.iloc[j]['valence mean']) / 2
            mid_y = (df.iloc[i]['arousal mean'] + df.iloc[j]['arousal mean']) / 2
            ax3.scatter(mid_x, mid_y, color='black', s=30, alpha=0.7, marker='x')

ax3.set_xlabel('Valence', fontsize=12, fontweight='bold')
ax3.set_ylabel('Arousal', fontsize=12, fontweight='bold')
ax3.set_title('2D Distribution with Intersection Markers', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax3.legend()

# Subplot 2: 3D with connection lines showing potential overlaps
ax4 = fig2.add_subplot(122, projection='3d')

# Plot points
for i, (idx, row) in enumerate(df.iterrows()):
    ax4.scatter(row['valence mean'], row['arousal mean'], row['dominance mean'],
               color=colors[i], s=120, label=row['class'], alpha=0.8)

# Draw connection lines between emotionally similar distributions
emotional_groups = [
    ['angry', 'disgusted', 'fearful'],  # Negative high arousal
    ['sad', 'contempt'],                # Negative low arousal  
    ['happy', 'surprised']              # Positive arousal
]

for group in emotional_groups:
    group_points = []
    for emotion in group:
        idx = df[df['class'] == emotion].index[0]
        point = [df.iloc[idx]['valence mean'], 
                df.iloc[idx]['arousal mean'], 
                df.iloc[idx]['dominance mean']]
        group_points.append(point)
    
    # Connect points within the same emotional group
    for i in range(len(group_points)):
        for j in range(i+1, len(group_points)):
            ax4.plot([group_points[i][0], group_points[j][0]],
                    [group_points[i][1], group_points[j][1]],
                    [group_points[i][2], group_points[j][2]],
                    color='gray', alpha=0.6, linewidth=2, linestyle=':')

ax4.set_xlabel('Valence', fontsize=12, fontweight='bold')
ax4.set_ylabel('Arousal', fontsize=12, fontweight='bold')
ax4.set_zlabel('Dominance', fontsize=12, fontweight='bold')
ax4.set_title('3D Emotional Clusters and Connections', fontsize=14, fontweight='bold')
ax4.set_xlim(-1, 1)
ax4.set_ylim(-1, 1)
ax4.set_zlim(-1, 1)
ax4.legend()

plt.tight_layout()
plt.show()

# Print intersection analysis
print("="*60)
print("EMOTIONAL DISTRIBUTION INTERSECTION ANALYSIS")
print("="*60)

for i in range(len(df)):
    for j in range(i+1, len(df)):
        # Calculate Euclidean distance between emotion centers
        dist = np.sqrt(
            (df.iloc[i]['valence mean'] - df.iloc[j]['valence mean'])**2 +
            (df.iloc[i]['arousal mean'] - df.iloc[j]['arousal mean'])**2 +
            (df.iloc[i]['dominance mean'] - df.iloc[j]['dominance mean'])**2
        )
        
        # Calculate combined standard deviation (simplified overlap measure)
        combined_std = np.mean([
            df.iloc[i]['valence std'] + df.iloc[j]['valence std'],
            df.iloc[i]['arousal std'] + df.iloc[j]['arousal std'],
            df.iloc[i]['dominance std'] + df.iloc[j]['dominance std']
        ])
        
        overlap_ratio = combined_std / dist if dist > 0 else 1.0
        
        if overlap_ratio > 0.8:  # High overlap threshold
            print(f"🚨 HIGH OVERLAP: {df.iloc[i]['class']} ↔ {df.iloc[j]['class']}")
            print(f"   Distance: {dist:.3f}, Overlap ratio: {overlap_ratio:.3f}")
        elif overlap_ratio > 0.5:  # Medium overlap
            print(f"⚠️  MEDIUM OVERLAP: {df.iloc[i]['class']} ↔ {df.iloc[j]['class']}")
            print(f"   Distance: {dist:.3f}, Overlap ratio: {overlap_ratio:.3f}")
        elif overlap_ratio > 0.3:  # Low overlap
            print(f"🔍 LOW OVERLAP: {df.iloc[i]['class']} ↔ {df.iloc[j]['class']}")
            print(f"   Distance: {dist:.3f}, Overlap ratio: {overlap_ratio:.3f}")