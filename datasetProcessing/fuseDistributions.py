import numpy as np, argparse, pandas as pd
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

def calculate_intersection(args):
    """
    Calculate intersection volume between two 3D Gaussian distributions
    with proper error handling and numpy operations
    """
    i, j, dist1, dist2 = args
    
    try:
        # Define integration bounds (3D)
        bounds = [(-1, 1), (-1, 1), (-1, 1)]
        
        # Create grid for numerical integration
        grid_resolution = 30  # Reduced for better performance
        x = np.linspace(bounds[0][0], bounds[0][1], grid_resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], grid_resolution)
        z = np.linspace(bounds[2][0], bounds[2][1], grid_resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Vectorized PDF calculation with clipping for stability
        pdf1 = np.clip(dist1.pdf(points), 1e-10, 1e10)
        pdf2 = np.clip(dist2.pdf(points), 1e-10, 1e10)
        min_pdf = np.minimum(pdf1, pdf2)
        
        # Numerical integration using numpy
        dx = (bounds[0][1] - bounds[0][0]) / grid_resolution
        dy = (bounds[1][1] - bounds[1][0]) / grid_resolution
        dz = (bounds[2][1] - bounds[2][0]) / grid_resolution
        intersection_volume = np.sum(min_pdf) * dx * dy * dz
        
        # Calculate normalization factors safely
        vol1 = np.sum(pdf1) * dx * dy * dz
        vol2 = np.sum(pdf2) * dx * dy * dz
        
        # Avoid division by zero
        normalization = min(vol1, vol2)
        if normalization <= 0:
            return i, j, 0.0
            
        return i, j, intersection_volume / normalization
    
    except Exception as e:
        print(f"Error calculating intersection between {i} and {j}: {str(e)}")
        return i, j, 0.0

def generateDistributions(mean, covMatrix, points=1000):
    """Generate 3D Gaussian distributions"""
    return np.random.multivariate_normal(mean=mean, cov=covMatrix, size=points)

def fuse_distributions(emotions_data, max_workers=None):
    """
    Optimized fusion algorithm with proper numpy numeric operations
    """
    # Prepare 3D points for KDTree
    points_3d = emotions_data[['valence mean', 'arousal mean', 'dominance mean']].values
    
    maxValence = emotions_data['valence std'].max()
    arousalMax = emotions_data['arousal std'].max()
    dominanceMax = emotions_data['dominance std'].max()

    # Build KDTree for fast nearest neighbor queries
    kdtree = KDTree(points_3d)
    
    # Create multivariate normal distributions with validation
    distributions = []
    for idx, row in emotions_data.iterrows():
        try:
            mean = [row['valence mean'], row['arousal mean'], row['dominance mean']]
            cov = np.diag([
                max(row['valence std'], 1e-6)**2,  # Ensure positive
                max(row['arousal std'], 1e-6)**2,
                max(row['dominance std'], 1e-6)**2
            ])
            distributions.append(multivariate_normal(mean=mean, cov=cov))
        except Exception as e:
            print(f"Error creating distribution for row {idx}: {str(e)}")
            distributions.append(None)
    
    # Find 5 nearest neighbors for each emotion
    n_neighbors = 6  # 5 neighbors + itself
    _, indices = kdtree.query(points_3d, k=n_neighbors)
    
    # Prepare tasks for parallel processing
    tasks = []
    for i in range(len(distributions)):
        if distributions[i] is None:
            continue
            
        for j in indices[i][1:]:  # Skip self
            if j >= len(distributions) or distributions[j] is None:
                continue
            tasks.append((i, j, distributions[i], distributions[j]))
    
    # Parallel computation of intersections
    intersection_matrix = np.zeros((len(distributions), len(distributions)))
    max_workers = max_workers or min(cpu_count(), 32)
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(calculate_intersection, tasks), 
                              total=len(tasks), 
                              desc="Calculating intersections"))
        
        for i, j, value in results:
            if i < len(intersection_matrix) and j < len(intersection_matrix):
                intersection_matrix[i, j] = value
                intersection_matrix[j, i] = value  # Symmetric matrix
    except Exception as e:
        print(f"Error during parallel processing: {str(e)}")
    
    # Fusion process with numpy math operations
    fused_emotions = []
    used_indices = set()
    
    for i in range(len(distributions)):
        if i in used_indices or distributions[i] is None:
            continue
        
        name1 = emotions_data.iloc[i]['class']
        if name1 in ['happy', 'sad','surprised','contempt','fearful','angry','disgusted']:
            fused_emotions.append({
                'class': emotions_data.iloc[i]['class'],
                **emotions_data.iloc[i][['valence mean', 'valence std', 
                                       'arousal mean', 'arousal std',
                                       'dominance mean', 'dominance std']].astype(float).to_dict()
            })
            used_indices.add(i)
            continue
        # Find most overlapping neighbor among top 5
        neighbors = indices[i][1:]  # Exclude self
        max_overlap = 0
        best_j = -1
        
        for j in neighbors:
            if (j < len(intersection_matrix) and 
                intersection_matrix[i, j] > max_overlap and 
                j not in used_indices):
                max_overlap = intersection_matrix[i, j]
                best_j = j
        
        if max_overlap > 0.5 and best_j != -1:
            try:
                
                name2 = emotions_data.iloc[best_j]['class']
                if name2 in ['happy', 'sad','surprised','contempt','fearful','angry','disgusted']:
                    fused_emotions.append({
                        'class': emotions_data.iloc[best_j]['class'],
                        **emotions_data.iloc[best_j][['valence mean', 'valence std', 
                                            'arousal mean', 'arousal std',
                                            'dominance mean', 'dominance std']].astype(float).to_dict()
                    })
                    used_indices.add(best_j)
                    continue

                # Merge distributions using numpy properly
                mean1 = points_3d[i]
                cov1 = np.diag([
                    max(emotions_data.iloc[i]['valence std'], 1e-6)**2,  # Ensure positive
                    max(emotions_data.iloc[i]['arousal std'], 1e-6)**2,
                    max(emotions_data.iloc[i]['dominance std'], 1e-6)**2
                ])
                mean2 = points_3d[best_j]
                cov2 = np.diag([
                    max(emotions_data.iloc[best_j]['valence std'], 1e-6)**2,
                    max(emotions_data.iloc[best_j]['arousal std'], 1e-6)**2,
                    max(emotions_data.iloc[best_j]['dominance std'], 1e-6)**2
                ])

                pts1 = generateDistributions(mean1, cov1, points=1000)
                pts2 = generateDistributions(mean2, cov2, points=1000)
                pts = np.concatenate((pts1, pts2), axis=0)
                meanND = np.mean(pts, axis=0)
                stdND = np.std(pts, axis=0)
                
                # Check all three std deviations
                if (stdND[0] > maxValence or stdND[1] > arousalMax or stdND[2] > dominanceMax):
                    print(f"Skipping {i} and {best_j} due to high std deviation.")
                    continue

                corrMatrixND = np.corrcoef(pts, rowvar=False)

                # Create combined name
                new_name = f"{name1} + {name2}"
                
                fused_emotions.append({
                    'class': new_name,
                    'valence mean': float(meanND[0]),
                    'valence std': float(stdND[0]),
                    'arousal mean': float(meanND[1]),
                    'arousal std': float(stdND[1]),
                    'dominance mean': float(meanND[2]),
                    'dominance std': float(stdND[2])
                })

                used_indices.update([i, best_j])
            except Exception as e:
                print(f"Error fusing {i} and {best_j}: {str(e)}")
                fused_emotions.append({
                    'class': emotions_data.iloc[i]['class'],
                    **emotions_data.iloc[i][['valence mean', 'valence std', 
                                           'arousal mean', 'arousal std',
                                           'dominance mean', 'dominance std']].astype(float).to_dict()
                })
                used_indices.add(i)
        else:
            # Keep original distribution
            fused_emotions.append({
                'class': emotions_data.iloc[i]['class'],
                **emotions_data.iloc[i][['valence mean', 'valence std', 
                                       'arousal mean', 'arousal std',
                                       'dominance mean', 'dominance std']].astype(float).to_dict()
            })
            used_indices.add(i)
    
    return pd.DataFrame(fused_emotions)

def main():
    try:
        parser = argparse.ArgumentParser(description='Fuse distributions')
        parser.add_argument('--distFile', help='Path to emotion distribution file', required=True)    
        args = parser.parse_args()
        # Load data with proper numeric columns
        emotions_data = pd.read_csv(args.distFile)
        
        # Ensure numeric columns
        for col in ['valence mean', 'valence std', 'arousal mean', 'arousal std', 'dominance mean', 'dominance std']:
            emotions_data[col] = pd.to_numeric(emotions_data[col], errors='coerce')
        
        # Remove rows with missing values
        emotions_data = emotions_data.dropna(subset=['valence mean', 'arousal mean', 'dominance mean'])
        
        # Run fusion algorithm
        idx = 1
        before = 0
        while len(emotions_data) > 13 and before != len(emotions_data):
            before = len(emotions_data)
            print(f"Fusing {len(emotions_data)} distributions...")
            emotions_data = fuse_distributions(emotions_data, max_workers=8)
            emotions_data.to_csv(f"fused_emotions_3d_{idx}.csv", index=False)
            idx += 1            
        #fused_df = fuse_distributions(emotions_data)
        
        # Save results
        emotions_data.to_csv("fused_emotions_3d.csv", index=False)
        print("Fusion complete. Results saved to fused_emotions_3d.csv")
        
        return 0
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())