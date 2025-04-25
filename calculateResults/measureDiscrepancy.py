import numpy as np, pandas as pd, os, sys, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    classesDist = np.array([
        [0,0,0,0],
        [0.81,0.21,0.51,0.26], #happy
        [-0.63,0.23,-0.27,0.34], #sad
        [0.4,0.3,0.67,0.27], #surprised
        [-0.64,0.2,0.6,0.32],#fear
        [-0.6,0.2,0.35,0.41],#disgust
        [-0.51,0.2,0.59,0.33],#angry
        [-0.23,0.39,0.31,0.33],#contempt
    ])

    results = [0,0]
    images = getFilesInPath(os.path.join(args.pathBase,'images'))
    for i in images:
        fileName = i.split(os.path.sep)[-1][:-4]
        exp = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_exp.npy"))
        val = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_val.npy")).astype(np.float64)
        aro = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_aro.npy")).astype(np.float64)

        current_values = np.array([val, aro])

        distances = []
        for emotion in classesDist:
            # Pegar valores de val (posição 0) e aro (posição 2)
            emotion_values = np.array([emotion[0], emotion[2]])
            distance = np.linalg.norm(current_values - emotion_values)
            distances.append(distance)

        distances = np.array(distances)
        #print(f"Distances: {(-distances).argsort()[0]} Exp {int(exp)}")
        results[ int(distances.argsort()[0] == int(exp)) ] += 1

    print(results)

if __name__ == "__main__":
    main()