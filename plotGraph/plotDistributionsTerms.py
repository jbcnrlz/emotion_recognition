import matplotlib.pyplot as plt, argparse, pandas as pd, numpy as np, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import create_ellipse
from matplotlib import patches

def main():
    parser = argparse.ArgumentParser(description='Plot Terms Files')
    parser.add_argument('--termFiles', help='CSV file with terms', required=True, nargs='+')
    args = parser.parse_args()
    colors = ['red', 'blue', 'green', 'yellow']
    for idx, ft in enumerate(args.termFiles):
        csvTerms = pd.read_csv(ft)
        plt.scatter(csvTerms['valence mean'],csvTerms['arousal mean'],c=colors[idx])
        for c in range(len(csvTerms)):
            plt.annotate(csvTerms['class'][c],(csvTerms['valence mean'][c],csvTerms['arousal mean'][c]))
    plt.show()
    '''
    elipses = list(zip(csvTerms['valence mean'],csvTerms['arousal mean'],csvTerms['valence std'],csvTerms['arousal std'],[0] * len(csvTerms)))
    elipses = [create_ellipse(e[:2],e[2:-1]) for e in elipses]

    fig,ax = plt.subplots()

    ##these next few lines are pretty important because
    ##otherwise your ellipses might only be displayed partly
    ##or may be distorted
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_aspect('equal')


    for e in elipses:
        if e is None:
            continue
        ##second ellipse in red    
        verts2 = np.array(e.exterior.coords.xy)
        patch2 = patches.Polygon(verts2.T,color = 'red', alpha = 0.5)
        ax.add_patch(patch2)
    '''
    plt.show()
    
if __name__ == '__main__':
    main()

