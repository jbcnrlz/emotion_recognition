import argparse, os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, getFeatureFromText

def main():
    parser = argparse.ArgumentParser(description='Database statistics')
    parser.add_argument('--pathFile', help='Weights', required=True)
    parser.add_argument('--csvMapping', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    mapping = np.array(pd.read_csv(args.csvMapping))
    anots = getFilesInPath(os.path.join(args.pathFile,'relabel_joao'))
    terms = { 'neutral' : 0 }

    for m in mapping:
        if m[-1] == 2:
            terms[m[0]] = 0

    for a in anots:
        feat = getFeatureFromText(a)
        terms[feat] += 1

    fig1, ax1 = plt.subplots()
    ax1.pie([i / sum(terms.values()) *100 for i in terms.values()],labels=terms.keys(), autopct='%1.1f%%',shadow=True,startangle=90)
    ax1.axis('equal')
    plt.show()

    print(terms)

if __name__ == "__main__":
    main()