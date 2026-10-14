import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.models import Emotion_GCN

def main():
    model = Emotion_GCN(adj_file='adj.pkl', input_size=227)

if __name__ == '__main__':
    main()