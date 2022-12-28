import os, sys, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, getDirectoriesInPath

def main():
    parser = argparse.ArgumentParser(description='Evaluate folders')
    parser.add_argument('--pathBase', help='Path for cluster', required=True)
    parser.add_argument('--pathAffWild', help='Path for cluster', required=True)
    args = parser.parse_args()

    images = getFilesInPath(args.pathAffWild)
    clusters = getDirectoriesInPath(args.pathBase)
    
