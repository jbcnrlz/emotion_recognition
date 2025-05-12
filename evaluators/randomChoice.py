import argparse, os, sys, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, printProgressBar
import random

def main():
    parser = argparse.ArgumentParser(description='Sample random images')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--pathOutput', help='Size of the batch', required=True)    
    parser.add_argument('--numberChoices', help='Size of the batch', required=True, type=int)
    parser.add_argument('--numberFolds', help='Size of the batch', required=True, type=int)    
    args = parser.parse_args()

    if os.path.exists(args.pathOutput):
        shutil.rmtree(args.pathOutput)
    
    os.makedirs(args.pathOutput)
        

    images = getFilesInPath(args.pathBase,imagesOnly=True)
    alreadyUsed = []
    for i in range(args.numberFolds):
        os.makedirs(os.path.join(args.pathOutput, str(i)))
        for j in range(args.numberChoices):
            while True:
                choice = random.choice(images)
                if choice not in alreadyUsed:
                    alreadyUsed.append(choice)
                    printProgressBar(j+1, args.numberChoices, prefix = 'Progress', suffix = 'Complete', length = 50)
                    break                
            shutil.copy(choice, os.path.join(args.pathOutput, str(i), os.path.basename(choice)))

    

if __name__ == "__main__":
    main()