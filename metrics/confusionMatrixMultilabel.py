import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import multilabel_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description='Generate Confusion Matrix')
    parser.add_argument('--pathCSV', help='Weights', required=True)
    args = parser.parse_args()

    csvFiles = pd.read_csv(args.pathCSV)
    logits = np.array(csvFiles)[:,:-2]
    labels = np.array(csvFiles)[:,-1]
    emotionsLabels = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,1],
    ])
    
    emotionOrder = [ 'neutral', 'happy', 'sad', 'surprise','fear', 'disgust', 'anger', 'contempt']

    preds = []
    gt = []
    for idx, l in enumerate(labels):
        preds.append([])
        gt.append(emotionsLabels[l])
        for idEmotion in (-logits[idx].astype(np.float64)).argsort():
            if (logits[idx][idEmotion] >= 0.1):
                preds[-1].append(emotionOrder[idEmotion])
            else:
                preds[-1].append('none')

    preds = np.array(preds)
    gt = np.array(gt)

    print('oi')



    

if __name__ == '__main__':
    main()