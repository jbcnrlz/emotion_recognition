import pandas as pd, argparse, numpy as np, os

def main():
    parser = argparse.ArgumentParser(description='Measure DAN Accuracy')
    parser.add_argument('--dataWithPredictions', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    tFiles = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
    classesLabel = np.array(tFiles)

    datasetPreds = np.array(pd.read_csv(args.dataWithPredictions))
    labels = datasetPreds[:,0]
    pathsFiles = datasetPreds[:,1]
    acertou = 0
    total = 0
    for idx, p in enumerate(pathsFiles):
        currPath = p.split(os.path.sep)
        fileName = int(currPath[-1][:-4])
        currLabel = np.load(os.path.join(currPath[0],'annotations' ,'%d_exp.npy' % (fileName)))
        if int(currLabel == 7):
            continue
        acertou += int(labels[idx] == classesLabel[int(currLabel)])
        total += 1
        print(p)
    print('%f' % (acertou / total))


if __name__ == '__main__':
    main()