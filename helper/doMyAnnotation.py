import argparse, os, sys, pandas as pd, numpy as np, shutil, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, separatedSecondLevel

def saveNewAnnotation(pathFile,ann):
    with open(pathFile,'w') as pf:
        pf.write(ann)

def main():
    parser = argparse.ArgumentParser(description='Fix anotation')
    parser.add_argument('--affectNetFolder', help='Path for the terms file', required=True)
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--csvMapping', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    emotions = np.array(pd.read_csv(args.csvEmotions))
    mapping = np.array(pd.read_csv(args.csvMapping))

    folders = getDirectoriesInPath(args.affectNetFolder)
    for f in folders:
        if os.path.exists(os.path.join(args.affectNetFolder,f,'relabel_joao')):
            shutil.rmtree(os.path.join(args.affectNetFolder,f,'relabel_joao'))

        os.makedirs(os.path.join(args.affectNetFolder,f,'relabel_joao'))

        images = getFilesInPath(os.path.join(args.affectNetFolder,f,'images'))
        for i in images:
            imgName = int(i.split(os.path.sep)[-1][:-4])
            val = float(np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_val.npy' % (imgName))))
            aro = float(np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_aro.npy' % (imgName))))
            valar = np.array([val,aro])
            exp = int(np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_exp.npy' % (imgName))))
            if (emotions[int(exp)][0] == 'neutral'):
                saveNewAnnotation(os.path.join(args.affectNetFolder,f,'relabel_joao','%d_relabel.txt' % (imgName)),'neutral')
            else:
                if (emotions[int(exp)][0] == 'contempt'):
                    searchSpace = separatedSecondLevel('fear',mapping)
                else:
                    searchSpace = separatedSecondLevel(emotions[int(exp)][0],mapping)
                resultFinal = np.array([ np.linalg.norm(valar-s[[1,3]]) for s in searchSpace]).argmin()
                saveNewAnnotation(os.path.join(args.affectNetFolder,f,'relabel_joao','%d_relabel.txt' % (imgName)),searchSpace[resultFinal][0])
                


if __name__ == '__main__':
    main()