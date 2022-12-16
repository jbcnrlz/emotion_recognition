import argparse, os, sys, pandas as pd, numpy as np, shutil, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Fix anotation')
    parser.add_argument('--affectNetFolder', help='Path for the terms file', required=True)
    args = parser.parse_args()

    folders = getDirectoriesInPath(args.affectNetFolder)
    for f in folders:
        if os.path.exists(os.path.join(args.affectNetFolder,f,'relabel_joao')):
            shutil.rmtree(os.path.join(args.affectNetFolder,f,'relabel_joao'))

        os.makedirs(os.path.join(args.affectNetFolder,f,'relabel_joao'))

        images = getFilesInPath(os.path.join(args.affectNetFolder,f,'images'))
        for i in images:
            imgName = int(i[:-4])
            val = np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_val.npy' % (imgName)))
            aro = np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_aro.npy' % (imgName)))
            exp = np.load(os.path.join(args.affectNetFolder,f,'annotations','%d_aro.npy' % (imgName)))


if __name__ == '__main__':
    main()