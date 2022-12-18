import argparse, os, sys, pandas as pd, numpy as np, shutil, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath

def organizeFiles(excelContent):
    returndata = {}
    currImage = -1
    for e in  excelContent:
        try:
            if e[0] == 'Image :':
                currImage =  e[1]
                returndata[currImage] = []
            elif math.isnan(e[0]):
                returndata[currImage].append(''.join([i for i in e[1] if not i.isdigit() or i != '.']))
        except:
            continue

    return returndata
    
def main():
    parser = argparse.ArgumentParser(description='Fix anotation')
    parser.add_argument('--excelFiles', help='Path for the terms file', required=True)
    parser.add_argument('--affectNetFolder', help='Path for the terms file', required=True)
    args = parser.parse_args()

    xlsx = np.array(pd.read_excel(args.excelFiles))
    relabeled = organizeFiles(xlsx)
    folders = getDirectoriesInPath(args.affectNetFolder)
    for f in folders:
        if os.path.exists(os.path.join(args.affectNetFolder,f,'relabel')):
            shutil.rmtree(os.path.join(args.affectNetFolder,f,'relabel'))

        os.makedirs(os.path.join(args.affectNetFolder,f,'relabel'))

        images = getFilesInPath(os.path.join(args.affectNetFolder,f,'images'))
        for i in images:
            pathFile = int(i.split(os.path.sep)[-1][:-4])
            if pathFile not in relabeled.keys():
                print("Missing file annotation: %d" % (pathFile))
            else:
                with open(os.path.join(args.affectNetFolder,f,'relabel',"%d_relabel.txt" % (pathFile)),'w') as fileRelabel:
                    for r in relabeled[pathFile]:
                        fileRelabel.write(r + '\n')


if __name__ == '__main__':
    main()