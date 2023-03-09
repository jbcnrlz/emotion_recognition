import os, sys, argparse, pandas as pd, numpy as np, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from scipy.spatial import KDTree

def outputsNeighFiles(outputPath,data):
    with open(outputPath,'w') as op:
        for d in data:
            op.write(d)

def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--datasetFolder', help='Path for the terms file', required=True)
    parser.add_argument('--lastClassNumber', help='Path for the terms file', required=True, type=int)
    parser.add_argument('--k', help='Path for the terms file', required=True, type=int)
    parser.add_argument('--outputFile', help='Path for the terms file', required=True)
    args = parser.parse_args()

    imagesFolder = os.path.join(args.datasetFolder,'images')
    annotationFolder = os.path.join(args.datasetFolder,'annotations')
    vaValues = []
    nClass = []    
    images = []
    for i in range(args.lastClassNumber+1):
        if os.path.exists(os.path.join(imagesFolder,'%d.jpg' % (i))):
            images.append(os.path.join(imagesFolder,'%d.jpg' % (i)))
            print("Fazendo imagem %s" % (images[-1]))
            vaValues.append((float(np.load(os.path.join(annotationFolder,'%d_val.npy' % (i)))),float(np.load(os.path.join(annotationFolder,'%d_aro.npy' % (i))))))
            nClass.append(np.load(os.path.join(annotationFolder,'%d_exp.npy' % (i))))

    vaValues = np.array(vaValues)
    nnTree = KDTree(vaValues)
    outputMaterial = []
    for idxVa, v in enumerate(vaValues):
        print("Pegando vizinhos da imagem %d de %d" % (idxVa,len(vaValues)))
        _, neighs = nnTree.query(v,args.k+1)
        outputMaterial.append('%s,%f,%f,%d,%d\n' % (images[idxVa],v[0],v[1],nClass[idxVa],args.k))
        for idxN in neighs[1:]:
            outputMaterial.append("%f,%f,%d\n" % (vaValues[idxN][0],vaValues[idxN][1],nClass[idxN]))
    
    outputsNeighFiles(args.outputFile,outputMaterial)

if __name__ == '__main__':
    main()