import argparse, os, sys, shutil, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar

def openAffWildFeaturesFile(pathFile):
    returnData = []
    with open(pathFile,'r') as oFile:
        for p in oFile:
            try:
                returnData.append(int(p))
            except:
                continue

    return returnData

def main():
    parser = argparse.ArgumentParser(description='Generate NewFusedDataset')
    parser.add_argument('--pathBaseAffectNet', help='Path for datasets', required=True)
    parser.add_argument('--pathBaseAffWild', help='Path for datasets', required=True)
    parser.add_argument('--newDatasetPath', help='Path for datasets', required=True)
    args = parser.parse_args()
    if os.path.exists(args.newDatasetPath):
        shutil.rmtree(args.newDatasetPath)
    
    annotationFile = os.path.join(args.newDatasetPath,'train_set','annotations')
    imagesFolder = os.path.join(args.newDatasetPath,'train_set','images')

    os.makedirs(annotationFile)
    os.makedirs(imagesFolder)

    AffectNetfaces = getFilesInPath(os.path.join(args.pathBaseAffectNet,'images'),imagesOnly=True)
    classes = [0,0]
    lastClass = 0
    for a in AffectNetfaces:
        imageNumber = int(a.split(os.path.sep)[-1][:-4])
        if imageNumber > lastClass:
            lastClass = imageNumber
        currLabel = int(np.load(os.path.join(args.pathBaseAffectNet,'annotations' ,'%d_exp.npy' % (imageNumber))))
        fileName = a.split(os.path.sep)[-1]
        shutil.copy(a,os.path.join(imagesFolder,fileName))
        shutil.copy(os.path.join(args.pathBaseAffectNet,'annotations' ,'%d_exp.npy' % (imageNumber)),os.path.join(annotationFile,'%d_exp.npy' % (imageNumber)))
        classes[currLabel == 0] += 1

    files = getFilesInPath(os.path.join(args.pathBaseAffWild,'annotations','EXPR_Set','Train_Set'))
    imagePath = os.path.join(args.pathBaseAffWild,'cropped_aligned')
    for f in files:
        fileName = f.split(os.path.sep)[-1]            
        if fileName[-3:] != 'txt':
            continue

        featuresFromVideo = openAffWildFeaturesFile(f)
        for idxFrame, currFeat in enumerate(featuresFromVideo):
            if currFeat == 0:
                videoName = os.path.join(imagePath,fileName.split('.')[0])
                if '_terms' in videoName:
                    videoName = videoName[:-len('_terms')]

                lastClass += 1
                fileSource = os.path.join(videoName,"%05d.jpg" % (idxFrame))
                fileDest = os.path.join(imagesFolder,"%d.jpg" % (lastClass))
                try:
                    shutil.copy(fileSource,fileDest)
                    np.save(os.path.join(annotationFile,'%d_exp.npy' % (lastClass)),np.array([0]))
                    classes[1] += 1
                except:
                    lastClass -= 1
                    continue
                if classes[0] == classes[1]:
                    break
                


if __name__ == '__main__':
    main()