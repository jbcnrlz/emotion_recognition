import cv2, dlib
import matplotlib.pyplot as plt
import argparse, pandas as pd, numpy as np, sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def outputTXTAnnotation(filePath,term):
    with open(filePath,'w') as fp:
        fp.write(term)

def main():
    parser = argparse.ArgumentParser(description='Format Terms to AffectNet')
    parser.add_argument('--pathBase', help='Path for data', required=True)
    parser.add_argument('--annotationPath', help='Path for data', required=True)
    parser.add_argument('--lastClass', help='Path for data', required=True,type=int)
    args = parser.parse_args()
    win = dlib.image_window()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    for cidx in range(args.lastClass+1):
        if os.path.exists(os.path.join(args.pathBase,'%d.jpg' % (cidx))):
            print("Doing file %s" %(os.path.join(args.pathBase,'%d.jpg' % (cidx))))
            img = dlib.load_rgb_image(os.path.join(args.pathBase,'%d.jpg' % (cidx)))

            win.clear_overlay()
            win.set_image(img)

            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
                shape = predictor(img, d)
                print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))        
                win.add_overlay(shape)

            win.add_overlay(dets)

if __name__ == '__main__':
    main()