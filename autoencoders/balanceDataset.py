import os, sys, argparse, numpy as np, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--pathAnnotation', help='Path for the terms file', required=True)
    parser.add_argument('--lastClassNumber', help='Path for the terms file', required=True, type=int)
    parser.add_argument('--limit', help='Path for the terms file', required=False, type=int, default=3750)
    args = parser.parse_args()

    annFile = os.path.join(args.pathAnnotation,'annotations')
    images = os.path.join(args.pathAnnotation,'images')
    imagesBalanced = os.path.join(args.pathAnnotation,'images_balanced')

    if os.path.exists(imagesBalanced):
        shutil.rmtree(imagesBalanced)

    os.makedirs(imagesBalanced)

    classes = [0] * 8
    for i in range(0,args.lastClassNumber+1):
        if os.path.exists(os.path.join(images,'%d.jpg' % (i))):
            cNumber = int(np.load(os.path.join(annFile,'%d_exp.npy' % (i))))
            if classes[cNumber] < args.limit:
                classes[cNumber] += 1
                shutil.copy(os.path.join(images,'%d.jpg' % (i)),os.path.join(imagesBalanced,'%d.jpg' % (i)))

        if sum(classes) >= (args.limit * len(classes)):
            break

    print(classes)


if __name__ == '__main__':
    main()