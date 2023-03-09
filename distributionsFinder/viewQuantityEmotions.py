import os, sys, argparse, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--pathAnnotation', help='Path for the terms file', required=True)
    parser.add_argument('--lastClassNumber', help='Path for the terms file', required=True, type=int)
    args = parser.parse_args()

    classes = [0] * 8
    for i in range(0,args.lastClassNumber+1):
        if os.path.exists(os.path.join(args.pathAnnotation,'%d_exp.npy' % (i))):
            cNumber = int(np.load(os.path.join(args.pathAnnotation,'%d_exp.npy' % (i))))
            classes[cNumber] += 1

    print(classes)

if __name__ == '__main__':
    main()