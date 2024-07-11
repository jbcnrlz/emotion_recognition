import argparse, os, sys, pandas as pd, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def main():
    parser = argparse.ArgumentParser(description='Generate CSV')
    parser.add_argument('--logitsFile', help='Path for the terms file', required=True)
    args = parser.parse_args()
    csvFile = pd.read_csv(args.logitsFile)
    dataLogits = np.array(csvFile)
    logitsFloat = dataLogits[:,[0,1,2,3,4,5,6]].astype(np.float64)
    classes = dataLogits[:,[-1]].astype(np.uint8).flatten()
    total = 0
    considerd = 0
    for idx, l in enumerate(logitsFloat):
        if classes[idx] < len(l):
            total += int(l.argmax() == classes[idx])
            considerd += 1

    print(total / considerd)
    

if __name__ == '__main__':
    main()