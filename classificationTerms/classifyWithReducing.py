import argparse, torch, os, sys, numpy as np, pandas as pd

def separatedSecondLevel(firstLevel,emotions):
    found = 0
    output = []
    for e in emotions:
        if firstLevel == e[0]:
            found = 1
        else:
            if found == 1 and e[1] != 0:
                output.append(e)
            elif found == 1:
                break

    return np.array(output)            

def main():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--csvVA', help='Weights', required=True)
    parser.add_argument('--csvMapping', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)    
    args = parser.parse_args()
    emotions = np.array(pd.read_csv(args.csvEmotions))
    va = np.array(pd.read_csv(args.csvVA))
    mapping = np.array(pd.read_csv(args.csvMapping))
    output = []
    for e in emotions:
        if e[0] == 'neutral':
            output.append(['neutral',e[-1]])
            continue

        for v in va:
            if e[-1] == v[-1]:
                searchSpace = separatedSecondLevel(e[0],mapping)
                resultFinal = np.array([ np.linalg.norm(v[:-1]-s[[1,3]]) for s in searchSpace]).argmin()
                output.append([searchSpace[resultFinal][0],e[-1]])
                print('oi')

    with open(args.output,'w') as pcsv:
        pcsv.write('emotion,file\n')
        for o in output:
            pcsv.write('%s,%s\n' % (o[0],o[1]))


if __name__ == '__main__':
    main()