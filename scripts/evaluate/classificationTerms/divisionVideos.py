import argparse, pandas as pd, numpy as np, os, sys, itertools

def sign (p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def pointInTriangle (pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = ((d1 < 0) or (d2 < 0) or (d3 < 0))
    has_pos = ((d1 > 0) or (d2 > 0) or (d3 > 0))

    return not (has_neg and has_pos)

def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    args = parser.parse_args()

    points = [
        [(0,0),(0,1),(1,1)],
        [(0,0),(1,0),(1,1)],
        [(0,0),(1,0),(1,-1)],
        [(0,0),(0,-1),(1,-1)],
        [(0,0),(-1,0),(-1,-1)],
        [(0,0),(0,-1),(-1,-1)],
        [(0,0),(-1,0),(-1,1)],
        [(0,0),(0,1),(-1,1)],
    ]

    tFiles = np.array(pd.read_csv(args.termsCSV))

    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3]].astype(np.float32)
    fallsInto = {}
    for idx, v in enumerate(vaValues):
        fallsInto[classesLabel[idx]] = []
        for p in points:
            fallsInto[classesLabel[idx]].append(pointInTriangle(v,p[0],p[1],p[2]))

    print(fallsInto)

if __name__ == '__main__':
    main()