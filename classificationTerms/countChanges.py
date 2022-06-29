import argparse, pandas as pd, numpy as np, os, sys
from re import sub

def openCSVFile(csvPath):
    returnFile = {}
    with open(csvPath,'r') as cpath:
        cpath.readline()
        currVideo = 0
        for c in cpath:
            data = c.split(',')
            subject = int(data[-1])
            if subject != currVideo:
                returnFile[subject] = []
                currVideo = subject

            returnFile[subject].append(data[0])

    return returnFile

def main():
    parser = argparse.ArgumentParser(description='Count changes')
    parser.add_argument('--csvFile', help='Path for the terms file', required=True)
    args = parser.parse_args()

    organizedFile = openCSVFile(args.csvFile)
    changes = []
    for o in organizedFile:
        changes.append(0)
        for idxEm in range(1,len(organizedFile[o])):
            changes[-1] += int(organizedFile[o][idxEm-1] == organizedFile[o][idxEm])
    print(sum(changes) / len(changes))


if __name__ == '__main__':
    main()