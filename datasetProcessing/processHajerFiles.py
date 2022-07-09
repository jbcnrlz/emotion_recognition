import pandas as pd

def openCSVHajer(filePath):
    output = []
    with open(filePath,'r') as fp:
        for f in fp:
            output.append(f.strip().split(','))

    return output

def outputCSV(filePath,dataOuput):
    with open(filePath,'w') as fp:
        for d in dataOuput:
            fp.write(','.join(d) + '\n')

def main():
    featAfewVA = openCSVHajer('va_afew-va_mimamo.csv')
    newOutput = []
    for idx, f in enumerate(featAfewVA):
        if idx == 0:
            newOutput.append(f)
        else:
            if len(f) == 1:
                continue
            pathSplited = f[-1].split('/')
            filePath = int(pathSplited[-1].split('_')[-1][:-4]) - 1    
            newPath = '/'.join([pathSplited[1],pathSplited[3],"%05d.bmp" % (filePath)])
            newOutput.append([f[0],f[1],newPath])

    outputCSV('va_afew-va_mimamo_fixed.csv',newOutput)

if __name__ == '__main__':
    main()