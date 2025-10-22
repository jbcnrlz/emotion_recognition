import tarfile, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def compactar_tar_bz2(arquivos, destino):
    with tarfile.open(destino, 'w:bz2') as tar:
        for arquivo in arquivos:
            tar.add(arquivo)

def main():
    folders = ['train_set','val_set']
    for f in folders:
        probsFiles = getFilesInPath(f'C:\\Users\\joaoc\\AffectNetFusedDifferent\\{f}\\annotations')
        compactFiles = []
        for p in probsFiles:
            if '_dom.npy' in p:
                compactFiles.append(p)

        compactar_tar_bz2(compactFiles,f'{f}_comp.tgz')

if __name__ == '__main__':
    main()