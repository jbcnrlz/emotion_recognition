import argparse, csv
import base64
import os

def generateAnn():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--facesDir', help='Weights', required=True)
    args = parser.parse_args()

    
    

if __name__ == '__main__':
    generateAnn()