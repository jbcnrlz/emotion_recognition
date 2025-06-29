import argparse, sys, os, cv2, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import printProgressBar, getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Align CelebA dataset')
    parser.add_argument('--input', help='Input folder with CelebA images', required=True)
    parser.add_argument('--output', help='Output folder for aligned images', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = getFilesInPath(args.input, imagesOnly=True)
    printProgressBar(0, len(files), prefix='Aligning CelebA images:', suffix='Complete', length=50)

    for i, file in enumerate(files):
        output_file = os.path.join(args.output, os.path.basename(file))
        
        frame = cv2.imread(file)
        
        model_file = 'faceDetection/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        config_file = 'faceDetection/deploy.prototxt'
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()

        for j in range(detections.shape[2]):
            confidence = detections[0, 0, j, 2]

            # Filtro de confiança
            if confidence > 0.7:  # Ajuste este valor conforme necessário
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                roi = frame[y:y2, x:x2]
                cv2.imwrite(output_file, roi)
                break

        printProgressBar(i + 1, len(files), prefix='Aligning CelebA images:', suffix='Complete', length=50)

if __name__ == "__main__":
    main()