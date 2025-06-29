import cv2, numpy as np
from PIL import Image
class AlignFace(object):

    def __init__(self, configFile, modelFile):
        self.configFile = configFile
        self.modelFile = modelFile

    def __call__(self, sample):
        grayScale = np.array(sample)        
        h, w = grayScale.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(grayScale, (300, 300)), 1.0, (300, 300), [104, 117, 123])

        net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Ajuste este valor conforme necessário
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                roi = grayScale[y:y2, x:x2].astype(np.uint8)
                roi = Image.fromarray(roi)
                return roi

        return sample  # No faces detected
        