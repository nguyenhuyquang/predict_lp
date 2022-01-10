import cv2
import numpy as np
from imutils import perspective

from tools import convertImgToSquare
from detectPlate import detectPlate
from charReconition import CNN_Model

# List characters
ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2',
              24: '3', 25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

CHAR_CLASSIFICATION_WEIGHTS = './cnn_model_pretrain/weight.h5'

class load_model(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.outputs = []

    def predict(self, image):
        pts = detectPlate(image)
        image_crop = perspective.four_point_transform(image, pts)
        self.segmentation(image_crop)
        self.recognizeChar()
        license_plate = self.format()

        return license_plate

    def segmentation(self, image_crop):
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            componentMask = (labels == i).astype("uint8") * 255
            heightWidthRatio = w / float(h)

            if 10 < h < 50 and 1 < w < 25 and 0.1 < heightWidthRatio < 0.7:
                output = np.array(componentMask[y:y + h, x:x + w])
                output_convert = convertImgToSquare(output)
                output_convert = cv2.resize(output_convert, (28, 28), cv2.INTER_AREA)
                output_convert = output_convert.reshape((28, 28, 1))
                self.outputs.append((output_convert, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []

        for char, coordinate in self.outputs:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        self.outputs = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:
                continue
            self.outputs.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []
        second_line = []

        max_y_cor = 100
        for output, coordinate in self.outputs:
            if (coordinate[0] < max_y_cor):
                max_y_cor = coordinate[0]
        y_cor_min = max_y_cor - 10
        y_cor_max = max_y_cor + 10
        for output, coordinate in self.outputs:
            if y_cor_min < coordinate[0] < y_cor_max:
                first_line.append((output, coordinate[1]))
            else:
                second_line.append((output, coordinate[1]))

        def short_x_cor(s):
            return s[1]

        first_line = sorted(first_line, key=short_x_cor)
        second_line = sorted(second_line, key=short_x_cor)

        if len(second_line) == 0:
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join(
                [str(ele[0]) for ele in second_line])

        return license_plate