import numpy as np
import cv2

def convertImgToSquare(image):
    img_h = image.shape[0]
    img_w = image.shape[1]

    if img_h > img_w:
        diff = img_h - img_w
        left_side = np.zeros(shape=(img_h, diff//2))
        right_side = left_side
        squared_image = np.concatenate((left_side, image, right_side), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        left_side = np.zeros(shape=((diff//2), img_w))
        right_side = left_side
        squared_image = np.concatenate((left_side, image, right_side), axis=0)
    else:
        squared_image = image

    return squared_image