import numpy as np
import cv2

# --------------------------------------------
# Auxiliary methods:
# --------------------------------------------
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    if len(imRGB.shape) == 3:
        yiq_ = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
        imYI = np.dot(imRGB, yiq_.T.copy())
        return imYI

def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    if len(imYIQ.shape) == 3:
        rgb_ = np.array([[1.00, 0.956, 0.623],
                         [1.0, -0.272, -0.648],
                         [1.0, -1.105, 0.705]])
        imRGB = np.dot(imYIQ, rgb_.T.copy())
        return imRGB
