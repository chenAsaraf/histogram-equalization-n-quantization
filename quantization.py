
import numpy as np
import cv2
import matplotlib.pyplot as plt
import transform as t
from PIL import Image

# --------------------------------------------
# Quantization (by Lloyd max algorithm)
# --------------------------------------------
# loading & convert to YIQ, Y channel
imOrig = cv2.imread("image.jpg")
imOrig = imOrig.astype('uint8') #for using the the next converting from BGR to RGB
imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB)
imOrig = cv2.normalize(imOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
imyiq = t.transformRGB2YIQ(imOrig)
imOrig = imyiq[:, :, 0]


#normalization to 0-255 - not neccesary?
imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
imOrig = np.ceil(imOrig)
imOrig = imOrig.astype('uint8')
#cv2.imshow("imOrig", imOrig)
#cv2.waitKey()

nQuant = 4
nIter = 10
hist, bins = np.histogram(imOrig.flatten(),nQuant)
#plt.hist(imOrig.flatten(), nQuant, color='r')
#plt.show()
z = bins.astype('uint8')
q = np.zeros(nQuant)
print("bins: ",z)


for i in range(nIter):
    for j in range(nQuant):
        down = int(z[j])
        up = int(z[j+1])
        mask_pixels = np.ma.masked_inside(imOrig, down, up)
        q[j] = np.average(imOrig[abs(imOrig - down) <= up-down])
        if j != 0:
            z[j] = 0.5 * (q[j-1] + q[j])
        np.ma.set_fill_value(mask_pixels, q[j])
        imOrig = mask_pixels.filled()

cv2.imshow("imOrig", imOrig)
cv2.waitKey()
