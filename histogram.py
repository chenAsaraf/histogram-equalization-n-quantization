import numpy as np
import cv2
import matplotlib.pyplot as plt
import transform as t
from PIL import Image


# --------------------------------------------
# Histogram:
# --------------------------------------------

# original image, 3 channels range [0, 1]
imOrig = cv2.imread("badcontrast.jpg")
Oldimg = imOrig
imOrig = cv2.normalize(imOrig.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Because original image in range [0,1] and we are required to internally perform the equalization
# using 256 bin histograms so it will be best to normalize the images values to [0,255].
imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
imOrig = np.ceil(imOrig)
imOrig = imOrig.astype('uint8') #for using the the next converting from BGR to RGB
imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB)
imOrig = cv2.normalize(imOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Convert to YIQ and export the the Y channel:
imyiq = t.transformRGB2YIQ(imOrig)
imOrig = imyiq[:, :, 0]


cv2.imshow("imOrig", imOrig)
cv2.waitKey()

# normalize the imOrig values to [0 255]
imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
imOrig = np.ceil(imOrig)
imOrig = imOrig.astype('uint8')

# step 1: Calculate the image histogram - how many pixels in each gray level
hist, bins = np.histogram(imOrig.flatten(), 256)

# step 2: Calculate the normalized Cumulative Sum (CumSum)
cumSum = hist.cumsum()
cumSum_normalized = cumSum * hist.max()/ cumSum.max() #(LUT in assignment)

plt.plot(cumSum_normalized, color='b')
plt.hist(imOrig.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')
plt.show()


# step 3: Create a LookUpTable(LUT) with distribution for each gray level
# masked array from numpy - all operations are performed on non masked elements.
# i.e : all elements != 0
# https://en.wikipedia.org/wiki/Histogram_equalization
# In order to map the values back into their original range, the following
# simple transformation needs to be applied on the result:
cumSum_m = np.ma.masked_equal(cumSum,0) # mask all elements == 0
LUT = (cumSum_m - cumSum_m.min())*255/(cumSum_m.max()-cumSum_m.min())
LUT = np.ma.filled(LUT,0).astype('uint8')

# step 4: Replace each intesity i with LUT[i]
imNew = LUT[imOrig]

# plot the new histogram
plt.hist(imNew.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend('New Histogram', loc='upper left')
plt.show()

# back to rgb
imyiq[:, :, 0] = imNew
imNew = t.transformYIQ2RGB(imyiq)

imNew = cv2.normalize(imNew, None, 0, 255, cv2.NORM_MINMAX)
imNew = np.ceil(imNew)
imNew = imNew.astype('uint8')
#imNew = cv2.cvtColor(imOrig, cv2.COLOR_RGB2BGR)

cv2.imshow("imNew", imNew)
cv2.waitKey()

images = np.concatenate((Oldimg, imNew), axis=1)
cv2.imshow("CHANGES", images)
cv2.waitKey()
