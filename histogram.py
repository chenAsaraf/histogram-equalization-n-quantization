import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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
        plt.imshow(imRGB)
        return imRGB

# --------------------------------------------
# Histogram:
# --------------------------------------------

"""# original image, 3 channels range [0, 1]
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
imyiq = transformRGB2YIQ(imOrig)
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
imNew = transformYIQ2RGB(imyiq)

imNew = cv2.normalize(imNew, None, 0, 255, cv2.NORM_MINMAX)
imNew = np.ceil(imNew)
imNew = imNew.astype('uint8')
#imNew = cv2.cvtColor(imOrig, cv2.COLOR_RGB2BGR)

cv2.imshow("imNew", imNew)
cv2.waitKey()

images = np.concatenate((Oldimg, imNew), axis=1)
cv2.imshow("CHANGES", images)
cv2.waitKey()"""

# --------------------------------------------
# Quantization (by Lloyd max algorithm)
# loading & convert to YIQ, Y channel
imOrig = cv2.imread("image.jpg")
imOrig = imOrig.astype('uint8') #for using the the next converting from BGR to RGB
imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB)
imOrig = cv2.normalize(imOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
imyiq = transformRGB2YIQ(imOrig)
imOrig = imyiq[:, :, 0]


#normalization to 0-255 - not neccesary?
imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
imOrig = np.ceil(imOrig)
imOrig = imOrig.astype('uint8')
#cv2.imshow("imOrig", imOrig)
#cv2.waitKey()

# array = np.array([1,2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9])# range of 10 intesities
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

print("q : " , str(q))
print("Z: " , str(z))




