import pandas as q
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from skimage import exposure
from tensorflow.contrib.layers import flatten


from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.layers import flatten
import cv2
import glob
from sklearn.metrics import confusion_matrix

import urllib
import urllib.request



def showPostProcessedImage(bitmap):
    plt.imshow(bitmap[:, :, 0] * 255, plt.get_cmap('gray'))

def getClassification(output_vector):
    if np.argmax(output_vector) == 0:
        image_type = "Bit Coin"
    else:
        image_type = "Not Bit Coin !"

    return image_type


def url_to_image(url, new_size ):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, new_size)
    # return the image
    return image


def makeIndependentPrediction(url, model, new_size):
    downloaded_image = url_to_image(url, (new_size, new_size))
    downloaded_image_reshaped = np.reshape(downloaded_image, [1, downloaded_image.shape[0], downloaded_image.shape[1], downloaded_image.shape[2]])
    plt.imshow(cv2.cvtColor(downloaded_image_reshaped[0], cv2.COLOR_BGR2RGB))
    post_process_image = preprocessData(downloaded_image_reshaped)

    showPostProcessedImage(post_process_image[0])

    print(getClassification(model.predict(post_process_image)))

def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def loadBlurImg(path, imgSize, randomlyRotate=False):
    img = cv2.imread(path)
    if randomlyRotate:
        angle = np.random.randint(-180, 180)
        img = rotateImage(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, imgSize)
    return img


def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []

    for path in classPath:
        img = loadBlurImg(path, imgSize)
        x.append(img)
        y.append(classLable)

    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize,randomlyRotate=True)
        x.append(img)
        y.append(classLable)

    return x, y


def loadData(img_size, classSize, parent_directory):
    BitCoins = glob.glob(parent_directory + '/BitCoin/**/*.jpg', recursive=True)
    notBitCoins = glob.glob(parent_directory + '/not_BitCoin/**/*.jpg', recursive=True)

    imgSize = (img_size, img_size)
    xBitCoin, yBitCoin = loadImgClass(BitCoins, 0, classSize, imgSize)
    xNotBitCoin, yNotBitCoin = loadImgClass(notBitCoins, 1, classSize, imgSize)
    print("There are", len(xBitCoin), "BitCoin images")
    print("There are", len(xNotBitCoin), "not BitCoin images")

    X = np.array(xBitCoin + xNotBitCoin)
    y = np.array(yBitCoin + yNotBitCoin)
    # y = y.reshape(y.shape + (1,))
    return X, y


def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html

    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    images = images.reshape(images.shape + (1,))
    return images


def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)