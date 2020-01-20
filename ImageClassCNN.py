import os
os.chdir("/Users/uuuu/Dropbox/My_Image/NeuralNet/")
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import image_helper_functions as img_helper
import matplotlib.pyplot as plt


###############################################################################################################
# Settings Location
################################################################################################################
picture_location = '/Users/uuuu/temp/ProcessedPictures/'
model_location = '/Users/uuuu/Dropbox/My_Image/NeuralNet/My_Image_model.h5'
picture_size = 64
class_size = 5000
################################################################################################################



def kerasModel(input_shape):
    model = Sequential()
    model.add(Convolution2D(16, kernel_size = (8,8), padding = 'valid', strides=(4,4), input_shape = input_shape, activation='relu'))
    model.add(Convolution2D(32, kernel_size = (5,5), padding='same', strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, kernel_size= (5,5), padding = 'same', strides = (2,2), activation = 'relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return (model)

def loadScaledImages(picture_location, picture_size, class_size):
    scaled_X, y = img_helper.loadData(picture_size, class_size,picture_location)
    n_classes = len(np.unique(y))
    print("Number of classes =", n_classes)
    scaled_X = img_helper.preprocessData(scaled_X)
    y = to_categorical(y)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print("train shape X", X_train.shape)
    print("train shape y", y_train.shape)
    print("test shape X", y_train.shape)
    print("test shape y", y_test.shape)

    return scaled_X, y, X_train, X_test, y_train, y_test

def makeModel(picture_size, X_train, X_test, y_train, y_test):

    model = kerasModel([picture_size, picture_size, 1])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit(X_train, y_train, epochs = 20,verbose=1)
    label_binarizer = LabelBinarizer()
    y_one_hot_test = label_binarizer.fit_transform(y_test)


    metrics = model.evaluate(X_test, y_test)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))
    return(model)


################################################################################################################################
# Train the model
# Train and save the model
################################################################################################################################
scaled_pictures, y, X_train, X_test, y_train, y_test = loadScaledImages(picture_location=picture_location, picture_size=picture_size, class_size=class_size)
#model  = makeModel(picture_size=picture_size,  X_train = X_train, X_test=X_test, y_train = y_train, y_test=y_test)

model = load_model(model_location)
model.summary()

#Show a picture
img_helper.showPostProcessedImage(scaled_pictures[5000])
plt.show()

model.save(model_location)




########################################################################################
#Make Prediction
################################################################################################
#My_Image imag
img_helper.makeIndependentPrediction("https://upload.wikimedia.org/wikipedia/commons/File:Physical_bitcoin_statistic_coin.jpg", model, picture_size) #My_Image
plt.show()
#shoe
img_helper.makeIndependentPrediction("http://dimg.dillards.com/is/image/DillardsZoom/zoom/vionic-splendid-midi-perforated-leather-slip-on-sneakers/04831775_zi_light_blue.jpg", model, picture_size)
plt.show()
#cat
img_helper.makeIndependentPrediction("https://www.catster.com/wp-content/uploads/2017/08/A-fluffy-cat-looking-funny-surprised-or-concerned.jpg", model, picture_size)
plt.show()