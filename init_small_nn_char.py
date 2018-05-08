import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import cv2
import os
import pandas as pd


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
print("before reading csv")
train_db = pd.read_csv("emnist-letters-train.csv")
test_db  = pd.read_csv("emnist-letters-test.csv")
print("after reading csv")

num_pixels = 784
num_classes = 47

y_train = train_db.iloc[:,0]
y_train = np_utils.to_categorical(y_train, num_classes)

x_train = train_db.iloc[:,1:]
x_train = x_train.astype('float32')
x_train /= 255

y_test = test_db.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)

x_test = test_db.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    # model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=200, verbose=2)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model_json = model.to_json()
with open("small_model_char.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("small_model_char.h5")
